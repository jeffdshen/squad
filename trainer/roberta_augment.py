"""Train and test methods for SQuAD question augmentation

Author:
    Jeffrey Shen
"""

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torch.cuda.amp as amp


from collections import OrderedDict
from tqdm import tqdm
import ujson as json

from models import RoBERTa
from datasets.bpe_squad import QuestionsMLM, collate_fn, MLM
from preprocess.bpe import BPE
import trainer.trainer as base_trainer
import trainer.util as util
import trainer.stats as stats
import models.transformer as T
import trainer.scheduler as sched


def add_special_tokens(args):
    args.ignore_idx = -1
    args.padding_idx = 0
    args.cls_idx = 1
    args.sep_idx = 2
    args.mask_idx = 3


def get_args(args):
    # Compute derived args values
    device, args.gpu_ids = util.get_available_devices()

    args.batch_size_per_gpu = args.batch_size
    args.batch_size *= max(1, len(args.gpu_ids))
    args.num_steps = args.epoch_size // args.batch_size // args.gradient_accumulation
    if args.num_epochs >= 0:
        args.num_steps *= args.num_epochs

    if args.decay_forever:
        args.num_steps = float("inf")

    return args, device


def get_bpe(args):
    bpe = BPE()
    with open(args.bpe_file, "r") as file:
        bpe.load_state_dict(json.load(file))
    add_special_tokens(args)
    return bpe


def get_dataset(args, file, bpe, shuffle):
    # Don't need to supply special idxs, since they're the same.
    dataset = QuestionsMLM(
        file,
        max_tokens=len(bpe),
        mask_prob=args.mask_prob,
        unmask_prob=args.unmask_prob,
        randomize_prob=args.randomize_prob,
        use_v2=args.use_squad_v2,
    )
    loader = data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        collate_fn=dataset.get_collate_fn(),
    )
    return dataset, loader


def get_model(args, bpe):
    model = RoBERTa(
        dim=args.dim,
        n_heads=args.n_heads,
        ff_dim=args.ff_dim,
        activation=args.activation,
        dropout=args.dropout,
        attn_dropout=args.attn_dropout,
        act_dropout=args.act_dropout,
        n_layers=args.n_layers,
        max_positions=args.max_positions,
        max_tokens=len(bpe),
        padding_idx=args.padding_idx,
        ignore_idx=args.ignore_idx,
        prenorm=args.prenorm,
        qa=False,
    )
    return model


def train(args):
    trainer = base_trainer.Trainer()
    args, device = get_args(args)
    args, log, tbx = trainer.setup(args)

    # Get BPE
    log.info("Loading BPE...")
    bpe = get_bpe(args)
    log.info("Loaded {} BPE tokens".format(len(bpe)))

    # Get data loader
    log.info("Building dataset...")
    train_dataset, train_loader = get_dataset(
        args, args.train_record_file, bpe, shuffle=True
    )
    dev_dataset, dev_loader = get_dataset(
        args, args.train_record_file, bpe, shuffle=False
    )

    # Get model
    log.info("Building model...")
    model = get_model(args, bpe)
    model = trainer.setup_model(model, device)

    # Get optimizer, scheduler, and scaler
    optimizer = optim.AdamW(
        model.parameters(),
        args.lr,
        betas=(args.beta_1, args.beta_2),
        eps=args.eps,
        weight_decay=args.l2_wd,
    )

    scheduler = sched.get_linear_warmup_power_decay_scheduler(
        optimizer, args.warmup_steps, args.num_steps, power=args.power_decay
    )

    scaler = amp.GradScaler()
    optimizer, scheduler, scaler = trainer.setup_optimizer(optimizer, scheduler, scaler)

    # Train
    log.info("Training...")
    model.train()
    sample_num = 0
    samples_till_eval = args.eval_per_n_samples
    epoch = 0
    step = 0
    trainer.setup_saver()
    trainer.setup_random()
    sample_num, samples_till_eval, epoch, step = trainer.setup_step(
        step_vars=(sample_num, samples_till_eval, epoch, step)
    )
    trainer.setup_close()

    while epoch != args.num_epochs:
        trainer.save_checkpoint(step_vars=(sample_num, samples_till_eval, epoch, step))
        epoch += 1
        log.info(f"Starting epoch {epoch}...")
        # Print histogram of weights every epoch
        for tags, params in model.named_parameters():
            tbx.add_histogram(tags, params.data, epoch)
        with torch.enable_grad(), tqdm(total=len(train_loader.dataset)) as progress_bar:
            for x, y, _, _ in train_loader:
                batch_size = x.size(0)
                loss, loss_val, _ = forward(x, y, args, device, model)
                loss = loss / args.gradient_accumulation

                # Backward
                scaler.scale(loss).backward()
                if (step + 1) % args.gradient_accumulation == 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()

                # Log info
                step += 1
                sample_num += batch_size
                progress_bar.update(batch_size)
                progress_bar.set_postfix(epoch=epoch, NLL=loss_val)
                tbx.add_scalar("train/NLL", loss_val, sample_num)
                tbx.add_scalar("train/LR", optimizer.param_groups[0]["lr"], sample_num)
                tbx.add_scalar(
                    "train/steps", step // args.gradient_accumulation, sample_num
                )

    results, augs = augment(model, dev_loader, device, bpe, args)
    for k, v in results.items():
        tbx.add_scalar(f"dev/{k}", v, sample_num)
    save(args.train_aug_file, augs, "train aug")


def save(filename, obj, message=None):
    if message is not None:
        print(f"Saving {message}...")
        with open(filename, "w") as fh:
            json.dump(obj, fh)
    else:
        raise RuntimeError("Message missing")


def forward(x, y, args, device, model, autocast=True):
    # Setup for forward
    x = x.to(device)
    padding_mask = T.get_padding_mask(x, args.padding_idx)

    # Forward
    with amp.autocast(enabled=autocast):
        scores = model(x, padding_mask=padding_mask)
        scores = model.module.mask_scores(scores, padding_mask)
        y = y.to(device)
        loss = model.module.get_loss(scores, y)
        loss_val = loss.item()

    return loss, loss_val, scores


def sample_mlm_pred(model, x, y, scores, args):
    scores = scores.clone().detach()
    # Don't generate special tokens
    scores[:, :, args.padding_idx] = float("-inf")
    scores[:, :, args.cls_idx] = float("-inf")
    scores[:, :, args.sep_idx] = float("-inf")
    scores[:, :, args.mask_idx] = float("-inf")

    pred = model.sample(scores, 1, alpha=args.sample_temperature).squeeze(-1)
    ans = y.clone().detach()
    mask = y != args.ignore_idx
    ans[~mask] = x[~mask]

    acc = (pred[mask] == y[mask]).float().mean().item()
    pred[~mask] = x[~mask]

    ans = ans[:, 1:].detach().cpu().numpy()
    pred = pred[:, 1:].detach().cpu().numpy()

    return pred, ans, acc


def augment(model, data_loader, device, bpe, args):
    nll_meter = stats.AverageMeter()
    acc_meter = stats.AverageMeter()

    model.eval()
    augs = {}
    with torch.no_grad():
        for _ in range(args.augment_samples):
            for x, y, c, ids in data_loader:
                batch_size = x.size(0)
                _, loss_val, scores = forward(x, y, args, device, model)
                nll_meter.update(loss_val, batch_size)

                x = x.to(device)
                y = y.to(device)
                pred, ans, acc = sample_mlm_pred(model.module, x, y, scores, args)
                acc_meter.update(acc, batch_size)

                for i, id in enumerate(ids.tolist()):
                    if str(id) not in augs:
                        augs[str(id)] = {
                            "gold_question": "",
                            "aug_questions": [],
                            "gold_question_text": "",
                            "aug_question_texts": [],
                        }

                    aug = augs[str(id)]
                    gold = ans[i].tolist()
                    aug_q = pred[i].tolist()
                    assert not aug["gold_question"] or aug["gold_question"] == gold
                    aug["gold_question"] = gold
                    aug["aug_questions"].append(aug_q)
                    aug["gold_question_text"] = bpe.decode(
                        [token for token in gold if token != args.padding_idx]
                    )
                    aug["aug_question_texts"].append(
                        bpe.decode(
                            [token for token in aug_q if token != args.padding_idx]
                        )
                    )

    model.train()

    results_list = [("NLL", nll_meter.avg), ("acc", acc_meter.avg)]
    results = OrderedDict(results_list)

    return results, augs


def add_mlm_args(parser):
    parser.add_argument(
        "--mask_prob", type=float, default=0.15, help="Mask probability."
    )
    parser.add_argument(
        "--unmask_prob",
        type=float,
        default=0.25,
        help="Probability to leave mask unchanged.",
    )
    parser.add_argument(
        "--randomize_prob",
        type=float,
        default=0.25,
        help="Probability to use a random token instead of mask.",
    )


def add_aug_args(parser):
    parser.add_argument(
        "--augment_samples",
        type=int,
        default=2,
        help="Number of augmented samples to generate per question",
    )
    parser.add_argument(
        "--sample_temperature", type=float, default=1.0, help="Sample temperature."
    )


def add_train_args(parser):
    """Add arguments needed in train.py."""
    add_train_test_args(parser)
    base_trainer.add_train_args(parser)
    add_mlm_args(parser)
    add_aug_args(parser)

    parser.add_argument(
        "--eval_per_n_samples",
        type=int,
        default=12500,
        help="Number of samples between successive evaluations.",
    )
    parser.add_argument(
        "--gradient_accumulation",
        type=int,
        default=4,
    )
    parser.add_argument("--lr", type=float, default=0.025, help="Learning rate.")
    parser.add_argument(
        "--warmup_steps", type=float, default=500, help="Warmup optimizer steps."
    )
    parser.add_argument(
        "--power_decay", type=float, default=-0.5, help="Power of the decay."
    )
    parser.add_argument(
        "--decay_forever",
        type=lambda s: s.lower().startswith("t"),
        default=True,
        help="Whether the decay should reach end_lr at the end of training, or in the limit to infinity",
    )

    parser.add_argument("--l2_wd", type=float, default=0.01, help="AdamW weight decay.")
    parser.add_argument("--eps", type=float, default=1e-6, help="Adam epsilon.")
    parser.add_argument("--beta_1", type=float, default=0.9, help="Adam beta_1.")
    parser.add_argument("--beta_2", type=float, default=0.98, help="Adam beta_2.")
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=2,
        help="Number of epochs for which to train. Negative means forever.",
    )
    parser.add_argument(
        "--metric_name",
        type=str,
        default="NLL",
        help="Name of dev metric to determine best checkpoint.",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=5.0,
        help="Maximum gradient norm for gradient clipping.",
    )


def add_train_test_args(parser):
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of sub-processes to use per data loader.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size per GPU. Scales automatically when \
                              multiple GPUs are available.",
    )

    # Model params
    parser.add_argument(
        "--dim",
        type=int,
        default=768,
        help="Embedding dimension.",
    )
    parser.add_argument(
        "--n_heads",
        type=int,
        default=12,
        help="Attention heads.",
    )
    parser.add_argument(
        "--ff_dim",
        type=int,
        default=3072,
        help="Feedforward dimension.",
    )
    parser.add_argument(
        "--activation",
        choices=["relu", "gelu"],
        default="gelu",
        help="Feedforward activation function.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout probability.",
    )
    parser.add_argument(
        "--attn_dropout",
        type=float,
        default=0.1,
        help="Dropout probability for attention weights within self attn.",
    )
    parser.add_argument(
        "--act_dropout",
        type=float,
        default=0.0,
        help="Dropout probability after activation within FF.",
    )
    parser.add_argument(
        "--n_layers",
        type=int,
        default=12,
        help="Number of layers.",
    )
    parser.add_argument(
        "--max_positions",
        type=int,
        default=512,
        help="Maximum number of tokens.",
    )
    parser.add_argument(
        "--prenorm",
        type=lambda s: s.lower().startswith("t"),
        default=False,
        help="Whether to put LayerNorm after the residual or before.",
    )