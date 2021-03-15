"""Train and test methods for iterative denoising pretraining.

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
from ujson import load as json_load

from models import RoBERTa
from datasets.bpe_squad import collate_fn, MLM
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
        bpe.load_state_dict(json_load(file))
    add_special_tokens(args)
    return bpe


def get_dataset(args, epoch_size, file, bpe):
    # Don't need to supply special idxs, since they're the same.
    dataset = MLM(
        file,
        max_tokens=len(bpe),
        epoch_size=epoch_size,
        mask_prob=args.mask_prob,
        unmask_prob=args.unmask_prob,
        randomize_prob=args.randomize_prob,
        block_size=args.max_positions,
    )
    loader = data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
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
        args, args.epoch_size, args.train_record_file, bpe
    )
    dev_dataset, dev_loader = get_dataset(
        args, args.dev_epoch_size, args.dev_record_file, bpe
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
            for x, y in train_loader:
                total_lambda_weight = sum(
                    [args.lambda_weight ** i for i in range(args.mlm_samples + 1)]
                )
                loss_weight = args.gradient_accumulation * total_lambda_weight
                batch_size = x.size(0)
                loss, loss_val, scores = forward(x, y, args, device, model)
                loss = loss / loss_weight
                scaler.scale(loss).backward()

                # DIDAE
                loss_val_didae = {}
                for i in range(args.mlm_samples):
                    x, y = x.to(device), y.to(device)
                    x, y = sample_mlm_pred(model.module, x, y, scores, i, args)
                    # Try to free up scores
                    del scores

                    loss, loss_val_item, scores = forward(x, y, args, device, model)
                    loss = loss / loss_weight * (args.lambda_weight ** (i + 1))
                    loss_val_didae["NLL_{}".format(i + 1)] = loss_val_item
                    scaler.scale(loss).backward()

                # Optimizer step
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
                progress_bar.set_postfix(epoch=epoch, NLL=loss_val, **loss_val_didae)
                tbx.add_scalar("train/NLL", loss_val, sample_num)
                for k, v in loss_val_didae.items():
                    tbx.add_scalar(f"train/{k}", v, sample_num)
                tbx.add_scalar("train/LR", optimizer.param_groups[0]["lr"], sample_num)
                tbx.add_scalar(
                    "train/steps", step // args.gradient_accumulation, sample_num
                )

                samples_till_eval -= batch_size
                if samples_till_eval <= 0:
                    samples_till_eval = args.eval_per_n_samples

                    # Evaluate and save checkpoint
                    log.info(f"Evaluating at sample step {sample_num}...")
                    results, preds, preds_didae = evaluate(
                        model, dev_loader, device, args
                    )
                    trainer.save_best(sample_num, results[args.metric_name])

                    # Log to console
                    results_str = ", ".join(
                        f"{k}: {v:05.2f}" for k, v in results.items()
                    )
                    log.info(f"Dev {results_str}")

                    # Log to TensorBoard
                    log.info("Visualizing in TensorBoard...")
                    for k, v in results.items():
                        tbx.add_scalar(f"dev/{k}", v, sample_num)

                    visuals = select_visuals(len(preds), args.num_visuals)
                    visualize(
                        tbx,
                        preds=preds,
                        bpe=bpe,
                        sample_num=sample_num,
                        split="dev",
                        visuals=visuals,
                    )
                    for k, v in preds_didae.items():
                        visualize(
                            tbx,
                            preds=v,
                            bpe=bpe,
                            sample_num=sample_num,
                            split="dev_{}".format(k + 1),
                            visuals=visuals,
                        )


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


def select_visuals(pred_len, num_visuals):
    if num_visuals <= 0:
        return []
    if num_visuals > pred_len:
        num_visuals = pred_len

    visuals = random.sample(range(pred_len), k=num_visuals)
    return visuals


def visualize(tbx, preds, bpe, sample_num, split, visuals=None):
    """Visualize text examples to TensorBoard.

    Args:
        tbx (SummaryWriter): Summary writer.
        preds (list(tuple)): list of (pred, ques, ans)
        bpe (BPE): bpe encoder/decoder
        sample_num (int): Number of examples seen so far during training.
        split (str): Name of data split being visualized.
        visuals (list(int)): indices to select from preds
    """
    num_visuals = len(visuals)

    visuals = [preds[i] for i in visuals]
    for i, (pred, ques, ans) in enumerate(visuals):
        pred = bpe.decode(pred)
        ques = bpe.decode(ques)
        ans = bpe.decode(ans)
        tbl_fmt = (
            f"- **Answer:** {ans}\n"
            + f"- **MLM Question:** {ques}\n"
            + f"- **Prediction:** {pred}"
        )
        tbx.add_text(
            tag=f"{split}/{i+1}_of_{num_visuals}",
            text_string=tbl_fmt,
            global_step=sample_num,
        )


def get_mlm_pred(model, x, y, scores, args):
    pred = model.get_top(scores)
    mask = y != args.ignore_idx
    acc = (pred[mask] == y[mask]).float().mean().item()
    ans = y.clone().detach()
    ans[~mask] = x[~mask]
    pred[~mask] = x[~mask]
    pred = pred.tolist()
    pred = [[token for token in sample if token != args.padding_idx] for sample in pred]
    ques = x.tolist()
    ques = [[token for token in sample if token != args.padding_idx] for sample in ques]
    ans = ans.tolist()
    ans = [[token for token in sample if token != args.padding_idx] for sample in ans]
    return pred, ques, ans, acc


def sample_mlm_pred(model, x, y, scores, idx, args):
    y = y.clone().detach()
    x = x.clone().detach()
    scores = scores.clone().detach()
    mask = y != args.ignore_idx
    y[~mask] = x[~mask]

    # Just in case, ignore the padding idx
    padding_mask = T.get_padding_mask(x, args.padding_idx)
    y[padding_mask] = args.ignore_idx

    # Don't choose the padding idx!
    scores[:, :, args.padding_idx] = float("-inf")
    alpha = args.sample_temperature if idx > 0 else args.mlm_sample_temperature
    x[mask] = model.sample(scores[mask, :], 1, alpha=alpha).squeeze(-1)

    # Only give hints after first iteration
    if idx > 0:
        hint_mask = torch.rand(x.size(), device=x.device)
        hint_mask = hint_mask < args.hint_prob
        x[hint_mask] = y[hint_mask]

    cls_idx = args.cls_idx if idx < args.didae_cls else args.sep_idx
    x[:, 0] = args.cls_idx
    y[:, 0] = cls_idx
    return x, y


def evaluate(model, data_loader, device, args):
    nll_meter = stats.AverageMeter()
    acc_meter = stats.AverageMeter()
    nll_didae_meter = {i: stats.AverageMeter() for i in range(args.mlm_samples)}
    acc_didae_meter = {i: stats.AverageMeter() for i in range(args.mlm_samples)}

    model.eval()
    preds = []
    preds_didae = {i: [] for i in range(args.mlm_samples)}
    with torch.no_grad():
        for x, y in data_loader:
            batch_size = x.size(0)
            _, loss_val, scores = forward(x, y, args, device, model)
            nll_meter.update(loss_val, batch_size)

            x = x.to(device)
            y = y.to(device)
            pred, ques, ans, acc = get_mlm_pred(model.module, x, y, scores, args)
            acc_meter.update(acc, batch_size)
            preds += zip(pred, ques, ans)

            # DIDAE
            for i in range(args.mlm_samples):
                x, y = sample_mlm_pred(model.module, x, y, scores, i, args)
                # Try to free up scores
                del scores

                _, loss_val_item, scores = forward(x, y, args, device, model)
                nll_didae_meter[i].update(loss_val_item, batch_size)

                pred, ques, ans, acc = get_mlm_pred(model.module, x, y, scores, args)
                acc_didae_meter[i].update(acc, batch_size)
                preds_didae[i] += zip(pred, ques, ans)

    model.train()

    results_list = [("NLL", nll_meter.avg), ("acc", acc_meter.avg)]
    for i in range(args.mlm_samples):
        results_list += [
            ("NLL_{}".format(i + 1), nll_didae_meter[i].avg),
            ("acc_{}".format(i + 1), acc_didae_meter[i].avg),
        ]
    results = OrderedDict(results_list)

    return results, preds, preds_didae


def add_mlm_args(parser):
    parser.add_argument(
        "--mask_prob", type=float, default=0.25, help="Mask probability."
    )
    parser.add_argument(
        "--unmask_prob",
        type=float,
        default=0.1,
        help="Probability to leave mask unchanged.",
    )
    parser.add_argument(
        "--randomize_prob",
        type=float,
        default=0.1,
        help="Probability to use a random token instead of mask.",
    )


def add_didae_mlm_args(parser):
    parser.add_argument(
        "--mlm_sample_temperature",
        type=float,
        default=2.0,
        help="Sample temperature (after mlm).",
    )
    parser.add_argument(
        "--sample_temperature",
        type=float,
        default=0.5,
        help="Sample temperature (after didae).",
    )
    parser.add_argument(
        "--mlm_samples",
        type=int,
        default=3,
        help="How many times to iterate replace MLM",
    )
    parser.add_argument(
        "--lambda_weight",
        type=float,
        default=3.25,
        help="Lambda weight for each successive iteration",
    )
    parser.add_argument(
        "--hint_prob",
        type=float,
        default=0.5,
        help="Probability to replace incorrect tokens with the correct one each iteration",
    )
    parser.add_argument(
        "--didae_cls",
        type=int,
        default=1,
        help="Model predict [cls] (<didae_cls) [sep] (>=didae_cls)",
    )


def add_train_args(parser):
    """Add arguments needed in train.py."""
    add_train_test_args(parser)
    base_trainer.add_train_args(parser)
    add_mlm_args(parser)
    add_didae_mlm_args(parser)

    parser.add_argument(
        "--epoch_size", type=int, default=25000, help="Number of samples per epoch."
    )
    parser.add_argument(
        "--dev_epoch_size",
        type=int,
        default=1000,
        help="Number of samples for a dev evaluation.",
    )
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
        "--warmup_steps", type=float, default=7500, help="Warmup optimizer steps."
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
        default=30,
        help="Number of epochs for which to train. Negative means forever.",
    )
    parser.add_argument(
        "--metric_name",
        type=str,
        default="NLL_3",
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
        default=16,
        help="Batch size per GPU. Scales automatically when \
                              multiple GPUs are available.",
    )
    parser.add_argument(
        "--num_visuals",
        type=int,
        default=10,
        help="Number of examples to visualize in TensorBoard.",
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