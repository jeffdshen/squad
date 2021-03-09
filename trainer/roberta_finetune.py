"""Train and test methods for transformer qa.

Author:
    Chris Chute (chute@stanford.edu)
    Jeffrey Shen
"""


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torch.cuda.amp as amp

from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from ujson import load as json_load
from os.path import join

from models import RoBERTa
from datasets.bpe_squad import SQuAD
from preprocess.bpe import BPE
import eval
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
    return args, device


def get_num_steps(args):
    args.num_steps = args.epoch_size // args.batch_size // args.gradient_accumulation
    if args.num_epochs >= 0:
        args.num_steps *= args.num_epochs

    if args.decay_forever:
        args.num_steps = float("inf")

    return args.num_steps


def get_bpe(args):
    bpe = BPE()
    with open(args.bpe_file, "r") as file:
        bpe.load_state_dict(json_load(file))
    add_special_tokens(args)
    return bpe


def get_dataset(args, file, shuffle, randomize):
    # Don't need to supply special idxs, since they're the same.
    dataset = SQuAD(
        file,
        block_size=args.max_positions,
        use_v2=args.use_squad_v2,
    )
    loader = data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        collate_fn=dataset.get_sliding_window_collate(
            args.context_window_stride, randomize
        ),
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
        ignore_idx=None,
        prenorm=args.prenorm,
        qa=True,
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
    train_dataset, train_loader = get_dataset(args, args.train_record_file, True, True)
    dev_dataset, dev_loader = get_dataset(args, args.dev_record_file, False, True)
    args.epoch_size = len(train_dataset)
    log.info("Train has {} examples".format(args.epoch_size))

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

    get_num_steps(args)
    log.info("Scheduler will decay over {} steps".format(args.num_steps))
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
            for x, y, c_padding_mask, ids in train_loader:
                batch_size = x.size(0)
                loss, loss_val, _ = forward(x, y, c_padding_mask, args, device, model)
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

                samples_till_eval -= batch_size
                if samples_till_eval <= 0:
                    samples_till_eval = args.eval_per_n_samples

                    # Evaluate and save checkpoint
                    log.info(f"Evaluating at sample step {sample_num}...")
                    results, pred_dict = evaluate(
                        model, dev_loader, device, args.dev_eval_file, args
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
                    util.visualize(
                        tbx,
                        pred_dict=pred_dict,
                        eval_path=args.dev_eval_file,
                        step=sample_num,
                        split="dev",
                        num_visuals=args.num_visuals,
                    )


def forward(x, y, c_padding_mask, args, device, model, autocast=True):
    # Setup for forward
    x = x.to(device)
    padding_mask = T.get_padding_mask(x, args.padding_idx)

    # Forward
    with amp.autocast(enabled=autocast):
        scores = model(x, padding_mask=padding_mask)
        c_padding_mask = c_padding_mask.to(device)
        scores = model.module.mask_scores(scores, c_padding_mask)
        y = y.to(device)
        loss = model.module.get_loss(scores, y)
        loss_val = loss.item() * 2

    return loss, loss_val, scores


def evaluate(model, data_loader, device, eval_file, args):
    nll_meter = stats.AverageMeter()

    model.eval()
    pred_dict = {}
    with open(eval_file, "r") as fh:
        gold_dict = json_load(fh)
    with torch.no_grad():
        for x, y, c_padding_mask, ids in data_loader:
            batch_size = x.size(0)
            _, loss_val, scores = forward(x, y, c_padding_mask, args, device, model)
            nll_meter.update(loss_val, batch_size)

            # Get F1 and EM scores
            p1, p2 = model.module.get_prob(scores).split(1, dim=-1)
            p1, p2 = p1.squeeze(-1), p2.squeeze(-1)
            starts, ends = util.discretize(p1, p2, args.max_ans_len, args.use_squad_v2)

            preds, _ = util.convert_tokens(
                gold_dict,
                ids.tolist(),
                starts.tolist(),
                ends.tolist(),
                args.use_squad_v2,
            )
            pred_dict.update(preds)

    model.train()

    results = eval.eval_dicts(gold_dict, pred_dict, args.use_squad_v2)
    results_list = [
        ("NLL", nll_meter.avg),
        ("F1", results["F1"]),
        ("EM", results["EM"]),
    ]
    if args.use_squad_v2:
        results_list.append(("AvNA", results["AvNA"]))
    results = OrderedDict(results_list)

    return results, pred_dict


def add_train_args(parser):
    """Add arguments needed in train.py."""
    add_train_test_args(parser)
    base_trainer.add_train_args(parser)

    parser.add_argument(
        "--eval_per_n_samples",
        type=int,
        default=25000,
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
        default=2,
        help="Number of epochs for which to train. Negative means forever.",
    )
    parser.add_argument(
        "--metric_name",
        type=str,
        default="F1",
        choices=("NLL", "EM", "F1"),
        help="Name of dev metric to determine best checkpoint.",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=5.0,
        help="Maximum gradient norm for gradient clipping.",
    )


def test(args):
    trainer = base_trainer.Trainer(is_train=False)
    args, device = get_args(args)
    args, log, tbx = trainer.setup(args)

    # Get BPE
    log.info("Loading BPE...")
    bpe = get_bpe(args)
    log.info("Loaded {} BPE tokens".format(len(bpe)))

    # Get data loader
    log.info("Building dataset...")
    record_file = vars(args)[f"{args.split}_record_file"]
    dataset, data_loader = get_dataset(args, record_file, shuffle=False, randomize=False)

    # Get model
    log.info("Building model...")
    model = get_model(args, bpe)
    model = trainer.setup_model(model, device)
    model.eval()

    trainer.setup_close()

    # Evaluate
    log.info(f"Evaluating on {args.split} split...")
    nll_meter = stats.AverageMeter()
    pred_dict = {}  # Predictions for TensorBoard
    sub_dict = {}  # Predictions for submission
    eval_file = vars(args)[f"{args.split}_eval_file"]
    with open(eval_file, "r") as fh:
        gold_dict = json_load(fh)
    with torch.no_grad(), tqdm(total=len(dataset)) as progress_bar:
        for x, y, c_padding_mask, ids in data_loader:
            batch_size = x.size(0)
            _, loss_val, scores = forward(x, y, c_padding_mask, args, device, model)
            nll_meter.update(loss_val, batch_size)

            # Get F1 and EM scores
            p1, p2 = model.module.get_prob(scores).split(1, dim=-1)
            p1, p2 = p1.squeeze(-1), p2.squeeze(-1)
            starts, ends = util.discretize(p1, p2, args.max_ans_len, args.use_squad_v2)

            # Log info
            progress_bar.update(batch_size)
            if args.split != "test":
                # No labels for the test set, so NLL would be invalid
                progress_bar.set_postfix(NLL=nll_meter.avg)

            idx2pred, uuid2pred = util.convert_tokens(
                gold_dict,
                ids.tolist(),
                starts.tolist(),
                ends.tolist(),
                args.use_squad_v2,
            )
            pred_dict.update(idx2pred)
            sub_dict.update(uuid2pred)

    # Log results (except for test set, since it does not come with labels)
    if args.split != "test":
        results = eval.eval_dicts(gold_dict, pred_dict, args.use_squad_v2)
        results_list = [
            ("NLL", nll_meter.avg),
            ("F1", results["F1"]),
            ("EM", results["EM"]),
        ]
        if args.use_squad_v2:
            results_list.append(("AvNA", results["AvNA"]))
        results = OrderedDict(results_list)

        # Log to console
        results_str = ", ".join(f"{k}: {v:05.2f}" for k, v in results.items())
        log.info(f"{args.split.title()} {results_str}")

        # Log to TensorBoard
        tbx = SummaryWriter(args.save_dir)
        util.visualize(
            tbx,
            pred_dict=pred_dict,
            eval_path=eval_file,
            step=0,
            split=args.split,
            num_visuals=args.num_visuals,
        )

    # Write submission file
    if args.split == "dev":
        sub_path = join(args.save_dir, "val" + "_" + args.sub_file)
    else:
        sub_path = join(args.save_dir, args.split + "_" + args.sub_file)
    log.info(f"Writing submission file to {sub_path}...")
    eval.write_submission(sub_path, sub_dict)


def add_test_args(parser):
    """Get arguments needed in test.py."""
    add_train_test_args(parser)


def add_train_test_args(parser):
    """Add arguments common to train.py and test.py"""
    parser.add_argument(
        "--max_ans_len",
        type=int,
        default=15,
        help="Maximum length of a predicted answer.",
    )
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
        "--use_squad_v2",
        type=lambda s: s.lower().startswith("t"),
        default=True,
        help="Whether to use SQuAD 2.0 (unanswerable) questions.",
    )
    parser.add_argument(
        "--num_visuals",
        type=int,
        default=10,
        help="Number of examples to visualize in TensorBoard.",
    )
    parser.add_argument(
        "--context_window_stride",
        type=int,
        default=256,
        help="Stride for selecting sliding windows from the context.",
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