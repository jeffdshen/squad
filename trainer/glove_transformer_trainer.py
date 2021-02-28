"""Train and test methods for transformer qa.

Author:
    Chris Chute (chute@stanford.edu)
    Jeffrey Shen
"""

import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torch.cuda.amp as amp


from collections import OrderedDict
from json import dumps
from tensorboardX import SummaryWriter
from tqdm import tqdm
from ujson import load as json_load
from os.path import join

from models import GloveTransformerQA
from datasets.squad import collate_fn, SQuAD
import eval
import trainer.util as util
import trainer.stats as stats
import models.transformer as T
import trainer.scheduler as sched
from torchinfo import summary


def train(args):
    # Set up logging and devices
    log = util.get_logger(args.save_dir, args.name)
    tbx = SummaryWriter(args.save_dir)
    device, args.gpu_ids = util.get_available_devices()
    log.info(f"Args: {dumps(vars(args), indent=4, sort_keys=True)}")
    args.batch_size *= max(1, len(args.gpu_ids))

    # Set random seed
    log.info(f"Using random seed {args.seed}...")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Get data loader
    log.info("Building dataset...")
    train_dataset = SQuAD(args.train_record_file, args.use_squad_v2)
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    dev_dataset = SQuAD(args.dev_record_file, args.use_squad_v2)
    dev_loader = data.DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    # Get embeddings
    log.info("Loading embeddings...")
    word_vectors = util.torch_from_json(args.word_emb_file)

    # TODO: Hardcode padding idx
    padding_idx = 0

    # Get model
    log.info("Building model...")
    model = GloveTransformerQA(
        dim=args.dim,
        n_heads=args.n_heads,
        ff_dim=args.ff_dim,
        activation=args.activation,
        dropout=args.dropout,
        attn_dropout=args.attn_dropout,
        act_dropout=args.act_dropout,
        n_layers=args.n_layers,
        max_positions=args.max_positions,
        word_vectors=word_vectors,
    )
    model = nn.DataParallel(model, args.gpu_ids)
    if args.load_path:
        log.info(f"Loading checkpoint from {args.load_path}...")
        model, step = util.load_model(model, args.load_path, args.gpu_ids)
    else:
        step = 0
    model = model.to(device)

    log.info(model)
    log.info(
        summary(
            model,
            input_size=(args.max_positions, args.batch_size),
            dtypes=[torch.long],
            device=device,
            depth=5,
            verbose=0,
        )
    )
    model.train()

    # Get saver
    saver = util.CheckpointSaver(
        args.save_dir,
        max_checkpoints=args.max_checkpoints,
        metric_name=args.metric_name,
        maximize_metric=args.maximize_metric,
        log=log,
    )

    # Get optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), args.lr, weight_decay=args.l2_wd)
    if args.num_epochs < 0:
        max_num_steps = (
            len(train_loader) + args.num_workers
        ) // args.gradient_accumulation
    else:
        max_num_steps = (
            args.num_epochs
            * (len(train_loader) + args.num_workers)
            // args.gradient_accumulation
        )

    warmup_steps = int(max_num_steps * args.warmup_ratio)
    if args.decay_forever:
        max_num_steps = -1

    scheduler = sched.LinearWarmupPowerDecay(
        optimizer,
        args.start_lr,
        args.lr,
        args.end_lr,
        warmup_steps,
        max_num_steps,
        power=args.power_decay,
    )

    # Train
    log.info("Training...")
    steps_till_eval = args.eval_steps
    epoch = step // len(train_dataset)
    scaler = amp.GradScaler()
    fwd_step = 0

    while epoch != args.num_epochs:
        epoch += 1
        log.info(f"Starting epoch {epoch}...")
        with torch.enable_grad(), tqdm(total=len(train_loader.dataset)) as progress_bar:
            for cw_idxs, cc_idxs, qw_idxs, qc_idxs, y1, y2, ids in train_loader:
                batch_size = cw_idxs.size(0)
                loss, loss_val, _ = forward(
                    cw_idxs, qw_idxs, y1, y2, padding_idx, args, device, model
                )
                loss = loss / args.gradient_accumulation

                # Backward
                scaler.scale(loss).backward()
                if (fwd_step + 1) % args.gradient_accumulation == 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()

                # Log info
                fwd_step += 1
                step += batch_size
                progress_bar.update(batch_size)
                progress_bar.set_postfix(epoch=epoch, NLL=loss_val)
                tbx.add_scalar("train/NLL", loss_val, step)
                tbx.add_scalar("train/LR", optimizer.param_groups[0]["lr"], step)

                steps_till_eval -= batch_size
                if steps_till_eval <= 0:
                    steps_till_eval = args.eval_steps

                    # Evaluate and save checkpoint
                    log.info(f"Evaluating at step {step}...")
                    results, pred_dict = evaluate(
                        model,
                        dev_loader,
                        device,
                        args.dev_eval_file,
                        args.max_ans_len,
                        args.use_squad_v2,
                        args,
                        padding_idx,
                    )
                    saver.save(step, model, results[args.metric_name], device)

                    # Log to console
                    results_str = ", ".join(
                        f"{k}: {v:05.2f}" for k, v in results.items()
                    )
                    log.info(f"Dev {results_str}")

                    # Log to TensorBoard
                    log.info("Visualizing in TensorBoard...")
                    for k, v in results.items():
                        tbx.add_scalar(f"dev/{k}", v, step)
                    util.visualize(
                        tbx,
                        pred_dict=pred_dict,
                        eval_path=args.dev_eval_file,
                        step=step,
                        split="dev",
                        num_visuals=args.num_visuals,
                    )


# (N, C) + (N, Q) -> (N, S)
def concat_example(c, q, padding_idx, max_positions):
    c_len = (c != padding_idx).sum(dim=1)
    q_len = (q != padding_idx).sum(dim=1)
    length = c_len + q_len

    # TODO: Just truncate context for now, can do sliding window later.
    max_length = min(length.max().item(), max_positions)
    length = length.clamp(max=max_length)
    c_len = length - q_len

    x = torch.full(
        (length.size(0), max_length), padding_idx, dtype=c.dtype, device=c.device
    )
    x_range = torch.arange(max_length, device=x.device).unsqueeze(0)
    x_mask = x_range < c_len.unsqueeze(-1)
    c_mask = torch.arange(c.size(1), device=x.device).unsqueeze(0) < c_len.unsqueeze(-1)
    x[x_mask] = c[c_mask]

    x_mask = (x_range < length.unsqueeze(-1)) & ~x_mask
    q_mask = torch.arange(q.size(1), device=x.device).unsqueeze(0) < q_len.unsqueeze(-1)
    x[x_mask] = q[q_mask]

    return x


def forward(cw_idxs, qw_idxs, y1, y2, padding_idx, args, device, model, autocast=True):
    # Setup for forward
    x = concat_example(cw_idxs, qw_idxs, padding_idx, args.max_positions)
    x = x.transpose(0, 1)
    x = x.to(device)
    padding_mask = T.get_padding_mask(x, padding_idx)

    # Forward
    with amp.autocast(enabled=autocast):
        scores = model(x, padding_mask=padding_mask)
        y = torch.stack((y1, y2), dim=-1)
        y = y.clamp(max=args.max_positions - 1)
        y = y.to(device)
        loss = model.module.get_loss(scores, y)
        loss_val = loss.item() * 2

    return loss, loss_val, scores


def evaluate(
    model,
    data_loader,
    device,
    eval_file,
    max_len,
    use_squad_v2,
    args,
    padding_idx,
):
    nll_meter = stats.AverageMeter()

    model.eval()
    pred_dict = {}
    with open(eval_file, "r") as fh:
        gold_dict = json_load(fh)
    with torch.no_grad(), tqdm(total=len(data_loader.dataset)) as progress_bar:
        for cw_idxs, cc_idxs, qw_idxs, qc_idxs, y1, y2, ids in data_loader:
            batch_size = cw_idxs.size(0)
            _, loss_val, scores = forward(
                cw_idxs, qw_idxs, y1, y2, padding_idx, args, device, model
            )
            nll_meter.update(loss_val, batch_size)

            # Get F1 and EM scores
            p1, p2 = model.module.get_prob(scores).transpose(0, 1).split(1, dim=-1)
            p1, p2 = p1.squeeze(-1), p2.squeeze(-1)
            starts, ends = util.discretize(p1, p2, max_len, use_squad_v2)

            # Log info
            progress_bar.update(batch_size)
            progress_bar.set_postfix(NLL=nll_meter.avg)

            preds, _ = util.convert_tokens(
                gold_dict, ids.tolist(), starts.tolist(), ends.tolist(), use_squad_v2
            )
            pred_dict.update(preds)

    model.train()

    results = eval.eval_dicts(gold_dict, pred_dict, use_squad_v2)
    results_list = [
        ("NLL", nll_meter.avg),
        ("F1", results["F1"]),
        ("EM", results["EM"]),
    ]
    if use_squad_v2:
        results_list.append(("AvNA", results["AvNA"]))
    results = OrderedDict(results_list)

    return results, pred_dict


def add_train_args(parser):
    """Add arguments needed in train.py."""
    add_train_test_args(parser)

    parser.add_argument(
        "--eval_steps",
        type=int,
        default=50000,
        help="Number of steps between successive evaluations.",
    )
    parser.add_argument(
        "--gradient_accumulation",
        type=int,
        default=128,
    )
    parser.add_argument("--lr", type=float, default=1.5e-5, help="Learning rate.")
    parser.add_argument(
        "--warmup_ratio", type=float, default=0.06, help="Warmup steps / total steps."
    )
    parser.add_argument(
        "--power_decay", type=float, default=1, help="Power of the decay."
    )
    parser.add_argument(
        "--start_lr", type=float, default=1e-8, help="Starting learning rate."
    )
    parser.add_argument(
        "--end_lr", type=float, default=1e-8, help="Ending learning rate."
    )
    parser.add_argument(
        "--decay_forever",
        type=lambda s: s.lower().startswith("t"),
        default=False,
        help="Whether the decay should reach end_lr at the end of training, or in the limit to infinity",
    )
    parser.add_argument(
        "--warmup_ratio", type=float, default=0.06, help="Warmup steps / total steps."
    )

    parser.add_argument("--l2_wd", type=float, default=0, help="L2 weight decay.")
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
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
        "--max_checkpoints",
        type=int,
        default=5,
        help="Maximum number of checkpoints to keep on disk.",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=5.0,
        help="Maximum gradient norm for gradient clipping.",
    )
    parser.add_argument(
        "--seed", type=int, default=224, help="Random seed for reproducibility."
    )


def test(args):
    # Set up logging
    log = util.get_logger(args.save_dir, args.name)
    log.info(f"Args: {dumps(vars(args), indent=4, sort_keys=True)}")
    device, gpu_ids = util.get_available_devices()
    args.batch_size *= max(1, len(gpu_ids))

    # Get embeddings
    log.info("Loading embeddings...")
    word_vectors = util.torch_from_json(args.word_emb_file)

    # TODO: Hardcode padding_idx
    padding_idx = 0

    # Get model
    log.info("Building model...")
    model = GloveTransformerQA(
        dim=args.dim,
        n_heads=args.n_heads,
        ff_dim=args.ff_dim,
        activation=args.activation,
        dropout=args.dropout,
        attn_dropout=args.attn_dropout,
        act_dropout=args.act_dropout,
        n_layers=args.n_layers,
        max_positions=args.max_positions,
        word_vectors=word_vectors,
    )
    model = nn.DataParallel(model, gpu_ids)
    log.info(f"Loading checkpoint from {args.load_path}...")
    model = util.load_model(model, args.load_path, gpu_ids, return_step=False)
    model = model.to(device)
    model.eval()

    # Get data loader
    log.info("Building dataset...")
    record_file = vars(args)[f"{args.split}_record_file"]
    dataset = SQuAD(record_file, args.use_squad_v2)
    data_loader = data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    # Evaluate
    log.info(f"Evaluating on {args.split} split...")
    nll_meter = stats.AverageMeter()
    pred_dict = {}  # Predictions for TensorBoard
    sub_dict = {}  # Predictions for submission
    eval_file = vars(args)[f"{args.split}_eval_file"]
    with open(eval_file, "r") as fh:
        gold_dict = json_load(fh)
    with torch.no_grad(), tqdm(total=len(dataset)) as progress_bar:
        for cw_idxs, cc_idxs, qw_idxs, qc_idxs, y1, y2, ids in data_loader:
            batch_size = cw_idxs.size(0)
            _, loss_val, scores = forward(
                cw_idxs, qw_idxs, y1, y2, padding_idx, args, device, model
            )
            nll_meter.update(loss_val, batch_size)

            # Get F1 and EM scores
            p1, p2 = model.module.get_prob(scores).transpose(0, 1).split(1, dim=-1)
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
