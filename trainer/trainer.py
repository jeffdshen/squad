"""Generic trainer class that stores all state.

Author:
    Jeffrey Shen
"""
import numpy as np
import random
import torch
import torch.nn as nn

import os
import trainer.util as util
from torch.utils.tensorboard import SummaryWriter
from json import dumps
import queue
import shutil


class Trainer:
    def __init__(self):
        super().__init__()
        self.state_dict = None


    def setup(self, args):
        log = util.get_logger(args.save_dir, args.name)
        tbx = SummaryWriter(args.save_dir)

        if args.resume_dir:
            checkpoint_path = os.path.join(args.resume_dir, "checkpoint.pth.tar")
            self.state_dict = torch.load(checkpoint_path)
            self.args = self.state_dict["args"]
            self.args.save_dir = args.save_dir
            self.args.name = args.name
            args = self.args
            log.info("Resuming from checkpoint: {}".format(checkpoint_path))


        self.args = args

        log.info(f"Args: {dumps(vars(args), indent=4, sort_keys=True)}")

        log.info(f"Using random seed {args.seed}...")
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        self.log = log
        self.tbx = tbx

        return args, log, tbx

    def setup_saver(self):
        args = self.args
        log = self.log
        self.saver = ModelSaver(
            args.save_dir,
            max_checkpoints=args.max_checkpoints,
            metric_name=args.metric_name,
            maximize_metric=args.maximize_metric,
            log=log,
        )

    def setup_random(self):
        log = self.log
        if self.state_dict is not None:
            log.info("Reloading random state...")
            random.setstate(self.state_dict["random"])
            np.random.set_state(self.state_dict["np.random"])
            torch.set_rng_state(self.state_dict["torch.random"])
            torch.cuda.set_rng_state_all(self.state_dict["torch.cuda.random"])

    def setup_model(self, model, device):
        log = self.log
        args = self.args
        if self.state_dict is not None:
            log.info("Reloading model...")
            model.load_state_dict(self.state_dict["model"])
        elif args.load_path:
            log.info(f"Loading model from {args.load_path}...")
            model, _ = ModelSaver.load_model(model, args.load_path, device)
        model = nn.DataParallel(model, self.args.gpu_ids)
        model = model.to(device)
        self.model = model
        self.device = device

        log.info(model)
        return model

    def setup_optimizer(self, optimizer, scheduler, scaler):
        log = self.log
        if self.state_dict is not None:
            log.info("Reloading optimizer, scheduler, scaler...")
            optimizer.load_state_dict(self.state_dict["optimizer"])
            scheduler.load_state_dict(self.state_dict["scheduler"])
            scaler.load_state_dict(self.state_dict["scaler"])

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        return optimizer, scheduler, scaler

    def setup_step(self, step_vars):
        log = self.log
        if self.state_dict is not None:
            log.info("Reloading step: {}".format(self.state_dict["step"]))
            return self.state_dict["step"]

        return step_vars

    def setup_close(self):
        self.state_dict = None

    def save_checkpoint(self, step_vars):
        ckpt_dict = {
            "args" : self.args,
            "random": random.getstate(),
            "np.random" : np.random.get_state(),
            "torch.random" : torch.random.get_rng_state(),
            "torch.cuda.random" : torch.cuda.get_rng_state_all(),
            "model": self.model.module.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "step": step_vars,
        }

        checkpoint_path = os.path.join(self.args.save_dir, "checkpoint.pth.tar")
        torch.save(ckpt_dict, checkpoint_path)

    def save_best(self, step, metric_val):
        self.saver.save(step, self.model.module, metric_val)


def add_train_args(parser):
    parser.add_argument(
        "--max_checkpoints",
        type=int,
        default=5,
        help="Maximum number of checkpoints to keep on disk.",
    )
    parser.add_argument(
        "--resume_dir",
        type=str,
        default=None,
        help="Path to trainer checkpoint.",
    )
    parser.add_argument(
        "--seed", type=int, default=224, help="Random seed for reproducibility."
    )
    

class ModelSaver:
    """Class to save and load model checkpoints.

    Save the best checkpoints as measured by a metric value passed into the
    `save` method. Overwrite checkpoints with better checkpoints once
    `max_checkpoints` have been saved.

    Author:
        Chris Chute (chute@stanford.edu)

    Args:
        save_dir (str): Directory to save checkpoints.
        max_checkpoints (int): Maximum number of checkpoints to keep before
            overwriting old ones.
        metric_name (str): Name of metric used to determine best model.
        maximize_metric (bool): If true, best checkpoint is that which maximizes
            the metric value passed in via `save`. Otherwise, best checkpoint
            minimizes the metric.
        log (logging.Logger): Optional logger for printing information.
    """

    def __init__(
        self, save_dir, max_checkpoints, metric_name, maximize_metric=False, log=None
    ):
        super().__init__()

        self.save_dir = save_dir
        self.max_checkpoints = max_checkpoints
        self.metric_name = metric_name
        self.maximize_metric = maximize_metric
        self.best_val = None
        self.ckpt_paths = queue.PriorityQueue()
        self.log = log
        self._print(
            f"Saver will {'max' if maximize_metric else 'min'}imize {metric_name}..."
        )

    def is_best(self, metric_val):
        """Check whether `metric_val` is the best seen so far.

        Args:
            metric_val (float): Metric value to compare to prior checkpoints.
        """
        if metric_val is None:
            # No metric reported
            return False

        if self.best_val is None:
            # No checkpoint saved yet
            return True

        return (self.maximize_metric and self.best_val < metric_val) or (
            not self.maximize_metric and self.best_val > metric_val
        )

    def _print(self, message):
        """Print a message if logging is enabled."""
        if self.log is not None:
            self.log.info(message)

    def save(self, step, model, metric_vals):
        """Save model parameters to disk.

        Args:
            step (int): Total number of examples seen during training so far.
            model (torch.nn.Module): Model to save.
            metric_vals (tuple(float)): Determines whether checkpoint is best so far.
        """
        ckpt_dict = {
            "model": model.state_dict(),
            "step": step,
        }

        checkpoint_path = os.path.join(self.save_dir, f"step_{step}.pth.tar")
        torch.save(ckpt_dict, checkpoint_path)
        self._print(f"Saved checkpoint: {checkpoint_path}")

        if not isinstance(metric_vals, tuple):
            metric_vals = (metric_vals,)

        if self.is_best(metric_vals):
            # Save the best model
            self.best_val = metric_vals
            best_path = os.path.join(self.save_dir, "best.pth.tar")
            shutil.copy(checkpoint_path, best_path)
            self._print(f"New best checkpoint at step {step}...")

        # Add checkpoint path to priority queue (lowest priority removed first)
        if self.maximize_metric:
            priority_order = tuple(metric_val for metric_val in metric_vals)
        else:
            priority_order = tuple(-metric_val for metric_val in metric_vals)

        self.ckpt_paths.put((priority_order, checkpoint_path))

        # Remove a checkpoint if more than max_checkpoints have been saved
        if self.ckpt_paths.qsize() > self.max_checkpoints:
            _, worst_ckpt = self.ckpt_paths.get()
            try:
                os.remove(worst_ckpt)
                self._print(f"Removed checkpoint: {worst_ckpt}")
            except OSError:
                # Avoid crashing if checkpoint has been removed or protected
                pass

    @staticmethod
    def load_model(model, checkpoint_path, device):
        """Load model parameters from disk.

        Args:
            model (torch.nn.Module): Load parameters into this model.
            checkpoint_path (str): Path to checkpoint to load.
            device: device to reload to

        Returns:
            model (torch.nn.Module): Model loaded from checkpoint.
            step (int): Step at which checkpoint was saved.
        """
        ckpt_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt_dict["model"])
        step = ckpt_dict["step"]
        return model, step

