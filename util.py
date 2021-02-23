"""Utility classes and methods.

Author:
    Chris Chute (chute@stanford.edu)
"""

import os
import argparse


def get_save_dir(base_dir, name, training, id_max=100):
    """Get a unique save directory by appending the smallest positive integer
    `id < id_max` that is not already taken (i.e., no dir exists with that id).

    Args:
        base_dir (str): Base directory in which to make save directories.
        name (str): Name to identify this training run. Need not be unique.
        training (bool): Save dir. is for training (determines subdirectory).
        id_max (int): Maximum ID number before raising an exception.

    Returns:
        save_dir (str): Path to a new directory with a unique name.
    """
    for uid in range(1, id_max):
        subdir = "train" if training else "test"
        save_dir = os.path.join(base_dir, subdir, f"{name}-{uid:02d}")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            return save_dir

    raise RuntimeError(
        "Too many save directories created with the same name. \
                       Delete old save directories or use another name."
    )


def add_train_args(parser):
    add_train_test_args(parser)


def add_test_args(parser):
    add_train_test_args(parser)
    parser.add_argument(
        "--split",
        type=str,
        default="dev",
        choices=("train", "traind", "dev", "test"),
        help="Split to use for testing.",
    )
    parser.add_argument(
        "--sub_file",
        type=str,
        default="submission.csv",
        help="Name for submission file.",
    )


def add_train_test_args(parser):
    parser.add_argument(
        "--name",
        "-n",
        type=str,
        required=True,
        help="Name to identify training or test run.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./save/",
        help="Base directory for saving information.",
    )
    parser.add_argument(
        "--load_path",
        type=str,
        default=None,
        help="Path to load as a model checkpoint.",
    )


def add_data_args(parser):
    """Add arguments common to all 3 scripts: setup.py, train.py, test.py"""
    parser.add_argument("--data_dir", type=str, default="./data/")
    parser.add_argument("--train_record_file", type=str, default="train.npz")
    parser.add_argument("--dev_record_file", type=str, default="dev.npz")
    parser.add_argument("--test_record_file", type=str, default="test.npz")
    parser.add_argument("--word_emb_file", type=str, default="word_emb.json")
    parser.add_argument("--char_emb_file", type=str, default="char_emb.json")
    parser.add_argument("--train_eval_file", type=str, default="train_eval.json")
    parser.add_argument("--dev_eval_file", type=str, default="dev_eval.json")
    parser.add_argument("--test_eval_file", type=str, default="test_eval.json")

    # TODO delete?
    parser.add_argument("--dev_meta_file", type=str, default="dev_meta.json")
    parser.add_argument("--test_meta_file", type=str, default="test_meta.json")
    parser.add_argument("--word2idx_file", type=str, default="word2idx.json")
    parser.add_argument("--char2idx_file", type=str, default="char2idx.json")


def get_data_dir(base_dir, name):
    if name is None:
        return base_dir

    data_dir = os.path.join(base_dir, f"{name}")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    return data_dir


def build_data_dir_path(args):
    parser = argparse.ArgumentParser()
    add_data_args(parser)
    data_args = parser.parse_args([])
    for f in vars(data_args):
        if f == "data_dir":
            continue
        vars(args)[f] = os.path.join(args.data_dir, vars(args)[f])
