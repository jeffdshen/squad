"""Test a model and generate submission CSV.

Usage:
    > python test.py --split SPLIT --load_path PATH --name NAME
    where
    > SPLIT is either "dev" or "test"
    > PATH is a path to a checkpoint (e.g., save/train/model-01/best.pth.tar)
    > NAME is a name to identify the test run

Author:
    Chris Chute (chute@stanford.edu)
"""

import argparse

from trainer import bidaf_trainer, glove_transformer_trainer, roberta_finetune
import util


def add_subparser(name, data_sub_dir, subparsers, parent_parser, module):
    subparser = subparsers.add_parser(name, parents=[parent_parser])
    module.add_test_args(subparser)
    subparser.set_defaults(test=module.test)
    subparser.add_argument("--data_sub_dir", type=str, default=data_sub_dir)


def main():
    parser = argparse.ArgumentParser("Test a trained model on SQuAD")
    parent = argparse.ArgumentParser(add_help=False)

    util.add_data_args(parent)
    util.add_test_args(parent)
    subparsers = parser.add_subparsers()

    add_subparser("bidaf", "bidaf", subparsers, parent, bidaf_trainer)
    add_subparser(
        "glove_transformer", "bidaf", subparsers, parent, glove_transformer_trainer
    )
    add_subparser("roberta_finetune", "bpe", subparsers, parent, roberta_finetune)

    args = parser.parse_args()

    # Require load_path for test.py
    if not args.load_path:
        raise argparse.ArgumentError("Missing required argument --load_path")

    args.save_dir = util.get_save_dir(args.save_dir, args.name, training=False)
    args.data_dir = util.get_data_dir(args.data_dir, args.data_sub_dir)
    util.build_data_dir_path(args)

    test = args.test
    del args.test
    test(args)


if __name__ == "__main__":
    main()
