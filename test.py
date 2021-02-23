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

from trainer import bidaf_trainer
import util


def main(args):
    parser = argparse.ArgumentParser("Train a model on SQuAD")
    parent_parser = argparse.ArgumentParser(add_help=False)

    util.add_data_args(parent_parser)
    util.add_train_test_args(parent_parser)
    subparsers = parser.add_subparsers()

    bidaf = subparsers.add_parser("bidaf", parents=[parent_parser])
    bidaf_trainer.add_test_args(bidaf)
    bidaf.set_defaults(test=bidaf_trainer.test)
    bidaf.set_defaults(data_sub_dir="bidaf")

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
