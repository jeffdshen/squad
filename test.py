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


def main(args):
    parser = argparse.ArgumentParser("Train a model on SQuAD")
    subparsers = parser.add_subparsers()

    bidaf = subparsers.add_parser("bidaf")
    bidaf_trainer.add_test_args(bidaf)
    bidaf.set_defaults(test=bidaf_trainer.test)

    args = parser.parse_args()

    # Require load_path for test.py
    if not args.load_path:
        raise argparse.ArgumentError("Missing required argument --load_path")

    test = args.test
    del args.test
    test(args)


if __name__ == "__main__":
    main()
