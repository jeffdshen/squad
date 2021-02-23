"""Download and pre-process SQuAD and GloVe.

Usage:
    > source activate squad
    > python setup.py

Pre-processing code adapted from:
    > https://github.com/HKUST-KnowComp/R-Net/blob/master/prepro.py

Author:
    Chris Chute (chute@stanford.edu)
"""

import argparse

from preprocess import bidaf_setup, split_setup
import util


def main():
    parser = argparse.ArgumentParser("Download and pre-process SQuAD")
    parent_parser = argparse.ArgumentParser(add_help=False)

    util.add_data_args(parent_parser)

    subparsers = parser.add_subparsers()
    bidaf = subparsers.add_parser("bidaf", parents=[parent_parser])
    bidaf_setup.add_args(bidaf)
    bidaf.set_defaults(setup=bidaf_setup.setup)
    bidaf.set_defaults(data_sub_dir="bidaf")

    split = subparsers.add_parser("split", parents=[parent_parser])
    split_setup.add_args(split)
    split.set_defaults(setup=split_setup.setup)
    split.set_defaults(data_sub_dir=None)

    args = parser.parse_args()
    args.data_dir = util.get_data_dir(args.data_dir, args.data_sub_dir)
    util.build_data_dir_path(args)

    args.setup(args)


if __name__ == "__main__":
    main()
