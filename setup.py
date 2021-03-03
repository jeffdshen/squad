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

from preprocess import bidaf_setup, bpe_setup
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

    subparsers = parser.add_subparsers()
    bpe = subparsers.add_parser("bpe", parents=[parent_parser])
    bpe_setup.add_args(bpe)
    bpe.set_defaults(setup=bpe_setup.setup)
    bpe.set_defaults(data_sub_dir="bpe")

    args = parser.parse_args()
    args.data_dir = util.get_data_dir(args.data_dir, args.data_sub_dir)
    util.build_data_dir_path(args)

    args.setup(args)


if __name__ == "__main__":
    main()
