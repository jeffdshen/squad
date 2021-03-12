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


def add_subparser(name, data_sub_dir, subparsers, parent_parser, module):
    subparser = subparsers.add_parser(name, parents=[parent_parser])
    module.add_args(subparser)
    subparser.set_defaults(setup=module.setup)
    subparser.add_argument("--data_sub_dir", type=str, default=data_sub_dir)


def main():
    parser = argparse.ArgumentParser("Download and pre-process SQuAD")
    parent_parser = argparse.ArgumentParser(add_help=False)

    util.add_data_args(parent_parser)

    subparsers = parser.add_subparsers()

    add_subparser("bidaf", "bidaf", subparsers, parent_parser, bidaf_setup)
    add_subparser("bpe", "bpe", subparsers, parent_parser, bpe_setup)

    args = parser.parse_args()
    args.data_dir = util.get_data_dir(args.data_dir, args.data_sub_dir)
    util.build_data_dir_path(args)

    args.setup(args)


if __name__ == "__main__":
    main()
