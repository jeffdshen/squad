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

from preprocess import bidaf_setup

def main():
    parser = argparse.ArgumentParser('Download and pre-process SQuAD')
    subparsers = parser.add_subparsers()
    bidaf = subparsers.add_parser('bidaf')
    bidaf_setup.add_args(bidaf)
    bidaf.set_defaults(func=bidaf_setup.setup)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
