"""Train a model on SQuAD.

Author:
    Chris Chute (chute@stanford.edu)
"""
import argparse

from trainer import bidaf_trainer
import util


def main():
    parser = argparse.ArgumentParser("Train a model on SQuAD")
    parent_parser = argparse.ArgumentParser(add_help=False)

    util.add_data_args(parent_parser)
    util.add_train_args(parent_parser)
    subparsers = parser.add_subparsers()

    bidaf = subparsers.add_parser("bidaf", parents=[parent_parser])
    bidaf_trainer.add_train_args(bidaf)
    bidaf.set_defaults(train=bidaf_trainer.train)
    bidaf.set_defaults(data_sub_dir="bidaf")

    args = parser.parse_args()
    if args.metric_name == "NLL":
        # Best checkpoint is the one that minimizes negative log-likelihood
        args.maximize_metric = False
    elif args.metric_name in ("EM", "F1"):
        # Best checkpoint is the one that maximizes EM or F1
        args.maximize_metric = True
    else:
        raise ValueError(f'Unrecognized metric name: "{args.metric_name}"')

    args.save_dir = util.get_save_dir(args.save_dir, args.name, training=True)
    args.data_dir = util.get_data_dir(args.data_dir, args.data_sub_dir)
    util.build_data_dir_path(args)

    train = args.train
    del args.train
    train(args)


if __name__ == "__main__":
    main()
