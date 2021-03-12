"""Train a model on SQuAD.

Author:
    Chris Chute (chute@stanford.edu)
"""
import argparse

from trainer import (
    bidaf_trainer,
    glove_transformer_trainer,
    roberta_pretrainer,
    electra_pretrainer,
    roberta_finetune,
)
import util


def add_subparser(name, data_sub_dir, subparsers, parent_parser, module):
    subparser = subparsers.add_parser(name, parents=[parent_parser])
    module.add_train_args(subparser)
    subparser.set_defaults(train=module.train)
    subparser.add_argument("--data_sub_dir", type=str, default=data_sub_dir)


def main():
    parser = argparse.ArgumentParser("Train a model on SQuAD")
    parent = argparse.ArgumentParser(add_help=False)

    util.add_data_args(parent)
    util.add_train_args(parent)
    subparsers = parser.add_subparsers()

    add_subparser("bidaf", "bidaf", subparsers, parent, bidaf_trainer)
    add_subparser(
        "glove_transformer", "bidaf", subparsers, parent, glove_transformer_trainer
    )
    add_subparser("roberta_pretrain", "bpe", subparsers, parent, roberta_pretrainer)
    add_subparser("electra_pretrain", "bpe", subparsers, parent, electra_pretrainer)
    add_subparser("roberta_finetune", "bpe", subparsers, parent, roberta_finetune)

    args = parser.parse_args()
    if args.metric_name in ("NLL", "NLL_E"):
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
