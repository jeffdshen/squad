"""Train a model on SQuAD.

Author:
    Chris Chute (chute@stanford.edu)
"""
import argparse

from trainer import bidaf_trainer

def main():
    parser = argparse.ArgumentParser('Train a model on SQuAD')
    subparsers = parser.add_subparsers()

    bidaf = subparsers.add_parser('bidaf')
    bidaf_trainer.add_train_args(bidaf)
    bidaf.set_defaults(train=bidaf_trainer.train)

    args = parser.parse_args()
    if args.metric_name == 'NLL':
        # Best checkpoint is the one that minimizes negative log-likelihood
        args.maximize_metric = False
    elif args.metric_name in ('EM', 'F1'):
        # Best checkpoint is the one that maximizes EM or F1
        args.maximize_metric = True
    else:
        raise ValueError(f'Unrecognized metric name: "{args.metric_name}"')

    args.train(args)

if __name__ == '__main__':
    main()
