"""Split train SQuAD into train and train-dev.

Author:
    Jeffrey Shen
"""

import copy
import os
import spacy
import ujson as json

import random
from codecs import open
from tqdm import tqdm


def count_stats(filename):
    with open(filename, "r") as file:
        source = json.load(file)
        stats = {
            "articles": 0,
            "paragraphs": 0,
            "qas": 0,
            "ans": 0,
            "no_ans": 0,
        }
        articles = source["data"]
        for article in articles:
            for paragraph in article["paragraphs"]:
                qas = paragraph["qas"]
                for qa in qas:
                    if qa["is_impossible"]:
                        stats["ans"] += 1
                    else:
                        stats["no_ans"] += 1
                    stats["qas"] += 1
                stats["paragraphs"] += 1
            stats["articles"] += 1
    return stats


def split(source, rand, num):
    def has_no_ans(article):
        for paragraph in article["paragraphs"]:
            qas = paragraph["qas"]
            for qa in qas:
                if qa["is_impossible"]:
                    return True
        return False

    articles = source["data"]
    good = [article for article in articles if has_no_ans(article)]
    bad = [article for article in articles if not has_no_ans(article)]

    rand.shuffle(good)
    bad += good[num:]
    good = good[:num]

    source["data"] = good
    good = copy.deepcopy(source)
    source["data"] = bad
    bad = copy.deepcopy(source)
    return good, bad


def save(filename, obj, message):
    print(f"Saving {message}...")
    with open(filename, "w") as fh:
        json.dump(obj, fh)


def setup(args):
    dev_stats = count_stats(args.dev_url)
    train_stats = count_stats(args.train_url)

    print(f"Splitting train examples")
    with open(args.train_url, "r") as file:
        source = json.load(file)
        traind, trainx = split(source, random.Random(args.seed), dev_stats["articles"])

    save(args.trainx_url, trainx, "train x")
    save(args.traind_url, traind, 'train dev')

    trainx_stats = count_stats(args.trainx_url)
    traind_stats = count_stats(args.traind_url)
    print("dev_stats:", dev_stats)
    print("train_stats:", train_stats)
    print("trainx_stats:", trainx_stats)
    print("traind_stats:", traind_stats)


def add_args(parser):
    """Get arguments needed in setup.py."""
    parser.add_argument(
        "--train_url", type=str, default="./data/train-v2.0.json",
    )
    parser.add_argument(
        "--trainx_url", type=str, default="./data/trainx-v2.0.json",
    )
    parser.add_argument(
        "--traind_url", type=str, default="./data/traind-v2.0.json",
    )
    parser.add_argument(
        "--dev_url", type=str, default="./data/dev-v2.0.json",
    )
    parser.add_argument(
        "--seed", type=int, default=224, help="Random seed for selecting split."
    )

    return parser
