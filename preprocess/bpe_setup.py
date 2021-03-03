"""Pre-process SQuAD with BPE

Author:
    Jeffrey Shen
"""

import numpy as np
import os
import ujson as json

from codecs import open
from collections import Counter
from tqdm import tqdm
from preprocess.bpe import BPE


def process_bpe_file(filename, data_type, max_length):
    print(f"Getting vocab from {data_type} examples...")
    lines = Counter()
    with open(filename, "r") as file:
        source = json.load(file)
        for article in tqdm(source["data"]):
            for paragraph in article["paragraphs"]:
                context = paragraph["context"]
                qas = paragraph["qas"]

                # Weight by number of questions it has
                lines[context] += len(qas)
                for qa in qas:
                    question = qa["question"]
                    lines[question] += 1

    bpe = BPE()
    bpe.build_base_vocab()
    bpe.build_vocab(lines)
    print("Learning bpe on {} words for {}".format(len(bpe.vocab), data_type))
    bpe.learn_bpe(max_length)
    return bpe


def process_file(filename, data_type, bpe):
    print(f"Pre-processing {data_type} examples...")
    examples = []
    eval_examples = {}
    total = 0

    with open(filename, "r") as file:
        source = json.load(file)
        for article in tqdm(source["data"]):
            for para in article["paragraphs"]:
                context = para["context"]
                qas = para["qas"]
                context_tokens = bpe.encode(context)
                spans = bpe.get_spans(context_tokens, context)
                for qa in qas:
                    total += 1
                    ques = qa["question"]
                    ques_tokens = bpe.encode(ques)

                    y1s, y2s = [], []
                    answer_texts = []
                    for answer in qa["answers"]:
                        answer_text = answer["text"]
                        answer_start = answer["answer_start"]
                        answer_end = answer_start + len(answer_text)
                        answer_texts.append(answer_text)
                        answer_span = []
                        for idx, span in enumerate(spans):
                            if span[0] < answer_end and answer_start < span[1]:
                                answer_span.append(idx)
                        y1, y2 = answer_span[0], answer_span[-1]
                        y1s.append(y1)
                        y2s.append(y2)
                    example = {
                        "context_tokens": context_tokens,
                        "ques_tokens": ques_tokens,
                        "y1s": y1s,
                        "y2s": y2s,
                        "id": total,
                    }
                    examples.append(example)
                    eval_examples[str(total)] = {
                        "context": context,
                        "question": ques,
                        "spans": spans,
                        "answers": answer_texts,
                        "uuid": qa["id"],
                    }
        print(f"{len(examples)} questions in total")
    return examples, eval_examples


def build_features(examples, data_type, out_file):
    print(f"Converting {data_type} examples to indices...")
    para_limit = max(len(example["context_tokens"]) for example in examples)
    ques_limit = max(len(example["ques_tokens"]) for example in examples)

    total = 0
    meta = {"context_lengths": [], "ques_lengths": []}
    context_idxs = []
    ques_idxs = []
    y1s = []
    y2s = []
    ids = []
    for n, example in tqdm(enumerate(examples)):
        total += 1

        # Zero is the padding index
        context_idx = np.zeros([para_limit], dtype=np.int32)
        ques_idx = np.zeros([ques_limit], dtype=np.int32)

        for i, token in enumerate(example["context_tokens"]):
            context_idx[i] = token
        context_idxs.append(context_idx)

        for i, token in enumerate(example["ques_tokens"]):
            ques_idx[i] = token
        ques_idxs.append(ques_idx)

        if is_answerable(example):
            start, end = example["y1s"][-1], example["y2s"][-1]
        else:
            start, end = -1, -1

        y1s.append(start)
        y2s.append(end)
        ids.append(example["id"])

    np.savez_compressed(
        out_file,
        context_idxs=np.array(context_idxs),
        ques_idxs=np.array(ques_idxs),
        y1s=np.array(y1s),
        y2s=np.array(y2s),
        ids=np.array(ids),
    )
    print(f"Built {total} instances of features in total")
    meta["total"] = total
    return meta


def is_answerable(example):
    return len(example["y2s"]) > 0 and len(example["y1s"]) > 0


def try_load(filename):
    if not os.path.isfile(filename):
        return None

    with open(filename, "r") as file:
        return json.load(file)


def save(filename, obj, message=None):
    if message is not None:
        print(f"Saving {message}...")
        with open(filename, "w") as fh:
            json.dump(obj, fh)
    else:
        raise RuntimeError("Message missing")


def preprocess(args):
    # Process training set and use it to construct bpe
    bpe_state_dict = try_load(args.bpe_file)
    if bpe_state_dict is None:
        bpe = process_bpe_file(args.train_file, "train", args.max_tokens)
        save(args.bpe_file, bpe.state_dict(), "BPE")
    else:
        bpe = BPE()
        bpe.load_state_dict(bpe_state_dict)

    train_examples, train_eval = process_file(args.train_file, "train", bpe)
    train_meta = build_features(train_examples, "train", args.train_record_file)
    save(args.train_eval_file, train_eval, message="train eval")
    save(args.train_meta_file, train_meta, message="train meta")

    # Process dev and test sets
    dev_examples, dev_eval = process_file(args.dev_file, "dev", bpe)
    dev_meta = build_features(dev_examples, "dev", args.dev_record_file)
    save(args.dev_eval_file, dev_eval, message="dev eval")
    save(args.dev_meta_file, dev_meta, message="dev meta")

    if args.include_test_examples:
        test_examples, test_eval = process_file(args.test_file, "test", bpe)
        test_meta = build_features(test_examples, "test", args.test_record_file)
        save(args.test_eval_file, test_eval, message="test eval")
        save(args.test_meta_file, test_meta, message="test meta")


def setup(args):
    preprocess(args)


def add_args(parser):
    """Get arguments needed in setup.py."""
    parser.add_argument("--bpe_file", type=str, default="./data/bpe.json")
    parser.add_argument(
        "--train_file",
        type=str,
        default="./data/train-v2.0.json",
    )
    parser.add_argument(
        "--dev_file",
        type=str,
        default="./data/dev-v2.0.json",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default="./data/test-v2.0.json",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=50000,
        help="Max size of BPE tokens vocab",
    )
    parser.add_argument(
        "--include_test_examples",
        type=lambda s: s.lower().startswith("t"),
        default=True,
        help="Process examples from the test set",
    )

    return parser
