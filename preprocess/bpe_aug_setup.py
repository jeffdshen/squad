"""Pre-process SQuAD with BPE

Author:
    Jeffrey Shen
"""

import numpy as np
import os
import ujson as json

from codecs import open
from collections import defaultdict
from tqdm import tqdm
from preprocess.bpe import BPE
from zipfile import ZipFile


def load_features(out_file):
    dataset = np.load(out_file)
    return (
        dataset["context_idxs"].tolist(),
        dataset["ques_idxs"].tolist(),
        dataset["y1s"],
        dataset["y2s"],
        dataset["ids"],
    )


def load_aug(aug_file):
    with open(aug_file, "r") as file:
        return json.load(file)


def unzip(output_path):
    if os.path.exists(output_path):
        return

    zip_path = output_path + ".zip"
    if os.path.exists(zip_path):
        print(f"Unzipping {zip_path}...")
        with ZipFile(zip_path, "r") as zip_fh:
            zip_fh.extractall(output_path)


def build_features(data_type, out_file, aug_file, bpe):
    print(f"Adding {data_type} aug examples...")

    context_idxs, ques_idxs, y1s, y2s, ids = load_features(out_file)

    id_map = {}
    for i, id in enumerate(ids):
        assert str(id) not in id_map
        id_map[str(id)] = i
    total = ids.max()
    total_unfiltered = total

    ques_map = defaultdict()
    for i, ques in enumerate(ques_idxs):
        ques = [token for token in ques.tolist() if token != 0]
        ques = bpe.decode(ques)
        ques_map[ques].append(i)

    ques_limit = ques_idxs.shape[1]

    aug_examples = load_aug(aug_file)

    meta = {}
    aug_context_idxs = []
    aug_ques_idxs = []
    aug_y1s = []
    aug_y2s = []
    aug_ids = []
    gold_filtered = 0
    dup_filtered = 0
    for id, example in tqdm(aug_examples.items()):
        gold_question = example["gold_question"]
        aug_questions = example["aug_questions"]
        aug_question_texts = example["aug_question_texts"]
        assert len(gold_question) == ques_limit
        assert gold_question == ques_idxs[id_map[id]].tolist()
        aug_context = context_idxs[id_map[id]]

        for i, ques in enumerate(aug_questions):
            total_unfiltered += 1
            assert len(ques) == ques_limit

            # filter out if it equals the gold question
            if ques == gold_question:
                gold_filtered += 1
                continue

            # filter out if it equals another question with the same context
            ques_text = aug_question_texts[i]
            skip = False
            for j in ques_map[ques_text]:
                if context_idxs[j] == aug_context:
                    skip = True
                    break
            if skip:
                dup_filtered += 1
                continue

            total += 1
            aug_context_idxs.append(aug_context)
            aug_ques_idxs.append(ques)
            aug_y1s.append(-1)
            aug_y2s.append(-1)
            aug_ids.append(total)

    aug_context_idxs = np.array(aug_context_idxs)
    aug_ques_idxs = np.array(aug_ques_idxs)
    aug_y1s = np.array(aug_y1s)
    aug_y2s = np.array(aug_y2s)
    aug_ids = np.array(aug_ids)
    context_idxs = np.concatenate((context_idxs, aug_context_idxs), axis=0)
    ques_idxs = np.concatenate((ques_idxs, aug_ques_idxs), axis=0)
    y1s = np.concatenate((y1s, aug_y1s), axis=0)
    y2s = np.concatenate((y2s, aug_y2s), axis=0)
    ids = np.concatenate((ids, aug_ids), axis=0)

    np.savez_compressed(
        out_file,
        context_idxs=context_idxs,
        ques_idxs=ques_idxs,
        y1s=y1s,
        y2s=y2s,
        ids=ids,
    )

    print(f"Built {total} instances of features in total")
    meta["total_unfiltered"] = total_unfiltered
    meta["total"] = total
    meta["gold_filtered"] = gold_filtered
    meta["dup_filtered"] = dup_filtered
    return meta


def get_bpe(args):
    bpe = BPE()
    with open(args.bpe_file, "r") as file:
        bpe.load_state_dict(json.load(file))
    return bpe


def preprocess(args):
    # Process training set and use it to construct bpe
    bpe = get_bpe(args)

    train_meta = build_features("train", args.train_record_file, args.train_aug_file, bpe)
    print("Meta: {}".format(train_meta))


def setup(args):
    preprocess(args)


def add_args(parser):
    return parser
