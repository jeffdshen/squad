"""Download and pre-process SQuAD and GloVe.

Usage:
    > source activate squad
    > python setup.py

Pre-processing code adapted from:
    > https://github.com/HKUST-KnowComp/R-Net/blob/master/prepro.py

Author:
    Chris Chute (chute@stanford.edu)
"""

import numpy as np
import os
import spacy
import ujson as json
import urllib.request

from codecs import open
from collections import Counter
from subprocess import run
from tqdm import tqdm
from zipfile import ZipFile


def download_url(url, output_path, show_progress=True):
    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    if show_progress:
        # Download with a progress bar
        with DownloadProgressBar(
            unit="B", unit_scale=True, miniters=1, desc=url.split("/")[-1]
        ) as t:
            urllib.request.urlretrieve(
                url, filename=output_path, reporthook=t.update_to
            )
    else:
        # Simple download with no progress bar
        urllib.request.urlretrieve(url, output_path)


def url_to_data_path(data_dir, url):
    return os.path.join(data_dir, url.split("/")[-1])


def download(args):
    downloads = [
        # Can add other downloads here (e.g., other word vectors)
        ("GloVe word vectors", args.glove_url),
    ]

    for name, url in downloads:
        output_path = url_to_data_path(args.data_dir, url)
        if not os.path.exists(output_path):
            print(f"Downloading {name}...")
            download_url(url, output_path)

        if os.path.exists(output_path) and output_path.endswith(".zip"):
            extracted_path = output_path.replace(".zip", "")
            if not os.path.exists(extracted_path):
                print(f"Unzipping {name}...")
                with ZipFile(output_path, "r") as zip_fh:
                    zip_fh.extractall(extracted_path)

    print("Downloading spacy language model...")
    run(["python", "-m", "spacy", "download", "en"])


def word_tokenize(nlp, sent):
    doc = nlp(sent)
    return [token.text for token in doc]


def convert_idx(text, tokens):
    current = 0
    spans = []
    for token in tokens:
        current = text.find(token, current)
        if current < 0:
            print(f"Token {token} cannot be found")
            raise Exception()
        spans.append((current, current + len(token)))
        current += len(token)
    return spans


def process_file(filename, data_type, nlp):
    print(f"Pre-processing {data_type} examples...")
    word_counter, char_counter = Counter(), Counter()
    examples = []
    eval_examples = {}
    total = 0
    with open(filename, "r") as fh:
        source = json.load(fh)
        for article in tqdm(source["data"]):
            for para in article["paragraphs"]:
                context = para["context"].replace("''", '" ').replace("``", '" ')
                context_tokens = word_tokenize(nlp, context)
                context_chars = [list(token) for token in context_tokens]
                spans = convert_idx(context, context_tokens)
                for token in context_tokens:
                    word_counter[token] += len(para["qas"])
                    for char in token:
                        char_counter[char] += len(para["qas"])
                for qa in para["qas"]:
                    total += 1
                    ques = qa["question"].replace("''", '" ').replace("``", '" ')
                    ques_tokens = word_tokenize(nlp, ques)
                    ques_chars = [list(token) for token in ques_tokens]
                    for token in ques_tokens:
                        word_counter[token] += 1
                        for char in token:
                            char_counter[char] += 1
                    y1s, y2s = [], []
                    answer_texts = []
                    for answer in qa["answers"]:
                        answer_text = answer["text"]
                        answer_start = answer["answer_start"]
                        answer_end = answer_start + len(answer_text)
                        answer_texts.append(answer_text)
                        answer_span = []
                        for idx, span in enumerate(spans):
                            if not (answer_end <= span[0] or answer_start >= span[1]):
                                answer_span.append(idx)
                        y1, y2 = answer_span[0], answer_span[-1]
                        y1s.append(y1)
                        y2s.append(y2)
                    example = {
                        "context_tokens": context_tokens,
                        "context_chars": context_chars,
                        "ques_tokens": ques_tokens,
                        "ques_chars": ques_chars,
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
    return examples, eval_examples, word_counter, char_counter


def get_embedding(
    counter, data_type, limit=-1, emb_file=None, vec_size=None, num_vectors=None
):
    print(f"Pre-processing {data_type} vectors...")
    embedding_dict = {}
    filtered_elements = [k for k, v in counter.items() if v > limit]
    if emb_file is not None:
        assert vec_size is not None
        with open(emb_file, "r", encoding="utf-8") as fh:
            for line in tqdm(fh, total=num_vectors):
                array = line.split()
                word = "".join(array[0:-vec_size])
                vector = list(map(float, array[-vec_size:]))
                if word in counter and counter[word] > limit:
                    embedding_dict[word] = vector
        print(
            f"{len(embedding_dict)} / {len(filtered_elements)} tokens have corresponding {data_type} embedding vector"
        )
    else:
        assert vec_size is not None
        for token in filtered_elements:
            embedding_dict[token] = [
                np.random.normal(scale=0.1) for _ in range(vec_size)
            ]
        print(
            f"{len(filtered_elements)} tokens have corresponding {data_type} embedding vector"
        )

    NULL = "--NULL--"
    OOV = "--OOV--"
    token2idx_dict = {token: idx for idx, token in enumerate(embedding_dict.keys(), 2)}
    token2idx_dict[NULL] = 0
    token2idx_dict[OOV] = 1
    embedding_dict[NULL] = [0.0 for _ in range(vec_size)]
    embedding_dict[OOV] = [0.0 for _ in range(vec_size)]
    idx2emb_dict = {idx: embedding_dict[token] for token, idx in token2idx_dict.items()}
    emb_mat = [idx2emb_dict[idx] for idx in range(len(idx2emb_dict))]
    return emb_mat, token2idx_dict


def is_answerable(example):
    return len(example["y2s"]) > 0 and len(example["y1s"]) > 0


def build_features(
    args, examples, data_type, out_file, word2idx_dict, char2idx_dict, is_test=True
):
    para_limit = args.test_para_limit if is_test else args.para_limit
    ques_limit = args.test_ques_limit if is_test else args.ques_limit
    ans_limit = args.ans_limit
    char_limit = args.char_limit

    def drop_example(ex, is_test_=False):
        if is_test_:
            drop = False
        else:
            drop = (
                len(ex["context_tokens"]) > para_limit
                or len(ex["ques_tokens"]) > ques_limit
                or (is_answerable(ex) and ex["y2s"][0] - ex["y1s"][0] > ans_limit)
            )

        return drop

    print(f"Converting {data_type} examples to indices...")
    total = 0
    total_ = 0
    meta = {}
    context_idxs = []
    context_char_idxs = []
    ques_idxs = []
    ques_char_idxs = []
    y1s = []
    y2s = []
    ids = []
    for n, example in tqdm(enumerate(examples)):
        total_ += 1

        if drop_example(example, is_test):
            continue

        total += 1

        def _get_word(word):
            for each in (word, word.lower(), word.capitalize(), word.upper()):
                if each in word2idx_dict:
                    return word2idx_dict[each]
            return 1

        def _get_char(char):
            if char in char2idx_dict:
                return char2idx_dict[char]
            return 1

        context_idx = np.zeros([para_limit], dtype=np.int32)
        context_char_idx = np.zeros([para_limit, char_limit], dtype=np.int32)
        ques_idx = np.zeros([ques_limit], dtype=np.int32)
        ques_char_idx = np.zeros([ques_limit, char_limit], dtype=np.int32)

        for i, token in enumerate(example["context_tokens"]):
            context_idx[i] = _get_word(token)
        context_idxs.append(context_idx)

        for i, token in enumerate(example["ques_tokens"]):
            ques_idx[i] = _get_word(token)
        ques_idxs.append(ques_idx)

        for i, token in enumerate(example["context_chars"]):
            for j, char in enumerate(token):
                if j == char_limit:
                    break
                context_char_idx[i, j] = _get_char(char)
        context_char_idxs.append(context_char_idx)

        for i, token in enumerate(example["ques_chars"]):
            for j, char in enumerate(token):
                if j == char_limit:
                    break
                ques_char_idx[i, j] = _get_char(char)
        ques_char_idxs.append(ques_char_idx)

        if is_answerable(example):
            start, end = example["y1s"][-1], example["y2s"][-1]
        else:
            start, end = -1, -1

        y1s.append(start)
        y2s.append(end)
        ids.append(example["id"])

    np.savez(
        out_file,
        context_idxs=np.array(context_idxs),
        context_char_idxs=np.array(context_char_idxs),
        ques_idxs=np.array(ques_idxs),
        ques_char_idxs=np.array(ques_char_idxs),
        y1s=np.array(y1s),
        y2s=np.array(y2s),
        ids=np.array(ids),
    )
    print(f"Built {total} / {total_} instances of features in total")
    meta["total"] = total
    return meta


def save(filename, obj, message=None):
    if message is not None:
        print(f"Saving {message}...")
        with open(filename, "w") as fh:
            json.dump(obj, fh)


def pre_process(args, nlp):
    # Process training set and use it to decide on the word/character vocabularies
    trainx_examples, trainx_eval, word_counter, char_counter = process_file(
        args.trainx_file, "trainx", nlp
    )
    word_emb_mat, word2idx_dict = get_embedding(
        word_counter,
        "word",
        emb_file=args.glove_file,
        vec_size=args.glove_dim,
        num_vectors=args.glove_num_vecs,
    )
    char_emb_mat, char2idx_dict = get_embedding(
        char_counter, "char", emb_file=None, vec_size=args.char_dim
    )

    # Process dev and test sets
    trainx_meta = build_features(
        args,
        trainx_examples,
        "trainx",
        args.trainx_record_file,
        word2idx_dict,
        char2idx_dict,
        is_test=False,
    )
    save(args.trainx_eval_file, trainx_eval, message="trainx eval")
    save(args.trainx_meta_file, trainx_meta, message="trainx meta")

    traind_examples, traind_eval, _, _ = process_file(args.traind_file, "traind", nlp)
    traind_meta = build_features(
        args,
        traind_examples,
        "traind",
        args.traind_record_file,
        word2idx_dict,
        char2idx_dict,
    )
    save(args.traind_eval_file, traind_eval, message="traind eval")
    save(args.traind_meta_file, traind_meta, message="traind meta")

    dev_examples, dev_eval, _, _ = process_file(args.dev_file, "dev", nlp)
    dev_meta = build_features(
        args, dev_examples, "dev", args.dev_record_file, word2idx_dict, char2idx_dict
    )
    save(args.dev_eval_file, dev_eval, message="dev eval")
    save(args.dev_meta_file, dev_meta, message="dev meta")

    if args.include_test_examples:
        test_examples, test_eval = process_file(
            args.test_file, "test", word_counter, char_counter, nlp
        )
        test_meta = build_features(
            args,
            test_examples,
            "test",
            args.test_record_file,
            word2idx_dict,
            char2idx_dict,
        )
        save(args.test_eval_file, test_eval, message="test eval")
        save(args.test_meta_file, test_meta, message="test meta")

    save(args.word_emb_file, word_emb_mat, message="word embedding")
    save(args.char_emb_file, char_emb_mat, message="char embedding")
    save(args.word2idx_file, word2idx_dict, message="word dictionary")
    save(args.char2idx_file, char2idx_dict, message="char dictionary")


def setup(args):
    # Download resources
    download(args)

    # Import spacy language model
    nlp = spacy.blank("en")

    # Preprocess dataset
    glove_dir = url_to_data_path(args.data_dir, args.glove_url.replace(".zip", ""))
    glove_ext = f".txt" if glove_dir.endswith("d") else f".{args.glove_dim}d.txt"
    args.glove_file = os.path.join(glove_dir, os.path.basename(glove_dir) + glove_ext)
    pre_process(args, nlp)


def add_args(parser):
    """Get arguments needed in setup.py."""
    parser.add_argument(
        "--trainx_file", type=str, default="./data/trainx-v2.0.json",
    )
    parser.add_argument(
        "--traind_file", type=str, default="./data/traind-v2.0.json",
    )
    parser.add_argument(
        "--dev_file", type=str, default="./data/dev-v2.0.json",
    )
    parser.add_argument(
        "--test_file", type=str, default="./data/test-v2.0.json",
    )
    parser.add_argument(
        "--glove_url",
        type=str,
        default="http://nlp.stanford.edu/data/glove.840B.300d.zip",
    )
    parser.add_argument(
        "--para_limit", type=int, default=400, help="Max number of words in a paragraph"
    )
    parser.add_argument(
        "--ques_limit",
        type=int,
        default=50,
        help="Max number of words to keep from a question",
    )
    parser.add_argument(
        "--test_para_limit",
        type=int,
        default=1000,
        help="Max number of words in a paragraph at test time",
    )
    parser.add_argument(
        "--test_ques_limit",
        type=int,
        default=100,
        help="Max number of words in a question at test time",
    )
    parser.add_argument(
        "--char_dim",
        type=int,
        default=64,
        help="Size of char vectors (char-level embeddings)",
    )
    parser.add_argument(
        "--glove_dim", type=int, default=300, help="Size of GloVe word vectors to use"
    )
    parser.add_argument(
        "--glove_num_vecs", type=int, default=2196017, help="Number of GloVe vectors"
    )
    parser.add_argument(
        "--ans_limit",
        type=int,
        default=30,
        help="Max number of words in a training example answer",
    )
    parser.add_argument(
        "--char_limit",
        type=int,
        default=16,
        help="Max number of chars to keep from a word",
    )
    parser.add_argument(
        "--include_test_examples",
        type=lambda s: s.lower().startswith("t"),
        default=True,
        help="Process examples from the test set",
    )

    return parser
