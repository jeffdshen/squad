"""Implementation of (byte-level) bpe.

The tokenizer splits by spaces to construct a vocabulary. All words then use space
as an end of word symbol. All words are encoded to bytes via utf-8, after which
bpe can be applied.

The implementation uses a cache for blocks of words to quickly filter out blocks
without specific pairs.

vocab - during compression, refers to list of (word, count)
word - during compression, this is a tuple of ids
base_vocab - this is a list of byte strings, the index is the id
pairs - during compression, pair -> count
cache - during compression, list of pair -> count for each block

Author:
    Jeffrey Shen
"""

import collections
import copy
from tqdm import tqdm

def get_vocab_from_dict(lines, tokenizer):
    """Gets a vocab from a dict of lines to their count.

    Returns:
        vocab (list): (word as a tuple of int, count), sorted descending by count
    """
    vocab = collections.Counter()

    for line, count in lines.items():
        words = tokenizer.tokenize(line)
        for word in words:
            vocab[word] += count

    vocab.sort(lambda k: k[1], reverse=True)
    return vocab


def get_stats(vocab, block_size):
    pairs = collections.Counter()
    cache = []
    for ind, (word, count) in enumerate(vocab):
        if ind % block_size == 0:
            cache.append(collections.Counter())
        add_stats_for_word(word, count, cache[-1])
    for block in cache:
        pairs.update(block)
    return pairs, cache


def add_stats_for_word(word, count, counter):
    for i in range(len(word) - 1):
        counter[word[i], word[i + 1]] += count


def contains_pair(word, pair):
    for i in range(len(word) - 1):
        if (word[i], word[i + 1]) == pair:
            return True
    return False


def replace_pair(word, pair, num):
    next = []
    for i in range(len(word)):
        next.append(word[i])
        if len(next) < 2:
            continue
        if (next[-2], next[-1]) == pair:
            next.pop()
            next[-1] = num
    return tuple(next)


def subtract_counters(counter_a, counter_b):
    counter_a.subtract(counter_b)
    for k in counter_b:
        if k in counter_a and counter_a[k] == 0:
            del counter_a[k]


def merge_vocab(vocab, best, num, pairs, cache, block_size):
    for ind in range(0, len(vocab), block_size):
        c = cache[ind // block_size]
        if best not in c or c[best] == 0:
            continue

        for v in range(ind, min(len(vocab), ind + block_size)):
            word, count = vocab[v]

            if not contains_pair(word, best):
                continue

            diff = collections.Counter()
            add_stats_for_word(word, count, diff)
            next = replace_pair(word, best, num)
            sub = collections.Counter()
            add_stats_for_word(next, count, sub)
            subtract_counters(diff, sub)
            subtract_counters(c, diff)
            subtract_counters(pairs, diff)

            vocab[v] = (next, count)


def learn_bpe(vocab, max_length, base_vocab, block_size=256):
    """Performs bpe and returns the merge list.

    Returns:
        merges (list): (id pair, id result, count)
        vocab (list): (encoded word, count)
    """
    vocab = copy.deepcopy(vocab)
    last = len(base_vocab)
    merges = []
    pairs, cache = get_stats(vocab, block_size)

    for i in tqdm(range(last, max_length)):
        if len(pairs) == 0:
            break
        best = pairs.most_common(1)[0][0]
        merges.append((best, i, pairs[best]))
        merge_vocab(vocab, best, i, pairs, cache, block_size)

    return merges, vocab