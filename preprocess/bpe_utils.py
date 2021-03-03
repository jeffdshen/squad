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

    vocab = list(vocab.items())
    vocab.sort(key=lambda k: k[1], reverse=True)
    return vocab


def get_stats(vocab, l1_block_size, l2_block_size):
    pairs = collections.Counter()
    l1 = []
    l2 = []
    for ind, (word, count) in enumerate(vocab):
        if ind % l1_block_size == 0:
            l1.append(collections.Counter())
        if ind % l2_block_size == 0:
            l2.append(collections.Counter())
        add_stats_for_word(word, count, l1[-1])
        add_stats_for_word(word, count, l2[-1])
    for block in l1:
        pairs.update(block)
    return pairs, l1, l2


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


def merge_vocab(vocab, best, num, pairs, l1, l2, l1_bs, l2_bs):
    hit1 = 0
    miss1 = 0
    hit2 = 0
    miss2 = 0
    for i in range(0, len(vocab), l2_bs):
        c2 = l2[i // l2_bs]
        hit2 += 1
        if best not in c2:
            continue
        if c2[best] == 0:
            continue
        
        hit2 -= 1
        miss2 += 1

        for j in range(i, min(len(vocab), i + l2_bs), l1_bs):
            c1 = l1[i // l1_bs]
            hit1 += 1
            if best not in c1:
                continue
            if c1[best] == 0:
                continue
            hit1 -= 1
            miss1 += 1

            for v in range(j, min(len(vocab), j + l1_bs), 1):
                word, count = vocab[v]

                if not contains_pair(word, best):
                    continue

                diff = collections.Counter()
                add_stats_for_word(word, count, diff)
                next = replace_pair(word, best, num)
                sub = collections.Counter()
                add_stats_for_word(next, count, sub)
                subtract_counters(diff, sub)
                subtract_counters(l1, diff)
                subtract_counters(l2, diff)
                subtract_counters(pairs, diff)

                vocab[v] = (next, count)

    return hit1, miss1, hit2, miss2


def learn_bpe(vocab, max_length, base_vocab, l1_block_size=16, l2_block_size=256):
    """Performs bpe and returns the merge list.

    Returns:
        merges (list): (id pair, id result, count)
        vocab (list): (encoded word, count)
    """
    vocab = copy.deepcopy(vocab)
    last = len(base_vocab)
    merges = []
    pairs, l1, l2 = get_stats(vocab, l1_block_size, l2_block_size)

    pbar = tqdm(range(last, max_length))
    for i in pbar:
        if len(pairs) == 0:
            break
        best = pairs.most_common(1)[0][0]
        merges.append((best, i, pairs[best]))
        hit1, miss1, hit2, miss2 = merge_vocab(
            vocab, best, i, pairs, l1, l2, l1_block_size, l2_block_size
        )
        pbar.set_postfix({"hit2": hit2, "miss2": miss2, "hit1": hit1, "miss1": miss1, "pairs": len(pairs)})

    return merges, vocab