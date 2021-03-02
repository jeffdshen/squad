"""Implementation of (byte-level) bpe.

The tokenizer splits by spaces to construct a vocabulary. All words then use space
as an end of word symbol. All words are encoded to bytes via utf-8, after which
bpe can be applied.

The implementation uses a cache for blocks of words to quickly filter out blocks
without specific pairs.

Author:
    Jeffrey Shen
"""

import collections


def get_vocab_from_dict(lines, special_tokens):
    """Gets a vocab from a dict of lines to their count.
    
    Returns:
        vocab (list): (word as a tuple of int, count), sorted descending by count
        base_vocab (list): byte strings for each token
    """
    vocab = collections.Counter()

    for line, count in lines.items():
        words = line.strip().split()
        words = [tuple((word + " ").encode()) for word in words]
        for word in words:
            vocab[word] += count

    base_vocab = [token.encode() for token in special_tokens]
    vocab = list(vocab.items())
    vocab = [
        (tuple(c + len(special_tokens) for c in word), count) for (word, count) in vocab
    ]
    vocab.sort(lambda k: k[1], reverse=True)
    for x in range(256):
        base_vocab.append(bytes([x]))

    return vocab, base_vocab


def get_vocab_from_file(filename, special_tokens):
    with open(filename, encoding="utf-8") as file:
        lines = collections.Counter(file.readlines())
    return get_vocab_from_dict(lines, special_tokens)


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


def merge_vocab(vocab, best, num, pairs, cache, block_size=256):
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


def bpe(vocab, max_length, base_vocab, block_size=256):
    vocab = vocab.copy()
    last = len(base_vocab)
    merges = []
    pairs, cache = get_stats(vocab, block_size)
    merge_count = []

    for i in range(last, max_length):
        if len(pairs) == 0:
            break
        best = pairs.most_common(1)[0][0]
        merges.append((best, pairs[best]))
        merge_vocab(vocab, best, i, pairs, cache, block_size)
        if i % 100 == 0:
            print(i)

    return merges


def get_merge_dict(merges, base_vocab):
    return {k: (i + len(base_vocab), v) for i, (k, v) in enumerate(merges)}


def bpe_encode_word(word, merge_dict):
    while len(word) > 1:
        mn = float("inf")
        best = None
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            if pair not in merge_dict:
                continue

            num = merge_dict[pair][0]
            if num < mn:
                mn = num
                best = pair
        if best is not None:
            word = replace_pair(word, best, mn)
        else:
            break
    return word


def bpe_encode(line, base_vocab_dict, merge_dict):
    words = line.strip().split()
    words = [tuple((word + " ").encode()) for word in words]
    words = [tuple(base_vocab_dict[(ind,)] for ind in word) for word in words]
    return [bpe_encode_word(word, merge_dict) for word in words]


def bpe_decode_word(next, merge_dict):
    word = []
    next = list(next)
    while len(next) > 0:
        k = next.pop()
        if k in merge_dict:
            next.append(merge_dict[k][0][0])
            next.append(merge_dict[k][0][1])
        else:
            word.append(k)
    word.reverse()
    return tuple(word)


def bpe_decode(encoded, base_vocab_dict, merge_dict):
    words = [bpe_decode_word(next, merge_dict) for next in encoded]
    words = [bytes(x for ind in word for x in base_vocab_dict[ind]) for word in words]
    words = [word.decode() for word in words]
    return "".join(words).strip()