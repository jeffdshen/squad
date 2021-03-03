"""Interface for (byte-level) bpe. See bpe_utils.py for implementation details.

Author:
    Jeffrey Shen
"""

import collections
import preprocess.bpe_utils as bpe_utils


def get_lines_from_file(filename):
    with open(filename, encoding="utf-8") as file:
        lines = collections.Counter(file.readlines())
    return lines


class BPE:
    def __init__(self):
        super().__init__()
        self.special_tokens = None
        self.base_vocab = None
        self.tokenizer = None
        self.vocab = None
        self.merges = None
        self.encoded_vocab = None
        self.code = None

    def build_base_vocab(self, special_tokens=["[PAD]", "[CLS]", "[SEP]", "[MASK]"]):
        self.special_tokens = special_tokens
        self._build_base_vocab()

    def _build_base_vocab(self):
        self.base_vocab = BaseVocab(self.special_tokens)
        self.tokenizer = Tokenizer(self.base_vocab)

    def build_vocab(self, lines):
        self.vocab = bpe_utils.get_vocab_from_dict(lines, self.tokenizer)

    def learn_bpe(self, max_length, l1_bs=16, l2_bs=256):
        self.merges, self.encoded_vocab = bpe_utils.learn_bpe(
            self.vocab, max_length, self.base_vocab, l1_bs=l1_bs, l2_bs=l2_bs
        )
        self._build_encoder()

    def _build_encoder(self):
        self.code = Code(self.merges, self.encoded_vocab)
        self.encoder = Encoder(self.code, self.tokenizer)

    def encode(self, line):
        return self.encoder.encode(line)

    def decode(self, tokens):
        return self.encoder.decode(tokens)

    def state_dict(self):
        return {
            "special_tokens": self.special_tokens,
            "vocab": self.vocab,
            "merges": self.merges,
            "encoded_vocab": self.encoded_vocab,
        }

    def load_state_dict(self, state_dict):
        self.special_tokens = state_dict["special_tokens"]
        self.vocab = state_dict["vocab"]
        self.merges = state_dict["merges"]
        self.encoded_vocab = state_dict["encoded_vocab"]
        if self.special_tokens is not None:
            self._build_base_vocab()

        if self.merges is not None:
            self._build_encoder()


class Encoder:
    def __init__(self, code, tokenizer):
        super().__init__()
        self.code = code
        self.tokenizer = tokenizer

    def encode(self, line):
        words = self.tokenizer.tokenize(line)
        words = [self.code.encode(word) for word in words]
        tokens = [token for word in words for token in word]
        return tokens

    def decode(self, tokens):
        tokens = self.code.decode(tokens)
        line = self.tokenizer.detokenize(tokens)
        return line


class Code:
    """Maps tokens to tokens. Encode merges tokens. Decode unmerges to base_vocab."""

    def __init__(self, merges, encoded_vocab):
        super().__init__()
        self.merge_dict = {pair: num for pair, num, _ in merges}
        self.unmerge_dict = {num: pair for pair, num, _ in merges}
        Code._precompute_unmerge(self.unmerge_dict, merges)
        self.vocab_dict = Code._precompute_vocab(self.unmerge_dict, encoded_vocab)

    @staticmethod
    def _precompute_unmerge(unmerge_dict, merges):
        for pair, num, _ in merges:
            a, b = pair
            if a in unmerge_dict:
                a = unmerge_dict[a]
            else:
                a = (a,)

            if b in unmerge_dict:
                b = unmerge_dict[b]
            else:
                b = (b,)
            unmerge_dict[num] = a + b

    @staticmethod
    def _precompute_vocab(unmerge_dict, encoded_vocab):
        vocab_dict = {}
        for tokens, _ in encoded_vocab:
            decoded_tokens = []
            for token in tokens:
                if token in unmerge_dict:
                    decoded_tokens += unmerge_dict[token]
                else:
                    decoded_tokens.append(token)
            decoded_tokens = tuple(decoded_tokens)
            vocab_dict[decoded_tokens] = tokens
        return vocab_dict

    def encode(self, word):
        if word in self.vocab_dict:
            return self.vocab_dict[word]

        while len(word) > 1:
            best_num = float("inf")
            best = None
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                if pair not in self.merge_dict:
                    continue

                num = self.merge_dict[pair]
                if num < best_num:
                    best_num = num
                    best = pair
            if best is not None:
                word = bpe_utils.replace_pair(word, best, best_num)
            else:
                break
        return word

    def decode(self, tokens):
        result = []
        for token in tokens:
            if token in self.unmerge_dict:
                result += self.unmerge_dict[token]
            else:
                result.append(token)
        return tuple(result)


class BaseVocab:
    """Encodes bytes to tokens"""

    def __init__(self, special_tokens):
        super().__init__()
        self.base_vocab = [tuple(token.encode()) for token in special_tokens]
        self.base_dict = {}
        for x in range(256):
            token = tuple(bytes([x]))
            self.base_dict[token] = len(self.base_vocab)
            self.base_vocab.append(token)

    def encode(self, token):
        """tuple of bytes -> int token"""
        return self.base_dict[token]

    def decode(self, token):
        """int token -> tuple of bytes"""
        return self.base_vocab[token]

    def __len__(self):
        return len(self.base_vocab)


class Tokenizer:
    """Takes a unicode string and tokenizes them to words of base_vocab tokens.
    Special tokens cannot be tokenized, only detokenized."""

    def __init__(self, base_vocab):
        super().__init__()
        self.base_vocab = base_vocab

    def tokenize(self, line):
        """unicode -> list of words of base_vocab tokens"""
        words = line.strip().split()
        words = [tuple((word + " ").encode("utf-8", "ignore")) for word in words]
        words = [
            tuple(self.base_vocab.encode((ind,)) for ind in word) for word in words
        ]
        return words

    def detokenize(self, tokens):
        """Flat (!) list of tokens -> unicode"""

        words = bytes(x for token in tokens for x in self.base_vocab.decode(token))
        words = words.decode("utf-8", "ignore")
        return "".join(words).strip()
