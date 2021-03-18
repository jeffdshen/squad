"""Stanford Question Answering Dataset (SQuAD).

Includes MLM and QA tasks.
Author:
    Jeffrey Shen
"""
import torch
import torch.utils.data as data
import numpy as np
import random


class MLM(data.IterableDataset):
    """
    Each item in the dataset is a tuple with the following entries (in order):
        - x: Masked blocks of text, starting with [CLS], separated by [SEP]
        - y: Target blocks of text, starting with [CLS], separated by [SEP]

    Args:
        data_path (str): Path to .npz file containing pre-processed dataset.
        max_tokens (int): Range of indices to generate for the random tokens.
    """

    def __init__(
        self,
        data_path,
        max_tokens,
        epoch_size,
        mask_prob=0.15,
        unmask_prob=0.1,
        randomize_prob=0.1,
        block_size=512,
        ignore_idx=-1,
        padding_idx=0,
        cls_idx=1,
        sep_idx=2,
        mask_idx=3,
    ):
        super(MLM, self).__init__()

        self.epoch_size = epoch_size
        self.max_tokens = max_tokens
        self.mask_prob = mask_prob
        self.unmask_prob = unmask_prob
        self.randomize_prob = randomize_prob
        self.block_size = block_size
        self.ignore_idx = ignore_idx
        self.padding_idx = padding_idx
        self.cls_idx = cls_idx
        self.sep_idx = sep_idx
        self.mask_idx = mask_idx
        self.random_weights = [1] * self.max_tokens
        self.random_weights[self.padding_idx] = 0
        self.random_weights[self.cls_idx] = 0
        self.random_weights[self.sep_idx] = 0
        self.random_weights[self.mask_idx] = 0
        # Don't need to do ignore_idx, since it should always be outside the range

        dataset = np.load(data_path)
        self.context_idxs = torch.from_numpy(dataset["context_idxs"]).long()
        self.question_idxs = torch.from_numpy(dataset["ques_idxs"]).long()

    def mask(self, x, y):
        size = x.size(0)
        num_mask = int(self.mask_prob * size + random.random())
        masks = torch.tensor(random.sample(range(size), num_mask), dtype=torch.long)
        change_masks = torch.rand(num_mask)
        unmask = change_masks < self.unmask_prob
        random_mask = change_masks < (self.randomize_prob + self.unmask_prob)
        random_mask = random_mask & (~unmask)
        random_content = torch.tensor(
            random.choices(
                range(self.max_tokens),
                weights=self.random_weights,
                k=random_mask.sum().item(),
            ),
            dtype=torch.long,
        )

        masked = torch.tensor([False] * size, dtype=torch.bool)
        masked[masks] = True

        x[masks[~unmask]] = self.mask_idx
        x[masks[random_mask]] = random_content
        y[~masked] = self.ignore_idx

        return x, y

    def __len__(self):
        return self.epoch_size

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = 0
        num_workers = 1
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        epoch_size = self.epoch_size // num_workers
        next = torch.full((self.block_size,), self.padding_idx, dtype=torch.long)
        next[0] = self.cls_idx
        next_index = 1
        n_samples = 0
        while True:
            dataset_size = self.context_idxs.size(0)
            ids = list(range(worker_id, dataset_size, num_workers))

            random.shuffle(ids)
            for i in ids:
                for j in range(2):
                    if j == 0:
                        sample = self.context_idxs[i]
                    else:
                        sample = self.question_idxs[i]

                    sample_length = (sample != self.padding_idx).sum().item()
                    sample_index = 0
                    while sample_index < sample_length:
                        fill = min(
                            sample_length - sample_index, next.size(0) - next_index
                        )
                        next[next_index : next_index + fill] = sample[
                            sample_index : sample_index + fill
                        ]
                        next_index += fill
                        sample_index += fill

                        if next_index >= next.size(0):
                            x = next.clone().detach()
                            y = next.clone().detach()
                            yield self.mask(x, y)
                            next = torch.full(
                                (self.block_size,), self.padding_idx, dtype=torch.long
                            )
                            next[0] = self.cls_idx
                            next_index = 1
                            n_samples += 1
                            if n_samples >= epoch_size:
                                return
                        else:
                            next[next_index] = self.sep_idx
                            next_index += 1


def collate_fn(examples):
    # Group by tensor type
    x, y = zip(*examples)
    return torch.stack(x, dim=0), torch.stack(y, dim=0)


class SQuAD(data.Dataset):
    """Stanford Question Answering Dataset (SQuAD).

    Each item in the dataset is a tuple with the following entries (in order):
        - x: [CLS] context window [SEP] question
        - y: start and end indices, adjusted to the context window
        - c_padding_mask: mask out [SEP] question (True) or keep [CLS] context window (False)
        - ids: ids for each entry

    Args:
        data_path (str): Path to .npz file containing pre-processed dataset.
    """

    def __init__(
        self,
        data_path,
        block_size=512,
        ignore_idx=-1,
        padding_idx=0,
        cls_idx=1,
        sep_idx=2,
        mask_idx=3,
        use_v2=True,
    ):
        super(SQuAD, self).__init__()

        self.block_size = block_size
        self.ignore_idx = ignore_idx
        self.padding_idx = padding_idx
        self.cls_idx = cls_idx
        self.sep_idx = sep_idx
        self.mask_idx = mask_idx

        dataset = np.load(data_path)
        self.context_idxs = torch.from_numpy(dataset["context_idxs"]).long()
        self.question_idxs = torch.from_numpy(dataset["ques_idxs"]).long()
        self.y1s = torch.from_numpy(dataset["y1s"]).long()
        self.y2s = torch.from_numpy(dataset["y2s"]).long()
        self.ids = torch.from_numpy(dataset["ids"]).long()
        self.valid_idxs = [
            idx for idx in range(len(self.ids)) if use_v2 or self.y1s[idx].item() >= 0
        ]

    def __getitem__(self, idx):
        idx = self.valid_idxs[idx]
        example = (
            self.context_idxs[idx],
            self.question_idxs[idx],
            self.y1s[idx],
            self.y2s[idx],
            self.ids[idx],
        )

        return example

    def __len__(self):
        return len(self.valid_idxs)

    def get_sliding_window_collate(self, stride, randomize):
        """
        Gets a collate function which creates inputs at most the block size.
        If randomize is True, we get a single random sliding window (for training/dev).
        Otherwise, we keep all the sliding windows (for evaluation).
        """

        def sliding_window_collate(examples):
            windows = []
            for example in examples:
                c, q, y1, y2, id = example
                c_len = (c != self.padding_idx).sum()
                q_len = (q != self.padding_idx).sum()

                # We want to keep going so long as c_end = c_start + (block_size - q_len - 2)
                # has not been at least c_len for the first time, i.e. c_end < c_len + stride.
                # We also want to take at least one step.
                c_range = range(
                    0, max(1, c_len + q_len + 2 - self.block_size + stride), stride
                )
                if randomize:
                    c_start = random.sample(c_range, k=1)[0]
                    c_range = range(c_start, c_start + 1)

                for c_start in c_range:
                    c_end = min(self.block_size - q_len - 2 + c_start, c_len)
                    if y1 < c_start or y2 < c_start or y1 >= c_end or y2 >= c_end:
                        y1 = -1
                        y2 = -1
                    else:
                        y1 -= c_start
                        y2 -= c_start
                    windows.append((c[c_start:c_end], q[:q_len], y1, y2, c_start, id))

            # Collate windows
            max_len = max(len(window[0]) + len(window[1]) + 2 for window in windows)
            assert max_len <= self.block_size
            x = torch.full((len(windows), max_len), self.padding_idx, dtype=torch.long)
            y = torch.zeros(len(windows), 2, dtype=torch.long)
            c_padding_mask = torch.ones(len(windows), max_len, dtype=torch.bool)
            c_starts = torch.zeros(len(windows), dtype=torch.long)
            ids = torch.zeros(len(windows), dtype=torch.long)
            for i, window in enumerate(windows):
                c, q, y1, y2, c_start, id = window
                x[i, 0] = self.cls_idx
                x[i, 1 : 1 + len(c)] = c
                x[i, 1 + len(c)] = self.sep_idx
                x[i, 2 + len(c) : 2 + len(c) + len(q)] = q
                c_padding_mask[i][0 : 1 + len(c)] = False
                y[i, 0] = y1 + 1
                y[i, 1] = y2 + 1
                c_starts[i] = c_start
                ids[i] = id
            return x, y, c_padding_mask, c_starts, ids

        return sliding_window_collate


class QuestionsMLM(data.Dataset):
    """
    Args:
        data_path (str): Path to .npz file containing pre-processed dataset.
        max_tokens (int): Range of indices to generate for the random tokens.
    """

    def __init__(
        self,
        data_path,
        max_tokens,
        mask_prob=0.15,
        unmask_prob=0.1,
        randomize_prob=0.1,
        ignore_idx=-1,
        padding_idx=0,
        cls_idx=1,
        sep_idx=2,
        mask_idx=3,
        use_v2=True,
    ):
        super().__init__()

        self.max_tokens = max_tokens
        self.mask_prob = mask_prob
        self.unmask_prob = unmask_prob
        self.randomize_prob = randomize_prob
        self.ignore_idx = ignore_idx
        self.padding_idx = padding_idx
        self.cls_idx = cls_idx
        self.sep_idx = sep_idx
        self.mask_idx = mask_idx
        self.random_weights = [1] * self.max_tokens
        self.random_weights[self.padding_idx] = 0
        self.random_weights[self.cls_idx] = 0
        self.random_weights[self.sep_idx] = 0
        self.random_weights[self.mask_idx] = 0
        # Don't need to do ignore_idx, since it should always be outside the range

        dataset = np.load(data_path)
        self.context_idxs = torch.from_numpy(dataset["context_idxs"]).long()
        self.question_idxs = torch.from_numpy(dataset["ques_idxs"]).long()
        self.y1s = torch.from_numpy(dataset["y1s"]).long()
        self.y2s = torch.from_numpy(dataset["y2s"]).long()
        self.ids = torch.from_numpy(dataset["ids"]).long()
        self.valid_idxs = [
            idx for idx in range(len(self.ids)) if use_v2 or self.y1s[idx].item() >= 0
        ]
        self.max_id = torch.max(self.ids) + 1

    def mask(self, x, y):
        size = (x != self.padding_idx).sum().item()
        num_mask = int(self.mask_prob * size + random.random())
        masks = torch.tensor(random.sample(range(size), num_mask), dtype=torch.long)
        change_masks = torch.rand(num_mask)
        unmask = change_masks < self.unmask_prob
        random_mask = change_masks < (self.randomize_prob + self.unmask_prob)
        random_mask = random_mask & (~unmask)
        random_content = torch.tensor(
            random.choices(
                range(self.max_tokens),
                weights=self.random_weights,
                k=random_mask.sum().item(),
            ),
            dtype=torch.long,
        )

        masked = torch.tensor([False] * x.size(0), dtype=torch.bool)
        masked[masks] = True

        x[masks[~unmask]] = self.mask_idx
        x[masks[random_mask]] = random_content
        y[~masked] = self.ignore_idx

        return x, y

    def __getitem__(self, idx):
        idx = self.valid_idxs[idx]
        x = torch.full((self.question_idxs.size(-1) + 1,), self.padding_idx, dtype=torch.long)
        x[0] = self.cls_idx
        x[1:] = self.question_idxs[idx]
        x = x.clone().detach()
        y = x.clone().detach()
        x, y = self.mask(x, y)

        return x, y, self.context_idxs[idx], self.ids[idx]

    def __len__(self):
        return len(self.valid_idxs)

    @staticmethod
    def get_collate_fn():
        def mlm_collate_fn(examples):
            # Group by tensor type
            x, y, c, ids = zip(*examples)
            return torch.stack(x, dim=0), torch.stack(y, dim=0), torch.stack(c, dim=0), torch.stack(ids, dim=0)
        return mlm_collate_fn
