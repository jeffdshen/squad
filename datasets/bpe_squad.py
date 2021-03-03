"""Stanford Question Answering Dataset (SQuAD).

Includes MLM and QA tasks.
Author:
    Jeffrey Shen
"""
import torch
import torch.utils.data as data
import numpy as np
import random


class MLM(data.Dataset):
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
        mask_prob=0.15,
        unmask_prob=0.1,
        randomize_prob=0.1,
        block_size=512,
        ignore_index=-1,
        padding_idx=0,
        cls_idx=1,
        sep_idx=2,
        mask_idx=3,
    ):
        super(MLM, self).__init__()

        self.max_tokens = max_tokens
        self.mask_prob = mask_prob
        self.unmask_prob = unmask_prob
        self.randomize_prob = randomize_prob
        self.block_size = block_size
        self.ignore_index = ignore_index
        self.padding_idx = padding_idx
        self.cls_idx = cls_idx
        self.sep_idx = sep_idx
        self.mask_idx = mask_idx

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
        random_content = torch.randint(
            self.max_tokens, (random_mask.sum().item(),), dtype=torch.long
        )

        masked = torch.tensor([False] * size, dtype=torch.bool)
        masked[masks] = True

        x[masks[~unmask]] = self.mask_idx
        x[masks[random_mask]] = random_content
        y[~masked] = self.ignore_index

        return x, y

    def __iter__(self):
        next = torch.full((self.block_size,), self.padding_idx, dtype=torch.long)
        next[0] = self.cls_idx
        next_index = 1
        while True:
            dataset_size = self.context_idxs.size(0)
            ids = list(zip(range(dataset_size), [0] * dataset_size)) + list(
                zip(range(dataset_size), [1] * dataset_size)
            )

            random.random.shuffle(ids)
            for i, j in ids:
                if j == 0:
                    sample = self.context_idxs[i]
                else:
                    sample = self.question_idxs[i]

                sample_length = (sample != self.padding_idx).sum().item()
                sample_index = 0
                while sample_index < sample_length:
                    fill = min(sample_length - sample_index, next.size(0) - next_index)
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
                    else:
                        next[next_index] = self.sep_index
                        next_index += 1


def collate_fn(examples):
    # Group by tensor type
    x, y = zip(*examples)
    return torch.stack(x, dim=0), torch.stack(y, dim=0)
