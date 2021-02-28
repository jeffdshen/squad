"""Various full transformer models for QA.

Author:
    Jeffrey Shen
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp

import models.transformer as T


class GloveTransformerQA(nn.Module):
    def __init__(
        self,
        dim,
        n_heads,
        ff_dim,
        activation,
        dropout,
        attn_dropout,
        act_dropout,
        n_layers,
        max_positions,
        word_vectors,
    ):
        super().__init__()
        self.embed = T.TransformerEncoderEmbedding(
            T.PretrainedTokenEmbedding(word_vectors, dim),
            T.LearnedPositionalEmbedding(max_positions, dim),
            dim=dim,
            dropout=dropout,
        )
        self.encoder = T.TransformerEncoder(
            T.TransformerEncoderLayer(
                dim=dim,
                n_heads=n_heads,
                ff_dim=ff_dim,
                activation=activation,
                dropout=dropout,
                attn_dropout=attn_dropout,
                act_dropout=act_dropout,
            ),
            n_layers=n_layers,
        )
        self.head = T.LinearQAHead(dim=dim, output_logits=2)

    # (S, N), (S, N), (N, S) -> (S, N, 2)
    @amp.autocast()
    def forward(self, x, positions=None, padding_mask=None):
        if positions is None:
            positions = T.get_positions(x)
        x = self.embed(x, positions)
        x = self.encoder.forward(x, key_padding_mask=padding_mask)
        x = self.head(x)
        return x

    # (S, N, 2), (N, S) -> (S, N, 2)
    def mask_scores(self, x, padding_mask):
        return x.masked_fill(padding_mask.transpose(0, 1).unsqueeze(-1), float("-inf"))

    # (S, N, 2) -> (S, N, 2)
    def get_log_prob(self, x):
        return F.log_softmax(x, dim=0)

    # (S, N, 2) -> (S, N, 2)
    def get_prob(self, x):
        return F.softmax(x, dim=0)

    # (S, N, 2), (N, 2) -> (1, )
    def get_loss(self, scores, y):
        return self.head.get_loss(scores, y)
