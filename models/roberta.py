"""RoBERTa Model with LM and classification head

Author:
    Jeffrey Shen
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp

import models.transformer as T


class RoBERTa(nn.Module):
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
        max_tokens,
        padding_idx,
        ignore_idx,
    ):
        super().__init__()
        embed_tokens = T.LearnedTokenEmbedding(max_tokens, dim, padding_idx)
        self.embed = T.TransformerEncoderEmbedding(
            embed_tokens,
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
        self.head = T.LMHead(
            dim=dim,
            output_tokens=max_tokens,
            activation=activation,
            weight=embed_tokens.embed.weight,
        )
        self.ignore_idx = ignore_idx
        self.apply(lambda mod: T.init_params_bert(mod, 0.02))


    # (S, N), (S, N), (N, S) -> (S, N, O)
    @amp.autocast()
    def forward(self, x, positions=None, padding_mask=None):
        if positions is None:
            positions = T.get_positions(x)
        x = self.embed(x, positions)
        x = self.encoder.forward(x, key_padding_mask=padding_mask)
        x = self.head(x)
        return x

    # (S, N, O), (N, S) -> (S, N, O)
    @staticmethod
    def mask_scores(x, padding_mask):
        return x.masked_fill(padding_mask.transpose(0, 1).unsqueeze(-1), float("-inf"))

    # (S, N, O) -> (S, N)
    @staticmethod
    def get_top(x):
        return torch.argmax(x, dim=-1)

    # (S, N, O) -> (S, N, O*)
    @staticmethod
    def get_log_prob(x):
        return F.log_softmax(x, dim=-1)

    # (S, N, O) -> (S, N, O*)
    @staticmethod
    def get_prob(x):
        return F.softmax(x, dim=-1)

    # (S, N, O), (N, O) -> (1, )
    def get_loss(self, scores, y):
        return self.head.get_loss(scores, y, self.ignore_idx)