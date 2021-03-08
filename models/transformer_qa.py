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
        self.apply(lambda mod: T.init_params_bert(mod, 0.02))

    # (S, N), (S, N), (N, S) -> (S, N, 2)
    @amp.autocast()
    def forward(self, x, positions=None, padding_mask=None):
        x = x.transpose(0, 1)
        if positions is None:
            positions = T.get_positions(x)
        else:
            positions = positions.transpose(0, 1)
        x = self.embed(x, positions)
        x = self.encoder.forward(x, key_padding_mask=padding_mask)
        x = self.head(x)
        x = x.transpose(0, 1)
        return x

    # (N, S, 2), (N, S) -> (N, S, 2)
    def mask_scores(self, x, padding_mask):
        return self.head.mask_scores(x, padding_mask)

    # (N, S, 2) -> (N, S, 2)
    def get_log_prob(self, x):
        return self.head.get_log_prob(x)

    # (N, S, 2) -> (N, S, 2)
    def get_prob(self, x):
        return self.head.get_prob(x)

    # (N, S, 2), (N, 2) -> (1, )
    def get_loss(self, scores, y):
        return self.head.get_loss(scores, y)


class WordTransformerQA(nn.Module):
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
            T.LearnedTokenEmbedding(word_vectors.size(0), dim, 0),
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
        self.apply(lambda mod: T.init_params_bert(mod, 0.02))

    # (S, N), (S, N), (N, S) -> (S, N, 2)
    @amp.autocast()
    def forward(self, x, positions=None, padding_mask=None):
        x = x.transpose(0, 1)
        if positions is None:
            positions = T.get_positions(x)
        else:
            positions = positions.transpose(0, 1)
        x = self.embed(x, positions)
        x = self.encoder.forward(x, key_padding_mask=padding_mask)
        x = self.head(x)
        x = x.transpose(0, 1)
        return x

    # (N, S, 2), (N, S) -> (N, S, 2)
    def mask_scores(self, x, padding_mask):
        return self.head.mask_scores(x, padding_mask)

    # (N, S, 2) -> (N, S, 2)
    def get_log_prob(self, x):
        return self.head.get_log_prob(x)

    # (N, S, 2) -> (N, S, 2)
    def get_prob(self, x):
        return self.head.get_prob(x)

    # (N, S, 2), (N, 2) -> (1, )
    def get_loss(self, scores, y):
        return self.head.get_loss(scores, y)