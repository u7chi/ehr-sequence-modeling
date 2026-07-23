"""CEHR-BERT-style encoder with pluggable MLM / classification heads.

Embeddings summed per token (BEHRT / CEHR-BERT style):
    concept + token-type + visit-segment + age-at-visit + absolute-position
The transformer stack is plain PyTorch (nn.TransformerEncoder, pre-norm, GELU)
so nothing depends on a specific `transformers` internal API, and the backbone
can later be swapped for a long-context encoder for the no-window research arm.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .buckets import N_COUNT_BUCKETS, N_RECENCY_BUCKETS
from .vocab import TYPE_PAD


class EHRSeqEmbeddings(nn.Module):
    def __init__(self, vocab_size, hidden, max_len, max_visits, n_age_bins, n_types=6, dropout=0.1):
        super().__init__()
        self.concept = nn.Embedding(vocab_size, hidden, padding_idx=TYPE_PAD)
        self.token_type = nn.Embedding(n_types, hidden)
        self.segment = nn.Embedding(max_visits, hidden)
        self.age = nn.Embedding(n_age_bins + 1, hidden)
        self.count = nn.Embedding(N_COUNT_BUCKETS, hidden)       # set mode: occurrence count
        self.recency = nn.Embedding(N_RECENCY_BUCKETS, hidden)   # set mode: days-before-index
        self.position = nn.Embedding(max_len, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(dropout)
        self.max_visits = max_visits
        self.max_len = max_len

    def forward(self, input_ids, type_ids, segment_ids, age_ids, count_ids, recency_ids):
        B, L = input_ids.shape
        pos = torch.arange(L, device=input_ids.device).clamp_max(self.max_len - 1)
        seg = segment_ids.clamp_max(self.max_visits - 1)
        x = (
            self.concept(input_ids)
            + self.token_type(type_ids)
            + self.segment(seg)
            + self.age(age_ids)
            + self.count(count_ids)
            + self.recency(recency_ids)
            + self.position(pos)[None, :, :]
        )
        return self.drop(self.norm(x))


class EHRSeqEncoder(nn.Module):
    def __init__(self, cfg_model, vocab_size, max_len, n_age_bins):
        super().__init__()
        h = cfg_model.hidden
        self.embeddings = EHRSeqEmbeddings(
            vocab_size=vocab_size,
            hidden=h,
            max_len=max_len,
            max_visits=cfg_model.max_visits,
            n_age_bins=n_age_bins,
            dropout=cfg_model.dropout,
        )
        layer = nn.TransformerEncoderLayer(
            d_model=h,
            nhead=cfg_model.heads,
            dim_feedforward=cfg_model.ff_mult * h,
            dropout=cfg_model.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg_model.layers)
        self.hidden = h
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.padding_idx is not None:
                with torch.no_grad():
                    m.weight[m.padding_idx].zero_()

    def forward(self, input_ids, type_ids, segment_ids, age_ids, count_ids, recency_ids, attention_mask):
        x = self.embeddings(input_ids, type_ids, segment_ids, age_ids, count_ids, recency_ids)
        pad_mask = attention_mask == 0  # True where padded
        return self.encoder(x, src_key_padding_mask=pad_mask)


class MLMHead(nn.Module):
    """Predict the concept id; decoder weight tied to the concept embedding."""

    def __init__(self, hidden, concept_embedding: nn.Embedding):
        super().__init__()
        self.transform = nn.Linear(hidden, hidden)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(hidden)
        self.concept_embedding = concept_embedding
        self.bias = nn.Parameter(torch.zeros(concept_embedding.num_embeddings))

    def forward(self, hidden):
        h = self.norm(self.act(self.transform(hidden)))
        return F.linear(h, self.concept_embedding.weight, self.bias)


class EHRSeqForPretraining(nn.Module):
    def __init__(self, cfg_model, vocab_size, max_len, n_age_bins):
        super().__init__()
        self.encoder = EHRSeqEncoder(cfg_model, vocab_size, max_len, n_age_bins)
        self.mlm = MLMHead(self.encoder.hidden, self.encoder.embeddings.concept)

    def forward(self, input_ids, type_ids, segment_ids, age_ids, count_ids, recency_ids, attention_mask, mlm_labels=None):
        hidden = self.encoder(input_ids, type_ids, segment_ids, age_ids, count_ids, recency_ids, attention_mask)
        logits = self.mlm(hidden)
        loss = None
        if mlm_labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), mlm_labels.view(-1), ignore_index=-100)
        return loss, logits


class EHRSeqForClassification(nn.Module):
    def __init__(self, cfg_model, vocab_size, max_len, n_age_bins):
        super().__init__()
        self.encoder = EHRSeqEncoder(cfg_model, vocab_size, max_len, n_age_bins)
        h = self.encoder.hidden
        # pool = concat([CLS], masked-mean) — their ablation showed mean-pool helps
        self.classifier = nn.Sequential(
            nn.Linear(2 * h, h), nn.GELU(),
            nn.Dropout(cfg_model.dropout), nn.Linear(h, 1),
        )

    def forward(self, input_ids, type_ids, segment_ids, age_ids, count_ids, recency_ids, attention_mask, labels=None, pos_weight=None):
        hidden = self.encoder(input_ids, type_ids, segment_ids, age_ids, count_ids, recency_ids, attention_mask)
        cls = hidden[:, 0]
        keep = attention_mask.unsqueeze(-1).float()
        mean = (hidden * keep).sum(1) / keep.sum(1).clamp_min(1.0)
        logit = self.classifier(torch.cat([cls, mean], dim=-1)).squeeze(-1)  # (B,)
        loss = None
        if labels is not None:
            loss = F.binary_cross_entropy_with_logits(logit, labels, pos_weight=pos_weight)
        return loss, logit

    def load_encoder(self, state_dict, strict=True):
        enc = {k[len("encoder."):]: v for k, v in state_dict.items() if k.startswith("encoder.")}
        missing, unexpected = self.encoder.load_state_dict(enc, strict=False)
        return missing, unexpected
