"""Datasets and collators for MLM pretraining and CAD fine-tuning."""
from __future__ import annotations

import pickle

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from .vocab import TYPE_DX, TYPE_LAB


class SeqDataset(Dataset):
    def __init__(self, pkl_path: str):
        with open(pkl_path, "rb") as f:
            self.samples = pickle.load(f)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    @property
    def labels(self):
        return np.array([s["label"] for s in self.samples], dtype=np.int64)


def _pad_stack(seqs, pad_value, max_len):
    out = torch.full((len(seqs), max_len), pad_value, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = torch.as_tensor(s, dtype=torch.long)
    return out


def _pack(batch, pad_id):
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids = _pad_stack([b["input_ids"] for b in batch], pad_id, max_len)
    type_ids = _pad_stack([b["type_ids"] for b in batch], 0, max_len)
    seg_ids = _pad_stack([b["segment_ids"] for b in batch], 0, max_len)
    age_ids = _pad_stack([b["age_ids"] for b in batch], 0, max_len)
    count_ids = _pad_stack([b["count_ids"] for b in batch], 0, max_len)
    recency_ids = _pad_stack([b["recency_ids"] for b in batch], 0, max_len)
    attn = (input_ids != pad_id).long()
    return {
        "input_ids": input_ids,
        "type_ids": type_ids,
        "segment_ids": seg_ids,
        "age_ids": age_ids,
        "count_ids": count_ids,
        "recency_ids": recency_ids,
        "attention_mask": attn,
    }


class ClassificationCollator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch):
        out = _pack(batch, self.pad_id)
        out["labels"] = torch.tensor([b["label"] for b in batch], dtype=torch.float32)
        return out


class MLMCollator:
    """BERT-style masking applied only to dx/lab (concept) tokens."""

    def __init__(self, vocab, pad_id: int, mask_prob: float = 0.15):
        self.vocab = vocab
        self.pad_id = pad_id
        self.mask_prob = mask_prob
        self.mask_id = vocab.mask_id
        self.maskable = np.array(vocab.maskable)  # vocab ids eligible to be masked/random
        self.type_of = np.array(vocab.type_of)

    def __call__(self, batch):
        out = _pack(batch, self.pad_id)
        input_ids, type_ids = out["input_ids"], out["type_ids"]
        labels = torch.full_like(input_ids, -100)

        is_concept = (type_ids == TYPE_DX) | (type_ids == TYPE_LAB)
        prob = torch.rand(input_ids.shape)
        selected = is_concept & (prob < self.mask_prob)

        labels[selected] = input_ids[selected]

        # 80% -> [MASK], 10% -> random concept, 10% -> keep
        r = torch.rand(input_ids.shape)
        mask_tok = selected & (r < 0.8)
        rand_tok = selected & (r >= 0.8) & (r < 0.9)

        input_ids[mask_tok] = self.mask_id
        n_rand = int(rand_tok.sum())
        if n_rand > 0:
            rand_ids = torch.as_tensor(
                np.random.choice(self.maskable, size=n_rand), dtype=torch.long
            )
            input_ids[rand_tok] = rand_ids

        out["mlm_labels"] = labels
        return out


def make_balanced_sampler(labels: np.ndarray, generator=None) -> WeightedRandomSampler:
    """Oversample the minority (positive) class to ~balanced batches (undersamples the
    majority; kept as an alternative). For parity with the graph project use
    `make_upsampled_dataset` instead."""
    class_count = np.bincount(labels)
    weight_per_class = 1.0 / np.clip(class_count, 1, None)
    weights = weight_per_class[labels]
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(labels),
        replacement=True,
        generator=generator,
    )


class UpsampledDataset(Dataset):
    """Index wrapper that physically replicates minority-class samples."""

    def __init__(self, base, indices):
        self.base = base
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.base[self.indices[i]]


def make_upsampled_dataset(dataset, labels, target_class=1, logger=None):
    """Physically replicate the minority class (mirrors ehr-graph-risk-prediction's
    `upsampling`): replication_factor = majority // minority, keeping all majority
    samples. Every negative is seen once per epoch (unlike WeightedRandomSampler)."""
    labels = np.asarray(labels)
    counts = np.bincount(labels)
    majority = 1 - target_class
    maj_count = int(counts[majority]) if majority < len(counts) else 0
    min_count = int(counts[target_class]) if target_class < len(counts) else 0
    if min_count == 0 or min_count >= maj_count:
        return dataset
    rep = maj_count // min_count
    minority_idx = [i for i, l in enumerate(labels) if l == target_class]
    indices = list(range(len(dataset)))
    for _ in range(rep - 1):
        indices.extend(minority_idx)
    if logger:
        logger.info(
            f"upsample: minority {min_count}x{rep}={min_count * rep} vs majority {maj_count}; "
            f"epoch {len(dataset)} -> {len(indices)}"
        )
    return UpsampledDataset(dataset, indices)
