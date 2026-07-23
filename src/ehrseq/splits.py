"""Stratified patient-level train/valid/test split (mirrors the graph project)."""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split


def stratified_patient_split(
    ids: np.ndarray, labels: np.ndarray, train: float, valid: float, test: float, seed: int
) -> dict[str, list]:
    assert abs(train + valid + test - 1.0) < 1e-6, "split ratios must sum to 1"

    trainval_ids, test_ids, trainval_y, _ = train_test_split(
        ids, labels, test_size=test, random_state=seed, stratify=labels
    )
    train_ids, valid_ids, _, _ = train_test_split(
        trainval_ids,
        trainval_y,
        test_size=valid / (train + valid),
        random_state=seed,
        stratify=trainval_y,
    )
    return {
        "train": list(train_ids),
        "valid": list(valid_ids),
        "test": list(test_ids),
    }
