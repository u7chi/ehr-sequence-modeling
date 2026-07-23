"""Fit/apply lab discretization and age bucketing.

Lab bins are quantile-based and fit on TRAIN rows only (same convention as the
graph project's KBinsDiscretizer), then applied to every split.
"""
from __future__ import annotations

import numpy as np


def is_binary_column(values: np.ndarray) -> bool:
    """True if the non-nan values are a subset of {0, 1}."""
    v = values[~np.isnan(values)]
    if v.size == 0:
        return False
    u = np.unique(v)
    return u.size <= 2 and np.isin(u, [0.0, 1.0]).all()


def fit_lab_bins(train_values: dict[str, np.ndarray], n_bins: int) -> dict:
    """
    Args:
        train_values: {lab_col -> 1d array of TRAIN values (may contain NaN)}
    Returns:
        {lab_col -> {"binary": bool, "edges": np.ndarray}}  interior quantile edges.
    """
    spec = {}
    qs = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]  # interior edges only
    for col, vals in train_values.items():
        v = vals[~np.isnan(vals)]
        if is_binary_column(vals):
            spec[col] = {"binary": True, "edges": np.array([0.5])}
            continue
        if v.size == 0:
            spec[col] = {"binary": False, "edges": np.array([])}
            continue
        edges = np.unique(np.quantile(v, qs))
        spec[col] = {"binary": False, "edges": edges}
    return spec


def value_to_bin(value: float, edges: np.ndarray) -> int:
    """Return a 1-indexed bin id for a single value given interior edges."""
    return int(np.searchsorted(edges, value, side="right")) + 1


def lab_token(col: str, value: float, edges: np.ndarray) -> str:
    return f"LAB_{col}_b{value_to_bin(value, edges)}"


def enumerate_lab_tokens(spec: dict, n_bins: int) -> list[str]:
    """All possible LAB_<col>_b<k> tokens, so the vocab is fixed up front."""
    tokens = []
    for col, s in spec.items():
        n = 2 if s["binary"] else n_bins
        tokens.extend(f"LAB_{col}_b{k}" for k in range(1, n + 1))
    return tokens


def age_bin(age: float, width: int, n_bins: int) -> int:
    """Bucket an age into [0, n_bins]. NaN/invalid -> 0."""
    if age is None or not np.isfinite(age):
        return 0
    k = int(age // width)
    return max(0, min(k, n_bins))
