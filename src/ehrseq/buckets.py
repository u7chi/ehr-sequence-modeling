"""Recency (days-before-index) and occurrence-count buckets for `set` mode.

Ported from the u7chi/multi-kg-ehr-graph-risk-prediction transformer/data.py so the
`set` serialization matches the model that reaches ~0.75 AUPRC.
"""
from __future__ import annotations

import numpy as np

# recency = days before the index date. Cohort window ~1-6 years pre-index
# (observed range ~366..2192 days).
RECENCY_EDGES = np.array([400, 500, 650, 800, 1000, 1300, 1600, 1900], dtype=np.float64)
N_RECENCY_BUCKETS = len(RECENCY_EDGES) + 1  # 9

# occurrence count of a concept across the patient's history
COUNT_EDGES = np.array([1, 2, 3, 5, 10, 20, 50], dtype=np.float64)
N_COUNT_BUCKETS = len(COUNT_EDGES) + 1  # 8


def recency_bucket(delta_days) -> int:
    return int(np.searchsorted(RECENCY_EDGES, float(delta_days), side="right"))


def count_bucket(count) -> int:
    return int(np.searchsorted(COUNT_EDGES, float(count), side="right"))
