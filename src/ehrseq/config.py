"""Configuration: a nested dataclass loaded from YAML with CLI overrides."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from types import SimpleNamespace

import yaml


def _to_ns(d):
    """Recursively turn dicts into attribute-accessible namespaces."""
    if isinstance(d, dict):
        # coerce keys to str: YAML 1.1 turns bare on/off/yes/no into booleans
        return SimpleNamespace(**{str(k): _to_ns(v) for k, v in d.items()})
    return d


def _to_dict(ns):
    if isinstance(ns, SimpleNamespace):
        return {k: _to_dict(v) for k, v in vars(ns).items()}
    return ns


def load_config(path: str, overrides: dict | None = None) -> SimpleNamespace:
    """Load YAML config and apply dotted-key overrides (e.g. {'sequence.mode': 'event_stream'})."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    if overrides:
        for key, val in overrides.items():
            node = raw
            parts = key.split(".")
            for p in parts[:-1]:
                node = node.setdefault(p, {})
            node[parts[-1]] = val

    cfg = _to_ns(raw)
    cfg._raw = copy.deepcopy(raw)  # keep a serializable copy for logging
    return cfg


def config_to_dict(cfg: SimpleNamespace) -> dict:
    d = _to_dict(cfg)
    d.pop("_raw", None)
    return d
