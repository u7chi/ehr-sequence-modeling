"""Small shared utilities."""
from __future__ import annotations

import logging
import os
import random
import sys

import numpy as np


def seed_everything(seed: int, deterministic: bool = True):
    """Seed all RNGs. With deterministic=True, also force deterministic CUDA/cuDNN
    kernels for fully reproducible runs (slightly slower)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # required for deterministic cuBLAS; must be set before CUDA init
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                pass
            try:
                # flash / mem-efficient SDPA have non-deterministic backward;
                # force the deterministic math attention kernel
                torch.backends.cuda.enable_flash_sdp(False)
                torch.backends.cuda.enable_mem_efficient_sdp(False)
                torch.backends.cuda.enable_math_sdp(True)
            except Exception:
                pass
    except ImportError:
        pass


def seed_worker(worker_id: int):
    """DataLoader worker_init_fn: seed numpy/random per worker from torch's
    per-worker seed (which PyTorch derives deterministically from the main seed)."""
    import torch

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int):
    """A seeded torch.Generator to make DataLoader shuffling / sampling reproducible."""
    import torch

    g = torch.Generator()
    g.manual_seed(seed)
    return g


def get_logger(name: str = "ehrseq", logfile: str | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if logfile:
        os.makedirs(os.path.dirname(logfile), exist_ok=True)
        fh = logging.FileHandler(logfile)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def describe(name: str, arr, logger) -> None:
    a = np.asarray(arr)
    logger.info(
        f"{name}: n={a.size} mean={a.mean():.1f} med={np.median(a):.0f} "
        f"p90={np.quantile(a, 0.9):.0f} p95={np.quantile(a, 0.95):.0f} "
        f"p99={np.quantile(a, 0.99):.0f} max={a.max():.0f} "
        f">{512}: {(a > 512).mean() * 100:.1f}%"
    )
