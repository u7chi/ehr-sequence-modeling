"""Optimizer + linear-warmup / cosine-decay schedule."""
from __future__ import annotations

import math

import torch


def build_optimizer_scheduler(model, lr, weight_decay, total_steps, warmup_frac):
    # no weight decay on norms / biases / embeddings
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or "norm" in n.lower() or n.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr, betas=(0.9, 0.98), eps=1e-6,
    )
    warmup = max(1, int(total_steps * warmup_frac))

    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        p = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    return opt, sched
