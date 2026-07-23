"""MLM pretraining of the CEHR-BERT-style encoder.

    python -m ehrseq.pretrain --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import os

import torch
from torch.utils.data import ConcatDataset, DataLoader

from .config import load_config
from .dataset import MLMCollator, SeqDataset
from .model import EHRSeqForPretraining
from .optim import build_optimizer_scheduler
from .util import get_logger, make_generator, seed_everything, seed_worker
from .vocab import Vocab


def cache_dir(cfg, mode=None, window_days=None):
    mode = mode or cfg.sequence.mode
    w = cfg.sequence.window_days if window_days is None else window_days
    return os.path.join(cfg.data.cache_dir, f"{mode}_w{w}")


@torch.no_grad()
def eval_mlm(model, loader, device, use_amp):
    model.eval()
    tot, n = 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast("cuda", enabled=use_amp):
            loss, _ = model(**batch)
        tot += loss.item() * batch["input_ids"].size(0)
        n += batch["input_ids"].size(0)
    return tot / max(1, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--mode", default=None)
    ap.add_argument("--window_days", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    mode = args.mode or cfg.sequence.mode
    window_days = args.window_days if args.window_days is not None else cfg.sequence.window_days
    cdir = cache_dir(cfg, mode, window_days)
    ckpt_dir = "outputs/ckpts"
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = get_logger("pretrain", logfile=f"outputs/logs/{cfg.pretrain.ckpt_name}.log")
    seed_everything(cfg.seed, deterministic=getattr(cfg, "deterministic", True))
    g = make_generator(cfg.seed)

    device = cfg.device if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"
    vocab = Vocab.load(os.path.join(cdir, "vocab.json"))
    logger.info(f"cache={cdir} vocab={len(vocab)} device={device}")

    train_sets = [SeqDataset(os.path.join(cdir, "train.pkl"))]
    if cfg.pretrain.source == "train+valid":
        train_sets.append(SeqDataset(os.path.join(cdir, "valid.pkl")))
    train_ds = ConcatDataset(train_sets)
    valid_ds = SeqDataset(os.path.join(cdir, "valid.pkl"))

    collate = MLMCollator(vocab, pad_id=vocab.pad_id, mask_prob=cfg.pretrain.mask_prob)
    train_loader = DataLoader(train_ds, batch_size=cfg.pretrain.batch_size, shuffle=True,
                              collate_fn=collate, num_workers=4, pin_memory=True, drop_last=True,
                              worker_init_fn=seed_worker, generator=g)
    valid_loader = DataLoader(valid_ds, batch_size=cfg.pretrain.batch_size, shuffle=False,
                              collate_fn=collate, num_workers=2, pin_memory=True,
                              worker_init_fn=seed_worker, generator=g)

    model = EHRSeqForPretraining(cfg.model, len(vocab), cfg.sequence.max_len, cfg.sequence.n_age_bins).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"model params={n_params/1e6:.1f}M  train_seqs={len(train_ds)}")

    total_steps = len(train_loader) * cfg.pretrain.epochs
    opt, sched = build_optimizer_scheduler(model, cfg.pretrain.lr, cfg.pretrain.weight_decay,
                                           total_steps, cfg.pretrain.warmup_frac)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best = float("inf")
    ckpt_path = os.path.join(ckpt_dir, f"{cfg.pretrain.ckpt_name}_{mode}_w{window_days}.pt")
    for epoch in range(cfg.pretrain.epochs):
        model.train()
        running = 0.0
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=use_amp):
                loss, _ = model(**batch)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.item()
        val = eval_mlm(model, valid_loader, device, use_amp)
        logger.info(f"epoch {epoch+1}/{cfg.pretrain.epochs} train_mlm={running/len(train_loader):.4f} "
                    f"val_mlm={val:.4f} lr={sched.get_last_lr()[0]:.2e}")
        if val < best:
            best = val
            torch.save({"model": model.state_dict(), "cfg_model": vars(cfg.model),
                        "vocab_size": len(vocab), "epoch": epoch, "val_mlm": val}, ckpt_path)
            logger.info(f"  saved best -> {ckpt_path} (val_mlm={val:.4f})")
    logger.info(f"done. best val_mlm={best:.4f}")


if __name__ == "__main__":
    main()
