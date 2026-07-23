"""Fine-tune (or train from scratch) for CAD prediction.

    python -m ehrseq.finetune --config configs/default.yaml
    python -m ehrseq.finetune --config configs/default.yaml --from_scratch   # ablation
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import load_config
from .dataset import ClassificationCollator, SeqDataset, make_upsampled_dataset
from .metrics import compute_metrics
from .model import EHRSeqForClassification
from .optim import build_optimizer_scheduler
from .util import get_logger, make_generator, seed_everything, seed_worker
from .vocab import Vocab


def cache_dir(cfg, mode=None, window_days=None):
    mode = mode or cfg.sequence.mode
    w = cfg.sequence.window_days if window_days is None else window_days
    return os.path.join(cfg.data.cache_dir, f"{mode}_w{w}")


@torch.no_grad()
def evaluate(model, loader, device, use_amp):
    model.eval()
    probs, ys = [], []
    for batch in loader:
        y = batch.pop("labels")
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast("cuda", enabled=use_amp):
            _, logit = model(**batch)
        probs.append(torch.sigmoid(logit).float().cpu().numpy())
        ys.append(y.numpy())
    return compute_metrics(np.concatenate(ys), np.concatenate(probs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--mode", default=None)
    ap.add_argument("--window_days", type=int, default=None)
    ap.add_argument("--from_scratch", action="store_true", help="ignore pretrained encoder")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg.seed
    mode = args.mode or cfg.sequence.mode
    window_days = args.window_days if args.window_days is not None else cfg.sequence.window_days
    cdir = cache_dir(cfg, mode, window_days)
    os.makedirs("outputs/ckpts", exist_ok=True)
    os.makedirs("outputs/results", exist_ok=True)
    tag = f"{cfg.finetune.ckpt_name}_{mode}_w{window_days}_seed{seed}"
    logger = get_logger("finetune", logfile=f"outputs/logs/{tag}.log")
    seed_everything(seed, deterministic=getattr(cfg, "deterministic", True))
    g = make_generator(seed)

    device = cfg.device if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"
    vocab = Vocab.load(os.path.join(cdir, "vocab.json"))

    train_ds = SeqDataset(os.path.join(cdir, "train.pkl"))
    valid_ds = SeqDataset(os.path.join(cdir, "valid.pkl"))
    test_ds = SeqDataset(os.path.join(cdir, "test.pkl"))
    collate = ClassificationCollator(pad_id=vocab.pad_id)

    imbalance = cfg.finetune.imbalance
    pos_weight = None
    if imbalance == "upsample":
        train_used = make_upsampled_dataset(train_ds, train_ds.labels, target_class=1, logger=logger)
        train_loader = DataLoader(train_used, batch_size=cfg.finetune.batch_size, shuffle=True,
                                  collate_fn=collate, num_workers=4, pin_memory=True, drop_last=True,
                                  worker_init_fn=seed_worker, generator=g)
    else:
        train_loader = DataLoader(train_ds, batch_size=cfg.finetune.batch_size, shuffle=True,
                                  collate_fn=collate, num_workers=4, pin_memory=True, drop_last=True,
                                  worker_init_fn=seed_worker, generator=g)
        if imbalance == "weight":
            y = train_ds.labels
            pos_weight = torch.tensor([(y == 0).sum() / max(1, (y == 1).sum())], device=device)
    valid_loader = DataLoader(valid_ds, batch_size=cfg.finetune.batch_size, shuffle=False,
                              collate_fn=collate, num_workers=2, pin_memory=True,
                              worker_init_fn=seed_worker, generator=g)
    test_loader = DataLoader(test_ds, batch_size=cfg.finetune.batch_size, shuffle=False,
                             collate_fn=collate, num_workers=2, pin_memory=True,
                             worker_init_fn=seed_worker, generator=g)

    model = EHRSeqForClassification(cfg.model, len(vocab), cfg.sequence.max_len, cfg.sequence.n_age_bins).to(device)

    use_pretrained = cfg.finetune.from_pretrained and not args.from_scratch
    if use_pretrained:
        pt = os.path.join("outputs/ckpts", f"{cfg.pretrain.ckpt_name}_{mode}_w{window_days}.pt")
        if os.path.exists(pt):
            missing, unexpected = model.load_encoder(torch.load(pt, map_location=device)["model"])
            logger.info(f"loaded pretrained encoder from {pt} (missing={len(missing)} unexpected={len(unexpected)})")
        else:
            logger.info(f"WARNING: no pretrained ckpt at {pt}; training encoder from scratch")
            use_pretrained = False
    if cfg.finetune.freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad = False

    logger.info(f"seed={seed} cache={cdir} pretrained={use_pretrained} imbalance={imbalance} "
                f"train={len(train_ds)} valid={len(valid_ds)} test={len(test_ds)}")

    total_steps = len(train_loader) * cfg.finetune.epochs
    opt, sched = build_optimizer_scheduler(model, cfg.finetune.lr, cfg.finetune.weight_decay,
                                           total_steps, cfg.finetune.warmup_frac)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    sel = cfg.finetune.select_metric
    patience = getattr(cfg.finetune, "patience", 10**9)
    best_val, best_state, best_epoch, no_improve = -1.0, None, -1, 0
    for epoch in range(cfg.finetune.epochs):
        model.train()
        running = 0.0
        for batch in train_loader:
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=use_amp):
                loss, _ = model(**batch, labels=labels, pos_weight=pos_weight)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.item()
        val = evaluate(model, valid_loader, device, use_amp)
        logger.info(f"epoch {epoch+1}/{cfg.finetune.epochs} loss={running/len(train_loader):.4f} "
                    f"val_auprc={val['auprc']:.4f} val_auroc={val['auroc']:.4f} val_f1={val['f1_best']:.4f}")
        if val[sel] > best_val:
            best_val, best_epoch, no_improve = val[sel], epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info(f"early stop at epoch {epoch+1} (no val_{sel} improvement for {patience})")
                break

    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, device, use_amp)
    logger.info(f"BEST epoch={best_epoch+1} val_{sel}={best_val:.4f}")
    logger.info(f"TEST  auprc={test_metrics['auprc']:.4f} auroc={test_metrics['auroc']:.4f} "
                f"f1_best={test_metrics['f1_best']:.4f} f1@0.5={test_metrics['f1']:.4f}")

    result = {"seed": seed, "mode": cfg.sequence.mode, "window_days": cfg.sequence.window_days,
              "pretrained": use_pretrained, "best_epoch": best_epoch + 1,
              f"val_{sel}": best_val, "test": test_metrics}
    with open(os.path.join("outputs/results", f"{tag}.json"), "w") as f:
        json.dump(result, f, indent=2)
    torch.save(best_state, os.path.join("outputs/ckpts", f"{tag}.pt"))


if __name__ == "__main__":
    main()
