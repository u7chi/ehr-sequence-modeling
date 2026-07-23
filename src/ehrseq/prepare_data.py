"""Build and cache tokenized patient sequences from the CCS/lab pivots.

Run once per (mode, window_days). Produces, under outputs/cache/<mode>_w<W>/:
    vocab.json                 shared vocabulary
    {train,valid,test}.pkl     list of samples (token channels + label)
    meta.json                  bin spec, split sizes, length stats

Usage:
    python -m ehrseq.prepare_data --config configs/default.yaml
    python -m ehrseq.prepare_data --config configs/default.yaml --mode event_stream
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .buckets import count_bucket, recency_bucket
from .config import load_config
from .discretize import age_bin, enumerate_lab_tokens, fit_lab_bins, value_to_bin
from .sequence_builder import DatedEvents, build_sequence
from .splits import stratified_patient_split
from .util import describe, get_logger, seed_everything
from .vocab import TYPE_DX, TYPE_LAB, TYPE_META, TYPE_SPECIAL

META_COLS = {"ID", "age", "gender", "date", "CAD", "indexDate"}


def _feature_cols(path):
    cols = pd.read_csv(path, nrows=0).columns.tolist()
    return cols, [c for c in cols if c not in META_COLS]


def _read_ccs(path, dx_cols):
    dtype = {c: "int8" for c in dx_cols}
    dtype.update({"ID": "string", "gender": "float32", "age": "float32", "CAD": "int8"})
    try:
        df = pd.read_csv(path, dtype=dtype, parse_dates=["date"])
    except (ValueError, TypeError):
        # fallback: code cols may contain NaN -> read as float then fill
        df = pd.read_csv(path, parse_dates=["date"])
        df[dx_cols] = df[dx_cols].fillna(0).astype("int8")
    return df.sort_values(["ID", "date"]).reset_index(drop=True)


def _read_lab(path, lab_cols):
    dtype = {c: "float32" for c in lab_cols}
    dtype.update({"ID": "string", "age": "float32", "gender": "float32", "CAD": "int8"})
    df = pd.read_csv(path, dtype=dtype, parse_dates=["date"])
    df = df.replace(-1, np.nan)  # -1 encodes "not measured" (same as graph loader)
    return df.sort_values(["ID", "date"]).reset_index(drop=True)


def _build_set_sample(dx_by_row, cdates, lab_by_row, ldates, cages, gender, index_date, vocab, cfg):
    """`set` mode: collapse the whole history into one token per unique concept, each
    carrying an occurrence-count bucket and a most-recent recency bucket (days before
    index). Mirrors the u7chi transformer/data.py `set` serialization."""
    if index_date is None:
        return None
    idx = np.datetime64(index_date)
    if np.isnat(idx):
        return None

    def delta(dt):
        return int((idx - np.datetime64(dt)) / np.timedelta64(1, "D"))

    dx_deltas = {}
    for i, toks in enumerate(dx_by_row):
        if toks:
            d = delta(cdates[i])
            for t in toks:
                dx_deltas.setdefault(t, []).append(d)

    lab_occ = {}
    for i, toks in enumerate(lab_by_row):
        if toks:
            d = delta(ldates[i])
            for t in toks:
                item = t.rsplit("_b", 1)[0]
                lab_occ.setdefault(item, []).append((d, t))

    if not dx_deltas and not lab_occ:
        return None

    age_val = float(np.nanmax(cages)) if len(cages) else float("nan")
    age_idx = age_bin(age_val, cfg.sequence.age_bin_width, cfg.sequence.n_age_bins)

    input_ids = [vocab.cls_id, vocab.id(f"[AGE_{age_idx}]"), vocab.id(f"[GENDER_{int(gender)}]")]
    type_ids = [TYPE_SPECIAL, TYPE_META, TYPE_META]
    segment_ids = [0, 0, 0]
    age_ids = [age_idx, age_idx, age_idx]
    count_ids = [0, 0, 0]
    recency_ids = [0, 0, 0]

    body = []  # (token, type, count_bucket, recency_bucket, recency_delta)
    for t, deltas in dx_deltas.items():
        body.append((t, TYPE_DX, count_bucket(len(deltas)), recency_bucket(min(deltas)), min(deltas)))
    for item, occ in lab_occ.items():
        occ.sort(key=lambda x: x[0])   # most-recent (smallest delta) first
        rd, rtok = occ[0]
        body.append((rtok, TYPE_LAB, count_bucket(len(occ)), recency_bucket(rd), rd))

    budget = cfg.sequence.max_len - len(input_ids)
    if len(body) > budget:
        body.sort(key=lambda x: x[4])  # keep most-recent
        body = body[:budget]

    for tok, typ, cb, rb, _ in body:
        input_ids.append(vocab.id(tok))
        type_ids.append(typ)
        segment_ids.append(1)
        age_ids.append(age_idx)
        count_ids.append(cb)
        recency_ids.append(rb)

    return {"input_ids": input_ids, "type_ids": type_ids, "segment_ids": segment_ids,
            "age_ids": age_ids, "count_ids": count_ids, "recency_ids": recency_ids}


def _fit_qtime_edges(ccs, lab, index_series, train_ids, n_buckets=9):
    """9 quantile time buckets fit on unique (patient, day-before-index) in train
    (mirrors baselines/qtime9_q4 fit_qtime_edges)."""
    train = set(train_ids)
    days = []
    for df in (ccs, lab):
        sub = df[df["ID"].isin(train)][["ID", "date"]].drop_duplicates()
        idx = sub["ID"].map(index_series).to_numpy()
        delta = (idx - sub["date"].to_numpy()) / np.timedelta64(1, "D")
        days.append(delta[np.isfinite(delta)])
    days = np.concatenate(days).astype(np.int64)
    qs = np.arange(1, n_buckets) / n_buckets
    return np.unique(np.quantile(days, qs, method="nearest").astype(np.int64))


def _build_qtime_sample(dx_by_row, cdates, lab_by_row, ldates, cages, gender, index_date, qtime_edges, vocab, cfg):
    """`qtime` mode: 9 quantile-time pseudo-visits (oldest->newest), each a bag of
    deduped dx + latest-value labs, carrying count + qtime-bucket channels."""
    if index_date is None:
        return None
    idx = np.datetime64(index_date)
    if np.isnat(idx):
        return None

    def info(dt):
        d = int((idx - np.datetime64(dt)) / np.timedelta64(1, "D"))
        return d, int(np.searchsorted(qtime_edges, d, side="right"))

    buckets = {}  # b -> {"dx": {tok: count}, "lab": {item: [min_delta, tok, count]}}
    for i, toks in enumerate(dx_by_row):
        if not toks:
            continue
        _, b = info(cdates[i])
        bk = buckets.setdefault(b, {"dx": {}, "lab": {}})
        for t in toks:
            bk["dx"][t] = bk["dx"].get(t, 0) + 1
    for i, toks in enumerate(lab_by_row):
        if not toks:
            continue
        d, b = info(ldates[i])
        bk = buckets.setdefault(b, {"dx": {}, "lab": {}})
        for t in toks:
            item = t.rsplit("_b", 1)[0]
            cur = bk["lab"].get(item)
            if cur is None:
                bk["lab"][item] = [d, t, 1]
            else:
                cur[2] += 1
                if d < cur[0]:
                    cur[0], cur[1] = d, t
    if not buckets:
        return None

    age_val = float(np.nanmax(cages)) if len(cages) else float("nan")
    age_idx = age_bin(age_val, cfg.sequence.age_bin_width, cfg.sequence.n_age_bins)
    input_ids = [vocab.cls_id, vocab.id(f"[AGE_{age_idx}]"), vocab.id(f"[GENDER_{int(gender)}]")]
    type_ids = [TYPE_SPECIAL, TYPE_META, TYPE_META]
    segment_ids, age_ids, count_ids, recency_ids = [0, 0, 0], [age_idx] * 3, [0, 0, 0], [0, 0, 0]

    blocks = []  # one block per pseudo-visit, oldest (high bucket) -> newest
    for seg, b in enumerate(sorted(buckets.keys(), reverse=True), start=1):
        blk = [(vocab.id(t), TYPE_DX, count_bucket(c), b, seg) for t, c in buckets[b]["dx"].items()]
        blk += [(vocab.id(t), TYPE_LAB, count_bucket(c), b, seg) for _, t, c in buckets[b]["lab"].values()]
        blocks.append(blk)

    budget = cfg.sequence.max_len - len(input_ids)
    total = sum(len(b) for b in blocks)
    while total > budget and len(blocks) > 1:  # drop oldest whole pseudo-visit first
        total -= len(blocks.pop(0))
    flat = [tok for blk in blocks for tok in blk][:budget]
    for tid, typ, cb, rb, seg in flat:
        input_ids.append(tid)
        type_ids.append(typ)
        segment_ids.append(min(seg, cfg.model.max_visits - 1))
        age_ids.append(age_idx)
        count_ids.append(cb)
        recency_ids.append(rb)

    return {"input_ids": input_ids, "type_ids": type_ids, "segment_ids": segment_ids,
            "age_ids": age_ids, "count_ids": count_ids, "recency_ids": recency_ids}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--mode", default=None, choices=["windowed", "event_stream", "set", "qtime"])
    ap.add_argument("--window_days", type=int, default=None)
    ap.add_argument("--develop_n", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    mode = args.mode or cfg.sequence.mode
    window_days = args.window_days if args.window_days is not None else cfg.sequence.window_days
    develop_n = args.develop_n if args.develop_n is not None else cfg.data.develop_n

    out_dir = os.path.join(cfg.data.cache_dir, f"{mode}_w{window_days}")
    os.makedirs(out_dir, exist_ok=True)
    logger = get_logger("prepare", logfile=os.path.join(out_dir, "prepare.log"))
    seed_everything(cfg.seed)
    logger.info(f"mode={mode} window_days={window_days} develop_n={develop_n} -> {out_dir}")

    # ---- headers / feature columns ----
    _, dx_cols = _feature_cols(cfg.data.ccs_path)
    _, lab_cols = _feature_cols(cfg.data.lab_path)
    logger.info(f"dx codes={len(dx_cols)}  lab items={len(lab_cols)}")

    # ---- load ----
    t = time.time()
    ccs = _read_ccs(cfg.data.ccs_path, dx_cols)
    logger.info(f"loaded CCS: {len(ccs):,} rows in {time.time()-t:.1f}s")
    t = time.time()
    lab = _read_lab(cfg.data.lab_path, lab_cols)
    logger.info(f"loaded LAB: {len(lab):,} rows in {time.time()-t:.1f}s")

    # ---- patient labels & split ----
    labels_by_id = ccs.groupby("ID")[cfg.data.label_col].first()
    all_ids = labels_by_id.index.to_numpy()
    all_y = labels_by_id.to_numpy()

    if develop_n and develop_n < len(all_ids):
        rng = np.random.RandomState(cfg.seed)
        idx = rng.choice(len(all_ids), size=develop_n, replace=False)
        all_ids, all_y = all_ids[idx], all_y[idx]
        keep = set(all_ids.tolist())
        ccs = ccs[ccs["ID"].isin(keep)]
        lab = lab[lab["ID"].isin(keep)]
        logger.info(f"DEVELOP mode: subsampled to {len(all_ids)} patients")

    split = stratified_patient_split(
        all_ids, all_y, cfg.split.train, cfg.split.valid, cfg.split.test, cfg.split.seed
    )
    id2split = {}
    for name, ids in split.items():
        for i in ids:
            id2split[i] = name
    logger.info(f"split: train={len(split['train'])} valid={len(split['valid'])} test={len(split['test'])}")

    # ---- fit lab bins on TRAIN rows only ----
    train_ids = set(split["train"])
    lab_train = lab[lab["ID"].isin(train_ids)]
    train_lab_vals = {c: lab_train[c].to_numpy(dtype="float32") for c in lab_cols}
    bin_spec = fit_lab_bins(train_lab_vals, cfg.sequence.n_lab_bins)
    lab_tokens = enumerate_lab_tokens(bin_spec, cfg.sequence.n_lab_bins)
    logger.info(f"fit lab bins on {len(lab_train):,} train rows -> {len(lab_tokens)} lab tokens")

    # ---- vocab ----
    from .vocab import build_vocab

    gender_values = sorted({int(g) for g in ccs["gender"].dropna().unique().tolist()})
    vocab = build_vocab(
        dx_codes=dx_cols,
        lab_tokens=lab_tokens,
        n_age_bins=cfg.sequence.n_age_bins,
        gender_values=gender_values,
    )
    vocab.save(os.path.join(out_dir, "vocab.json"))
    logger.info(f"vocab size={len(vocab)}  gender values={gender_values}")

    # ---- precompute per-column helpers ----
    dxstr = [f"DX_{c}" for c in dx_cols]
    lab_edges = [bin_spec[c]["edges"] for c in lab_cols]

    ccs_g = ccs.groupby("ID", sort=False)
    lab_g = lab.groupby("ID", sort=False)
    lab_groups = lab_g.groups

    index_map = {}
    qtime_edges = None
    if mode in ("set", "qtime"):
        idx_df = pd.read_csv(cfg.data.index_path)
        idx_df["indexDate"] = pd.to_datetime(idx_df["indexDate"], errors="coerce")
        index_map = {str(k): v for k, v in zip(idx_df["ID"], idx_df["indexDate"].to_numpy())}
        logger.info(f"loaded {len(index_map)} index dates for {mode} mode")
    if mode == "qtime":
        qtime_edges = _fit_qtime_edges(ccs, lab, pd.Series(index_map), split["train"])
        logger.info(f"qtime edges (days before index): {qtime_edges.tolist()}")

    samples = {"train": [], "valid": [], "test": []}
    lengths = {"train": [], "valid": [], "test": []}

    for pid, cblock in tqdm(ccs_g, total=ccs["ID"].nunique(), desc=f"build[{mode}]"):
        sp = id2split.get(pid)
        if sp is None:
            continue
        label = int(cblock[cfg.data.label_col].iloc[0])
        g = cblock["gender"].iloc[0]
        gender = 0 if pd.isna(g) else int(g)

        # dx tokens per ccs row (vectorized nonzero)
        cdates = cblock["date"].to_numpy()
        cages = cblock["age"].to_numpy()
        cmat = cblock[dx_cols].to_numpy()
        rr, cc = np.nonzero(cmat == 1)
        dx_by_row = [[] for _ in range(len(cblock))]
        for r, c in zip(rr.tolist(), cc.tolist()):
            dx_by_row[r].append(dxstr[c])

        # lab tokens per lab row
        lab_by_row, ldates, lages = [], np.array([], dtype="datetime64[ns]"), np.array([])
        if pid in lab_groups:
            lblock = lab.loc[lab_groups[pid]]
            ldates = lblock["date"].to_numpy()
            lages = lblock["age"].to_numpy()
            lmat = lblock[lab_cols].to_numpy()
            rr, cc = np.nonzero(~np.isnan(lmat))
            vals = lmat[rr, cc]
            lab_by_row = [[] for _ in range(len(lblock))]
            for r, c, v in zip(rr.tolist(), cc.tolist(), vals.tolist()):
                b = value_to_bin(v, lab_edges[c])
                lab_by_row[r].append(f"LAB_{lab_cols[c]}_b{b}")

        if mode == "set":
            seq = _build_set_sample(
                dx_by_row, cdates, lab_by_row, ldates, cages, gender,
                index_map.get(str(pid)), vocab, cfg,
            )
            if seq is None:
                continue
        elif mode == "qtime":
            seq = _build_qtime_sample(
                dx_by_row, cdates, lab_by_row, ldates, cages, gender,
                index_map.get(str(pid)), qtime_edges, vocab, cfg,
            )
            if seq is None:
                continue
        else:
            # per-date accumulator: date -> [dx, lab, age]
            by_date: dict = {}
            for i in range(len(cblock)):
                d = cdates[i]
                e = by_date.get(d)
                if e is None:
                    by_date[d] = [list(dx_by_row[i]), [], cages[i]]
                else:
                    e[0].extend(dx_by_row[i])
            for i in range(len(lab_by_row)):
                d = ldates[i]
                e = by_date.get(d)
                if e is None:
                    by_date[d] = [[], list(lab_by_row[i]), lages[i]]
                else:
                    e[1].extend(lab_by_row[i])
            if not by_date:
                continue
            dates_sorted = sorted(by_date.keys())
            min_day = dates_sorted[0]
            events = [
                DatedEvents(
                    day=int((d - min_day) / np.timedelta64(1, "D")),
                    dx_tokens=by_date[d][0],
                    lab_tokens=by_date[d][1],
                    age=float(by_date[d][2]),
                )
                for d in dates_sorted
            ]
            seq = build_sequence(
                events,
                gender_value=gender,
                vocab=vocab,
                mode=mode,
                window_days=window_days,
                max_len=cfg.sequence.max_len,
                age_width=cfg.sequence.age_bin_width,
                n_age_bins=cfg.sequence.n_age_bins,
                max_visits=cfg.model.max_visits,
                truncate_side=cfg.sequence.truncate_side,
            )

        seq["label"] = label
        seq["pid"] = str(pid)
        samples[sp].append(seq)
        lengths[sp].append(len(seq["input_ids"]))

    # ---- save ----
    for sp in ("train", "valid", "test"):
        with open(os.path.join(out_dir, f"{sp}.pkl"), "wb") as f:
            pickle.dump(samples[sp], f)
        n_pos = sum(s["label"] for s in samples[sp])
        logger.info(f"[{sp}] n={len(samples[sp])} pos={n_pos} ({n_pos/max(1,len(samples[sp])):.3f})")
        describe(f"[{sp}] seq_len", lengths[sp], logger)

    meta = {
        "mode": mode,
        "window_days": window_days,
        "vocab_size": len(vocab),
        "n_dx": len(dx_cols),
        "n_lab_items": len(lab_cols),
        "n_lab_tokens": len(lab_tokens),
        "gender_values": gender_values,
        "split_sizes": {k: len(v) for k, v in samples.items()},
        "max_len": cfg.sequence.max_len,
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"done -> {out_dir}")


if __name__ == "__main__":
    main()
