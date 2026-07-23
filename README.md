# ehr-sequence-modeling

A CEHR-BERT-style **sequence-model framework** for EHR risk prediction (CAD), built
to study **how input-aggregation granularity affects performance** — from a raw
per-event stream all the way to a bag-of-unique-concepts. One tokenizer + one
encoder, four interchangeable aggregation modes, MLM pretraining, supervised
fine-tuning, and fully reproducible runs.

Same cohort / patient-level stratified split / AUPRC metric as the
`ehr-graph-risk-prediction` GCN baseline, so numbers are directly comparable.

## Aggregation modes (the research axis)

The whole point: hold the model fixed and vary **how much the patient history is
collapsed** before it hits the encoder.

| `--mode` | aggregation unit | seq median | test AUPRC (seed 1) |
|---|---|---|---|
| `event_stream` | one token per event (+ `[ATT]` time-gap tokens) | ~410 | *(running)* |
| `windowed` (`--window_days 0`) | one calendar **date** = a visit | 525 | — |
| `windowed` (`--window_days 90`) | 90-day fixed window | 328 | — |
| `qtime` | 9 quantile-time pseudo-visits (data-adaptive) | 201 | **0.742** |
| `set` | whole history → one token per **unique concept** | 64 | **0.736** |

Reference points on the same cohort: KG-GCN ≈ 0.70; a source visit/set Transformer
with MLM pretrain ≈ 0.75. Each token can carry **count** (occurrence frequency) and
**recency / qtime** buckets, so aggregation doesn't throw away frequency/timing.

## Model

`src/ehrseq/model.py` — embeddings summed per token
(`concept + token-type + visit-segment + age + count + recency + position`), a
pre-norm `nn.TransformerEncoder` (pure PyTorch, no `transformers` dependency, so the
backbone is swappable), an MLM head (weight-tied) for pretraining, and a
`[CLS] + masked-mean` pooled classification head.

## Setup

- Python 3.10, PyTorch 2.x + CUDA. `pip install -r requirements.txt`
  (or reuse the `ehr-graph` conda env).
- **Data**: three pivot CSVs — see [`data/README.md`](data/README.md). Point the
  `data.*_path` fields in `configs/*.yaml` at them (or symlink into `data/`).

## Run

```bash
export PYTHONPATH=src
PY=python                      # or /path/to/env/bin/python

# 1) build + cache tokenized sequences for a mode (once per mode)
$PY -m ehrseq.prepare_data --config configs/default.yaml --mode set
# 2) MLM pretrain the encoder
$PY -m ehrseq.pretrain     --config configs/default.yaml --mode set
# 3) fine-tune for CAD (a seed)
$PY -m ehrseq.finetune     --config configs/default.yaml --mode set --seed 1
```

`--mode ∈ {event_stream, windowed, set, qtime}`; `windowed` also takes
`--window_days`. Checkpoints/results are namespaced by `mode`+`window_days`
(`outputs/ckpts/…`, `outputs/results/cad_*_<mode>_w<W>_seed<S>.json`). Convenience
wrappers live in `scripts/`.

## Reproducibility

`deterministic: true` (config, default in real runs) forces bit-exact kernels
(math attention — needs `batch_size 32` at `max_len 1024`; cuDNN deterministic;
`CUBLAS_WORKSPACE_CONFIG`). `false` uses flash attention (fast, batch 64, ~very
close but not bit-identical). Python/NumPy/Torch seeds + DataLoader `worker_init_fn`
+ generators are all fixed.

## Layout

```
src/ehrseq/
  config.py vocab.py discretize.py buckets.py     # tokenizer + bucketing
  sequence_builder.py prepare_data.py             # history -> token channels (all modes)
  dataset.py model.py optim.py                     # data / encoder / schedule
  pretrain.py finetune.py metrics.py splits.py util.py
configs/    default.yaml (real runs) · smoke.yaml (fast end-to-end check)
scripts/    00_prepare · 01_pretrain · 02_finetune · run_smoke
docs/       temporal-aggregation-survey.md         # related work on learned aggregation
DESIGN.md                                          # design rationale + evidence
```

## Notes

- The pivots are **not** committed (private data). Data files, `outputs/`, and
  caches are git-ignored.
- `configs/smoke.yaml` runs the whole pipeline on a small patient subset in minutes
  — use it to verify a clean checkout works before a full run.
# ehr-sequence-modeling
