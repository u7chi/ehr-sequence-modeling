# Design: sequence modeling for CAD risk prediction

## 1. Task (mirrors `ehr-graph-risk-prediction` for comparability)

Binary **CAD** prediction from a patient's 5-year diagnosis + lab history up to
an index date. Same cohort, same patient-level **stratified** split
(0.64 / 0.16 / 0.20), same class-imbalance handling, and **AUPRC** as the
model-selection / primary metric — so numbers are directly comparable to the
GNN models.

### Cohort facts (measured)
| | |
|---|---|
| patients | 53,567 |
| CAD positive | 6,272 (11.7%) |
| diagnosis vocab | 276 CCS codes |
| lab vocab | 80 items |
| visits (distinct dates) / patient | median 38, p95 120, p99 226, max 866 |
| dx / visit | mean 3.7 |
| event-stream tokens / patient (dx only) | mean 180, p95 440, p99 829, max 3178 |

## 2. Why "no time windows" is the research question

- **Windowed** (aggregate events in a window → one bagged visit): sequence length
  = number of visits → p95 ≈ 120 positions, comfortably inside a 512 context.
- **Event-stream** (one token per event): p95 ≈ 440 (dx only) and ≈ 800–900 with
  labs, tail > 3000 → blows past 512 → **truncation drops the oldest history** →
  empirically worse than windowing.

So the model that currently *wins* is the windowed one. The goal is a method that
matches/beats it **without** hand-chosen calendar windows and without the length
blow-up. This repo builds the windowed CEHR-BERT baseline first, with the
event-stream arm wired through the same tokenizer/model so the comparison is clean.

## 3. Token schema (CEHR-BERT style)

```
[CLS] [AGE_k] [GENDER_v] [SEP]                     # demographics prefix (segment 0)
[VS] DX_101 DX_53 LAB_72-101_b3 LAB_72-140_b1 [VE] # visit 1  (bag of codes)
[ATT_3m]                                           # inter-visit gap token
[VS] DX_98 LAB_72-135_b4 [VE]                      # visit 2
... [SEP]
```

- **Diagnoses** → `DX_<ccs>`.
- **Labs** → `LAB_<item>_b{1..4}`: continuous labs discretized into **4 quantile
  bins fit on train** (reuses the graph project's convention); binary labs → 2
  categories. (The old PyHealth baseline kept only lab *presence* — we keep value.)
- **Time** → `[ATT_*]` buckets of the inter-visit gap in days
  (`1d,1w,2w,1m,3m,6m,1y,LT`). This is the bridge to the event-stream arm.
- **Demographics** → `[AGE_k]` (5-y buckets) + `[GENDER_v]` up front.

### Embeddings (summed per token, BEHRT/CEHR-BERT)
`concept + token-type + visit-segment + age-at-visit + absolute-position`,
LayerNorm + dropout → transformer encoder.

- token-type ∈ {special, meta, dx, lab, time}
- visit-segment = which visit the token belongs to
- age = age bucket at that visit

### Two build modes, one tokenizer
- `windowed` (`window_days=0` → per-date visit; `>0` → N-day aggregation): baseline.
- `event_stream`: one token per event + `[ATT]` gaps, no `[VS]/[VE]`: research arm.

Truncation keeps the **most recent** visits (index date is the anchor).

## 4. Model

Plain-PyTorch `nn.TransformerEncoder` (pre-norm, GELU) — no dependence on
`transformers` internals, so the backbone is swappable for a long-context encoder
(Longformer/Mamba) in the research arm.

- **Pretraining**: BERT MLM, masking only dx/lab (concept) tokens (15%, 80/10/10);
  decoder tied to the concept embedding.
  - CEHR-BERT actually uses **two** objectives: MLM (primary) + **visit-type
    prediction** (auxiliary, needs OMOP `visit_concept_id`). Our pivots have no
    visit-type field, so we use MLM-only. If a secondary signal is wanted, the
    data-available analogues are: predict the next-visit code set, or predict the
    `[ATT]` time-gap / age bucket — added as a second head on the same encoder.
- **Fine-tuning**: `[CLS]` pooling → MLP → 1 logit, `BCEWithLogits`.
  Imbalance via balanced oversampling (default) or `pos_weight`.

Default size: hidden 256, 6 layers, 8 heads (~small BERT; vocab ≈ 600 so this is
plenty). Config in `configs/default.yaml`.

## 5. SOTA lineage referenced
BEHRT (codes + visit-segment + position) → Med-BERT (MLM + aux pretraining, scale)
→ **CEHR-BERT** (artificial time tokens — what we implement) → CORE-BEHRT (2024
tuned recipe). No-window frontier: time-token / continuous-time models (CEHR-BERT,
ETHOS, DT-Transformer) and long-context backbones (EHRMamba, Longformer,
Context-Clues ICLR'25).

## 6. Research roadmap (after the baseline)
1. **Baseline table**: windowed CEHR-BERT vs event-stream CEHR-BERT (same 512) —
   quantify the truncation penalty.
2. **No-window candidates** to beat the windowed baseline:
   - continuous-time position (sinusoidal age/Δt) instead of calendar bins;
   - long-context backbone (Mamba / Longformer) on the raw event stream;
   - **learned/content-based aggregation** (hierarchical event→segment→patient)
     so segmentation is learned, not a fixed calendar window.
3. Report AUPRC/AUROC across seeds vs the GNN models.
