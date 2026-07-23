# Results — CAD risk prediction

## Setup
- **Cohort**: 53,567 patients, CAD positive 11.7%.
- **Split**: patient-level stratified, 0.64 / 0.16 / 0.20.
- **Metric**: test **AUPRC** (primary) + AUROC; model selection on validation AUPRC.
- **Recipe**: CEHR-BERT-style encoder, MLM pretrain (60 ep) → CAD fine-tune (early
  stop), 7× minority upsampling (matches the graph baseline).
- **Seed**: single seed (seed 1) unless noted; multi-seed sweep pending.
- Explore config (flash attention, `deterministic: false`). Empty cells = not yet run.

## Aggregation spectrum — our framework (current code, seed 1)

Same tokenizer + encoder; only the **aggregation granularity** changes.

| mode | aggregation unit | seq median | test AUPRC | test AUROC | test F1-macro |
|---|---|---|---|---|---|
| `event_stream` | one token per event (+ time tokens) | ~410 | 0.7218 | 0.9351 | 0.7872 |
| `windowed` (per-date) | one calendar date = visit | 525 | | | |
| `windowed` (90-day) | 90-day fixed window | 328 | | | |
| `qtime` | 9 quantile-time pseudo-visits | 201 | 0.7418 | 0.9404 | 0.7927 |
| `set` | whole history → unique concept | 64 | 0.7364 | 0.9359 | 0.7807 |
| **`our-method`** | learned adaptive (emergent operators) | | | | |

Notes:
- `windowed` per-date/90-day: pending re-run on the **current** architecture (an
  earlier run on the pre-refactor architecture — before the count/recency channels
  and `[CLS]+mean` pooling — gave per-date ≈ 0.716; not directly comparable, so left
  blank here).
- `our-method`: the learned adaptive-operator method (intervention-identifiable
  emergent operators) — not yet implemented.

## Reference baselines (source repo, for context)

`u7chi/multi-kg-ehr-graph-risk-prediction`, same cohort:

| method | representation | test AUPRC | note |
|---|---|---|---|
| KG-GCN | graph over concepts | ~0.70 | reference |
| Transformer `seq` | one token per event | 0.658 | least aggregated |
| Transformer `set` (+ MLM pretrain) | unique concept | 0.750 | most aggregated |
| `qtime9_q4` (visit) | 9 quantile pseudo-visits | 0.748 | **validation** AUPRC, seed 0 |

## Reading so far (preliminary — single seed)

Full spectrum (our framework, seed 1, test AUPRC):

```
event_stream 0.722   <   set 0.736   <   qtime 0.742
 (per event)            (bag)            (9 quantile visits)
```

- The raw per-event stream is the **weakest** (0.722) — consistent with truncation
  (36% of its sequences exceed 512 tokens, the tail hitting the 1024 cap).
- Moderate, order-preserving aggregation (`qtime`) is best (0.742), a hair above the
  pure bag (`set`, 0.736).
- So it is **neither** a flat plateau **nor** "finer resolution always wins" — there
  is a mild sweet spot at moderate aggregation.
- The whole spread is ~0.02 on a **single seed**; treat as suggestive only. A
  multi-seed sweep (and re-running `windowed` on the current architecture) is needed
  before any firm claim.
