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
| `event_stream` | one token per event (+ time tokens) | ~410 | | | |
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
- Aggregation dominates: heavily-aggregated (`set`/`qtime`) clearly beats the
  least-aggregated `seq` (0.658) in the source repo.
- In our framework, `qtime` (0.742) slightly edges `set` (0.736) — coarse temporal
  order helps a little beyond a pure bag.
- Whether `event_stream` (full temporal resolution) climbs back above `set`/`qtime`
  will show whether finer temporal structure keeps paying off on CAD — pending.
- Differences of ~0.005–0.01 are within single-seed noise; a multi-seed sweep is
  needed before drawing firm conclusions.
