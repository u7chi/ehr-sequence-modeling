# Temporal aggregation without information loss — related work

**Goal of this project (the reason for this survey).** Find a representation for
long patient event sequences that **loses no useful information** yet **filters
out uninformative noise/redundancy** — as an alternative to fixed calendar
time-window binning. In our own data (see `DESIGN.md`), fixed per-date windowing
already truncates the *median* patient at `max_len=512`, while the raw event
stream (p95 ≈ 800–900 tokens, tail > 3000) truncates even more and underperforms.
So neither extreme is satisfactory; we want the top-right of the design space below.

---

## 1. The design space

Every method trades off along two axes we care about:

- **Information preservation** — does it keep exact timing, exact values, event
  identity, and long-range history? (fixed windows destroy timing + average values)
- **Noise / redundancy filtering** — does it suppress high-frequency, repeated,
  or irrelevant events so the model isn't drowned or length-blown? (raw streams don't)

```
                 filters noise / redundancy →
                 low                              high
 preserves   ┌───────────────────────┬───────────────────────────┐
 info  high  │  RAW EVENT STREAM      │  ★ TARGET REGION ★         │
             │  (lossless but long,  │  learned/adaptive          │
             │   noisy → truncation) │  aggregation; selective    │
             │  STraTS, SeFT, ODEs   │  compression; multi-scale  │
             ├───────────────────────┼───────────────────────────┤
       low   │  (naive subsampling)  │  FIXED TIME WINDOWS        │
             │                       │  (short + denoised but     │
             │                       │   destroys timing/values)  │
             └───────────────────────┴───────────────────────────┘
```

Fixed windows sit bottom-right (denoise + short, but lossy). Raw streams sit
top-left (lossless, but noisy/long). **The research target is top-right: keep the
information of the raw stream while getting the length/denoising benefit of
windows — by making the aggregation _learned and content-adaptive_ rather than a
fixed calendar grid.**

Summary of the mechanism families:

| Family | Mechanism | Preserves info | Filters noise | Shortens seq | Key refs |
|---|---|---|---|---|---|
| A. Time-aware tokenization | keep events, encode time as tokens/embeddings | ✅ high | ❌ | ❌ | BEHRT, Med-BERT, CEHR-BERT, Time2Vec |
| B. Binning-free continuous-time / set | (time,var,value) triples; ODE state | ✅ highest | ~ | ❌ | STraTS, SeFT, mTAND, GRU-D, Latent-ODE |
| C. Learned / adaptive aggregation | learn what to merge/drop from content | ✅ (learned) | ✅ | ✅ | Charformer-GBST, ToMe, Perceiver, Funnel, DuETT |
| D. Multi-scale / hierarchical | encode at several resolutions | ✅ (coarse+fine) | ✅ | ✅ | Hi-BEHRT, N-HiTS, Pyraformer, PatchTST, XTSFormer |
| E. Long-context backbones | just fit the whole stream, selectively | ✅ high | ✅ (selective) | n/a | Mamba/EHRMamba, Longformer, Context-Clues |

---

## 2. Family A — time-aware tokenization (keep events, encode time)

Keep every event as a token and inject time as **tokens or embeddings** instead of
binning. Lossless on timing, but does **not** shorten the sequence, so it doesn't
solve the length/noise problem by itself. This is our current baseline family.

- **BEHRT** (Li et al., 2020) — diagnosis tokens + visit-segment + position
  embeddings. [arXiv:1907.09538](https://arxiv.org/abs/1907.09538)
- **Med-BERT** (Rasmy et al., 2021) — MLM + auxiliary pretraining at scale.
  [npj Digital Medicine](https://www.nature.com/articles/s41746-021-00455-y)
- **CEHR-BERT** (Pang et al., 2021) — **artificial time tokens [ATT]** encode the
  gap between consecutive visits + age/time embeddings. *This is our baseline.*
  [arXiv:2111.08585](https://arxiv.org/abs/2111.08585)
- **CORE-BEHRT** (Odgaard et al., 2024) — carefully-tuned recipe; gains from adding
  medications and temporal age encoding. [arXiv:2404.15201](https://arxiv.org/abs/2404.15201)
- **Time2Vec** (Kazemi et al., 2019) — learnable sinusoidal time representation;
  a drop-in continuous encoding of timestamps. [arXiv:1907.05321](https://arxiv.org/abs/1907.05321)

**Takeaway:** necessary (time must be encoded, not binned) but insufficient — it
addresses info-preservation, not length/noise.

---

## 3. Family B — binning-free, continuous-time & set-based (the "no info loss" purists)

These deliberately **avoid binning, imputation, and aggregation**. They are the
strongest answer to "lose no information," and the clinical-time-series literature
shows they beat fixed-window baselines on irregular data.

- **STraTS** (Tipirneni & Reddy, 2022) — represents a record as a **set of
  (time, variable, value) triplets** with a **Continuous Value Embedding** so
  continuous values need no binning; a transformer attends over the triplet set.
  Explicitly "preserves the fine-grained information lost when the time axis is
  discretized." **Closest existing model to our stated goal.**
  [arXiv:2107.14293](https://arxiv.org/abs/2107.14293)
- **SeFT — Set Functions for Time Series** (Horn et al., ICML 2020) — treats the
  series as an unordered set of observations, aggregated by a permutation-invariant
  set function with sinusoidal time encodings. [arXiv:1909.12064](https://arxiv.org/abs/1909.12064)
- **mTAND — Multi-Time Attention Networks** (Shukla & Marlin, ICLR 2021) —
  continuous-time attention over a learned set of reference time points; interpolates
  irregular observations without a fixed grid. [arXiv:2101.10318](https://arxiv.org/abs/2101.10318)
- **GRU-D** (Che et al., 2018) — trainable **decay** between observations models
  missingness/irregular gaps directly. [arXiv:1606.01865](https://arxiv.org/abs/1606.01865)
- **Latent ODE / ODE-RNN** (Rubanova et al., 2019) — a continuous-time latent state
  evolves by an ODE between observations → native irregular sampling.
  [arXiv:1907.03907](https://arxiv.org/abs/1907.03907)
- **GRU-ODE-Bayes** (De Brouwer et al., 2019) — continuous-time GRU state evolution
  + Bayesian update at each observation. [arXiv:1905.12374](https://arxiv.org/abs/1905.12374)

**Takeaway:** maximal information preservation and native irregular time. The open
problem for *us* is scale: a set/triplet over a whole 5-year history is huge, and
these papers mostly target short ICU windows (e.g., 24–48 h). Combining B with a
compression or selective backbone (C/E) is the interesting direction.

---

## 4. Family C — learned / content-adaptive aggregation (the core idea)

Instead of a fixed calendar grid, **learn** which events to merge or drop from
their content. This is the most direct realization of "keep signal, drop noise."

- **Charformer / GBST** (Tay et al., ICLR 2022) — a **gradient-based subword
  tokenization** block that *learns* a latent segmentation: it enumerates candidate
  blocks and scores them position-wise, so the model learns its own "visits"
  end-to-end instead of using fixed windows. **Direct analogue of "learned
  time-windows."** [arXiv:2106.12672](https://arxiv.org/abs/2106.12672)
- **Token Merging (ToMe)** (Bolya et al., ICLR 2023) — merges the most **redundant**
  (similar) tokens each layer, keeping distinct/important ones. Content-aware
  redundancy removal ≈ "filter noise without dropping signal."
  [arXiv:2210.09461](https://arxiv.org/abs/2210.09461)
- **Perceiver / Perceiver IO** (Jaegle et al., 2021) — cross-attends a long input
  into a small **fixed set of latent vectors** (information bottleneck), decoupling
  compute from input length. Lets a very long event stream be compressed to a
  learned latent summary without a calendar grid.
  [Perceiver](https://arxiv.org/abs/2103.03206) · [Perceiver IO](https://arxiv.org/abs/2107.14795)
- **Funnel-Transformer / Hourglass** (Dai et al., 2020) — progressively **pool the
  sequence length** with learned (strided/attention) pooling, then optionally
  up-sample. [arXiv:2006.03236](https://arxiv.org/abs/2006.03236)
- **DuETT — Dual Event Time Transformer** (Labach et al., MLHC 2023) — attends over
  **both** the event-type and time axes and aggregates sparse EHR series into a
  fixed-length regular sequence in a **learned** way (an EHR-native version of this
  idea). [arXiv:2304.13017](https://arxiv.org/abs/2304.13017)
- **Adaptive segmentation of clinical events** (Luo et al., 2019, "Learning
  Hierarchical Representations of EHR") — segments an event sequence into groups by
  **irregular record time**; events within a group are treated as exchangeable,
  which *removes short-range ordering noise* and shortens the sequence.
  [arXiv:1903.08652](https://arxiv.org/abs/1903.08652)
- **Temporal Latent Bottleneck** (Didolkar et al., 2022) — splits processing into a
  fast per-chunk stream and a slow bottleneck that carries only compressed
  cross-chunk state. [arXiv:2205.14794](https://arxiv.org/abs/2205.14794)

**Takeaway:** this family is where "learned windows" live. GBST-style learned
segmentation and Perceiver-style latent bottlenecks are the two most transferable
mechanisms for replacing fixed windows in our event stream.

---

## 5. Family D — multi-scale / hierarchical (fine + coarse simultaneously)

Rather than pick one resolution, encode several. Keeps fine detail *and* a denoised
coarse view.

- **Hi-BEHRT** (Li et al., 2022) — **hierarchical** EHR transformer: a local encoder
  over sliding windows + a global aggregator, expanding the receptive field to very
  long histories; reports +1–8% AUPRC, larger for patients with **long** history —
  exactly our truncation regime. [arXiv:2106.11360](https://arxiv.org/abs/2106.11360)
- **PatchTST** (Nie et al., 2023) — splits a series into **patches** (subseries) as
  tokens; patching "suppresses high-frequency noise and models local dependencies"
  while cutting length. (Note: patches are still *fixed*-size — a learned-patch
  version is an open direction.) [arXiv:2211.14730](https://arxiv.org/abs/2211.14730)
- **N-HiTS** (Challu et al., 2023) — hierarchical interpolation + **multi-rate
  sampling** to model multiple resolutions cheaply. [arXiv:2201.12886](https://arxiv.org/abs/2201.12886)
- **Pyraformer** (Liu et al., ICLR 2022) — pyramidal attention with intra/inter-scale
  edges for multi-resolution, low-complexity long series.
  [OpenReview](https://openreview.net/forum?id=0EXmFzUn5I)
- **Pathformer** (Chen et al., 2024) — **adaptive** multi-scale: patch divisions of
  varying sizes with data-dependent routing. [arXiv:2402.05956](https://arxiv.org/abs/2402.05956)
- **XTSFormer** (Wu et al., 2024) — cross-temporal-scale transformer for
  **irregular-time clinical events**: hierarchical pooling + cross-scale attention so
  short-interval events interact at fine scale and distant ones at coarse scale.
  [arXiv:2402.02258](https://arxiv.org/abs/2402.02258)
- **Multi-resolution Time-Series Transformer** (Zhang et al., AISTATS 2024) —
  simultaneous multi-resolution patching. [PMLR v238](https://proceedings.mlr.press/v238/zhang24l/zhang24l.pdf)

**Takeaway:** hierarchical is the pragmatic middle ground and has direct EHR
evidence (Hi-BEHRT, XTSFormer). "Adaptive-scale" variants (Pathformer) start to
overlap with Family C.

---

## 6. Family E — long-context backbones (don't compress; just fit it, selectively)

If the backbone scales to the full stream, you may not need to aggregate at all —
*and* a **selective** backbone filters noise internally.

- **Mamba** (Gu & Dao, 2023) — selective state-space model; its core property is
  *"filtering out irrelevant information so the context can be compressed into an
  efficient state."* Linear time, million-token sequences.
  [arXiv:2312.00752](https://arxiv.org/abs/2312.00752)
- **EHRMamba** (Fallahpour et al., 2024) — Mamba foundation model for EHR; targets
  the quadratic-cost / short-context limits of EHR transformers, ~3× longer
  sequences. [ResearchGate](https://www.researchgate.net/publication/380847209)
- **ReTAMamba** (2026) — **Reliability-Aware Temporal Aggregation with Mamba** for
  *irregular* clinical time series — Mamba + explicit temporal aggregation, very
  on-topic. [arXiv:2605.16380](https://arxiv.org/abs/2605.16380)
- **Longformer / BigBird** — sparse attention for long documents; drop-in long-context
  transformers.
- **Context Clues** (Wornow et al., ICLR 2025) — systematic study of context length
  for clinical prediction on EHR; evidence on when longer context helps.
  [arXiv:2412.16178](https://arxiv.org/abs/2412.16178)

**Takeaway:** Mamba's *selectivity* is itself a learned noise filter, and EHRMamba/
ReTAMamba show it works on EHR. This is the lowest-friction swap in our repo (the
encoder is already pluggable).

---

## 7. Synthesis — what actually hits "no info loss + filter noise"

No single family reaches the top-right alone:

- **B** maximizes info preservation but doesn't scale to long histories.
- **C/D** shorten + denoise, but fixed-patch versions still lose info; the *learned*
  variants (GBST, ToMe, Perceiver, Pathformer, DuETT, adaptive segmentation) are the
  ones that keep info **and** filter.
- **E** avoids aggregation entirely and filters via selectivity, but a raw million-token
  stream can still bury signal.

The promising recipe is a **composition**:

> **binning-free input (B) → learned/adaptive compression (C) or selective
> long-context backbone (E), with time encoded continuously (A/Time2Vec).**

Framed information-theoretically: we want a **learned bottleneck** (Perceiver /
Funnel / GBST / Mamba-state) trained so the retained representation is *sufficient*
for the outcome while discarding redundancy — an information-bottleneck objective
rather than a hand-set calendar window.

---

## 8. Concrete candidates for this repo (`ehr-sequence-modeling`)

All three drop into our `event_stream` mode (one token per event + `[ATT]`), reuse
the tokenizer/MLM pretraining, and are benchmarked against the windowed CEHR-BERT
baseline on CAD (AUPRC), including the **truncation-penalty** test.

1. **Selective backbone (fastest to try).** Keep the event stream; swap
   `nn.TransformerEncoder` → **Mamba** (or Longformer). Tests whether selectivity
   alone recovers the windowed baseline without any hand aggregation. *Effort: low —
   the encoder is already pluggable.* (cf. EHRMamba, ReTAMamba)

2. **Learned aggregation / "learned visits" (most aligned with the thesis).** Insert
   a **GBST-style** learned segmentation or a **Funnel/Perceiver** pooling layer over
   the event stream, so the model learns which events to merge into a "visit" instead
   of a fixed `window_days`. Ablate learned-window vs fixed-window at equal length.
   *Effort: medium.* (cf. Charformer-GBST, Perceiver, DuETT)

3. **Triplet / binning-free input (max info preservation).** Replace the pivot-derived
   tokens with **(time, variable, value) triplets + continuous value embedding** à la
   STraTS, so lab values aren't bucketed into 4 bins at all — combined with (1) or (2)
   for length. *Effort: medium–high; changes the input pipeline.* (cf. STraTS, SeFT)

**Recommended order:** (1) to get a no-window number fast → (2) as the core
contribution (learned vs fixed windows) → (3) as the "no information loss anywhere"
ablation (also removes lab binning).

### Evaluation protocol
- Baselines: windowed CEHR-BERT (`window_days=0`), fixed coarse window
  (`window_days=30/90`), raw `event_stream` (truncated) — quantify the truncation gap.
- Metric: AUPRC (primary) + AUROC, multi-seed, same split — comparable to the GNN models.
- Diagnostics: performance vs **history length** (does the method win most on long
  histories, like Hi-BEHRT?); effective sequence length after compression; ablate
  learned vs fixed aggregation at matched length.

---

## References (grouped)

**EHR transformers:** BEHRT [1907.09538] · Med-BERT [npj 2021] · CEHR-BERT [2111.08585] ·
CORE-BEHRT [2404.15201] · Hi-BEHRT [2106.11360] · DuETT [2304.13017] · Context Clues [2412.16178]
**Binning-free / irregular TS:** STraTS [2107.14293] · SeFT [1909.12064] · mTAND [2101.10318] ·
GRU-D [1606.01865] · Latent-ODE [1907.03907] · GRU-ODE-Bayes [1905.12374] · Time2Vec [1907.05321]
**Learned aggregation / compression:** Charformer-GBST [2106.12672] · ToMe [2210.09461] ·
Perceiver [2103.03206] / Perceiver IO [2107.14795] · Funnel [2006.03236] · Temporal Latent Bottleneck [2205.14794]
**Multi-scale:** PatchTST [2211.14730] · N-HiTS [2201.12886] · Pyraformer [ICLR'22] ·
Pathformer [2402.05956] · XTSFormer [2402.02258] · Multi-res TS Transformer [AISTATS'24]
**Long-context / selective:** Mamba [2312.00752] · EHRMamba [2024] · ReTAMamba [2605.16380]
**Adaptive clinical segmentation:** Luo et al. [1903.08652]

*Compiled 2026-07-23 for the `ehr-sequence-modeling` project. Links are to arXiv/venue
pages; a few IDs are best-effort — verify before formal citation.*
