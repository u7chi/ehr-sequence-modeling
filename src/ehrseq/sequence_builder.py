"""Turn a patient's dated events into parallel token channels.

Output channels (unpadded python lists, same length):
    input_ids   : vocab ids
    type_ids    : token-type ids (special/meta/dx/lab/time)
    segment_ids : visit index (0 for the [CLS]/meta prefix, 1..K for visits)
    age_ids     : age-bucket id at that point in time

Two modes share this code:
    windowed     : events in a time window -> one bagged visit  [VS] ... [VE]
    event_stream : one token per event, [ATT] gaps, no [VS]/[VE]  (research arm)
"""
from __future__ import annotations

from dataclasses import dataclass

from .discretize import age_bin
from .vocab import (
    TYPE_META,
    TYPE_SPECIAL,
    TYPE_TIME,
    Vocab,
    att_bucket,
)


@dataclass
class DatedEvents:
    day: int              # days since the patient's first record (>= 0)
    dx_tokens: list[str]
    lab_tokens: list[str]
    age: float            # age at this date


def _dedup(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _reduce_lab_max_bin(tokens):
    """When a lab item repeats within an aggregated visit, keep only its MAX bin
    (e.g. LAB_X_b1, LAB_X_b3 -> LAB_X_b3). Non-binned tokens are just deduped."""
    best = {}
    order = []
    for t in tokens:
        item, sep, b = t.rpartition("_b")
        if sep and b.isdigit():
            k = int(b)
            if item not in best:
                best[item] = k
                order.append(item)
            else:
                best[item] = max(best[item], k)
        else:
            if t not in best:
                best[t] = None
                order.append(t)
    return [key if best[key] is None else f"{key}_b{best[key]}" for key in order]


def _group_units(events: list[DatedEvents], mode: str, window_days: int):
    """Collapse dated events into ordered units (a window, or a single date).
    In windowed mode, repeated labs in a visit are reduced to their max bin."""
    reduce_lab = _reduce_lab_max_bin if mode == "windowed" else _dedup

    if mode == "event_stream" or window_days <= 0:
        # one unit per distinct date
        return [
            {"rep_day": e.day, "age": e.age,
             "dx": _dedup(e.dx_tokens), "lab": reduce_lab(e.lab_tokens)}
            for e in events
        ]
    # aggregate every `window_days` into one visit
    units = {}
    order = []
    for e in events:
        w = e.day // window_days
        if w not in units:
            units[w] = {"rep_day": e.day, "age": e.age, "dx": [], "lab": []}
            order.append(w)
        u = units[w]
        u["rep_day"] = e.day          # last date in window
        u["age"] = e.age
        u["dx"].extend(e.dx_tokens)
        u["lab"].extend(e.lab_tokens)
    out = []
    for w in order:
        u = units[w]
        u["dx"] = _dedup(u["dx"])
        u["lab"] = reduce_lab(u["lab"])
        out.append(u)
    return out


def build_sequence(
    events: list[DatedEvents],
    gender_value: int,
    vocab: Vocab,
    mode: str,
    window_days: int,
    max_len: int,
    age_width: int,
    n_age_bins: int,
    max_visits: int,
    truncate_side: str = "left",
) -> dict:
    units = _group_units(events, mode, window_days)

    def age_tok_id(age):
        return vocab.id(f"[AGE_{age_bin(age, age_width, n_age_bins)}]")

    # ---- prefix: [CLS] [AGE_index] [GENDER] [SEP] (segment 0) ----
    index_age = units[-1]["age"] if units else float("nan")
    prefix_ids = [
        vocab.cls_id,
        age_tok_id(index_age),
        vocab.id(f"[GENDER_{int(gender_value)}]"),
        vocab.sep_id,
    ]
    prefix_types = [TYPE_SPECIAL, TYPE_META, TYPE_META, TYPE_SPECIAL]
    index_age_id = age_bin(index_age, age_width, n_age_bins)

    # ---- one "block" per unit (includes its leading [ATT]) ----
    blocks = []
    prev_day = None
    for vi, u in enumerate(units, start=1):
        seg = min(vi, max_visits - 1)
        age_id = age_bin(u["age"], age_width, n_age_bins)

        ids, types = [], []
        # leading time-gap token
        if prev_day is not None:
            att = att_bucket(u["rep_day"] - prev_day)
            if att is not None:
                ids.append(vocab.id(att))
                types.append(TYPE_TIME)
        # visit body
        body_tokens = u["dx"] + u["lab"]
        body_ids = vocab.encode(body_tokens)
        body_types = [vocab.type_of[i] for i in body_ids]
        if mode == "windowed":
            ids = ids + [vocab.vs_id] + body_ids + [vocab.ve_id]
            types = types + [TYPE_SPECIAL] + body_types + [TYPE_SPECIAL]
        else:  # event_stream: no VS/VE wrapper
            ids = ids + body_ids
            types = types + body_types

        blocks.append({
            "ids": ids, "types": types,
            "seg": seg, "age_id": age_id,
        })
        prev_day = u["rep_day"]

    # ---- recency-preserving truncation (keep whole newest blocks) ----
    reserved = len(prefix_ids) + 1  # + trailing [SEP]
    budget = max_len - reserved
    kept: list[dict] = []
    used = 0
    iterator = reversed(blocks) if truncate_side == "left" else iter(blocks)
    for blk in iterator:
        blen = len(blk["ids"])
        if used + blen <= budget:
            kept.append(blk)
            used += blen
        elif not kept:
            # single block bigger than budget -> hard-truncate its interior, keep recent events
            keep_n = max(0, budget)
            blk = dict(blk)
            blk["ids"] = blk["ids"][-keep_n:]
            blk["types"] = blk["types"][-keep_n:]
            kept.append(blk)
            used += len(blk["ids"])
            break
        else:
            break
    if truncate_side == "left":
        kept = list(reversed(kept))

    # drop a dangling leading [ATT] that would reference a dropped older visit
    if kept and kept[0]["types"] and kept[0]["types"][0] == TYPE_TIME:
        kept[0] = dict(kept[0])
        kept[0]["ids"] = kept[0]["ids"][1:]
        kept[0]["types"] = kept[0]["types"][1:]

    # ---- assemble channels ----
    input_ids = list(prefix_ids)
    type_ids = list(prefix_types)
    segment_ids = [0] * len(prefix_ids)
    age_ids = [index_age_id] * len(prefix_ids)

    for blk in kept:
        input_ids.extend(blk["ids"])
        type_ids.extend(blk["types"])
        segment_ids.extend([blk["seg"]] * len(blk["ids"]))
        age_ids.extend([blk["age_id"]] * len(blk["ids"]))

    input_ids.append(vocab.sep_id)
    type_ids.append(TYPE_SPECIAL)
    segment_ids.append(segment_ids[-1] if segment_ids else 0)
    age_ids.append(index_age_id)

    n = len(input_ids)
    return {
        "input_ids": input_ids,
        "type_ids": type_ids,
        "segment_ids": segment_ids,
        "age_ids": age_ids,
        "count_ids": [0] * n,       # only used by set mode
        "recency_ids": [0] * n,
    }
