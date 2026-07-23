"""Token vocabulary for the CEHR-BERT-style EHR tokenizer.

A single vocabulary covers every token type so the windowed baseline and the
event-stream research arm are perfectly comparable:

    special : [PAD] [UNK] [CLS] [SEP] [MASK] [VS] [VE]
    meta    : [AGE_k]  [GENDER_v]
    dx      : DX_<ccs>
    lab     : LAB_<item>_b<bin>
    time    : [ATT_*]  (inter-visit gap buckets, CEHR-BERT style)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

# token-type ids (used by the type embedding)
TYPE_PAD = 0
TYPE_SPECIAL = 1
TYPE_META = 2
TYPE_DX = 3
TYPE_LAB = 4
TYPE_TIME = 5

SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "[VS]", "[VE]"]

# inter-visit time-gap buckets (days), ordered coarse->fine is irrelevant; kept human-readable
ATT_TOKENS = [
    "[ATT_1d]", "[ATT_1w]", "[ATT_2w]", "[ATT_1m]",
    "[ATT_3m]", "[ATT_6m]", "[ATT_1y]", "[ATT_LT]",
]


def att_bucket(days: int) -> str | None:
    """Map an inter-visit gap in days to a CEHR-BERT-style [ATT] token (None if same day)."""
    if days <= 0:
        return None
    if days == 1:
        return "[ATT_1d]"
    if days <= 7:
        return "[ATT_1w]"
    if days <= 14:
        return "[ATT_2w]"
    if days <= 30:
        return "[ATT_1m]"
    if days <= 90:
        return "[ATT_3m]"
    if days <= 180:
        return "[ATT_6m]"
    if days <= 365:
        return "[ATT_1y]"
    return "[ATT_LT]"


@dataclass
class Vocab:
    itos: list[str]
    stoi: dict[str, int]
    type_of: list[int]          # token-type id per vocab id
    maskable: list[int]         # vocab ids eligible for MLM masking (dx + lab)

    # cached special ids
    pad_id: int = field(default=0)
    unk_id: int = field(default=0)
    cls_id: int = field(default=0)
    sep_id: int = field(default=0)
    mask_id: int = field(default=0)
    vs_id: int = field(default=0)
    ve_id: int = field(default=0)

    def __len__(self):
        return len(self.itos)

    def encode(self, tokens: list[str]) -> list[int]:
        s = self.stoi
        unk = self.unk_id
        return [s.get(t, unk) for t in tokens]

    def id(self, token: str) -> int:
        return self.stoi.get(token, self.unk_id)

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(
                {"itos": self.itos, "type_of": self.type_of, "maskable": self.maskable},
                f,
            )

    @classmethod
    def load(cls, path: str) -> "Vocab":
        with open(path) as f:
            d = json.load(f)
        return build_from_itos(d["itos"], d["type_of"], d["maskable"])


def build_from_itos(itos, type_of, maskable) -> Vocab:
    stoi = {t: i for i, t in enumerate(itos)}
    v = Vocab(itos=itos, stoi=stoi, type_of=type_of, maskable=maskable)
    v.pad_id = stoi["[PAD]"]
    v.unk_id = stoi["[UNK]"]
    v.cls_id = stoi["[CLS]"]
    v.sep_id = stoi["[SEP]"]
    v.mask_id = stoi["[MASK]"]
    v.vs_id = stoi["[VS]"]
    v.ve_id = stoi["[VE]"]
    return v


def build_vocab(
    dx_codes: list[str],
    lab_tokens: list[str],
    n_age_bins: int,
    gender_values: list[int],
) -> Vocab:
    """Build the full vocabulary. `lab_tokens` are the fully-formed LAB_<item>_b<bin> strings."""
    itos: list[str] = []
    type_of: list[int] = []

    def add(tok, typ):
        itos.append(tok)
        type_of.append(typ)

    for t in SPECIAL_TOKENS:
        add(t, TYPE_SPECIAL)
    for k in range(n_age_bins + 1):
        add(f"[AGE_{k}]", TYPE_META)
    for g in gender_values:
        add(f"[GENDER_{g}]", TYPE_META)
    for t in ATT_TOKENS:
        add(t, TYPE_TIME)
    for c in dx_codes:
        add(f"DX_{c}", TYPE_DX)
    for t in lab_tokens:
        add(t, TYPE_LAB)

    maskable = [i for i, ty in enumerate(type_of) if ty in (TYPE_DX, TYPE_LAB)]
    return build_from_itos(itos, type_of, maskable)
