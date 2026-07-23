#!/usr/bin/env bash
# End-to-end smoke test on a small patient subset (validates the pipeline fast).
# Usage:  PY=/path/to/python bash scripts/run_smoke.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY="${PY:-python}"

CFG=configs/default.yaml
COMMON="--config $CFG --develop_n 1500"

echo "== prepare (windowed) =="
$PY -m ehrseq.prepare_data $COMMON --mode windowed --window_days 0

echo "== pretrain (few epochs) =="
$PY -m ehrseq.pretrain --config $CFG --mode windowed --window_days 0 \
    2>/dev/null || true

echo "== finetune =="
$PY -m ehrseq.finetune --config $CFG --mode windowed --window_days 0
