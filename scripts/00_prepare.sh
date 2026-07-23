#!/usr/bin/env bash
# Build tokenized-sequence caches. Windowed = baseline; event_stream = research arm.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
PY="${PY:-python}"
CFG="${CFG:-configs/default.yaml}"

$PY -m ehrseq.prepare_data --config "$CFG" --mode windowed     --window_days 0
# uncomment to also build the no-window research arm:
# $PY -m ehrseq.prepare_data --config "$CFG" --mode event_stream --window_days 0
