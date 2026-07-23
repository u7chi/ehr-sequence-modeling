#!/usr/bin/env bash
# MLM-pretrain the encoder on the windowed cache.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
PY="${PY:-python}"
CFG="${CFG:-configs/default.yaml}"

$PY -m ehrseq.pretrain --config "$CFG" --mode windowed --window_days 0
