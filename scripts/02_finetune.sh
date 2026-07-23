#!/usr/bin/env bash
# Fine-tune for CAD. Loops seeds to mirror the graph project's multi-run protocol.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
PY="${PY:-python}"
CFG="${CFG:-configs/default.yaml}"
SEEDS="${SEEDS:-1 2 3 4 5}"

for s in $SEEDS; do
  $PY -m ehrseq.finetune --config "$CFG" --mode windowed --window_days 0 --seed "$s"
done

# ablation (no pretraining):
# for s in $SEEDS; do
#   $PY -m ehrseq.finetune --config "$CFG" --from_scratch --seed "$s"
# done
