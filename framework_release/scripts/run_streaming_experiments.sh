#!/usr/bin/env bash
# Phase A: complete streaming matrix. Phase B: globally deduplicated native PSNR.
# One process/GPU owner; per-run and per-image checkpoints survive interruption.
set -euo pipefail
TASK4_REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$TASK4_REPO"
TASK4_PYTHON=${STREAMING_EXPERIMENT_PYTHON:-/home/fil/miniconda3/envs/Hoang/bin/python}
TASK4_OUTPUT=${STREAMING_EXPERIMENT_OUTPUT:-experiments/streaming_evaluation/runs/runtime_discretized_virtual_v1}
mkdir -p "$TASK4_OUTPUT/logs"
TASK4_OUTPUT=$(cd -- "$TASK4_OUTPUT" && pwd)
exec > >(tee -a "$TASK4_OUTPUT/logs/execution.log") 2>&1
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="$TASK4_REPO/.cache/streaming-matplotlib"
printf 'Streaming experiments started: %s\n' "$(date --iso-8601=seconds)"
set +e
"$TASK4_PYTHON" -m experiments.streaming_evaluation.two_phase --root "$TASK4_OUTPUT" "$@"
TASK4_RUN_STATUS=$?
if [[ -f "$TASK4_OUTPUT/manifest.json" ]]; then
    "$TASK4_PYTHON" -m experiments.streaming_evaluation.analyze_runtime_results --root "$TASK4_OUTPUT" --update-report
    TASK4_AUDIT_STATUS=$?
    "$TASK4_PYTHON" -m experiments.streaming_evaluation.two_phase_report --root "$TASK4_OUTPUT"
    if [[ "$?" -ne 0 ]]; then TASK4_AUDIT_STATUS=1; fi
else
    TASK4_AUDIT_STATUS=0
fi
if [[ "$TASK4_RUN_STATUS" -ne 0 ]]; then
    exit "$TASK4_RUN_STATUS"
fi
exit "$TASK4_AUDIT_STATUS"
