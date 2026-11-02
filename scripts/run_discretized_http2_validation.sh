#!/usr/bin/env bash
# Bounded launcher for the existing Phase A runner: 2 modes x 3 N x 5 methods.
# Each policy gets a fresh process so VmHWM is meaningful for that validation.
set -euo pipefail
VALIDATION_REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$VALIDATION_REPO"
VALIDATION_PYTHON=${STREAMING_EXPERIMENT_PYTHON:-/home/fil/miniconda3/envs/Hoang/bin/python}
VALIDATION_ROOT=${STREAMING_VALIDATION_OUTPUT:-experiments/streaming_evaluation/runs/discretized_http2_comparison_v2}
VALIDATION_CACHE=${STREAMING_VISIBILITY_SOURCE:-experiments/streaming_evaluation/runs/runtime_six_second_h2_validation_v1}
VALIDATION_DEPS=${STREAMING_HTTP2_DEPS:-/tmp/streaming-http2-deps}
export PYTHONPATH="$VALIDATION_DEPS:$VALIDATION_REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$VALIDATION_ROOT"
VALIDATION_FAILURES=0
# A separate real protocol/shaping proof is mandatory before Lumos playback.
"$VALIDATION_PYTHON" -m experiments.streaming_evaluation.real_http2 --validate \
  --output "$VALIDATION_ROOT/http2_transport_validation.json"
for VALIDATION_MODE in virtual real_http2; do
  for VALIDATION_N in 5 7 15; do
    for VALIDATION_METHOD in NAVA VISIBILITY_PSNR VISIBILITY_MSE SOLO_SSE VISIBILITY_WEIGHTED_SSE; do
      VALIDATION_CASE="$VALIDATION_ROOT/$VALIDATION_MODE/N$VALIDATION_N/$VALIDATION_METHOD"
      mkdir -p "$VALIDATION_CASE/logs"
      printf 'Phase A: %s N=%s %s\n' "$VALIDATION_MODE" "$VALIDATION_N" "$VALIDATION_METHOD"
      if "$VALIDATION_PYTHON" -m experiments.streaming_evaluation.two_phase \
          --root "$VALIDATION_CASE" --phase A --transport "$VALIDATION_MODE" \
          --object-counts "$VALIDATION_N" --playback-durations 10 --scenarios camera_rotation \
          --methods "$VALIDATION_METHOD" --delta-bps 250000 --max-dp-cells 1000000 \
          --max-wall-seconds 600 --visibility-cache-from "$VALIDATION_CACHE" \
          > "$VALIDATION_CASE/logs/execution.log" 2>&1; then
        printf '0\n' > "$VALIDATION_CASE/exit_code.txt"
      else
        VALIDATION_EXIT=$?
        printf '%s\n' "$VALIDATION_EXIT" > "$VALIDATION_CASE/exit_code.txt"
        VALIDATION_FAILURES=$((VALIDATION_FAILURES + 1))
      fi
      "$VALIDATION_PYTHON" -m experiments.streaming_evaluation.analyze_runtime_results \
        --root "$VALIDATION_ROOT" --compare-phase-a-modes
    done
  done
done
printf 'Completed bounded validation; failed processes: %s\n' "$VALIDATION_FAILURES"
test "$VALIDATION_FAILURES" -eq 0
