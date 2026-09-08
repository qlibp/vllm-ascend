#!/bin/bash
# Run the P/D dual-stream microbenchmark (native chunked-prefill vs P/D dual-stream).
#
# This is an OFFLINE benchmark: it drives the worker's model runner directly
# through fake SchedulerOutput objects, bypassing the scheduler entirely.  It
# does NOT need a running server.
#
# Usage (run each scenario in its own process for clean device state):
#   MODE=baseline MODEL=/path/to/model ./run_pd_dual_stream_microbench.sh
#   MODE=pd       MODEL=/path/to/model ./run_pd_dual_stream_microbench.sh
#
#   # both back-to-back in one process (needs ~2x device memory)
#   MODE=both MODEL=/path/to/model ./run_pd_dual_stream_microbench.sh
#
#   # with profiler
#   MODE=pd MODEL=/path/to/model PROFILE=1 PROFILE_DIR=/tmp/pd_traces \
#     ./run_pd_dual_stream_microbench.sh
#
#   # with a custom python interpreter / source tree
#   PYTHON_BIN=/path/to/python \
#   VLLM_SRC=/path/to/vllm VLLM_ASCEND_SRC=/path/to/vllm-ascend \
#   MODE=baseline MODEL=/path/to/model ./run_pd_dual_stream_microbench.sh
#
# All parameters are read from environment variables.  Defaults match the
# target scenario: prefill=200, beam-width=128, output-tokens=2.

set -euo pipefail

# --- Paths ----------------------------------------------------------------- #
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BENCH="${SCRIPT_DIR}/pd_dual_stream_microbench.py"

# --- Python interpreter & import paths ------------------------------------ #
# The benchmark imports both ``vllm`` and ``vllm_ascend``.  Use PYTHON_BIN to
# select the interpreter that has them installed; use VLLM_SRC / VLLM_ASCEND_SRC
# to prepend source trees to PYTHONPATH (for a from-source checkout).
PYTHON_BIN="${PYTHON_BIN:-python3}"
VLLM_SRC="${VLLM_SRC:-}"
VLLM_ASCEND_SRC="${VLLM_ASCEND_SRC:-}"

if [[ -n "$VLLM_SRC" ]]; then
  export PYTHONPATH="${VLLM_SRC}${PYTHONPATH:+:${PYTHONPATH}}"
fi
if [[ -n "$VLLM_ASCEND_SRC" ]]; then
  export PYTHONPATH="${VLLM_ASCEND_SRC}${PYTHONPATH:+:${PYTHONPATH}}"
fi

# --- Benchmark parameters (env-var driven) --------------------------------- #
MODEL="${MODEL:-}"
# baseline | pd | both.  Default to a single scenario so each run has clean
# device/memory state; run the two modes in separate processes and compare.
MODE="${MODE:-baseline}"
PREFILL_LEN="${PREFILL_LEN:-200}"
BEAM_WIDTH="${BEAM_WIDTH:-128}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-2}"
CHUNK_SIZE="${CHUNK_SIZE:-128}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
WARMUP="${WARMUP:-5}"
ITERS="${ITERS:-20}"
PROFILE="${PROFILE:-0}"
PROFILE_DIR="${PROFILE_DIR:-/tmp/pd_dual_stream_traces}"

# Note: do NOT enable ENFORCE_EAGER for the PD scenario.  With eager mode the
# runner skips capture_model(), so PDDualStreamModelRunner falls back to eager
# execution and the dual-stream path is never measured.
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"

if [[ -z "$MODEL" ]]; then
  echo "MODEL is required." >&2
  echo "Usage: MODEL=/path/to/model $0" >&2
  exit 1
fi

# --- Build command --------------------------------------------------------- #
CMD=(
  "$PYTHON_BIN" "$BENCH"
  --model "$MODEL"
  --mode "$MODE"
  --prefill-len "$PREFILL_LEN"
  --beam-width "$BEAM_WIDTH"
  --output-tokens "$OUTPUT_TOKENS"
  --chunk-size "$CHUNK_SIZE"
  --max-num-seqs "$MAX_NUM_SEQS"
  --block-size "$BLOCK_SIZE"
  --max-model-len "$MAX_MODEL_LEN"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --warmup "$WARMUP"
  --iters "$ITERS"
)

if [[ -n "$MAX_NUM_BATCHED_TOKENS" ]]; then
  CMD+=(--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS")
fi

if [[ "$ENFORCE_EAGER" == "1" || "$ENFORCE_EAGER" == "true" || "$ENFORCE_EAGER" == "on" ]]; then
  CMD+=(--enforce-eager)
fi

if [[ "$PROFILE" == "1" || "$PROFILE" == "true" || "$PROFILE" == "on" ]]; then
  CMD+=(--profile --profile-dir "$PROFILE_DIR")
fi

# --- Run ------------------------------------------------------------------- #
echo "Running P/D dual-stream microbenchmark from: ${REPO_ROOT}"
echo "Command:"
printf '  %q' "${CMD[@]}"
printf '\n\n'

cd "$REPO_ROOT"
"${CMD[@]}"
