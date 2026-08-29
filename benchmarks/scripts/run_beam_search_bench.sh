#!/bin/bash
# Run a vLLM online-serving beam-search benchmark.
#
# The OpenAI-compatible server must already be running (e.g. `vllm serve ...`).
# Beam search is non-streaming and uses the /v1/completions endpoint.
#
# Usage:
#   BASE_URL=http://127.0.0.1:8000 \
#   MODEL=your-model \
#   BEAM_WIDTH=4 \
#   ./run_beam_search_bench.sh

set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
MODEL="${MODEL:-}"
DATASET_NAME="${DATASET_NAME:-random}"
INPUT_LEN="${INPUT_LEN:-128}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"
NUM_PROMPTS="${NUM_PROMPTS:-100}"
BEAM_WIDTH="${BEAM_WIDTH:-4}"
REQUEST_RATE="${REQUEST_RATE:-inf}"

MODEL_ARGS=()
if [[ -n "$MODEL" ]]; then
  MODEL_ARGS=(--model "$MODEL")
fi

vllm bench serve \
  --backend openai \
  --base-url "$BASE_URL" \
  "${MODEL_ARGS[@]}" \
  --dataset-name "$DATASET_NAME" \
  --input-len "$INPUT_LEN" \
  --output-len "$OUTPUT_LEN" \
  --num-prompts "$NUM_PROMPTS" \
  --use-beam-search \
  --n "$BEAM_WIDTH" \
  --request-rate "$REQUEST_RATE"
