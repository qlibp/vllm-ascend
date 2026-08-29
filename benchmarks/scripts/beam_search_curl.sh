#!/bin/bash
# Send a single beam-search request to the OpenAI-compatible completions endpoint.
#
# Beam search is non-streaming and `n` is the beam width.
#
# Usage:
#   BASE_URL=http://127.0.0.1:8000 \
#   MODEL=your-model \
#   BEAM_WIDTH=4 \
#   PROMPT="What is the capital of France?" \
#   ./beam_search_curl.sh

set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
MODEL="${MODEL:-}"
PROMPT="${PROMPT:-What is the capital of France?}"
MAX_TOKENS="${MAX_TOKENS:-128}"
BEAM_WIDTH="${BEAM_WIDTH:-4}"
TEMPERATURE="${TEMPERATURE:-0.0}"

if [[ -z "$MODEL" ]]; then
  MODEL=$(curl -s "$BASE_URL/v1/models" | jq -r '.data[0].id')
fi

curl -s "$BASE_URL/v1/completions" \
  -H "Content-Type: application/json" \
  -d "$(jq -nc \
    --arg model "$MODEL" \
    --arg prompt "$PROMPT" \
    --argjson max_tokens "$MAX_TOKENS" \
    --argjson n "$BEAM_WIDTH" \
    --argjson temperature "$TEMPERATURE" \
    '{model:$model, prompt:$prompt, use_beam_search:true, n:$n, max_tokens:$max_tokens, temperature:$temperature, stream:false}')" \
  | jq .
