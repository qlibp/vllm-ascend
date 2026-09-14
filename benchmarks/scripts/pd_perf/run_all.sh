#!/usr/bin/env bash
# Run each P/D performance probe in its own process for clean device state.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== test01: dual-stream replay overlap ==="
python "$DIR/test01_dual_stream_replay_overlap.py" "$@"

echo
echo "=== test02: set_stream_limit effect ==="
python "$DIR/test02_set_stream_limit_effect.py" "$@"

echo
echo "=== test03: host enqueue overhead ==="
python "$DIR/test03_host_enqueue_overhead.py" "$@"

echo
echo "=== test04: memory bandwidth contention ==="
python "$DIR/test04_memory_bandwidth_contention.py" "$@"
