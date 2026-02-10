#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$SCRIPT_DIR"

HISIM_BENCHMARK_OUT_DIR="$SCRIPT_DIR/tmp/server" \
python3 $SCRIPT_DIR/serving_hook/sglang_launch_server.py \
    --model-path="Qwen/Qwen3-8B"


: <<'EOF'
PYTHONPATH=$SCRIPT_DIR/serving_hook \
HISIM_BENCHMARK_OUT_DIR=$SCRIPT_DIR/tmp/server \
python3 python3 -m sglang.launch_server \
    --model-path="Qwen/Qwen3-8B"
EOF