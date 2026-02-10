#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$SCRIPT_DIR"

# ================= Configuration =================
HISIM_BENCHMARK_OUT_DIR="$SCRIPT_DIR/tmp/server"
BASE_OUTPUT_DIR="$SCRIPT_DIR/data"
SERVER_URL="http://localhost:30000"
request_rates=(1 2 4 6 8 12 16 20 24 28 32)
# ===========================================

echo "Starting benchmark loop..."

for rate in "${request_rates[@]}"; do
    echo "========================================"
    echo "Processing Request Rate: $rate"
    
    OUR_DIR="$BASE_OUTPUT_DIR/$rate"
    mkdir -p "$OUR_DIR"

    echo "Flushing cache..."
    if ! curl -s "$SERVER_URL/flush_cache" > /dev/null; then
        echo "Error: Failed to flush cache. Is the server running?"
        exit 1
    fi

    echo "Clean server..."
    curl -s "$SERVER_URL/start_profile" > /dev/null

    echo "Running benchmark..."
    python3 -m sglang.bench_serving --backend sglang \
        --dataset-name random --num-prompts 100 \
        --random-input 2000 --random-output 512 \
        --random-range-ratio 0.5 \
        --request-rate "$rate" \
        --warmup-requests 0 \
        --output-file "$OUR_DIR/metrics.json"

    echo "Export data..."
    curl -s "$SERVER_URL/stop_profile" > /dev/null

    echo "Collecting logs..."
    
    if [ -f "$HISIM_BENCHMARK_OUT_DIR/TP0.requests.jsonl" ]; then
        mv "$HISIM_BENCHMARK_OUT_DIR/TP0.requests.jsonl" "$OUR_DIR/requests.jsonl"
    else
        echo "Warning: TP0.requests.jsonl not found for rate $rate"
    fi

    echo "Rate $rate completed."
done

echo "All benchmarks finished."