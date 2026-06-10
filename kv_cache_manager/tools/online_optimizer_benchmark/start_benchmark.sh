#!/bin/bash
# Start the online_optimizer benchmark.
# Designed for Drogo deployment: expects $HIPPO_APP_REAL_WORKDIR to be set.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Online Optimizer Benchmark ==="
echo "Working directory: ${SCRIPT_DIR}"

# Install dependencies if requirements.txt exists
if [ -f "${SCRIPT_DIR}/requirements.txt" ]; then
    echo "Installing Python dependencies..."
    pip3 install --quiet -r "${SCRIPT_DIR}/requirements.txt"
fi

# Auto-detect bundled trace data for trace_replay mode
if [ "${BENCH_MODE:-}" = "trace_replay" ] && [ -z "${BENCH_TRACE_DATA_DIR:-}" ]; then
    BUNDLED_TRACE_DIR="${SCRIPT_DIR}/trace_data"
    if [ -d "${BUNDLED_TRACE_DIR}" ]; then
        export BENCH_TRACE_DATA_DIR="${BUNDLED_TRACE_DIR}"
        echo "Using bundled trace data: ${BUNDLED_TRACE_DIR}"
    fi
fi

# Run the benchmark (must cd to SCRIPT_DIR so 'python3 -m benchmark.main' finds the package)
echo "Starting benchmark..."
cd "${SCRIPT_DIR}"
python3 -m benchmark.main

echo "Benchmark finished."
