#!/bin/bash
# Pack the online_optimizer benchmark into a deployable tarball.
# Usage:
#   ./pack_benchmark.sh                          # pack without trace data
#   ./pack_benchmark.sh /path/to/trace_data_dir  # pack with trace data
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

TRACE_DATA_DIR="${1:-}"

PACK_DIR="${PROJECT_ROOT}/bazel-bin/package"
BIN_DIR="${PACK_DIR}/bin"

rm -rf "${PACK_DIR}"
mkdir -p "${BIN_DIR}/benchmark"

# Copy benchmark Python sources
cp -r "${SCRIPT_DIR}/"*.py "${BIN_DIR}/benchmark/"

# Copy start script
cp "${SCRIPT_DIR}/start_benchmark.sh" "${BIN_DIR}/start_benchmark.sh"
chmod +x "${BIN_DIR}/start_benchmark.sh"

# Copy trace data if provided
if [ -n "${TRACE_DATA_DIR}" ]; then
    if [ ! -d "${TRACE_DATA_DIR}" ]; then
        echo "ERROR: trace data directory not found: ${TRACE_DATA_DIR}"
        exit 1
    fi
    TRACE_DEST="${BIN_DIR}/trace_data"
    mkdir -p "${TRACE_DEST}"
    echo "Copying trace data from ${TRACE_DATA_DIR} ..."
    cp -r "${TRACE_DATA_DIR}/"*.jsonl "${TRACE_DEST}/"
    FILE_COUNT=$(ls -1 "${TRACE_DEST}/"*.jsonl 2>/dev/null | wc -l)
    echo "Copied ${FILE_COUNT} trace files."
fi

# Create requirements.txt
cat > "${BIN_DIR}/requirements.txt" <<EOF
requests>=2.28.0
EOF

# Create tarball (no compression for speed, especially with large trace data)
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
TARBALL="${PROJECT_ROOT}/bazel-bin/package/online_optimizer_benchmark_${TIMESTAMP}.tar"
PACK_SIZE=$(du -sh "${PACK_DIR}" | cut -f1)
echo "Packing ${PACK_SIZE} into tarball..."
(cd "${PACK_DIR}" && tar -cf "${TARBALL}" .)

echo "Benchmark packed: ${TARBALL}"
echo "Contents:"
tar -tf "${TARBALL}" | head -20
