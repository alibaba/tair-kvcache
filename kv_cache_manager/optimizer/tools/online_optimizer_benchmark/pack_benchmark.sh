#!/bin/bash
# Pack the online_optimizer benchmark into a deployable tarball.
# Usage:
#   ./pack_benchmark.sh                          # pack without trace data
#   ./pack_benchmark.sh /path/to/trace_data_dir  # pack with trace data
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

TRACE_DATA_DIR="${1:-}"

PACK_DIR="${PROJECT_ROOT}/bazel-bin/package"
BIN_DIR="${PACK_DIR}/bin"

rm -rf "${PACK_DIR}"
mkdir -p "${BIN_DIR}/benchmark"
mkdir -p "${BIN_DIR}/kv_cache_manager/optimizer/client"
mkdir -p "${BIN_DIR}/kv_cache_manager/protocol/protobuf"

# Copy benchmark Python sources
cp "${SCRIPT_DIR}/__init__.py" "${BIN_DIR}/benchmark/"
cp "${SCRIPT_DIR}/main.py" "${BIN_DIR}/benchmark/"
cp -r "${SCRIPT_DIR}/core" "${BIN_DIR}/benchmark/"
cp -r "${SCRIPT_DIR}/runners" "${BIN_DIR}/benchmark/"
cp -r "${SCRIPT_DIR}/workload" "${BIN_DIR}/benchmark/"

# Copy Online Optimizer Python client SDK (gRPC + HTTP) and proto definition.
cp "${PROJECT_ROOT}/kv_cache_manager/__init__.py" "${BIN_DIR}/kv_cache_manager/"
cp "${PROJECT_ROOT}/kv_cache_manager/optimizer/__init__.py" "${BIN_DIR}/kv_cache_manager/optimizer/"
cp "${PROJECT_ROOT}/kv_cache_manager/optimizer/client/__init__.py" "${BIN_DIR}/kv_cache_manager/optimizer/client/"
cp "${PROJECT_ROOT}/kv_cache_manager/optimizer/client/base.py" "${BIN_DIR}/kv_cache_manager/optimizer/client/"
cp "${PROJECT_ROOT}/kv_cache_manager/optimizer/client/grpc_client.py" "${BIN_DIR}/kv_cache_manager/optimizer/client/"
cp "${PROJECT_ROOT}/kv_cache_manager/optimizer/client/http_client.py" "${BIN_DIR}/kv_cache_manager/optimizer/client/"
cp "${PROJECT_ROOT}/kv_cache_manager/protocol/__init__.py" "${BIN_DIR}/kv_cache_manager/protocol/"
cp "${PROJECT_ROOT}/kv_cache_manager/protocol/protobuf/__init__.py" "${BIN_DIR}/kv_cache_manager/protocol/protobuf/"
cp "${PROJECT_ROOT}/kv_cache_manager/protocol/protobuf/optimizer_service.proto" "${BIN_DIR}/kv_cache_manager/protocol/protobuf/"

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
grpcio>=1.48.0
grpcio-tools>=1.48.0
protobuf>=3.19.0
EOF

# Create tarball (no compression for speed, especially with large trace data)
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
TARBALL="${PROJECT_ROOT}/bazel-bin/package/online_optimizer_benchmark_${TIMESTAMP}.tar"
PACK_SIZE=$(du -sh "${PACK_DIR}" --exclude='*.tar' 2>/dev/null | cut -f1)
echo "Packing ${PACK_SIZE} into tarball..."
(
    cd "${PACK_DIR}"
    tar \
        --exclude='./online_optimizer_benchmark_*.tar' \
        --exclude='*/__pycache__' \
        --exclude='*/__pycache__/*' \
        -cf "${TARBALL}" .
)

echo "Benchmark packed: ${TARBALL}"
echo "Contents:"
set +o pipefail
tar -tf "${TARBALL}" | head -20
set -o pipefail
