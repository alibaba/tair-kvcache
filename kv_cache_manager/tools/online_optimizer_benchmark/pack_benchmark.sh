#!/bin/bash
# Pack the online_optimizer benchmark into a deployable tarball.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

PACK_DIR="${PROJECT_ROOT}/benchmark_pack"
BIN_DIR="${PACK_DIR}/bin"

rm -rf "${PACK_DIR}"
mkdir -p "${BIN_DIR}/benchmark"

# Copy benchmark Python sources
cp -r "${SCRIPT_DIR}/"*.py "${BIN_DIR}/benchmark/"

# Copy start script
cp "${SCRIPT_DIR}/start_benchmark.sh" "${BIN_DIR}/start_benchmark.sh"
chmod +x "${BIN_DIR}/start_benchmark.sh"

# Create requirements.txt
cat > "${BIN_DIR}/requirements.txt" <<EOF
requests>=2.28.0
EOF

# Create tarball
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
TARBALL="${PROJECT_ROOT}/online_optimizer_benchmark_${TIMESTAMP}.tar.gz"
(cd "${PACK_DIR}" && tar -czf "${TARBALL}" .)

echo "Benchmark packed: ${TARBALL}"
echo "Contents:"
tar -tzf "${TARBALL}"

rm -rf "${PACK_DIR}"
