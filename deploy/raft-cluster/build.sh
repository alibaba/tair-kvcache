#!/bin/bash
set -e

OUTPUT_DIR="/opt/kvcm"
BINARY="$OUTPUT_DIR/kv_cache_manager_bin"

if [ -f "$BINARY" ]; then
    echo "=== Binary already cached, skipping build ==="
    exit 0
fi

echo "=== Building kv_cache_manager_bin ==="
cd /src
bazelisk build //kv_cache_manager:kv_cache_manager_bin 2>&1

cp -f bazel-bin/kv_cache_manager/kv_cache_manager_bin "$BINARY"
chmod +x "$BINARY"
echo "=== Build complete ==="
