#!/bin/bash
set -e

NODE_CONF="${NODE_CONF:-/conf/node1.conf}"
SRC_DIR="/src"
BINARY="$SRC_DIR/bazel-bin/kv_cache_manager/kv_cache_manager_bin"

mkdir -p /data/raft /data/nfs

cd "$SRC_DIR"

if [ ! -f "$BINARY" ]; then
    echo "=== Building kv_cache_manager_bin (first run, cached afterwards) ==="
    bazelisk build //kv_cache_manager:kv_cache_manager_bin 2>&1
    echo "=== Build complete ==="
fi

echo "=== Starting KVCM with config: $NODE_CONF ==="
exec "$BINARY" -c "$NODE_CONF" -l /conf/logger.conf
