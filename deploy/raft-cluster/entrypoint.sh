#!/bin/bash
set -e

NODE_CONF="${NODE_CONF:-/conf/node1.conf}"
BINARY="/opt/kvcm/kv_cache_manager_bin"

mkdir -p /data/raft /data/nfs

echo "=== Starting KVCM with config: $NODE_CONF ==="
exec "$BINARY" -c "$NODE_CONF" -l /conf/logger.conf
