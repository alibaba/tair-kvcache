#!/bin/bash

set -x

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")

# Spectrum 服务发现依赖本地 sidecar 容器暴露的 HTTP 接口：
#   http://127.0.0.1:8880/api/v1/discovery/virtual-services/{id}/instances
# 在 kvcm 启动前需要确保该接口已经联通（sidecar 已就绪），
# 否则 SpectrumServiceDiscovery 初始化阶段拉取实例会失败。
#
# virtual-id 优先取 SPECTRUM_APPLICATION_SERVICE_ID，未设置时用 fake-id。
# 只要 HTTP 状态码是 200 就认为 sidecar 已可服务，不关心返回内容。
SPECTRUM_HOST=${SPECTRUM_HOST:-"127.0.0.1"}
SPECTRUM_PORT=${SPECTRUM_PORT:-8880}
SPECTRUM_VIRTUAL_ID=${SPECTRUM_APPLICATION_SERVICE_ID:-fake-id}
SPECTRUM_PROBE_INTERVAL_SEC=${SPECTRUM_PROBE_INTERVAL_SEC:-1}
SPECTRUM_PROBE_TIMEOUT_SEC=${SPECTRUM_PROBE_TIMEOUT_SEC:-2}
SPECTRUM_PROBE_MAX_ATTEMPTS=${SPECTRUM_PROBE_MAX_ATTEMPTS:-60}

if [ -z "${SPECTRUM_PROBE_URL:-}" ]; then
    SPECTRUM_PROBE_URL="http://${SPECTRUM_HOST}:${SPECTRUM_PORT}/api/v1/discovery/virtual-services/${SPECTRUM_VIRTUAL_ID}/instances"
fi

function wait_for_spectrum_sidecar() {
    echo "wait for spectrum sidecar: $SPECTRUM_PROBE_URL (max_attempts=$SPECTRUM_PROBE_MAX_ATTEMPTS)"
    local attempt=0
    while [ "$attempt" -lt "$SPECTRUM_PROBE_MAX_ATTEMPTS" ]; do
        attempt=$((attempt + 1))
        http_code=$(curl -s -o /dev/null \
            -m "$SPECTRUM_PROBE_TIMEOUT_SEC" \
            -w "%{http_code}" \
            "$SPECTRUM_PROBE_URL" || echo "000")
        if [ "$http_code" = "200" ]; then
            echo "spectrum sidecar ready after ${attempt} attempt(s), http_code=$http_code"
            return 0
        fi
        echo "spectrum sidecar not ready, attempt=${attempt}/${SPECTRUM_PROBE_MAX_ATTEMPTS}, http_code=$http_code, retry in ${SPECTRUM_PROBE_INTERVAL_SEC}s"
        sleep "$SPECTRUM_PROBE_INTERVAL_SEC"
    done
    echo "spectrum sidecar not ready after ${SPECTRUM_PROBE_MAX_ATTEMPTS} attempt(s): $SPECTRUM_PROBE_URL" >&2
    return 1
}

function main() {
    wait_for_spectrum_sidecar || exit 1
    exec "$SCRIPT_DIR/start_server.sh" "$@"
}

main "$@"
