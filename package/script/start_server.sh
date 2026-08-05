#!/bin/bash

set -x

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
ROOT_PATH=${SCRIPT_DIR%/bin}
KVCM_OPS_WHEEL_PATH=$ROOT_PATH/kvcm_ops-0.1.0-py3-none-any.whl
BINARY_PATH=$ROOT_PATH/bin
CONFIG_PATH=$ROOT_PATH/etc
DEFAULT_SERVER_CONFIG=$CONFIG_PATH/default_server_config.conf
DEFAULT_LOGGER_CONFIG=$CONFIG_PATH/default_logger_config.conf
BINARY=$BINARY_PATH/kv_cache_manager_bin

function configure_jemalloc() {
    if [ "${KVCM_USE_JEMALLOC:-1}" = "0" ]; then
        echo "jemalloc disabled by KVCM_USE_JEMALLOC=0"
        return 0
    fi

    local arch
    arch=$(uname -m)
    local candidates=()
    if [ -n "${KVCM_JEMALLOC_PATH:-}" ]; then
        candidates+=("$KVCM_JEMALLOC_PATH")
    fi
    case "$arch" in
        x86_64 | amd64)
            candidates+=(
                "/usr/lib/x86_64-linux-gnu/libjemalloc.so.2"
                "/usr/lib64/libjemalloc.so.2"
            )
            ;;
        aarch64 | arm64)
            candidates+=(
                "/usr/lib/aarch64-linux-gnu/libjemalloc.so.2"
                "/usr/lib64/libjemalloc.so.2"
            )
            ;;
        *)
            echo "unsupported architecture for jemalloc auto-detection: $arch" >&2
            return 0
            ;;
    esac

    local jemalloc_path=""
    local candidate
    for candidate in "${candidates[@]}"; do
        if [ -r "$candidate" ]; then
            jemalloc_path=$candidate
            break
        fi
    done
    if [ -z "$jemalloc_path" ]; then
        echo "jemalloc library not found for architecture $arch; continue with the default allocator" >&2
        return 0
    fi

    case ":${LD_PRELOAD// /:}:" in
        *":$jemalloc_path:"*) ;;
        *) export LD_PRELOAD="$jemalloc_path${LD_PRELOAD:+:$LD_PRELOAD}" ;;
    esac
    echo "jemalloc enabled: LD_PRELOAD=$LD_PRELOAD"
}

function install_kvcm_ops() {
    python3 -m pip install "$KVCM_OPS_WHEEL_PATH"
}

function start_server() {
    echo "start server at: "$BINARY
    exec $BINARY -c $DEFAULT_SERVER_CONFIG -l $DEFAULT_LOGGER_CONFIG "$@"
}

function main() {
    configure_jemalloc
    install_kvcm_ops
    start_server "$@"
}

main "$@"
