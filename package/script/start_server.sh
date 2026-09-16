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
HOME_ADMIN_LOGGER_CONFIG=$CONFIG_PATH/home_admin_logger_config.conf
KVCM_LOG_TARGET_DIR=/home/admin/logs/kvcm
LOGGER_CONFIG=$DEFAULT_LOGGER_CONFIG
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
    echo "jemalloc enabled: LD_PRELOAD=$LD_PRELOAD MALLOC_CONF=${MALLOC_CONF:-<unset>}"
}

function install_kvcm_ops() {
    python3 -m pip install "$KVCM_OPS_WHEEL_PATH"
}

function use_home_admin_logger_config_if_writable() {
    local probe_file
    local probe_content="kvcm-log-directory-check"
    local read_content
    local log_file
    local log_files=(
        "$KVCM_LOG_TARGET_DIR/kv_cache_manager.log"
        "$KVCM_LOG_TARGET_DIR/access.log"
        "$KVCM_LOG_TARGET_DIR/metrics.log"
        "$KVCM_LOG_TARGET_DIR/event_publisher.log"
    )

    if [ ! -r "$HOME_ADMIN_LOGGER_CONFIG" ]; then
        echo "logger config is not readable, use default config: $HOME_ADMIN_LOGGER_CONFIG" >&2
        return 1
    fi

    if ! mkdir -p "$KVCM_LOG_TARGET_DIR"; then
        echo "failed to create log directory, use default config: $KVCM_LOG_TARGET_DIR" >&2
        return 1
    fi

    for log_file in "${log_files[@]}"; do
        if [ -e "$log_file" ] && { [ ! -f "$log_file" ] || [ ! -w "$log_file" ]; }; then
            echo "log file is not writable, use default config: $log_file" >&2
            return 1
        fi
    done

    if ! probe_file=$(mktemp "$KVCM_LOG_TARGET_DIR/.kvcm-log-check.XXXXXX"); then
        echo "failed to create file in log directory, use default config: $KVCM_LOG_TARGET_DIR" >&2
        return 1
    fi

    if ! printf '%s\n' "$probe_content" > "$probe_file"; then
        echo "failed to write log directory, use default config: $KVCM_LOG_TARGET_DIR" >&2
        rm -f "$probe_file"
        return 1
    fi

    if ! IFS= read -r read_content < "$probe_file" || [ "$read_content" != "$probe_content" ]; then
        echo "failed to read log directory, use default config: $KVCM_LOG_TARGET_DIR" >&2
        rm -f "$probe_file"
        return 1
    fi

    if ! rm -f "$probe_file"; then
        echo "failed to remove probe file, use default config: $probe_file" >&2
        return 1
    fi

    LOGGER_CONFIG=$HOME_ADMIN_LOGGER_CONFIG
    echo "use logger config: $LOGGER_CONFIG"
}

function start_server() {
    echo "start server at: $BINARY"
    exec "$BINARY" -c "$DEFAULT_SERVER_CONFIG" -l "$LOGGER_CONFIG" "$@"
}

function main() {
    configure_jemalloc
    install_kvcm_ops
    use_home_admin_logger_config_if_writable || true
    start_server "$@"
}

main "$@"
