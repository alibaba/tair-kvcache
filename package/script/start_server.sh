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
BOOTSTRAP_RETRY_INTERVAL_SECONDS=5
BOOTSTRAP_MAX_RETRIES=2
BOOTSTRAP_FOLLOWER_EXIT_CODE=2
BOOTSTRAP_RESTART_REQUIRED_EXIT_CODE=3
BOOTSTRAP_MAX_SERVER_RESTARTS=1
server_pid=""
bootstrap_pid=""
shutdown_requested=0
shutdown_signal=""

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

function forward_signal() {
    local signal=$1
    if [ -n "$server_pid" ] && kill -0 "$server_pid" 2>/dev/null; then
        kill "-$signal" "$server_pid" 2>/dev/null || true
    fi
}

function forward_bootstrap_signal() {
    local signal=$1
    if [ -n "$bootstrap_pid" ] && kill -0 "$bootstrap_pid" 2>/dev/null; then
        kill "-$signal" "$bootstrap_pid" 2>/dev/null || true
    fi
}

function handle_shutdown_signal() {
    local signal=$1
    shutdown_requested=1
    shutdown_signal=$signal
    forward_signal "$signal"
    forward_bootstrap_signal "$signal"
}

function wait_for_pid() {
    local pid=$1
    local child_status=0
    while true; do
        if wait "$pid"; then
            child_status=0
        else
            child_status=$?
        fi
        if ! kill -0 "$pid" 2>/dev/null; then
            return "$child_status"
        fi
    done
}

function wait_for_server() {
    if [ -z "$server_pid" ]; then
        return 0
    fi
    wait_for_pid "$server_pid"
}

function wait_for_admin_health() {
    while kill -0 "$server_pid" 2>/dev/null; do
        if command -v curl >/dev/null 2>&1; then
            if curl -fsS --max-time 2 http://127.0.0.1:6492/api/healthy >/dev/null 2>&1; then
                return 0
            fi
        elif python3 -c 'import requests,sys; r=requests.get("http://127.0.0.1:6492/api/healthy", timeout=2); sys.exit(0 if r.status_code == 200 else 1)' >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
    done
    return 1
}

function wait_for_bootstrap_retry() {
    local elapsed=0
    while [ "$elapsed" -lt "$BOOTSTRAP_RETRY_INTERVAL_SECONDS" ]; do
        if ! kill -0 "$server_pid" 2>/dev/null; then
            return 1
        fi
        sleep 1
        elapsed=$((elapsed + 1))
    done
    return 0
}

function run_bootstrap_command() {
    local bootstrap_status=0
    python3 -m kvcm_ops bootstrap &
    bootstrap_pid=$!
    wait_for_pid "$bootstrap_pid" || bootstrap_status=$?
    bootstrap_pid=""
    return "$bootstrap_status"
}

function run_bootstrap_monitor() {
    if ! wait_for_admin_health; then
        return 0
    fi

    local retry_count=0
    while kill -0 "$server_pid" 2>/dev/null; do
        local bootstrap_status=0
        run_bootstrap_command || bootstrap_status=$?
        if [ "$shutdown_requested" -eq 1 ]; then
            return 0
        fi
        if [ "$bootstrap_status" -eq 0 ]; then
            return 0
        fi
        if [ "$bootstrap_status" -eq "$BOOTSTRAP_RESTART_REQUIRED_EXIT_CODE" ]; then
            return "$BOOTSTRAP_RESTART_REQUIRED_EXIT_CODE"
        fi
        if [ "$bootstrap_status" -eq "$BOOTSTRAP_FOLLOWER_EXIT_CODE" ]; then
            if ! wait_for_bootstrap_retry; then
                return 0
            fi
            continue
        fi
        if [ "$retry_count" -ge "$BOOTSTRAP_MAX_RETRIES" ]; then
            echo "ERROR: KVCM bootstrap failed after $BOOTSTRAP_MAX_RETRIES retries; automatic bootstrap stopped and KVCM remains running" >&2
            return 0
        fi
        retry_count=$((retry_count + 1))
        echo "ERROR: KVCM bootstrap failed; retry $retry_count/$BOOTSTRAP_MAX_RETRIES in $BOOTSTRAP_RETRY_INTERVAL_SECONDS seconds" >&2
        if ! wait_for_bootstrap_retry; then
            return 0
        fi
    done
}

function launch_server() {
    start_server "$@" &
    server_pid=$!
}

function main() {
    configure_jemalloc
    install_kvcm_ops

    trap 'handle_shutdown_signal TERM' TERM
    trap 'handle_shutdown_signal INT' INT

    local restart_count=0
    while true; do
        if [ "$shutdown_requested" -eq 1 ]; then
            return 0
        fi
        launch_server "$@"
        if [ "$shutdown_requested" -eq 1 ]; then
            local shutdown_status=0
            forward_signal "$shutdown_signal"
            wait_for_server || shutdown_status=$?
            return "$shutdown_status"
        fi

        local bootstrap_status=0
        run_bootstrap_monitor || bootstrap_status=$?
        if [ "$bootstrap_status" -eq "$BOOTSTRAP_RESTART_REQUIRED_EXIT_CODE" ]; then
            if [ "$restart_count" -ge "$BOOTSTRAP_MAX_SERVER_RESTARTS" ]; then
                echo "ERROR: KVCM bootstrap still requires restart after $restart_count automatic restart(s)" >&2
                forward_signal TERM
                wait_for_server
                return 1
            fi
            echo "KVCM MetaIndexer configuration changed; restarting server to recover instances with the new configuration"
            forward_signal TERM
            wait_for_server || true
            server_pid=""
            restart_count=$((restart_count + 1))
            continue
        fi

        local server_status=0
        wait_for_server || server_status=$?
        return "$server_status"
    done
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
