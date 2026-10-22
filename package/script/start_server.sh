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
BOOTSTRAP_RETRY_INTERVAL_SECONDS=5
BOOTSTRAP_MAX_RETRIES=2
BOOTSTRAP_FOLLOWER_EXIT_CODE=2
ADMIN_URL=""
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

function resolve_admin_url() {
    local config_file=$DEFAULT_SERVER_CONFIG
    local arg_port=""
    local arg_value
    local arg
    local key
    local value
    local index
    local args=("$@")
    for ((index = 0; index < ${#args[@]}; index++)); do
        arg=${args[index]}
        case "$arg" in
            -c|--config_file)
                index=$((index + 1))
                config_file=${args[index]}
                ;;
            --config_file=*) config_file=${arg#*=} ;;
            -c?*) config_file=${arg#-c} ;;
            -e|--env)
                index=$((index + 1))
                arg_value=${args[index]}
                ;;
            --env=*) arg_value=${arg#*=} ;;
            -e?*) arg_value=${arg#-e} ;;
            *) continue ;;
        esac
        if [[ "$arg" == -e || "$arg" == --env || "$arg" == --env=* || "$arg" == -e?* ]]; then
            key=${arg_value%%=*}
            value=${arg_value#*=}
            # CommandArgs::ParseCmdLine keeps the first value for a repeated -e key.
            if [ "$key" = "kvcm.service.admin_http_port" ] && [ -z "$arg_port" ]; then
                arg_port=$value
            fi
        fi
    done

    if [ ! -r "$config_file" ]; then
        echo "ERROR: cannot read KVCM config file: $config_file" >&2
        return 1
    fi
    local port
    port=$(awk '
        { sub(/#.*/, ""); split($0, setting, "=");
          key = setting[1]; value = setting[2];
          gsub(/^[[:space:]]+|[[:space:]]+$/, "", key);
          gsub(/^[[:space:]]+|[[:space:]]+$/, "", value);
          if (key == "kvcm.service.admin_http_port") port = value }
        END { print port }
    ' "$config_file")
    if [ -n "$arg_port" ]; then
        port=$arg_port
    fi
    local env_port
    env_port=$(printenv 'kvcm.service.admin_http_port' 2>/dev/null || true)
    if [ -z "$env_port" ]; then
        env_port=${kvcm_service_admin_http_port:-}
    fi
    if [ -n "$env_port" ]; then
        port=$env_port
    fi
    port=${port//[[:space:]]/}
    if [[ ! "$port" =~ ^[0-9]+$ ]] || ((10#$port < 1 || 10#$port > 65535)); then
        echo "ERROR: invalid kvcm.service.admin_http_port: $port" >&2
        return 1
    fi
    printf 'http://127.0.0.1:%d\n' "$((10#$port))"
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
            if curl -fsS --max-time 2 "$ADMIN_URL/api/healthy" >/dev/null 2>&1; then
                return 0
            fi
        elif python3 -c 'import requests,sys; r=requests.get(sys.argv[1], timeout=2); sys.exit(0 if r.status_code == 200 else 1)' "$ADMIN_URL/api/healthy" >/dev/null 2>&1; then
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
    python3 -m kvcm_ops bootstrap --host "$ADMIN_URL" &
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
    use_home_admin_logger_config_if_writable || true
    ADMIN_URL=$(resolve_admin_url "$@") || return 1

    trap 'handle_shutdown_signal TERM' TERM
    trap 'handle_shutdown_signal INT' INT

    launch_server "$@"
    if [ "$shutdown_requested" -eq 1 ]; then
        local shutdown_status=0
        forward_signal "$shutdown_signal"
        wait_for_server || shutdown_status=$?
        return "$shutdown_status"
    fi

    run_bootstrap_monitor

    local server_status=0
    wait_for_server || server_status=$?
    return "$server_status"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
