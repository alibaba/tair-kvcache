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

function start_server() {
    echo "start server at: "$BINARY
    local env_args=()
    while IFS='=' read -r key value; do
        env_args+=(-e "${key}=${value}")
    done < <(env | grep '^kvcm\.')
    exec $BINARY -c $DEFAULT_SERVER_CONFIG -l $DEFAULT_LOGGER_CONFIG "${env_args[@]}" "$@"
}

function main() {
    python3 -m pip install "$KVCM_OPS_WHEEL_PATH"
    start_server "$@"
}

main "$@"