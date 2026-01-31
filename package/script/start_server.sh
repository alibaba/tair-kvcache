#!/bin/bash

set -x

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
ROOT_PATH=${SCRIPT_DIR%/bin}
BINARY_PATH=$ROOT_PATH/bin
CONFIG_PATH=$ROOT_PATH/etc
DEFAULT_SERVER_CONFIG=$CONFIG_PATH/default_server_config.conf
DEFAULT_LOGGER_CONFIG=$CONFIG_PATH/default_logger_config.conf
BINARY=$BINARY_PATH/kv_cache_manager_bin

function start_server() {
    echo "start server at: "$BINARY
    exec $BINARY -c $DEFAULT_SERVER_CONFIG -l $DEFAULT_LOGGER_CONFIG "$@"
}

function main() {
    start_server "$@"
}

main "$@"