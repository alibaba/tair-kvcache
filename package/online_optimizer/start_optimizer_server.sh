#!/bin/bash

set -x

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
ROOT_PATH=${SCRIPT_DIR%/bin}
BINARY_PATH=$ROOT_PATH/bin
CONFIG_PATH=$ROOT_PATH/etc
DEFAULT_CONFIG=$CONFIG_PATH/default_optimizer_config.json
DEFAULT_LOGGER_CONFIG=$CONFIG_PATH/default_logger_config.conf
BINARY=$BINARY_PATH/online_optimizer_server_main

function start_server() {
    echo "start optimizer server at: "$BINARY
    exec $BINARY -c $DEFAULT_CONFIG "$@"
}

start_server "$@"
