#!/bin/bash
# Wrapper to run py_tpu_client_test.py with the conda/uv venv activated.
# Bazel's hermetic Python lacks jax/numpy; we source the venv to get them.

set -e

# In Bazel runfiles, SCRIPT_DIR is the package directory containing the .so and .py
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Activate venv (must contain jax, numpy)
source ~/vllm_tpu_env_yemu/bin/activate

# Run test with .so on PYTHONPATH
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"
export TPU_LOG_DIR="${TPU_LOG_DIR:-/tmp/tpu_logs}"

exec python3 "${SCRIPT_DIR}/py_tpu_client_test.py" "$@"
