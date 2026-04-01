#!/usr/bin/env bash
#
# test_musa_build.sh — Verify MUSA build configuration and compilation.
#
# Usage:
#   ./3rdparty/gpus/test_musa_build.sh [--run-tests]
#
# This script verifies:
#   1. MUSA Bazel config exists and is parseable
#   2. MUSA build system files are present and well-formed
#   3. SDK builds with --config=musa (requires MUSA toolkit at /usr/local/musa
#      or $MUSA_TOOLKIT_PATH)
#   4. Optionally runs MUSA tests (--run-tests, requires Moore Threads GPU)
#
# On machines without MUSA toolkit installed, only steps 1-2 run (structural
# validation). This is useful for CI environments that lack MUSA hardware.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

RUN_TESTS=false
if [[ "${1:-}" == "--run-tests" ]]; then
    RUN_TESTS=true
fi

pass() { echo -e "${GREEN}[PASS]${NC} $1"; }
fail() { echo -e "${RED}[FAIL]${NC} $1"; exit 1; }
info() { echo -e "${YELLOW}[INFO]${NC} $1"; }
skip() { echo -e "${YELLOW}[SKIP]${NC} $1"; }

MUSA_TOOLKIT="${MUSA_TOOLKIT_PATH:-/usr/local/musa}"

# --------------------------------------------------------------------------
# Test 1: Verify MUSA Bazel config entries in .bazelrc
# --------------------------------------------------------------------------
info "Test 1: Checking .bazelrc for MUSA config entries..."

check_bazelrc() {
    local pattern="$1"
    local desc="$2"
    if grep -q "$pattern" .bazelrc; then
        pass "$desc found in .bazelrc"
    else
        fail "$desc NOT found in .bazelrc"
    fi
}

check_bazelrc 'build:musa --copt="-DUSING_MUSA=1"'          "USING_MUSA define"
check_bazelrc 'build:musa --define=using_musa=true'          "using_musa define"
check_bazelrc 'build:client_with_musa --config=client'       "client_with_musa config"
check_bazelrc 'build:client_with_musa --config=musa'         "client_with_musa -> musa"

# --------------------------------------------------------------------------
# Test 2: Verify MUSA build system files exist and are well-formed
# --------------------------------------------------------------------------
info "Test 2: Checking MUSA build system files..."

check_file() {
    local filepath="$1"
    local desc="$2"
    if [[ -f "$filepath" ]]; then
        pass "$desc exists: $filepath"
    else
        fail "$desc MISSING: $filepath"
    fi
}

check_file "3rdparty/gpus/musa_configure.bzl"                 "musa_configure.bzl"
check_file "3rdparty/gpus/musa/BUILD.tpl"                    "MUSA BUILD template"
check_file "3rdparty/gpus/musa/build_defs.bzl.tpl"           "MUSA build_defs template"
check_file "3rdparty/gpus/musa/musa_config.h.tpl"            "MUSA config header template"

# Verify WORKSPACE has musa_configure
if grep -q 'musa_configure' WORKSPACE; then
    pass "WORKSPACE contains musa_configure"
else
    fail "WORKSPACE missing musa_configure"
fi

# Verify SDK BUILD has using_musa config_setting
if grep -q 'name = "using_musa"' kv_cache_manager/client/src/internal/sdk/BUILD; then
    pass "SDK BUILD has using_musa config_setting"
else
    fail "SDK BUILD missing using_musa config_setting"
fi

# --------------------------------------------------------------------------
# Test 3: Verify MUSA C++ source files exist
# --------------------------------------------------------------------------
info "Test 3: Checking MUSA C++ source files..."

SDK_DIR="kv_cache_manager/client/src/internal/sdk"

check_file "$SDK_DIR/musa_util.h"                            "MUSA error-check macros"
check_file "$SDK_DIR/hf3fs_musa_util.h"                      "Hf3fsMusaUtil class"
check_file "$SDK_DIR/sdk_buffer_check_util.mu"               "MUSA CRC32 kernel"
check_file "$SDK_DIR/sdk_buffer_check_util_musa.cc"          "MUSA host-side buffer check"

# Verify GpuStream_t alias exists in header
if grep -q 'using GpuStream_t' "$SDK_DIR/sdk_buffer_check_util.h"; then
    pass "GpuStream_t type alias present"
else
    fail "GpuStream_t type alias missing from sdk_buffer_check_util.h"
fi

# Verify conditional compilation guards (accept both #ifdef USING_MUSA and defined(USING_MUSA))
check_musa_guard() {
    local filepath="$1"
    local desc="$2"
    if grep -qE '(defined\(USING_MUSA\)|#ifdef USING_MUSA)' "$filepath"; then
        pass "USING_MUSA guard in $desc"
    else
        fail "USING_MUSA guard missing from $desc"
    fi
}

check_musa_guard "$SDK_DIR/sdk_buffer_check_util.h" "sdk_buffer_check_util.h"
check_musa_guard "$SDK_DIR/local_file_sdk.cc"        "local_file_sdk.cc"
check_musa_guard "$SDK_DIR/hf3fs_usrbio_client.h"    "hf3fs_usrbio_client.h"

# --------------------------------------------------------------------------
# Test 4: Verify Python device-agnostic changes
# --------------------------------------------------------------------------
info "Test 4: Checking Python connector changes..."

PY_DIR="kv_cache_manager/py_connector"

if grep -q '_get_device_module' "$PY_DIR/vllm/data_transfer.py"; then
    pass "_get_device_module() helper in data_transfer.py"
else
    fail "_get_device_module() helper missing from data_transfer.py"
fi

if grep -q '_device_mod' "$PY_DIR/vllm/v1_connector.py"; then
    pass "_device_mod imported in v1_connector.py"
else
    fail "_device_mod not found in v1_connector.py"
fi

if grep -q "hasattr(tl, 'float8e4nv')" "$PY_DIR/kernel/batch_gather_scatter_helper.py"; then
    pass "Conditional float8 dtype in batch_gather_scatter_helper.py"
else
    fail "Conditional float8 dtype missing from batch_gather_scatter_helper.py"
fi

# No torch.cuda.* should remain in modified connector files
for f in "$PY_DIR/vllm/data_transfer.py" "$PY_DIR/vllm/v1_connector.py"; do
    if grep -qE '^\s.*torch\.cuda\.' "$f" 2>/dev/null; then
        fail "Residual torch.cuda.* call found in $f"
    else
        pass "No residual torch.cuda.* in $(basename "$f")"
    fi
done

# --------------------------------------------------------------------------
# Test 5: MUSA compilation (only if toolkit is available)
# --------------------------------------------------------------------------
if [[ -d "$MUSA_TOOLKIT" ]]; then
    info "Test 5: MUSA toolkit found at $MUSA_TOOLKIT — attempting build..."

    if bazel build //kv_cache_manager/client/src/internal/sdk:sdk --config=musa 2>&1; then
        pass "MUSA SDK build succeeded"
    else
        fail "MUSA SDK build failed"
    fi

    if bazel build //kv_cache_manager/client/... --config=client_with_musa 2>&1; then
        pass "client_with_musa build succeeded"
    else
        fail "client_with_musa build failed"
    fi

    if $RUN_TESTS; then
        info "Running MUSA unit tests..."
        if bazel test //kv_cache_manager/client/src/internal/sdk/test:LocalFileSdkTest --config=musa 2>&1; then
            pass "LocalFileSdkTest (MUSA) passed"
        else
            fail "LocalFileSdkTest (MUSA) failed"
        fi
    fi
else
    skip "Test 5: MUSA toolkit not found at $MUSA_TOOLKIT — skipping compilation tests"
    skip "(Set MUSA_TOOLKIT_PATH env var if installed elsewhere)"
fi

echo ""
echo -e "${GREEN}=== All MUSA validation checks passed ===${NC}"
