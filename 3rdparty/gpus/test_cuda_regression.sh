#!/usr/bin/env bash
#
# test_cuda_regression.sh — Verify that MUSA changes do not break existing CUDA builds/tests.
#
# Usage:
#   ./3rdparty/gpus/test_cuda_regression.sh [--skip-gpu-tests]
#
# This script performs three levels of verification:
#   1. Non-GPU compilation: build client without any GPU config (GpuStream_t = void* path)
#   2. CUDA compilation: build the full SDK with --config=cuda12
#   3. CUDA tests: run existing CUDA unit tests (requires NVIDIA GPU, skippable)
#
# Exit code 0 means all checks passed.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

SKIP_GPU_TESTS=false
if [[ "${1:-}" == "--skip-gpu-tests" ]]; then
    SKIP_GPU_TESTS=true
fi

pass() { echo -e "${GREEN}[PASS]${NC} $1"; }
fail() { echo -e "${RED}[FAIL]${NC} $1"; exit 1; }
info() { echo -e "${YELLOW}[INFO]${NC} $1"; }

# --------------------------------------------------------------------------
# Test 1: Non-GPU build (no CUDA, no MUSA — exercises GpuStream_t = void* path)
# --------------------------------------------------------------------------
info "Test 1/3: Building client without GPU config (GpuStream_t = void* fallback)..."
if bazel build //kv_cache_manager/client/src/internal/sdk:sdk 2>&1; then
    pass "Non-GPU client SDK build succeeded"
else
    fail "Non-GPU client SDK build failed — GpuStream_t abstraction may be broken"
fi

# --------------------------------------------------------------------------
# Test 2: CUDA build (exercises GpuStream_t = cudaStream_t path)
# --------------------------------------------------------------------------
info "Test 2/3: Building SDK with --config=cuda12..."
if bazel build //kv_cache_manager/client/src/internal/sdk:sdk --config=cuda12 2>&1; then
    pass "CUDA SDK build succeeded"
else
    fail "CUDA SDK build failed — MUSA changes may have broken CUDA compilation"
fi

info "Building CUDA buffer check util..."
if bazel build //kv_cache_manager/client/src/internal/sdk:sdk_buffer_check_util --config=cuda12 2>&1; then
    pass "CUDA sdk_buffer_check_util build succeeded"
else
    fail "CUDA sdk_buffer_check_util build failed"
fi

info "Building transfer client with CUDA..."
if bazel build //kv_cache_manager/client/... --config=client_with_cuda 2>&1; then
    pass "client_with_cuda build succeeded"
else
    fail "client_with_cuda build failed"
fi

# --------------------------------------------------------------------------
# Test 3: CUDA unit tests (requires NVIDIA GPU)
# --------------------------------------------------------------------------
if $SKIP_GPU_TESTS; then
    info "Test 3/3: Skipping GPU tests (--skip-gpu-tests flag set)"
else
    info "Test 3/3: Running CUDA unit tests..."
    if bazel test //kv_cache_manager/client/src/internal/sdk/test:SdkBufferCheckUtilTest --config=cuda12 2>&1; then
        pass "SdkBufferCheckUtilTest passed"
    else
        fail "SdkBufferCheckUtilTest failed — existing CUDA functionality is broken"
    fi

    info "Running LocalFileSdkTest..."
    if bazel test //kv_cache_manager/client/src/internal/sdk/test:LocalFileSdkTest --config=cuda12 2>&1; then
        pass "LocalFileSdkTest passed"
    else
        fail "LocalFileSdkTest failed"
    fi

    info "Running SdkWrapperTest..."
    if bazel test //kv_cache_manager/client/src/internal/sdk/test:SdkWrapperTest --config=cuda12 2>&1; then
        pass "SdkWrapperTest passed"
    else
        fail "SdkWrapperTest failed"
    fi
fi

echo ""
echo -e "${GREEN}=== All CUDA regression checks passed ===${NC}"
