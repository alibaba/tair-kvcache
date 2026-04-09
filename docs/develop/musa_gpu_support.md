# MUSA GPU Support for TAIR-KVCACHE

## 1. Overview

This document describes the design and implementation of Moore Threads MUSA GPU
support in TAIR-KVCACHE. MUSA (Moore Threads Unified Software Architecture) is
a GPU programming platform whose API with a `musa` prefix
(e.g., `cudaMalloc` -> `musaMalloc`, `cudaStream_t` -> `musaStream_t`).

MUSA support is added as a **parallel, mutually exclusive** GPU backend
alongside the existing CUDA backend. When building with `--config=musa`, the
`USING_MUSA` preprocessor macro is defined and all GPU-specific code paths use
MUSA APIs instead of CUDA APIs. No existing CUDA code paths are altered.

## 2. Motivation

- Enable TAIR-KVCACHE to run on Moore Threads GPU hardware
- Establish a pattern for adding additional GPU backends (AMD ROCm, Intel
  oneAPI, etc.) in the future

## 3. Design Principles

| Principle | Description |
|---|---|
| **Zero CUDA impact** | All changes are additive: new `#elif defined(USING_MUSA)` branches or new files. Existing CUDA code is unchanged (verified by test). |
| **Symmetry** | Every CUDA construct has a 1:1 MUSA counterpart: same class names with `Musa` prefix, same file layout, same error handling patterns. |
| **Build isolation** | `--config=musa` and `--config=cuda` are mutually exclusive Bazel configs. They cannot be combined. |
| **Type aliasing** | Shared headers use `GpuStream_t` (aliased to `cudaStream_t` or `musaStream_t`) so GPU-agnostic code compiles under either backend. |

## 4. Architecture

The changes span three layers:

```
┌─────────────────────────────────────────────────────┐
│                   Build System                      │
│  .bazelrc  ·  WORKSPACE  ·  3rdparty/gpus/musa*/     │
│  sdk/BUILD (using_musa config_setting)              │
├─────────────────────────────────────────────────────┤
│              C++ Runtime Abstraction                │
│  musa_util.h  ·  hf3fs_musa_util.h                 │
│  sdk_buffer_check_util.mu / _musa.cc               │
│  sdk_buffer_check_util.h (GpuStream_t alias)       │
│  local_file_sdk.h/cc  ·  transfer_client_impl.h/cc │
│  hf3fs_usrbio_client.h  ·  hf3fs_sdk.cc            │
├─────────────────────────────────────────────────────┤
│             Python Connector Layer                  │
│  data_transfer.py (_device_mod helper)              │
│  v1_connector.py (device-agnostic stream/event)     │
│  batch_gather_scatter_helper.py (float8 compat)     │
└─────────────────────────────────────────────────────┘
```

## 5. Detailed Changes

### 5.1 Build System

| File | Change |
|---|---|
| `.bazelrc` | Add `build:musa` config block (`-DUSING_MUSA=1`, `using_musa=true`, MUSA toolkit env vars) and `build:client_with_musa` |
| `WORKSPACE` | Load and call `musa_configure(name = "local_config_musa")` |
| `3rdparty/gpus/musa_configure.bzl` + `3rdparty/gpus/musa/` (new) | Repository rule to detect MUSA toolkit, generate `@local_config_musa//:musa` cc_library with headers and `libmusart.so` |
| `sdk/BUILD` | Add `using_musa` config_setting, MUSA conditional deps in `sdk` library, `sdk_buffer_check_util_musa` cc_library target |

### 5.2 C++ Runtime Layer

**New files:**

- `musa_util.h` — `CHECK_MUSA_ERROR` / `CHECK_MUSA_ERROR_RETURN` macros, mirrors `cuda_util.h`
- `hf3fs_musa_util.h` — `Hf3fsMusaUtil` class (stream management, host register, async memcpy), mirrors `Hf3fsCudaUtil` with no-op fallback when `USING_MUSA` is not defined
- `sdk_buffer_check_util.mu` — MUSA CRC32 GPU kernel (compiled by `mcc`; same syntax as CUDA)
- `sdk_buffer_check_util_musa.mu` — MUSA kernel + `GetIovsCrc(..., stream)` and `min_cal_byte_size_` (compiled by `mcc` to `.o`); handles `stream == nullptr` by creating/destroying a temporary stream
- `sdk_buffer_check_util_musa.cc` — Host-only buffer check (GetBlocksHash, pool, overloads); links with the `.o` from the `.mu` file

**Modified files:**

- `sdk_buffer_check_util.h` — Replaced unconditional `#include "cuda_util.h"` with conditional include; introduced `GpuStream_t` type alias; renamed `Cell::cuda_stream` to `Cell::gpu_stream`
- `sdk_buffer_check_util.cc` / `.cu` — Updated signatures to use `GpuStream_t` and `gpu_stream` field name
- `local_file_sdk.h/cc` — Added `#elif defined(USING_MUSA)` branches for stream creation, memcpy, host register, sync, and destruction (~10 blocks)
- `transfer_client_impl.h/cc` — Widened `#ifdef USING_CUDA` to `#if defined(USING_CUDA) || defined(USING_MUSA)` for buffer check initialization and usage
- `hf3fs_usrbio_client.h` — Added `musa_util` shared_ptr to `Hf3fsIovHandle` under `#ifdef USING_MUSA`
- `hf3fs_sdk.cc` — Added MUSA util initialization and cleanup path

**Key abstraction — `GpuStream_t`:**

```cpp
#if defined(USING_CUDA)
using GpuStream_t = cudaStream_t;
#elif defined(USING_MUSA)
using GpuStream_t = musaStream_t;
#else
using GpuStream_t = void *;
#endif
```

This enables `SdkBufferCheckUtil`, `SdkBufferCheckPool`, and `TransferClientImpl`
to work with either backend through a single type alias. CUDA builds resolve
`GpuStream_t` to `cudaStream_t` — no ABI or behavior change.

### 5.3 Python Connector Layer

- `data_transfer.py` — Added `_get_device_module(device)` helper and select device module from runtime device (`torch.get_device_module(device)` when available). Replaced hard-coded `torch.cuda.*` stream/event calls with device-module equivalents.
- `v1_connector.py` — Select device module from actual KV cache tensor device in `register_kv_caches`, then initialize stream/event on that module. This keeps connector device choice aligned with inference engine runtime.
- `batch_gather_scatter_helper.py` — Made `tl.float8e4nv` conditional: `tl.float8e4nv if hasattr(tl, 'float8e4nv') else tl.float8e4b8` to support non-NVIDIA Triton backends.

## 6. File Change Summary

| File | Action | Description |
|---|---|---|
| `.bazelrc` | Modify | Add `build:musa`, `build:client_with_musa` |
| `WORKSPACE` | Modify | Add `musa_configure` call |
| `3rdparty/gpus/musa_configure.bzl` | **New** | MUSA toolchain auto-detection repository rule |
| `3rdparty/gpus/musa/` | **New dir** | MUSA BUILD templates + config header |
| `sdk/BUILD` | Modify | Add `using_musa` config, MUSA library target |
| `sdk/musa_util.h` | **New** | MUSA error-checking macros |
| `sdk/hf3fs_musa_util.h` | **New** | `Hf3fsMusaUtil` class |
| `sdk/sdk_buffer_check_util.mu` | **New** | MUSA CRC32 kernel (legacy/single-file) |
| `sdk/sdk_buffer_check_util_musa.mu` | **New** | MUSA kernel + stream GetIovsCrc (mcc → .o) |
| `sdk/sdk_buffer_check_util_musa.cc` | **New** | MUSA host-side buffer check (links with .o) |
| `sdk/sdk_buffer_check_util.h` | Modify | `GpuStream_t` type alias, `gpu_stream` field |
| `sdk/sdk_buffer_check_util.cc` | Modify | Use `GpuStream_t` in signatures |
| `sdk/sdk_buffer_check_util.cu` | Modify | Use `GpuStream_t` in signatures |
| `sdk/local_file_sdk.h` | Modify | Add `USING_MUSA` branch |
| `sdk/local_file_sdk.cc` | Modify | Add `USING_MUSA` branches (~10 blocks) |
| `sdk/hf3fs_usrbio_client.h` | Modify | Add `musa_util` field |
| `sdk/hf3fs_sdk.cc` | Modify | Add MUSA util init/cleanup path |
| `transfer_client_impl.h` | Modify | Widen guard to `USING_CUDA || USING_MUSA` |
| `transfer_client_impl.cc` | Modify | Widen guard to `USING_CUDA || USING_MUSA` |
| `test/sdk_buffer_check_util_test.cc` | Modify | Use `gpu_stream` field name |
| `data_transfer.py` | Modify | Device-agnostic stream/event |
| `v1_connector.py` | Modify | Device-agnostic stream/event |
| `batch_gather_scatter_helper.py` | Modify | Conditional float8 dtype |

## 7. How to Build
Only the client component depends on CUDA/MUSA; the rest of `kv_cache_manager` does not. Use client-only configs:

```bash
# Client with CUDA
bazel build //kv_cache_manager/client/... --config=client_with_cuda
# Client with MUSA
bazel build //kv_cache_manager/client/... --config=client_with_musa
```

## 8. Testing

### 8.1 CUDA Regression (No MUSA Hardware Required)

Run the existing test suite with `--config=cuda12` to verify zero regression:

```bash
bazel test //kv_cache_manager/client/src/internal/sdk/test/... --config=cuda12
```

The included `3rdparty/gpus/test_cuda_regression.sh` automates this check and also
verifies that C++ compilation succeeds in non-GPU mode (the `GpuStream_t = void*`
fallback path).

### 8.2 MUSA Validation (Requires MUSA Hardware)

On a Moore Threads GPU machine with `mcc` and MUSA toolkit installed:

```bash
bazel test //kv_cache_manager/client/src/internal/sdk/test/... --config=musa
```

### 8.3 Python Layer Validation (No GPU Required)

```bash
python 3rdparty/gpus/test_musa_python_compat.py
```

This script validates:
- The `_get_device_module()` helper correctly falls back to `torch.cuda`
- The `_device_mod` module has the required `Stream`, `Event`, `stream`,
  `set_device`, `current_stream` attributes
- The float8 dtype mapping in `batch_gather_scatter_helper.py` works regardless
  of whether `tl.float8e4nv` exists

## 9. Commit Structure

This PR is split into multiple commits for reviewer convenience:

1. **`[client] add GpuStream_t abstraction and rename Cell::cuda_stream to gpu_stream`**
   — Pure refactoring of existing CUDA code. Introduces `GpuStream_t` type alias,
   renames the `Cell` struct field. No new functionality. Reviewers can verify this
   commit in isolation builds/passes all existing CUDA tests unchanged.

2. **`[client/build] add MUSA GPU backend support`**
   — All new MUSA files, build system changes, `#elif defined(USING_MUSA)`
   branches, Python device-agnostic helpers, and test scripts.


## 10. Future Work

- Add `build:mooncake_musa` / `build:hf3fs_musa` / `build:tair_mempool_musa`
  once those backends need MUSA support
- Consider a unified `GpuUtil` base class or `std::variant` to replace
  parallel `cuda_util` / `musa_util` fields in `Hf3fsIovHandle`
- Use the same pattern to add AMD ROCm / Intel oneAPI backends

## 11. License and Distribution Notes

- `3rdparty/gpus/musa/BUILD.tpl` uses `licenses(["restricted"])` to mark
  third-party runtime constraints.
- MUSA runtime libraries are linked from local toolkit installation via
  `musa_configure`; this repository does not vendor or re-distribute proprietary
  MUSA driver/runtime binaries.
- Production distribution should ensure target environments install MUSA
  toolkit/driver according to vendor terms before enabling `--config=musa`.
