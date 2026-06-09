# TPU Client (TpuClient)

TpuClient is the low-level SDK wrapper in KVCacheManager for accessing
TPU devices. It is built on top of the PJRT C API (Pretty Much Just a
Runtime — the unified device abstraction from OpenXLA) and loads
`libtpu.so` dynamically via `dlopen`, equivalent to how JAX loads the
TPU plugin in Python.

Source: `kv_cache_manager/client/src/internal/sdk/tpu_client.{h,cc}`

## Why dlopen Instead of Static Linking

libtpu.so internally depends on Google's infrastructure (flags, logging,
filesystem, etc.). Linking it directly in a standalone C++ process
triggers a fatal assertion: `CheckInitGoogleIsDone(): InitGoogle() has
not finished yet`. The dlopen approach lets us manually invoke the
internal `RealInitGoogle()` before calling `PJRT_Client_Create`.

See "Initialization Flow" below.

## Core Capabilities

| Capability | Method | Description |
|---|---|---|
| Client init | `Init()` | dlopen libtpu → RealInitGoogle → GetPjrtApi → PJRT_Client_Create → enumerate devices |
| Teardown | `Destroy()` | PJRT_Client_Destroy; shared dlopen handle is not dlclose'd |
| DMA registration | `DmaMap()` / `DmaUnmap()` | Host memory DMA mapping (no-op on current TPU) |
| Sync H2D transfer | `BufferFromHost()` | Host → TPU device, creates a new PJRT_Buffer, blocks until complete |
| Sync D2H transfer | `BufferToHost()` | TPU device → host, reads from an existing PJRT_Buffer, blocks until complete |
| Async H2D transfer | `BufferFromHostAsync()` | Host → TPU device, returns PJRT_Event*, non-blocking |
| Async D2H transfer | `BufferToHostAsync()` | TPU device → host, returns PJRT_Event*, non-blocking |
| Single event wait | `WaitEvent()` | Block until a single PJRT_Event completes, then destroy it |
| Batch event wait | `WaitEvents()` | Block until all PJRT_Events complete, then destroy them |
| Event destroy | `DestroyEvent()` | Manually destroy a single PJRT_Event |
| Buffer destroy | `DestroyBuffer()` | Releases a PJRT_Buffer |
| RawBuffer create | `CreateRawAlias()` | Create a RawBuffer alias from PJRT_Buffer (no concurrency protection) |
| RawBuffer H2D | `RawBufferFromHost()` | Raw H2D copy with offset support, returns Event |
| RawBuffer D2H | `RawBufferToHost()` | Raw D2H copy with offset support, returns Event |
| RawBuffer size query | `RawBufferGetDeviceSize()` | Query on-device physical byte size of a RawBuffer |
| RawBuffer destroy | `DestroyRawBuffer()` | Release a RawBuffer (does not affect original Buffer) |
| Error extraction | `GetErrorMessage()` | Static utility to extract text from PJRT_Error |

Note: `CreateViewOfDeviceBuffer` is **not supported** on TPU. All data
transfers use `BufferFromHostBuffer` / `ToHostBuffer`.

## Initialization Flow

`Init()` executes the following steps:

```
1. dlopen(libtpu.so, RTLD_NOW | RTLD_GLOBAL)
2. Compute RealInitGoogle address = dlopen_base + kRealInitGoogleOffset
3. Call RealInitGoogle("kvcm_tpu", ...) — first time only (static guard)
4. dlsym("GetPjrtApi") → obtain the PJRT_Api function-pointer table
5. PJRT_Client_Create() → create the PJRT client
6. PJRT_Client_AddressableDevices() → enumerate addressable TPU devices
7. LogDeviceInfo() → query and log platform/device/memory information
8. ProbeApiCapabilities() → probe and log all API support status
```

### RealInitGoogle

`RealInitGoogle()` inside libtpu is equivalent to Google's standard
`InitGoogle()`. It initializes flags, logging, filesystem, and other
subsystems. It is **not idempotent** — calling it twice will abort.
TpuClient uses a `static bool google_initialized` to guarantee a
process-wide single invocation.

### Offset

`kRealInitGoogleOffset` is tied to the libtpu version (currently
v0.0.41). Upgrading libtpu requires recomputing this offset via
`nm`/`readelf` on the new `.so`.

## Device Information Query (LogDeviceInfo)

Called automatically during `Init()`. Queries and logs read-only PJRT
API information:

### Platform-Level Info

| API | Example Output |
|---|---|
| `PJRT_Client_PlatformName` | `tpu` |
| `PJRT_Client_PlatformVersion` | `TFRT TPU7x Built on May 13 2026 ...` |
| `PJRT_Client_ProcessIndex` | `0` (always 0 in single-process) |
| `PJRT_Client_Devices` | Total device count (including non-addressable) |
| `PJRT_Client_AddressableMemories` | Client-level addressable memory count |

### Per-Device Info

| API | Example Output |
|---|---|
| `PJRT_Device_IsAddressable` | `true` |
| `PJRT_Device_LocalHardwareId` | `0` (local hardware ID) |
| `PJRT_DeviceDescription_Id` | `0` (globally unique device ID) |
| `PJRT_DeviceDescription_ProcessIndex` | `0` |
| `PJRT_DeviceDescription_Kind` | `TPU7x` |
| `PJRT_DeviceDescription_ToString` | `TpuDevice(id=0, process_index=0, coords=(0,0,0), core_on_chip=0)` |
| `PJRT_DeviceDescription_DebugString` | `TPU_0(process=0,(0,0,0,0))` |

### Per-Memory Info (Default Memory)

| API | Example Output |
|---|---|
| `PJRT_Memory_Id` | `0` |
| `PJRT_Memory_Kind` | `device` |
| `PJRT_Memory_Kind_Id` | Platform-dependent ID |
| `PJRT_Memory_DebugString` | `TpuHbmMemory(id=0, process_index=0, client=tpu)` |
| `PJRT_Device_AddressableMemories` | Per-device addressable memory count (e.g. 3) |
| `PJRT_Device_MemoryStats` | `bytes_in_use`, `bytes_limit`, `peak_bytes`, etc. |

MemoryStats may return `UNIMPLEMENTED`; the log will show `unavailable`
in that case.

## API Capability Probe (ProbeApiCapabilities)

Called automatically during `Init()`. Iterates all 138 function pointers
in the `PJRT_Api` table and classifies each as `[OK]` or `[NULL]`.

Also traverses the PJRT extension chain, logging for each extension:

| Field | Meaning |
|---|---|
| `type` | Extension type number and human-readable name (e.g. `[8] RawBuffer`) |
| `struct_size` | Byte size of the extension struct, used for PJRT ABI versioning |

### RawBuffer Extension

`PJRT_RawBuffer_*` APIs live outside the main `PJRT_Api` struct. They
are provided through the `PJRT_RawBuffer_Extension` struct linked via
the extension chain. ProbeApiCapabilities probes them separately:

| API | Description |
|---|---|
| `PJRT_RawBuffer_CreateRawAliasOfBuffer` | Create a raw alias buffer |
| `PJRT_RawBuffer_Destroy` | Destroy a raw buffer |
| `PJRT_RawBuffer_GetOnDeviceSizeInBytes` | Get on-device byte count |
| `PJRT_RawBuffer_GetMemorySpace` | Get memory space |
| `PJRT_RawBuffer_CopyRawHostToDevice` | Raw H2D copy |
| `PJRT_RawBuffer_CopyRawDeviceToHost` | Raw D2H copy |
| `PJRT_RawBuffer_GetHostPointer` | Get host pointer |

### RawBuffer Known Limitations

#### Host Memory Must Be 32-Byte Aligned

TPU's `PJRT_RawBuffer_CopyRawHostToDevice` / `CopyRawDeviceToHost`
require the host memory pointer to be **32-byte aligned**. Otherwise
the PJRT runtime returns an error in the Event:
`"Host buffer is not aligned to 32 bytes"`.

Standard containers like `std::vector` and `std::string` do **not**
guarantee 32-byte alignment. Callers should use `std::aligned_alloc(32, size)`
or equivalent:

```cpp
size_t aligned_size = (size + 31) & ~static_cast<size_t>(31);
void* host_buf = std::aligned_alloc(32, aligned_size);
// ... use host_buf for RawBuffer copy ...
free(host_buf);
```

#### Physical Byte Layout Differs from Logical Data

TPU stores device memory in tile layout. RawBuffer D2H reads back
**physical bytes**, whose order may differ from the logical data written
via `BufferFromHost`. For example:

```
Logical write: [0, 1, 2, 3, 4, 5, ...]  (row-major)
Physical read: [0, 1, 2, ..., 7, 128, 129, ...]  (tile 8×128 reordered)
```

Therefore RawBuffer D2H results **cannot** be directly `memcmp`'d against
logical data. To verify data correctness, use the logical interface
`BufferToHost()` instead.

> **Future work**: Upper layers (e.g. tpu-raiden) need to compute physical
> offsets according to tile layout and parse/fill RawBuffer data per tile.
> TpuClient does not perform automatic conversion at this level.

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `TPU_LIBRARY_PATH` | Full path to libtpu.so | Virtualenv built-in path |

## Thread Safety

- `Init()` / `Destroy()` are **not** thread-safe; callers must
  synchronize externally.
- The internal `static bool google_initialized` ensures `RealInitGoogle`
  runs once, but concurrent `Init()` calls still have a data race.
- A single `TpuClient` instance should not be used concurrently from
  multiple threads.

## Constraints

- KVCache is reused only within the same `instance_id`; no cross-Instance
  matching.
- TPU does not support `CreateViewOfDeviceBuffer`; all transfers go
  through `BufferFromHostBuffer` / `ToHostBuffer`.
- libtpu version changes require updating `kRealInitGoogleOffset`.
- RawBuffer operations require 32-byte aligned host memory
  (see RawBuffer Known Limitations).
- RawBuffer D2H/H2D transfers physical bytes; under tile layout the
  byte order differs from logical data.

## Testing

```bash
bazelisk test //kv_cache_manager/client/src/internal/sdk/test:TpuClientTest \
    --config=client_with_tpu \
    --config=debug \
    --test_env=TPU_LIBRARY_PATH=<path-to-libtpu.so>
```

Test coverage: InitSucceeds, DoubleInit, DestroyClearsState,
DmaMapUnmap, DmaMapUninitReturnsError, BufferRoundTrip,
BufferFromHostZeroSize, BufferToHostZeroSize,
BufferOpsUninitReturnsError, DestroyBufferNullSafe,
GetErrorMessageNullSafe, LargeBufferRoundTrip,
AsyncBufferFromHostRoundTrip, AsyncBufferToHostRoundTrip,
WaitEventsBatch, DestroyEventNullSafe, WaitEventNullSafe,
AsyncBufferOpsUninitReturnsError, HasRawBufferExtensionAfterInit,
RawBufferRoundTrip, RawBufferOpsUninitReturnsError,
DestroyRawBufferNullSafe, RawBufferBatchAsyncD2H.
