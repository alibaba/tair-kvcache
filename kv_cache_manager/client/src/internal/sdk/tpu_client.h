#pragma once

#ifdef USING_TPU

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_raw_buffer_extension.h"
#include "kv_cache_manager/client/include/common.h"
#include "kv_cache_manager/common/logger.h"

namespace kv_cache_manager {

/**
 * TpuClient - Manages TPU device access and data transfers using PJRT C API
 *
 * This class provides a clean interface to TPU devices through the PJRT C API.
 * It loads libtpu.so dynamically via dlopen to ensure proper runtime initialization
 * (equivalent to how JAX loads the TPU plugin).
 *
 * Supports:
 * - Client initialization via dlopen + GetPjrtApi()
 * - Host memory registration for DMA (DmaMap/DmaUnmap)
 * - Synchronous H2D/D2H transfers via BufferFromHost() / BufferToHost()
 * - Asynchronous H2D/D2H transfers via BufferFromHostAsync() / BufferToHostAsync()
 * - Event management: WaitEvent() / WaitEvents() / DestroyEvent()
 * - RawBuffer extension: raw alias creation, raw H2D/D2H copies (no concurrency protection)
 *
 * Note: CreateViewOfDeviceBuffer is NOT supported on TPU, so this class uses
 * BufferFromHostBuffer() and ToHostBuffer() for all transfers.
 */
class TpuClient {
public:
    TpuClient() = default;
    ~TpuClient();

    // Non-copyable
    TpuClient(const TpuClient&) = delete;
    TpuClient& operator=(const TpuClient&) = delete;

    /**
     * Initialize the TPU client
     * @return ER_OK on success, ER_TPU_PJRT_INIT_ERROR on failure
     */
    ClientErrorCode Init();

    /**
     * Destroy the client and release all resources
     */
    void Destroy();

    /**
     * Register host memory for DMA access by TPU
     * @param data Host memory pointer
     * @param size Size of memory region
     * @return ER_OK on success, ER_TPU_DMA_MAP_ERROR on failure
     */
    ClientErrorCode DmaMap(void* data, size_t size);

    /**
     * Unregister previously registered host memory
     * @param data Host memory pointer (must match DmaMap call)
     * @return ER_OK on success, ER_TPU_DMA_MAP_ERROR on failure
     */
    ClientErrorCode DmaUnmap(void* data);

    // =====================================================================
    // Synchronous Buffer Transfers (blocking, wait for completion)
    // =====================================================================

    /**
     * Create a device buffer from host data (H2D transfer, synchronous)
     * Creates a new PjRtBuffer on TPU containing a copy of host data.
     * Blocks until the transfer is complete.
     * @param host_src Source host memory pointer
     * @param size Size of data to transfer
     * @param out_buffer Output parameter for the created buffer (caller owns)
     * @return ER_OK on success, ER_TPU_BUFFER_TRANSFER_ERROR on failure
     */
    ClientErrorCode BufferFromHost(const void* host_src, size_t size, PJRT_Buffer*& out_buffer);

    /**
     * Copy device buffer data to host (D2H transfer, synchronous)
     * Blocks until the transfer is complete.
     * @param buffer Source PjRtBuffer on TPU
     * @param host_dst Destination host memory pointer
     * @param size Size of data to transfer
     * @return ER_OK on success, ER_TPU_BUFFER_TRANSFER_ERROR on failure
     */
    ClientErrorCode BufferToHost(PJRT_Buffer* buffer, void* host_dst, size_t size);

    /**
     * Destroy a PjRtBuffer
     * @param buffer Buffer to destroy
     */
    void DestroyBuffer(PJRT_Buffer* buffer);

    // =====================================================================
    // Asynchronous Buffer Transfers (non-blocking, return PJRT_Event*)
    // =====================================================================

    /**
     * Create a device buffer from host data (H2D transfer, asynchronous)
     * Returns immediately with a PJRT_Event* that signals when the host
     * buffer can be safely reused/freed.
     * @param host_src Source host memory pointer
     * @param size Size of data to transfer
     * @param out_buffer Output parameter for the created buffer (caller owns)
     * @param out_event Output parameter for the completion event (caller must destroy)
     * @return ER_OK on success, ER_TPU_BUFFER_TRANSFER_ERROR on failure
     */
    ClientErrorCode BufferFromHostAsync(const void* host_src, size_t size,
                                        PJRT_Buffer*& out_buffer, PJRT_Event*& out_event);

    /**
     * Copy device buffer data to host (D2H transfer, asynchronous)
     * Returns immediately with a PJRT_Event* that signals when the data
     * has been fully copied to the host buffer.
     * @param buffer Source PjRtBuffer on TPU
     * @param host_dst Destination host memory pointer
     * @param size Size of data to transfer
     * @param out_event Output parameter for the completion event (caller must destroy)
     * @return ER_OK on success, ER_TPU_BUFFER_TRANSFER_ERROR on failure
     */
    ClientErrorCode BufferToHostAsync(PJRT_Buffer* buffer, void* host_dst, size_t size,
                                      PJRT_Event*& out_event);

    // =====================================================================
    // Event Management
    // =====================================================================

    /**
     * Block until a single PJRT_Event completes, then destroy it.
     * @param event Event to wait on (will be destroyed after completion)
     * @return ER_OK on success, ER_TPU_EVENT_ERROR if the event signaled an error
     */
    ClientErrorCode WaitEvent(PJRT_Event* event);

    /**
     * Block until all events complete, then destroy them all.
     * @param events Vector of events to wait on (all will be destroyed)
     * @return ER_OK on success, ER_TPU_EVENT_ERROR if any event signaled an error
     */
    ClientErrorCode WaitEvents(std::vector<PJRT_Event*>& events);

    /**
     * Destroy a single PJRT_Event without waiting.
     * @param event Event to destroy (nullptr is safe)
     */
    void DestroyEvent(PJRT_Event* event);

    // =====================================================================
    // RawBuffer Extension (unsafe, no concurrency protection)
    // =====================================================================

    /**
     * Check whether the RawBuffer extension is available on this TPU plugin.
     * @return true if RawBuffer extension is present and usable
     */
    bool HasRawBufferExtension() const;

    /**
     * Create a raw alias of an existing PJRT_Buffer.
     * The alias shares the same device memory but bypasses concurrency protection.
     * @param buffer Source PJRT_Buffer
     * @param out_raw Output parameter for the created PJRT_RawBuffer (caller must destroy)
     * @return ER_OK on success, ER_TPU_RAWBUFFER_ERROR on failure
     */
    ClientErrorCode CreateRawAlias(PJRT_Buffer* buffer, PJRT_RawBuffer*& out_raw);

    /**
     * Raw H2D copy: host → device memory at specified offset (no concurrency protection).
     * @param raw Target PJRT_RawBuffer
     * @param src Source host memory pointer
     * @param offset Device memory offset in bytes
     * @param size Number of bytes to copy
     * @param out_event Output parameter for the completion event (caller must destroy)
     * @return ER_OK on success, ER_TPU_RAWBUFFER_ERROR on failure
     */
    ClientErrorCode RawBufferFromHost(PJRT_RawBuffer* raw, const void* src,
                                      int64_t offset, int64_t size, PJRT_Event*& out_event);

    /**
     * Raw D2H copy: device memory at specified offset → host (no concurrency protection).
     * @param raw Source PJRT_RawBuffer
     * @param dst Destination host memory pointer
     * @param offset Device memory offset in bytes
     * @param size Number of bytes to copy
     * @param out_event Output parameter for the completion event (caller must destroy)
     * @return ER_OK on success, ER_TPU_RAWBUFFER_ERROR on failure
     */
    ClientErrorCode RawBufferToHost(PJRT_RawBuffer* raw, void* dst,
                                    int64_t offset, int64_t size, PJRT_Event*& out_event);

    /**
     * Get the on-device size in bytes of a PJRT_RawBuffer.
     * @param raw The PJRT_RawBuffer to query
     * @param out_size Output parameter for the size in bytes
     * @return ER_OK on success, ER_TPU_RAWBUFFER_ERROR on failure
     */
    ClientErrorCode RawBufferGetDeviceSize(PJRT_RawBuffer* raw, size_t& out_size);

    /**
     * Destroy a PJRT_RawBuffer (does not affect the original PJRT_Buffer).
     * @param raw RawBuffer to destroy (nullptr is safe)
     */
    void DestroyRawBuffer(PJRT_RawBuffer* raw);

    // =====================================================================
    // Utility
    // =====================================================================

    /**
     * Utility: extract error message from a PJRT_Error
     */
    static std::string GetErrorMessage(const PJRT_Api* api, PJRT_Error* error);

private:
    /**
     * Probe PJRT API: print all supported/unsupported function pointers.
     * Called automatically during Init() to inspect runtime capabilities.
     */
    void ProbeApiCapabilities() const;

    /**
     * Query and log platform/device/memory info via read-only PJRT APIs.
     * Called automatically during Init() for diagnostic purposes.
     */
    void LogDeviceInfo(size_t num_addressable_devices) const;

    /**
     * Find the RawBuffer extension in the PJRT extension chain.
     * @return pointer to the extension, or nullptr if not present
     */
    const PJRT_RawBuffer_Extension* FindRawBufferExtension() const;

    void* libtpu_handle_ = nullptr;  // dlopen handle for libtpu.so
    const PJRT_Api* api_ = nullptr;
    PJRT_Client* client_ = nullptr;
    PJRT_Device* device_ = nullptr;
    const PJRT_RawBuffer_Extension* rawbuf_ext_ = nullptr;  // cached RawBuffer extension
};

} // namespace kv_cache_manager

#endif // USING_TPU
