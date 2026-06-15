// TPU client implementation using PJRT C API
// Provides clean interface without CreateViewOfDeviceBuffer (unsupported on TPU)

#ifdef USING_TPU

#include "kv_cache_manager/client/src/internal/sdk/tpu_client.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <vector>

#include "xla/pjrt/c/pjrt_c_api_raw_buffer_extension.h"

namespace kv_cache_manager {

// =====================================================================
// PJRT C API helper macros — reduce boilerplate for Args struct init
// and error handling patterns that repeat throughout this file.
// =====================================================================

// Initialize a PJRT *_Args struct with struct_size and extension_start.
// Usage: PJRT_INIT(PJRT_Client_Create);  → declares `args` of type PJRT_Client_Create_Args
#define PJRT_INIT(type) \
    type##_Args args{}; \
    args.struct_size = type##_Args_STRUCT_SIZE; \
    args.extension_start = nullptr

// Call a PJRT API function and return error_code if it returns a PJRT_Error*.
// Uses api_ (TpuClient member) for error handling.
// Usage: PJRT_CHECK(PJRT_Client_Create(&args), ER_TPU_PJRT_INIT_ERROR);
#define PJRT_CHECK(call, error_code) \
    do { \
        PJRT_Error* _pjrt_err = (call); \
        if (_pjrt_err != nullptr) { \
            std::string _msg = TpuClient::GetErrorMessage(api_, _pjrt_err); \
            KVCM_LOG_ERROR(#call " failed: %s", _msg.c_str()); \
            return error_code; \
        } \
    } while (0)

// Guard: return error_code if api_ (and optionally client_) is null.
#define PJRT_GUARD(error_code) \
    do { if (!api_ || !client_) return error_code; } while (0)

#define PJRT_GUARD_API(error_code) \
    do { if (!api_) return error_code; } while (0)

// =====================================================================
// PyArray → PJRT_Buffer* extraction bridge (runtime registration)
// =====================================================================
namespace {
PyArrayBufferExtractorFn g_py_array_buffer_extractor = nullptr;
}  // namespace

void RegisterPyArrayBufferExtractor(PyArrayBufferExtractorFn fn) {
    g_py_array_buffer_extractor = fn;
}

PyArrayBufferExtractorFn GetPyArrayBufferExtractor() {
    return g_py_array_buffer_extractor;
}

namespace {
// Default libtpu.so search path. Override via TPU_LIBRARY_PATH env var.
// When JAX is installed, libtpu is typically at:
//   <python-site-packages>/libtpu/libtpu.so
constexpr const char* kDefaultLibtpuPath = "libtpu.so";

const char* GetLibtpuPath() {
    const char* env_path = std::getenv("TPU_LIBRARY_PATH");
    return (env_path && env_path[0] != '\0') ? env_path : kDefaultLibtpuPath;
}
} // anonymous namespace

TpuClient::~TpuClient() {
    Destroy();
}

std::string TpuClient::GetErrorMessage(const PJRT_Api* api, PJRT_Error* error) {
    if (!api || !error) return "unknown error";

    std::string msg;
    {
        PJRT_INIT(PJRT_Error_Message);
        args.error = error;
        api->PJRT_Error_Message(&args);
        msg.assign(args.message, args.message_size);
    }
    {
        PJRT_INIT(PJRT_Error_Destroy);
        args.error = error;
        api->PJRT_Error_Destroy(&args);
    }
    return msg;
}

ClientErrorCode TpuClient::Init() {
    // If already initialized, destroy the old client first
    if (client_) {
        Destroy();
    }

    // Load libtpu.so on first call only.
    // PJRT_Plugin_Initialize() is not idempotent — calling it twice will abort.
    // If libtpu is already loaded (e.g. by JAX's TPU plugin), we skip
    // both dlopen and PJRT_Plugin_Initialize, and just reuse the existing PJRT_Api.
    static bool plugin_initialized = false;
    static void* shared_libtpu_handle = nullptr;
    static bool needs_plugin_init = false;

    if (!plugin_initialized) {
        const char* lib_path = GetLibtpuPath();

        // Check if libtpu.so is already loaded in this process
        // (e.g. JAX imported before us). RTLD_NOLOAD returns a handle
        // without loading the library if it's not already loaded.
        void* existing_handle = dlopen(lib_path, RTLD_NOW | RTLD_GLOBAL | RTLD_NOLOAD);
        if (existing_handle) {
            // libtpu already loaded by another component (e.g. JAX).
            // PJRT_Plugin_Initialize has already run — skip to avoid abort.
            shared_libtpu_handle = existing_handle;
            needs_plugin_init = false;
            KVCM_LOG_INFO("libtpu already loaded (by JAX or another component), reusing handle");
        } else {
            // First loader in this process — load the library
            shared_libtpu_handle = dlopen(lib_path, RTLD_NOW | RTLD_GLOBAL);
            if (!shared_libtpu_handle) {
                KVCM_LOG_ERROR("dlopen(%s) failed: %s", lib_path, dlerror());
                return ER_TPU_PJRT_INIT_ERROR;
            }
            KVCM_LOG_INFO("libtpu loaded from: %s", lib_path);
            needs_plugin_init = true;
        }
        plugin_initialized = true;
    }

    libtpu_handle_ = shared_libtpu_handle;

    // Get the PJRT API function pointer table
    using GetPjrtApiFunc = const PJRT_Api* (*)();
    auto get_pjrt_api = reinterpret_cast<GetPjrtApiFunc>(dlsym(libtpu_handle_, "GetPjrtApi"));
    if (!get_pjrt_api) {
        KVCM_LOG_ERROR("dlsym(GetPjrtApi) failed: %s", dlerror());
        return ER_TPU_PJRT_INIT_ERROR;
    }

    api_ = get_pjrt_api();
    if (!api_) {
        KVCM_LOG_ERROR("GetPjrtApi returned nullptr");
        return ER_TPU_PJRT_INIT_ERROR;
    }

    // Call PJRT_Plugin_Initialize if we are the first loader.
    // Skip if JAX (or another component) already initialized the plugin.
    if (needs_plugin_init) {
        PJRT_INIT(PJRT_Plugin_Initialize);
        PJRT_CHECK(api_->PJRT_Plugin_Initialize(&args), ER_TPU_PJRT_INIT_ERROR);
        KVCM_LOG_INFO("PJRT_Plugin_Initialize succeeded");
        needs_plugin_init = false;
    }

    // Create PJRT client
    {
        PJRT_INIT(PJRT_Client_Create);
        args.create_options = nullptr;
        args.num_options = 0;
        args.kv_get_callback = nullptr;
        args.kv_get_user_arg = nullptr;
        args.kv_put_callback = nullptr;
        args.kv_put_user_arg = nullptr;
        PJRT_CHECK(api_->PJRT_Client_Create(&args), ER_TPU_PJRT_INIT_ERROR);
        client_ = args.client;
    }

    // Get addressable devices
    size_t num_devices = 0;
    {
        PJRT_INIT(PJRT_Client_AddressableDevices);
        args.client = client_;
        api_->PJRT_Client_AddressableDevices(&args);

        if (args.num_addressable_devices == 0) {
            KVCM_LOG_ERROR("No addressable TPU devices found");
            Destroy();
            return ER_TPU_PJRT_INIT_ERROR;
        }

        device_ = args.addressable_devices[0];
        num_devices = args.num_addressable_devices;
        KVCM_LOG_INFO("TPU PJRT client initialized with %zu devices", num_devices);
    }

    LogDeviceInfo(num_devices);

    // Probe and log all API capabilities
    ProbeApiCapabilities();

    // Cache RawBuffer extension (may be nullptr if plugin doesn't support it)
    rawbuf_ext_ = FindRawBufferExtension();
    if (rawbuf_ext_) {
        KVCM_LOG_INFO("RawBuffer extension cached during Init()");
    } else {
        KVCM_LOG_INFO("RawBuffer extension not available");
    }

    return ER_OK;
}

void TpuClient::Destroy() {
    if (client_ && api_) {
        PJRT_INIT(PJRT_Client_Destroy);
        args.client = client_;
        api_->PJRT_Client_Destroy(&args);
    }
    client_ = nullptr;
    device_ = nullptr;
    api_ = nullptr;
    rawbuf_ext_ = nullptr;
    libtpu_handle_ = nullptr;
}

ClientErrorCode TpuClient::DmaMap(void* data, size_t size) {
    PJRT_GUARD(ER_TPU_DMA_MAP_ERROR);
    return ER_OK;
}

ClientErrorCode TpuClient::DmaUnmap(void* data) {
    PJRT_GUARD(ER_TPU_DMA_MAP_ERROR);
    return ER_OK;
}

ClientErrorCode TpuClient::BufferFromHost(const void* host_src, size_t size, PJRT_Buffer*& out_buffer) {
    PJRT_Event* event = nullptr;
    auto ec = BufferFromHostAsync(host_src, size, out_buffer, event);
    if (ec != ER_OK) return ec;
    return WaitEvent(event);
}

ClientErrorCode TpuClient::BufferToHost(PJRT_Buffer* buffer, void* host_dst, size_t size) {
    PJRT_Event* event = nullptr;
    auto ec = BufferToHostAsync(buffer, host_dst, size, event);
    if (ec != ER_OK) return ec;
    return WaitEvent(event);
}

void TpuClient::DestroyBuffer(PJRT_Buffer* buffer) {
    if (!api_ || !buffer) return;
    PJRT_INIT(PJRT_Buffer_Destroy);
    args.buffer = buffer;
    api_->PJRT_Buffer_Destroy(&args);
}

// =========================================================================
// Asynchronous Buffer Transfers
// =========================================================================

ClientErrorCode TpuClient::BufferFromHostAsync(const void* host_src, size_t size,
                                                PJRT_Buffer*& out_buffer,
                                                PJRT_Event*& out_event) {
    if (!api_ || !client_ || !device_) return ER_TPU_BUFFER_TRANSFER_ERROR;
    if (size == 0) return ER_INVALID_PARAMS;

    int64_t dim = static_cast<int64_t>(size);
    PJRT_INIT(PJRT_Client_BufferFromHostBuffer);
    args.client = client_;
    args.data = host_src;
    args.type = PJRT_Buffer_Type_U8;
    args.dims = &dim;
    args.num_dims = 1;
    args.byte_strides = nullptr;
    args.num_byte_strides = 0;
    args.host_buffer_semantics = PJRT_HostBufferSemantics_kImmutableOnlyDuringCall;
    args.device = device_;
    args.memory = nullptr;
    args.device_layout = nullptr;
    PJRT_CHECK(api_->PJRT_Client_BufferFromHostBuffer(&args), ER_TPU_BUFFER_TRANSFER_ERROR);

    out_buffer = args.buffer;
    out_event = args.done_with_host_buffer;
    return ER_OK;
}

ClientErrorCode TpuClient::BufferToHostAsync(PJRT_Buffer* buffer, void* host_dst,
                                              size_t size, PJRT_Event*& out_event) {
    if (!api_ || !buffer) return ER_TPU_BUFFER_TRANSFER_ERROR;
    if (size == 0) return ER_INVALID_PARAMS;

    PJRT_INIT(PJRT_Buffer_ToHostBuffer);
    args.src = buffer;
    args.host_layout = nullptr;
    args.dst = host_dst;
    args.dst_size = size;
    PJRT_CHECK(api_->PJRT_Buffer_ToHostBuffer(&args), ER_TPU_BUFFER_TRANSFER_ERROR);

    out_event = args.event;
    return ER_OK;
}

// =========================================================================
// Event Management
// =========================================================================

ClientErrorCode TpuClient::WaitEvent(PJRT_Event* event) {
    if (!api_ || !event) return ER_OK;

    PJRT_INIT(PJRT_Event_Await);
    args.event = event;
    PJRT_CHECK(api_->PJRT_Event_Await(&args), ER_TPU_EVENT_ERROR);

    DestroyEvent(event);
    return ER_OK;
}

ClientErrorCode TpuClient::WaitEvents(std::vector<PJRT_Event*>& events) {
    ClientErrorCode result = ER_OK;
    for (auto* event : events) {
        auto ec = WaitEvent(event);
        if (ec != ER_OK) {
            result = ec;
            // Continue waiting/destroying remaining events to avoid leaks
        }
    }
    events.clear();
    return result;
}

void TpuClient::DestroyEvent(PJRT_Event* event) {
    if (!api_ || !event) return;
    PJRT_INIT(PJRT_Event_Destroy);
    args.event = event;
    api_->PJRT_Event_Destroy(&args);
}

// =========================================================================
// RawBuffer Extension
// =========================================================================

const PJRT_RawBuffer_Extension* TpuClient::FindRawBufferExtension() const {
    if (!api_) return nullptr;

    const PJRT_Extension_Base* ext = api_->extension_start;
    while (ext != nullptr) {
        if (ext->type == PJRT_Extension_Type_RawBuffer) {
            return reinterpret_cast<const PJRT_RawBuffer_Extension*>(ext);
        }
        ext = ext->next;
    }
    return nullptr;
}

bool TpuClient::HasRawBufferExtension() const {
    return rawbuf_ext_ != nullptr;
}

ClientErrorCode TpuClient::CreateRawAlias(PJRT_Buffer* buffer, PJRT_RawBuffer*& out_raw) {
    if (!api_ || !rawbuf_ext_ || !rawbuf_ext_->PJRT_RawBuffer_CreateRawAliasOfBuffer)
        return ER_TPU_RAWBUFFER_ERROR;
    if (!buffer) return ER_INVALID_PARAMS;

    PJRT_RawBuffer_CreateRawAliasOfBuffer_Args args{};
    args.struct_size = PJRT_RawBuffer_CreateRawAliasOfBuffer_Args_STRUCT_SIZE;
    args.extension_start = nullptr;
    args.buffer = buffer;
    PJRT_CHECK(rawbuf_ext_->PJRT_RawBuffer_CreateRawAliasOfBuffer(&args), ER_TPU_RAWBUFFER_ERROR);

    out_raw = args.raw_buffer;
    return ER_OK;
}

ClientErrorCode TpuClient::RawBufferFromHost(PJRT_RawBuffer* raw, const void* src,
                                              int64_t offset, int64_t size,
                                              PJRT_Event*& out_event) {
    if (!api_ || !rawbuf_ext_ || !rawbuf_ext_->PJRT_RawBuffer_CopyRawHostToDevice)
        return ER_TPU_RAWBUFFER_ERROR;
    if (!raw || !src || size <= 0) return ER_INVALID_PARAMS;

    PJRT_RawBuffer_CopyRawHostToDevice_Args args{};
    args.struct_size = PJRT_RawBuffer_CopyRawHostToDevice_Args_STRUCT_SIZE;
    args.extension_start = nullptr;
    args.buffer = raw;
    args.src = src;
    args.offset = offset;
    args.transfer_size = size;
    PJRT_CHECK(rawbuf_ext_->PJRT_RawBuffer_CopyRawHostToDevice(&args), ER_TPU_RAWBUFFER_ERROR);

    out_event = args.event;
    return ER_OK;
}

ClientErrorCode TpuClient::RawBufferToHost(PJRT_RawBuffer* raw, void* dst,
                                            int64_t offset, int64_t size,
                                            PJRT_Event*& out_event) {
    if (!api_ || !rawbuf_ext_ || !rawbuf_ext_->PJRT_RawBuffer_CopyRawDeviceToHost)
        return ER_TPU_RAWBUFFER_ERROR;
    if (!raw || !dst || size <= 0) return ER_INVALID_PARAMS;

    PJRT_RawBuffer_CopyRawDeviceToHost_Args args{};
    args.struct_size = PJRT_RawBuffer_CopyRawDeviceToHost_Args_STRUCT_SIZE;
    args.extension_start = nullptr;
    args.buffer = raw;
    args.dst = dst;
    args.offset = offset;
    args.transfer_size = size;
    PJRT_CHECK(rawbuf_ext_->PJRT_RawBuffer_CopyRawDeviceToHost(&args), ER_TPU_RAWBUFFER_ERROR);

    out_event = args.event;
    return ER_OK;
}

ClientErrorCode TpuClient::RawBufferGetDeviceSize(PJRT_RawBuffer* raw, size_t& out_size) {
    if (!api_ || !rawbuf_ext_ || !rawbuf_ext_->PJRT_RawBuffer_GetOnDeviceSizeInBytes)
        return ER_TPU_RAWBUFFER_ERROR;
    if (!raw) return ER_INVALID_PARAMS;

    PJRT_RawBuffer_GetOnDeviceSizeInBytes_Args args{};
    args.struct_size = PJRT_RawBuffer_GetOnDeviceSizeInBytes_Args_STRUCT_SIZE;
    args.extension_start = nullptr;
    args.buffer = raw;
    PJRT_CHECK(rawbuf_ext_->PJRT_RawBuffer_GetOnDeviceSizeInBytes(&args), ER_TPU_RAWBUFFER_ERROR);

    out_size = args.on_device_size_in_bytes;
    return ER_OK;
}

void TpuClient::DestroyRawBuffer(PJRT_RawBuffer* raw) {
    if (!api_ || !rawbuf_ext_ || !raw || !rawbuf_ext_->PJRT_RawBuffer_Destroy) return;
    PJRT_RawBuffer_Destroy_Args args{};
    args.struct_size = PJRT_RawBuffer_Destroy_Args_STRUCT_SIZE;
    args.extension_start = nullptr;
    args.buffer = raw;
    rawbuf_ext_->PJRT_RawBuffer_Destroy(&args);
}

// Helper macro: check if a PJRT_Api function pointer is non-null and log it
#define _PJRT_PROBE_FIELD(api, field) do { \
    if (api->field != nullptr) { \
        supported.push_back(#field); \
    } else { \
        unsupported.push_back(#field); \
    } \
} while (0)

void TpuClient::ProbeApiCapabilities() const {
    if (!api_) {
        KVCM_LOG_ERROR("ProbeApiCapabilities: api_ is null, call Init() first");
        return;
    }

    // Print API version (embedded struct, not a function call)
    KVCM_LOG_INFO("===== PJRT API Capability Probe =====");
    KVCM_LOG_INFO("PJRT API version: %d.%d",
                   api_->pjrt_api_version.major_version,
                   api_->pjrt_api_version.minor_version);

    std::vector<const char*> supported;
    std::vector<const char*> unsupported;

    // --- Error ---
    _PJRT_PROBE_FIELD(api_, PJRT_Error_Destroy);
    _PJRT_PROBE_FIELD(api_, PJRT_Error_Message);
    _PJRT_PROBE_FIELD(api_, PJRT_Error_GetCode);
    _PJRT_PROBE_FIELD(api_, PJRT_Error_ForEachPayload);

    // --- Plugin ---
    _PJRT_PROBE_FIELD(api_, PJRT_Plugin_Initialize);
    _PJRT_PROBE_FIELD(api_, PJRT_Plugin_Attributes);

    // --- Event ---
    _PJRT_PROBE_FIELD(api_, PJRT_Event_Create);
    _PJRT_PROBE_FIELD(api_, PJRT_Event_Set);
    _PJRT_PROBE_FIELD(api_, PJRT_Event_Destroy);
    _PJRT_PROBE_FIELD(api_, PJRT_Event_IsReady);
    _PJRT_PROBE_FIELD(api_, PJRT_Event_Error);
    _PJRT_PROBE_FIELD(api_, PJRT_Event_Await);
    _PJRT_PROBE_FIELD(api_, PJRT_Event_OnReady);

    // --- Client ---
    _PJRT_PROBE_FIELD(api_, PJRT_Client_Create);
    _PJRT_PROBE_FIELD(api_, PJRT_Client_Destroy);
    _PJRT_PROBE_FIELD(api_, PJRT_Client_PlatformName);
    _PJRT_PROBE_FIELD(api_, PJRT_Client_ProcessIndex);
    _PJRT_PROBE_FIELD(api_, PJRT_Client_PlatformVersion);
    _PJRT_PROBE_FIELD(api_, PJRT_Client_Devices);
    _PJRT_PROBE_FIELD(api_, PJRT_Client_AddressableDevices);
    _PJRT_PROBE_FIELD(api_, PJRT_Client_LookupDevice);
    _PJRT_PROBE_FIELD(api_, PJRT_Client_LookupAddressableDevice);
    _PJRT_PROBE_FIELD(api_, PJRT_Client_AddressableMemories);
    _PJRT_PROBE_FIELD(api_, PJRT_Client_Compile);
    _PJRT_PROBE_FIELD(api_, PJRT_Client_DefaultDeviceAssignment);
    _PJRT_PROBE_FIELD(api_, PJRT_Client_BufferFromHostBuffer);
    _PJRT_PROBE_FIELD(api_, PJRT_Client_CreateViewOfDeviceBuffer);
    _PJRT_PROBE_FIELD(api_, PJRT_Client_TopologyDescription);
    _PJRT_PROBE_FIELD(api_, PJRT_Client_CreateUninitializedBuffer);
    _PJRT_PROBE_FIELD(api_, PJRT_Client_UpdateGlobalProcessInfo);
    _PJRT_PROBE_FIELD(api_, PJRT_Client_CreateAliasBuffer);
    _PJRT_PROBE_FIELD(api_, PJRT_Client_FulfillAliasBuffer);
    _PJRT_PROBE_FIELD(api_, PJRT_Client_CreateErrorBuffer);
    _PJRT_PROBE_FIELD(api_, PJRT_Client_CreateBuffersForAsyncHostToDevice);
    _PJRT_PROBE_FIELD(api_, PJRT_Client_DmaMap);
    _PJRT_PROBE_FIELD(api_, PJRT_Client_DmaUnmap);
    _PJRT_PROBE_FIELD(api_, PJRT_Client_Load);

    // --- DeviceDescription ---
    _PJRT_PROBE_FIELD(api_, PJRT_DeviceDescription_Id);
    _PJRT_PROBE_FIELD(api_, PJRT_DeviceDescription_ProcessIndex);
    _PJRT_PROBE_FIELD(api_, PJRT_DeviceDescription_Attributes);
    _PJRT_PROBE_FIELD(api_, PJRT_DeviceDescription_Kind);
    _PJRT_PROBE_FIELD(api_, PJRT_DeviceDescription_DebugString);
    _PJRT_PROBE_FIELD(api_, PJRT_DeviceDescription_ToString);

    // --- Device ---
    _PJRT_PROBE_FIELD(api_, PJRT_Device_GetDescription);
    _PJRT_PROBE_FIELD(api_, PJRT_Device_IsAddressable);
    _PJRT_PROBE_FIELD(api_, PJRT_Device_LocalHardwareId);
    _PJRT_PROBE_FIELD(api_, PJRT_Device_AddressableMemories);
    _PJRT_PROBE_FIELD(api_, PJRT_Device_DefaultMemory);
    _PJRT_PROBE_FIELD(api_, PJRT_Device_MemoryStats);
    _PJRT_PROBE_FIELD(api_, PJRT_Device_PoisonExecution);
    _PJRT_PROBE_FIELD(api_, PJRT_Device_CreateAsyncTrackingEvent);
    _PJRT_PROBE_FIELD(api_, PJRT_Device_GetAttributes);
    _PJRT_PROBE_FIELD(api_, PJRT_Device_ClearMemoryStats);

    // --- Memory ---
    _PJRT_PROBE_FIELD(api_, PJRT_Memory_Id);
    _PJRT_PROBE_FIELD(api_, PJRT_Memory_Kind);
    _PJRT_PROBE_FIELD(api_, PJRT_Memory_DebugString);
    _PJRT_PROBE_FIELD(api_, PJRT_Memory_ToString);
    _PJRT_PROBE_FIELD(api_, PJRT_Memory_AddressableByDevices);
    _PJRT_PROBE_FIELD(api_, PJRT_Memory_Kind_Id);

    // --- Buffer ---
    _PJRT_PROBE_FIELD(api_, PJRT_Buffer_Destroy);
    _PJRT_PROBE_FIELD(api_, PJRT_Buffer_ElementType);
    _PJRT_PROBE_FIELD(api_, PJRT_Buffer_Dimensions);
    _PJRT_PROBE_FIELD(api_, PJRT_Buffer_UnpaddedDimensions);
    _PJRT_PROBE_FIELD(api_, PJRT_Buffer_DynamicDimensionIndices);
    _PJRT_PROBE_FIELD(api_, PJRT_Buffer_GetMemoryLayout);
    _PJRT_PROBE_FIELD(api_, PJRT_Buffer_OnDeviceSizeInBytes);
    _PJRT_PROBE_FIELD(api_, PJRT_Buffer_Device);
    _PJRT_PROBE_FIELD(api_, PJRT_Buffer_Memory);
    _PJRT_PROBE_FIELD(api_, PJRT_Buffer_Delete);
    _PJRT_PROBE_FIELD(api_, PJRT_Buffer_IsDeleted);
    _PJRT_PROBE_FIELD(api_, PJRT_Buffer_CopyToDevice);
    _PJRT_PROBE_FIELD(api_, PJRT_Buffer_CopyToMemory);
    _PJRT_PROBE_FIELD(api_, PJRT_Buffer_ToHostBuffer);
    _PJRT_PROBE_FIELD(api_, PJRT_Buffer_IsOnCpu);
    _PJRT_PROBE_FIELD(api_, PJRT_Buffer_ReadyEvent);
    _PJRT_PROBE_FIELD(api_, PJRT_Buffer_UnsafePointer);
    _PJRT_PROBE_FIELD(api_, PJRT_Buffer_IncreaseExternalReferenceCount);
    _PJRT_PROBE_FIELD(api_, PJRT_Buffer_DecreaseExternalReferenceCount);
    _PJRT_PROBE_FIELD(api_, PJRT_Buffer_OpaqueDeviceMemoryDataPointer);
    _PJRT_PROBE_FIELD(api_, PJRT_Buffer_CopyRawToHost);
    _PJRT_PROBE_FIELD(api_, PJRT_Buffer_CopyRawToHostFuture);
    _PJRT_PROBE_FIELD(api_, PJRT_Buffer_DonateWithControlDependency);
    _PJRT_PROBE_FIELD(api_, PJRT_Buffer_Bitcast);

    // --- Executable ---
    _PJRT_PROBE_FIELD(api_, PJRT_Executable_Destroy);
    _PJRT_PROBE_FIELD(api_, PJRT_Executable_Name);
    _PJRT_PROBE_FIELD(api_, PJRT_Executable_NumReplicas);
    _PJRT_PROBE_FIELD(api_, PJRT_Executable_NumPartitions);
    _PJRT_PROBE_FIELD(api_, PJRT_Executable_NumOutputs);
    _PJRT_PROBE_FIELD(api_, PJRT_Executable_SizeOfGeneratedCodeInBytes);
    _PJRT_PROBE_FIELD(api_, PJRT_Executable_GetCostAnalysis);
    _PJRT_PROBE_FIELD(api_, PJRT_Executable_OutputMemoryKinds);
    _PJRT_PROBE_FIELD(api_, PJRT_Executable_OptimizedProgram);
    _PJRT_PROBE_FIELD(api_, PJRT_Executable_Serialize);
    _PJRT_PROBE_FIELD(api_, PJRT_Executable_OutputElementTypes);
    _PJRT_PROBE_FIELD(api_, PJRT_Executable_OutputDimensions);
    _PJRT_PROBE_FIELD(api_, PJRT_Executable_Fingerprint);
    _PJRT_PROBE_FIELD(api_, PJRT_Executable_GetCompiledMemoryStats);
    _PJRT_PROBE_FIELD(api_, PJRT_Executable_GetCompileOptions);
    _PJRT_PROBE_FIELD(api_, PJRT_Executable_ParameterMemoryKinds);

    // --- LoadedExecutable ---
    _PJRT_PROBE_FIELD(api_, PJRT_LoadedExecutable_Destroy);
    _PJRT_PROBE_FIELD(api_, PJRT_LoadedExecutable_GetExecutable);
    _PJRT_PROBE_FIELD(api_, PJRT_LoadedExecutable_AddressableDevices);
    _PJRT_PROBE_FIELD(api_, PJRT_LoadedExecutable_Delete);
    _PJRT_PROBE_FIELD(api_, PJRT_LoadedExecutable_IsDeleted);
    _PJRT_PROBE_FIELD(api_, PJRT_LoadedExecutable_Execute);
    _PJRT_PROBE_FIELD(api_, PJRT_Executable_DeserializeAndLoad);
    _PJRT_PROBE_FIELD(api_, PJRT_LoadedExecutable_Fingerprint);
    _PJRT_PROBE_FIELD(api_, PJRT_LoadedExecutable_GetDeviceAssignment);
    _PJRT_PROBE_FIELD(api_, PJRT_LoadedExecutable_AddressableDeviceLogicalIds);

    // --- CopyToDeviceStream ---
    _PJRT_PROBE_FIELD(api_, PJRT_CopyToDeviceStream_Destroy);
    _PJRT_PROBE_FIELD(api_, PJRT_CopyToDeviceStream_AddChunk);
    _PJRT_PROBE_FIELD(api_, PJRT_CopyToDeviceStream_TotalBytes);
    _PJRT_PROBE_FIELD(api_, PJRT_CopyToDeviceStream_GranuleSize);
    _PJRT_PROBE_FIELD(api_, PJRT_CopyToDeviceStream_CurrentBytes);

    // --- AsyncHostToDeviceTransferManager ---
    _PJRT_PROBE_FIELD(api_, PJRT_AsyncHostToDeviceTransferManager_Destroy);
    _PJRT_PROBE_FIELD(api_, PJRT_AsyncHostToDeviceTransferManager_TransferData);
    _PJRT_PROBE_FIELD(api_, PJRT_AsyncHostToDeviceTransferManager_TransferLiteral);
    _PJRT_PROBE_FIELD(api_, PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer);
    _PJRT_PROBE_FIELD(api_, PJRT_AsyncHostToDeviceTransferManager_Device);
    _PJRT_PROBE_FIELD(api_, PJRT_AsyncHostToDeviceTransferManager_BufferCount);
    _PJRT_PROBE_FIELD(api_, PJRT_AsyncHostToDeviceTransferManager_BufferSize);
    _PJRT_PROBE_FIELD(api_, PJRT_AsyncHostToDeviceTransferManager_SetBufferError);
    _PJRT_PROBE_FIELD(api_, PJRT_AsyncHostToDeviceTransferManager_AddMetadata);

    // --- TopologyDescription ---
    _PJRT_PROBE_FIELD(api_, PJRT_TopologyDescription_Create);
    _PJRT_PROBE_FIELD(api_, PJRT_TopologyDescription_Destroy);
    _PJRT_PROBE_FIELD(api_, PJRT_TopologyDescription_PlatformName);
    _PJRT_PROBE_FIELD(api_, PJRT_TopologyDescription_PlatformVersion);
    _PJRT_PROBE_FIELD(api_, PJRT_TopologyDescription_GetDeviceDescriptions);
    _PJRT_PROBE_FIELD(api_, PJRT_TopologyDescription_Serialize);
    _PJRT_PROBE_FIELD(api_, PJRT_TopologyDescription_Deserialize);
    _PJRT_PROBE_FIELD(api_, PJRT_TopologyDescription_Attributes);
    _PJRT_PROBE_FIELD(api_, PJRT_TopologyDescription_Fingerprint);
    _PJRT_PROBE_FIELD(api_, PJRT_TopologyDescription_MakeCanonicalShapeForMemorySpace);
    _PJRT_PROBE_FIELD(api_, PJRT_TopologyDescription_GetMemorySpaceKindIds);

    // --- Misc ---
    _PJRT_PROBE_FIELD(api_, PJRT_Compile);
    _PJRT_PROBE_FIELD(api_, PJRT_ExecuteContext_Create);
    _PJRT_PROBE_FIELD(api_, PJRT_ExecuteContext_Destroy);
    _PJRT_PROBE_FIELD(api_, PJRT_AsyncTrackingEvent_Destroy);

    // Print results
    KVCM_LOG_INFO("Supported APIs (%zu/%zu):", supported.size(),
                   supported.size() + unsupported.size());
    for (auto* name : supported) {
        KVCM_LOG_INFO("  [OK]   %s", name);
    }

    KVCM_LOG_INFO("Unsupported APIs (%zu/%zu):", unsupported.size(),
                   supported.size() + unsupported.size());
    for (auto* name : unsupported) {
        KVCM_LOG_INFO("  [NULL] %s", name);
    }

    // Enumerate extensions with human-readable names
    KVCM_LOG_INFO("Extension chain (struct_size = ABI versioning footprint of each extension struct):");
    const PJRT_Extension_Base* ext = api_->extension_start;
    int ext_count = 0;
    const PJRT_RawBuffer_Extension* rawbuf_ext = nullptr;
    while (ext != nullptr) {
        const char* type_name = "Unknown";
        switch (ext->type) {
            case PJRT_Extension_Type_Gpu_Custom_Call:    type_name = "Gpu_Custom_Call"; break;
            case PJRT_Extension_Type_Profiler:           type_name = "Profiler"; break;
            case PJRT_Extension_Type_Custom_Partitioner: type_name = "Custom_Partitioner"; break;
            case PJRT_Extension_Type_Stream:             type_name = "Stream"; break;
            case PJRT_Extension_Type_Layouts:            type_name = "Layouts"; break;
            case PJRT_Extension_Type_FFI:                type_name = "FFI"; break;
            case PJRT_Extension_Type_MemoryDescriptions: type_name = "MemoryDescriptions"; break;
            case PJRT_Extension_Type_Triton:             type_name = "Triton"; break;
            case PJRT_Extension_Type_RawBuffer:          type_name = "RawBuffer"; break;
            case PJRT_Extension_Type_PhaseCompile:       type_name = "PhaseCompile"; break;
            case PJRT_Extension_Type_Example:            type_name = "Example"; break;
            case PJRT_Extension_Type_Unknown:            type_name = "Unknown"; break;
            case PJRT_Extension_Type_CrossHostTransfers: type_name = "CrossHostTransfers"; break;
            case PJRT_Extension_Type_ExecutableMetadata: type_name = "ExecutableMetadata"; break;
            case PJRT_Extension_Type_Callback:           type_name = "Callback"; break;
            case PJRT_Extension_Type_HostAllocator:      type_name = "HostAllocator"; break;
            case PJRT_Extension_Type_TpuTopology:        type_name = "TpuTopology"; break;
            case PJRT_Extension_Type_TpuExecutable:      type_name = "TpuExecutable"; break;
            case PJRT_Extension_Type_Megascale:          type_name = "Megascale"; break;
            case PJRT_Extension_Type_Shardings:          type_name = "Shardings"; break;
            case PJRT_Extension_Type_AbiVersion:         type_name = "AbiVersion"; break;
            case PJRT_Extension_Type_Collectives:        type_name = "Collectives"; break;
            case PJRT_Extension_Type_MultiSlice:         type_name = "MultiSlice"; break;
            case PJRT_Extension_Type_HostMemoryAllocator:type_name = "HostMemoryAllocator"; break;
            case PJRT_Extension_Type_XlaTransform:       type_name = "XlaTransform"; break;
        }
        KVCM_LOG_INFO("  [%d] %-22s (struct_size=%zu bytes)",
                       static_cast<int>(ext->type), type_name, ext->struct_size);
        if (ext->type == PJRT_Extension_Type_RawBuffer) {
            rawbuf_ext = reinterpret_cast<const PJRT_RawBuffer_Extension*>(ext);
        }
        ext = ext->next;
        ext_count++;
    }
    if (ext_count == 0) {
        KVCM_LOG_INFO("  (no extensions)");
    }

    // Probe RawBuffer extension APIs (lives outside main PJRT_Api struct)
    if (rawbuf_ext) {
        std::vector<const char*> rb_supported;
        std::vector<const char*> rb_unsupported;
        #define _PJRT_PROBE_RB_FIELD(field) do { \
            if (rawbuf_ext->field != nullptr) { \
                rb_supported.push_back(#field); \
            } else { \
                rb_unsupported.push_back(#field); \
            } \
        } while (0)
        _PJRT_PROBE_RB_FIELD(PJRT_RawBuffer_CreateRawAliasOfBuffer);
        _PJRT_PROBE_RB_FIELD(PJRT_RawBuffer_Destroy);
        _PJRT_PROBE_RB_FIELD(PJRT_RawBuffer_GetOnDeviceSizeInBytes);
        _PJRT_PROBE_RB_FIELD(PJRT_RawBuffer_GetMemorySpace);
        _PJRT_PROBE_RB_FIELD(PJRT_RawBuffer_CopyRawHostToDevice);
        _PJRT_PROBE_RB_FIELD(PJRT_RawBuffer_CopyRawDeviceToHost);
        _PJRT_PROBE_RB_FIELD(PJRT_RawBuffer_GetHostPointer);
        #undef _PJRT_PROBE_RB_FIELD
        KVCM_LOG_INFO("RawBuffer Extension APIs (%zu/%zu supported):",
                       rb_supported.size(),
                       rb_supported.size() + rb_unsupported.size());
        for (auto* name : rb_supported) {
            KVCM_LOG_INFO("  [OK]   %s", name);
        }
        for (auto* name : rb_unsupported) {
            KVCM_LOG_INFO("  [NULL] %s", name);
        }
    } else {
        KVCM_LOG_INFO("RawBuffer Extension: NOT PRESENT (PJRT_RawBuffer_* APIs unavailable)");
    }

    KVCM_LOG_INFO("===== End PJRT API Probe =====");
}

#undef _PJRT_PROBE_FIELD

void TpuClient::LogDeviceInfo(size_t num_addressable_devices) const {
    PJRT_GUARD((void)0);

    KVCM_LOG_INFO("===== TPU Platform & Device Info =====");

    // --- Client-level info ---
    {
        PJRT_INIT(PJRT_Client_PlatformName);
        args.client = client_;
        if (api_->PJRT_Client_PlatformName(&args) == nullptr)
            KVCM_LOG_INFO("Platform name   : %.*s",
                           static_cast<int>(args.platform_name_size), args.platform_name);
    }
    {
        PJRT_INIT(PJRT_Client_PlatformVersion);
        args.client = client_;
        if (api_->PJRT_Client_PlatformVersion(&args) == nullptr)
            KVCM_LOG_INFO("Platform version: %.*s",
                           static_cast<int>(args.platform_version_size), args.platform_version);
    }
    {
        PJRT_INIT(PJRT_Client_ProcessIndex);
        args.client = client_;
        if (api_->PJRT_Client_ProcessIndex(&args) == nullptr)
            KVCM_LOG_INFO("Process index   : %d", args.process_index);
    }
    {
        PJRT_INIT(PJRT_Client_Devices);
        args.client = client_;
        if (api_->PJRT_Client_Devices(&args) == nullptr)
            KVCM_LOG_INFO("Total devices   : %zu (addressable: %zu)",
                           args.num_devices, num_addressable_devices);
    }
    {
        PJRT_INIT(PJRT_Client_AddressableMemories);
        args.client = client_;
        if (api_->PJRT_Client_AddressableMemories(&args) == nullptr)
            KVCM_LOG_INFO("Client addressable memories: %zu", args.num_addressable_memories);
    }

    // --- Per-device info ---
    PJRT_INIT(PJRT_Client_AddressableDevices);
    args.client = client_;
    if (api_->PJRT_Client_AddressableDevices(&args) != nullptr) {
        KVCM_LOG_INFO("===== End TPU Platform & Device Info =====");
        return;
    }

    for (size_t i = 0; i < args.num_addressable_devices; ++i) {
        PJRT_Device* dev = args.addressable_devices[i];
        KVCM_LOG_INFO("--- Device [%zu] ---", i);

        {
            PJRT_INIT(PJRT_Device_IsAddressable);
            args.device = dev;
            if (api_->PJRT_Device_IsAddressable(&args) == nullptr)
                KVCM_LOG_INFO("  is_addressable    : %s", args.is_addressable ? "true" : "false");
        }
        {
            PJRT_INIT(PJRT_Device_LocalHardwareId);
            args.device = dev;
            if (api_->PJRT_Device_LocalHardwareId(&args) == nullptr)
                KVCM_LOG_INFO("  local_hardware_id : %d", args.local_hardware_id);
        }

        // DeviceDescription
        {
            PJRT_INIT(PJRT_Device_GetDescription);
            args.device = dev;
            if (api_->PJRT_Device_GetDescription(&args) == nullptr) {
                PJRT_DeviceDescription* desc = args.device_description;

                { PJRT_INIT(PJRT_DeviceDescription_Id);
                  args.device_description = desc;
                  if (api_->PJRT_DeviceDescription_Id(&args) == nullptr)
                      KVCM_LOG_INFO("  device_id         : %d", args.id); }

                { PJRT_INIT(PJRT_DeviceDescription_ProcessIndex);
                  args.device_description = desc;
                  if (api_->PJRT_DeviceDescription_ProcessIndex(&args) == nullptr)
                      KVCM_LOG_INFO("  device_proc_index : %d", args.process_index); }

                { PJRT_INIT(PJRT_DeviceDescription_Kind);
                  args.device_description = desc;
                  if (api_->PJRT_DeviceDescription_Kind(&args) == nullptr)
                      KVCM_LOG_INFO("  device_kind       : %.*s",
                                     static_cast<int>(args.device_kind_size), args.device_kind); }

                { PJRT_INIT(PJRT_DeviceDescription_ToString);
                  args.device_description = desc;
                  if (api_->PJRT_DeviceDescription_ToString(&args) == nullptr)
                      KVCM_LOG_INFO("  to_string         : %.*s",
                                     static_cast<int>(args.to_string_size), args.to_string); }

                { PJRT_INIT(PJRT_DeviceDescription_DebugString);
                  args.device_description = desc;
                  if (api_->PJRT_DeviceDescription_DebugString(&args) == nullptr)
                      KVCM_LOG_INFO("  debug_string      : %.*s",
                                     static_cast<int>(args.debug_string_size), args.debug_string); }
            }
        }

        // DefaultMemory + Memory details
        {
            PJRT_INIT(PJRT_Device_DefaultMemory);
            args.device = dev;
            if (api_->PJRT_Device_DefaultMemory(&args) == nullptr && args.memory) {
                PJRT_Memory* mem = args.memory;

                { PJRT_INIT(PJRT_Memory_Id);
                  args.memory = mem;
                  if (api_->PJRT_Memory_Id(&args) == nullptr)
                      KVCM_LOG_INFO("  default_memory_id : %d", args.id); }

                { PJRT_INIT(PJRT_Memory_Kind);
                  args.memory = mem;
                  if (api_->PJRT_Memory_Kind(&args) == nullptr)
                      KVCM_LOG_INFO("  default_memory_kind: %.*s",
                                     static_cast<int>(args.kind_size), args.kind); }

                { PJRT_INIT(PJRT_Memory_Kind_Id);
                  args.memory = mem;
                  if (api_->PJRT_Memory_Kind_Id(&args) == nullptr)
                      KVCM_LOG_INFO("  memory_kind_id    : %d", args.kind_id); }

                { PJRT_INIT(PJRT_Memory_DebugString);
                  args.memory = mem;
                  if (api_->PJRT_Memory_DebugString(&args) == nullptr)
                      KVCM_LOG_INFO("  memory_debug_str  : %.*s",
                                     static_cast<int>(args.debug_string_size), args.debug_string); }
            }
        }

        // Addressable memories for this device
        {
            PJRT_INIT(PJRT_Device_AddressableMemories);
            args.device = dev;
            if (api_->PJRT_Device_AddressableMemories(&args) == nullptr)
                KVCM_LOG_INFO("  addressable_memories: %zu", args.num_memories);
        }

        // MemoryStats (best-effort, may return UNIMPLEMENTED)
        {
            PJRT_INIT(PJRT_Device_MemoryStats);
            args.device = dev;
            PJRT_Error* ms_err = api_->PJRT_Device_MemoryStats(&args);
            if (ms_err == nullptr) {
                KVCM_LOG_INFO("  memory.bytes_in_use: %ld", args.bytes_in_use);
                if (args.bytes_limit_is_set)
                    KVCM_LOG_INFO("  memory.bytes_limit : %ld", args.bytes_limit);
                if (args.peak_bytes_in_use_is_set)
                    KVCM_LOG_INFO("  memory.peak_bytes  : %ld", args.peak_bytes_in_use);
                if (args.largest_alloc_size_is_set)
                    KVCM_LOG_INFO("  memory.largest_alloc: %ld", args.largest_alloc_size);
                if (args.bytes_reserved_is_set)
                    KVCM_LOG_INFO("  memory.bytes_reserved: %ld", args.bytes_reserved);
                if (args.pool_bytes_is_set)
                    KVCM_LOG_INFO("  memory.pool_bytes  : %ld", args.pool_bytes);
            } else {
                std::string msg = GetErrorMessage(api_, ms_err);
                KVCM_LOG_INFO("  memory_stats      : unavailable (%s)", msg.c_str());
            }
        }
    }

    KVCM_LOG_INFO("===== End TPU Platform & Device Info =====");
}

} // namespace kv_cache_manager

#endif // USING_TPU
