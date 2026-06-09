// TPU client implementation using PJRT C API
// Provides clean interface without CreateViewOfDeviceBuffer (unsupported on TPU)

#ifdef USING_TPU

#include "kv_cache_manager/client/src/internal/sdk/tpu_client.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <link.h>
#include <vector>

#include "xla/pjrt/c/pjrt_c_api_raw_buffer_extension.h"

namespace kv_cache_manager {

namespace {
// Default libtpu.so path (installed by uv in the vllm_tpu_env_yemu virtualenv)
constexpr const char* kDefaultLibtpuPath =
    "/home/zhaotaonan_ztn/vllm_tpu_env_yemu/lib/python3.12/site-packages/libtpu/libtpu.so";

// Offset of RealInitGoogle in libtpu.so v0.0.41.
// Demangled: RealInitGoogle(std::string_view, int*, char***, bool, bool)
// This is the internal initialization function that sets up Google's
// infrastructure (flags, logging, etc.) required by PJRT_Client_Create.
// In standalone C++ (unlike JAX/Python), we must call this explicitly.
// TODO: make this version-aware or use dlinfo+symbol lookup if possible.
constexpr uintptr_t kRealInitGoogleOffset = 0x1d654780;

// Function signature matching the demangled RealInitGoogle:
// std::string_view is passed as (const char* data, size_t len) in libc++ ABI
using RealInitGoogleFunc = void (*)(const char* progname, size_t len,
                                     int* argc, char*** argv,
                                     bool remove_flags, bool install_signal_handlers);

const char* GetLibtpuPath() {
    const char* env_path = std::getenv("TPU_LIBRARY_PATH");
    return (env_path && env_path[0] != '\0') ? env_path : kDefaultLibtpuPath;
}

// Call RealInitGoogle() to set up Google's internal infrastructure
// (flags, logging, filesystem, etc.) required by PJRT_Client_Create.
// This is equivalent to what JAX's jaxlib does during Python import.
bool CallRealInitGoogle(void* libtpu_handle) {
    struct link_map* lm = nullptr;
    if (dlinfo(libtpu_handle, RTLD_DI_LINKMAP, &lm) != 0) {
        KVCM_LOG_ERROR("dlinfo failed: %s", dlerror());
        return false;
    }

    uintptr_t base = static_cast<uintptr_t>(lm->l_addr);
    auto init_google = reinterpret_cast<RealInitGoogleFunc>(base + kRealInitGoogleOffset);

    // Call with minimal args: program name, no argc/argv manipulation,
    // don't remove flags, don't install signal handlers.
    static const char* kProgName = "kvcm_tpu";
    static int dummy_argc = 1;
    static char* dummy_argv[] = {const_cast<char*>(kProgName), nullptr};
    static char** dummy_argv_ptr = dummy_argv;

    init_google(kProgName, strlen(kProgName), &dummy_argc, &dummy_argv_ptr,
                false, false);

    KVCM_LOG_INFO("RealInitGoogle called successfully (base=0x%lx)", base);
    return true;
}
} // anonymous namespace

TpuClient::~TpuClient() {
    Destroy();
}

std::string TpuClient::GetErrorMessage(const PJRT_Api* api, PJRT_Error* error) {
    if (!api || !error) return "unknown error";
    PJRT_Error_Message_Args args{};
    args.struct_size = PJRT_Error_Message_Args_STRUCT_SIZE;
    args.extension_start = nullptr;
    args.error = error;
    api->PJRT_Error_Message(&args);
    std::string msg(args.message, args.message_size);

    // Destroy the error
    PJRT_Error_Destroy_Args destroy_args{};
    destroy_args.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
    destroy_args.extension_start = nullptr;
    destroy_args.error = error;
    api->PJRT_Error_Destroy(&destroy_args);
    return msg;
}

ClientErrorCode TpuClient::Init() {
    // If already initialized, destroy the old client first
    if (client_) {
        Destroy();
    }

    // Load libtpu.so and run one-time Google init on first call only.
    // RealInitGoogle() is not idempotent - calling it twice will abort.
    // PJRT_Plugin_Initialize() is also not idempotent.
    // If libtpu is already loaded (e.g. by JAX's TPU plugin), we skip
    // both dlopen and RealInitGoogle, and just reuse the existing PJRT_Api.
    static bool google_initialized = false;
    static void* shared_libtpu_handle = nullptr;

    if (!google_initialized) {
        const char* lib_path = GetLibtpuPath();

        // Check if libtpu.so is already loaded in this process
        // (e.g. JAX imported before us). RTLD_NOLOAD returns a handle
        // without loading the library if it's not already loaded.
        void* existing_handle = dlopen(lib_path, RTLD_NOW | RTLD_GLOBAL | RTLD_NOLOAD);
        if (existing_handle) {
            // libtpu already loaded by another component (e.g. JAX).
            // RealInitGoogle and PJRT_Plugin_Initialize have already run —
            // skip them to avoid duplicate-init abort.
            shared_libtpu_handle = existing_handle;
            KVCM_LOG_INFO("libtpu already loaded (by JAX or another component), reusing handle");
        } else {
            // First loader in this process — load and initialize
            shared_libtpu_handle = dlopen(lib_path, RTLD_NOW | RTLD_GLOBAL);
            if (!shared_libtpu_handle) {
                KVCM_LOG_ERROR("dlopen(%s) failed: %s", lib_path, dlerror());
                return ER_TPU_PJRT_INIT_ERROR;
            }
            KVCM_LOG_INFO("libtpu loaded from: %s", lib_path);

            if (!CallRealInitGoogle(shared_libtpu_handle)) {
                KVCM_LOG_ERROR("Failed to call RealInitGoogle");
                dlclose(shared_libtpu_handle);
                shared_libtpu_handle = nullptr;
                return ER_TPU_PJRT_INIT_ERROR;
            }
        }
        google_initialized = true;
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

    // Create PJRT client
    PJRT_Client_Create_Args create_args{};
    create_args.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
    create_args.extension_start = nullptr;
    create_args.create_options = nullptr;
    create_args.num_options = 0;
    create_args.kv_get_callback = nullptr;
    create_args.kv_get_user_arg = nullptr;
    create_args.kv_put_callback = nullptr;
    create_args.kv_put_user_arg = nullptr;

    PJRT_Error* err = api_->PJRT_Client_Create(&create_args);
    if (err != nullptr) {
        std::string msg = GetErrorMessage(api_, err);
        KVCM_LOG_ERROR("PJRT_Client_Create failed: %s", msg.c_str());
        return ER_TPU_PJRT_INIT_ERROR;
    }
    client_ = create_args.client;

    // Get addressable devices
    PJRT_Client_AddressableDevices_Args devices_args{};
    devices_args.struct_size = PJRT_Client_AddressableDevices_Args_STRUCT_SIZE;
    devices_args.extension_start = nullptr;
    devices_args.client = client_;
    api_->PJRT_Client_AddressableDevices(&devices_args);

    if (devices_args.num_addressable_devices == 0) {
        KVCM_LOG_ERROR("No addressable TPU devices found");
        Destroy();
        return ER_TPU_PJRT_INIT_ERROR;
    }

    device_ = devices_args.addressable_devices[0];
    KVCM_LOG_INFO("TPU PJRT client initialized with %zu devices",
                   devices_args.num_addressable_devices);

    // Log platform/device/memory info via read-only PJRT APIs
    LogDeviceInfo(devices_args.num_addressable_devices);

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
        PJRT_Client_Destroy_Args args{};
        args.struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE;
        args.extension_start = nullptr;
        args.client = client_;
        api_->PJRT_Client_Destroy(&args);
    }
    client_ = nullptr;
    device_ = nullptr;
    api_ = nullptr;
    rawbuf_ext_ = nullptr;
    // Note: libtpu_handle_ points to a shared static handle, don't dlclose here.
    libtpu_handle_ = nullptr;
}

ClientErrorCode TpuClient::DmaMap(void* data, size_t size) {
    if (!api_ || !client_) {
        return ER_TPU_DMA_MAP_ERROR;
    }
    // DMA mapping for TPU - return success as basic operations don't require it
    return ER_OK;
}

ClientErrorCode TpuClient::DmaUnmap(void* data) {
    if (!api_ || !client_) {
        return ER_TPU_DMA_MAP_ERROR;
    }
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

    PJRT_Buffer_Destroy_Args args{};
    args.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
    args.extension_start = nullptr;
    args.buffer = buffer;
    api_->PJRT_Buffer_Destroy(&args);
}

// =========================================================================
// Asynchronous Buffer Transfers
// =========================================================================

ClientErrorCode TpuClient::BufferFromHostAsync(const void* host_src, size_t size,
                                                PJRT_Buffer*& out_buffer,
                                                PJRT_Event*& out_event) {
    if (!api_ || !client_ || !device_) {
        return ER_TPU_BUFFER_TRANSFER_ERROR;
    }

    if (size == 0) {
        return ER_INVALID_PARAMS;
    }

    int64_t dim = static_cast<int64_t>(size);

    PJRT_Client_BufferFromHostBuffer_Args args{};
    args.struct_size = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
    args.extension_start = nullptr;
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

    PJRT_Error* err = api_->PJRT_Client_BufferFromHostBuffer(&args);
    if (err != nullptr) {
        std::string msg = GetErrorMessage(api_, err);
        KVCM_LOG_ERROR("BufferFromHostBuffer(async) failed: %s", msg.c_str());
        return ER_TPU_BUFFER_TRANSFER_ERROR;
    }

    out_buffer = args.buffer;
    out_event = args.done_with_host_buffer;  // may be nullptr
    return ER_OK;
}

ClientErrorCode TpuClient::BufferToHostAsync(PJRT_Buffer* buffer, void* host_dst,
                                              size_t size, PJRT_Event*& out_event) {
    if (!api_ || !buffer) {
        return ER_TPU_BUFFER_TRANSFER_ERROR;
    }

    if (size == 0) {
        return ER_INVALID_PARAMS;
    }

    PJRT_Buffer_ToHostBuffer_Args args{};
    args.struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
    args.extension_start = nullptr;
    args.src = buffer;
    args.host_layout = nullptr;
    args.dst = host_dst;
    args.dst_size = size;

    PJRT_Error* err = api_->PJRT_Buffer_ToHostBuffer(&args);
    if (err != nullptr) {
        std::string msg = GetErrorMessage(api_, err);
        KVCM_LOG_ERROR("Buffer_ToHostBuffer(async) failed: %s", msg.c_str());
        return ER_TPU_BUFFER_TRANSFER_ERROR;
    }

    out_event = args.event;  // may be nullptr
    return ER_OK;
}

// =========================================================================
// Event Management
// =========================================================================

ClientErrorCode TpuClient::WaitEvent(PJRT_Event* event) {
    if (!api_ || !event) {
        return ER_OK;  // nothing to wait on
    }

    PJRT_Event_Await_Args await_args{};
    await_args.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
    await_args.extension_start = nullptr;
    await_args.event = event;
    PJRT_Error* err = api_->PJRT_Event_Await(&await_args);

    if (err != nullptr) {
        std::string msg = GetErrorMessage(api_, err);
        KVCM_LOG_ERROR("Event_Await failed: %s", msg.c_str());
        DestroyEvent(event);
        return ER_TPU_EVENT_ERROR;
    }

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

    PJRT_Event_Destroy_Args args{};
    args.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
    args.extension_start = nullptr;
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
    if (!api_ || !rawbuf_ext_ || !rawbuf_ext_->PJRT_RawBuffer_CreateRawAliasOfBuffer) {
        return ER_TPU_RAWBUFFER_ERROR;
    }
    if (!buffer) {
        return ER_INVALID_PARAMS;
    }

    PJRT_RawBuffer_CreateRawAliasOfBuffer_Args args{};
    args.struct_size = PJRT_RawBuffer_CreateRawAliasOfBuffer_Args_STRUCT_SIZE;
    args.extension_start = nullptr;
    args.buffer = buffer;

    PJRT_Error* err = rawbuf_ext_->PJRT_RawBuffer_CreateRawAliasOfBuffer(&args);
    if (err != nullptr) {
        std::string msg = GetErrorMessage(api_, err);
        KVCM_LOG_ERROR("CreateRawAliasOfBuffer failed: %s", msg.c_str());
        return ER_TPU_RAWBUFFER_ERROR;
    }

    out_raw = args.raw_buffer;
    return ER_OK;
}

ClientErrorCode TpuClient::RawBufferFromHost(PJRT_RawBuffer* raw, const void* src,
                                              int64_t offset, int64_t size,
                                              PJRT_Event*& out_event) {
    if (!api_ || !rawbuf_ext_ || !rawbuf_ext_->PJRT_RawBuffer_CopyRawHostToDevice) {
        return ER_TPU_RAWBUFFER_ERROR;
    }
    if (!raw || !src || size <= 0) {
        return ER_INVALID_PARAMS;
    }

    PJRT_RawBuffer_CopyRawHostToDevice_Args args{};
    args.struct_size = PJRT_RawBuffer_CopyRawHostToDevice_Args_STRUCT_SIZE;
    args.extension_start = nullptr;
    args.buffer = raw;
    args.src = src;
    args.offset = offset;
    args.transfer_size = size;

    PJRT_Error* err = rawbuf_ext_->PJRT_RawBuffer_CopyRawHostToDevice(&args);
    if (err != nullptr) {
        std::string msg = GetErrorMessage(api_, err);
        KVCM_LOG_ERROR("RawBuffer_CopyRawHostToDevice failed: %s", msg.c_str());
        return ER_TPU_RAWBUFFER_ERROR;
    }

    out_event = args.event;
    return ER_OK;
}

ClientErrorCode TpuClient::RawBufferToHost(PJRT_RawBuffer* raw, void* dst,
                                            int64_t offset, int64_t size,
                                            PJRT_Event*& out_event) {
    if (!api_ || !rawbuf_ext_ || !rawbuf_ext_->PJRT_RawBuffer_CopyRawDeviceToHost) {
        return ER_TPU_RAWBUFFER_ERROR;
    }
    if (!raw || !dst || size <= 0) {
        return ER_INVALID_PARAMS;
    }

    PJRT_RawBuffer_CopyRawDeviceToHost_Args args{};
    args.struct_size = PJRT_RawBuffer_CopyRawDeviceToHost_Args_STRUCT_SIZE;
    args.extension_start = nullptr;
    args.buffer = raw;
    args.dst = dst;
    args.offset = offset;
    args.transfer_size = size;

    PJRT_Error* err = rawbuf_ext_->PJRT_RawBuffer_CopyRawDeviceToHost(&args);
    if (err != nullptr) {
        std::string msg = GetErrorMessage(api_, err);
        KVCM_LOG_ERROR("RawBuffer_CopyRawDeviceToHost failed: %s", msg.c_str());
        return ER_TPU_RAWBUFFER_ERROR;
    }

    out_event = args.event;
    return ER_OK;
}

ClientErrorCode TpuClient::RawBufferGetDeviceSize(PJRT_RawBuffer* raw, size_t& out_size) {
    if (!api_ || !rawbuf_ext_ || !rawbuf_ext_->PJRT_RawBuffer_GetOnDeviceSizeInBytes) {
        return ER_TPU_RAWBUFFER_ERROR;
    }
    if (!raw) {
        return ER_INVALID_PARAMS;
    }

    PJRT_RawBuffer_GetOnDeviceSizeInBytes_Args args{};
    args.struct_size = PJRT_RawBuffer_GetOnDeviceSizeInBytes_Args_STRUCT_SIZE;
    args.extension_start = nullptr;
    args.buffer = raw;

    PJRT_Error* err = rawbuf_ext_->PJRT_RawBuffer_GetOnDeviceSizeInBytes(&args);
    if (err != nullptr) {
        std::string msg = GetErrorMessage(api_, err);
        KVCM_LOG_ERROR("RawBuffer_GetOnDeviceSizeInBytes failed: %s", msg.c_str());
        return ER_TPU_RAWBUFFER_ERROR;
    }

    out_size = args.on_device_size_in_bytes;
    return ER_OK;
}

void TpuClient::DestroyRawBuffer(PJRT_RawBuffer* raw) {
    if (!api_ || !rawbuf_ext_ || !raw) return;
    if (!rawbuf_ext_->PJRT_RawBuffer_Destroy) return;

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
    if (!api_ || !client_) return;

    KVCM_LOG_INFO("===== TPU Platform & Device Info =====");

    // --- Client-level info ---
    PJRT_Client_PlatformName_Args name_args{};
    name_args.struct_size = PJRT_Client_PlatformName_Args_STRUCT_SIZE;
    name_args.extension_start = nullptr;
    name_args.client = client_;
    if (api_->PJRT_Client_PlatformName(&name_args) == nullptr) {
        KVCM_LOG_INFO("Platform name   : %.*s",
                       static_cast<int>(name_args.platform_name_size),
                       name_args.platform_name);
    }

    PJRT_Client_PlatformVersion_Args ver_args{};
    ver_args.struct_size = PJRT_Client_PlatformVersion_Args_STRUCT_SIZE;
    ver_args.extension_start = nullptr;
    ver_args.client = client_;
    if (api_->PJRT_Client_PlatformVersion(&ver_args) == nullptr) {
        KVCM_LOG_INFO("Platform version: %.*s",
                       static_cast<int>(ver_args.platform_version_size),
                       ver_args.platform_version);
    }

    PJRT_Client_ProcessIndex_Args pi_args{};
    pi_args.struct_size = PJRT_Client_ProcessIndex_Args_STRUCT_SIZE;
    pi_args.extension_start = nullptr;
    pi_args.client = client_;
    if (api_->PJRT_Client_ProcessIndex(&pi_args) == nullptr) {
        KVCM_LOG_INFO("Process index   : %d", pi_args.process_index);
    }

    // Total devices (including non-addressable)
    PJRT_Client_Devices_Args devs_args{};
    devs_args.struct_size = PJRT_Client_Devices_Args_STRUCT_SIZE;
    devs_args.extension_start = nullptr;
    devs_args.client = client_;
    if (api_->PJRT_Client_Devices(&devs_args) == nullptr) {
        KVCM_LOG_INFO("Total devices   : %zu (addressable: %zu)",
                       devs_args.num_devices, num_addressable_devices);
    }

    // Addressable memories at client level
    PJRT_Client_AddressableMemories_Args cmem_args{};
    cmem_args.struct_size = PJRT_Client_AddressableMemories_Args_STRUCT_SIZE;
    cmem_args.extension_start = nullptr;
    cmem_args.client = client_;
    if (api_->PJRT_Client_AddressableMemories(&cmem_args) == nullptr) {
        KVCM_LOG_INFO("Client addressable memories: %zu",
                       cmem_args.num_addressable_memories);
    }

    // --- Per-device info ---
    // Get addressable devices again for iteration
    PJRT_Client_AddressableDevices_Args addr_args{};
    addr_args.struct_size = PJRT_Client_AddressableDevices_Args_STRUCT_SIZE;
    addr_args.extension_start = nullptr;
    addr_args.client = client_;
    if (api_->PJRT_Client_AddressableDevices(&addr_args) != nullptr) {
        KVCM_LOG_INFO("===== End TPU Platform & Device Info =====");
        return;
    }

    for (size_t i = 0; i < addr_args.num_addressable_devices; ++i) {
        PJRT_Device* dev = addr_args.addressable_devices[i];
        KVCM_LOG_INFO("--- Device [%zu] ---", i);

        // IsAddressable
        PJRT_Device_IsAddressable_Args ia_args{};
        ia_args.struct_size = PJRT_Device_IsAddressable_Args_STRUCT_SIZE;
        ia_args.extension_start = nullptr;
        ia_args.device = dev;
        if (api_->PJRT_Device_IsAddressable(&ia_args) == nullptr) {
            KVCM_LOG_INFO("  is_addressable    : %s",
                           ia_args.is_addressable ? "true" : "false");
        }

        // LocalHardwareId
        PJRT_Device_LocalHardwareId_Args hwid_args{};
        hwid_args.struct_size = PJRT_Device_LocalHardwareId_Args_STRUCT_SIZE;
        hwid_args.extension_start = nullptr;
        hwid_args.device = dev;
        if (api_->PJRT_Device_LocalHardwareId(&hwid_args) == nullptr) {
            KVCM_LOG_INFO("  local_hardware_id : %d",
                           hwid_args.local_hardware_id);
        }

        // DeviceDescription
        PJRT_Device_GetDescription_Args desc_args{};
        desc_args.struct_size = PJRT_Device_GetDescription_Args_STRUCT_SIZE;
        desc_args.extension_start = nullptr;
        desc_args.device = dev;
        if (api_->PJRT_Device_GetDescription(&desc_args) == nullptr) {
            PJRT_DeviceDescription* desc = desc_args.device_description;

            PJRT_DeviceDescription_Id_Args id_args{};
            id_args.struct_size = PJRT_DeviceDescription_Id_Args_STRUCT_SIZE;
            id_args.extension_start = nullptr;
            id_args.device_description = desc;
            if (api_->PJRT_DeviceDescription_Id(&id_args) == nullptr) {
                KVCM_LOG_INFO("  device_id         : %d", id_args.id);
            }

            PJRT_DeviceDescription_ProcessIndex_Args dpi_args{};
            dpi_args.struct_size = PJRT_DeviceDescription_ProcessIndex_Args_STRUCT_SIZE;
            dpi_args.extension_start = nullptr;
            dpi_args.device_description = desc;
            if (api_->PJRT_DeviceDescription_ProcessIndex(&dpi_args) == nullptr) {
                KVCM_LOG_INFO("  device_proc_index : %d",
                               dpi_args.process_index);
            }

            PJRT_DeviceDescription_Kind_Args kind_args{};
            kind_args.struct_size = PJRT_DeviceDescription_Kind_Args_STRUCT_SIZE;
            kind_args.extension_start = nullptr;
            kind_args.device_description = desc;
            if (api_->PJRT_DeviceDescription_Kind(&kind_args) == nullptr) {
                KVCM_LOG_INFO("  device_kind       : %.*s",
                               static_cast<int>(kind_args.device_kind_size),
                               kind_args.device_kind);
            }

            PJRT_DeviceDescription_ToString_Args str_args{};
            str_args.struct_size = PJRT_DeviceDescription_ToString_Args_STRUCT_SIZE;
            str_args.extension_start = nullptr;
            str_args.device_description = desc;
            if (api_->PJRT_DeviceDescription_ToString(&str_args) == nullptr) {
                KVCM_LOG_INFO("  to_string         : %.*s",
                               static_cast<int>(str_args.to_string_size),
                               str_args.to_string);
            }

            PJRT_DeviceDescription_DebugString_Args dbg_args{};
            dbg_args.struct_size = PJRT_DeviceDescription_DebugString_Args_STRUCT_SIZE;
            dbg_args.extension_start = nullptr;
            dbg_args.device_description = desc;
            if (api_->PJRT_DeviceDescription_DebugString(&dbg_args) == nullptr) {
                KVCM_LOG_INFO("  debug_string      : %.*s",
                               static_cast<int>(dbg_args.debug_string_size),
                               dbg_args.debug_string);
            }
        }

        // DefaultMemory + Memory details
        PJRT_Device_DefaultMemory_Args dm_args{};
        dm_args.struct_size = PJRT_Device_DefaultMemory_Args_STRUCT_SIZE;
        dm_args.extension_start = nullptr;
        dm_args.device = dev;
        if (api_->PJRT_Device_DefaultMemory(&dm_args) == nullptr && dm_args.memory) {
            PJRT_Memory* mem = dm_args.memory;

            PJRT_Memory_Id_Args mid_args{};
            mid_args.struct_size = PJRT_Memory_Id_Args_STRUCT_SIZE;
            mid_args.extension_start = nullptr;
            mid_args.memory = mem;
            if (api_->PJRT_Memory_Id(&mid_args) == nullptr) {
                KVCM_LOG_INFO("  default_memory_id : %d", mid_args.id);
            }

            PJRT_Memory_Kind_Args mk_args{};
            mk_args.struct_size = PJRT_Memory_Kind_Args_STRUCT_SIZE;
            mk_args.extension_start = nullptr;
            mk_args.memory = mem;
            if (api_->PJRT_Memory_Kind(&mk_args) == nullptr) {
                KVCM_LOG_INFO("  default_memory_kind: %.*s",
                               static_cast<int>(mk_args.kind_size),
                               mk_args.kind);
            }

            PJRT_Memory_Kind_Id_Args mki_args{};
            mki_args.struct_size = PJRT_Memory_Kind_Id_Args_STRUCT_SIZE;
            mki_args.extension_start = nullptr;
            mki_args.memory = mem;
            if (api_->PJRT_Memory_Kind_Id(&mki_args) == nullptr) {
                KVCM_LOG_INFO("  memory_kind_id    : %d",
                               mki_args.kind_id);
            }

            PJRT_Memory_DebugString_Args mdbg_args{};
            mdbg_args.struct_size = PJRT_Memory_DebugString_Args_STRUCT_SIZE;
            mdbg_args.extension_start = nullptr;
            mdbg_args.memory = mem;
            if (api_->PJRT_Memory_DebugString(&mdbg_args) == nullptr) {
                KVCM_LOG_INFO("  memory_debug_str  : %.*s",
                               static_cast<int>(mdbg_args.debug_string_size),
                               mdbg_args.debug_string);
            }
        }

        // Addressable memories for this device
        PJRT_Device_AddressableMemories_Args dam_args{};
        dam_args.struct_size = PJRT_Device_AddressableMemories_Args_STRUCT_SIZE;
        dam_args.extension_start = nullptr;
        dam_args.device = dev;
        if (api_->PJRT_Device_AddressableMemories(&dam_args) == nullptr) {
            KVCM_LOG_INFO("  addressable_memories: %zu",
                           dam_args.num_memories);
        }

        // MemoryStats (best-effort, may return UNIMPLEMENTED)
        PJRT_Device_MemoryStats_Args ms_args{};
        ms_args.struct_size = PJRT_Device_MemoryStats_Args_STRUCT_SIZE;
        ms_args.extension_start = nullptr;
        ms_args.device = dev;
        PJRT_Error* ms_err = api_->PJRT_Device_MemoryStats(&ms_args);
        if (ms_err == nullptr) {
            KVCM_LOG_INFO("  memory.bytes_in_use: %ld", ms_args.bytes_in_use);
            if (ms_args.bytes_limit_is_set) {
                KVCM_LOG_INFO("  memory.bytes_limit : %ld",
                               ms_args.bytes_limit);
            }
            if (ms_args.peak_bytes_in_use_is_set) {
                KVCM_LOG_INFO("  memory.peak_bytes  : %ld",
                               ms_args.peak_bytes_in_use);
            }
            if (ms_args.largest_alloc_size_is_set) {
                KVCM_LOG_INFO("  memory.largest_alloc: %ld",
                               ms_args.largest_alloc_size);
            }
            if (ms_args.bytes_reserved_is_set) {
                KVCM_LOG_INFO("  memory.bytes_reserved: %ld",
                               ms_args.bytes_reserved);
            }
            if (ms_args.pool_bytes_is_set) {
                KVCM_LOG_INFO("  memory.pool_bytes  : %ld",
                               ms_args.pool_bytes);
            }
        } else {
            // MemoryStats may return UNIMPLEMENTED - that's OK
            std::string msg = GetErrorMessage(api_, ms_err);
            KVCM_LOG_INFO("  memory_stats      : unavailable (%s)",
                           msg.c_str());
        }
    }

    KVCM_LOG_INFO("===== End TPU Platform & Device Info =====");
}

} // namespace kv_cache_manager

#endif // USING_TPU
