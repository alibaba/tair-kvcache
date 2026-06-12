#include "kv_cache_manager/client/src/internal/sdk/local_file_sdk.h"

#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#if defined(USING_CUDA)
#include "kv_cache_manager/client/src/internal/sdk/cuda_util.h"
#elif defined(USING_MUSA)
#include "kv_cache_manager/client/src/internal/sdk/musa_util.h"
#elif defined(USING_TPU)
#include "kv_cache_manager/client/src/internal/sdk/tpu_client.h"
#endif
#include "kv_cache_manager/client/src/internal/util/debug_string_util.h"
#include "kv_cache_manager/common/logger.h"

namespace {
class MmapHelper {
public:
    MmapHelper(int fd, void *file_mem, size_t file_size) : fd_(fd), file_mem_(file_mem), file_size_(file_size) {}
    kv_cache_manager::ClientErrorCode RegisterGpu(unsigned int register_flag) {
#if defined(USING_CUDA)
        CHECK_CUDA_ERROR_RETURN(cudaHostRegister(file_mem_, file_size_, register_flag),
                                kv_cache_manager::ER_CUDA_HOST_REGISTER_ERROR,
                                "register host mem [%p] fail, size: %zu, register_flag: %u",
                                file_mem_,
                                file_size_,
                                register_flag);
        is_mem_registered = true;
#elif defined(USING_MUSA)
        CHECK_MUSA_ERROR_RETURN(musaHostRegister(file_mem_, file_size_, register_flag),
                                kv_cache_manager::ER_CUDA_HOST_REGISTER_ERROR,
                                "register host mem [%p] fail, size: %zu, register_flag: %u",
                                file_mem_,
                                file_size_,
                                register_flag);
        is_mem_registered = true;
#endif
        return kv_cache_manager::ER_OK;
    }

#if defined(USING_TPU)
    kv_cache_manager::ClientErrorCode RegisterTpu(kv_cache_manager::TpuClient *tpu_client) {
        tpu_client_ = tpu_client;
        auto ec = tpu_client_->DmaMap(file_mem_, file_size_);
        if (ec != kv_cache_manager::ER_OK) {
            return ec;
        }
        is_mem_registered = true;
        return kv_cache_manager::ER_OK;
    }
#endif

    void SkipRegistration() {
#if defined(USING_CUDA) || defined(USING_MUSA) || defined(USING_TPU)
        is_mem_registered = false;
#endif
    }

    ~MmapHelper() {
#if defined(USING_CUDA)
        if (is_mem_registered) {
            CHECK_CUDA_ERROR(cudaHostUnregister(file_mem_), "unregister host mem [%p] fail", file_mem_);
        }
#elif defined(USING_MUSA)
        if (is_mem_registered) {
            CHECK_MUSA_ERROR(musaHostUnregister(file_mem_), "unregister host mem [%p] fail", file_mem_);
        }
#elif defined(USING_TPU)
        if (is_mem_registered && tpu_client_) {
            tpu_client_->DmaUnmap(file_mem_);
        }
#endif
        munmap(file_mem_, file_size_);
        close(fd_);
    }

private:
    int fd_;
    void *file_mem_;
    size_t file_size_;
#if defined(USING_CUDA) || defined(USING_MUSA)
    bool is_mem_registered = false;
#elif defined(USING_TPU)
    bool is_mem_registered = false;
    kv_cache_manager::TpuClient *tpu_client_ = nullptr;
#endif
};

[[maybe_unused]] int getGpusDeviceCount() {
    int count = 0;
#if defined(USING_CUDA)
    CHECK_CUDA_ERROR_RETURN(cudaGetDeviceCount(&count), -1, "cudaGetDeviceCount failed");
#elif defined(USING_MUSA)
    CHECK_MUSA_ERROR_RETURN(musaGetDeviceCount(&count), -1, "musaGetDeviceCount failed");
#endif
    return count;
}

[[maybe_unused]] bool allGpusSupportHostRegister() {
    int count = getGpusDeviceCount();
    if (count < 0) {
        return false;
    }

    for (int dev = 0; dev < count; ++dev) {
        int value = 0;
#if defined(USING_CUDA)
        CHECK_CUDA_ERROR_RETURN(cudaDeviceGetAttribute(&value, cudaDevAttrHostRegisterSupported, dev), false, "get cudaDevAttrHostRegisterSupported failed");
#elif defined(USING_MUSA)
        CHECK_MUSA_ERROR_RETURN(musaDeviceGetAttribute(&value, musaDevAttrHostRegisterSupported, dev), false, "get musaDevAttrHostRegisterSupported failed");
#endif
        if (value != 1) {
            return false;
        }
    }
    return true;
}

[[maybe_unused]] bool allGpusSupportHostRegisterReadOnly() {
    int count = getGpusDeviceCount();
    if (count < 0) {
        return false;
    }

    for (int dev = 0; dev < count; ++dev) {
        int value = 0;
#if defined(USING_CUDA)
        CHECK_CUDA_ERROR_RETURN(cudaDeviceGetAttribute(&value, cudaDevAttrHostRegisterReadOnlySupported, dev), false, "get cudaDevAttrHostRegisterReadOnlySupported failed");
#elif defined(USING_MUSA)
        CHECK_MUSA_ERROR_RETURN(musaDeviceGetAttribute(&value, musaDevAttrHostRegisterReadOnlySupported, dev), false, "get musaDevAttrHostRegisterReadOnlySupported failed");
#endif
        if (value != 1) {
            return false;
        }
    }
    return true;
}

// Check if all GPUs support direct access to pageable memory (including mmap'd memory)
// If true, cudaHostRegister is not needed for mmap'd memory
[[maybe_unused]] bool allGpusSupportPageableMemoryAccess() {
    int count = getGpusDeviceCount();
    if (count < 0) {
        return false;
    }

    for (int dev = 0; dev < count; ++dev) {
        int value = 0;
#if defined(USING_CUDA)
        CHECK_CUDA_ERROR_RETURN(cudaDeviceGetAttribute(&value, cudaDevAttrPageableMemoryAccess, dev), false, "get cudaDevAttrPageableMemoryAccess failed");
#elif defined(USING_MUSA)
        // MUSA equivalent - adjust if needed
        CHECK_MUSA_ERROR_RETURN(musaDeviceGetAttribute(&value, musaDevAttrPageableMemoryAccess, dev), false, "get musaDevAttrPageableMemoryAccess failed");
#endif
        if (value != 1) {
            return false;
        }
    }
    return true;
}

} // namespace

namespace kv_cache_manager {

LocalFileSdk::~LocalFileSdk() {
#if defined(USING_CUDA)
    if (cuda_stream_) {
        CHECK_CUDA_ERROR(cudaStreamDestroy(cuda_stream_), "destroy cuda stream error");
    }
#elif defined(USING_MUSA)
    if (musa_stream_) {
        CHECK_MUSA_ERROR(musaStreamDestroy(musa_stream_), "destroy musa stream error");
    }
#elif defined(USING_TPU)
    tpu_client_.Destroy();
#endif
}

LocalFileItem LocalFileItem::FromUri(const DataStorageUri &uri) {
    LocalFileItem item;
    item.file_path = uri.GetPath();
    uri.GetParamAs<uint64_t>("blkid", item.blkid);
    uri.GetParamAs<size_t>("size", item.size);
    return item;
}
ClientErrorCode LocalFileSdk::Init(const std::shared_ptr<SdkBackendConfig> &sdk_backend_config,
                                   const std::shared_ptr<StorageConfig> &storage_config) {
    if (!sdk_backend_config) {
        KVCM_LOG_WARN("Init local file sdk failed, sdk backend config is null");
        return ER_INVALID_SDKBACKEND_CONFIG;
    }
    spec_byte_sizes_per_block_ = sdk_backend_config->spec_byte_sizes_per_block();
    if (spec_byte_sizes_per_block_.empty()) {
        KVCM_LOG_WARN("Init local file sdk failed, spec_byte_sizes_per_block is empty");
        return ER_INVALID_SDKBACKEND_CONFIG;
    }
#if defined(USING_CUDA)
    CHECK_CUDA_ERROR_RETURN(cudaStreamCreateWithFlags(&cuda_stream_, cudaStreamNonBlocking),
                            ER_CUDA_STREAM_CREATE_ERROR,
                            "Init local file sdk failed");
    if (!allGpusSupportHostRegister()) {
        KVCM_LOG_ERROR("gpu not support HostRegister");
        return ER_SDKINIT_ERROR;
    }

    support_register_readonly_ = allGpusSupportHostRegisterReadOnly();
    KVCM_LOG_INFO("gpu support register readonly [%d]", static_cast<int>(support_register_readonly_));
    
    // Check if GPUs support direct pageable memory access
    // If true, we can skip cudaHostRegister for mmap'd memory
    support_pageable_memory_access_ = allGpusSupportPageableMemoryAccess();
    KVCM_LOG_INFO("gpu support pageable memory access [%d]", static_cast<int>(support_pageable_memory_access_));
#elif defined(USING_MUSA)
    CHECK_MUSA_ERROR_RETURN(musaStreamCreateWithFlags(&musa_stream_, musaStreamNonBlocking),
                            ER_CUDA_STREAM_CREATE_ERROR,
                            "Init local file sdk failed");
    if (!allGpusSupportHostRegister()) {
        KVCM_LOG_ERROR("gpu not support HostRegister");
        return ER_SDKINIT_ERROR;
    }

    support_register_readonly_ = allGpusSupportHostRegisterReadOnly();
    KVCM_LOG_INFO("gpu support register readonly [%d]", static_cast<int>(support_register_readonly_));
    
    // Check if GPUs support direct pageable memory access
    support_pageable_memory_access_ = allGpusSupportPageableMemoryAccess();
    KVCM_LOG_INFO("gpu support pageable memory access [%d]", static_cast<int>(support_pageable_memory_access_));
#elif defined(USING_TPU)
    auto tpu_ec = tpu_client_.Init();
    if (tpu_ec != ER_OK) {
        KVCM_LOG_ERROR("Init local file sdk failed, TPU PJRT client init failed");
        return ER_TPU_PJRT_INIT_ERROR;
    }
    KVCM_LOG_INFO("TPU PJRT client initialized successfully");
#endif
    return ER_OK;
}

SdkType LocalFileSdk::Type() { return SdkType::LOCAL_FILE; }

ClientErrorCode LocalFileSdk::Get(const std::vector<DataStorageUri> &remote_uris, const BlockBuffers &local_buffers) {
    if (remote_uris.size() != local_buffers.size()) {
        KVCM_LOG_ERROR("Get failed, remote_uris size not equal to local_buffers size");
        return ER_INVALID_PARAMS;
    }
    // const_cast: Get writes into destination buffers (CPU/GPU iov.base).
    // For TPU, iov.base holds a PyArray* (read-only input); data is written
    // directly into the jax.Array's existing device memory via RawBuffer.
    auto &mutable_buffers = const_cast<BlockBuffers &>(local_buffers);
    auto group_map = SplitByPath(remote_uris, mutable_buffers);
    for (const auto &group : group_map) {
        auto ec = DoGet(group.second.remote_uris, mutable_buffers, group.second.buffer_indices);
        if (ec != ER_OK) {
            KVCM_LOG_ERROR("DoGet failed, errorcode: %d", ec);
            return ER_SDKREAD_ERROR;
        }
    }
    return ER_OK;
}

ClientErrorCode LocalFileSdk::Put(const std::vector<DataStorageUri> &remote_uris,
                                  const BlockBuffers &local_buffers,
                                  std::shared_ptr<std::vector<DataStorageUri>> actual_remote_uris) {
    actual_remote_uris->clear();
    if (remote_uris.size() != local_buffers.size()) {
        KVCM_LOG_ERROR("Put failed, remote_uris size not equal to local_buffers size");
        return ER_INVALID_PARAMS;
    }
    auto group_map = SplitByPath(remote_uris, local_buffers);
    for (const auto &group : group_map) {
        std::string file_path = group.first;
        if (!std::filesystem::exists(file_path)) {
            auto ec = Alloc(group.second.remote_uris, *actual_remote_uris);
            if (ec != ER_OK) {
                KVCM_LOG_ERROR("Put failed, alloc failed, errorcode: %d", ec);
                return ER_SDKALLOC_ERROR;
            }
        } else {
            actual_remote_uris->insert(
                actual_remote_uris->end(), group.second.remote_uris.begin(), group.second.remote_uris.end());
        }
        auto ec = DoPut(group.second.remote_uris, group.second.local_buffers);
        if (ec != ER_OK) {
            KVCM_LOG_ERROR("Put failed, DoPut failed, errorcode: %d", ec);
            return ER_SDKWRITE_ERROR;
        }
    }
    return ER_OK;
}

ClientErrorCode LocalFileSdk::Alloc(const std::vector<DataStorageUri> &remote_uris,
                                    std::vector<DataStorageUri> &alloc_uris) {
    if (remote_uris.empty()) {
        KVCM_LOG_WARN("Alloc failed, remote_uris is empty");
        return ER_OK;
    }
    std::string file_path = remote_uris[0].GetPath();
    std::filesystem::path path(file_path);
    std::error_code ec;
    std::filesystem::create_directories(path.parent_path(), ec);
    if (ec) {
        KVCM_LOG_WARN(
            "Alloc failed, failed to create parent directories for %s: %s", file_path.c_str(), ec.message().c_str());
        return ER_FILE_IO_ERROR;
    }
    std::ofstream ofs(file_path, std::ios::app);
    if (!ofs) {
        KVCM_LOG_WARN("Alloc failed, failed to open or create file %s", file_path.c_str());
        return ER_FILE_IO_ERROR;
    }
    ofs.close();
    alloc_uris.insert(alloc_uris.end(), remote_uris.begin(), remote_uris.end());
    return ER_OK;
}

ClientErrorCode LocalFileSdk::DoGet(const std::vector<DataStorageUri> &remote_uris, BlockBuffers &local_buffers,
                                     const std::vector<size_t> &buffer_indices) {
    if (remote_uris.size() != buffer_indices.size() || remote_uris.empty()) {
        KVCM_LOG_ERROR("Do Get failed, remote_uris size not equal to buffer_indices size");
        return ER_INVALID_PARAMS;
    }

    std::string file_path = remote_uris[0].GetPath();
    if (!std::filesystem::exists(file_path)) {
        KVCM_LOG_WARN("Get failed, file %s is not exist", file_path.c_str());
        return ER_FILE_IO_ERROR;
    }
    int fd = ::open(file_path.c_str(), O_RDONLY);
    if (fd < 0) {
        KVCM_LOG_ERROR("Get failed, open file %s failed", file_path.c_str());
        return ER_FILE_IO_ERROR;
    }

    struct stat st;
    if (fstat(fd, &st) != 0) {
        KVCM_LOG_ERROR("Get failed, fstat file %s failed", file_path.c_str());
        close(fd);
        return ER_FILE_IO_ERROR;
    }
    size_t file_size = st.st_size;
    KVCM_LOG_DEBUG("Get file path [%s] size [%zu] block buffer [%s]",
                   file_path.c_str(),
                   file_size,
                   DebugStringUtil::ToString(local_buffers).c_str());
    int prot = PROT_READ;
#if !defined(USING_TPU)
    if (!support_register_readonly_) {
        prot |= PROT_WRITE;
    }
#else
    prot |= PROT_WRITE; // TPU DmaMap may need write access
#endif
    void *file_mem = mmap(nullptr, file_size, prot, MAP_PRIVATE, fd, 0);
    if (file_mem == MAP_FAILED) {
        KVCM_LOG_ERROR("Get failed, mmap file %s failed", file_path.c_str());
        close(fd);
        return ER_FILE_IO_ERROR;
    }

    MmapHelper helper(fd, file_mem, file_size);
#if defined(USING_CUDA)
    bool exist_gpu_iov = false; 
    // If GPU supports direct pageable memory access, skip cudaHostRegister
    // This allows direct DMA transfer between GPU and mmap'd memory without pinning
    if (!support_pageable_memory_access_) {
        auto register_ec = helper.RegisterGpu(support_register_readonly_ ? cudaHostRegisterReadOnly : cudaHostRegisterDefault);
        if (register_ec != ER_OK) {
            return register_ec;
        }
    } else {
        KVCM_LOG_DEBUG("Skipping cudaHostRegister - GPU supports direct pageable memory access");
        helper.SkipRegistration(); // Mark as not registered since we don't need to
    }
#elif defined(USING_MUSA)
    bool exist_gpu_iov = false;
    if (!support_pageable_memory_access_) {
        auto register_ec = helper.RegisterGpu(support_register_readonly_ ? musaHostRegisterReadOnly : musaHostRegisterDefault);
        if (register_ec != ER_OK) {
            return register_ec;
        }
    }
#elif defined(USING_TPU)
    [[maybe_unused]] bool exist_tpu_iov = false;
    {
        auto register_ec = helper.RegisterTpu(&tpu_client_);
        if (register_ec != ER_OK) {
            return register_ec;
        }
    }
    std::vector<PJRT_Event*> tpu_events;
    std::vector<PJRT_Buffer*> tpu_new_bufs;   // newly created buffers (Get output)
    std::vector<void**> tpu_iov_bases;         // iov.base pointers to update (Get output)
    std::vector<std::vector<char>> tpu_temp_bufs;  // temp host buffers for H2D
#endif

    size_t offset = 0;
    char *src = static_cast<char *>(file_mem);
    // asume that url is sorted by blkid
    for (size_t i = 0; i < remote_uris.size(); ++i) {
        auto &remote_uri = remote_uris[i];
        auto &local_buffer = local_buffers[buffer_indices[i]];
        if (remote_uri.GetPath().empty()) {
            KVCM_LOG_ERROR("Get failed, remote_uri is invalid");
            return ER_INVALID_PARAMS;
        }
        auto item = LocalFileItem::FromUri(remote_uri);

        // 防御性校验：URI 的 size 必须在允许的 spec 范围内
        bool size_valid = false;
        for (const auto &[spec_name, byte_size_per_block] : spec_byte_sizes_per_block_) {
            if (item.size == byte_size_per_block) {
                size_valid = true;
                break;
            }
        }
        if (!size_valid) {
            KVCM_LOG_ERROR("Get failed, URI size [%zu] not in allowed spec_byte_sizes_per_block, uri: %s",
                           item.size,
                           remote_uri.ToUriString().c_str());
            return ER_INVALID_PARAMS;
        }

        // 使用 URI 的 size 计算 offset
        // ASSUMPTION: All items in a single batch must have the same `size`.
        // The formula `blkid * size` produces correct, non-overlapping offsets
        // only under this invariant.  The current calling convention guarantees
        // this (separate sessions for different spec sizes), but the SDK does
        // not enforce it explicitly.
        offset = item.blkid * item.size;

        for (auto &iov : local_buffer.iovs) {
            if (offset + iov.size > file_size) {
                KVCM_LOG_ERROR("Get failed, IOV size exceeds file size");
                return ER_INVALID_PARAMS;
            }

            if (!iov.ignore && iov.size > 0 && (iov.base || iov.type == MemoryType::TPU)) {
                if (iov.type == MemoryType::CPU) {
                    std::memcpy(iov.base, src + offset, iov.size);
                } else if (iov.type == MemoryType::GPU) {
#if defined(USING_CUDA)
                    exist_gpu_iov = true;
                    CHECK_CUDA_ERROR_RETURN(
                        cudaMemcpyAsync(iov.base, src + offset, iov.size, cudaMemcpyHostToDevice, cuda_stream_),
                        ER_CUDAMEMCPY_ERROR,
                        "cuda memcpy async fail");
#elif defined(USING_MUSA)
                    exist_gpu_iov = true;
                    CHECK_MUSA_ERROR_RETURN(
                        musaMemcpyAsync(iov.base, src + offset, iov.size, musaMemcpyHostToDevice, musa_stream_),
                        ER_CUDAMEMCPY_ERROR,
                        "musa memcpy async fail");
#endif
                } else if (iov.type == MemoryType::TPU) {
#if defined(USING_TPU)
                    exist_tpu_iov = true;
                    // Get: file → TPU. Create a new PJRT_Buffer* from file data.
                    // The bridge extracts the existing PJRT_Buffer* from iov.base
                    // (used only for validation), but we create a NEW buffer since
                    // BufferFromHost handles tile layout correctly.
                    tpu_temp_bufs.emplace_back(iov.size);
                    std::memcpy(tpu_temp_bufs.back().data(), src + offset, iov.size);
                    PJRT_Buffer* new_buf = nullptr;
                    PJRT_Event* event = nullptr;
                    auto buf_ec = tpu_client_.BufferFromHostAsync(
                        tpu_temp_bufs.back().data(), iov.size, new_buf, event);
                    if (buf_ec != ER_OK) {
                        KVCM_LOG_ERROR("TPU BufferFromHostAsync failed in Get");
                        for (auto* e : tpu_events) tpu_client_.DestroyEvent(e);
                        for (auto* b : tpu_new_bufs) tpu_client_.DestroyBuffer(b);
                        return buf_ec;
                    }
                    tpu_events.push_back(event);
                    tpu_new_bufs.push_back(new_buf);
                    tpu_iov_bases.push_back(&iov.base);
#endif
                }
            }
            offset += iov.size;
        }
    }

#if defined(USING_TPU)
    if (exist_tpu_iov) {
        auto wait_ec = tpu_client_.WaitEvents(tpu_events);
        if (wait_ec != ER_OK) {
            KVCM_LOG_ERROR("TPU WaitEvents failed in Get");
            for (auto* b : tpu_new_bufs) tpu_client_.DestroyBuffer(b);
            return wait_ec;
        }
        // All async H2D completed; assign new buffer handles to iov.base
        for (size_t j = 0; j < tpu_new_bufs.size(); ++j) {
            *tpu_iov_bases[j] = reinterpret_cast<void*>(tpu_new_bufs[j]);
        }
    }
#elif defined(USING_CUDA)
    if (exist_gpu_iov) {
        CHECK_CUDA_ERROR_RETURN(
            cudaStreamSynchronize(cuda_stream_), ER_CUDA_STREAM_SYNCHRONIZE_ERROR, "cuda stream synchronize fail");
    }
#elif defined(USING_MUSA)
    if (exist_gpu_iov) {
        CHECK_MUSA_ERROR_RETURN(
            musaStreamSynchronize(musa_stream_), ER_CUDA_STREAM_SYNCHRONIZE_ERROR, "musa stream synchronize fail");
    }
#endif

    return ER_OK;
} // namespace kv_cache_manager

ClientErrorCode LocalFileSdk::DoPut(const std::vector<DataStorageUri> &remote_uris, const BlockBuffers &local_buffers) {
    if (remote_uris.size() != local_buffers.size() || remote_uris.empty()) {
        KVCM_LOG_ERROR("Do Put failed, remote_uris size not equal to local_buffers size");
        return ER_INVALID_PARAMS;
    }

    std::string file_path = remote_uris[0].GetPath();

    size_t max_blkid = 0;
    size_t required_size = 0;
    std::vector<LocalFileItem> items;
    for (size_t i = 0; i < remote_uris.size(); ++i) {
        auto &remote_uri = remote_uris[i];
        if (remote_uri.GetPath().empty()) {
            KVCM_LOG_ERROR("Put failed, remote_uri is invalid");
            return ER_INVALID_PARAMS;
        }
        auto item = LocalFileItem::FromUri(remote_uri);

        // 防御性校验：URI 的 size 必须在允许的 spec 范围内
        bool size_valid = false;
        for (const auto &[spec_name, byte_size_per_block] : spec_byte_sizes_per_block_) {
            if (item.size == byte_size_per_block) {
                size_valid = true;
                break;
            }
        }
        if (!size_valid) {
            KVCM_LOG_ERROR("Put failed, URI size [%zu] not in allowed spec_byte_sizes_per_block, uri: %s",
                           item.size,
                           remote_uri.ToUriString().c_str());
            return ER_INVALID_PARAMS;
        }

        max_blkid = std::max(max_blkid, item.blkid);
        required_size = std::max(required_size, (max_blkid + 1) * item.size);
        items.push_back(item);
    }

    int fd = ::open(file_path.c_str(), O_RDWR, 0644);
    if (fd < 0) {
        KVCM_LOG_ERROR("Put failed, open file %s failed", file_path.c_str());
        return ER_FILE_IO_ERROR;
    }

    if (fallocate(fd, 0, 0, required_size) != 0) {
        KVCM_LOG_ERROR("Put failed, fallocate file %s failed", file_path.c_str());
        close(fd);
        return ER_FILE_IO_ERROR;
    }

    if (0 == required_size) {
        KVCM_LOG_ERROR("required size is 0, something is wrong");
        close(fd);
        return ER_INVALID_PARAMS;
    }

    void *file_mem = mmap(nullptr, required_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (file_mem == MAP_FAILED) {
        KVCM_LOG_ERROR("Put failed, mmap file %s failed", file_path.c_str());
        close(fd);
        return ER_FILE_IO_ERROR;
    }

    KVCM_LOG_DEBUG("Put file path [%s] size [%zu] block buffer [%s]",
                   file_path.c_str(),
                   required_size,
                   DebugStringUtil::ToString(local_buffers).c_str());
    MmapHelper helper(fd, file_mem, required_size);
#if defined(USING_CUDA)
    bool exist_gpu_iov = false;
    // If GPU supports direct pageable memory access, skip cudaHostRegister
    // This allows direct DMA transfer between GPU and mmap'd memory without pinning
    if (!support_pageable_memory_access_) {
        auto register_ec = helper.RegisterGpu(cudaHostRegisterDefault);
        if (register_ec != ER_OK) {
            return register_ec;
        }
    } else {
        KVCM_LOG_DEBUG("Skipping cudaHostRegister - GPU supports direct pageable memory access");
        helper.SkipRegistration(); // Mark as not registered since we don't need to
    }
#elif defined(USING_MUSA)
    bool exist_gpu_iov = false;
    if (!support_pageable_memory_access_) {
        auto register_ec = helper.RegisterGpu(musaHostRegisterDefault);
        if (register_ec != ER_OK) {
            return register_ec;
        }
    }
#elif defined(USING_TPU)
    [[maybe_unused]] bool exist_tpu_iov = false;
    {
        auto register_ec = helper.RegisterTpu(&tpu_client_);
        if (register_ec != ER_OK) {
            return register_ec;
        }
    }
    std::vector<PJRT_Event*> tpu_events;
    std::vector<std::vector<char>> tpu_temp_bufs;
    std::vector<std::pair<char*, size_t>> tpu_d2h_dsts;  // (dst in file, size)
#endif

    char *dst = static_cast<char *>(file_mem);
    // url assumed sorted by blkid
    // ASSUMPTION: same as DoGet — all items in a batch must share the same `size`.
    for (size_t i = 0; i < items.size(); ++i) {
        auto &item = items[i];
        auto &local_buffer = local_buffers[i];
        size_t offset = item.blkid * item.size;

        for (auto &iov : local_buffer.iovs) {
            if (offset + iov.size > required_size) {
                KVCM_LOG_ERROR(
                    "Put failed, IOV size [%zu] offset[%zu] exceeds file size [%zu]", iov.size, offset, required_size);
                return ER_INVALID_PARAMS;
            }

            if (!iov.ignore && iov.base && iov.size > 0) {
                if (iov.type == MemoryType::CPU) {
                    std::memcpy(dst + offset, iov.base, iov.size);
                } else if (iov.type == MemoryType::GPU) {
#if defined(USING_CUDA)
                    exist_gpu_iov = true;
                    CHECK_CUDA_ERROR_RETURN(
                        cudaMemcpyAsync(dst + offset, iov.base, iov.size, cudaMemcpyDeviceToHost, cuda_stream_),
                        ER_CUDAMEMCPY_ERROR,
                        "cuda memcpy async fail");
#elif defined(USING_MUSA)
                    exist_gpu_iov = true;
                    CHECK_MUSA_ERROR_RETURN(
                        musaMemcpyAsync(dst + offset, iov.base, iov.size, musaMemcpyDeviceToHost, musa_stream_),
                        ER_CUDAMEMCPY_ERROR,
                        "musa memcpy async fail");
#endif
                } else if (iov.type == MemoryType::TPU) {
#if defined(USING_TPU)
                    exist_tpu_iov = true;
                    // Put: TPU → file. Extract PJRT_Buffer* from iov.base via bridge,
                    // then use BufferToHostAsync (handles tile layout correctly).
                    auto extractor = GetPyArrayBufferExtractor();
                    if (!extractor) {
                        KVCM_LOG_ERROR("TPU PyArray buffer extractor not registered in Put");
                        for (auto* e : tpu_events) tpu_client_.DestroyEvent(e);
                        return ER_TPU_BUFFER_TRANSFER_ERROR;
                    }
                    PJRT_Buffer* c_buf = extractor(iov.base);
                    if (!c_buf) {
                        KVCM_LOG_ERROR("TPU PyArray buffer extraction returned null in Put");
                        for (auto* e : tpu_events) tpu_client_.DestroyEvent(e);
                        return ER_TPU_BUFFER_TRANSFER_ERROR;
                    }
                    // D2H: device memory → temp host buffer via regular API
                    tpu_temp_bufs.emplace_back(iov.size);
                    PJRT_Event* event = nullptr;
                    auto buf_ec = tpu_client_.BufferToHostAsync(
                        c_buf, tpu_temp_bufs.back().data(), iov.size, event);
                    if (buf_ec != ER_OK) {
                        KVCM_LOG_ERROR("TPU BufferToHostAsync failed in Put");
                        for (auto* e : tpu_events) tpu_client_.DestroyEvent(e);
                        return buf_ec;
                    }
                    tpu_events.push_back(event);
                    tpu_d2h_dsts.emplace_back(dst + offset, iov.size);
#endif
                }
            }
            offset += iov.size;
        }
    }
#if defined(USING_TPU)
    if (exist_tpu_iov) {
        auto wait_ec = tpu_client_.WaitEvents(tpu_events);
        if (wait_ec != ER_OK) {
            KVCM_LOG_ERROR("TPU WaitEvents failed in Put");
            return wait_ec;
        }
        // All async D2H completed; copy temp buffers to file
        for (size_t j = 0; j < tpu_d2h_dsts.size(); ++j) {
            std::memcpy(tpu_d2h_dsts[j].first, tpu_temp_bufs[j].data(), tpu_d2h_dsts[j].second);
        }
    }
#elif defined(USING_CUDA)
    if (exist_gpu_iov) {
        CHECK_CUDA_ERROR_RETURN(
            cudaStreamSynchronize(cuda_stream_), ER_CUDA_STREAM_SYNCHRONIZE_ERROR, "cuda stream synchronize fail");
    }
#elif defined(USING_MUSA)
    if (exist_gpu_iov) {
        CHECK_MUSA_ERROR_RETURN(
            musaStreamSynchronize(musa_stream_), ER_CUDA_STREAM_SYNCHRONIZE_ERROR, "musa stream synchronize fail");
    }
#endif
    if (msync(file_mem, required_size, MS_SYNC) != 0) {
        KVCM_LOG_WARN("Put msync failed for file %s", file_path.c_str());
    }

    return ER_OK;
}

} // namespace kv_cache_manager