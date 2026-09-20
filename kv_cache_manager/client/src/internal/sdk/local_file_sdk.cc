#include "kv_cache_manager/client/src/internal/sdk/local_file_sdk.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#if defined(USING_CUDA)
#include "kv_cache_manager/client/src/internal/sdk/cuda_util.h"
#elif defined(USING_MUSA)
#include "kv_cache_manager/client/src/internal/sdk/musa_util.h"
#endif
#include "kv_cache_manager/client/src/internal/sdk/deadline_util.h"
#include "kv_cache_manager/client/src/internal/util/debug_string_util.h"
#include "kv_cache_manager/common/logger.h"

namespace {

#if !defined(USING_CUDA) && !defined(USING_MUSA)
bool HasActiveGpuBuffer(const kv_cache_manager::BlockBuffers &buffers) {
    for (const auto &buffer : buffers) {
        for (const auto &iov : buffer.iovs) {
            if (!iov.ignore && iov.size > 0 && iov.type == kv_cache_manager::MemoryType::GPU) {
                return true;
            }
        }
    }
    return false;
}
#endif

bool IsExactObjectBuffer(const kv_cache_manager::BlockBuffer &buffer, const std::size_t expected_size) {
    if (buffer.iovs.empty()) {
        return false;
    }
    std::size_t total_size = 0;
    for (const auto &iov : buffer.iovs) {
        if (iov.ignore || iov.base == nullptr || iov.size == 0 ||
            (iov.type != kv_cache_manager::MemoryType::CPU && iov.type != kv_cache_manager::MemoryType::GPU) ||
            total_size > expected_size || iov.size > expected_size - total_size) {
            return false;
        }
        total_size += iov.size;
    }
    return total_size == expected_size;
}

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

    void SkipRegistration() {
#if defined(USING_CUDA) || defined(USING_MUSA)
        is_mem_registered = false;
#endif
    }

    bool Close() noexcept {
        bool success = true;
#if defined(USING_CUDA)
        if (is_mem_registered) {
            const cudaError_t error = cudaHostUnregister(file_mem_);
            if (error != cudaSuccess) {
                KVCM_LOG_WARN("cuda error [%d] [%s] | unregister host mem [%p] fail",
                              error,
                              cudaGetErrorString(error),
                              file_mem_);
                success = false;
            }
            is_mem_registered = false;
        }
#elif defined(USING_MUSA)
        if (is_mem_registered) {
            const musaError_t error = musaHostUnregister(file_mem_);
            if (error != musaSuccess) {
                KVCM_LOG_WARN("musa error [%d] [%s] | unregister host mem [%p] fail",
                              error,
                              musaGetErrorString(error),
                              file_mem_);
                success = false;
            }
            is_mem_registered = false;
        }
#endif
        if (file_mem_ != MAP_FAILED) {
            if (munmap(file_mem_, file_size_) != 0) {
                success = false;
            }
            file_mem_ = MAP_FAILED;
        }
        if (fd_ >= 0) {
            if (close(fd_) != 0) {
                success = false;
            }
            fd_ = -1;
        }
        return success;
    }

    ~MmapHelper() { (void)Close(); }

private:
    int fd_;
    void *file_mem_;
    size_t file_size_;
#if defined(USING_CUDA) || defined(USING_MUSA)
    bool is_mem_registered = false;
#endif
};

// A variable-size KVMeta path names one immutable allocation generation. If
// writing that newly created file fails before publication, remove only the
// inode created by this call. The identity check prevents an abort path from
// unlinking a replacement installed by another process in a shared namespace.
class ExclusiveFileCleanupGuard {
public:
    explicit ExclusiveFileCleanupGuard(const std::string &path) : path_(path) {}

    ~ExclusiveFileCleanupGuard() {
        if (!active_) {
            return;
        }
        struct stat current_stat;
        if (::lstat(path_.c_str(), &current_stat) != 0) {
            if (errno != ENOENT) {
                KVCM_LOG_WARN(
                    "KVMeta failed-write cleanup could not inspect file %s: %s", path_.c_str(), std::strerror(errno));
            }
            return;
        }
        if (!S_ISREG(current_stat.st_mode) || current_stat.st_dev != device_ || current_stat.st_ino != inode_) {
            KVCM_LOG_WARN("KVMeta failed-write cleanup retained a replaced path: %s", path_.c_str());
            return;
        }
        if (::unlink(path_.c_str()) != 0 && errno != ENOENT) {
            KVCM_LOG_WARN(
                "KVMeta failed-write cleanup could not unlink file %s: %s", path_.c_str(), std::strerror(errno));
        }
    }

    void Arm(const struct stat &created_stat) noexcept {
        device_ = created_stat.st_dev;
        inode_ = created_stat.st_ino;
        active_ = true;
    }
    void Release() noexcept { active_ = false; }

private:
    const std::string &path_;
    dev_t device_{0};
    ino_t inode_{0};
    bool active_{false};
};

// RAII guard：析构时若仍有在飞的 GPU async copy（cudaMemcpyAsync/musaMemcpyAsync 已入队），
// 先同步 stream 再返回。目的：hard 契约—— LocalFileSdk 返回后
// 不得再有异步 DMA 写 caller buffer（GPU 显存）。
// 必须覆盖所有提前返回路径（超时、既有的各种错误分支），因此用 RAII 而不是逐点调用
// （逐点调用极易漏掉某条 return 路径，重新打开静默数据损坏窗口）。
// 构造顺序要求：必须在 MmapHelper 之后构造（后构造先析构），保证 stream 同步先于
// MmapHelper 的 cudaHostUnregister + munmap 执行（否则 async copy 的源/目标地址
// 可能已被 unmap）。
class GpuStreamDrainGuard {
public:
#if defined(USING_CUDA)
    explicit GpuStreamDrainGuard(cudaStream_t stream, bool *has_inflight)
        : stream_(stream), has_inflight_(has_inflight) {}
    ~GpuStreamDrainGuard() {
        if (stream_ && has_inflight_ && *has_inflight_) {
            CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_),
                             "gpu stream synchronize fail on abort path, async copy may still write caller buffer");
        }
    }
#elif defined(USING_MUSA)
    explicit GpuStreamDrainGuard(musaStream_t stream, bool *has_inflight)
        : stream_(stream), has_inflight_(has_inflight) {}
    ~GpuStreamDrainGuard() {
        if (stream_ && has_inflight_ && *has_inflight_) {
            CHECK_MUSA_ERROR(musaStreamSynchronize(stream_),
                             "musa stream synchronize fail on abort path, async copy may still write caller buffer");
        }
    }
#else
    explicit GpuStreamDrainGuard(bool *has_inflight) { (void)has_inflight; }
    ~GpuStreamDrainGuard() = default;
#endif
    GpuStreamDrainGuard(const GpuStreamDrainGuard &) = delete;
    GpuStreamDrainGuard &operator=(const GpuStreamDrainGuard &) = delete;

private:
#if defined(USING_CUDA)
    cudaStream_t stream_;
    bool *has_inflight_;
#elif defined(USING_MUSA)
    musaStream_t stream_;
    bool *has_inflight_;
#endif
};

// 超时中止日志：定位是哪个 block、哪块 caller buffer 停在了半路。
void LogTimeoutAbort(const char *op,
                     size_t done_blocks,
                     size_t total_blocks,
                     size_t in_flight_block,
                     const kv_cache_manager::BlockBuffer &in_flight_buffer) {
    std::string buf_addrs;
    for (const auto &iov : in_flight_buffer.iovs) {
        char addr[32];
        std::snprintf(addr, sizeof(addr), "%p", iov.base);
        if (!buf_addrs.empty()) {
            buf_addrs += ",";
        }
        buf_addrs += addr;
    }
    KVCM_LOG_ERROR("local file sdk timeout abort: op=%s done_blocks=%zu/%zu in_flight_block=%zu "
                   "caller_buffer_addrs=[%s]",
                   op,
                   done_blocks,
                   total_blocks,
                   in_flight_block,
                   buf_addrs.c_str());
}

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
        CHECK_CUDA_ERROR_RETURN(cudaDeviceGetAttribute(&value, cudaDevAttrHostRegisterSupported, dev),
                                false,
                                "get cudaDevAttrHostRegisterSupported failed");
#elif defined(USING_MUSA)
        CHECK_MUSA_ERROR_RETURN(musaDeviceGetAttribute(&value, musaDevAttrHostRegisterSupported, dev),
                                false,
                                "get musaDevAttrHostRegisterSupported failed");
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
        CHECK_CUDA_ERROR_RETURN(cudaDeviceGetAttribute(&value, cudaDevAttrHostRegisterReadOnlySupported, dev),
                                false,
                                "get cudaDevAttrHostRegisterReadOnlySupported failed");
#elif defined(USING_MUSA)
        CHECK_MUSA_ERROR_RETURN(musaDeviceGetAttribute(&value, musaDevAttrHostRegisterReadOnlySupported, dev),
                                false,
                                "get musaDevAttrHostRegisterReadOnlySupported failed");
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
        CHECK_CUDA_ERROR_RETURN(cudaDeviceGetAttribute(&value, cudaDevAttrPageableMemoryAccess, dev),
                                false,
                                "get cudaDevAttrPageableMemoryAccess failed");
#elif defined(USING_MUSA)
        // MUSA equivalent - adjust if needed
        CHECK_MUSA_ERROR_RETURN(musaDeviceGetAttribute(&value, musaDevAttrPageableMemoryAccess, dev),
                                false,
                                "get musaDevAttrPageableMemoryAccess failed");
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
    variable_object_size_enabled_ = sdk_backend_config->variable_object_size_enabled();
    max_variable_object_bytes_ = sdk_backend_config->max_variable_object_bytes();
    if (variable_object_size_enabled_ && max_variable_object_bytes_ == 0) {
        KVCM_LOG_WARN("Init local file sdk failed, max variable object bytes is zero");
        return ER_INVALID_SDKBACKEND_CONFIG;
    }
    timeout_config_ = sdk_backend_config->timeout_config();
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
#endif
    return ER_OK;
}

SdkType LocalFileSdk::Type() { return SdkType::LOCAL_FILE; }

bool LocalFileSdk::IsAllowedObjectSize(std::size_t size) const {
    if (variable_object_size_enabled_) {
        return size > 0 && size <= max_variable_object_bytes_;
    }
    return std::any_of(spec_byte_sizes_per_block_.begin(), spec_byte_sizes_per_block_.end(), [size](const auto &entry) {
        return entry.second > 0 && size == static_cast<std::size_t>(entry.second);
    });
}

ClientErrorCode LocalFileSdk::Get(const std::vector<DataStorageUri> &remote_uris, const BlockBuffers &local_buffers) {
    if (remote_uris.size() != local_buffers.size() || (variable_object_size_enabled_ && remote_uris.empty())) {
        KVCM_LOG_ERROR("Get failed, URI/buffer count mismatches or exact-object request is empty");
        return ER_INVALID_PARAMS;
    }
#if !defined(USING_CUDA) && !defined(USING_MUSA)
    if (HasActiveGpuBuffer(local_buffers)) {
        KVCM_LOG_ERROR("Get failed, GPU buffer requires a CUDA or MUSA client build");
        return ER_UNSUPPORTED_MEMORY_TYPE;
    }
#endif
    // 静态预算：Init 时由 wrapper 注入，从自身任务起点起算 deadline。
    const int64_t deadline_ms = SteadyClockMs() + timeout_config_.get_timeout_ms();
    auto group_map = SplitByPath(remote_uris, local_buffers);
    size_t done_blocks = 0;
    for (const auto &group : group_map) {
        // 组级准入：deadline 已过则不再为后续组做
        // open/mmap 等准备工作，直接返回超时。
        if (DeadlineExpired(deadline_ms)) {
            LogTimeoutAbort(
                "get", done_blocks, remote_uris.size(), group.second.indices[0], group.second.local_buffers[0]);
            return ER_SDK_TIMEOUT;
        }
        auto ec = DoGet(group.second.remote_uris, group.second.local_buffers, deadline_ms);
        if (ec == ER_SDK_TIMEOUT) {
            // 透传超时错误码，供 wrapper 层归因（不要把超时吞成普通读错误）。
            return ER_SDK_TIMEOUT;
        }
        if (ec != ER_OK) {
            KVCM_LOG_ERROR("DoGet failed, errorcode: %d", ec);
            return ER_SDKREAD_ERROR;
        }
        done_blocks += group.second.remote_uris.size();
    }
    return ER_OK;
}

ClientErrorCode LocalFileSdk::Put(const std::vector<DataStorageUri> &remote_uris,
                                  const BlockBuffers &local_buffers,
                                  std::shared_ptr<std::vector<DataStorageUri>> actual_remote_uris) {
    // A stale URI is unsafe for an exact-object caller: an upper layer could
    // otherwise publish a prior allocation after this call fails before the
    // normal resize/rollback path. Keep legacy output behavior unchanged.
    if (actual_remote_uris && variable_object_size_enabled_) {
        actual_remote_uris->clear();
    }
    if (!actual_remote_uris || remote_uris.size() != local_buffers.size() ||
        (variable_object_size_enabled_ && remote_uris.empty())) {
        KVCM_LOG_ERROR("Put failed, output is null, URI/buffer count mismatches, or exact-object request is empty");
        return ER_INVALID_PARAMS;
    }
#if !defined(USING_CUDA) && !defined(USING_MUSA)
    if (HasActiveGpuBuffer(local_buffers)) {
        KVCM_LOG_ERROR("Put failed, GPU buffer requires a CUDA or MUSA client build");
        return ER_UNSUPPORTED_MEMORY_TYPE;
    }
#endif
    // 静态预算：Init 时由 wrapper 注入，从自身任务起点起算 deadline。
    const int64_t deadline_ms = SteadyClockMs() + timeout_config_.put_timeout_ms();
    // 预分配并按原始下标回填，保证同序契约：actual_remote_uris[i] 对应 remote_uris[i]。
    // SplitByPath 按 path 分组后迭代顺序是不确定的（unordered_map），不能按分组
    // 顺序 append 结果，否则交错输入（同 backend 多 path）会乱序。
    actual_remote_uris->resize(remote_uris.size());
    auto group_map = SplitByPath(remote_uris, local_buffers);
    size_t done_blocks = 0;
    for (const auto &group : group_map) {
        // 组级准入：deadline 已过则不再为后续组做 exists/Alloc/mmap 等准备工作。
        if (DeadlineExpired(deadline_ms)) {
            LogTimeoutAbort(
                "put", done_blocks, remote_uris.size(), group.second.indices[0], group.second.local_buffers[0]);
            if (variable_object_size_enabled_) {
                actual_remote_uris->clear();
            }
            return ER_SDK_TIMEOUT;
        }
        const std::string &file_path = group.first;
        const BlockGroup &block_group = group.second;
        std::vector<DataStorageUri> group_actual_uris;
        // Every KVMeta URI names one immutable allocation generation. Always
        // create and hold the same descriptor through write+fsync. A separate
        // create/close/reopen sequence would leave a replacement race. The
        // fixed-block path deliberately keeps its historical allocation flow.
        if (variable_object_size_enabled_) {
            group_actual_uris = block_group.remote_uris;
        } else {
            std::error_code exists_ec;
            const bool file_exists = std::filesystem::exists(file_path, exists_ec);
            if (exists_ec) {
                KVCM_LOG_ERROR(
                    "Put failed, cannot inspect file %s: %s", file_path.c_str(), exists_ec.message().c_str());
                return ER_SDKALLOC_ERROR;
            }
            if (!file_exists) {
                auto ec = Alloc(block_group.remote_uris, group_actual_uris);
                if (ec != ER_OK) {
                    KVCM_LOG_ERROR("Put failed, alloc failed, errorcode: %d", ec);
                    return ER_SDKALLOC_ERROR;
                }
                if (group_actual_uris.size() != block_group.indices.size()) {
                    KVCM_LOG_ERROR("Put failed, alloc returned %zu uris but group has %zu blocks, path: %s",
                                   group_actual_uris.size(),
                                   block_group.indices.size(),
                                   file_path.c_str());
                    return ER_SDKALLOC_ERROR;
                }
            } else {
                group_actual_uris = block_group.remote_uris;
            }
        }
        // Preserve the fixed-block output timing. KVMeta publishes its exact
        // URI only after the exclusive write has completed successfully.
        if (!variable_object_size_enabled_) {
            for (size_t k = 0; k < block_group.indices.size(); ++k) {
                (*actual_remote_uris)[block_group.indices[k]] = group_actual_uris[k];
            }
        }
        auto ec = DoPut(block_group.remote_uris, block_group.local_buffers, deadline_ms);
        if (ec == ER_SDK_TIMEOUT) {
            if (variable_object_size_enabled_) {
                actual_remote_uris->clear();
            }
            // 透传超时错误码，供 wrapper 层归因（不要把超时吞成普通写错误）。
            return ER_SDK_TIMEOUT;
        }
        if (ec != ER_OK) {
            if (variable_object_size_enabled_) {
                actual_remote_uris->clear();
            }
            KVCM_LOG_ERROR("Put failed, DoPut failed, errorcode: %d", ec);
            return ER_SDKWRITE_ERROR;
        }
        // 保序回填：indices[k] 是该组第 k 个元素在原始入参中的下标。
        if (variable_object_size_enabled_) {
            for (size_t k = 0; k < block_group.indices.size(); ++k) {
                (*actual_remote_uris)[block_group.indices[k]] = group_actual_uris[k];
            }
        }
        done_blocks += block_group.remote_uris.size();
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

ClientErrorCode LocalFileSdk::DoGet(const std::vector<DataStorageUri> &remote_uris,
                                    const BlockBuffers &local_buffers,
                                    int64_t deadline_ms) {
    if (remote_uris.size() != local_buffers.size() || remote_uris.empty()) {
        KVCM_LOG_ERROR("Do Get failed, remote_uris size not equal to local_buffers size");
        return ER_INVALID_PARAMS;
    }

    std::string file_path = remote_uris[0].GetPath();
    if (variable_object_size_enabled_ && remote_uris.size() != 1) {
        KVCM_LOG_ERROR("Get failed, KVMeta local-file allocation must contain exactly one block");
        return ER_INVALID_PARAMS;
    }
    std::error_code exists_ec;
    const bool file_exists = std::filesystem::exists(file_path, exists_ec);
    if (exists_ec) {
        KVCM_LOG_ERROR("Get failed, cannot inspect file %s: %s", file_path.c_str(), exists_ec.message().c_str());
        return ER_FILE_IO_ERROR;
    }
    if (!file_exists) {
        KVCM_LOG_WARN("Get failed, file %s is not exist", file_path.c_str());
        return ER_FILE_IO_ERROR;
    }
    const int open_flags = variable_object_size_enabled_ ? O_RDONLY | O_CLOEXEC | O_NOFOLLOW : O_RDONLY;
    int fd = ::open(file_path.c_str(), open_flags);
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
    if (st.st_size < 0 || static_cast<uintmax_t>(st.st_size) > std::numeric_limits<size_t>::max() ||
        (variable_object_size_enabled_ && !S_ISREG(st.st_mode))) {
        KVCM_LOG_ERROR("Get failed, KVMeta path is not a regular file or has an invalid size: %s", file_path.c_str());
        close(fd);
        return ER_FILE_IO_ERROR;
    }
    size_t file_size = static_cast<size_t>(st.st_size);
    if (variable_object_size_enabled_) {
        for (const auto &remote_uri : remote_uris) {
            const auto item = LocalFileItem::FromUri(remote_uri);
            if (item.blkid != 0 || !IsAllowedObjectSize(item.size) || item.size != file_size ||
                !IsExactObjectBuffer(local_buffers.front(), item.size)) {
                KVCM_LOG_ERROR("Get failed, KVMeta file size or singleton block identity mismatches uri: %s",
                               remote_uri.ToUriString().c_str());
                close(fd);
                return ER_FILE_IO_ERROR;
            }
        }
    }
    KVCM_LOG_DEBUG("Get file path [%s] size [%zu] block buffer [%s]",
                   file_path.c_str(),
                   file_size,
                   DebugStringUtil::ToString(local_buffers).c_str());
    int prot = PROT_READ;
    if (!support_register_readonly_) {
        prot |= PROT_WRITE;
    }
    void *file_mem = mmap(nullptr, file_size, prot, MAP_PRIVATE, fd, 0);
    if (file_mem == MAP_FAILED) {
        KVCM_LOG_ERROR("Get failed, mmap file %s failed", file_path.c_str());
        close(fd);
        return ER_FILE_IO_ERROR;
    }

    MmapHelper helper(fd, file_mem, file_size);
    // GpuStreamDrainGuard 必须在 MmapHelper 之后构造（后构造先析构）：任何提前返回
    // 路径上，guard 析构会先同步 GPU stream，再执行 MmapHelper 的 unregister/munmap。
    bool gpu_copy_enqueued = false;
#if defined(USING_CUDA)
    GpuStreamDrainGuard gpu_drain(cuda_stream_, &gpu_copy_enqueued);
    // If GPU supports direct pageable memory access, skip cudaHostRegister
    // This allows direct DMA transfer between GPU and mmap'd memory without pinning
    if (!support_pageable_memory_access_) {
        auto register_ec =
            helper.RegisterGpu(support_register_readonly_ ? cudaHostRegisterReadOnly : cudaHostRegisterDefault);
        if (register_ec != ER_OK) {
            // 此时尚无 async copy 入队，guard 为空操作；helper 析构 unregister/munmap 安全。
            return register_ec;
        }
    } else {
        KVCM_LOG_DEBUG("Skipping cudaHostRegister - GPU supports direct pageable memory access");
        helper.SkipRegistration(); // Mark as not registered since we don't need to
    }
#elif defined(USING_MUSA)
    GpuStreamDrainGuard gpu_drain(musa_stream_, &gpu_copy_enqueued);
    if (!support_pageable_memory_access_) {
        auto register_ec =
            helper.RegisterGpu(support_register_readonly_ ? musaHostRegisterReadOnly : musaHostRegisterDefault);
        if (register_ec != ER_OK) {
            // 此时尚无 async copy 入队，guard 为空操作；helper 析构 unregister/munmap 安全。
            return register_ec;
        }
    }
#else
    GpuStreamDrainGuard gpu_drain(&gpu_copy_enqueued);
#endif

    size_t offset = 0;
    char *src = static_cast<char *>(file_mem);
    // asume that url is sorted by blkid
    for (size_t i = 0; i < remote_uris.size(); ++i) {
        // 逐 block 准入：deadline 已过则停止搬运，不再发起
        // 后续 memcpy/async copy。若已有 GPU async copy 入队，guard 析构会在返回前同步。
        if (DeadlineExpired(deadline_ms)) {
            LogTimeoutAbort("get",
                            /*done=*/i,
                            remote_uris.size(),
                            /*in_flight=*/i,
                            local_buffers[i]);
            return ER_SDK_TIMEOUT;
        }
        auto &remote_uri = remote_uris[i];
        auto &local_buffer = local_buffers[i];
        if (remote_uri.GetPath().empty()) {
            KVCM_LOG_ERROR("Get failed, remote_uri is invalid");
            return ER_INVALID_PARAMS;
        }
        auto item = LocalFileItem::FromUri(remote_uri);

        // 防御性校验：URI 的 size 必须在允许的 spec 范围内
        if (!IsAllowedObjectSize(item.size)) {
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

            if (!iov.ignore && iov.base && iov.size > 0) {
                if (iov.type == MemoryType::CPU) {
                    std::memcpy(iov.base, src + offset, iov.size);
                } else if (iov.type == MemoryType::GPU) {
                    // 先置位再入队：只要 async copy 入队成功，flag 必为 true；
                    // 即使入队失败提前返回，guard 同步空闲 stream 也无害。
                    gpu_copy_enqueued = true;
#if defined(USING_CUDA)
                    CHECK_CUDA_ERROR_RETURN(
                        cudaMemcpyAsync(iov.base, src + offset, iov.size, cudaMemcpyHostToDevice, cuda_stream_),
                        ER_CUDAMEMCPY_ERROR,
                        "cuda memcpy async fail");
#elif defined(USING_MUSA)
                    CHECK_MUSA_ERROR_RETURN(
                        musaMemcpyAsync(iov.base, src + offset, iov.size, musaMemcpyHostToDevice, musa_stream_),
                        ER_CUDAMEMCPY_ERROR,
                        "musa memcpy async fail");
#endif
                }
            }
            offset += iov.size;
        }
    }

#if defined(USING_CUDA)
    if (gpu_copy_enqueued) {
        CHECK_CUDA_ERROR_RETURN(
            cudaStreamSynchronize(cuda_stream_), ER_CUDA_STREAM_SYNCHRONIZE_ERROR, "cuda stream synchronize fail");
        // 已同步完成，guard 析构无需重复同步。
        gpu_copy_enqueued = false;
    }
#elif defined(USING_MUSA)
    if (gpu_copy_enqueued) {
        CHECK_MUSA_ERROR_RETURN(
            musaStreamSynchronize(musa_stream_), ER_CUDA_STREAM_SYNCHRONIZE_ERROR, "musa stream synchronize fail");
        // 已同步完成，guard 析构无需重复同步。
        gpu_copy_enqueued = false;
    }
#endif

    return ER_OK;
}

ClientErrorCode LocalFileSdk::DoPut(const std::vector<DataStorageUri> &remote_uris,
                                    const BlockBuffers &local_buffers,
                                    int64_t deadline_ms) {
    if (remote_uris.size() != local_buffers.size() || remote_uris.empty()) {
        KVCM_LOG_ERROR("Do Put failed, remote_uris size not equal to local_buffers size");
        return ER_INVALID_PARAMS;
    }

    std::string file_path = remote_uris[0].GetPath();

    size_t required_size = 0;
    std::vector<LocalFileItem> items;
    items.reserve(remote_uris.size());
    for (size_t i = 0; i < remote_uris.size(); ++i) {
        auto &remote_uri = remote_uris[i];
        if (remote_uri.GetPath().empty()) {
            KVCM_LOG_ERROR("Put failed, remote_uri is invalid");
            return ER_INVALID_PARAMS;
        }
        auto item = LocalFileItem::FromUri(remote_uri);

        // 防御性校验：URI 的 size 必须在允许的 spec 范围内
        if (!IsAllowedObjectSize(item.size)) {
            KVCM_LOG_ERROR("Put failed, URI size [%zu] not in allowed spec_byte_sizes_per_block, uri: %s",
                           item.size,
                           remote_uri.ToUriString().c_str());
            return ER_INVALID_PARAMS;
        }
        if (variable_object_size_enabled_ && (remote_uris.size() != 1 || item.blkid != 0)) {
            KVCM_LOG_ERROR("Put failed, KVMeta local-file allocation must contain exactly one block at offset zero");
            return ER_INVALID_PARAMS;
        }
        if (variable_object_size_enabled_ && !IsExactObjectBuffer(local_buffers[i], item.size)) {
            KVCM_LOG_ERROR("Put failed, KVMeta local buffer does not exactly match the object size");
            return ER_INVALID_LOCAL_BUFFERS;
        }
        if (item.blkid > std::numeric_limits<size_t>::max() - 1 ||
            item.size > std::numeric_limits<size_t>::max() / (static_cast<size_t>(item.blkid) + 1)) {
            KVCM_LOG_ERROR("Put failed, URI block range overflows addressable file size: %s",
                           remote_uri.ToUriString().c_str());
            return ER_INVALID_PARAMS;
        }
        required_size = std::max(required_size, (static_cast<size_t>(item.blkid) + 1) * static_cast<size_t>(item.size));
        items.push_back(item);
    }

    if (required_size == 0 || required_size > static_cast<std::uintmax_t>(std::numeric_limits<off_t>::max())) {
        KVCM_LOG_ERROR("Put failed, required file size is invalid: %zu", required_size);
        return ER_INVALID_PARAMS;
    }
    if (variable_object_size_enabled_) {
        std::error_code directory_ec;
        std::filesystem::create_directories(std::filesystem::path(file_path).parent_path(), directory_ec);
        if (directory_ec) {
            KVCM_LOG_ERROR("Put failed, cannot create KVMeta parent directories for %s: %s",
                           file_path.c_str(),
                           directory_ec.message().c_str());
            return ER_FILE_IO_ERROR;
        }
    }
    const int open_flags = variable_object_size_enabled_ ? O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW : O_RDWR;
    int fd = ::open(file_path.c_str(), open_flags, 0644);
    if (fd < 0) {
        KVCM_LOG_ERROR("Put failed, open file %s failed: %s", file_path.c_str(), std::strerror(errno));
        return ER_FILE_IO_ERROR;
    }
    ExclusiveFileCleanupGuard failed_write_cleanup(file_path);
    if (variable_object_size_enabled_) {
        struct stat opened_file_stat;
        if (fstat(fd, &opened_file_stat) != 0 || !S_ISREG(opened_file_stat.st_mode)) {
            KVCM_LOG_ERROR("Put failed, KVMeta path is not a regular file: %s", file_path.c_str());
            close(fd);
            return ER_FILE_IO_ERROR;
        }
        failed_write_cleanup.Arm(opened_file_stat);
    }

    if (fallocate(fd, 0, 0, static_cast<off_t>(required_size)) != 0) {
        KVCM_LOG_ERROR("Put failed, fallocate file %s failed", file_path.c_str());
        close(fd);
        return ER_FILE_IO_ERROR;
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
    // GpuStreamDrainGuard 必须在 MmapHelper 之后构造（后构造先析构）：任何提前返回
    // 路径上，guard 析构会先同步 GPU stream，再执行 MmapHelper 的 unregister/munmap。
    bool gpu_copy_enqueued = false;
#if defined(USING_CUDA)
    GpuStreamDrainGuard gpu_drain(cuda_stream_, &gpu_copy_enqueued);
    // If GPU supports direct pageable memory access, skip cudaHostRegister
    // This allows direct DMA transfer between GPU and mmap'd memory without pinning
    if (!support_pageable_memory_access_) {
        auto register_ec = helper.RegisterGpu(cudaHostRegisterDefault);
        if (register_ec != ER_OK) {
            // 此时尚无 async copy 入队，guard 为空操作；helper 析构 unregister/munmap 安全。
            return register_ec;
        }
    } else {
        KVCM_LOG_DEBUG("Skipping cudaHostRegister - GPU supports direct pageable memory access");
        helper.SkipRegistration(); // Mark as not registered since we don't need to
    }
#elif defined(USING_MUSA)
    GpuStreamDrainGuard gpu_drain(musa_stream_, &gpu_copy_enqueued);
    if (!support_pageable_memory_access_) {
        auto register_ec = helper.RegisterGpu(musaHostRegisterDefault);
        if (register_ec != ER_OK) {
            // 此时尚无 async copy 入队，guard 为空操作；helper 析构 unregister/munmap 安全。
            return register_ec;
        }
    }
#else
    GpuStreamDrainGuard gpu_drain(&gpu_copy_enqueued);
#endif

    char *dst = static_cast<char *>(file_mem);
    // url assumed sorted by blkid
    // ASSUMPTION: same as DoGet — all items in a batch must share the same `size`.
    for (size_t i = 0; i < items.size(); ++i) {
        // 逐 block 准入：deadline 已过则停止搬运，不再发起
        // 后续 memcpy/async copy。若已有 GPU async copy 入队，guard 析构会在返回前同步。
        if (DeadlineExpired(deadline_ms)) {
            LogTimeoutAbort("put",
                            /*done=*/i,
                            items.size(),
                            /*in_flight=*/i,
                            local_buffers[i]);
            return ER_SDK_TIMEOUT;
        }
        auto &item = items[i];
        auto &local_buffer = local_buffers[i];
        size_t offset = static_cast<size_t>(item.blkid) * item.size;

        for (auto &iov : local_buffer.iovs) {
            if (offset > required_size || iov.size > required_size - offset) {
                KVCM_LOG_ERROR(
                    "Put failed, IOV size [%zu] offset[%zu] exceeds file size [%zu]", iov.size, offset, required_size);
                return ER_INVALID_PARAMS;
            }

            if (!iov.ignore && iov.base && iov.size > 0) {
                if (iov.type == MemoryType::CPU) {
                    std::memcpy(dst + offset, iov.base, iov.size);
                } else if (iov.type == MemoryType::GPU) {
                    // 先置位再入队：只要 async copy 入队成功，flag 必为 true；
                    // 即使入队失败提前返回，guard 同步空闲 stream 也无害。
                    gpu_copy_enqueued = true;
#if defined(USING_CUDA)
                    CHECK_CUDA_ERROR_RETURN(
                        cudaMemcpyAsync(dst + offset, iov.base, iov.size, cudaMemcpyDeviceToHost, cuda_stream_),
                        ER_CUDAMEMCPY_ERROR,
                        "cuda memcpy async fail");
#elif defined(USING_MUSA)
                    CHECK_MUSA_ERROR_RETURN(
                        musaMemcpyAsync(dst + offset, iov.base, iov.size, musaMemcpyDeviceToHost, musa_stream_),
                        ER_CUDAMEMCPY_ERROR,
                        "musa memcpy async fail");
#endif
                }
            }
            offset += iov.size;
        }
    }
#if defined(USING_CUDA)
    if (gpu_copy_enqueued) {
        CHECK_CUDA_ERROR_RETURN(
            cudaStreamSynchronize(cuda_stream_), ER_CUDA_STREAM_SYNCHRONIZE_ERROR, "cuda stream synchronize fail");
        // 已同步完成，guard 析构无需重复同步。
        gpu_copy_enqueued = false;
    }
#elif defined(USING_MUSA)
    if (gpu_copy_enqueued) {
        CHECK_MUSA_ERROR_RETURN(
            musaStreamSynchronize(musa_stream_), ER_CUDA_STREAM_SYNCHRONIZE_ERROR, "musa stream synchronize fail");
        // 已同步完成，guard 析构无需重复同步。
        gpu_copy_enqueued = false;
    }
#endif
    if (!SyncMappedFile(file_mem, required_size)) {
        if (variable_object_size_enabled_) {
            // KVMeta publishes the location only after Put succeeds. Returning
            // success here would make an unflushed NFS object durable metadata
            // and could leave a poisoned cache hit after process or host loss.
            // Fail closed so the write session is aborted and GC owns cleanup.
            KVCM_LOG_ERROR("KVMeta Put msync failed for file %s", file_path.c_str());
            return ER_FILE_IO_ERROR;
        }
        // Preserve the established fixed-block behavior. KVMeta enables the
        // strict branch through its isolated variable-object SDK config.
        KVCM_LOG_WARN("Put msync failed for file %s", file_path.c_str());
    }
    if (variable_object_size_enabled_ && !SyncFileDescriptor(fd)) {
        // msync covers the mapped pages, while fsync also persists inode
        // metadata such as the fallocated file size. Do not publish a cache hit
        // when either half of that durability barrier is uncertain.
        KVCM_LOG_ERROR("KVMeta Put fsync failed for file %s", file_path.c_str());
        return ER_FILE_IO_ERROR;
    }
    if (variable_object_size_enabled_) {
        if (!helper.Close()) {
            KVCM_LOG_ERROR("KVMeta Put could not close the synchronized file %s", file_path.c_str());
            return ER_FILE_IO_ERROR;
        }
        failed_write_cleanup.Release();
    }

    return ER_OK;
}

bool LocalFileSdk::SyncMappedFile(void *address, const std::size_t length) const {
    return msync(address, length, MS_SYNC) == 0;
}

bool LocalFileSdk::SyncFileDescriptor(const int fd) const { return fsync(fd) == 0; }

} // namespace kv_cache_manager
