#include "kv_cache_manager/client/src/internal/sdk/local_file_sdk.h"

#include <cstdio>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
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

    ~MmapHelper() {
#if defined(USING_CUDA)
        if (is_mem_registered) {
            CHECK_CUDA_ERROR(cudaHostUnregister(file_mem_), "unregister host mem [%p] fail", file_mem_);
        }
#elif defined(USING_MUSA)
        if (is_mem_registered) {
            CHECK_MUSA_ERROR(musaHostUnregister(file_mem_), "unregister host mem [%p] fail", file_mem_);
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
#endif
};

// RAII guard：析构时若仍有在飞的 GPU async copy（cudaMemcpyAsync/musaMemcpyAsync 已入队），
// 先同步 stream 再返回。目的：hard 契约（docs/design/client_sdk_io_contract.md）—— LocalFileSdk 返回后
// 不得再有异步 DMA 写 caller buffer（GPU 显存）。
//
// 必须覆盖所有提前返回路径（超时、既有的各种错误分支），因此用 RAII 而不是逐点调用
// （逐点调用极易漏掉某条 return 路径，重新打开静默数据损坏窗口）。
//
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

ClientErrorCode LocalFileSdk::Get(const std::vector<DataStorageUri> &remote_uris,
                                  const BlockBuffers &local_buffers,
                                  int64_t deadline_ms) {
    if (remote_uris.size() != local_buffers.size()) {
        KVCM_LOG_ERROR("Get failed, remote_uris size not equal to local_buffers size");
        return ER_INVALID_PARAMS;
    }
    auto group_map = SplitByPath(remote_uris, local_buffers);
    size_t done_blocks = 0;
    for (const auto &group : group_map) {
        // 组级准入（契约 §2 第2条）：deadline 已过则不再为后续组做
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
                                  std::shared_ptr<std::vector<DataStorageUri>> actual_remote_uris,
                                  int64_t deadline_ms) {
    if (remote_uris.size() != local_buffers.size()) {
        KVCM_LOG_ERROR("Put failed, remote_uris size not equal to local_buffers size");
        return ER_INVALID_PARAMS;
    }
    // 保序契约（契约 §6）：actual_remote_uris[i] 必须对应 remote_uris[i]。
    // 先按总大小 resize，再按 BlockGroup::indices 回填原位；禁止 clear() 后 append
    // （那会依赖 unordered_map 的迭代序，交错多 path 输入必然错位）。
    actual_remote_uris->resize(remote_uris.size());
    auto group_map = SplitByPath(remote_uris, local_buffers);
    size_t done_blocks = 0;
    for (const auto &group : group_map) {
        // 组级准入：deadline 已过则不再为后续组做 exists/Alloc/mmap 等准备工作。
        if (DeadlineExpired(deadline_ms)) {
            LogTimeoutAbort(
                "put", done_blocks, remote_uris.size(), group.second.indices[0], group.second.local_buffers[0]);
            return ER_SDK_TIMEOUT;
        }
        std::string file_path = group.first;
        std::vector<DataStorageUri> group_actual_uris;
        if (!std::filesystem::exists(file_path)) {
            auto ec = Alloc(group.second.remote_uris, group_actual_uris);
            if (ec != ER_OK) {
                KVCM_LOG_ERROR("Put failed, alloc failed, errorcode: %d", ec);
                return ER_SDKALLOC_ERROR;
            }
            if (group_actual_uris.size() != group.second.indices.size()) {
                KVCM_LOG_ERROR("Put failed, alloc returned %zu uris but group has %zu blocks, path: %s",
                               group_actual_uris.size(),
                               group.second.indices.size(),
                               file_path.c_str());
                return ER_SDKALLOC_ERROR;
            }
        } else {
            // 文件已存在：实际位置就是原 uri，但仍须按 indices 回填（保序契约）。
            group_actual_uris = group.second.remote_uris;
        }
        // 保序回填：indices[k] 是该组第 k 个元素在原始入参中的下标（下标是 block 的唯一身份）。
        // 禁止 append 到 actual_remote_uris 末尾 —— 多 path 交错输入会错位（F3 保序）。
        for (size_t k = 0; k < group.second.indices.size(); ++k) {
            (*actual_remote_uris)[group.second.indices[k]] = group_actual_uris[k];
        }
        auto ec = DoPut(group.second.remote_uris, group.second.local_buffers, deadline_ms);
        if (ec == ER_SDK_TIMEOUT) {
            // 透传超时错误码，供 wrapper 层归因（不要把超时吞成普通写错误）。
            return ER_SDK_TIMEOUT;
        }
        if (ec != ER_OK) {
            KVCM_LOG_ERROR("Put failed, DoPut failed, errorcode: %d", ec);
            return ER_SDKWRITE_ERROR;
        }
        done_blocks += group.second.remote_uris.size();
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
        // 逐 block 准入（契约 §2 第2条）：deadline 已过则停止搬运，不再发起
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
} // namespace kv_cache_manager

ClientErrorCode LocalFileSdk::DoPut(const std::vector<DataStorageUri> &remote_uris,
                                    const BlockBuffers &local_buffers,
                                    int64_t deadline_ms) {
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
        // 逐 block 准入（契约 §2 第2条）：deadline 已过则停止搬运，不再发起
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
    if (msync(file_mem, required_size, MS_SYNC) != 0) {
        KVCM_LOG_WARN("Put msync failed for file %s", file_path.c_str());
    }

    return ER_OK;
}

} // namespace kv_cache_manager