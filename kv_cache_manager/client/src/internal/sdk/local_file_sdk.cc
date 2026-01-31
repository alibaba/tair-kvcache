#include "kv_cache_manager/client/src/internal/sdk/local_file_sdk.h"

#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#ifdef USING_CUDA
#include "kv_cache_manager/client/src/internal/sdk/cuda_util.h"
#endif
#include "kv_cache_manager/client/src/internal/util/debug_string_util.h"
#include "kv_cache_manager/common/logger.h"

namespace {
class MmapHelper {
public:
    MmapHelper(int fd, void *file_mem, size_t file_size) : fd_(fd), file_mem_(file_mem), file_size_(file_size) {}
    kv_cache_manager::ClientErrorCode RegisterCuda(unsigned int register_flag) {
#ifdef USING_CUDA
        CHECK_CUDA_ERROR_RETURN(cudaHostRegister(file_mem_, file_size_, register_flag),
                                kv_cache_manager::ER_CUDA_HOST_REGISTER_ERROR,
                                "register host mem [%p] fail, size: %zu, register_flag: %u",
                                file_mem_,
                                file_size_,
                                register_flag);
        is_mem_registered = true;
#endif
        return kv_cache_manager::ER_OK;
    }

    ~MmapHelper() {
#ifdef USING_CUDA
        if (is_mem_registered) {
            CHECK_CUDA_ERROR(cudaHostUnregister(file_mem_), "unregister host mem [%p] fail", file_mem_);
        }
#endif
        munmap(file_mem_, file_size_);
        close(fd_);
    }

private:
    int fd_;
    void *file_mem_;
    size_t file_size_;
#ifdef USING_CUDA
    bool is_mem_registered = false;
#endif
};
} // namespace

namespace kv_cache_manager {

LocalFileSdk::~LocalFileSdk() {
#ifdef USING_CUDA
    if (cuda_stream_) {
        CHECK_CUDA_ERROR(cudaStreamDestroy(cuda_stream_), "destroy cuda stream error");
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
    byte_size_per_block_ = sdk_backend_config->byte_size_per_block();
    if (byte_size_per_block_ <= 0) {
        KVCM_LOG_WARN("Init local file sdk failed, invalid byte_size_per_block [%ld]", byte_size_per_block_);
        return ER_INVALID_SDKBACKEND_CONFIG;
    }
#ifdef USING_CUDA
    CHECK_CUDA_ERROR_RETURN(cudaStreamCreateWithFlags(&cuda_stream_, cudaStreamNonBlocking),
                            ER_CUDA_STREAM_CREATE_ERROR,
                            "Init local file sdk failed");
#endif
    return ER_OK;
}

SdkType LocalFileSdk::Type() { return SdkType::LOCAL_FILE; }

ClientErrorCode LocalFileSdk::Get(const std::vector<DataStorageUri> &remote_uris, const BlockBuffers &local_buffers) {
    if (remote_uris.size() != local_buffers.size()) {
        KVCM_LOG_ERROR("Get failed, remote_uris size not equal to local_buffers size");
        return ER_INVALID_PARAMS;
    }
    auto group_map = SplitByPath(remote_uris, local_buffers);
    for (const auto &group : group_map) {
        auto ec = DoGet(group.second.remote_uris, group.second.local_buffers);
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

ClientErrorCode LocalFileSdk::DoGet(const std::vector<DataStorageUri> &remote_uris, const BlockBuffers &local_buffers) {
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
    void *file_mem = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (file_mem == MAP_FAILED) {
        KVCM_LOG_ERROR("Get failed, mmap file %s failed", file_path.c_str());
        close(fd);
        return ER_FILE_IO_ERROR;
    }
    MmapHelper helper(fd, file_mem, file_size);
#ifdef USING_CUDA
    auto register_ec = helper.RegisterCuda(cudaHostRegisterReadOnly);
    if (register_ec != ER_OK) {
        return register_ec;
    }
    bool exist_gpu_iov = false;
#endif

    size_t offset = 0;
    char *src = static_cast<char *>(file_mem);
    // asume that url is sorted by blkid
    for (size_t i = 0; i < remote_uris.size(); ++i) {
        auto &remote_uri = remote_uris[i];
        auto &local_buffer = local_buffers[i];
        if (remote_uri.GetPath().empty()) {
            KVCM_LOG_ERROR("Get failed, remote_uri is invalid");
            return ER_INVALID_PARAMS;
        }
        auto item = LocalFileItem::FromUri(remote_uri);
        if (byte_size_per_block_ != item.size) {
            KVCM_LOG_ERROR("Get failed, byte_size_per_block_ [%ld] not equal to uri size [%zu], origin uri: [%s]",
                           byte_size_per_block_,
                           item.size,
                           remote_uri.ToUriString().c_str());
            return ER_INVALID_PARAMS;
        }
        offset = item.blkid * byte_size_per_block_;

        for (auto &iov : local_buffer.iovs) {
            if (offset + iov.size > file_size) {
                KVCM_LOG_ERROR("Get failed, IOV size exceeds file size");
                return ER_INVALID_PARAMS;
            }

            if (!iov.ignore && iov.base && iov.size > 0) {
                if (iov.type == MemoryType::CPU) {
                    std::memcpy(iov.base, src + offset, iov.size);
                } else if (iov.type == MemoryType::GPU) {
#ifdef USING_CUDA
                    exist_gpu_iov = true;
                    CHECK_CUDA_ERROR_RETURN(
                        cudaMemcpyAsync(iov.base, src + offset, iov.size, cudaMemcpyHostToDevice, cuda_stream_),
                        ER_CUDAMEMCPY_ERROR,
                        "cuda memcpy async fail");
#endif
                }
            }
            offset += iov.size;
        }
    }

#ifdef USING_CUDA
    if (exist_gpu_iov) {
        CHECK_CUDA_ERROR_RETURN(
            cudaStreamSynchronize(cuda_stream_), ER_CUDA_STREAM_SYNCHRONIZE_ERROR, "cuda stream synchronize fail");
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
        if (byte_size_per_block_ != item.size) {
            KVCM_LOG_ERROR("Get failed, byte_size_per_block_ [%ld] not equal to uri size [%zu], origin uri: [%s]",
                           byte_size_per_block_,
                           item.size,
                           remote_uri.ToUriString().c_str());
            return ER_INVALID_PARAMS;
        }
        max_blkid = std::max(max_blkid, item.blkid);
        required_size = std::max(required_size, (max_blkid + 1) * byte_size_per_block_);
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
#ifdef USING_CUDA
    auto register_ec = helper.RegisterCuda(cudaHostRegisterDefault);
    if (register_ec != ER_OK) {
        return register_ec;
    }
    bool exist_gpu_iov = false;
#endif

    char *dst = static_cast<char *>(file_mem);
    // url assumed sorted by blkid
    for (size_t i = 0; i < items.size(); ++i) {
        auto &item = items[i];
        auto &local_buffer = local_buffers[i];
        size_t offset = item.blkid * byte_size_per_block_;

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
#ifdef USING_CUDA
                    exist_gpu_iov = true;
                    CHECK_CUDA_ERROR_RETURN(
                        cudaMemcpyAsync(dst + offset, iov.base, iov.size, cudaMemcpyDeviceToHost, cuda_stream_),
                        ER_CUDAMEMCPY_ERROR,
                        "cuda memcpy async fail");
#endif
                }
            }
            offset += iov.size;
        }
    }
#ifdef USING_CUDA
    if (exist_gpu_iov) {
        CHECK_CUDA_ERROR_RETURN(
            cudaStreamSynchronize(cuda_stream_), ER_CUDA_STREAM_SYNCHRONIZE_ERROR, "cuda stream synchronize fail");
    }
#endif
    if (msync(file_mem, required_size, MS_SYNC) != 0) {
        KVCM_LOG_WARN("Put msync failed for file %s", file_path.c_str());
    }

    return ER_OK;
}

} // namespace kv_cache_manager