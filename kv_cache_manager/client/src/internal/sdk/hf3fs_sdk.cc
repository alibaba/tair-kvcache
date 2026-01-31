#include "kv_cache_manager/client/src/internal/sdk/hf3fs_sdk.h"

#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "kv_cache_manager/client/src/internal/sdk/hf3fs_cuda_util.h"
#include "kv_cache_manager/client/src/internal/sdk/hf3fs_mempool.h"
#include "kv_cache_manager/common/logger.h"

namespace kv_cache_manager {

Hf3fsSdk::~Hf3fsSdk() {
    KVCM_LOG_INFO("Hf3fsSdk destructor");
    ReleaseIovHandle(read_iov_handle_);
    ReleaseIovHandle(write_iov_handle_);
}

ClientErrorCode Hf3fsSdk::Init(const std::shared_ptr<SdkBackendConfig> &sdk_backend_config,
                               const std::shared_ptr<StorageConfig> &storage_config) {
    // delete remaining iov shm to avoid shm space not enough
    DeleteRemainingIovShm();

    auto hf3fs_config = std::dynamic_pointer_cast<Hf3fsSdkConfig>(sdk_backend_config);
    if (!hf3fs_config) {
        KVCM_LOG_WARN("init 3fs sdk failed, unexpected config type: %d",
                      static_cast<uint8_t>(sdk_backend_config->type()));
        return ER_INVALID_SDKBACKEND_CONFIG;
    }
    KVCM_LOG_INFO("init 3fs sdk, config: %s", hf3fs_config->ToString().c_str());

    if (!CheckConfig(*hf3fs_config)) {
        KVCM_LOG_WARN("3fs init failed, check init params failed");
        return ER_INVALID_SDKBACKEND_CONFIG;
    }
    config_ = hf3fs_config;

    auto cuda_util = std::make_shared<Hf3fsCudaUtil>();
    if (!cuda_util->Init()) {
        KVCM_LOG_WARN("dist storage 3fs init failed, cuda util init failed");
        return ER_SDKINIT_ERROR;
    }

    usrbio_api_ = std::make_shared<Hf3fsUsrbioApi>();

    // read iov
    if (!InitIovHandle(read_iov_handle_, config_->read_iov_block_size(), config_->read_iov_size(), cuda_util)) {
        KVCM_LOG_WARN("init read iov handle failed");
        return ER_SDKINIT_ERROR;
    }

    // write iov
    if (!InitIovHandle(write_iov_handle_, config_->write_iov_block_size(), config_->write_iov_size(), cuda_util)) {
        KVCM_LOG_WARN("init write iov handle failed");
        ReleaseIovHandle(read_iov_handle_);
        return ER_SDKINIT_ERROR;
    }

    // TODO(LXQ): metrics

    return ER_OK;
}

ClientErrorCode Hf3fsSdk::Get(const std::vector<DataStorageUri> &remote_uris, const BlockBuffers &local_buffers) {
    if (remote_uris.size() != local_buffers.size()) {
        KVCM_LOG_WARN(
            "get failed, size mismatch, uris size: %lu, buffers size: %lu", remote_uris.size(), local_buffers.size());
        return ER_INVALID_PARAMS;
    }

    for (size_t i = 0; i < remote_uris.size(); ++i) {
        if (Get(remote_uris[i], local_buffers[i]) != ER_OK) {
            return ER_SDKREAD_ERROR;
        }
    }

    return ER_OK;
}

ClientErrorCode Hf3fsSdk::Get(const DataStorageUri &uri, const BlockBuffer &block_buffer) const {
    if (block_buffer.iovs.empty()) {
        return ER_OK;
    }

    const auto path = uri.GetPath();
    if (path.empty()) {
        KVCM_LOG_WARN("get failed, path is empty or not exists: %s", path.c_str());
        return ER_INVALID_PARAMS;
    }

    const auto offset_opt = GetFileOffset(uri);
    if (!offset_opt.has_value()) {
        return ER_INVALID_PARAMS;
    }

    auto iovs = block_buffer.iovs;
    iovs.insert(iovs.begin(), Iov{MemoryType::CPU, nullptr, offset_opt.value(), true});

    auto usrbio_client = std::make_shared<Hf3fsUsrbioClient>(
        BuildHf3fsFileConfig(path), read_iov_handle_, write_iov_handle_, usrbio_api_);
    if (!usrbio_client->Read(iovs)) {
        KVCM_LOG_WARN("get failed, read failed, path: %s", path.c_str());
        return ER_SDKREAD_ERROR;
    }

    return ER_OK;
}

ClientErrorCode Hf3fsSdk::Put(const std::vector<DataStorageUri> &remote_uris,
                              const BlockBuffers &local_buffers,
                              std::shared_ptr<std::vector<DataStorageUri>> actual_remote_uris) {
    if (remote_uris.size() != local_buffers.size()) {
        KVCM_LOG_WARN(
            "put failed, size mismatch, uris size: %lu, buffers size: %lu", remote_uris.size(), local_buffers.size());
        return ER_INVALID_PARAMS;
    }

    if (Alloc(remote_uris, *actual_remote_uris) != ER_OK) {
        return ER_SDKALLOC_ERROR;
    }

    for (size_t i = 0; i < remote_uris.size(); ++i) {
        if (Put(remote_uris[i], local_buffers[i]) != ER_OK) {
            return ER_SDKWRITE_ERROR;
        }
    }

    return ER_OK;
}

ClientErrorCode Hf3fsSdk::Put(const DataStorageUri &uri, const BlockBuffer &block_buffer) const {
    if (block_buffer.iovs.empty()) {
        return ER_OK;
    }

    const auto path = uri.GetPath();
    if (path.empty()) {
        KVCM_LOG_WARN("put failed, path is empty");
        return ER_INVALID_PARAMS;
    }

    const auto offset_opt = GetFileOffset(uri);
    if (!offset_opt.has_value()) {
        return ER_INVALID_PARAMS;
    }

    auto iovs = block_buffer.iovs;
    iovs.insert(iovs.begin(), Iov{MemoryType::CPU, nullptr, offset_opt.value(), true});

    auto usrbio_client = std::make_shared<Hf3fsUsrbioClient>(
        BuildHf3fsFileConfig(path), read_iov_handle_, write_iov_handle_, usrbio_api_);
    if (!usrbio_client->Write(iovs)) {
        KVCM_LOG_WARN("put failed, write failed, path: %s", path.c_str());
        return ER_SDKWRITE_ERROR;
    }

    return ER_OK;
}

ClientErrorCode Hf3fsSdk::Alloc(const std::vector<DataStorageUri> &remote_uris,
                                std::vector<DataStorageUri> &alloc_uris) {
    alloc_uris = remote_uris;

    for (const auto &uri : remote_uris) {
        const auto path = uri.GetPath();
        if (path.empty()) {
            KVCM_LOG_WARN("alloc failed, path is empty");
            return ER_SDKALLOC_ERROR;
        }
        if (!CreateDir(std::filesystem::path(path).parent_path())) {
            return ER_SDKALLOC_ERROR;
        }
    }
    return ER_OK;
}

bool Hf3fsSdk::CheckConfig(const Hf3fsSdkConfig &hf3fs_config) const {
    // TODO(LXQ): maybe need to move to Hf3fsSdkConfig::Validate()
    const auto &mountpoint = hf3fs_config.mountpoint();
    if (mountpoint.empty()) {
        KVCM_LOG_WARN("init failed, 3fs mountpoint is empty");
        return false;
    }
    if (!std::filesystem::exists(mountpoint)) {
        KVCM_LOG_WARN("init failed, 3fs mountpoint not exists: %s", mountpoint.c_str());
        return false;
    }

    const auto &root_dir = hf3fs_config.root_dir();
    if (root_dir.empty()) {
        KVCM_LOG_WARN("init failed, 3fs root dir is empty");
        return false;
    }
    const auto root_dir_path = std::filesystem::path(mountpoint) / root_dir;
    if (!std::filesystem::exists(root_dir_path)) {
        KVCM_LOG_WARN("init failed, 3fs root dir not exists: %s", root_dir_path.string().c_str());
        return false;
    }
    return true;
}

void Hf3fsSdk::DeleteRemainingIovShm() const {
    // 删除旧的 shm iov , 避免 shm 空间不够用
    namespace fs = std::filesystem;

    const std::string shm_path = "/dev/shm/";
    const std::string prefix = "hf3fs-iov-";

    try {
        fs::path dir(shm_path);
        if (!fs::exists(dir)) {
            return;
        }
        if (!fs::is_directory(dir)) {
            return;
        }

        const auto threshold = std::chrono::seconds(300); // 5min
        const auto now = fs::file_time_type::clock::now();

        for (const auto &entry : fs::directory_iterator(dir)) {
            if (!fs::is_regular_file(entry.status())) {
                continue;
            }

            std::string filename = entry.path().filename().string();
            if (filename.find(prefix) != 0) {
                continue;
            }

            const auto age = now - fs::last_write_time(entry.path());
            if (age > threshold) {
                try {
                    KVCM_LOG_INFO("remove old shm iov file: %s", filename.c_str());
                    fs::remove(entry.path());
                } catch (const fs::filesystem_error &e) {
                    KVCM_LOG_WARN(
                        "found exception when remove old iov file: %s, exception: %s", filename.c_str(), e.what());
                }
            }
        }
    } catch (const fs::filesystem_error &e) {
        KVCM_LOG_WARN("found exception when remove old iov file, exception: %s", e.what());
    }
}

bool Hf3fsSdk::InitIovHandle(Hf3fsIovHandle &handle,
                             size_t iov_block_size,
                             size_t iov_size,
                             const std::shared_ptr<Hf3fsCudaUtil> &cuda_util) const {
    if (iov_block_size != 0 && iov_size % iov_block_size != 0) {
        iov_size = (iov_size / iov_block_size + 1) * iov_block_size;
    }

    auto iov = CreateIov(config_->mountpoint(), iov_size, iov_block_size);
    if (iov == nullptr) {
        KVCM_LOG_WARN("create iov failed, iov size: %zu, iov block size: %zu", iov_size, iov_block_size);
        return false;
    }

    auto mempool = std::make_shared<Hf3fsMempool>(iov->base, iov_size, iov_block_size);
    if (!mempool->Init()) {
        KVCM_LOG_WARN("mempool init failed, iov base: %p, iov size: %zu, iov block size: %zu",
                      iov->base,
                      iov_size,
                      iov_block_size);
        DestroyIov(iov);
        return false;
    }

    if (!cuda_util->RegisterHost(iov->base, iov_size)) {
        KVCM_LOG_WARN("cuda register iov failed, iov base: %p, expect iov size: %zu, actual iov size: %zu",
                      iov->base,
                      iov_size,
                      iov->size);
        DestroyIov(iov);
        return false;
    }
    handle = {iov, nullptr, iov_size, iov_block_size, mempool, cuda_util};
    return true;
}

void Hf3fsSdk::ReleaseIovHandle(Hf3fsIovHandle &iov_handle) {
    if (iov_handle.iov != nullptr) {
        if (iov_handle.cuda_util) {
            iov_handle.cuda_util->UnregisterHost(iov_handle.iov->base);
        }
        DestroyIov(iov_handle.iov);
        iov_handle.iov = nullptr;
    }
    iov_handle.iov_mempool.reset();
    iov_handle.cuda_util.reset();
}

struct hf3fs_iov *Hf3fsSdk::CreateIov(const std::string &mountpoint, size_t iov_size, size_t iov_block_size) const {
    if (mountpoint.empty()) {
        KVCM_LOG_WARN("create iov failed, mountpoint is empty");
        return nullptr;
    }
    if (iov_size <= 0) {
        KVCM_LOG_WARN("create iov failed, iov size is invalid: %zu", iov_size);
        return nullptr;
    }

    auto iov = new struct hf3fs_iov();
    auto ret = usrbio_api_->Hf3fsIovCreate(iov, mountpoint.c_str(), iov_size, iov_block_size, -1);
    if (ret != 0) {
        KVCM_LOG_WARN("3fs iov create failed, errno: %s, mountpoint: %s, iov size: %lu, iov block size: %lu",
                      strerror(-ret),
                      mountpoint.c_str(),
                      iov_size,
                      iov_block_size);
        delete iov;
        iov = nullptr;
        return nullptr;
    }
    return iov;
}

void Hf3fsSdk::DestroyIov(struct hf3fs_iov *iov) const {
    if (iov == nullptr) {
        return;
    }
    usrbio_api_->Hf3fsIovDestroy(iov);
    delete iov;
}

Hf3fsFileConfig Hf3fsSdk::BuildHf3fsFileConfig(const std::string &filepath) const {
    Hf3fsFileConfig config;
    config.filepath = filepath;
    config.mountpoint = config_->mountpoint();
    return config;
}

bool Hf3fsSdk::CreateDir(const std::filesystem::path &dir) const {
    if (std::filesystem::exists(dir)) {
        return true;
    }

    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec) {
        KVCM_LOG_WARN("create directory failed, dir: %s, error: %s", dir.string().c_str(), ec.message().c_str());
        return false;
    }
    return true;
}

std::optional<size_t> Hf3fsSdk::GetFileOffset(const DataStorageUri &uri) const {
    uint64_t blkid = 0;
    size_t size = 0;
    uri.GetParamAs<uint64_t>("blkid", blkid);
    uri.GetParamAs<size_t>("size", size);
    if (size == 0) {
        KVCM_LOG_WARN("get file offset failed, uri size is 0, uri: %s", uri.ToUriString().c_str());
        return std::nullopt;
    }
    return blkid * size;
}

} // namespace kv_cache_manager
