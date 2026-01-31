#pragma once

#include <filesystem>

#include "kv_cache_manager/client/src/internal/sdk/hf3fs_usrbio_client.h"
#include "sdk_interface.h"

namespace kv_cache_manager {

class Hf3fsUsrbioApi;

class Hf3fsSdk : public SdkInterface {
public:
    Hf3fsSdk() {}
    ~Hf3fsSdk() override;

    SdkType Type() override { return SdkType::HF3FS; }
    ClientErrorCode Init(const std::shared_ptr<SdkBackendConfig> &sdk_backend_config,
                         const std::shared_ptr<StorageConfig> &storage_config) override;
    ClientErrorCode Get(const std::vector<DataStorageUri> &remote_uris, const BlockBuffers &local_buffers) override;
    ClientErrorCode Put(const std::vector<DataStorageUri> &remote_uris,
                        const BlockBuffers &local_buffers,
                        std::shared_ptr<std::vector<DataStorageUri>> actual_remote_uris) override;

protected:
    ClientErrorCode Alloc(const std::vector<DataStorageUri> &remote_uris,
                          std::vector<DataStorageUri> &alloc_uris) override;

private:
    ClientErrorCode Get(const DataStorageUri &uri, const BlockBuffer &block_buffer) const;
    ClientErrorCode Put(const DataStorageUri &uri, const BlockBuffer &block_buffer) const;

    bool CheckConfig(const Hf3fsSdkConfig &hf3fs_config) const;
    void DeleteRemainingIovShm() const;
    bool InitIovHandle(Hf3fsIovHandle &handle,
                       size_t iov_block_size,
                       size_t iov_size,
                       const std::shared_ptr<Hf3fsCudaUtil> &cuda_util) const;
    void ReleaseIovHandle(Hf3fsIovHandle &handle);
    struct hf3fs_iov *CreateIov(const std::string &mountpoint, size_t iov_size, size_t iov_block_size) const;
    void DestroyIov(struct hf3fs_iov *iov) const;
    Hf3fsFileConfig BuildHf3fsFileConfig(const std::string &filepath) const;
    bool CreateDir(const std::filesystem::path &dir) const;
    std::optional<size_t> GetFileOffset(const DataStorageUri &uri) const;

private:
    int64_t byte_size_per_block_;
    std::shared_ptr<Hf3fsSdkConfig> config_;
    std::shared_ptr<Hf3fsUsrbioApi> usrbio_api_;
    Hf3fsIovHandle read_iov_handle_;
    Hf3fsIovHandle write_iov_handle_;
};

} // namespace kv_cache_manager