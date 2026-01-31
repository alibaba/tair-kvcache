#pragma once

#include "kv_cache_manager/client/src/internal/sdk/sdk_interface.h"

namespace kv_cache_manager {

class TairMempoolSdk : public SdkInterface {
public:
    TairMempoolSdk() {}
    ~TairMempoolSdk() override = default;

    ClientErrorCode Init(const std::shared_ptr<SdkBackendConfig> &sdk_backend_config,
                         const std::shared_ptr<StorageConfig> &storage_config) override;
    SdkType Type() override;
    ClientErrorCode Get(const std::vector<DataStorageUri> &remote_uris, const BlockBuffers &local_buffers) override;
    ClientErrorCode Put(const std::vector<DataStorageUri> &remote_uris,
                        const BlockBuffers &local_buffers,
                        std::shared_ptr<std::vector<DataStorageUri>> actual_remote_uris) override;

protected:
    ClientErrorCode Alloc(const std::vector<DataStorageUri> &remote_uris,
                          std::vector<DataStorageUri> &alloc_uris) override;
};

} // namespace kv_cache_manager