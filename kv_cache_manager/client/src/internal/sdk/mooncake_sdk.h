#pragma once

#include "3rdparty/mooncake/client_c.h"
#include "kv_cache_manager/common/error_code.h"
#include "kv_cache_manager/common/logger.h"
#include "sdk_interface.h"
namespace kv_cache_manager {

class MooncakeSdk : public SdkInterface {
public:
    MooncakeSdk() {}
    ~MooncakeSdk();

    ClientErrorCode Close();

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

private:
    std::pair<size_t, bool>
    extractSlices(const MooncakeRemoteItem &item, const BlockBuffer &buffer, std::vector<Slice_t> &slices) const;

private:
    client_t client_{nullptr};
    std::shared_ptr<MooncakeSdkConfig> sdk_backend_config_;
    std::shared_ptr<StorageConfig> storage_config_;
};

} // namespace kv_cache_manager