#pragma once

#include <memory>
#include <unordered_map>

#include "kv_cache_manager/client/include/common.h"
#include "kv_cache_manager/client/src/internal/config/sdk_config.h"
#include "kv_cache_manager/client/src/internal/sdk/sdk_type.h"
#include "kv_cache_manager/data_storage/data_storage_uri.h"
#include "kv_cache_manager/data_storage/storage_config.h"

namespace kv_cache_manager {

class SdkInterface {
public:
    SdkInterface() {}
    virtual ~SdkInterface() = default;
    virtual ClientErrorCode Init(const std::shared_ptr<SdkBackendConfig> &sdk_backend_config,
                                 const std::shared_ptr<StorageConfig> &storage_config) = 0;

    virtual SdkType Type() = 0;

    // 一个remote_uri和一个Blockbuffer对应一个block
    virtual ClientErrorCode Get(const std::vector<DataStorageUri> &remote_uris, const BlockBuffers &local_buffers) = 0;
    // actual_remote_uris是实际存储的远端地址
    virtual ClientErrorCode Put(const std::vector<DataStorageUri> &remote_uris,
                                const BlockBuffers &local_buffers,
                                std::shared_ptr<std::vector<DataStorageUri>> actual_remote_uris) = 0;

protected:
    virtual ClientErrorCode Alloc(const std::vector<DataStorageUri> &remote_uris,
                                  std::vector<DataStorageUri> &alloc_uris) = 0;

    using GroupMap = std::unordered_map<std::string, BlockGroup>;
    GroupMap SplitByPath(const std::vector<DataStorageUri> &remote_uris, const BlockBuffers &local_buffers);
};

} // namespace kv_cache_manager