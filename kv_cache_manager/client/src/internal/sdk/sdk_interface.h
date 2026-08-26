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
    //
    // 同序契约（Get/Put 共同遵守）：出参与入参按位置一一对应。
    //   - Get：local_buffers[i] 必须被 remote_uris[i] 指向的 block 填充。
    //   - Put：返回时 (*actual_remote_uris)[i] 必须是 remote_uris[i] /
    //     local_buffers[i] 这个 block 实际写入的远端位置，且
    //     actual_remote_uris->size() == remote_uris.size()。
    // 实现不得按内部分组（如按 path 聚合）的迭代顺序回填 actual_remote_uris；
    // 即使同一请求内多个 block 落在不同 path（交错出现），也必须保证上述同序性。
    // 上层 SdkWrapper 依赖该契约把各 SDK 的返回值回填到原始请求位置。
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