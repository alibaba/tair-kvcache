#pragma once

#include <memory>
#include <unordered_map>

#include "kv_cache_manager/client/include/common.h"
#include "kv_cache_manager/client/src/internal/config/sdk_config.h"
#include "kv_cache_manager/client/src/internal/sdk/sdk_type.h"
#include "kv_cache_manager/data_storage/data_storage_uri.h"
#include "kv_cache_manager/data_storage/storage_config.h"

namespace kv_cache_manager {

// 各存储后端 SDK 的统一接口。
//
// deadline 语义：int64_t deadline_ms 是绝对时间点（steady_clock 毫秒）。
// 到达 deadline 后实现方不得再触碰 caller 的 local buffers。能做到固然好，
// 做不到（soft 级后端）须如实声明。0 表示无 deadline。
// 规范全文见 docs/design/client_sdk_io_contract.md。
class SdkInterface {
public:
    SdkInterface() {}
    virtual ~SdkInterface() = default;
    virtual ClientErrorCode Init(const std::shared_ptr<SdkBackendConfig> &sdk_backend_config,
                                 const std::shared_ptr<StorageConfig> &storage_config) = 0;

    virtual SdkType Type() = 0;

    // remote_uris[i] ↔ local_buffers[i] 保序。
    // deadline_ms：绝对时间点（steady_clock 毫秒），0=无 deadline。
    // 超时路径必须输出可归因日志。
    virtual ClientErrorCode
    Get(const std::vector<DataStorageUri> &remote_uris, const BlockBuffers &local_buffers, int64_t deadline_ms) = 0;
    // actual_remote_uris[i] 对应 remote_uris[i]（下标即身份）。分组处理后须按 indices 回填原位。
    virtual ClientErrorCode Put(const std::vector<DataStorageUri> &remote_uris,
                                const BlockBuffers &local_buffers,
                                std::shared_ptr<std::vector<DataStorageUri>> actual_remote_uris,
                                int64_t deadline_ms) = 0;

protected:
    virtual ClientErrorCode Alloc(const std::vector<DataStorageUri> &remote_uris,
                                  std::vector<DataStorageUri> &alloc_uris) = 0;

    using GroupMap = std::unordered_map<std::string, BlockGroup>;
    // 按 path 分组，并记录每个元素在原始入参中的下标到 BlockGroup::indices。
    GroupMap SplitByPath(const std::vector<DataStorageUri> &remote_uris, const BlockBuffers &local_buffers);
};

} // namespace kv_cache_manager
