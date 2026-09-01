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
// 超时契约（静态，无调用参数）：SdkWrapper 在 Init 阶段把自身配置的静态预算注入
// SdkBackendConfig::timeout_config()（get/put_timeout_ms）。后端从自身任务起点
// 起算 deadline，在预算内完成或内部取消；到点后不得再触碰 caller 的 local
// buffers。能做到的（localfile/hf3fs）须如实做到，做不到的（mooncake，上游无
// 取消语义）须声明 soft 并输出可归因日志。不读取该字段的后端（tair_mempool）
// 自行管理内部超时，要求其内部超时严格小于 wrapper 预算。
// 规范全文见 docs/design/client_sdk_io_contract.md。
class SdkInterface {
public:
    SdkInterface() {}
    virtual ~SdkInterface() = default;
    virtual ClientErrorCode Init(const std::shared_ptr<SdkBackendConfig> &sdk_backend_config,
                                 const std::shared_ptr<StorageConfig> &storage_config) = 0;

    virtual SdkType Type() = 0;

    // remote_uris[i] ↔ local_buffers[i] 保序。超时路径必须输出可归因日志。
    virtual ClientErrorCode Get(const std::vector<DataStorageUri> &remote_uris, const BlockBuffers &local_buffers) = 0;
    // actual_remote_uris[i] 对应 remote_uris[i]（下标即身份）。分组处理后须按 indices 回填原位。
    virtual ClientErrorCode Put(const std::vector<DataStorageUri> &remote_uris,
                                const BlockBuffers &local_buffers,
                                std::shared_ptr<std::vector<DataStorageUri>> actual_remote_uris) = 0;

protected:
    virtual ClientErrorCode Alloc(const std::vector<DataStorageUri> &remote_uris,
                                  std::vector<DataStorageUri> &alloc_uris) = 0;

    using GroupMap = std::unordered_map<std::string, BlockGroup>;
    // 按 path 分组，并记录每个元素在原始入参中的下标到 BlockGroup::indices。
    GroupMap SplitByPath(const std::vector<DataStorageUri> &remote_uris, const BlockBuffers &local_buffers);
};

} // namespace kv_cache_manager
