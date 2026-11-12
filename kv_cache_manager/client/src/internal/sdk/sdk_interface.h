#pragma once

#include <memory>
#include <unordered_map>
#include <utility>

#include "kv_cache_manager/client/include/common.h"
#include "kv_cache_manager/client/src/internal/config/sdk_config.h"
#include "kv_cache_manager/client/src/internal/sdk/sdk_type.h"
#include "kv_cache_manager/data_storage/data_storage_uri.h"
#include "kv_cache_manager/data_storage/storage_config.h"

namespace kv_cache_manager {

// 各存储后端 SDK 的统一接口。
// Wrapper 在有调用方期限时通过 WithDeadline 传入绝对 steady_clock 毫秒期限，
// 后端在已有逐组/逐块检查和等待位置使用它，与自身配置的阶段上限取最小值。
// 直接调用旧 Get/Put 仍使用 Init 时注入的静态阶段上限。
// 规范全文见 docs/design/client_sdk_io_contract.md。
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
    // 超时路径必须输出可归因日志（超时契约见类前注释）。
    virtual ClientErrorCode Get(const std::vector<DataStorageUri> &remote_uris, const BlockBuffers &local_buffers) = 0;
    // actual_remote_uris是实际存储的远端地址
    virtual ClientErrorCode Put(const std::vector<DataStorageUri> &remote_uris,
                                const BlockBuffers &local_buffers,
                                std::shared_ptr<std::vector<DataStorageUri>> actual_remote_uris) = 0;

    // Legacy/custom backends keep their existing behavior unless they override
    // these methods to consume the deadline.
    virtual ClientErrorCode GetWithDeadline(const std::vector<DataStorageUri> &remote_uris,
                                            const BlockBuffers &local_buffers,
                                            int64_t /*deadline_ms*/) {
        return Get(remote_uris, local_buffers);
    }
    virtual ClientErrorCode PutWithDeadline(const std::vector<DataStorageUri> &remote_uris,
                                            const BlockBuffers &local_buffers,
                                            std::shared_ptr<std::vector<DataStorageUri>> actual_remote_uris,
                                            int64_t /*deadline_ms*/) {
        return Put(remote_uris, local_buffers, std::move(actual_remote_uris));
    }

protected:
    virtual ClientErrorCode Alloc(const std::vector<DataStorageUri> &remote_uris,
                                  std::vector<DataStorageUri> &alloc_uris) = 0;

    using GroupMap = std::unordered_map<std::string, BlockGroup>;
    // 按 path 分组，并记录每个元素在原始入参中的下标到 BlockGroup::indices。
    GroupMap SplitByPath(const std::vector<DataStorageUri> &remote_uris, const BlockBuffers &local_buffers);
};

} // namespace kv_cache_manager
