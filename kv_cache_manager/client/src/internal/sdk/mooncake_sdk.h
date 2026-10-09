#pragma once

#include "3rdparty/mooncake/client_c.h"
#include "kv_cache_manager/common/error_code.h"
#include "kv_cache_manager/common/logger.h"
#include "sdk_interface.h"

namespace kv_cache_manager {

// mooncake transfer engine 无取消语义：caller 的 iov.base 直接作为 slices 提交给网卡 DMA，
// 超时返回后 buffer 可能仍被在飞 DMA 写入（soft 级，唯一达不到 hard 的后端）。
// 逐 key 准入检查把超时时刻的暴露面限制为最多 1 个 block。
// 后端能力矩阵见 docs/design/client_sdk_io_contract.md。
class MooncakeSdk : public SdkInterface {
public:
    MooncakeSdk() {}
    ~MooncakeSdk();

    ClientErrorCode Close();

    ClientErrorCode Init(const std::shared_ptr<SdkBackendConfig> &sdk_backend_config,
                         const std::shared_ptr<StorageConfig> &storage_config) override;
    SdkType Type() override;
    ClientErrorCode Get(const std::vector<DataStorageUri> &remote_uris,
                        const BlockBuffers &local_buffers) override;

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