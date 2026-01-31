#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "kv_cache_manager/client/include/common.h"
#include "kv_cache_manager/client/src/internal/config/client_config.h"
#include "kv_cache_manager/client/src/internal/config/sdk_config.h"
#include "kv_cache_manager/client/src/internal/sdk/sdk_type.h"
#include "kv_cache_manager/data_storage/data_storage_uri.h"

namespace kv_cache_manager {
class StorageConfig;
class LockFreeThreadPool;
class SdkFactory;
class SdkInterface;

class SdkWrapper {
public:
    SdkWrapper();
    ~SdkWrapper();

public:
    ClientErrorCode Init(const std::unique_ptr<ClientConfig> &client_config, const InitParams &init_params);

    ClientErrorCode Get(const std::vector<DataStorageUri> &remote_uris, const BlockBuffers &local_buffers);
    ClientErrorCode Put(const std::vector<DataStorageUri> &remote_uris,
                        const BlockBuffers &local_buffers,
                        std::shared_ptr<std::vector<DataStorageUri>> actual_remote_uris);

private:
    enum class OpType : uint8_t {
        GET = 0,
        PUT = 1,
    };
    ClientErrorCode Valid(const std::vector<DataStorageUri> &remote_uris, const BlockBuffers local_buffers);
    std::shared_ptr<SdkInterface> GetSdk(const DataStorageUri &remote_uri);

    std::string getOpTypeString(OpType op_type) const;
    ClientErrorCode RunWithTimeout(OpType op_type, const std::function<ClientErrorCode()> &func, int timeout_ms) const;
    ClientErrorCode UpdateMooncakeSdkConfig(const std::shared_ptr<SdkBackendConfig> &sdk_backend_config,
                                            RegistSpan *span,
                                            const std::string &self_location_spec_name);

private:
    SdkFactory *sdk_factory_;
    std::shared_ptr<SdkWrapperConfig> wrapper_config_;
    std::vector<std::shared_ptr<StorageConfig>> storage_configs_;
    std::unique_ptr<LockFreeThreadPool> wait_task_thread_pool_;
    // storage unique name -> storage_sdk
    std::map<std::string, std::shared_ptr<SdkInterface>> sdk_map_;
};

} // namespace kv_cache_manager