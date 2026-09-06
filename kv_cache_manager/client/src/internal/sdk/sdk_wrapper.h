#pragma once

#include <cstdint>
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
    ClientErrorCode Init(const std::unique_ptr<ClientConfig> &client_config,
                         const InitParams &init_params,
                         const SharedMemoryRegistration *shared_memory_registration = nullptr);

    // KVMeta-only initialization. It enables bounded variable-size validation
    // in the SDK instances created by this wrapper without changing the
    // regular TransferClient policy.
    ClientErrorCode InitForKvMeta(const std::unique_ptr<ClientConfig> &client_config,
                                  const InitParams &init_params,
                                  std::uint64_t max_object_bytes,
                                  const SharedMemoryRegistration *shared_memory_registration = nullptr);

    ClientErrorCode Get(const std::vector<DataStorageUri> &remote_uris, const BlockBuffers &local_buffers);
    ClientErrorCode Put(const std::vector<DataStorageUri> &remote_uris,
                        const BlockBuffers &local_buffers,
                        std::shared_ptr<std::vector<DataStorageUri>> actual_remote_uris);
    ClientErrorCode GetKvMetaObjects(const std::vector<DataStorageUri> &remote_uris,
                                     const std::vector<std::uint64_t> &value_sizes,
                                     const BlockBuffers &local_buffers);
    ClientErrorCode PutKvMetaObjects(const std::vector<DataStorageUri> &remote_uris,
                                     const std::vector<std::uint64_t> &value_sizes,
                                     const BlockBuffers &local_buffers,
                                     std::shared_ptr<std::vector<DataStorageUri>> actual_remote_uris);

private:
    enum class OpType : uint8_t {
        GET = 0,
        PUT = 1,
    };

    // 按 SDK 分组的 URI 和 buffer 组
    struct SdkGroup {
        std::shared_ptr<SdkInterface> sdk;
        std::vector<size_t> indices; // 原始索引，用于结果重排
        std::vector<DataStorageUri> uris;
        BlockBuffers buffers;
    };

    ClientErrorCode Valid(const std::vector<DataStorageUri> &remote_uris, const BlockBuffers local_buffers);
    ClientErrorCode ValidateKvMetaObjects(const std::vector<DataStorageUri> &remote_uris,
                                          const std::vector<std::uint64_t> &value_sizes,
                                          const BlockBuffers &local_buffers) const;
    ClientErrorCode InitInternal(const std::unique_ptr<ClientConfig> &client_config,
                                 const InitParams &init_params,
                                 bool variable_object_size_enabled,
                                 std::uint64_t max_object_bytes,
                                 const SharedMemoryRegistration *shared_memory_registration);
    std::shared_ptr<SdkInterface> GetSdk(const DataStorageUri &remote_uri);

    // 按 URI hostname 分组，每组关联对应 SDK；任一 hostname 无对应 SDK 时返回错误
    ClientErrorCode GroupBySdk(const std::vector<DataStorageUri> &remote_uris,
                               const BlockBuffers &local_buffers,
                               std::vector<SdkGroup> &groups);

    std::string getOpTypeString(OpType op_type) const;
    ClientErrorCode RunWithTimeoutParallel(OpType op_type,
                                           std::vector<std::function<ClientErrorCode()>> &&tasks,
                                           int timeout_ms,
                                           bool wait_for_inflight = false) const;
    ClientErrorCode UpdateMooncakeSdkConfig(const std::shared_ptr<SdkBackendConfig> &sdk_backend_config,
                                            RegistSpan *span,
                                            const std::string &self_location_spec_name);
    ClientErrorCode PrepareSharedMemoryRegistration(const SharedMemoryRegistration &shared_memory_registration,
                                                    SharedMemoryRegistration &prepared_registration);
    ClientErrorCode UpdateTairMempoolSdkConfig(const std::shared_ptr<SdkBackendConfig> &sdk_backend_config,
                                               const SharedMemoryRegistration *shared_memory_registration);

private:
    SdkFactory *sdk_factory_;
    std::shared_ptr<SdkWrapperConfig> wrapper_config_;
    std::vector<std::shared_ptr<StorageConfig>> storage_configs_;
    std::unique_ptr<LockFreeThreadPool> wait_task_thread_pool_;
    // storage unique name -> storage_sdk
    std::map<std::string, std::shared_ptr<SdkInterface>> sdk_map_;
    int owned_shm_fd_{-1};
    bool variable_object_size_enabled_{false};
    std::uint64_t max_variable_object_bytes_{0};
};

} // namespace kv_cache_manager
