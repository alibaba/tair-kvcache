#pragma once
#if defined(USING_CUDA)
#include <cuda_runtime.h>
#elif defined(USING_MUSA)
#include <musa_runtime.h>
#endif
#include <shared_mutex>

#include "kv_cache_manager/client/include/common.h"
#include "kv_cache_manager/client/include/transfer_client.h"
#include "kv_cache_manager/data_storage/data_storage_uri.h"

namespace kv_cache_manager {
class ClientConfig;
class SdkWrapper;
class SdkBufferCheckPool;

class TransferClientImpl : public TransferClient {
public:
    TransferClientImpl();
    ~TransferClientImpl() override;

    using TransferClient::LoadKvCaches;
    using TransferClient::SaveKvCaches;

    ClientErrorCode LoadKvCaches(const UriStrVec &uri_str_vec,
                                 const BlockBuffers &block_buffers,
                                 const LoadKvCachesOptions &options) override;
    std::pair<ClientErrorCode, UriStrVec> SaveKvCaches(const UriStrVec &uri_str_vec,
                                                       const BlockBuffers &block_buffers,
                                                       const SaveKvCachesOptions &options) override;

protected:
    ClientErrorCode Init(const std::string &client_config, const InitParams &init_params) override;

private:
    ClientErrorCode IsValid(const std::unique_ptr<ClientConfig> &client_config) const;
    // Parse init_params.storage_configs to (1) reject enable_inline_header=true (Scheme B
    // is reserved but not implemented) and (2) detect any spec opting into
    // enable_meta_checksum so we can auto-initialize SdkBufferCheckPool.
    ClientErrorCode ValidateStorageConfigsForIntegrity(const std::string &storage_configs_json,
                                                       bool &any_meta_checksum_enabled) const;
    std::vector<DataStorageUri> ParseLocations(const UriStrVec &uri_str_vec);
    UriStrVec ConstructLocations(const std::vector<DataStorageUri> &uris);
    void PrintBlockChecksumAndUri(const std::string &prefix,
                                  const UriStrVec &uri_str_vec,
                                  const std::vector<int64_t> &block_checksums,
                                  const std::shared_ptr<TransferTraceInfo> &trace_info) const;

private:
    friend class TransferClient;
    std::unique_ptr<ClientConfig> client_config_;
    InitParams init_params_;
    std::unique_ptr<SdkWrapper> sdk_wrapper_;
    mutable std::shared_mutex config_mutex_;
#if defined(USING_CUDA) || defined(USING_MUSA)
    bool is_check_buffer_ = false;       // KVCM_SDK_CHECK env var: log-only fallback
    bool meta_checksum_enabled_ = false; // spec.integrity.enable_meta_checksum gate
    size_t max_check_iov_num_;
    std::shared_ptr<SdkBufferCheckPool> sdk_buffer_check_pool_;
#endif
};

} // namespace kv_cache_manager
