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

    ClientErrorCode LoadKvCaches(const UriStrVec &uri_str_vec,
                                 const BlockBuffers &block_buffers,
                                 std::shared_ptr<TransferTraceInfo> trace_info = nullptr,
                                 const std::vector<int64_t> *expected_hashes = nullptr) override;
    std::pair<ClientErrorCode, UriStrVec> SaveKvCaches(const UriStrVec &uri_str_vec,
                                                       const BlockBuffers &block_buffers,
                                                       std::shared_ptr<TransferTraceInfo> trace_info = nullptr,
                                                       std::vector<int64_t> *out_block_hashes = nullptr) override;

protected:
    ClientErrorCode Init(const std::string &client_config, const InitParams &init_params) override;

private:
    ClientErrorCode IsValid(const std::unique_ptr<ClientConfig> &client_config) const;
    // 任务 82620492：解析 init_params.storage_configs JSON，拒绝 enable_inline_header=true
    // 的 spec (方案 B 接口预留)，并探测是否有任何 spec 开启 enable_meta_checksum (方案 A)
    // 以便自动启用 SdkBufferCheckPool。
    ClientErrorCode ValidateStorageConfigsForIntegrity(const std::string &storage_configs_json,
                                                       bool &any_meta_checksum_enabled) const;
    std::vector<DataStorageUri> ParseLocations(const UriStrVec &uri_str_vec);
    UriStrVec ConstructLocations(const std::vector<DataStorageUri> &uris);
    void PrintBlockHashAndUri(const std::string &prefix,
                              const UriStrVec &uri_str_vec,
                              const std::vector<int64_t> &block_hashs,
                              const std::shared_ptr<TransferTraceInfo> &trace_info) const;

private:
    friend class TransferClient;
    std::unique_ptr<ClientConfig> client_config_;
    InitParams init_params_;
    std::unique_ptr<SdkWrapper> sdk_wrapper_;
    mutable std::shared_mutex config_mutex_;
#if defined(USING_CUDA) || defined(USING_MUSA)
    bool is_check_buffer_ = false;       // KVCM_SDK_CHECK env var：仅日志 print
    bool meta_checksum_enabled_ = false; // spec.integrity.enable_meta_checksum：影响 Save/Load 校验路径
    size_t max_check_iov_num_;
    std::shared_ptr<SdkBufferCheckPool> sdk_buffer_check_pool_;
#endif
};

} // namespace kv_cache_manager