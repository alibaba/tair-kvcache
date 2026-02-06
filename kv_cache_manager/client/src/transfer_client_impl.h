#pragma once
#ifdef USING_CUDA
#include <cuda_runtime.h>
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

    ClientErrorCode LoadKvCaches(const UriStrVec &uri_str_vec, const BlockBuffers &block_buffers) override;
    std::pair<ClientErrorCode, UriStrVec> SaveKvCaches(const UriStrVec &uri_str_vec,
                                                       const BlockBuffers &block_buffers) override;

protected:
    ClientErrorCode Init(const std::string &client_config, const InitParams &init_params) override;

private:
    ClientErrorCode IsValid(const std::unique_ptr<ClientConfig> &client_config) const;
    std::vector<DataStorageUri> ParseLocations(const UriStrVec &uri_str_vec);
    UriStrVec ConstructLocations(const std::vector<DataStorageUri> &uris);
    void PrintBlockHashAndUri(const std::string &prefix,
                              const UriStrVec &uri_str_vec,
                              const std::vector<int64_t> &block_hashs) const;

private:
    friend class TransferClient;
    std::unique_ptr<ClientConfig> client_config_;
    InitParams init_params_;
    std::unique_ptr<SdkWrapper> sdk_wrapper_;
    mutable std::shared_mutex config_mutex_;
#ifdef USING_CUDA
    bool is_check_buffer_ = false;
    size_t max_check_iov_num_;
    std::shared_ptr<SdkBufferCheckPool> sdk_buffer_check_pool_;
#endif
};

} // namespace kv_cache_manager