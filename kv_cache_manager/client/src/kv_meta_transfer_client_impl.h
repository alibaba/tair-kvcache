#pragma once

#include <memory>

#include "kv_cache_manager/client/include/kv_meta_transfer_client.h"
#include "kv_cache_manager/data_storage/data_storage_uri.h"

namespace kv_cache_manager {

class ClientConfig;
class SdkWrapper;

class KvMetaTransferClientImpl final : public KvMetaTransferClient {
public:
    KvMetaTransferClientImpl() = default;
    ~KvMetaTransferClientImpl() override = default;

    ClientErrorCode LoadObjects(const UriStrVec &uri_str_vec,
                                const std::vector<std::uint64_t> &value_sizes,
                                const BlockBuffers &object_buffers) override;
    std::pair<ClientErrorCode, UriStrVec>
    SaveObjects(const UriStrVec &uri_str_vec,
                const std::vector<std::uint64_t> &value_sizes,
                const BlockBuffers &object_buffers) override;

private:
    friend class KvMetaTransferClient;

    ClientErrorCode Init(const std::string &client_config,
                         const InitParams &init_params,
                         std::uint64_t max_object_bytes,
                         const SharedMemoryRegistration *shared_memory_registration);
    static std::vector<DataStorageUri> ParseLocations(const UriStrVec &uri_str_vec);
    static UriStrVec ConstructLocations(const std::vector<DataStorageUri> &uris);

    std::unique_ptr<ClientConfig> client_config_;
    std::unique_ptr<SdkWrapper> sdk_wrapper_;
};

} // namespace kv_cache_manager
