#pragma once

#include <memory>
#include <mutex>

#include "kv_cache_manager/client/include/kv_meta_transfer_client.h"
#include "kv_cache_manager/data_storage/data_storage_uri.h"
namespace kv_cache_manager {

class ClientConfig;
class SdkWrapper;

// Performs the KVMeta-only static checks that do not depend on the storage
// configs returned by RegisterInstance.  KvMetaObjectClient uses this before
// the registration RPC so a malformed data-plane config cannot leave behind a
// remotely registered instance.
ClientErrorCode ValidateKvMetaTransferClientConfig(const std::string &client_config,
                                                   const InitParams &init_params,
                                                   const std::string *expected_instance_group = nullptr,
                                                   const std::string *expected_instance_id = nullptr);

class KvMetaTransferClientImpl final : public KvMetaTransferClient {
public:
    KvMetaTransferClientImpl() = default;
    ~KvMetaTransferClientImpl() override = default;

    ClientErrorCode LoadObjects(const UriStrVec &uri_str_vec,
                                const std::vector<std::uint64_t> &value_sizes,
                                const BlockBuffers &object_buffers) override;
    std::pair<ClientErrorCode, UriStrVec> SaveObjects(const UriStrVec &uri_str_vec,
                                                      const std::vector<std::uint64_t> &value_sizes,
                                                      const BlockBuffers &object_buffers) override;

private:
    friend class KvMetaTransferClient;

    ClientErrorCode Init(const std::string &client_config,
                         const InitParams &init_params,
                         std::uint64_t max_object_bytes,
                         const SharedMemoryRegistration *shared_memory_registration);
    ClientErrorCode ValidateObjects(const UriStrVec &uri_str_vec,
                                    const std::vector<std::uint64_t> &value_sizes,
                                    const BlockBuffers &object_buffers) const;
    void SetAllowedObjectSizes(const std::vector<std::uint64_t> &value_sizes);
    static std::vector<DataStorageUri> ParseLocations(const UriStrVec &uri_str_vec);
    static UriStrVec ConstructLocations(const std::vector<DataStorageUri> &uris);

    std::unique_ptr<ClientConfig> client_config_;
    std::unique_ptr<SdkWrapper> sdk_wrapper_;
    std::uint64_t max_object_bytes_{0};
    // TairMempoolSdk's existing fixed-spec guard reads its SdkBackendConfig
    // during I/O. KVMeta updates only its private config with this batch's
    // exact sizes, so config update and SDK access must stay in one critical
    // section. Ordinary TransferClient instances are untouched.
    std::mutex io_mutex_;
};

} // namespace kv_cache_manager
