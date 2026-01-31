#include "stub_source/kv_cache_manager/client/src/internal/sdk/tair_mempool_sdk.h"

#include "kv_cache_manager/common/logger.h"
namespace kv_cache_manager {
TairMempoolRemoteItem TairMempoolRemoteItem::FromUri(const DataStorageUri &storage_uri) {
    TairMempoolRemoteItem item;
    storage_uri.GetParamAs<uint16_t>("media_type", item.media_type);
    storage_uri.GetParamAs<uint16_t>("node_id", item.node_id);
    std::string path = storage_uri.GetPath();
    item.offset = (path.size() < 2) ? 0 : std::stoull(path.substr(1));
    storage_uri.GetParamAs<uint16_t>("range_id", item.range_id);
    storage_uri.GetParamAs<uint64_t>("size", item.size);
    return item;
}

// TairMempool has not been open-sourced yet
ClientErrorCode TairMempoolSdk::Init(const std::shared_ptr<SdkBackendConfig> &sdk_backend_config,
                                     const std::shared_ptr<StorageConfig> &storage_config) {

    KVCM_LOG_ERROR("no implementation for TairMempoolSdk");
    return ER_SDKINIT_ERROR;
}

SdkType TairMempoolSdk::Type() { return SdkType::TAIR_MEMPOOL; }

ClientErrorCode TairMempoolSdk::Get(const std::vector<DataStorageUri> &remote_uris, const BlockBuffers &local_buffers) {
    KVCM_LOG_ERROR("no implementation for TairMempoolSdk");
    return ER_SDKREAD_ERROR;
}

ClientErrorCode TairMempoolSdk::Put(const std::vector<DataStorageUri> &remote_uris,
                                    const BlockBuffers &local_buffers,
                                    std::shared_ptr<std::vector<DataStorageUri>> actual_remote_uris) {
    KVCM_LOG_ERROR("no implementation for TairMempoolSdk");
    return ER_SDKWRITE_ERROR;
}

ClientErrorCode TairMempoolSdk::Alloc(const std::vector<DataStorageUri> &remote_uris,
                                      std::vector<DataStorageUri> &alloc_uris) {
    KVCM_LOG_ERROR("no implementation for TairMempoolSdk");
    return ER_SDKALLOC_ERROR;
}
} // namespace kv_cache_manager
