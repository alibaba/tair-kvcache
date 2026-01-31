#include "stub_source/kv_cache_manager/data_storage/tair_mempool_backend.h"

#include <memory>
#include <string>
#include <utility>

#include "kv_cache_manager/common/logger.h"
namespace kv_cache_manager {

TairMempoolDataStorageItem TairMempoolDataStorageItem::FromUri(const DataStorageUri &storage_uri) {
    TairMempoolDataStorageItem item;
    storage_uri.GetParamAs<uint16_t>("media_type", item.media_type);
    storage_uri.GetParamAs<uint16_t>("node_id", item.node_id);
    std::string path = storage_uri.GetPath();
    item.offset = (path.size() < 2) ? 0 : std::stoull(path.substr(1));
    storage_uri.GetParamAs<uint16_t>("range_id", item.range_id);
    storage_uri.GetParamAs<uint64_t>("size", item.size);
    return item;
}

TairMempoolBackend::TairMempoolBackend(std::shared_ptr<MetricsRegistry> metrics_registry)
    : DataStorageBackend(std::move(metrics_registry)) {}

TairMempoolBackend::~TairMempoolBackend() { KVCM_LOG_ERROR("no implementation for TairMempoolBackend"); }

DataStorageType TairMempoolBackend::GetType() { return DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL; }

bool TairMempoolBackend::Available() {
    KVCM_LOG_ERROR("no implementation for TairMempoolBackend");
    return false;
}

double TairMempoolBackend::GetStorageUsageRatio(const std::string &trace_id) const {
    KVCM_LOG_ERROR("no implementation for TairMempoolBackend");
    return 0.0;
}

ErrorCode TairMempoolBackend::DoOpen(const StorageConfig &storage_config, const std::string &trace_id) {
    KVCM_LOG_ERROR("no implementation for TairMempoolBackend");
    return EC_ERROR;
}

ErrorCode TairMempoolBackend::Close() {
    KVCM_LOG_ERROR("no implementation for TairMempoolBackend");
    return EC_ERROR;
}

std::vector<std::pair<ErrorCode, DataStorageUri>> TairMempoolBackend::Create(const std::vector<std::string> &keys,
                                                                             size_t size_per_key,
                                                                             const std::string &trace_id,
                                                                             std::function<void()> cb) {
    KVCM_LOG_ERROR("no implementation for TairMempoolBackend");
    return {};
}

std::vector<ErrorCode> TairMempoolBackend::Delete(const std::vector<DataStorageUri> &storage_uris,
                                                  const std::string &trace_id,
                                                  std::function<void()> cb) {
    KVCM_LOG_ERROR("no implementation for TairMempoolBackend");
    std::vector<ErrorCode> result(storage_uris.size(), EC_ERROR);
    return result;
}

std::vector<bool> TairMempoolBackend::Exist(const std::vector<DataStorageUri> &storage_uris) {
    KVCM_LOG_ERROR("no implementation for TairMempoolBackend");
    std::vector<bool> result(storage_uris.size(), false);
    return result;
}
std::vector<ErrorCode> TairMempoolBackend::Lock(const std::vector<DataStorageUri> &storage_uris) {
    KVCM_LOG_ERROR("no implementation for TairMempoolBackend");
    std::vector<ErrorCode> result(storage_uris.size(), EC_ERROR);
    // not supported yet
    return result;
}
std::vector<ErrorCode> TairMempoolBackend::UnLock(const std::vector<DataStorageUri> &storage_uris) {
    KVCM_LOG_ERROR("no implementation for TairMempoolBackend");
    std::vector<ErrorCode> result(storage_uris.size(), EC_ERROR);
    // not supported yet
    return result;
}

} // namespace kv_cache_manager
