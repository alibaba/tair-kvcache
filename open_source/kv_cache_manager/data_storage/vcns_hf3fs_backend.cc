#include "stub_source/kv_cache_manager/data_storage/vcns_hf3fs_backend.h"

#include "kv_cache_manager/common/logger.h"

namespace kv_cache_manager {

VcnsHf3fsBackend::VcnsHf3fsBackend(std::shared_ptr<MetricsRegistry> metrics_registry)
    : DataStorageBackend(std::move(metrics_registry)) {}

// use same client protocol here
DataStorageType VcnsHf3fsBackend::GetType() { return DataStorageType::DATA_STORAGE_TYPE_HF3FS; }

bool VcnsHf3fsBackend::Available() { return IsOpen() && IsAvailable(); }

double VcnsHf3fsBackend::GetStorageUsageRatio(const std::string &trace_id) const {
    KVCM_LOG_ERROR("no implementation for VcnsHf3fsBackend");
    return 0.0;
}

ErrorCode VcnsHf3fsBackend::DoOpen(const StorageConfig &storage_config, const std::string &trace_id) {
    KVCM_LOG_ERROR("no implementation for VcnsHf3fsBackend");
    return EC_ERROR;
};

ErrorCode VcnsHf3fsBackend::Close() {
    KVCM_LOG_ERROR("no implementation for VcnsHf3fsBackend");
    return EC_ERROR;
};

std::vector<std::pair<ErrorCode, DataStorageUri>> VcnsHf3fsBackend::Create(const std::vector<std::string> &keys,
                                                                           size_t size_per_key,
                                                                           const std::string &trace_id,
                                                                           std::function<void()> cb) {
    KVCM_LOG_ERROR("no implementation for TairMempoolBackend");
    return {};
};

std::vector<ErrorCode> VcnsHf3fsBackend::Delete(const std::vector<DataStorageUri> &storage_uris,
                                                const std::string &trace_id,
                                                std::function<void()> cb) {
    KVCM_LOG_ERROR("no implementation for TairMempoolBackend");
    std::vector<ErrorCode> result(storage_uris.size(), EC_ERROR);
    return result;
};

std::vector<bool> VcnsHf3fsBackend::Exist(const std::vector<DataStorageUri> &storage_uris) {
    KVCM_LOG_ERROR("no implementation for TairMempoolBackend");
    std::vector<bool> result(storage_uris.size(), false);
    return result;
}

std::vector<ErrorCode> VcnsHf3fsBackend::Lock(const std::vector<DataStorageUri> &storage_uris) {
    KVCM_LOG_ERROR("no implementation for TairMempoolBackend");
    std::vector<ErrorCode> result(storage_uris.size(), EC_ERROR);
    return result;
}

std::vector<ErrorCode> VcnsHf3fsBackend::UnLock(const std::vector<DataStorageUri> &storage_uris) {
    KVCM_LOG_ERROR("no implementation for TairMempoolBackend");
    std::vector<ErrorCode> result(storage_uris.size(), EC_ERROR);
    return result;
}

} // namespace kv_cache_manager
