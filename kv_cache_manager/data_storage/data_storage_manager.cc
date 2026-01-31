#include "data_storage_manager.h"

#include <memory>
#include <string>
#include <utility>

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/data_storage/hf3fs_backend.h"
#include "kv_cache_manager/data_storage/mooncake_backend.h"
#include "kv_cache_manager/data_storage/nfs_backend.h"
#include "kv_cache_manager/data_storage/storage_config.h"
#include "kv_cache_manager/metrics/metrics_collector.h"
#include "kv_cache_manager/metrics/metrics_registry.h"
#include "stub_source/kv_cache_manager/data_storage/tair_mempool_backend.h"
#include "stub_source/kv_cache_manager/data_storage/vcns_hf3fs_backend.h"

namespace kv_cache_manager {

DataStorageManager::DataStorageManager(std::shared_ptr<MetricsRegistry> metrics_registry)
    : metrics_registry_(std::move(metrics_registry)) {}

std::vector<std::string> DataStorageManager::GetAllStorageNames() const {
    std::shared_lock<std::shared_mutex> lock(rw_lock_);
    std::vector<std::string> all;
    std::for_each(storage_map_.begin(), storage_map_.end(), [&all](auto const &pair) {
        if (pair.second) {
            all.emplace_back(pair.first);
        }
    });
    return all;
}

std::vector<std::shared_ptr<DataStorageBackend>> DataStorageManager::GetAvailableStorages() {
    // 改到后台更新
    std::shared_lock<std::shared_mutex> lock(rw_lock_);
    std::vector<std::shared_ptr<DataStorageBackend>> availableStorages;
    std::for_each(storage_map_.begin(), storage_map_.end(), [&availableStorages](auto const &pair) {
        if (pair.second && pair.second->Available()) {
            availableStorages.emplace_back(pair.second);
        }
    });
    return availableStorages;
}

std::vector<StorageConfig> DataStorageManager::ListStorageConfig() {
    std::shared_lock<std::shared_mutex> lock(rw_lock_);
    std::vector<StorageConfig> result;
    std::for_each(storage_map_.begin(), storage_map_.end(), [&result](auto const &pair) {
        // TODO: check available
        result.push_back(pair.second->GetStorageConfig());
    });
    return result;
}

std::shared_ptr<DataStorageBackend> DataStorageManager::GetDataStorageBackend(const std::string &name) {
    std::shared_lock<std::shared_mutex> lock(rw_lock_);
    auto it = storage_map_.find(name);
    if (it != storage_map_.end()) {
        return it->second;
    }
    KVCM_LOG_WARN("GetDataStorageBackend failed, name: %s not exist", name.c_str());
    return nullptr;
}

ErrorCode DataStorageManager::EnableStorage(const std::string &name) {
    std::unique_lock<std::shared_mutex> lock(rw_lock_);
    auto iter = storage_map_.find(name);
    if (iter == storage_map_.end()) {
        KVCM_LOG_WARN("EnableStorage failed, name: %s not exist", name.c_str());
        return EC_NOENT;
    }
    iter->second->SetAvailable(true);
    KVCM_LOG_INFO("storage [%s] is enabled", name.c_str());
    return EC_OK;
}

ErrorCode DataStorageManager::DisableStorage(const std::string &name) {
    std::unique_lock<std::shared_mutex> lock(rw_lock_);
    auto iter = storage_map_.find(name);
    if (iter == storage_map_.end()) {
        KVCM_LOG_WARN("DisableStorage failed, name: %s not exist", name.c_str());
        return EC_NOENT;
    }
    iter->second->SetAvailable(false);
    KVCM_LOG_INFO("storage [%s] is disabled", name.c_str());
    return EC_OK;
}

ErrorCode DataStorageManager::RegisterStorage(RequestContext *request_context,
                                              const std::string &name,
                                              const StorageConfig &storage_config) {
    SPAN_TRACER(request_context);
    std::unique_lock<std::shared_mutex> lock(rw_lock_);
    const std::string &trace_id = request_context->trace_id();
    if (storage_map_.find(name) != storage_map_.end()) {
        KVCM_LOG_WARN("RegisterStorage failed, name: %s already exist", name.c_str());
        return EC_EXIST;
    }
    std::shared_ptr<DataStorageBackend> storage_backend = CreateStorageBackend(storage_config.type());
    if (storage_backend == nullptr) {
        KVCM_LOG_WARN("RegisterStorage failed, name: %s, type: %d, create storage backend failed",
                      name.c_str(),
                      static_cast<uint8_t>(storage_config.type()));
        return EC_ERROR;
    }
    auto ec = storage_backend->Open(storage_config, trace_id);
    if (ec != EC_OK) {
        KVCM_LOG_WARN("RegisterStorage failed, name: %s, type: %d, open storage backend failed with error code: %d",
                      name.c_str(),
                      static_cast<uint8_t>(storage_config.type()),
                      ec);
        return ec;
    }
    KVCM_LOG_INFO("RegisterStorage success, name: %s", name.c_str());
    storage_map_[name] = storage_backend;
    return ec;
}

ErrorCode DataStorageManager::UnRegisterStorage(const std::string &name) {
    std::unique_lock<std::shared_mutex> lock(rw_lock_);
    auto iter = storage_map_.find(name);
    if (iter == storage_map_.end()) {
        KVCM_LOG_WARN("UnRegisterStorage failed, name: %s not exist", name.c_str());
        return EC_NOENT;
    }
    auto ec = iter->second->Close();
    if (ec != EC_OK) {
        KVCM_LOG_WARN("UnRegisterStorage failed, name: %s, type: %d, close storage backend failed with error code: %d",
                      name.c_str(),
                      static_cast<uint8_t>(iter->second->GetType()),
                      ec);
        return ec;
    }
    storage_map_.erase(iter);
    return ec;
}

ErrorCode DataStorageManager::DoCleanup() {
    std::unique_lock<std::shared_mutex> lock(rw_lock_);
    KVCM_LOG_INFO("data storage manager start cleanup");
    ErrorCode result = EC_OK;

    // clean all storage in DataStorageManager
    for (const auto &pair : storage_map_) {
        auto ec = pair.second->Close();
        if (ec != EC_OK) {
            KVCM_LOG_WARN("close storage backend name: %s, type: %d, failed with error code: %d",
                          pair.first.c_str(),
                          static_cast<uint8_t>(pair.second->GetType()),
                          ec);
            result = ec;
        }
    }
    storage_map_.clear();

    KVCM_LOG_INFO("data storage manager cleanup completed");
    return result;
}

std::shared_ptr<DataStorageBackend> DataStorageManager::CreateStorageBackend(const DataStorageType &type) {
    switch (type) {
    case DataStorageType::DATA_STORAGE_TYPE_HF3FS:
        return std::make_shared<Hf3fsBackend>(metrics_registry_);
    case DataStorageType::DATA_STORAGE_TYPE_VCNS_HF3FS:
        return std::make_shared<VcnsHf3fsBackend>(metrics_registry_);
    case DataStorageType::DATA_STORAGE_TYPE_MOONCAKE:
        return std::make_shared<MooncakeBackend>(metrics_registry_);
    case DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL:
        return std::make_shared<TairMempoolBackend>(metrics_registry_);
    case DataStorageType::DATA_STORAGE_TYPE_NFS:
        return std::make_shared<NfsBackend>(metrics_registry_);
    default:
        return nullptr;
    }
}

std::vector<std::pair<ErrorCode, DataStorageUri>> DataStorageManager::Create(RequestContext *request_context,
                                                                             const std::string &unique_name,
                                                                             const std::vector<std::string> &keys,
                                                                             size_t size_per_key,
                                                                             std::function<void()> cb) {
    SPAN_TRACER(request_context);
    std::shared_lock<std::shared_mutex> lock(rw_lock_);
    const std::string &trace_id = request_context->trace_id();
    auto iter = storage_map_.find(unique_name);
    if (iter == storage_map_.end()) {
        KVCM_LOG_WARN("Storage name: %s not exist", unique_name.c_str());
        return {};
    }
    auto storage_backend = iter->second;
    const auto dsmc = storage_backend->GetMetricsCollector();
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_BEGIN(dsmc, DataStorageCreate);
    std::vector<std::pair<ErrorCode, DataStorageUri>> create_result =
        storage_backend->Create(keys, size_per_key, trace_id, cb);
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_END(dsmc, DataStorageCreate);
    KVCM_METRICS_COLLECTOR_SET_METRICS(dsmc, data_storage, create_keys_qps, keys.size());
    if (request_context) {
        request_context->GetMetricsCollectorsVehicle().AddMetricsCollector(dsmc);
    }
    std::for_each(create_result.begin(), create_result.end(), [&unique_name](auto &pair) {
        if (pair.first == EC_OK) {
            pair.second.SetHostName(unique_name);
        }
    });
    return create_result;
}

std::vector<ErrorCode> DataStorageManager::Delete(RequestContext *request_context,
                                                  const std::string &unique_name,
                                                  const std::vector<DataStorageUri> &storage_uris,
                                                  std::function<void()> cb) {
    SPAN_TRACER(request_context);
    if (storage_uris.empty()) {
        return {};
    }
    std::shared_lock<std::shared_mutex> lock(rw_lock_);
    const std::string &trace_id = request_context->trace_id();
    auto iter = storage_map_.find(unique_name);
    if (iter == storage_map_.end()) {
        KVCM_LOG_WARN("Storage name: %s not exist", unique_name.c_str());
        return {};
    }
    auto storage_backend = iter->second;
    return storage_backend->Delete(storage_uris, trace_id, cb);
}

std::vector<bool> DataStorageManager::Exist(const std::string &unique_name,
                                            const std::vector<DataStorageUri> &storage_uris) {
    std::shared_lock<std::shared_mutex> lock(rw_lock_);
    auto iter = storage_map_.find(unique_name);
    if (iter == storage_map_.end()) {
        KVCM_LOG_WARN("Storage name: %s not exist", unique_name.c_str());
        return {};
    }
    auto storage_backend = iter->second;
    return storage_backend->Exist(storage_uris);
}

std::vector<ErrorCode> DataStorageManager::Lock(const std::string &unique_name,
                                                const std::vector<DataStorageUri> &storage_uris) {
    std::shared_lock<std::shared_mutex> lock(rw_lock_);
    auto iter = storage_map_.find(unique_name);
    if (iter == storage_map_.end()) {
        KVCM_LOG_WARN("Storage name: %s not exist", unique_name.c_str());
        return {};
    }
    auto storage_backend = iter->second;
    return storage_backend->Lock(storage_uris);
}

std::vector<ErrorCode> DataStorageManager::UnLock(const std::string &unique_name,
                                                  const std::vector<DataStorageUri> &storage_uris) {
    std::shared_lock<std::shared_mutex> lock(rw_lock_);
    auto iter = storage_map_.find(unique_name);
    if (iter == storage_map_.end()) {
        KVCM_LOG_WARN("Storage name: %s not exist", unique_name.c_str());
        return {};
    }
    auto storage_backend = iter->second;
    return storage_backend->UnLock(storage_uris);
}
} // namespace kv_cache_manager