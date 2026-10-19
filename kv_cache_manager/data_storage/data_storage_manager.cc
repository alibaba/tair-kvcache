#include "data_storage_manager.h"

#include <memory>
#include <string>
#include <utility>

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/data_storage/dummy_backend.h"
#include "kv_cache_manager/data_storage/event_report_backend.h"
#include "kv_cache_manager/data_storage/hf3fs_backend.h"
#ifdef ENABLE_MOONCAKE
#include "kv_cache_manager/data_storage/mooncake_backend.h"
#endif
#include "kv_cache_manager/data_storage/nfs_backend.h"
#include "kv_cache_manager/data_storage/storage_config.h"
#include "kv_cache_manager/metrics/metrics_collector.h"
#include "kv_cache_manager/metrics/metrics_registry.h"
#include "stub_source/kv_cache_manager/data_storage/tair_mempool_backend.h"
#ifdef ENABLE_VCNS
#include "stub_source/kv_cache_manager/data_storage/vcns_hf3fs_backend.h"
#endif

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
    // Serializes with local Copy acceptance/publication and physical Create/Delete,
    // but never waits for a remote Copy or for a pending cleanup future.
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

StorageBackendSnapshot DataStorageManager::CaptureStorageBackends(const std::vector<std::string> &names) const {
    std::shared_lock<std::shared_mutex> lock(rw_lock_);
    if (names.empty()) {
        return std::make_shared<const std::map<std::string, std::shared_ptr<DataStorageBackend>>>(storage_map_);
    }
    auto snapshot = std::make_shared<std::map<std::string, std::shared_ptr<DataStorageBackend>>>();
    for (const auto &name : names) {
        const auto it = storage_map_.find(name);
        snapshot->emplace(name, it == storage_map_.end() ? nullptr : it->second);
    }
    return snapshot;
}

bool DataStorageManager::StorageBackendCurrentLocked(const StorageBackendSnapshot &snapshot,
                                                     const std::string &name) const {
    if (!snapshot) {
        return true; // Legacy callers do not bind a registration.
    }
    const auto expected = snapshot->find(name);
    const auto current = storage_map_.find(name);
    return expected != snapshot->end() && expected->second && current != storage_map_.end() &&
           current->second == expected->second;
}

bool DataStorageManager::StorageBackendsCurrentLocked(const StorageBackendSnapshot &snapshot) const {
    return !snapshot || std::all_of(snapshot->begin(), snapshot->end(), [&](const auto &entry) {
        return StorageBackendCurrentLocked(snapshot, entry.first);
    });
}

bool DataStorageManager::IsStorageBackendCurrent(const StorageBackendSnapshot &snapshot,
                                                 const std::string &name) const {
    std::shared_lock<std::shared_mutex> lock(rw_lock_);
    return StorageBackendCurrentLocked(snapshot, name);
}

bool DataStorageManager::WithCurrentStorageBackends(const StorageBackendSnapshot &snapshot,
                                                    const std::function<void()> &action) const {
    std::shared_lock<std::shared_mutex> lock(rw_lock_);
    if (!StorageBackendsCurrentLocked(snapshot)) {
        return false;
    }
    // action must not call back into DataStorageManager. This is a local
    // admission/publication barrier, never a wait for Copy completion.
    action();
    return true;
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
#ifdef ENABLE_VCNS
    case DataStorageType::DATA_STORAGE_TYPE_VCNS_HF3FS:
        return std::make_shared<VcnsHf3fsBackend>(metrics_registry_);
#endif
#ifdef ENABLE_MOONCAKE
    case DataStorageType::DATA_STORAGE_TYPE_MOONCAKE:
        return std::make_shared<MooncakeBackend>(metrics_registry_);
#endif
    case DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL:
    case DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL_SSD:
        return std::make_shared<TairMempoolBackend>(metrics_registry_);
    case DataStorageType::DATA_STORAGE_TYPE_NFS:
        return std::make_shared<NfsBackend>(metrics_registry_);
    case DataStorageType::DATA_STORAGE_TYPE_DUMMY:
        return std::make_shared<DummyBackend>(metrics_registry_);
    case DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L1P5:
    case DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2:
        return std::make_shared<EventReportBackend>(metrics_registry_);
    default:
        return nullptr;
    }
}

std::vector<std::pair<ErrorCode, DataStorageUri>> DataStorageManager::Create(RequestContext *request_context,
                                                                             const std::string &unique_name,
                                                                             const std::vector<std::string> &keys,
                                                                             size_t size_per_key,
                                                                             std::function<void()> cb,
                                                                             const StorageBackendSnapshot &snapshot) {
    SPAN_TRACER(request_context);
    std::shared_lock<std::shared_mutex> lock(rw_lock_);
    if (!StorageBackendsCurrentLocked(snapshot)) {
        return std::vector<std::pair<ErrorCode, DataStorageUri>>(keys.size(), {EC_NOENT, DataStorageUri{}});
    }
    const std::string &trace_id = request_context->trace_id();
    auto iter = storage_map_.find(unique_name);
    if (iter == storage_map_.end()) {
        KVCM_LOG_WARN("Storage name: %s not exist", unique_name.c_str());
        return {};
    }
    auto storage_backend = iter->second;
    // DisableStorage and this check are serialized by rw_lock_. Keep the lock through Create so a
    // target cannot become disabled between admission and backend allocation.
    if (storage_backend == nullptr || !storage_backend->Writable()) {
        KVCM_LOG_WARN("Storage name: %s is not writable, reject create", unique_name.c_str());
        return std::vector<std::pair<ErrorCode, DataStorageUri>>(keys.size(), {EC_NOENT, DataStorageUri{}});
    }
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
                                                  std::function<void()> cb,
                                                  const StorageBackendSnapshot &snapshot,
                                                  bool *storage_removed) {
    SPAN_TRACER(request_context);
    if (storage_uris.empty()) {
        return {};
    }
    std::shared_lock<std::shared_mutex> lock(rw_lock_);
    if (storage_removed) {
        *storage_removed = false;
    }
    if (!StorageBackendCurrentLocked(snapshot, unique_name)) {
        if (storage_removed) {
            *storage_removed = true;
        }
        return {}; // Retirement is not a successful physical Delete.
    }
    const std::string &trace_id = request_context->trace_id();
    auto iter = storage_map_.find(unique_name);
    if (iter == storage_map_.end()) {
        KVCM_LOG_WARN("Storage name: %s not exist", unique_name.c_str());
        return {};
    }
    auto storage_backend = iter->second;
    return storage_backend->Delete(storage_uris, trace_id, cb);
}

std::vector<ErrorCode> DataStorageManager::Copy(RequestContext *request_context,
                                                const std::string &unique_name,
                                                const std::vector<DataStorageUri> &src_uris,
                                                const std::vector<DataStorageUri> &dst_uris,
                                                const StorageBackendSnapshot &snapshot) {
    SPAN_TRACER(request_context);
    if (src_uris.size() != dst_uris.size()) {
        KVCM_LOG_WARN("Copy src/dst size mismatch, storage: %s, src: %zu, dst: %zu",
                      unique_name.c_str(),
                      src_uris.size(),
                      dst_uris.size());
        return std::vector<ErrorCode>(src_uris.size(), ErrorCode::EC_BADARGS);
    }
    if (src_uris.empty()) {
        return {};
    }
    std::shared_lock<std::shared_mutex> lock(rw_lock_);
    if (!StorageBackendsCurrentLocked(snapshot)) {
        return std::vector<ErrorCode>(src_uris.size(), EC_NOENT);
    }
    const std::string &trace_id = request_context->trace_id();
    auto iter = storage_map_.find(unique_name);
    if (iter == storage_map_.end()) {
        KVCM_LOG_WARN("Storage name: %s not exist", unique_name.c_str());
        return std::vector<ErrorCode>(src_uris.size(), ErrorCode::EC_NOENT);
    }
    auto storage_backend = iter->second;
    return storage_backend->Copy(src_uris, dst_uris, trace_id);
}

bool DataStorageManager::SupportsAsyncCopy(const std::string &unique_name) const {
    std::shared_ptr<DataStorageBackend> storage_backend;
    {
        std::shared_lock<std::shared_mutex> lock(rw_lock_);
        const auto iter = storage_map_.find(unique_name);
        if (iter == storage_map_.end()) {
            return false;
        }
        storage_backend = iter->second;
    }
    return storage_backend != nullptr && storage_backend->SupportsAsyncCopy();
}

AsyncCopySubmitResult DataStorageManager::CopyAsync(RequestContext *request_context,
                                                    const std::string &unique_name,
                                                    const std::vector<DataStorageUri> &src_uris,
                                                    const std::vector<DataStorageUri> &dst_uris,
                                                    const std::string &operation_id,
                                                    const AsyncCopyOptions &options,
                                                    AsyncCopyCompletion completion,
                                                    const StorageBackendSnapshot &snapshot) {
    SPAN_TRACER(request_context);
    AsyncCopySubmitResult rejected;
    rejected.operation_id = operation_id;
    if (src_uris.size() != dst_uris.size() || src_uris.empty() || !completion) {
        rejected.status = EC_BADARGS;
        rejected.detail = "invalid asynchronous copy input";
        return rejected;
    }

    // CopyAsync only hands work to the local coordinator. Keep this short
    // handoff serialized with Close, without retaining the lock in completion.
    std::shared_lock<std::shared_mutex> lock(rw_lock_);
    if (!StorageBackendsCurrentLocked(snapshot)) {
        rejected.status = EC_NOENT;
        rejected.detail = "copy storage registration was removed";
        return rejected;
    }
    std::shared_ptr<DataStorageBackend> storage_backend;
    {
        const auto iter = storage_map_.find(unique_name);
        if (iter == storage_map_.end() || !iter->second) {
            rejected.status = EC_NOENT;
            rejected.detail = "storage does not exist";
            return rejected;
        }
        storage_backend = iter->second;
        if (!storage_backend->Available()) {
            rejected.status = EC_NOENT;
            rejected.detail = "storage is unavailable";
            return rejected;
        }
    }

    if (!storage_backend->SupportsAsyncCopy()) {
        rejected.status = EC_UNIMPLEMENTED;
        rejected.detail = "storage backend does not support asynchronous copy";
        return rejected;
    }
    const std::string trace_id = request_context ? request_context->trace_id() : std::string();
    // Keep the backend alive until the exactly-once completion callback.  The
    // local shared_ptr above only protects the submit call itself; without the
    // capture an unregister could destroy the coordinator while PACE is still
    // executing the operation.
    auto completion_with_backend_lease =
        [storage_backend, completion = std::move(completion)](AsyncCopyBatchResult result) mutable {
            completion(std::move(result));
        };
    return storage_backend->CopyAsync(src_uris,
                                      dst_uris,
                                      operation_id,
                                      trace_id,
                                      options,
                                      std::move(completion_with_backend_lease));
}

ErrorCode DataStorageManager::RequestCancelAsyncCopy(const std::string &unique_name,
                                                     const std::string &operation_id,
                                                     const StorageBackendSnapshot &snapshot) {
    if (operation_id.empty()) {
        return EC_BADARGS;
    }
    std::shared_lock<std::shared_mutex> lock(rw_lock_);
    if (!StorageBackendCurrentLocked(snapshot, unique_name)) {
        return EC_NOENT;
    }
    std::shared_ptr<DataStorageBackend> storage_backend;
    {
        const auto iter = storage_map_.find(unique_name);
        if (iter == storage_map_.end() || !iter->second) {
            return EC_NOENT;
        }
        storage_backend = iter->second;
    }
    return storage_backend->RequestCancelAsyncCopy(operation_id);
}

std::vector<bool> DataStorageManager::Exist(const std::string &unique_name,
                                            const std::vector<DataStorageUri> &storage_uris,
                                            bool fastpath) {
    std::shared_lock<std::shared_mutex> lock(rw_lock_);
    auto iter = storage_map_.find(unique_name);
    if (iter == storage_map_.end()) {
        KVCM_LOG_WARN("Storage name: %s not exist", unique_name.c_str());
        return {};
    }
    auto storage_backend = iter->second;
    return fastpath ? storage_backend->MightExist(storage_uris) : storage_backend->Exist(storage_uris);
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

void DataStorageManager::RecordWriteBytes(const std::string &unique_name, std::uint64_t bytes) {
    if (bytes == 0) {
        return;
    }
    std::shared_lock<std::shared_mutex> lock(rw_lock_);
    auto iter = storage_map_.find(unique_name); // iter->second 指向 DataStorageBackend 对象
    if (iter == storage_map_.end() || iter->second == nullptr) {
        KVCM_LOG_WARN("RecordWriteBytes: storage [%s] not found, drop %llu bytes",
                      unique_name.c_str(), static_cast<unsigned long long>(bytes));
        return;
    }
    const auto collector = iter->second->GetMetricsCollector(); // 指向 DataStorageMetricsCollector 对象
    if (collector == nullptr) {
        return;
    }
    collector->AddWriteBytes(bytes);
}

} // namespace kv_cache_manager
