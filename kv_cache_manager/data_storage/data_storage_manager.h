#pragma once

#include <map>
#include <memory>
#include <shared_mutex>
#include <string>
#include <thread>
#include <vector>

#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/data_storage/data_storage_backend.h"

namespace kv_cache_manager {

class MetricsRegistry;

class DataStorageManager {
public:
    DataStorageManager() = delete;
    explicit DataStorageManager(std::shared_ptr<MetricsRegistry> metrics_registry);
    ~DataStorageManager() = default;

    DataStorageManager(const DataStorageManager &) = delete;
    DataStorageManager &operator=(const DataStorageManager &) = delete;

public:
    std::vector<std::string> GetAllStorageNames() const;
    std::vector<std::shared_ptr<DataStorageBackend>> GetAvailableStorages();
    std::shared_ptr<DataStorageBackend> GetDataStorageBackend(const std::string &name);
    std::vector<StorageConfig> ListStorageConfig();

    ErrorCode EnableStorage(const std::string &name);
    ErrorCode DisableStorage(const std::string &name);
    ErrorCode
    RegisterStorage(RequestContext *request_context, const std::string &name, const StorageConfig &storage_config);
    ErrorCode UnRegisterStorage(const std::string &name);
    ErrorCode DoCleanup();

    std::vector<std::pair<ErrorCode, DataStorageUri>> Create(RequestContext *request_context,
                                                             const std::string &unique_name,
                                                             const std::vector<std::string> &keys,
                                                             size_t size_per_key,
                                                             std::function<void()> cb);

    std::vector<ErrorCode> Delete(RequestContext *request_context,
                                  const std::string &unique_name,
                                  const std::vector<DataStorageUri> &storage_uris,
                                  std::function<void()> cb);
    std::vector<bool> Exist(const std::string &unique_name, const std::vector<DataStorageUri> &storage_uris);
    std::vector<ErrorCode> Lock(const std::string &unique_name, const std::vector<DataStorageUri> &storage_uris);
    std::vector<ErrorCode> UnLock(const std::string &unique_name, const std::vector<DataStorageUri> &storage_uris);

private:
    std::shared_ptr<DataStorageBackend> CreateStorageBackend(const DataStorageType &type);
    std::string ToString(DataStorageType type);

private:
    mutable std::shared_mutex rw_lock_;
    std::thread heartbeat_thread_;
    // stroage unique name -> storage_backend
    std::map<std::string, std::shared_ptr<DataStorageBackend>> storage_map_;
    std::shared_ptr<MetricsRegistry> metrics_registry_;
};

} // namespace kv_cache_manager
