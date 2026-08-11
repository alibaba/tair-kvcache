#pragma once

#include <map>
#include <memory>
#include <shared_mutex>
#include <string>
#include <thread>
#include <vector>
#include <cstdint>

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
    // 跨存储层复制：用 unique_name 对应的 backend 执行 src_uris[i] -> dst_uris[i] 的复制。
    // 可选能力，backend 不支持时返回逐项 EC_UNIMPLEMENTED；storage 不存在返回逐项 EC_NOENT。
    std::vector<ErrorCode> Copy(RequestContext *request_context,
                                const std::string &unique_name,
                                const std::vector<DataStorageUri> &src_uris,
                                const std::vector<DataStorageUri> &dst_uris);
    std::vector<bool>
    Exist(const std::string &unique_name, const std::vector<DataStorageUri> &storage_uris, bool fastpath = false);
    std::vector<ErrorCode> Lock(const std::string &unique_name, const std::vector<DataStorageUri> &storage_uris);
    std::vector<ErrorCode> UnLock(const std::string &unique_name, const std::vector<DataStorageUri> &storage_uris);

    // 写入量统计统一入口：按 unique_name 找到 backend 的 collector 并累加预估写入字节数。
    // 普通写入（StartWriteCache）与 Migration Copy 共用。
    void RecordWriteBytes(const std::string &unique_name, std::uint64_t bytes);

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
