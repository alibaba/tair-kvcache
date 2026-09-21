#pragma once

#include <functional>
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
    void SetAsyncCopyReferenceChecker(std::function<bool(const std::string &)> checker);
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
    bool SupportsAsyncCopy(const std::string &unique_name) const;
    AsyncCopySubmitResult CopyAsync(RequestContext *request_context,
                                    const std::string &unique_name,
                                    const std::vector<DataStorageUri> &src_uris,
                                    const std::vector<DataStorageUri> &dst_uris,
                                    const std::string &operation_id,
                                    const AsyncCopyOptions &options,
                                    AsyncCopyRemoteSubmitCompletion remote_submit_completion,
                                    AsyncCopyCompletion completion);
    AsyncCopySubmitResult ResumeAsyncCopy(RequestContext *request_context,
                                          const std::string &unique_name,
                                          const std::vector<std::string> &backend_task_ids,
                                          size_t expected_items,
                                          const std::string &operation_id,
                                          const AsyncCopyOptions &options,
                                          AsyncCopyCompletion completion);
    ErrorCode RequestCancelAsyncCopy(const std::string &unique_name, const std::string &operation_id);
    std::vector<bool>
    Exist(const std::string &unique_name, const std::vector<DataStorageUri> &storage_uris, bool fastpath = false);
    std::vector<ErrorCode> Lock(const std::string &unique_name, const std::vector<DataStorageUri> &storage_uris);
    std::vector<ErrorCode> UnLock(const std::string &unique_name, const std::vector<DataStorageUri> &storage_uris);

    // 写入量统计统一入口：按 unique_name 找到 backend 的 collector 并累加。
    // Adds `bytes` to the named storage's data_storage.write_bytes_dispatched_total
    // counter. Dispatched-view semantics: callers invoke this once a write has been
    // dispatched (locations handed to the client / copy task accepted by executor),
    // NOT when bytes are confirmed written to the backend. See the metric definition
    // in metrics/metrics_collector.h for the full semantic contract.
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
    std::function<bool(const std::string &)> async_copy_reference_checker_;
};

} // namespace kv_cache_manager
