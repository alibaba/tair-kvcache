#pragma once

#include <memory>
#include <string>

#include "kv_cache_manager/data_storage/data_storage_backend.h"

namespace kv_cache_manager {

class MetricsRegistry;

class NfsBackend : public DataStorageBackend {
public:
    NfsBackend() = delete;
    explicit NfsBackend(std::shared_ptr<MetricsRegistry> metrics_registry);
    ~NfsBackend() override = default;
    DataStorageType GetType() override;
    bool Available() override;
    double GetStorageUsageRatio(const std::string &trace_id) const override;

public:
    ErrorCode DoOpen(const StorageConfig &storage_config, const std::string &trace_id) override;
    ErrorCode Close() override;

    std::vector<std::pair<ErrorCode, DataStorageUri>> Create(const std::vector<std::string> &keys,
                                                             size_t size_per_key,
                                                             const std::string &trace_id,
                                                             std::function<void()> cb) override;
    // NFS 写共享根路径, 真实生产无需 affinity; 这里把 preferred_node_ids[0]
    // 拼进 URI 作为 query param, 仅用于测试可观察策略层的偏好透传。
    std::vector<LocationDescriptor> CreateWithHints(const std::vector<std::string> &keys,
                                                    size_t size_per_key,
                                                    const WriteHints &hints,
                                                    bool strict,
                                                    const std::string &trace_id,
                                                    std::function<void()> cb) override;
    std::vector<ErrorCode> Delete(const std::vector<DataStorageUri> &storage_uris,
                                  const std::string &trace_id,
                                  std::function<void()> cb) override;
    std::vector<bool> Exist(const std::vector<DataStorageUri> &storage_uris) override;
    std::vector<ErrorCode> Lock(const std::vector<DataStorageUri> &storage_uris) override;
    std::vector<ErrorCode> UnLock(const std::vector<DataStorageUri> &storage_uris) override;

    // 这里默认上报"本机 IP"作为唯一节点
    std::vector<NodeMetrics> SnapshotPerNodeMetrics() const override;

private:
    NfsStorageSpec spec_;
    std::string local_node_id_;
};

} // namespace kv_cache_manager
