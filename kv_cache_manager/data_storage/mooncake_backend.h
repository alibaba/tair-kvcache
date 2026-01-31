#pragma once

#include <memory>

#include "3rdparty/mooncake/client_c.h"
#include "kv_cache_manager/data_storage/data_storage_backend.h"

namespace kv_cache_manager {

class MetricsRegistry;

class MooncakeBackend : public DataStorageBackend {
public:
    MooncakeBackend() = delete;
    explicit MooncakeBackend(std::shared_ptr<MetricsRegistry> metrics_registry);
    ~MooncakeBackend() override;
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
    std::vector<ErrorCode> Delete(const std::vector<DataStorageUri> &storage_uris,
                                  const std::string &trace_id,
                                  std::function<void()> cb) override;
    std::vector<bool> Exist(const std::vector<DataStorageUri> &storage_uris) override;
    std::vector<ErrorCode> Lock(const std::vector<DataStorageUri> &storage_uris) override;
    std::vector<ErrorCode> UnLock(const std::vector<DataStorageUri> &storage_uris) override;

private:
    client_t client_{nullptr};
    MooncakeStorageSpec spec_;
};

} // namespace kv_cache_manager
