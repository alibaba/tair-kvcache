#pragma once

#include "kv_cache_manager/data_storage/data_storage_backend.h"

namespace kv_cache_manager {

class VcnsHf3fsBackend : public DataStorageBackend {
public:
    VcnsHf3fsBackend() = delete;
    explicit VcnsHf3fsBackend(std::shared_ptr<MetricsRegistry> metrics_registry);
    ~VcnsHf3fsBackend() override = default;
    DataStorageType GetType() override;
    bool Available() override;
    double GetStorageUsageRatio(const std::string &trace_id) const override;

public:
    ErrorCode DoOpen(const StorageConfig &storage_config, const std::string &trace_id) override;
    ErrorCode Close() override;

    std::vector<SpecCreateResult> Create(const CreateBlocksRequest &request,
                                        const std::string &trace_id,
                                        std::function<void()> cb) override;
    std::vector<ErrorCode> Delete(const std::vector<DataStorageUri> &storage_uris,
                                  const std::string &trace_id,
                                  std::function<void()> cb) override;
    std::vector<bool> Exist(const std::vector<DataStorageUri> &storage_uris) override;
    std::vector<ErrorCode> Lock(const std::vector<DataStorageUri> &storage_uris) override;
    std::vector<ErrorCode> UnLock(const std::vector<DataStorageUri> &storage_uris) override;
};

} // namespace kv_cache_manager
