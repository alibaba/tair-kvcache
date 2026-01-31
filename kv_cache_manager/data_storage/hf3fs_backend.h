#pragma once

#include <filesystem>
#include <memory>

#include "kv_cache_manager/data_storage/data_storage_backend.h"

namespace kv_cache_manager {

class MetricsRegistry;

class Hf3fsBackend : public DataStorageBackend {
public:
    Hf3fsBackend() = delete;
    explicit Hf3fsBackend(std::shared_ptr<MetricsRegistry> metrics_registry);
    ~Hf3fsBackend() override = default;
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
    bool DoInit();
    bool TouchFile(const std::string &file_path);

private:
    ThreeFSStorageSpec spec_;
    std::filesystem::path base_path_;
};

} // namespace kv_cache_manager
