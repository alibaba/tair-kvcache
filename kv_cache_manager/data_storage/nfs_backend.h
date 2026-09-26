#pragma once

#include <memory>

#include "kv_cache_manager/data_storage/data_storage_backend.h"

namespace kv_cache_manager {

class MetricsRegistry;

class NfsBackend : public DataStorageBackend, public KvMetaDataStorageBackendExtension {
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
    std::vector<ErrorCode> Delete(const std::vector<DataStorageUri> &storage_uris,
                                  const std::string &trace_id,
                                  std::function<void()> cb) override;
    // Keep the legacy Delete behavior unchanged for fixed-block callers. The
    // KVMeta side interface performs and verifies exact singleton-file removal.
    std::vector<ErrorCode> DeleteForKvMeta(const std::vector<DataStorageUri> &storage_uris,
                                           const std::string &trace_id,
                                           std::function<void()> cb) override;
    bool IsKvMetaDeleteRetrySafe() const noexcept override { return true; }
    std::vector<bool> Exist(const std::vector<DataStorageUri> &storage_uris) override;
    std::vector<ErrorCode> Lock(const std::vector<DataStorageUri> &storage_uris) override;
    std::vector<ErrorCode> UnLock(const std::vector<DataStorageUri> &storage_uris) override;

protected:
    // Persist the namespace mutation before KVMeta is allowed to erase its
    // durable cleanup ledger. Virtual only for deterministic fault injection.
    virtual bool SyncKvMetaDeleteDirectory(const std::string &object_path) const noexcept;

private:
    NfsStorageSpec spec_;
};

} // namespace kv_cache_manager
