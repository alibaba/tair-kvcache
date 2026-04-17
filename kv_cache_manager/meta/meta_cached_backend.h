#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_set>

#include "kv_cache_manager/meta/meta_local_base_backend.h"

namespace kv_cache_manager {

// Two-phase composite backend: Recover → Running.
//
// Recover: background scans persistent → PutIfAbsent into local.
//   Read: local first, miss → persistent fallback.
//   Write: persistent-first, then local. Delete records key for backfill filtering.
//
// Running: read local only, write dual-write (persistent-first).
class MetaCachedBackend : public MetaStorageBackend {
public:
    enum class RecoverState {
        kRecover,
        kRunning,
    };

    MetaCachedBackend() = default;
    ~MetaCachedBackend() override;

    std::string GetStorageType() noexcept override;

    ErrorCode Init(const std::string &instance_id,
                   const std::shared_ptr<MetaStorageBackendConfig> &config) noexcept override;
    ErrorCode Open() noexcept override;
    ErrorCode Close() noexcept override;

    // write
    std::vector<ErrorCode> Put(const KeyTypeVec &keys, const FieldMapVec &field_maps) noexcept override;
    std::vector<ErrorCode> UpdateFields(const KeyTypeVec &keys, const FieldMapVec &field_maps) noexcept override;
    std::vector<ErrorCode> Upsert(const KeyTypeVec &keys, const FieldMapVec &field_maps) noexcept override;
    std::vector<ErrorCode> Delete(const KeyTypeVec &keys) noexcept override;

    // read
    std::vector<ErrorCode> Get(const KeyTypeVec &keys,
                               const std::vector<std::string> &field_names,
                               FieldMapVec &out_field_maps) noexcept override;
    std::vector<ErrorCode> GetAllFields(const KeyTypeVec &keys, FieldMapVec &out_field_maps) noexcept override;
    std::vector<ErrorCode> Exists(const KeyTypeVec &keys, std::vector<bool> &out_is_exist_vec) noexcept override;
    ErrorCode ListKeys(const std::string &cursor,
                       const int64_t limit,
                       std::string &out_next_cursor,
                       std::vector<KeyType> &out_keys) noexcept override;
    ErrorCode RandomSample(const int64_t count, std::vector<KeyType> &out_keys) noexcept override;
    ErrorCode SampleReclaimKeys(const int64_t count, std::vector<KeyType> &out_keys) noexcept override;

    // meta data
    ErrorCode PutMetaData(const FieldMap &field_maps) noexcept override;
    ErrorCode GetMetaData(FieldMap &field_maps) noexcept override;

    RecoverState GetRecoverState() const noexcept { return recover_state_.load(std::memory_order_acquire); }

protected:
    virtual std::unique_ptr<MetaStorageBackend>
    CreatePersistentBackend(const std::string &instance_id,
                            const std::shared_ptr<MetaStorageBackendConfig> &config) const;
    virtual std::unique_ptr<MetaLocalBaseBackend>
    CreateLocalBackend(const std::string &instance_id, const std::shared_ptr<MetaStorageBackendConfig> &config) const;

private:
    void AsyncRecoverTask() noexcept;
    // Returns the number of keys successfully backfilled.
    int64_t BackfillKeysToLocal(const KeyTypeVec &keys,
                                const FieldMapVec &field_maps,
                                const std::vector<ErrorCode> &get_error_codes) noexcept;
    void EnsureKeyInLocal(const KeyTypeVec &keys) noexcept;

    std::string instance_id_;
    std::shared_ptr<MetaStorageBackendConfig> config_;
    std::unique_ptr<MetaStorageBackend> persistent_backend_;
    std::unique_ptr<MetaLocalBaseBackend> local_backend_;

    std::atomic<RecoverState> recover_state_{RecoverState::kRecover};
    std::atomic<bool> is_closed_{false};
    std::thread recover_thread_;

    mutable std::mutex deleted_keys_mutex_;
    std::unordered_set<KeyType> deleted_keys_;
};

} // namespace kv_cache_manager
