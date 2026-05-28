#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "kv_cache_manager/config/meta_indexer_config.h"
#include "kv_cache_manager/data_storage/storage_config.h"
#include "kv_cache_manager/meta/cache_location.h"
#include "kv_cache_manager/meta/common.h"
#include "kv_cache_manager/meta/meta_storage_backend_manager.h"
#include "kv_cache_manager/meta/storage_usage_data.h"
#include "kv_cache_manager/meta/types.h"

namespace kv_cache_manager {

class MetaIndexerConfig;
class MetaSearchCache;
class RequestContext;
class MetricsCollector;
class MetricsRegistry;

class MetaIndexer {
public:
    // Per-key result.
    struct Result {
        ErrorCode ec = EC_OK;
        std::vector<ErrorCode> error_codes; // per_key_ec
        explicit Result(ErrorCode error_code) : ec(error_code) {}
        explicit Result(size_t count) : ec(EC_OK), error_codes(count, EC_OK) {}
    };

    // Per-(key, location_id) result for location-granularity APIs.
    struct LocationResult {
        ErrorCode ec = EC_OK;
        // [i][j] is the ec of keys[i]'s j-th selected location.
        std::vector<std::vector<ErrorCode>> per_location_error_codes;
        explicit LocationResult(ErrorCode error_code) : ec(error_code) {}
        explicit LocationResult(const LocationIdsPerKey &location_ids) : ec(EC_OK) {
            per_location_error_codes.resize(location_ids.size());
            for (size_t i = 0; i < location_ids.size(); ++i) {
                per_location_error_codes[i].assign(location_ids[i].size(), EC_OK);
            }
        }
    };

public:
    MetaIndexer() = default;
    ~MetaIndexer();

    ErrorCode Init(const std::string &instance_id, const std::shared_ptr<MetaIndexerConfig> &config) noexcept;

    // ---------- WRITE ----------
    Result Put(RequestContext *request_context,
               const KeyVector &keys,
               CacheLocationMapVector &location_maps,
               PropertyMapVector &properties) noexcept;
    Result Delete(RequestContext *request_context, const KeyVector &keys) noexcept;
    // Block-level RMW: modifier sees only existing location id list per key.
    Result ReadModifyWriteBlock(RequestContext *request_context,
                                const KeyVector &keys,
                                const BlockIdsOnlyModifierFunc &modifier) noexcept;
    // Location-level RMW: modifier sees per-key CacheLocation vector.
    LocationResult ReadModifyWriteLocation(RequestContext *request_context,
                                           const KeyVector &keys,
                                           const LocationIdsPerKey &location_ids,
                                           const LocationModifierFunc &modifier) noexcept;

    // ---------- READ ----------
    Result Exist(RequestContext *request_context, const KeyVector &keys, std::vector<bool> &out_exists) noexcept;
    Result Get(RequestContext *request_context,
               const KeyVector &keys,
               CacheLocationMapVector &out_location_maps,
               PropertyMapVector &out_properties) noexcept;
    Result GetLocations(RequestContext *request_context,
                        const KeyVector &keys,
                        CacheLocationMapVector &out_location_maps) noexcept;
    LocationResult GetLocations(RequestContext *request_context,
                                const KeyVector &keys,
                                const LocationIdsPerKey &location_ids,
                                LocationsPerKey &out_locations) noexcept;
    Result GetProperties(RequestContext *request_context,
                         const KeyVector &keys,
                         const std::vector<std::string> &property_names,
                         PropertyMapVector &out_properties) noexcept;
    ErrorCode Scan(RequestContext *request_context,
                   const std::string &cursor,
                   const size_t limit,
                   std::string &out_next_cursor,
                   KeyVector &out_keys) noexcept;
    ErrorCode RandomSample(RequestContext *request_context, const size_t count, KeyVector &out_keys) const noexcept;
    ErrorCode
    SampleReclaimKeys(RequestContext *request_context, const int64_t count, KeyVector &out_keys) const noexcept;

    void PersistMetaData() noexcept;
    size_t GetKeyCount() const noexcept;
    size_t GetMaxKeyCount() const noexcept;
    size_t GetMemUsage() const noexcept;
    int64_t GetOldestAccessTime() const noexcept;

    // Synchronously flush pending writes for the given keys to persistent storage.
    bool Sync(const KeyVector &keys) noexcept;

    // Returns async write path stats from async backend.
    MetaStorageBackend::AsyncWriteStats GetAsyncWriteStats() noexcept;

    // storage usage interfaces
    [[nodiscard]] std::uint64_t GetStorageUsage() const noexcept;
    [[nodiscard]] std::uint64_t GetStorageUsageByType(const DataStorageType &type) const noexcept;
    void SetStorageUsageByType(const DataStorageType &type, std::uint64_t value) noexcept;
    std::uint64_t AddStorageUsageByType(const DataStorageType &type, std::uint64_t value) noexcept;
    std::uint64_t SubStorageUsageByType(const DataStorageType &type, std::uint64_t value) noexcept;

private:
    class ScopedBatchLock;

private:
    std::vector<BatchMetaData> MakeBatches(const KeyVector &keys,
                                           const LocationIdsPerKey &location_ids,
                                           CacheLocationMapVector &locations,
                                           PropertyMapVector &properties) const noexcept;

    ErrorCode RecoverMetaData() noexcept;
    void AdjustKeyCountMeta(const int32_t delta) noexcept;
    int32_t ProcessErrorCodes(const std::string &trace_id,
                              const std::vector<ErrorCode> &error_codes,
                              const std::vector<int32_t> &indexs,
                              const KeyVector &keys,
                              const std::string &op_name,
                              Result &result) const noexcept;
    void ProcessErrorResult(const std::string &trace_id,
                            const std::string &op_name,
                            const int32_t error_count,
                            const int32_t key_count,
                            Result &result) const noexcept;

    // ----- ReadModifyWrite helpers -----
    struct RmwStats {
        int64_t get_io_time_us = 0;
        int64_t upsert_io_time_us = 0;
        int64_t delete_io_time_us = 0;
        int64_t index_serialize_time_us = 0;
        int64_t index_deserialize_time_us = 0;
        bool has_index_deserialize = false;
        int64_t lock_wait_time_us = 0; // accumulated time waiting for shard locks
        int64_t async_enqueue_timeout_key_count = 0;
        int64_t async_enqueue_time_us = 0;
        int64_t cache_backend_upsert_time_us = 0;
        int64_t cache_backend_delete_time_us = 0;
        int64_t put_key_count = 0;     // brand-new keys created by upsert
        int64_t update_key_count = 0;  // existing keys updated by upsert
        int64_t delete_key_count = 0;  // keys deleted by whole-key delete
    };
    // Returns {error_count, put_success_count}.
    std::pair<int32_t, int32_t> ExecuteRmwUpsert(const std::string &trace_id,
                                                 RequestContext *request_context,
                                                 BatchMetaData &upsert_batch,
                                                 const std::vector<int32_t> &put_global_indexs,
                                                 const KeyVector &all_keys,
                                                 RmwStats &stats,
                                                 Result &result) noexcept;
    // Returns {error_count, delete_success_count}.
    std::pair<int32_t, int32_t> ExecuteRmwDelete(const std::string &trace_id,
                                                 RequestContext *request_context,
                                                 const BatchMetaData &delete_batch,
                                                 const KeyVector &all_keys,
                                                 RmwStats &stats,
                                                 Result &result) noexcept;
    void
    EmitRmwMetrics(MetricsCollector *metrics_collector, const RmwStats &stats, size_t total_key_count) const noexcept;

private:
    std::vector<std::unique_ptr<std::mutex>> mutex_shards_;
    std::unique_ptr<MetaStorageBackendManager> backend_manager_;

    std::atomic<int64_t> key_count_ = {0};
    int64_t last_persist_metadata_time_ = 0;
    int64_t persist_metadata_interval_time_ms_ = 0;
    size_t max_key_count_ = MetaIndexerConfig::kDefaultMaxKeyCount;
    size_t mutex_shard_mask_ = MetaIndexerConfig::kDefaultMutexShardNum - 1;
    size_t batch_key_size_ = MetaIndexerConfig::kDefaultBatchKeySize;
    std::string instance_id_;
    StorageUsageData storage_usage_data_;
};

} // namespace kv_cache_manager
