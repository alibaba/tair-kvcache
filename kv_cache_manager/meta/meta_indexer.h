#pragma once

#include <array>
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

    // ---------- Put ----------
    // Whole-block put: replace keys[i]'s entire location set with location_maps[i].
    Result Put(RequestContext *request_context,
               const KeyVector &keys,
               LocationMapVector &location_maps,
               PropertyMapVector &properties) noexcept;
    // Whole-block update: overwrite keys[i]'s location set with location_maps[i]
    // and update its block-level properties at the same time.
    Result Update(RequestContext *request_context,
                  const KeyVector &keys,
                  LocationMapVector &location_maps,
                  PropertyMapVector &properties) noexcept;

    // ---------- Get ----------
    Result Exist(RequestContext *request_context, const KeyVector &keys, std::vector<bool> &out_exists) noexcept;
    // Whole-block get + business properties.
    Result Get(RequestContext *request_context,
               const KeyVector &keys,
               LocationMapVector &out_location_maps,
               PropertyMapVector &out_properties) noexcept;
    // Whole-block get: fetch every location of each block_key.
    Result
    GetLocations(RequestContext *request_context, const KeyVector &keys, LocationMapVector &out_location_maps) noexcept;
    // Per-location get: fetch only the specified location_ids for each block_key.
    LocationResult GetLocations(RequestContext *request_context,
                                const KeyVector &keys,
                                const LocationIdsPerKey &location_ids,
                                LocationsPerKey &out_locations) noexcept;
    // Fetch only block-level business properties (no location data is read).
    Result GetProperties(RequestContext *request_context,
                         const KeyVector &keys,
                         const std::vector<std::string> &property_names,
                         PropertyMapVector &out_properties) noexcept;

    // ---------- Modify ----------
    // Block-level RMW: modifier sees only existing location id list per key.
    Result ReadModifyWriteBlock(RequestContext *request_context,
                                const KeyVector &keys,
                                const BlockIdsOnlyModifierFunc &modifier) noexcept;
    // Location-level RMW: modifier sees per-key CacheLocation vector.
    LocationResult ReadModifyWriteLocation(RequestContext *request_context,
                                           const KeyVector &keys,
                                           const LocationIdsPerKey &location_ids,
                                           const LocationModifierFunc &modifier) noexcept;

    // ---------- Delete ----------
    Result Delete(RequestContext *request_context, const KeyVector &keys) noexcept;

    // ---------- Scan / Sample ----------
    ErrorCode
    Scan(const std::string &cursor, const size_t limit, std::string &out_next_cursor, KeyVector &out_keys) noexcept;
    ErrorCode RandomSample(RequestContext *request_context, const size_t count, KeyVector &out_keys) const noexcept;
    ErrorCode
    SampleReclaimKeys(RequestContext *request_context, const int64_t count, KeyVector &out_keys) const noexcept;

    void PersistMetaData() noexcept;
    size_t GetKeyCount() const noexcept;
    size_t GetMaxKeyCount() const noexcept;
    size_t GetMemUsage() const noexcept;

    // storage usage interfaces
    //
    // notes about backward compatibility:
    //
    // GetStorageUsage() is accurate for hybrid model arch, and should
    // only be used under InstanceVersion::VERSION_1
    //
    // similar logic for Get/Add/Sub-StorageUsageByType() are already
    // being used under InstanceVersion::VERSION_0; however they are
    // only accurate under InstanceVersion::VERSION_1
    //
    // see also notes for instance behavior version
    [[nodiscard]] std::uint64_t GetStorageUsage() const noexcept;
    [[nodiscard]] std::uint64_t GetStorageUsageByType(const DataStorageType &type) const noexcept;
    void SetStorageUsageByType(const DataStorageType &type, std::uint64_t value) noexcept;
    std::uint64_t AddStorageUsageByType(const DataStorageType &type, std::uint64_t value) noexcept;
    std::uint64_t SubStorageUsageByType(const DataStorageType &type, std::uint64_t value) noexcept;

    // instance behavior version
    enum class InstanceVersion : std::uint8_t {
        // version 0: metadata persisted = [
        //     METADATA_PROPERTY_KEY_COUNT,
        // ]
        VERSION_0 = 0,

        // version 1: metadata persisted = [
        //     METADATA_PROPERTY_KEY_COUNT,
        //     METADATA_PROPERTY_STORAGE_USAGE_DATA,
        // ]
        VERSION_1 = 1,
    };

    [[nodiscard]] InstanceVersion GetVersion() const noexcept;

private:
    class ScopedBatchLock;

    // instance storage usage data
    class StorageUsageData : public Jsonizable {
    public:
        StorageUsageData() = default;
        ~StorageUsageData() override = default;

        [[nodiscard]] std::uint64_t GetStorageUsage() const noexcept;
        [[nodiscard]] std::uint64_t GetStorageUsageByType(const DataStorageType &type) const noexcept;

        void Reset() noexcept;
        void SetStorageUsageByType(const DataStorageType &type, std::uint64_t value) noexcept;

        std::uint64_t AddStorageUsageByType(const DataStorageType &type, std::uint64_t value) noexcept;
        std::uint64_t SubStorageUsageByType(const DataStorageType &type, std::uint64_t value) noexcept;

        void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override;
        bool FromRapidValue(const rapidjson::Value &rapid_value) override;

        [[nodiscard]] std::string Serialize() const noexcept;
        ErrorCode Deserialize(const std::string &str) noexcept;

    private:
        using array_t_ = std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(DataStorageType::COUNT)>;
        using size_t_ = array_t_::size_type;

        // storage usage data array aggregated by storage type
        // slot 0: DATA_STORAGE_TYPE_UNKNOWN **UNUSED**
        // slot 1: DATA_STORAGE_TYPE_HF3FS usage data
        // slot 2: DATA_STORAGE_TYPE_MOONCAKE usage data
        // slot 3: DATA_STORAGE_TYPE_TAIR_MEMPOOL usage data
        // slot 4: DATA_STORAGE_TYPE_NFS usage data
        // slot 5: DATA_STORAGE_TYPE_VCNS_HF3FS **UNUSED** (merged into HF3FS)
        // slot 6: DATA_STORAGE_TYPE_DUMMY usage data (testing only)
        array_t_ storage_usage_by_type_;
    };

private:
    std::vector<BatchMetaData> MakeBatches(const KeyVector &keys,
                                           const LocationIdsPerKey &location_ids,
                                           LocationMapVector &locations,
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
        int64_t put_key_count = 0;    // brand-new keys created by upsert
        int64_t update_key_count = 0; // existing keys updated by upsert
        int64_t delete_key_count = 0; // keys deleted by whole-key delete
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
    InstanceVersion version_ = InstanceVersion::VERSION_1;
};

} // namespace kv_cache_manager
