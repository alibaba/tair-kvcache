#pragma once

#include <atomic>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "kv_cache_manager/config/meta_indexer_config.h"
#include "kv_cache_manager/meta/common.h"
#include "kv_cache_manager/meta/meta_storage_backend.h"

namespace kv_cache_manager {

class MetaIndexerConfig;
class MetaSearchCache;
class RequestContext;
class MetricsCollector;
class MetricsRegistry;

class MetaIndexer {
public:
    using KeyType = int64_t;
    using KeyVector = std::vector<KeyType>;
    using UriType = std::string;
    using UriVector = std::vector<UriType>;
    using PropertyMap = std::map<std::string, std::string>;
    using PropertyMapVector = std::vector<PropertyMap>;
    // Define the uri modifier function type for ReadModifyWrite
    // Parameters: uri string, get uri error code, index
    // Return value: std::pair<ModifierAction, ErrorCode>
    enum ModifierAction {
        MA_OK = 1,
        MA_FAIL = 2,
        MA_SKIP = 3,
        MA_DELETE = 4
    };
    using ModifierResult = std::pair<ModifierAction, ErrorCode>;
    using ModifierFunc = std::function<ModifierResult(std::string &, ErrorCode, size_t, PropertyMap &)>;

public:
    struct Result {
        ErrorCode ec = EC_OK;
        std::vector<ErrorCode> error_codes; // per_key_ec
        Result(ErrorCode error_code) : ec(error_code) {}
        Result(size_t count) : ec(EC_OK), error_codes(count, EC_OK) {}
    };

public:
    /// @brief Initialize the meta indexer.
    MetaIndexer() = default;
    ~MetaIndexer() = default;

    ErrorCode Init(const std::string &instance_id, const std::shared_ptr<MetaIndexerConfig> &config) noexcept;

    Result Put(RequestContext *request_context,
               const KeyVector &keys,
               UriVector &uris,
               PropertyMapVector &properties) noexcept;

    Result Update(RequestContext *request_context, const KeyVector &keys, PropertyMapVector &properties) noexcept;
    Result Update(RequestContext *request_context,
                  const KeyVector &keys,
                  UriVector &uris,
                  PropertyMapVector &properties) noexcept;
    Result
    ReadModifyWrite(RequestContext *request_context, const KeyVector &keys, const ModifierFunc &modifier) noexcept;

    Result Delete(RequestContext *request_context, const KeyVector &keys) noexcept;

    Result Exist(RequestContext *request_context, const KeyVector &keys, std::vector<bool> &out_exists) noexcept;
    Result Get(RequestContext *request_context, const KeyVector &keys, UriVector &out_uris) noexcept;
    Result Get(RequestContext *request_context,
               const KeyVector &keys,
               UriVector &out_uris,
               PropertyMapVector &out_properties) noexcept;
    Result GetProperties(RequestContext *request_context,
                         const KeyVector &keys,
                         const std::vector<std::string> &property_names,
                         PropertyMapVector &out_properties) noexcept;

    ErrorCode
    Scan(const std::string &cursor, const size_t limit, std::string &out_next_cursor, KeyVector &out_keys) noexcept;
    ErrorCode RandomSample(const size_t count, KeyVector &out_keys) noexcept;

    void PersistMetaData() noexcept;
    size_t GetKeyCount() const noexcept;
    size_t GetMaxKeyCount() const noexcept;
    size_t GetCacheUsage() const noexcept;

private:
    class ScopedBatchLock;
    struct BatchMetaData {
        std::vector<std::vector<int32_t>> batch_shard_indexs; // shard mutex index
        std::vector<std::vector<int32_t>> batch_indexs;       // raw index in KeyVector
        std::vector<KeyVector> batch_keys;
        std::vector<PropertyMapVector> batch_properties;
    };

private:
    void MakeBatches(const KeyVector &keys, PropertyMapVector &properties, BatchMetaData &batch_data) const noexcept;
    ErrorCode RecoverMetaData() noexcept;
    int32_t GetShardIndex(KeyType key) const noexcept;
    void AdjustKeyCountMeta(const int32_t delta) noexcept;
    Result DoGetWithCache(RequestContext *request_context, const KeyVector &keys, UriVector &out_uris) noexcept;
    Result DoGetWithoutCache(RequestContext *request_context, const KeyVector &keys, UriVector &out_uris) noexcept;
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

private:
    std::vector<std::unique_ptr<std::mutex>> mutex_shards_;
    std::unique_ptr<MetaStorageBackend> storage_;
    std::shared_ptr<MetaSearchCache> cache_;

    std::atomic<int64_t> key_count_ = {0};
    int64_t last_persist_metadata_time_ = 0;
    int64_t persist_metadata_interval_time_ms_ = 0;
    size_t max_key_count_ = MetaIndexerConfig::kDefaultMaxKeyCount;
    size_t mutex_shard_num_ = MetaIndexerConfig::kDefaultMutexShardNum;
    size_t batch_key_size_ = MetaIndexerConfig::kDefaultBatchKeySize;
    std::string instance_id_;

public:
    std::array<std::atomic<std::uint64_t>, 5> storage_usage_array_ = {0, 0, 0, 0, 0};
};

} // namespace kv_cache_manager
