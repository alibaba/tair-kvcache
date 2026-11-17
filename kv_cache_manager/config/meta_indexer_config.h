#pragma once
#include <cassert>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>

#include "kv_cache_manager/common/jsonizable.h"
#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/config/meta_storage_backend_config.h"

namespace kv_cache_manager {

class MetaIndexerConfig : public Jsonizable {
public:
    static constexpr size_t kDefaultMaxKeyCount = std::numeric_limits<size_t>::max();
    static constexpr size_t kDefaultMutexShardNum = 1024;
    static constexpr size_t kDefaultBatchKeySize = 128;
    static constexpr size_t kDefaultPersistMetaDataIntervalTimeMs = 1000;

public:
    MetaIndexerConfig()
        : max_key_count_(kDefaultMaxKeyCount)
        , mutex_shard_num_(kDefaultMutexShardNum)
        , batch_key_size_(kDefaultBatchKeySize)
        , persist_metadata_interval_time_ms_(kDefaultPersistMetaDataIntervalTimeMs)
        , meta_storage_backend_config_(std::make_shared<MetaStorageBackendConfig>()) {}

    MetaIndexerConfig(size_t max_key_count,
                      size_t mutex_shard_num,
                      size_t batch_key_size,
                      const std::shared_ptr<MetaStorageBackendConfig> &meta_storage_backend_config)
        : max_key_count_(max_key_count)
        , mutex_shard_num_(mutex_shard_num)
        , batch_key_size_(batch_key_size)
        , persist_metadata_interval_time_ms_(kDefaultPersistMetaDataIntervalTimeMs)
        , meta_storage_backend_config_(meta_storage_backend_config) {}

    ~MetaIndexerConfig() override;
    size_t GetMaxKeyCount() const { return max_key_count_; }
    size_t GetMutexShardNum() const { return mutex_shard_num_; }
    size_t GetBatchKeySize() const { return batch_key_size_; }
    size_t GetPersistMetaDataIntervalTimeMs() const { return persist_metadata_interval_time_ms_; }
    size_t GetLegacySearchCacheCapacity() const { return legacy_search_cache_capacity_; }
    int32_t GetLegacySearchCacheShardBits() const { return legacy_search_cache_shard_bits_; }

    const std::shared_ptr<MetaStorageBackendConfig> &GetMetaStorageBackendConfig() const {
        return meta_storage_backend_config_;
    }

    void SetMaxKeyCount(size_t max_key_count) { max_key_count_ = max_key_count; }
    void SetMutexShardNum(size_t mutex_shard_num) { mutex_shard_num_ = mutex_shard_num; }
    void SetBatchKeySize(size_t batch_key_size) { batch_key_size_ = batch_key_size; }
    void SetLegacySearchCacheCapacity(size_t capacity) { legacy_search_cache_capacity_ = capacity; }
    void SetLegacySearchCacheShardBits(int32_t shard_bits) { legacy_search_cache_shard_bits_ = shard_bits; }
    void SetPersistMetaDataIntervalTimeMs(size_t persist_metadata_interval_time_ms) {
        persist_metadata_interval_time_ms_ = persist_metadata_interval_time_ms;
    }
    void SetMetaStorageBackendConfig(const std::shared_ptr<MetaStorageBackendConfig> &meta_storage_backend_config) {
        meta_storage_backend_config_ = meta_storage_backend_config;
    }

public:
    bool FromRapidValue(const rapidjson::Value &rapid_value) override {
        KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "max_key_count", max_key_count_, kDefaultMaxKeyCount);
        KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "mutex_shard_num", mutex_shard_num_, kDefaultMutexShardNum);
        KVCM_JSON_GET_DEFAULT_MACRO(rapid_value,
                                    "persist_metadata_interval_time_ms",
                                    persist_metadata_interval_time_ms_,
                                    kDefaultPersistMetaDataIntervalTimeMs);
        KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "batch_key_size", batch_key_size_, kDefaultBatchKeySize);
        KVCM_JSON_GET_MACRO(rapid_value, "meta_storage_backend_config", meta_storage_backend_config_);
        legacy_search_cache_capacity_ = 0;
        legacy_search_cache_shard_bits_ = 6;
        if (rapid_value.HasMember("meta_cache_policy_config") && !rapid_value["meta_cache_policy_config"].IsNull()) {
            const rapidjson::Value &legacy = rapid_value["meta_cache_policy_config"];
            if (!legacy.IsObject()) {
                return false;
            }
            KVCM_JSON_GET_DEFAULT_MACRO(legacy, "capacity", legacy_search_cache_capacity_, size_t(0));
            KVCM_JSON_GET_DEFAULT_MACRO(legacy, "cache_shard_bits", legacy_search_cache_shard_bits_, int32_t(6));
        }
        if (mutex_shard_num_ <= 0 || (mutex_shard_num_ & (mutex_shard_num_ - 1)) != 0) {
            KVCM_LOG_ERROR("mutex_shard_num[%lu] is not valid, should be 2^n", mutex_shard_num_);
            return false;
        }
        return true;
    }

    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override {
        Put(writer, "max_key_count", max_key_count_);
        Put(writer, "mutex_shard_num", mutex_shard_num_);
        Put(writer, "batch_key_size", batch_key_size_);
        Put(writer, "persist_metadata_interval_time_ms", persist_metadata_interval_time_ms_);
        Put(writer, "meta_storage_backend_config", meta_storage_backend_config_);
        // Legacy bootstrap compares these values; they never configure a runtime cache.
        [[maybe_unused]] const bool started = writer.Key("meta_cache_policy_config") && writer.StartObject();
        assert(started);
        Put(writer, "type", std::string("LRU"));
        Put(writer, "capacity", legacy_search_cache_capacity_);
        Put(writer, "cache_shard_bits", legacy_search_cache_shard_bits_);
        Put(writer, "high_pri_pool_ratio", 0.0);
        [[maybe_unused]] const bool finished = writer.EndObject();
        assert(finished);
    }
    bool ValidateRequiredFields(std::string &invalid_fields) const {
        bool valid = true;
        std::string local_invalid_fields;
        if (meta_storage_backend_config_ == nullptr) {
            valid = false;
            local_invalid_fields += "{meta_storage_backend_config}";
        } else if (!meta_storage_backend_config_->ValidateRequiredFields(local_invalid_fields)) {
            valid = false;
        }
        if (!valid) {
            invalid_fields += "{MetaIndexerConfig: " + local_invalid_fields + "}";
        }
        return valid;
    }

private:
    size_t max_key_count_;
    size_t mutex_shard_num_;
    size_t batch_key_size_;
    size_t persist_metadata_interval_time_ms_;
    std::shared_ptr<MetaStorageBackendConfig> meta_storage_backend_config_;
    size_t legacy_search_cache_capacity_ = 0;
    int32_t legacy_search_cache_shard_bits_ = 6;
};
} // namespace kv_cache_manager
