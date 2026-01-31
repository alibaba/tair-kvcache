#pragma once
#include <limits>
#include <memory>
#include <string>

#include "kv_cache_manager/common/jsonizable.h"
#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/config/meta_cache_policy_config.h"
#include "kv_cache_manager/config/meta_storage_backend_config.h"

namespace kv_cache_manager {

class MetaIndexerConfig : public Jsonizable {
public:
    static constexpr size_t kDefaultMaxKeyCount = std::numeric_limits<size_t>::max();
    static constexpr size_t kDefaultMutexShardNum = 16;
    static constexpr size_t kDefaultBatchKeySize = 16;
    static constexpr size_t kDefaultPersistMetaDataIntervalTimeMs = 1000;

public:
    MetaIndexerConfig()
        : max_key_count_(kDefaultMaxKeyCount)
        , mutex_shard_num_(kDefaultMutexShardNum)
        , batch_key_size_(kDefaultBatchKeySize)
        , persist_metadata_interval_time_ms_(kDefaultPersistMetaDataIntervalTimeMs)
        , meta_storage_backend_config_(std::make_shared<MetaStorageBackendConfig>())
        , meta_cache_policy_config_(std::make_shared<MetaCachePolicyConfig>()) {}

    MetaIndexerConfig(size_t max_key_count,
                      size_t mutex_shard_num,
                      size_t batch_key_size,
                      const std::shared_ptr<MetaStorageBackendConfig> &meta_storage_backend_config,
                      const std::shared_ptr<MetaCachePolicyConfig> &meta_cache_policy_config = nullptr)
        : max_key_count_(max_key_count)
        , mutex_shard_num_(mutex_shard_num)
        , batch_key_size_(batch_key_size)
        , meta_storage_backend_config_(meta_storage_backend_config)
        , meta_cache_policy_config_(meta_cache_policy_config) {}

    ~MetaIndexerConfig() override;
    size_t GetMaxKeyCount() const { return max_key_count_; }
    size_t GetMutexShardNum() const { return mutex_shard_num_; }
    size_t GetBatchKeySize() const { return batch_key_size_; }
    size_t GetPersistMetaDataIntervalTimeMs() const { return persist_metadata_interval_time_ms_; }

    const std::shared_ptr<MetaStorageBackendConfig> &GetMetaStorageBackendConfig() const {
        return meta_storage_backend_config_;
    }

    const std::shared_ptr<MetaCachePolicyConfig> &GetMetaCachePolicyConfig() const { return meta_cache_policy_config_; }
    void SetMaxKeyCount(size_t max_key_count) { max_key_count_ = max_key_count; }
    void SetMutexShardNum(size_t mutex_shard_num) { mutex_shard_num_ = mutex_shard_num; }
    void SetBatchKeySize(size_t batch_key_size) { batch_key_size_ = batch_key_size; }
    void SetPersistMetaDataIntervalTimeMs(size_t persist_metadata_interval_time_ms) {
        persist_metadata_interval_time_ms_ = persist_metadata_interval_time_ms;
    }
    void SetMetaStorageBackendConfig(const std::shared_ptr<MetaStorageBackendConfig> &meta_storage_backend_config) {
        meta_storage_backend_config_ = meta_storage_backend_config;
    }
    void SetMetaCachePolicyConfig(const std::shared_ptr<MetaCachePolicyConfig> &meta_cache_policy_config) {
        meta_cache_policy_config_ = meta_cache_policy_config;
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
        KVCM_JSON_GET_MACRO(rapid_value, "meta_cache_policy_config", meta_cache_policy_config_);
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
        Put(writer, "meta_storage_backend_config", meta_storage_backend_config_);
        Put(writer, "meta_cache_policy_config", meta_cache_policy_config_);
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
        if (meta_cache_policy_config_ == nullptr) {
            valid = false;
            local_invalid_fields += "{meta_cache_policy_config}";
        } else if (!meta_cache_policy_config_->ValidateRequiredFields(local_invalid_fields)) {
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
    std::shared_ptr<MetaCachePolicyConfig> meta_cache_policy_config_;
};
} // namespace kv_cache_manager
