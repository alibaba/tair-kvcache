#pragma once

#include <atomic>
#include <ctime>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/string_util.h"
#include "kv_cache_manager/common/timestamp_util.h"
#include "kv_cache_manager/config/meta_cache_policy_config.h"
#include "kv_cache_manager/meta/common.h"
#include "kv_cache_manager/meta/meta_local_base_backend.h"
#include "kv_cache_manager/meta/meta_search_cache.h"

namespace kv_cache_manager {

// 简化的FieldMap序列化器，不进行转义以提高性能
class FieldMapSerializer {
public:
    // 将FieldMap序列化为字符串，使用简单的格式
    static std::string Serialize(const MetaStorageBackend::FieldMap &field_map) {
        if (field_map.empty()) {
            return "";
        }

        std::ostringstream oss;
        bool first = true;

        for (const auto &pair : field_map) {
            if (!first) {
                oss << "\n";
            }
            oss << pair.first << "\t" << pair.second;
            first = false;
        }

        return oss.str();
    }

    // 从字符串反序列化为FieldMap
    static MetaStorageBackend::FieldMap Deserialize(const std::string &serialized) {
        MetaStorageBackend::FieldMap field_map;

        if (serialized.empty()) {
            return field_map;
        }

        std::string line;
        std::istringstream stream(serialized);

        while (std::getline(stream, line)) {
            size_t delimiter_pos = line.find('\t');
            if (delimiter_pos != std::string::npos) {
                std::string key = line.substr(0, delimiter_pos);
                std::string value = line.substr(delimiter_pos + 1);
                field_map[key] = value;
            }
        }

        return field_map;
    }
};

// 多shard的KeyTracker，使用多个队列和锁分片
class KeyTracker {
public:
    using KeyType = MetaStorageBackend::KeyType;

    explicit KeyTracker(size_t shard_num) : shard_num_(shard_num), shards_(shard_num) {}

    // 添加key到对应的shard
    void AddKey(const KeyType &key) {
        size_t shard_idx = GetShardIdx(key);
        std::lock_guard<std::mutex> lock(shards_[shard_idx].mutex_);
        auto &shard = shards_[shard_idx];
        // 检查是否已存在
        if (shard.index_map_.count(key) == 0) {
            shard.index_map_[key] = shard.keys_.size();
            shard.keys_.push_back(key);
            ++key_count_;
        }
    }

    // 访问一个key，目前不做任何操作（简化版本）
    void AccessKey(const KeyType &key) {
        // 在随机采样版本中，不需要更新访问顺序
    }

    // 删除key
    bool RemoveKey(const KeyType &key) {
        size_t shard_idx = GetShardIdx(key);
        std::lock_guard<std::mutex> lock(shards_[shard_idx].mutex_);
        auto &shard = shards_[shard_idx];
        auto iter = shard.index_map_.find(key);
        if (iter == shard.index_map_.end()) {
            return false;
        }
        int32_t index = iter->second;
        if (index < shard.keys_.size() - 1) {
            // 将最后一个元素移到被删位置
            KeyType last_key = shard.keys_.back();
            shard.keys_[index] = last_key;
            shard.index_map_[last_key] = index;
        }
        shard.keys_.pop_back();
        shard.index_map_.erase(iter);
        --key_count_;
        return true;
    }

    ErrorCode
    Scan(const std::string &cursor, const int64_t limit, std::string &out_next_cursor, std::vector<KeyType> &result) {
        // 解析 cursor 为 shard index 和 key index
        int64_t shard_idx = 0;
        int64_t key_idx = 0;

        if (!cursor.empty() && cursor != SCAN_BASE_CURSOR) {
            size_t pos = cursor.find(',');
            if (pos != std::string::npos) {
                std::string shard_idx_str = cursor.substr(0, pos);
                std::string key_idx_str = cursor.substr(pos + 1);
                bool is_success = StringUtil::StrToInt64(shard_idx_str.c_str(), shard_idx);
                is_success = is_success && StringUtil::StrToInt64(key_idx_str.c_str(), key_idx);
                if (!is_success) {
                    KVCM_LOG_ERROR("Scan fail, cannot convert shard_idx[%s] and key_idx[%s] to int64_t",
                                   shard_idx_str.c_str(),
                                   key_idx_str.c_str());
                    return EC_BADARGS;
                }
            } else {
                KVCM_LOG_ERROR("Scan fail, cannot convert cursor[%s], should be shard_idx,key_idx", cursor.c_str());
                return EC_BADARGS;
            }
        }

        size_t current_shard = shard_idx;
        size_t current_key_idx = key_idx;
        int64_t collected_count = 0;

        // 扫描直到收集到足够的 key 或者遍历完所有 shards
        while (collected_count < limit && current_shard < shard_num_) {
            auto &shard = shards_[current_shard];
            int64_t last_collected_count = collected_count;
            if (current_key_idx < shard.keys_.size()) {
                std::lock_guard<std::mutex> lock(shard.mutex_);
                // 从当前 key index 开始收集 keys
                for (size_t i = current_key_idx; i < shard.keys_.size() && collected_count < limit; ++i) {
                    result.push_back(shard.keys_[i]);
                    collected_count++;
                }
            }

            // 如果当前 shard 的 keys 已经收集完毕，移动到下一个 shard
            if (collected_count < limit) {
                current_shard++;
                current_key_idx = 0; // 下一个 shard 从头开始
            } else {
                current_key_idx += collected_count - last_collected_count;
            }
        }

        // 设置下一个游标
        if (current_shard >= shard_num_) {
            // 扫描已完成
            out_next_cursor = SCAN_BASE_CURSOR;
        } else {
            // 返回下一个扫描位置
            out_next_cursor = std::to_string(current_shard) + "," + std::to_string(current_key_idx);
        }
        return EC_OK;
    }

    // 随机采样keys，从多个shard中选择
    void RandomSample(int64_t count, std::vector<KeyType> &result) {
        int64_t start_time = TimestampUtil::GetSteadyTimeUs();
        if (count <= 0) {
            return;
        }
        count = std::min(count, key_count_.load());

        // 为了提高性能，我们随机选择shard来收集key
        static std::random_device rd;
        static std::mt19937 gen(rd());

        // 首先尝试从单个shard获取足够的key
        std::uniform_int_distribution<size_t> shard_dist(0, shard_num_ - 1);
        // 随机选择一个shard开始
        std::unordered_set<KeyType> selected_indices;
        size_t start_shard_idx = shard_dist(gen);
        size_t current_shard_idx = start_shard_idx;
        size_t remaining_count = count;
        size_t select_shard_num = 0;
        do {
            std::lock_guard<std::mutex> lock(shards_[current_shard_idx].mutex_);
            auto &shard = shards_[current_shard_idx];
            size_t shard_key_size = shard.keys_.size();

            if (shard_key_size > 0) {
                // 从当前shard获取尽可能多的key（最多随机shard_key_size次）
                size_t need_random_count = std::min(remaining_count, shard_key_size);

                // 随机选择current_random_count个key，避免重复
                std::uniform_int_distribution<size_t> key_dist(0, shard_key_size - 1);
                size_t current_select_count = 0;
                size_t current_random_count = 0;
                while (current_random_count < need_random_count) {
                    size_t random_idx = key_dist(gen);
                    KeyType current_key = shard.keys_[random_idx];
                    if (selected_indices.insert(current_key).second) {
                        result.push_back(current_key);
                        ++current_select_count;
                    }
                    ++current_random_count;
                }
                remaining_count -= current_select_count;
            }
            // 移动到下一个shard
            current_shard_idx = (current_shard_idx + 1) % shard_num_;
            ++select_shard_num;
        } while (remaining_count > 0 && current_shard_idx != start_shard_idx);
        KVCM_LOG_INFO("tianran select_shard_num: %lu, select_key_count: %lu, need_key_count: %lu, use_time[%lu]",
                      select_shard_num,
                      selected_indices.size(),
                      count,
                      TimestampUtil::GetSteadyTimeUs() - start_time);
    }

    // 获取当前key的数量
    size_t Size() const { return key_count_.load(); }

private:
    size_t GetShardIdx(const KeyType &key) const { return static_cast<uint32_t>(key) & (shard_num_ - 1); }

private:
    struct Shard {
        std::mutex mutex_;
        std::vector<KeyType> keys_;
        std::unordered_map<KeyType, int32_t> index_map_; // 快速检查key是否存在
    };

    std::atomic<int64_t> key_count_ = {0};
    size_t shard_num_;
    std::vector<Shard> shards_;
};

class MetaLocalBackend : public MetaLocalBaseBackend {
public:
    // 构造函数增加了mutex_shard_num_参数
    MetaLocalBackend(size_t mutex_shard_num = 16) : mutex_shard_num_(mutex_shard_num) {}

    ~MetaLocalBackend() = default;

    std::string GetStorageType() noexcept override;

    ErrorCode Init(const std::string &instance_id,
                   const std::shared_ptr<MetaStorageBackendConfig> &config) noexcept override;
    ErrorCode Open() noexcept override;
    ErrorCode Close() noexcept override;

    // write
    std::vector<ErrorCode> Put(const KeyTypeVec &keys, const FieldMapVec &field_maps) noexcept override;
    std::vector<ErrorCode> UpdateFields(const KeyTypeVec &keys, const FieldMapVec &field_maps) noexcept override;
    std::vector<ErrorCode> Upsert(const KeyTypeVec &keys, const FieldMapVec &field_maps) noexcept override;
    std::vector<ErrorCode> IncrFields(const KeyTypeVec &keys,
                                      const std::map<std::string, int64_t> &field_amounts) noexcept override;
    std::vector<ErrorCode> Delete(const KeyTypeVec &keys) noexcept override;
    std::vector<ErrorCode> PutIfAbsent(const KeyTypeVec &keys, const FieldMapVec &field_maps) noexcept override;

    // Conditional write: only processes keys where previous_error_codes[i] == EC_OK.
    std::vector<ErrorCode> Put(const KeyTypeVec &keys,
                               const FieldMapVec &field_maps,
                               const std::vector<ErrorCode> &previous_error_codes) noexcept override;
    std::vector<ErrorCode> UpdateFields(const KeyTypeVec &keys,
                                        const FieldMapVec &field_maps,
                                        const std::vector<ErrorCode> &previous_error_codes) noexcept override;
    std::vector<ErrorCode> Upsert(const KeyTypeVec &keys,
                                  const FieldMapVec &field_maps,
                                  const std::vector<ErrorCode> &previous_error_codes) noexcept override;
    std::vector<ErrorCode> IncrFields(const KeyTypeVec &keys,
                                      const std::map<std::string, int64_t> &field_amounts,
                                      const std::vector<ErrorCode> &previous_error_codes) noexcept override;
    std::vector<ErrorCode> Delete(const KeyTypeVec &keys,
                                  const std::vector<ErrorCode> &previous_error_codes) noexcept override;
    std::vector<ErrorCode> PutIfAbsent(const KeyTypeVec &keys,
                                       const FieldMapVec &field_maps,
                                       const std::vector<ErrorCode> &previous_error_codes) noexcept override;

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

    // meta data
    ErrorCode PutMetaData(const FieldMap &field_maps) noexcept override;
    ErrorCode GetMetaData(FieldMap &field_maps) noexcept override;

private:
    ErrorCode PutForOneKey(const KeyType &key, const FieldMap &field_map);
    ErrorCode PutIfAbsentForOneKey(const KeyType &key, const FieldMap &field_map);
    ErrorCode UpdateFieldsForOneKey(const KeyType &key, const FieldMap &field_map);
    ErrorCode UpsertForOneKey(const KeyType &key, const FieldMap &field_map);
    ErrorCode IncrFieldsForOneKey(const KeyType &key, const std::map<std::string, int64_t> &field_amounts);
    ErrorCode DeleteForOneKey(const KeyType &key);

    ErrorCode GetForOneKey(const KeyType &key, const std::vector<std::string> &field_names, FieldMap &out_field_map);
    ErrorCode GetAllFieldsForOneKey(const KeyType &key, FieldMap &out_field_map);
    ErrorCode ExistsForOneKey(const KeyType &key, bool &out_is_exist);

private:
    std::string path_;
    std::unique_ptr<MetaSearchCache> cache_;
    std::unique_ptr<KeyTracker> key_tracker_;
    std::shared_ptr<MetaCachePolicyConfig> cache_config_;
    bool enable_persistence_ = false; // 不再使用
    size_t mutex_shard_num_;
};

} // namespace kv_cache_manager
