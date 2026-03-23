#include "kv_cache_manager/meta/meta_local_backend.h"

#include <utility>

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/config/meta_storage_backend_config.h"

namespace kv_cache_manager {

std::string MetaLocalBackend::GetStorageType() noexcept { return "local"; }

ErrorCode MetaLocalBackend::Init(const std::string &instance_id,
                                 const std::shared_ptr<MetaStorageBackendConfig> &config) noexcept {
    if (instance_id.empty()) {
        KVCM_LOG_ERROR("fail to init meta redis backend, invalid empty instance id");
        return EC_BADARGS;
    }
    if (!config) {
        KVCM_LOG_ERROR("fail to init meta local backend, invalid nullptr config");
        return EC_BADARGS;
    }
    // 创建默认的MetaCachePolicyConfig
    cache_config_ = std::make_shared<MetaCachePolicyConfig>();
    // 这里可以根据需要设置特定的配置参数
    cache_config_->SetCapacity(1 * 1024 * 1024); // 1TB
    cache_config_->SetCacheShardBits(10);
    // 初始化cache和key_tracker，使用指定的shard数
    cache_ = std::make_unique<MetaSearchCache>();
    key_tracker_ = std::make_unique<KeyTracker>(config->GetMutexShardNum());

    return EC_OK;
}

ErrorCode MetaLocalBackend::Open() noexcept {
    if (!cache_config_) {
        KVCM_LOG_ERROR("Config is not initialized");
        return EC_ERROR;
    }

    ErrorCode ret = cache_->Init(cache_config_);
    if (ret != EC_OK) {
        KVCM_LOG_ERROR("Failed to initialize MetaSearchCache: %d", static_cast<int>(ret));
        return ret;
    }

    return EC_OK;
}

ErrorCode MetaLocalBackend::Close() noexcept {
    cache_.reset();
    key_tracker_.reset();
    return EC_OK;
}

std::vector<ErrorCode> MetaLocalBackend::Put(const KeyTypeVec &keys, const FieldMapVec &field_maps) noexcept {
    std::vector<ErrorCode> results(keys.size(), EC_OK);

    for (size_t i = 0; i < keys.size(); ++i) {
        results[i] = PutForOneKey(keys[i], field_maps[i]);
    }

    return results;
}

std::vector<ErrorCode> MetaLocalBackend::UpdateFields(const KeyTypeVec &keys, const FieldMapVec &field_maps) noexcept {
    std::vector<ErrorCode> results(keys.size(), EC_OK);

    for (size_t i = 0; i < keys.size(); ++i) {
        results[i] = UpdateFieldsForOneKey(keys[i], field_maps[i]);
    }

    return results;
}

std::vector<ErrorCode> MetaLocalBackend::Upsert(const KeyTypeVec &keys, const FieldMapVec &field_maps) noexcept {
    std::vector<ErrorCode> results(keys.size(), EC_OK);

    for (size_t i = 0; i < keys.size(); ++i) {
        results[i] = UpsertForOneKey(keys[i], field_maps[i]);
    }

    return results;
}

std::vector<ErrorCode> MetaLocalBackend::IncrFields(const KeyTypeVec &keys,
                                                    const std::map<std::string, int64_t> &field_amounts) noexcept {
    std::vector<ErrorCode> results(keys.size(), EC_OK);

    for (size_t i = 0; i < keys.size(); ++i) {
        results[i] = IncrFieldsForOneKey(keys[i], field_amounts);
    }

    return results;
}

std::vector<ErrorCode> MetaLocalBackend::Delete(const KeyTypeVec &keys) noexcept {
    std::vector<ErrorCode> results(keys.size(), EC_OK);

    for (size_t i = 0; i < keys.size(); ++i) {
        results[i] = DeleteForOneKey(keys[i]);
    }

    return results;
}

std::vector<ErrorCode> MetaLocalBackend::Get(const KeyTypeVec &keys,
                                             const std::vector<std::string> &field_names,
                                             FieldMapVec &out_field_maps) noexcept {
    std::vector<ErrorCode> results(keys.size(), EC_OK);
    out_field_maps.resize(keys.size());

    for (size_t i = 0; i < keys.size(); ++i) {
        results[i] = GetForOneKey(keys[i], field_names, out_field_maps[i]);
    }

    return results;
}

std::vector<ErrorCode> MetaLocalBackend::GetAllFields(const KeyTypeVec &keys, FieldMapVec &out_field_maps) noexcept {
    std::vector<ErrorCode> results(keys.size(), EC_OK);
    out_field_maps.resize(keys.size());

    for (size_t i = 0; i < keys.size(); ++i) {
        results[i] = GetAllFieldsForOneKey(keys[i], out_field_maps[i]);
    }

    return results;
}

std::vector<ErrorCode> MetaLocalBackend::Exists(const KeyTypeVec &keys, std::vector<bool> &out_is_exist_vec) noexcept {
    std::vector<ErrorCode> results(keys.size(), EC_OK);
    out_is_exist_vec.resize(keys.size());

    for (size_t i = 0; i < keys.size(); ++i) {
        bool is_exist;
        results[i] = ExistsForOneKey(keys[i], is_exist);
        out_is_exist_vec[i] = is_exist;
    }

    return results;
}

ErrorCode MetaLocalBackend::ListKeys(const std::string &cursor,
                                     const int64_t limit,
                                     std::string &out_next_cursor,
                                     std::vector<KeyType> &out_keys) noexcept {
    return key_tracker_->Scan(cursor, limit, out_next_cursor, out_keys);
}

ErrorCode MetaLocalBackend::RandomSample(const int64_t count, std::vector<KeyType> &out_keys) noexcept {
    key_tracker_->RandomSample(count, out_keys);
    return EC_OK;
}

std::vector<ErrorCode> MetaLocalBackend::PutIfAbsent(const KeyTypeVec &keys, const FieldMapVec &field_maps) noexcept {
    std::vector<ErrorCode> results(keys.size(), EC_OK);
    for (size_t i = 0; i < keys.size(); ++i) {
        results[i] = PutIfAbsentForOneKey(keys[i], field_maps[i]);
    }
    return results;
}

// ==================== Conditional write operations ====================

std::vector<ErrorCode> MetaLocalBackend::Put(const KeyTypeVec &keys,
                                             const FieldMapVec &field_maps,
                                             const std::vector<ErrorCode> &previous_error_codes) noexcept {
    std::vector<ErrorCode> results(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        if (previous_error_codes[i] != EC_OK) {
            results[i] = previous_error_codes[i];
            continue;
        }
        results[i] = PutForOneKey(keys[i], field_maps[i]);
    }
    return results;
}

std::vector<ErrorCode> MetaLocalBackend::UpdateFields(const KeyTypeVec &keys,
                                                      const FieldMapVec &field_maps,
                                                      const std::vector<ErrorCode> &previous_error_codes) noexcept {
    std::vector<ErrorCode> results(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        if (previous_error_codes[i] != EC_OK) {
            results[i] = previous_error_codes[i];
            continue;
        }
        results[i] = UpdateFieldsForOneKey(keys[i], field_maps[i]);
    }
    return results;
}

std::vector<ErrorCode> MetaLocalBackend::Upsert(const KeyTypeVec &keys,
                                                const FieldMapVec &field_maps,
                                                const std::vector<ErrorCode> &previous_error_codes) noexcept {
    std::vector<ErrorCode> results(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        if (previous_error_codes[i] != EC_OK) {
            results[i] = previous_error_codes[i];
            continue;
        }
        results[i] = UpsertForOneKey(keys[i], field_maps[i]);
    }
    return results;
}

std::vector<ErrorCode> MetaLocalBackend::IncrFields(const KeyTypeVec &keys,
                                                    const std::map<std::string, int64_t> &field_amounts,
                                                    const std::vector<ErrorCode> &previous_error_codes) noexcept {
    std::vector<ErrorCode> results(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        if (previous_error_codes[i] != EC_OK) {
            results[i] = previous_error_codes[i];
            continue;
        }
        results[i] = IncrFieldsForOneKey(keys[i], field_amounts);
    }
    return results;
}

std::vector<ErrorCode> MetaLocalBackend::Delete(const KeyTypeVec &keys,
                                                const std::vector<ErrorCode> &previous_error_codes) noexcept {
    std::vector<ErrorCode> results(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        if (previous_error_codes[i] != EC_OK) {
            results[i] = previous_error_codes[i];
            continue;
        }
        results[i] = DeleteForOneKey(keys[i]);
    }
    return results;
}

std::vector<ErrorCode> MetaLocalBackend::PutIfAbsent(const KeyTypeVec &keys,
                                                     const FieldMapVec &field_maps,
                                                     const std::vector<ErrorCode> &previous_error_codes) noexcept {
    std::vector<ErrorCode> results(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        if (previous_error_codes[i] != EC_OK) {
            results[i] = previous_error_codes[i];
            continue;
        }
        results[i] = PutIfAbsentForOneKey(keys[i], field_maps[i]);
    }
    return results;
}

ErrorCode MetaLocalBackend::PutMetaData(const FieldMap &field_maps) noexcept { return EC_OK; }

ErrorCode MetaLocalBackend::GetMetaData(FieldMap &field_maps) noexcept { return EC_NOENT; }

// 私有方法实现
ErrorCode MetaLocalBackend::PutIfAbsentForOneKey(const KeyType &key, const FieldMap &field_map) {
    std::string existing_value;
    ErrorCode exists_ec = cache_->Get(key, &existing_value);
    if (exists_ec == EC_OK) {
        return EC_OK;
    }
    return PutForOneKey(key, field_map);
}

ErrorCode MetaLocalBackend::PutForOneKey(const KeyType &key, const FieldMap &field_map) {
    // 序列化FieldMap
    std::string serialized = FieldMapSerializer::Serialize(field_map);

    // 存储到cache
    ErrorCode ret = cache_->Put(key, serialized);
    if (ret == EC_OK) {
        // 更新key tracker
        key_tracker_->AddKey(key);
    }

    return ret;
}

ErrorCode MetaLocalBackend::UpdateFieldsForOneKey(const KeyType &key, const FieldMap &field_map) {
    // 先获取现有的field map
    std::string serialized_value;
    ErrorCode ret = cache_->Get(key, &serialized_value);
    if (ret != EC_OK) {
        // 如果不存在，则不进行更新
        return EC_NOENT;
    }

    // 反序列化现有值
    FieldMap existing_map = FieldMapSerializer::Deserialize(serialized_value);

    // 更新字段
    for (const auto &pair : field_map) {
        existing_map[pair.first] = pair.second;
    }

    // 重新序列化并存储
    std::string new_serialized = FieldMapSerializer::Serialize(existing_map);
    ret = cache_->Put(key, new_serialized);
    if (ret == EC_OK) {
        key_tracker_->AccessKey(key);
    }

    return ret;
}

ErrorCode MetaLocalBackend::UpsertForOneKey(const KeyType &key, const FieldMap &field_map) {
    // 先获取现有的field map
    std::string serialized_value;
    ErrorCode ret = cache_->Get(key, &serialized_value);
    if (ret != EC_OK) {
        // 如果不存在，则当作新键处理
        return PutForOneKey(key, field_map);
    }
    return UpdateFieldsForOneKey(key, field_map);
}

ErrorCode MetaLocalBackend::IncrFieldsForOneKey(const KeyType &key,
                                                const std::map<std::string, int64_t> &field_amounts) {
    // 先获取现有的field map
    std::string serialized_value;
    ErrorCode ret = cache_->Get(key, &serialized_value);
    if (ret != EC_OK) {
        return ret;
    }

    // 反序列化现有值
    FieldMap existing_map = FieldMapSerializer::Deserialize(serialized_value);

    // 增加指定字段的值
    for (const auto &[field_name, amount] : field_amounts) {
        const auto field_iter = existing_map.find(field_name);
        if (field_iter == existing_map.end()) {
            KVCM_LOG_ERROR("incr fields fail, cannot find field[%s] for key[%ld]", field_name.c_str(), key);
            return EC_BADARGS;
        }
        const auto &old_field_value = field_iter->second;
        int64_t old_field_value_num = 0;
        if (!StringUtil::StrToInt64(old_field_value.c_str(), old_field_value_num)) {
            KVCM_LOG_ERROR("incr fields fail, cannot convert field[%s] value[%s] to int64_t for key[%ld]",
                           field_name.c_str(),
                           old_field_value.c_str(),
                           key);
            return EC_BADARGS;
        }
        existing_map[field_name] = std::to_string(old_field_value_num + amount);
    }

    // 重新序列化并存储
    std::string new_serialized = FieldMapSerializer::Serialize(existing_map);
    ret = cache_->Put(key, new_serialized);
    if (ret == EC_OK) {
        key_tracker_->AccessKey(key);
    }

    return ret;
}

ErrorCode MetaLocalBackend::DeleteForOneKey(const KeyType &key) {
    // 从cache中删除
    cache_->Delete(key);

    // 从key tracker中删除
    bool is_delete = key_tracker_->RemoveKey(key);

    return is_delete ? EC_OK : EC_NOENT;
}

ErrorCode MetaLocalBackend::GetForOneKey(const KeyType &key,
                                         const std::vector<std::string> &field_names,
                                         FieldMap &out_field_map) {
    std::string serialized_value;
    ErrorCode ret = cache_->Get(key, &serialized_value);
    if (ret != EC_OK) {
        return ret;
    }

    // 反序列化整个field map
    FieldMap full_map = FieldMapSerializer::Deserialize(serialized_value);
    for (const auto &field_name : field_names) {
        auto it = full_map.find(field_name);
        if (it != full_map.end()) {
            out_field_map[field_name] = it->second;
        } else {
            out_field_map[field_name] = "";
        }
    }

    // 更新key的访问时间
    key_tracker_->AccessKey(key);

    return EC_OK;
}

ErrorCode MetaLocalBackend::GetAllFieldsForOneKey(const KeyType &key, FieldMap &out_field_map) {
    std::string serialized_value;
    ErrorCode ret = cache_->Get(key, &serialized_value);
    if (ret != EC_OK) {
        return ret;
    }

    // 反序列化整个field map
    out_field_map = FieldMapSerializer::Deserialize(serialized_value);

    // 更新key的访问时间
    key_tracker_->AccessKey(key);

    return EC_OK;
}

ErrorCode MetaLocalBackend::ExistsForOneKey(const KeyType &key, bool &out_is_exist) {
    std::string serialized_value;
    ErrorCode ret = cache_->Get(key, &serialized_value);

    out_is_exist = (ret == EC_OK);

    if (out_is_exist) {
        // 更新key的访问时间
        key_tracker_->AccessKey(key);
    }

    return EC_OK;
}

} // namespace kv_cache_manager
