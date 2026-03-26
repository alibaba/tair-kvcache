#include "kv_cache_manager/meta/meta_local_backend.h"

#include <random>
#include <utility>

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/standard_uri.h"
#include "kv_cache_manager/config/meta_storage_backend_config.h"

namespace kv_cache_manager {

std::string MetaLocalBackend::GetStorageType() noexcept { return "local"; }

ErrorCode MetaLocalBackend::Init(const std::string &instance_id,
                                 const std::shared_ptr<MetaStorageBackendConfig> &config) noexcept {
    if (instance_id.empty()) {
        KVCM_LOG_ERROR("fail to init meta local backend, invalid empty instance id");
        return EC_BADARGS;
    }
    if (!config) {
        KVCM_LOG_ERROR("fail to init meta local backend, invalid nullptr config");
        return EC_BADARGS;
    }

    // Parse capacity, num_shard_bits, and sample_times from storage_uri.
    // Fall back to defaults if the URI is empty, invalid, or missing parameters.
    size_t capacity = META_LOCAL_BACKEND_DEFAULT_CAPACITY;
    int32_t num_shard_bits = META_LOCAL_BACKEND_DEFAULT_NUM_SHARD_BITS;
    sample_times_ = META_LOCAL_BACKEND_DEFAULT_SAMPLE_TIMES;

    const std::string &storage_uri = config->GetStorageUri();
    if (!storage_uri.empty()) {
        StandardUri uri = StandardUri::FromUri(storage_uri);
        if (uri.Valid()) {
            // capacity mb
            uri.GetParamAs("capacity", capacity);
            uri.GetParamAs("num_shard_bits", num_shard_bits);
            uri.GetParamAs("sample_times", sample_times_);
        } else {
            KVCM_LOG_ERROR("invalid storage uri[%s]", storage_uri.c_str());
            return EC_BADARGS;
        }
    }
    if (capacity <= 0 || num_shard_bits < 0 || sample_times_ <= 0) {
        KVCM_LOG_ERROR(
            "invalid local backend parameters, capacity[%lu] num_shard_bits[%d] sample_times[%ld], storage uri[%s]",
            capacity,
            num_shard_bits,
            sample_times_,
            storage_uri.c_str());
        return EC_BADARGS;
    }

    shard_mask_ = (1 << num_shard_bits) - 1;
    cache_ = NewLRUCache(capacity * 1024 * 1024ULL, num_shard_bits);
    if (!cache_) {
        KVCM_LOG_ERROR("fail to create LRUCache");
        return EC_ERROR;
    }
    cache_item_helper_ = std::make_shared<Cache::CacheItemHelper>();
    cache_item_helper_->del_cb = MetaMemCacheItem::Deleter;

    KVCM_LOG_INFO("local backend init ok, instance[%s] capacity[%lu] num_shard_bits[%d] sample_times[%ld]",
                  instance_id.c_str(),
                  capacity,
                  num_shard_bits,
                  sample_times_);
    return EC_OK;
}

ErrorCode MetaLocalBackend::Open() noexcept {
    if (!cache_) {
        KVCM_LOG_ERROR("Cache is not initialized");
        return EC_ERROR;
    }
    return EC_OK;
}

ErrorCode MetaLocalBackend::Close() noexcept {
    cache_.reset();
    cache_item_helper_.reset();
    return EC_OK;
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

ErrorCode MetaLocalBackend::CreateAndInsert(const std::string &key_str, const FieldMap &fields) {
    MetaMemCacheItem *item = MetaMemCacheItem::Create(fields);
    size_t charge = item->Size();
    ErrorCode ret = cache_->Insert(key_str, item, cache_item_helper_.get(), charge);
    if (ret != EC_OK) {
        MetaMemCacheItem::Deleter(item, nullptr);
    }
    return ret;
}

ErrorCode MetaLocalBackend::CreateAndInsert(const std::string &key_str, FieldMap &&fields) {
    MetaMemCacheItem *item = MetaMemCacheItem::Create(std::move(fields));
    size_t charge = item->Size();
    ErrorCode ret = cache_->Insert(key_str, item, cache_item_helper_.get(), charge);
    if (ret != EC_OK) {
        MetaMemCacheItem::Deleter(item, nullptr);
    }
    return ret;
}

ErrorCode MetaLocalBackend::CreateAndInsertIfAbsent(const std::string &key_str, const FieldMap &fields) {
    MetaMemCacheItem *item = MetaMemCacheItem::Create(fields);
    size_t charge = item->Size();
    ErrorCode ret = cache_->InsertIfAbsent(key_str, item, cache_item_helper_.get(), charge);
    if (ret != EC_OK && ret != EC_EXIST) {
        MetaMemCacheItem::Deleter(item, nullptr);
    }
    return ret;
}

bool MetaLocalBackend::LookupFields(const std::string &key_str, FieldMap &out_fields) {
    Cache::Handle *handle = cache_->Lookup(key_str);
    if (!handle) {
        return false;
    }
    auto *existing = static_cast<MetaMemCacheItem *>(cache_->Value(handle));
    out_fields = existing->GetFields();
    cache_->Release(handle);
    return true;
}

ErrorCode MetaLocalBackend::UpdateFieldsInPlace(const std::string &key_str, const FieldMap &updates) {
    Cache::Handle *handle = cache_->Lookup(key_str);
    if (!handle) {
        return EC_NOENT;
    }
    // Safe to modify fields_ in place: the handle holds a reference so the
    // item cannot be evicted or freed by another thread.  The FieldMap itself
    // is not involved in any LRU list or hash table operations.
    auto *existing = static_cast<MetaMemCacheItem *>(cache_->Value(handle));
    for (const auto &[key, value] : updates) {
        existing->GetMutableFields()[key] = value;
    }
    cache_->Release(handle);
    return EC_OK;
}

// ---------------------------------------------------------------------------
// Per-key write helpers
// ---------------------------------------------------------------------------

ErrorCode MetaLocalBackend::UpsertForOneKey(KeyType key, const FieldMap &field_map) {
    std::string key_str = std::to_string(key);
    if (UpdateFieldsInPlace(key_str, field_map) == EC_OK) {
        return EC_OK;
    }
    FieldMap fields = field_map;
    return CreateAndInsert(key_str, std::move(fields));
}

ErrorCode MetaLocalBackend::DeleteForOneKey(KeyType key) {
    std::string key_str = std::to_string(key);
    Cache::Handle *handle = cache_->Lookup(key_str);
    if (!handle) {
        return EC_NOENT;
    }
    cache_->Release(handle);
    cache_->Erase(key_str);
    return EC_OK;
}

// ---------------------------------------------------------------------------
// Write operations
// ---------------------------------------------------------------------------

std::vector<ErrorCode> MetaLocalBackend::Put(const KeyTypeVec &keys, const FieldMapVec &field_maps) noexcept {
    std::vector<ErrorCode> results(keys.size(), EC_OK);
    for (size_t i = 0; i < keys.size(); ++i) {
        results[i] = CreateAndInsert(std::to_string(keys[i]), field_maps[i]);
    }
    return results;
}

std::vector<ErrorCode> MetaLocalBackend::Put(const KeyTypeVec &keys,
                                             const FieldMapVec &field_maps,
                                             const std::vector<ErrorCode> &previous_error_codes) noexcept {
    std::vector<ErrorCode> results(keys.size(), EC_OK);
    for (size_t i = 0; i < keys.size(); ++i) {
        results[i] = (previous_error_codes[i] == EC_OK) ? CreateAndInsert(std::to_string(keys[i]), field_maps[i])
                                                        : previous_error_codes[i];
    }
    return results;
}

std::vector<ErrorCode> MetaLocalBackend::UpdateFields(const KeyTypeVec &keys, const FieldMapVec &field_maps) noexcept {
    std::vector<ErrorCode> results(keys.size(), EC_OK);
    for (size_t i = 0; i < keys.size(); ++i) {
        results[i] = UpdateFieldsInPlace(std::to_string(keys[i]), field_maps[i]);
    }
    return results;
}

std::vector<ErrorCode> MetaLocalBackend::UpdateFields(const KeyTypeVec &keys,
                                                      const FieldMapVec &field_maps,
                                                      const std::vector<ErrorCode> &previous_error_codes) noexcept {
    std::vector<ErrorCode> results(keys.size(), EC_OK);
    for (size_t i = 0; i < keys.size(); ++i) {
        results[i] = (previous_error_codes[i] == EC_OK) ? UpdateFieldsInPlace(std::to_string(keys[i]), field_maps[i])
                                                        : previous_error_codes[i];
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

std::vector<ErrorCode> MetaLocalBackend::Upsert(const KeyTypeVec &keys,
                                                const FieldMapVec &field_maps,
                                                const std::vector<ErrorCode> &previous_error_codes) noexcept {
    std::vector<ErrorCode> results(keys.size(), EC_OK);
    for (size_t i = 0; i < keys.size(); ++i) {
        results[i] =
            (previous_error_codes[i] == EC_OK) ? UpsertForOneKey(keys[i], field_maps[i]) : previous_error_codes[i];
    }
    return results;
}

std::vector<ErrorCode> MetaLocalBackend::IncrFields(const KeyTypeVec &keys,
                                                    const std::map<std::string, int64_t> &field_amounts) noexcept {
    return std::vector<ErrorCode>(keys.size(), EC_UNIMPLEMENTED);
}

std::vector<ErrorCode> MetaLocalBackend::IncrFields(const KeyTypeVec &keys,
                                                    const std::map<std::string, int64_t> &field_amounts,
                                                    const std::vector<ErrorCode> &previous_error_codes) noexcept {
    return std::vector<ErrorCode>(keys.size(), EC_UNIMPLEMENTED);
}

std::vector<ErrorCode> MetaLocalBackend::Delete(const KeyTypeVec &keys) noexcept {
    std::vector<ErrorCode> results(keys.size(), EC_OK);
    for (size_t i = 0; i < keys.size(); ++i) {
        results[i] = DeleteForOneKey(keys[i]);
    }
    return results;
}

std::vector<ErrorCode> MetaLocalBackend::Delete(const KeyTypeVec &keys,
                                                const std::vector<ErrorCode> &previous_error_codes) noexcept {
    std::vector<ErrorCode> results(keys.size(), EC_OK);
    for (size_t i = 0; i < keys.size(); ++i) {
        results[i] = (previous_error_codes[i] == EC_OK) ? DeleteForOneKey(keys[i]) : previous_error_codes[i];
    }
    return results;
}

// ---------------------------------------------------------------------------
// Read operations
// ---------------------------------------------------------------------------

std::vector<ErrorCode> MetaLocalBackend::Get(const KeyTypeVec &keys,
                                             const std::vector<std::string> &field_names,
                                             FieldMapVec &out_field_maps) noexcept {
    std::vector<ErrorCode> results(keys.size(), EC_OK);
    out_field_maps.resize(keys.size());

    for (size_t i = 0; i < keys.size(); ++i) {
        std::string key_str = std::to_string(keys[i]);

        Cache::Handle *handle = cache_->Lookup(key_str);
        if (!handle) {
            results[i] = EC_NOENT;
            continue;
        }

        auto *item = static_cast<MetaMemCacheItem *>(cache_->Value(handle));
        const auto &fields = item->GetFields();
        FieldMap &out_field_map = out_field_maps[i];
        for (const auto &field_name : field_names) {
            auto it = fields.find(field_name);
            if (it != fields.end()) {
                out_field_map[field_name] = it->second;
            }
        }
        cache_->Release(handle);
        results[i] = EC_OK;
    }

    return results;
}

std::vector<ErrorCode> MetaLocalBackend::GetAllFields(const KeyTypeVec &keys, FieldMapVec &out_field_maps) noexcept {
    std::vector<ErrorCode> results(keys.size(), EC_OK);
    out_field_maps.resize(keys.size());

    for (size_t i = 0; i < keys.size(); ++i) {
        std::string key_str = std::to_string(keys[i]);

        Cache::Handle *handle = cache_->Lookup(key_str);
        if (!handle) {
            results[i] = EC_NOENT;
            continue;
        }

        auto *item = static_cast<MetaMemCacheItem *>(cache_->Value(handle));
        out_field_maps[i] = item->GetFields();
        cache_->Release(handle);
        results[i] = EC_OK;
    }

    return results;
}

std::vector<ErrorCode> MetaLocalBackend::Exists(const KeyTypeVec &keys, std::vector<bool> &out_is_exist_vec) noexcept {
    std::vector<ErrorCode> results(keys.size(), EC_OK);
    out_is_exist_vec.resize(keys.size());

    for (size_t i = 0; i < keys.size(); ++i) {
        std::string key_str = std::to_string(keys[i]);

        Cache::Handle *handle = cache_->Lookup(key_str);
        out_is_exist_vec[i] = (handle != nullptr);

        if (handle) {
            cache_->Release(handle);
        }

        results[i] = EC_OK;
    }

    return results;
}

ErrorCode MetaLocalBackend::ListKeys(const std::string &cursor,
                                     const int64_t limit,
                                     std::string &out_next_cursor,
                                     std::vector<KeyType> &out_keys) noexcept {
    // Treat cursor as shard_id; SCAN_BASE_CURSOR ("0") means start from shard 0.
    int64_t start_shard = 0;
    if (!StringUtil::StrToInt64(cursor.c_str(), start_shard) || start_shard < 0 || start_shard > shard_mask_) {
        return EC_BADARGS;
    }

    uint32_t num_shards = shard_mask_ + 1;
    int64_t collected = 0;

    for (uint32_t i = static_cast<uint32_t>(start_shard); i < num_shards; ++i) {
        cache_->ApplyToSingleShard(i,
                                   [&](const std::string_view &key,
                                       Cache::ObjectPtr /*value*/,
                                       size_t /*charge*/,
                                       const Cache::CacheItemHelper * /*helper*/) {
                                       KeyType parsed_key = 0;
                                       if (StringUtil::StrToInt64(std::string(key).c_str(), parsed_key)) {
                                           out_keys.push_back(parsed_key);
                                           ++collected;
                                       }
                                   });

        // Check after finishing the entire shard — never truncate mid-shard.
        if (collected >= limit) {
            // Point the next cursor to the following shard for continuation.
            uint32_t next_shard = i + 1;
            out_next_cursor = (next_shard >= num_shards) ? SCAN_BASE_CURSOR : std::to_string(next_shard);
            return EC_OK;
        }
    }

    // All shards exhausted without reaching the limit.
    out_next_cursor = SCAN_BASE_CURSOR;
    return EC_OK;
}

ErrorCode MetaLocalBackend::RandomSample(const int64_t count, std::vector<KeyType> &out_keys) noexcept {
    if (!cache_ || count <= 0) {
        return EC_OK;
    }

    return GetOldestKeysFromRandomShard(static_cast<size_t>(count), out_keys);
}

ErrorCode MetaLocalBackend::SampleReclaimKeys(const int64_t count, std::vector<KeyType> &out_keys) noexcept {
    // Validate input parameters
    if (!cache_ || count <= 0) {
        return EC_OK;
    }

    int64_t remaining = count;
    int64_t batch_size = count / std::min(sample_times_, count);
    // Loop until we have collected enough keys or no more keys available
    while (remaining > 0) {
        // Calculate batch size for this iteration using member variable sample_times
        batch_size = std::min(batch_size, remaining);

        // Get oldest keys from a random shard
        int64_t last_size = out_keys.size();
        ErrorCode ec = GetOldestKeysFromRandomShard(static_cast<size_t>(batch_size), out_keys);
        if (ec != EC_OK) {
            return ec;
        }

        // Add collected keys to output
        int64_t current_sample_size = out_keys.size() - last_size;
        remaining -= current_sample_size;

        // If no keys were collected in this iteration, break to avoid infinite loop
        if (current_sample_size == 0) {
            break;
        }
    }

    return EC_OK;
}

ErrorCode MetaLocalBackend::PutMetaData(const FieldMap & /*field_maps*/) noexcept { return EC_OK; }

ErrorCode MetaLocalBackend::GetMetaData(FieldMap & /*field_maps*/) noexcept { return EC_NOENT; }

std::vector<ErrorCode> MetaLocalBackend::PutIfAbsent(const KeyTypeVec &keys,
                                                     const FieldMapVec &field_maps,
                                                     const std::vector<ErrorCode> &previous_error_codes) noexcept {
    std::vector<ErrorCode> results(keys.size(), EC_OK);
    for (size_t i = 0; i < keys.size(); ++i) {
        results[i] = (previous_error_codes[i] == EC_OK)
                         ? CreateAndInsertIfAbsent(std::to_string(keys[i]), field_maps[i])
                         : previous_error_codes[i];
    }
    return results;
}

ErrorCode MetaLocalBackend::GetOldestKeysFromRandomShard(size_t count, std::vector<KeyType> &out_keys) {
    if (!cache_ || count == 0) {
        return EC_BADARGS;
    }

    static thread_local std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<uint32_t> dist(0, shard_mask_);
    uint32_t shard_id = dist(rng);

    std::vector<std::string> string_keys;
    string_keys.reserve(count);
    uint32_t shard_collect_count = 0;
    size_t key_collect_count = 0;
    while (key_collect_count < count && shard_collect_count <= shard_mask_) {
        string_keys.clear();
        cache_->GetOldestKeysInShard(shard_id, count - key_collect_count, string_keys);
        KeyType parsed_key = 0;
        for (const auto &key_str : string_keys) {
            if (StringUtil::StrToInt64(key_str.c_str(), parsed_key)) {
                out_keys.push_back(parsed_key);
            }
        }
        key_collect_count += string_keys.size();
        ++shard_collect_count;
        shard_id = (shard_id + 1) & shard_mask_;
    }

    return EC_OK;
}

} // namespace kv_cache_manager
