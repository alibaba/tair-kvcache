#include "kv_cache_manager/meta/meta_local_backend.h"

#include <algorithm>
#include <climits>
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
            "invalid local backend parameters, capacity[%lu] num_shard_bits[%d] sample_times[%zu], storage uri[%s]",
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

    // Initialize per-shard oldest access time tracking.
    size_t num_shards = shard_mask_ + 1;
    shard_oldest_access_time_ = std::make_unique<std::atomic<int64_t>[]>(num_shards);
    for (size_t i = 0; i < num_shards; ++i) {
        shard_oldest_access_time_[i].store(INT64_MAX, std::memory_order_relaxed);
    }
    // Register tail-change callback: whenever a shard's LRU tail changes,
    // read the tail item's last_access_time and store it in the atomic array.
    cache_->SetTailChangeCallback([this](uint32_t shard_id, Cache::ObjectPtr tail_value) {
        if (tail_value == nullptr) {
            shard_oldest_access_time_[shard_id].store(INT64_MAX, std::memory_order_relaxed);
        } else {
            auto *item = static_cast<MetaMemCacheItem *>(tail_value);
            int64_t access_time = item->GetLastAccessTime();
            shard_oldest_access_time_[shard_id].store(access_time, std::memory_order_relaxed);
        }
    });

    KVCM_LOG_INFO("local backend init ok, instance[%s] capacity[%lu] num_shard_bits[%d] sample_times[%zu]",
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
    if (cache_) {
        cache_->SetTailChangeCallback(nullptr);
    }
    cache_.reset();
    cache_item_helper_.reset();
    return EC_OK;
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

ErrorCode MetaLocalBackend::CreateAndInsert(const std::string &key_str, const FieldMap &fields) {
    MetaMemCacheItem *item = MetaMemCacheItem::Create(fields);
    item->TouchAccessTime();
    size_t charge = item->Size();
    ErrorCode ret = cache_->Insert(key_str, item, cache_item_helper_.get(), charge);
    if (ret != EC_OK) {
        MetaMemCacheItem::Deleter(item, nullptr);
    }
    return ret;
}

ErrorCode MetaLocalBackend::CreateAndInsert(const std::string &key_str, FieldMap &&fields) {
    MetaMemCacheItem *item = MetaMemCacheItem::Create(std::move(fields));
    item->TouchAccessTime();
    size_t charge = item->Size();
    ErrorCode ret = cache_->Insert(key_str, item, cache_item_helper_.get(), charge);
    if (ret != EC_OK) {
        MetaMemCacheItem::Deleter(item, nullptr);
    }
    return ret;
}

ErrorCode MetaLocalBackend::CreateAndInsertIfAbsent(const std::string &key_str, const FieldMap &fields) {
    MetaMemCacheItem *item = MetaMemCacheItem::Create(fields);
    item->TouchAccessTime();
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
    existing->TouchAccessTime();
    {
        std::shared_lock lock(existing->GetMutex());
        out_fields = existing->GetFields();
    }
    cache_->Release(handle);
    return true;
}

ErrorCode MetaLocalBackend::UpdateFieldsInPlace(const std::string &key_str, const FieldMap &updates) {
    Cache::Handle *handle = cache_->Lookup(key_str);
    if (!handle) {
        return EC_NOENT;
    }
    auto *existing = static_cast<MetaMemCacheItem *>(cache_->Value(handle));
    existing->TouchAccessTime();
    {
        std::unique_lock lock(existing->GetMutex());
        for (const auto &[key, value] : updates) {
            existing->GetMutableFields()[key] = value;
        }
    }
    cache_->Release(handle);
    return EC_OK;
}

// ---------------------------------------------------------------------------
// Per-key helpers
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

ErrorCode MetaLocalBackend::DeleteFieldsForOneKey(KeyType key, const std::vector<std::string> &field_names) {
    std::string key_str = std::to_string(key);
    Cache::Handle *handle = cache_->Lookup(key_str);
    if (!handle) {
        return EC_NOENT;
    }
    auto *item = static_cast<MetaMemCacheItem *>(cache_->Value(handle));
    item->TouchAccessTime();
    {
        std::unique_lock lock(item->GetMutex());
        for (const auto &field_name : field_names) {
            item->GetMutableFields().erase(field_name);
        }
    }
    cache_->Release(handle);
    return EC_OK;
}

ErrorCode
MetaLocalBackend::GetForOneKey(KeyType key, const std::vector<std::string> &field_names, FieldMap &out_field_map) {
    out_field_map.clear();

    std::string key_str = std::to_string(key);
    Cache::Handle *handle = cache_->Lookup(key_str);
    if (!handle) {
        return EC_NOENT;
    }

    auto *item = static_cast<MetaMemCacheItem *>(cache_->Value(handle));
    int64_t stored_time = item->GetLastAccessTime();
    item->TouchAccessTime();
    {
        std::shared_lock lock(item->GetMutex());
        const auto &fields = item->GetFields();
        for (const auto &field_name : field_names) {
            if (field_name == PROPERTY_LRU_TIME) {
                out_field_map[PROPERTY_LRU_TIME] = std::to_string(stored_time);
                continue;
            }
            auto it = fields.find(field_name);
            if (it != fields.end()) {
                out_field_map[field_name] = it->second;
            }
        }
    }
    cache_->Release(handle);
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

std::vector<ErrorCode> MetaLocalBackend::PutIfAbsent(const KeyTypeVec &keys, const FieldMapVec &field_maps) noexcept {
    std::vector<ErrorCode> results(keys.size(), EC_OK);
    for (size_t i = 0; i < keys.size(); ++i) {
        results[i] = CreateAndInsertIfAbsent(std::to_string(keys[i]), field_maps[i]);
    }
    return results;
}

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

std::vector<ErrorCode>
MetaLocalBackend::DeleteFields(const KeyTypeVec &keys,
                               const std::vector<std::vector<std::string>> &field_names_vec) noexcept {
    std::vector<ErrorCode> results(keys.size(), EC_OK);
    for (size_t i = 0; i < keys.size(); ++i) {
        results[i] = DeleteFieldsForOneKey(keys[i], field_names_vec[i]);
    }
    return results;
}

std::vector<ErrorCode> MetaLocalBackend::DeleteFields(const KeyTypeVec &keys,
                                                      const std::vector<std::vector<std::string>> &field_names_vec,
                                                      const std::vector<ErrorCode> &previous_error_codes) noexcept {
    std::vector<ErrorCode> results(keys.size(), EC_OK);
    for (size_t i = 0; i < keys.size(); ++i) {
        results[i] = (previous_error_codes[i] == EC_OK) ? DeleteFieldsForOneKey(keys[i], field_names_vec[i])
                                                        : previous_error_codes[i];
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
        results[i] = GetForOneKey(keys[i], field_names, out_field_maps[i]);
    }
    return results;
}

std::vector<ErrorCode> MetaLocalBackend::Get(const KeyTypeVec &keys,
                                             const std::vector<std::vector<std::string>> &field_names_vec,
                                             FieldMapVec &out_field_maps) noexcept {
    std::vector<ErrorCode> results(keys.size(), EC_OK);
    out_field_maps.resize(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        results[i] = GetForOneKey(keys[i], field_names_vec[i], out_field_maps[i]);
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
        int64_t stored_time = item->GetLastAccessTime();
        item->TouchAccessTime();
        {
            std::shared_lock lock(item->GetMutex());
            out_field_maps[i] = item->GetFields();
        }
        out_field_maps[i][PROPERTY_LRU_TIME] = std::to_string(stored_time);
        cache_->Release(handle);
        results[i] = EC_OK;
    }

    return results;
}

std::vector<ErrorCode> MetaLocalBackend::Exists(const KeyTypeVec &keys, std::vector<bool> &out_is_exist_vec) noexcept {
    std::vector<ErrorCode> results(keys.size(), EC_OK);
    out_is_exist_vec.resize(keys.size(), false);

    for (size_t i = 0; i < keys.size(); ++i) {
        std::string key_str = std::to_string(keys[i]);

        Cache::Handle *handle = cache_->Lookup(key_str);
        out_is_exist_vec[i] = (handle != nullptr);

        if (handle) {
            auto *item = static_cast<MetaMemCacheItem *>(cache_->Value(handle));
            item->TouchAccessTime();
            cache_->Release(handle);
        }

        results[i] = EC_OK;
    }

    return results;
}

std::vector<ErrorCode> MetaLocalBackend::ExistsFieldWithPrefix(const KeyTypeVec &keys,
                                                               const std::string &field_prefix,
                                                               std::vector<bool> &out_exists_vec) noexcept {
    std::vector<ErrorCode> results(keys.size(), EC_OK);
    out_exists_vec.resize(keys.size(), false);

    for (size_t i = 0; i < keys.size(); ++i) {
        std::string key_str = std::to_string(keys[i]);
        Cache::Handle *handle = cache_->Lookup(key_str);
        if (!handle) {
            results[i] = EC_NOENT;
            continue;
        }

        auto *item = static_cast<MetaMemCacheItem *>(cache_->Value(handle));
        item->TouchAccessTime();
        {
            std::shared_lock lock(item->GetMutex());
            const auto &fields = item->GetFields();
            for (auto it = fields.lower_bound(field_prefix); it != fields.end(); ++it) {
                if (it->first.size() < field_prefix.size() ||
                    it->first.compare(0, field_prefix.size(), field_prefix) != 0) {
                    break;
                }
                // Skip tombstones (empty value) — they are not valid locations.
                if (!it->second.empty()) {
                    out_exists_vec[i] = true;
                    break;
                }
            }
        }
        cache_->Release(handle);
    }

    return results;
}

std::vector<ErrorCode>
MetaLocalBackend::GetFieldNamesWithPrefix(const KeyTypeVec &keys,
                                          const std::string &field_prefix,
                                          std::vector<std::vector<std::string>> &out_field_names_vec) noexcept {
    std::vector<ErrorCode> results(keys.size(), EC_OK);
    out_field_names_vec.resize(keys.size());

    for (size_t i = 0; i < keys.size(); ++i) {
        std::string key_str = std::to_string(keys[i]);
        Cache::Handle *handle = cache_->Lookup(key_str);
        if (!handle) {
            results[i] = EC_NOENT;
            continue;
        }

        auto *item = static_cast<MetaMemCacheItem *>(cache_->Value(handle));
        item->TouchAccessTime();
        {
            std::shared_lock lock(item->GetMutex());
            const auto &fields = item->GetFields();
            for (auto it = fields.lower_bound(field_prefix); it != fields.end(); ++it) {
                if (it->first.size() < field_prefix.size() ||
                    it->first.compare(0, field_prefix.size(), field_prefix) != 0) {
                    break;
                }
                // Skip tombstones (empty value) — they are not valid locations.
                if (!it->second.empty()) {
                    out_field_names_vec[i].emplace_back(it->first);
                }
            }
        }
        cache_->Release(handle);
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
    if (!cache_) {
        KVCM_LOG_ERROR("local backend not inited");
        return EC_ERROR;
    }
    if (count <= 0) {
        return EC_OK;
    }

    static thread_local std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<uint32_t> dist(0, shard_mask_);
    uint32_t shard_id = dist(rng);
    uint32_t shard_collect_count = 0;
    size_t key_collect_count = 0;
    while (key_collect_count < count && shard_collect_count <= shard_mask_) {
        key_collect_count += CollectOldestKeysFromShard(shard_id, count - key_collect_count, out_keys);
        ++shard_collect_count;
        shard_id = (shard_id + 1) & shard_mask_;
    }

    return EC_OK;
}

ErrorCode MetaLocalBackend::SampleReclaimKeys(const int64_t count, std::vector<KeyType> &out_keys) noexcept {
    out_keys.clear();
    if (!cache_) {
        KVCM_LOG_ERROR("local backend not inited");
        return EC_ERROR;
    }
    if (count <= 0) {
        return EC_OK;
    }

    size_t num_shards = shard_mask_ + 1;
    size_t num_rounds = std::min(sample_times_, num_shards);
    num_rounds = std::min(num_rounds, static_cast<size_t>(count));
    int64_t per_round_count = (count + num_rounds - 1) / num_rounds;
    std::vector<std::pair<int64_t, uint32_t>> shard_times;
    shard_times.reserve(num_shards);
    for (uint32_t s = 0; s < num_shards; ++s) {
        int64_t access_time = shard_oldest_access_time_[s].load(std::memory_order_relaxed);
        if (access_time < INT64_MAX) {
            shard_times.emplace_back(access_time, s);
        }
    }
    if (shard_times.empty()) {
        return EC_OK;
    }

    size_t select_count = std::min(num_rounds, shard_times.size());
    std::partial_sort(shard_times.begin(), shard_times.begin() + select_count, shard_times.end());
    int64_t remaining = count;
    for (size_t i = 0; i < select_count && remaining > 0; ++i) {
        size_t batch = static_cast<size_t>(std::min(per_round_count, remaining));
        size_t collected = CollectOldestKeysFromShard(shard_times[i].second, batch, out_keys);
        remaining -= static_cast<int64_t>(collected);
    }
    return EC_OK;
}

// return OK to avoid error in MetaIndexer::PersistMetaData()
ErrorCode MetaLocalBackend::PutMetaData(const FieldMap & /*field_maps*/) noexcept { return EC_OK; }

ErrorCode MetaLocalBackend::GetMetaData(FieldMap & /*field_maps*/) noexcept { return EC_NOENT; }

size_t MetaLocalBackend::GetMemUsage() const noexcept { return cache_->GetUsage(); }

size_t MetaLocalBackend::CollectOldestKeysFromShard(uint32_t shard_id, size_t count, std::vector<KeyType> &out_keys) {
    std::vector<std::string> string_keys;
    string_keys.reserve(count);
    cache_->GetOldestKeysInShard(shard_id, count, string_keys);
    KeyType parsed_key = 0;
    for (const auto &key_str : string_keys) {
        if (StringUtil::StrToInt64(key_str.c_str(), parsed_key)) {
            out_keys.push_back(parsed_key);
        }
    }
    return string_keys.size();
}

} // namespace kv_cache_manager
