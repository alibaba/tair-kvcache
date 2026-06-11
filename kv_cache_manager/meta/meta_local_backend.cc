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
    cache_ = NewLRUCache(capacity * 1024 * 1024ULL,
                         num_shard_bits,
                         /*strict_capacity_limit=*/true,
                         /*no_evict_on_insert=*/true);
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
    KVCM_LOG_INFO("local backend open ok");
    return EC_OK;
}

ErrorCode MetaLocalBackend::Close() noexcept {
    if (cache_) {
        cache_->SetTailChangeCallback(nullptr);
    }
    cache_.reset();
    cache_item_helper_.reset();
    KVCM_LOG_INFO("local backend close ok");
    return EC_OK;
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

ErrorCode MetaLocalBackend::CreateAndInsert(std::string_view key_sv,
                                            const CacheLocationMap &locations,
                                            const PropertyMap &properties) {
    MetaMemCacheItem *item = MetaMemCacheItem::Create(locations, properties);
    item->TouchAccessTime();
    size_t charge = item->Size();
    // We must pass &handle (not nullptr) so that when strict_capacity_limit
    // is enabled and capacity is exceeded, Insert returns EC_NOSPC instead
    // of silently discarding the entry with EC_OK.
    Cache::Handle *handle = nullptr;
    ErrorCode ret = cache_->Insert(key_sv, item, cache_item_helper_.get(), charge, &handle);
    if (ret != EC_OK) {
        MetaMemCacheItem::Deleter(item, nullptr);
    } else if (handle) {
        cache_->Release(handle);
    }
    return ret;
}

ErrorCode MetaLocalBackend::CreateAndInsertIfAbsent(std::string_view key_sv,
                                                    const CacheLocationMap &locations,
                                                    const PropertyMap &properties) {
    MetaMemCacheItem *item = MetaMemCacheItem::Create(locations, properties);
    item->TouchAccessTime();
    size_t charge = item->Size();
    Cache::Handle *handle = nullptr;
    ErrorCode ret = cache_->InsertIfAbsent(key_sv, item, cache_item_helper_.get(), charge, &handle);
    if (ret != EC_OK && ret != EC_EXIST) {
        MetaMemCacheItem::Deleter(item, nullptr);
    } else if (handle) {
        cache_->Release(handle);
    }
    return ret;
}

ErrorCode MetaLocalBackend::UpdateInPlace(std::string_view key_sv,
                                          const CacheLocationMap &locations,
                                          const PropertyMap &properties) {
    Cache::Handle *handle = cache_->Lookup(key_sv);
    if (!handle) {
        return EC_NOENT;
    }
    auto *existing = static_cast<MetaMemCacheItem *>(cache_->Value(handle));
    existing->TouchAccessTime();
    ssize_t charge_delta = 0;
    {
        std::unique_lock lock(existing->GetMutex());
        auto &existing_locations = existing->GetMutableLocations();
        for (const auto &[loc_id, loc_ptr] : locations) {
            auto it = existing_locations.find(loc_id);
            if (it != existing_locations.end()) {
                ssize_t old_usage = it->second ? static_cast<ssize_t>(it->second->EstimateMemUsage()) : 0;
                it->second = loc_ptr;
                ssize_t new_usage = loc_ptr ? static_cast<ssize_t>(loc_ptr->EstimateMemUsage()) : 0;
                charge_delta += new_usage - old_usage;
            } else {
                ssize_t new_usage = loc_ptr ? static_cast<ssize_t>(loc_ptr->EstimateMemUsage()) : 0;
                charge_delta += static_cast<ssize_t>(sizeof(void *) * 4 + loc_id.size()) + new_usage;
                existing_locations[loc_id] = loc_ptr;
            }
        }
        auto &existing_properties = existing->GetMutableProperties();
        for (const auto &[prop_name, prop_value] : properties) {
            auto it = existing_properties.find(prop_name);
            if (it != existing_properties.end()) {
                charge_delta += static_cast<ssize_t>(prop_value.size()) - static_cast<ssize_t>(it->second.size());
                it->second = prop_value;
            } else {
                charge_delta += static_cast<ssize_t>(sizeof(void *) * 4 + prop_name.size() + prop_value.size());
                existing_properties[prop_name] = prop_value;
            }
        }
    }
    if (charge_delta != 0) {
        cache_->AdjustCharge(handle, charge_delta);
    }
    cache_->Release(handle);
    return EC_OK;
}

// ---------------------------------------------------------------------------
// Per-key helpers
// ---------------------------------------------------------------------------

ErrorCode
MetaLocalBackend::UpsertForOneKey(KeyType key, const CacheLocationMap &locations, const PropertyMap &properties) {
    std::string_view key_sv = KeyToView(key);
    ErrorCode update_ec = UpdateInPlace(key_sv, locations, properties);
    if (update_ec != EC_OK && update_ec != EC_NOENT) {
        KVCM_LOG_ERROR("local backend fail to update key[%ld] in upsert, ec[%d]", key, update_ec);
        return update_ec;
    }
    if (update_ec == EC_OK) {
        return EC_OK;
    }
    ErrorCode insert_ec = CreateAndInsert(key_sv, locations, properties);
    if (insert_ec != EC_OK) {
        KVCM_LOG_ERROR("local backend fail to insert key[%ld] in upsert, ec[%d]", key, insert_ec);
        return insert_ec;
    }
    return EC_OK;
}

ErrorCode MetaLocalBackend::DeleteForOneKey(KeyType key) { return cache_->Erase(KeyToView(key)) ? EC_OK : EC_NOENT; }

ErrorCode MetaLocalBackend::DeleteLocationsForOneKey(KeyType key, const std::vector<LocationId> &location_ids) {
    std::string_view key_sv = KeyToView(key);
    Cache::Handle *handle = cache_->Lookup(key_sv);
    if (!handle) {
        return EC_NOENT;
    }
    auto *item = static_cast<MetaMemCacheItem *>(cache_->Value(handle));
    item->TouchAccessTime();
    ssize_t charge_delta = 0;
    {
        std::unique_lock lock(item->GetMutex());
        auto &locs = item->GetMutableLocations();
        for (const auto &loc_id : location_ids) {
            auto it = locs.find(loc_id);
            if (it != locs.end()) {
                ssize_t loc_usage = it->second ? static_cast<ssize_t>(it->second->EstimateMemUsage()) : 0;
                charge_delta -= static_cast<ssize_t>(sizeof(void *) * 4 + it->first.size()) + loc_usage;
                locs.erase(it);
            }
        }
    }
    if (charge_delta != 0) {
        cache_->AdjustCharge(handle, charge_delta);
    }
    cache_->Release(handle);
    return EC_OK;
}

// ---------------------------------------------------------------------------
// Write operations (unconditional)
// ---------------------------------------------------------------------------

std::vector<ErrorCode> MetaLocalBackend::Put(RequestContext * /*request_context*/,
                                             const KeyTypeVec &keys,
                                             const CacheLocationMapVector &locations,
                                             const PropertyMapVector &properties) noexcept {
    std::vector<ErrorCode> results(keys.size(), EC_OK);
    for (size_t i = 0; i < keys.size(); ++i) {
        results[i] = CreateAndInsert(KeyToView(keys[i]), locations[i], properties[i]);
    }
    return results;
}

std::vector<ErrorCode> MetaLocalBackend::PutIfAbsent(RequestContext * /*request_context*/,
                                                     const KeyTypeVec &keys,
                                                     const CacheLocationMapVector &locations,
                                                     const PropertyMapVector &properties) noexcept {
    std::vector<ErrorCode> results(keys.size(), EC_OK);
    for (size_t i = 0; i < keys.size(); ++i) {
        results[i] = CreateAndInsertIfAbsent(KeyToView(keys[i]), locations[i], properties[i]);
    }
    return results;
}

std::vector<ErrorCode> MetaLocalBackend::Upsert(RequestContext * /*request_context*/,
                                                const KeyTypeVec &keys,
                                                const CacheLocationMapVector &locations,
                                                const PropertyMapVector &properties) noexcept {
    std::vector<ErrorCode> results(keys.size(), EC_OK);
    for (size_t i = 0; i < keys.size(); ++i) {
        results[i] = UpsertForOneKey(keys[i], locations[i], properties[i]);
    }
    return results;
}

std::vector<ErrorCode> MetaLocalBackend::Delete(RequestContext * /*request_context*/, const KeyTypeVec &keys) noexcept {
    std::vector<ErrorCode> results(keys.size(), EC_OK);
    for (size_t i = 0; i < keys.size(); ++i) {
        results[i] = DeleteForOneKey(keys[i]);
    }
    return results;
}

std::vector<ErrorCode> MetaLocalBackend::DeleteLocations(RequestContext * /*request_context*/,
                                                         const KeyTypeVec &keys,
                                                         const LocationIdsPerKey &location_ids) noexcept {
    std::vector<ErrorCode> results(keys.size(), EC_OK);
    for (size_t i = 0; i < keys.size(); ++i) {
        if (location_ids[i].empty()) {
            continue;
        }
        results[i] = DeleteLocationsForOneKey(keys[i], location_ids[i]);
    }
    return results;
}

// ---------------------------------------------------------------------------
// Write operations (PutIfAbsent + conditional with previous_error_codes)
// ---------------------------------------------------------------------------

std::vector<ErrorCode> MetaLocalBackend::Put(RequestContext *request_context,
                                             const KeyTypeVec &keys,
                                             const CacheLocationMapVector &locations,
                                             const PropertyMapVector &properties,
                                             const std::vector<ErrorCode> &previous_error_codes) noexcept {
    std::vector<ErrorCode> results(keys.size(), EC_OK);
    for (size_t i = 0; i < keys.size(); ++i) {
        results[i] = (previous_error_codes[i] == EC_OK)
                         ? CreateAndInsert(KeyToView(keys[i]), locations[i], properties[i])
                         : previous_error_codes[i];
    }
    return results;
}

std::vector<ErrorCode> MetaLocalBackend::PutIfAbsent(RequestContext *request_context,
                                                     const KeyTypeVec &keys,
                                                     const CacheLocationMapVector &locations,
                                                     const PropertyMapVector &properties,
                                                     const std::vector<ErrorCode> &previous_error_codes) noexcept {
    std::vector<ErrorCode> results(keys.size(), EC_OK);
    for (size_t i = 0; i < keys.size(); ++i) {
        results[i] = (previous_error_codes[i] == EC_OK)
                         ? CreateAndInsertIfAbsent(KeyToView(keys[i]), locations[i], properties[i])
                         : previous_error_codes[i];
    }
    return results;
}

std::vector<ErrorCode> MetaLocalBackend::Upsert(RequestContext *request_context,
                                                const KeyTypeVec &keys,
                                                const CacheLocationMapVector &locations,
                                                const PropertyMapVector &properties,
                                                const std::vector<ErrorCode> &previous_error_codes) noexcept {
    std::vector<ErrorCode> results(keys.size(), EC_OK);
    for (size_t i = 0; i < keys.size(); ++i) {
        results[i] = (previous_error_codes[i] == EC_OK) ? UpsertForOneKey(keys[i], locations[i], properties[i])
                                                        : previous_error_codes[i];
    }
    return results;
}

std::vector<ErrorCode> MetaLocalBackend::Delete(RequestContext *request_context,
                                                const KeyTypeVec &keys,
                                                const std::vector<ErrorCode> &previous_error_codes) noexcept {
    std::vector<ErrorCode> results(keys.size(), EC_OK);
    for (size_t i = 0; i < keys.size(); ++i) {
        results[i] = (previous_error_codes[i] == EC_OK) ? DeleteForOneKey(keys[i]) : previous_error_codes[i];
    }
    return results;
}

std::vector<ErrorCode> MetaLocalBackend::DeleteLocations(RequestContext *request_context,
                                                         const KeyTypeVec &keys,
                                                         const LocationIdsPerKey &location_ids,
                                                         const std::vector<ErrorCode> &previous_error_codes) noexcept {
    std::vector<ErrorCode> results(keys.size(), EC_OK);
    for (size_t i = 0; i < keys.size(); ++i) {
        if (location_ids[i].empty()) {
            results[i] = previous_error_codes[i];
            continue;
        }
        results[i] = (previous_error_codes[i] == EC_OK) ? DeleteLocationsForOneKey(keys[i], location_ids[i])
                                                        : previous_error_codes[i];
    }
    return results;
}

// ---------------------------------------------------------------------------
// Read operations
// ---------------------------------------------------------------------------

ErrorCode MetaLocalBackend::GetForOneKey(KeyType key,
                                         const std::vector<std::string> *field_names,
                                         CacheLocationMap *out_location_map,
                                         PropertyMap *out_property_map,
                                         std::vector<LocationId> *out_location_ids) {
    std::string_view key_sv = KeyToView(key);
    Cache::Handle *handle = cache_->Lookup(key_sv);
    if (!handle) {
        return EC_NOENT;
    }
    auto *item = static_cast<MetaMemCacheItem *>(cache_->Value(handle));
    int64_t stored_time = item->GetLastAccessTime();

    // Record revisit interval before updating access time
    if (revisit_histogram_ && stored_time > 0) {
        int64_t now = TimestampUtil::GetCurrentTimeUs();
        int64_t interval_us = now - stored_time;
        revisit_histogram_->Observe(interval_us);
    }

    item->TouchAccessTime();
    {
        std::shared_lock lock(item->GetMutex());
        if (out_location_map || out_location_ids) {
            const auto &locs = item->GetLocations();
            if (out_location_map) {
                *out_location_map = locs;
            }
            if (out_location_ids) {
                out_location_ids->reserve(locs.size());
                for (const auto &[loc_id, _] : locs) {
                    out_location_ids->push_back(loc_id);
                }
            }
        }
        if (out_property_map) {
            if (field_names) {
                const auto &props = item->GetProperties();
                for (const auto &field_name : *field_names) {
                    if (field_name == PROPERTY_LRU_TIME) {
                        (*out_property_map)[PROPERTY_LRU_TIME] = std::to_string(stored_time);
                        continue;
                    }
                    auto it = props.find(field_name);
                    if (it != props.end()) {
                        (*out_property_map)[field_name] = it->second;
                    }
                }
            } else {
                *out_property_map = item->GetProperties();
                (*out_property_map)[PROPERTY_LRU_TIME] = std::to_string(stored_time);
            }
        }
    }
    cache_->Release(handle);
    return EC_OK;
}

std::vector<ErrorCode> MetaLocalBackend::Get(RequestContext * /*request_context*/,
                                             const KeyTypeVec &keys,
                                             CacheLocationMapVector &out_locations,
                                             PropertyMapVector &out_properties) noexcept {
    std::vector<ErrorCode> results(keys.size(), EC_OK);
    out_locations.resize(keys.size());
    out_properties.resize(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        results[i] = GetForOneKey(keys[i], nullptr, &out_locations[i], &out_properties[i], nullptr);
    }
    return results;
}

std::vector<ErrorCode> MetaLocalBackend::GetLocations(RequestContext * /*request_context*/,
                                                      const KeyTypeVec &keys,
                                                      CacheLocationMapVector &out_locations) noexcept {
    std::vector<ErrorCode> results(keys.size(), EC_OK);
    out_locations.resize(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        results[i] = GetForOneKey(keys[i], nullptr, &out_locations[i], nullptr, nullptr);
    }
    return results;
}

std::vector<std::vector<ErrorCode>> MetaLocalBackend::GetLocations(RequestContext * /*request_context*/,
                                                                   const KeyTypeVec &keys,
                                                                   const LocationIdsPerKey &location_ids,
                                                                   LocationsPerKey &out_locations) noexcept {
    assert(keys.size() == location_ids.size());
    std::vector<std::vector<ErrorCode>> results(keys.size());
    out_locations.resize(keys.size());

    for (size_t i = 0; i < keys.size(); ++i) {
        out_locations[i].resize(location_ids[i].size());

        std::string_view key_sv = KeyToView(keys[i]);
        Cache::Handle *handle = cache_->Lookup(key_sv);
        if (!handle) {
            results[i].assign(location_ids[i].size(), EC_NOENT);
            continue;
        }
        auto *item = static_cast<MetaMemCacheItem *>(cache_->Value(handle));
        item->TouchAccessTime();
        results[i].resize(location_ids[i].size());
        {
            std::shared_lock lock(item->GetMutex());
            const auto &locs = item->GetLocations();
            for (size_t j = 0; j < location_ids[i].size(); ++j) {
                auto it = locs.find(location_ids[i][j]);
                if (it != locs.end()) {
                    out_locations[i][j] = it->second;
                    results[i][j] = EC_OK;
                } else {
                    results[i][j] = EC_NOENT;
                }
            }
        }
        cache_->Release(handle);
    }
    return results;
}

std::vector<ErrorCode> MetaLocalBackend::GetLocationIds(RequestContext * /*request_context*/,
                                                        const KeyTypeVec &keys,
                                                        LocationIdsPerKey &out_location_ids) noexcept {
    std::vector<ErrorCode> results(keys.size(), EC_OK);
    out_location_ids.resize(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        results[i] = GetForOneKey(keys[i], nullptr, nullptr, nullptr, &out_location_ids[i]);
    }
    return results;
}

std::vector<ErrorCode> MetaLocalBackend::GetProperties(RequestContext * /*request_context*/,
                                                       const KeyTypeVec &keys,
                                                       const std::vector<std::string> &field_names,
                                                       PropertyMapVector &out_properties) noexcept {
    std::vector<ErrorCode> results(keys.size(), EC_OK);
    out_properties.resize(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        results[i] = GetForOneKey(keys[i], &field_names, nullptr, &out_properties[i], nullptr);
    }
    return results;
}

std::vector<ErrorCode> MetaLocalBackend::Exists(RequestContext * /*request_context*/,
                                                const KeyTypeVec &keys,
                                                std::vector<bool> &out_is_exist_vec) noexcept {
    std::vector<ErrorCode> results(keys.size(), EC_OK);
    out_is_exist_vec.resize(keys.size(), false);

    for (size_t i = 0; i < keys.size(); ++i) {
        out_is_exist_vec[i] = cache_->Exists(KeyToView(keys[i]));
    }
    return results;
}

std::vector<ErrorCode> MetaLocalBackend::ExistsLocation(RequestContext * /*request_context*/,
                                                        const KeyTypeVec &keys,
                                                        std::vector<bool> &out_exists) noexcept {
    std::vector<ErrorCode> results(keys.size(), EC_OK);
    out_exists.resize(keys.size(), false);
    for (size_t i = 0; i < keys.size(); ++i) {
        std::string_view key_sv = KeyToView(keys[i]);
        Cache::Handle *handle = cache_->Lookup(key_sv);
        if (!handle) {
            results[i] = EC_NOENT;
            continue;
        }
        auto *item = static_cast<MetaMemCacheItem *>(cache_->Value(handle));
        {
            std::shared_lock lock(item->GetMutex());
            out_exists[i] = !item->GetLocations().empty();
        }
        cache_->Release(handle);
    }
    return results;
}

ErrorCode MetaLocalBackend::ListKeys(RequestContext * /*request_context*/,
                                     const std::string &cursor,
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
                                       if (key.size() == sizeof(KeyType)) {
                                           out_keys.push_back(ViewToKey(key));
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

ErrorCode MetaLocalBackend::RandomSample(RequestContext * /*request_context*/,
                                         const int64_t count,
                                         std::vector<KeyType> &out_keys) noexcept {
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

ErrorCode MetaLocalBackend::SampleReclaimKeys(RequestContext * /*request_context*/,
                                              const int64_t count,
                                              std::vector<KeyType> &out_keys) noexcept {
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

int64_t MetaLocalBackend::GetOldestAccessTime() const noexcept {
    int64_t oldest = INT64_MAX;
    size_t num_shards = shard_mask_ + 1;
    for (size_t s = 0; s < num_shards; ++s) {
        oldest = std::min(oldest, shard_oldest_access_time_[s].load(std::memory_order_relaxed));
    }
    return oldest;
}

size_t MetaLocalBackend::CollectOldestKeysFromShard(uint32_t shard_id, size_t count, std::vector<KeyType> &out_keys) {
    std::vector<std::string> string_keys;
    string_keys.reserve(count);
    cache_->GetOldestKeysInShard(shard_id, count, string_keys);
    for (const auto &key_str : string_keys) {
        if (key_str.size() == sizeof(KeyType)) {
            out_keys.push_back(ViewToKey(key_str));
        }
    }
    return string_keys.size();
}

} // namespace kv_cache_manager
