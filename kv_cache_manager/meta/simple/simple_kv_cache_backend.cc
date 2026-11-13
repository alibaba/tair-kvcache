#include "kv_cache_manager/meta/simple/simple_kv_cache_backend.h"

#include <algorithm>
#include <cassert>
#include <charconv>
#include <cstring>
#include <functional>
#include <limits>
#include <utility>

#include "kv_cache_manager/common/cache/allocation_size.h"
#include "kv_cache_manager/common/standard_uri.h"
#include "kv_cache_manager/config/meta_storage_backend_config.h"
#include "kv_cache_manager/meta/common.h"

namespace kv_cache_manager {
namespace {

std::string_view KeyToView(const KeyType &key) {
    return std::string_view(reinterpret_cast<const char *>(&key), sizeof(key));
}

KeyType ViewToKey(std::string_view view) {
    assert(view.size() == sizeof(KeyType));
    KeyType key = 0;
    std::memcpy(&key, view.data(), sizeof(key));
    return key;
}

} // namespace

SimpleKVCacheBackend::SimpleKVCacheBackend(SharedLocationSpecName spec_name) : spec_name_(std::move(spec_name)) {}

SimpleKVCacheBackend::~SimpleKVCacheBackend() {
    [[maybe_unused]] const ErrorCode ec = Close();
    assert(ec == EC_OK);
}

std::string SimpleKVCacheBackend::GetStorageType() noexcept { return META_LOCAL_BACKEND_TYPE_STR; }

ErrorCode SimpleKVCacheBackend::Init(const std::string &instance_id,
                                     const std::shared_ptr<MetaStorageBackendConfig> &config) noexcept {
    if (cache_) {
        return EC_EXIST;
    }
    if (instance_id.empty() || !config || !spec_name_ || spec_name_->empty()) {
        return EC_BADARGS;
    }
    size_t capacity_mb = META_LOCAL_BACKEND_DEFAULT_CAPACITY;
    int32_t shard_bits = META_LOCAL_BACKEND_DEFAULT_NUM_SHARD_BITS;
    if (!config->GetStorageUri().empty()) {
        StandardUri uri;
        if (!uri.Parse(config->GetStorageUri()) || !uri.Valid()) {
            return EC_BADARGS;
        }
        uri.GetParamAs("capacity", capacity_mb);
        uri.GetParamAs("num_shard_bits", shard_bits);
    }
    if (capacity_mb == 0 || capacity_mb > std::numeric_limits<size_t>::max() / (1024 * 1024ULL) || shard_bits < 0 ||
        shard_bits >= 20) {
        return EC_BADARGS;
    }
    LRUCacheOptions options;
    options.capacity = capacity_mb * 1024 * 1024ULL;
    options.num_shard_bits = shard_bits;
    options.strict_capacity_limit = true;
    options.no_evict_on_insert = true;
    options.high_pri_pool_ratio = 0.0;
    auto cache = std::make_shared<LRUCache>(options);
    auto helper = std::make_shared<Cache::CacheItemHelper>();
    helper->del_cb = DeleteCacheValue;
    const size_t shard_count = size_t{1} << shard_bits;
    std::vector<std::mutex> mutexes(shard_count);
    mutation_mutex_shards_.swap(mutexes);
    cache_item_helper_ = std::move(helper);
    cache_ = std::move(cache);
    return EC_OK;
}

ErrorCode SimpleKVCacheBackend::Open() noexcept {
    if (!cache_) {
        return EC_ERROR;
    }
    open_ = true;
    return EC_OK;
}

ErrorCode SimpleKVCacheBackend::Close() noexcept {
    open_ = false;
    cache_.reset();
    cache_item_helper_.reset();
    std::vector<std::mutex>().swap(mutation_mutex_shards_);
    return EC_OK;
}

void SimpleKVCacheBackend::DeleteCacheValue(Cache::ObjectPtr value, MemoryAllocator * /*allocator*/) {
    delete static_cast<CacheValue *>(value);
}

uint32_t SimpleKVCacheBackend::ShardIndex(KeyType key) const noexcept {
    return LRUCacheShard::ComputeHash(KeyToView(key), cache_->GetHashSeed()) &
           static_cast<uint32_t>(mutation_mutex_shards_.size() - 1);
}

std::shared_ptr<SimpleKVCacheItem> SimpleKVCacheBackend::LoadItem(KeyType key) {
    CacheValue item;
    auto load_item = [&item](Cache::ObjectPtr value, size_t, const Cache::CacheItemHelper *) -> ssize_t {
        item = *static_cast<CacheValue *>(value);
        return 0;
    };
    [[maybe_unused]] const bool found = cache_->ApplyToEntryNoTouch(KeyToView(key), std::ref(load_item));
    assert(found == static_cast<bool>(item));
    return item;
}

ErrorCode SimpleKVCacheBackend::CheckWriteCapacity(KeyType key,
                                                   const CacheLocationMap &locations,
                                                   const PropertyMap &properties,
                                                   bool replace) noexcept {
    if (!open_) {
        return EC_ERROR;
    }
    if (!properties.empty()) {
        return EC_UNIMPLEMENTED;
    }
    const CacheValue current = LoadItem(key);
    size_t growth = 0;
    const ErrorCode ec = SimpleKVCacheItem::EstimateWriteGrowth(current, locations, replace, growth);
    if (ec != EC_OK) {
        return ec;
    }
    if (!current) {
        growth += EstimateEntryOverhead() + EstimateAllocationSize(sizeof(LRUHandle) - 1 + sizeof(KeyType));
    }
    return growth <= cache_->GetAvailableCapacity(KeyToView(key)) ? EC_OK : EC_NOSPC;
}

size_t SimpleKVCacheBackend::EstimateEntryOverhead() noexcept {
    // A separately allocated shared_ptr control block needs the managed pointer as well as its counters.
    return EstimateAllocationSize(sizeof(CacheValue)) + EstimateAllocationSize(sizeof(void *) * 4);
}

ErrorCode SimpleKVCacheBackend::PublishItem(KeyType key, const std::shared_ptr<SimpleKVCacheItem> &item, bool replace) {
    if (!replace) {
        auto value = std::make_unique<CacheValue>(item);
        const size_t charge = item->Size() + EstimateEntryOverhead();
        Cache::Handle *handle = nullptr;
        const ErrorCode ec = cache_->Insert(KeyToView(key), value.get(), cache_item_helper_.get(), charge, &handle);
        if (ec != EC_OK) {
            return ec;
        }
        static_cast<void>(value.release()); // Insert 成功后由 LRU 持有 value。
        const bool erased = cache_->Release(handle);
        (void)erased;
        return EC_OK;
    }
    const size_t next_size = item->Size();
    CacheValue retired;
    auto publish_item = [&](Cache::ObjectPtr value, size_t, const Cache::CacheItemHelper *) -> ssize_t {
        CacheValue &cached = *static_cast<CacheValue *>(value);
        const size_t previous_size = cached->Size();
        retired = std::move(cached);
        cached = item;
        return next_size >= previous_size ? static_cast<ssize_t>(next_size - previous_size)
                                          : -static_cast<ssize_t>(previous_size - next_size);
    };
    [[maybe_unused]] const bool replaced = cache_->ApplyToEntryNoTouch(KeyToView(key), std::ref(publish_item));
    assert(replaced);
    return EC_OK;
}

std::vector<ErrorCode> SimpleKVCacheBackend::Write(const KeyTypeVec &keys,
                                                   const CacheLocationMapVector &locations,
                                                   const PropertyMapVector &properties,
                                                   const std::vector<ErrorCode> &previous_error_codes,
                                                   WriteMode mode) {
    if (locations.size() != keys.size() || properties.size() != keys.size() ||
        (!previous_error_codes.empty() && previous_error_codes.size() != keys.size())) {
        return std::vector<ErrorCode>(keys.size(), EC_BADARGS);
    }
    std::vector<ErrorCode> results(keys.size(), EC_OK);
    if (!previous_error_codes.empty()) {
        results = previous_error_codes;
    }
    if (!open_) {
        std::replace(results.begin(), results.end(), EC_OK, EC_ERROR);
        return results;
    }
    std::unordered_map<LocationId, InternedLocationId> ids;
    for (size_t i = 0; i < keys.size(); ++i) {
        if (results[i] != EC_OK) {
            continue;
        }
        if (!properties[i].empty()) {
            results[i] = EC_UNIMPLEMENTED;
            continue;
        }
        for (const CacheLocationMap::value_type &entry : locations[i]) {
            if (ids.find(entry.first) == ids.end()) {
                [[maybe_unused]] const bool inserted =
                    ids.emplace(entry.first, location_id_interner_.Intern(entry.first)).second;
                assert(inserted);
            }
        }
    }
    for (size_t i = 0; i < keys.size(); ++i) {
        if (results[i] != EC_OK) {
            continue;
        }
        std::lock_guard<std::mutex> mutation_lock(mutation_mutex_shards_[ShardIndex(keys[i])]);
        const CacheValue current = LoadItem(keys[i]);
        if (mode == WriteMode::PutIfAbsent && current) {
            continue;
        }
        if (mode == WriteMode::Upsert && current && locations[i].empty()) {
            continue;
        }
        if (previous_error_codes.empty() && current && mode != WriteMode::PutIfAbsent) {
            size_t growth = 0;
            results[i] = SimpleKVCacheItem::EstimateWriteGrowth(current, locations[i], mode == WriteMode::Put, growth);
            if (results[i] == EC_OK && growth > cache_->GetAvailableCapacity(KeyToView(keys[i]))) {
                results[i] = EC_NOSPC;
            }
            if (results[i] != EC_OK) {
                continue;
            }
        }
        Cache::ObjectPtr value = nullptr;
        if (mode == WriteMode::Upsert && current) {
            results[i] = SimpleKVCacheItem::Merge(*current, locations[i], spec_name_, ids, value);
        } else {
            results[i] = SimpleKVCacheItem::Create(locations[i], spec_name_, ids, value);
        }
        if (results[i] != EC_OK) {
            continue;
        }
        const CacheValue item(static_cast<SimpleKVCacheItem *>(value),
                              [](SimpleKVCacheItem *ptr) { SimpleKVCacheItem::Deleter(ptr, nullptr); });
        results[i] = PublishItem(keys[i], item, static_cast<bool>(current));
    }
    return results;
}

std::vector<ErrorCode> SimpleKVCacheBackend::DeleteKeys(const KeyTypeVec &keys,
                                                        const std::vector<ErrorCode> &previous_error_codes) {
    if (!previous_error_codes.empty() && previous_error_codes.size() != keys.size()) {
        return std::vector<ErrorCode>(keys.size(), EC_BADARGS);
    }
    std::vector<ErrorCode> results(keys.size(), EC_OK);
    if (!previous_error_codes.empty()) {
        results = previous_error_codes;
    }
    if (!open_) {
        std::replace(results.begin(), results.end(), EC_OK, EC_ERROR);
        return results;
    }
    for (size_t i = 0; i < keys.size(); ++i) {
        // 持久层 NOENT 也表示删除已收敛，仍需清除热层残留。
        if (results[i] != EC_OK && results[i] != EC_NOENT) {
            continue;
        }
        std::lock_guard<std::mutex> mutation_lock(mutation_mutex_shards_[ShardIndex(keys[i])]);
        results[i] = cache_->Erase(KeyToView(keys[i])) ? EC_OK : EC_NOENT;
    }
    return results;
}

std::vector<ErrorCode> SimpleKVCacheBackend::DeleteLocationBatch(const KeyTypeVec &keys,
                                                                 const LocationIdsPerKey &location_ids,
                                                                 const std::vector<ErrorCode> &previous_error_codes) {
    if (location_ids.size() != keys.size() ||
        (!previous_error_codes.empty() && previous_error_codes.size() != keys.size())) {
        return std::vector<ErrorCode>(keys.size(), EC_BADARGS);
    }
    std::vector<ErrorCode> results(keys.size(), EC_OK);
    if (!previous_error_codes.empty()) {
        results = previous_error_codes;
    }
    if (!open_) {
        std::replace(results.begin(), results.end(), EC_OK, EC_ERROR);
        return results;
    }
    for (size_t i = 0; i < keys.size(); ++i) {
        if ((results[i] != EC_OK && results[i] != EC_NOENT) || location_ids[i].empty()) {
            continue;
        }
        std::lock_guard<std::mutex> mutation_lock(mutation_mutex_shards_[ShardIndex(keys[i])]);
        const CacheValue current = LoadItem(keys[i]);
        if (!current) {
            results[i] = EC_NOENT;
            continue;
        }
        Cache::ObjectPtr value = nullptr;
        results[i] = SimpleKVCacheItem::Remove(*current, location_ids[i], value);
        if (results[i] != EC_OK) {
            continue;
        }
        const CacheValue item(static_cast<SimpleKVCacheItem *>(value),
                              [](SimpleKVCacheItem *ptr) { SimpleKVCacheItem::Deleter(ptr, nullptr); });
        results[i] = PublishItem(keys[i], item, true);
    }
    return results;
}

ErrorCode
SimpleKVCacheBackend::CollectScanBatch(const std::string &cursor,
                                       int64_t limit,
                                       bool with_values,
                                       std::vector<std::pair<KeyType, std::shared_ptr<SimpleKVCacheItem>>> &out_entries,
                                       std::string &out_next_cursor) {
    out_entries.clear();
    out_next_cursor = SCAN_BASE_CURSOR;
    if (limit <= 0) {
        return EC_BADARGS;
    }
    uint32_t shard = 0;
    size_t state = 0;
    if (cursor != SCAN_BASE_CURSOR) {
        const size_t split = cursor.find(':');
        if (split == std::string::npos) {
            return EC_BADARGS;
        }
        const char *begin = cursor.data();
        const char *end = begin + cursor.size();
        const std::from_chars_result shard_result = std::from_chars(begin, begin + split, shard);
        const std::from_chars_result state_result = std::from_chars(begin + split + 1, end, state);
        if (shard_result.ec != std::errc{} || shard_result.ptr != begin + split || state_result.ec != std::errc{} ||
            state_result.ptr != end || state == std::numeric_limits<size_t>::max() || shard >= cache_->GetNumShards()) {
            return EC_BADARGS;
        }
    }
    // 游标保存 shard 和高位 hash 进度；每次只取一个桶，允许并发更新导致重复，删除前由 indexer 复核。
    while (shard < cache_->GetNumShards() && out_entries.size() < static_cast<size_t>(limit)) {
        cache_->GetShard(shard).ApplyToSomeEntries(
            [&](const std::string_view &key, Cache::ObjectPtr value, size_t, const Cache::CacheItemHelper *) {
                CacheValue item;
                if (with_values) {
                    item = *static_cast<CacheValue *>(value);
                }
                out_entries.emplace_back(ViewToKey(key), std::move(item));
            },
            1,
            &state);
        if (state == std::numeric_limits<size_t>::max()) {
            ++shard;
            state = 0;
        }
    }
    if (shard < cache_->GetNumShards()) {
        out_next_cursor = std::to_string(shard) + ":" + std::to_string(state);
    }
    return EC_OK;
}

std::vector<ErrorCode> SimpleKVCacheBackend::ReadMaps(const KeyTypeVec &keys, CacheLocationMapVector &out_locations) {
    out_locations.clear();
    out_locations.resize(keys.size());
    std::vector<ErrorCode> results(keys.size(), EC_NOENT);
    if (!open_) {
        std::fill(results.begin(), results.end(), EC_ERROR);
        return results;
    }
    for (size_t i = 0; i < keys.size(); ++i) {
        const CacheValue item = LoadItem(keys[i]);
        if (!item) {
            continue;
        }
        results[i] = item->GetLocations(spec_name_, out_locations[i]);
    }
    return results;
}

std::vector<ErrorCode> SimpleKVCacheBackend::ReadIds(const KeyTypeVec &keys, LocationIdsPerKey &out_location_ids) {
    out_location_ids.clear();
    out_location_ids.resize(keys.size());
    std::vector<ErrorCode> results(keys.size(), EC_NOENT);
    if (!open_) {
        std::fill(results.begin(), results.end(), EC_ERROR);
        return results;
    }
    for (size_t i = 0; i < keys.size(); ++i) {
        const CacheValue item = LoadItem(keys[i]);
        if (!item) {
            continue;
        }
        results[i] = item->GetLocationIds(out_location_ids[i]);
    }
    return results;
}

std::vector<std::vector<ErrorCode>> SimpleKVCacheBackend::ReadLocations(const KeyTypeVec &keys,
                                                                        const LocationIdsPerKey &location_ids,
                                                                        LocationsPerKey &out_locations,
                                                                        std::vector<ErrorCode> &out_key_error_codes) {
    out_locations.clear();
    out_locations.resize(keys.size());
    out_key_error_codes.assign(keys.size(), EC_BADARGS);
    std::vector<std::vector<ErrorCode>> results(keys.size());
    for (size_t i = 0; i < keys.size() && i < location_ids.size(); ++i) {
        results[i].assign(location_ids[i].size(), EC_BADARGS);
        out_locations[i].resize(location_ids[i].size());
    }
    if (location_ids.size() != keys.size()) {
        return results;
    }
    if (!open_) {
        out_key_error_codes.assign(keys.size(), EC_ERROR);
        for (std::vector<ErrorCode> &codes : results) {
            std::fill(codes.begin(), codes.end(), EC_ERROR);
        }
        return results;
    }
    for (size_t i = 0; i < keys.size(); ++i) {
        const CacheValue item = LoadItem(keys[i]);
        out_key_error_codes[i] = item ? EC_OK : EC_NOENT;
        std::fill(results[i].begin(), results[i].end(), EC_NOENT);
        if (!item) {
            continue;
        }
        for (size_t j = 0; j < location_ids[i].size(); ++j) {
            results[i][j] = item->GetLocation(spec_name_, location_ids[i][j], out_locations[i][j]);
        }
    }
    return results;
}

std::vector<ErrorCode> SimpleKVCacheBackend::Get(RequestContext * /*request_context*/,
                                                 const KeyTypeVec &keys,
                                                 CacheLocationMapVector &out_locations,
                                                 PropertyMapVector &out_properties) noexcept {
    out_properties.assign(keys.size(), PropertyMap{});
    return ReadMaps(keys, out_locations);
}

std::vector<ErrorCode> SimpleKVCacheBackend::GetLocations(RequestContext * /*request_context*/,
                                                          const KeyTypeVec &keys,
                                                          CacheLocationMapVector &out_locations) noexcept {
    return ReadMaps(keys, out_locations);
}

std::vector<ErrorCode> SimpleKVCacheBackend::GetLocationValues(RequestContext * /*request_context*/,
                                                               const KeyTypeVec &keys,
                                                               LocationsPerKey &out_locations) noexcept {
    out_locations.clear();
    out_locations.resize(keys.size());
    std::vector<ErrorCode> results(keys.size(), EC_NOENT);
    if (!open_) {
        std::fill(results.begin(), results.end(), EC_ERROR);
        return results;
    }
    for (size_t i = 0; i < keys.size(); ++i) {
        const CacheValue item = LoadItem(keys[i]);
        if (!item) {
            continue;
        }
        results[i] = item->GetLocationValues(spec_name_, out_locations[i]);
    }
    return results;
}

std::vector<ErrorCode> SimpleKVCacheBackend::GetLocationValuesCompact(RequestContext * /*request_context*/,
                                                                      const KeyType *keys,
                                                                      size_t key_count,
                                                                      CompactLocationsPerKey &out_locations) noexcept {
    out_locations.Clear(0, 0);
    if (keys == nullptr && key_count != 0) {
        out_locations.Clear(key_count, 0);
        for (size_t i = 0; i < key_count; ++i) {
            out_locations.FinishKey();
        }
        return std::vector<ErrorCode>(key_count, EC_BADARGS);
    }
    std::vector<ErrorCode> results(key_count, EC_NOENT);
    if (!open_) {
        std::fill(results.begin(), results.end(), EC_ERROR);
        for (size_t i = 0; i < key_count; ++i) {
            out_locations.FinishKey();
        }
        return results;
    }
    out_locations.Clear(key_count, key_count);
    for (size_t i = 0; i < key_count; ++i) {
        const CacheValue item = LoadItem(keys[i]);
        if (item) {
            results[i] = item->AppendLocationValues(spec_name_, out_locations.values);
        }
        out_locations.FinishKey();
    }
    return results;
}

std::vector<std::vector<ErrorCode>> SimpleKVCacheBackend::GetLocations(RequestContext * /*request_context*/,
                                                                       const KeyTypeVec &keys,
                                                                       const LocationIdsPerKey &location_ids,
                                                                       LocationsPerKey &out_locations) noexcept {
    std::vector<ErrorCode> key_results;
    return ReadLocations(keys, location_ids, out_locations, key_results);
}

std::vector<std::vector<ErrorCode>>
SimpleKVCacheBackend::GetLocationsWithKeyStatus(RequestContext * /*request_context*/,
                                                const KeyTypeVec &keys,
                                                const LocationIdsPerKey &location_ids,
                                                LocationsPerKey &out_locations,
                                                std::vector<ErrorCode> &out_key_error_codes) noexcept {
    return ReadLocations(keys, location_ids, out_locations, out_key_error_codes);
}

std::vector<ErrorCode>
SimpleKVCacheBackend::GetSingleLocationsWithKeyStatus(RequestContext * /*request_context*/,
                                                      const KeyTypeVec &keys,
                                                      const LocationIdRefVector &location_ids,
                                                      CacheLocationVector &out_locations,
                                                      std::vector<ErrorCode> &out_key_error_codes) noexcept {
    out_locations.assign(keys.size(), CacheLocationConstPtr{});
    out_key_error_codes.assign(keys.size(), EC_BADARGS);
    if (location_ids.size() != keys.size() ||
        std::find(location_ids.begin(), location_ids.end(), nullptr) != location_ids.end()) {
        return std::vector<ErrorCode>(keys.size(), EC_BADARGS);
    }
    std::vector<ErrorCode> results(keys.size(), EC_NOENT);
    if (!open_) {
        out_key_error_codes.assign(keys.size(), EC_ERROR);
        std::fill(results.begin(), results.end(), EC_ERROR);
        return results;
    }
    for (size_t i = 0; i < keys.size(); ++i) {
        const CacheValue item = LoadItem(keys[i]);
        out_key_error_codes[i] = item ? EC_OK : EC_NOENT;
        if (!item) {
            continue;
        }
        results[i] = item->GetLocation(spec_name_, *location_ids[i], out_locations[i]);
    }
    return results;
}

std::vector<ErrorCode> SimpleKVCacheBackend::GetLocationsForMaintenance(
    RequestContext * /*request_context*/, const KeyTypeVec &keys, CacheLocationMapVector &out_locations) noexcept {
    return ReadMaps(keys, out_locations);
}

std::vector<std::vector<ErrorCode>>
SimpleKVCacheBackend::GetLocationsForMaintenance(RequestContext * /*request_context*/,
                                                 const KeyTypeVec &keys,
                                                 const LocationIdsPerKey &location_ids,
                                                 LocationsPerKey &out_locations) noexcept {
    std::vector<ErrorCode> key_results;
    return ReadLocations(keys, location_ids, out_locations, key_results);
}

std::vector<ErrorCode> SimpleKVCacheBackend::GetLocationIds(RequestContext * /*request_context*/,
                                                            const KeyTypeVec &keys,
                                                            LocationIdsPerKey &out_location_ids) noexcept {
    return ReadIds(keys, out_location_ids);
}

std::vector<ErrorCode> SimpleKVCacheBackend::GetLocationIdsForMaintenance(
    RequestContext * /*request_context*/, const KeyTypeVec &keys, LocationIdsPerKey &out_location_ids) noexcept {
    return ReadIds(keys, out_location_ids);
}

std::vector<ErrorCode> SimpleKVCacheBackend::GetLocationMapsForMaintenance(
    RequestContext * /*request_context*/, const KeyTypeVec &keys, CacheLocationMapVector &out_locations) noexcept {
    return ReadMaps(keys, out_locations);
}

std::vector<ErrorCode> SimpleKVCacheBackend::GetProperties(RequestContext * /*request_context*/,
                                                           const KeyTypeVec &keys,
                                                           const std::vector<std::string> & /*field_names*/,
                                                           PropertyMapVector &out_properties) noexcept {
    out_properties.assign(keys.size(), PropertyMap{});
    return std::vector<ErrorCode>(keys.size(), EC_UNIMPLEMENTED);
}

std::vector<ErrorCode> SimpleKVCacheBackend::Exists(RequestContext * /*request_context*/,
                                                    const KeyTypeVec &keys,
                                                    std::vector<bool> &out_is_exist_vec) noexcept {
    out_is_exist_vec.assign(keys.size(), false);
    std::vector<ErrorCode> results(keys.size(), EC_OK);
    if (!open_) {
        std::fill(results.begin(), results.end(), EC_ERROR);
        return results;
    }
    for (size_t i = 0; i < keys.size(); ++i) {
        out_is_exist_vec[i] = cache_->Exists(KeyToView(keys[i]));
    }
    return results;
}

std::vector<ErrorCode> SimpleKVCacheBackend::ExistsLocation(RequestContext * /*request_context*/,
                                                            const KeyTypeVec &keys,
                                                            std::vector<bool> &out_exists) noexcept {
    out_exists.assign(keys.size(), false);
    std::vector<ErrorCode> results(keys.size(), EC_NOENT);
    if (!open_) {
        std::fill(results.begin(), results.end(), EC_ERROR);
        return results;
    }
    for (size_t i = 0; i < keys.size(); ++i) {
        const CacheValue item = LoadItem(keys[i]);
        if (item) {
            out_exists[i] = item->LocationCount() != 0;
            results[i] = EC_OK;
        }
    }
    return results;
}

ErrorCode SimpleKVCacheBackend::ListKeys(RequestContext * /*request_context*/,
                                         const std::string &cursor,
                                         const int64_t limit,
                                         std::string &out_next_cursor,
                                         std::vector<KeyType> &out_keys) noexcept {
    out_keys.clear();
    out_next_cursor = SCAN_BASE_CURSOR;
    if (!open_) {
        return EC_ERROR;
    }
    std::vector<std::pair<KeyType, CacheValue>> entries;
    const ErrorCode ec = CollectScanBatch(cursor, limit, false, entries, out_next_cursor);
    if (ec == EC_OK) {
        out_keys.reserve(entries.size());
        for (const std::pair<KeyType, CacheValue> &entry : entries) {
            out_keys.push_back(entry.first);
        }
    }
    return ec;
}

ErrorCode SimpleKVCacheBackend::ScanLocationsForMaintenance(RequestContext * /*request_context*/,
                                                            const std::string &cursor,
                                                            int64_t limit,
                                                            MaintenanceScanBatch &out) noexcept {
    out.Clear();
    if (!open_) {
        return EC_ERROR;
    }
    std::vector<std::pair<KeyType, CacheValue>> entries;
    const ErrorCode ec = CollectScanBatch(cursor, limit, true, entries, out.next_cursor);
    if (ec != EC_OK) {
        return ec;
    }
    out.keys.reserve(entries.size());
    out.locations.resize(entries.size());
    out.location_results.reserve(entries.size());
    for (size_t i = 0; i < entries.size(); ++i) {
        out.keys.push_back(entries[i].first);
        out.location_results.push_back(entries[i].second->GetLocations(spec_name_, out.locations[i]));
    }
    return EC_OK;
}

std::vector<ErrorCode>
SimpleKVCacheBackend::GetLastAccessTimesForMaintenance(RequestContext * /*request_context*/,
                                                       const KeyTypeVec &keys,
                                                       std::vector<int64_t> &out_last_access_times) noexcept {
    out_last_access_times.assign(keys.size(), 0);
    return std::vector<ErrorCode>(keys.size(), EC_UNIMPLEMENTED);
}

// 实例元信息由原 cached manager 写入持久层，本地沿用原 local 的空实现。
ErrorCode SimpleKVCacheBackend::PutMetaData(const FieldMap & /*field_maps*/) noexcept { return EC_OK; }

ErrorCode SimpleKVCacheBackend::GetMetaData(FieldMap & /*field_maps*/) noexcept { return EC_NOENT; }

ErrorCode SimpleKVCacheBackend::DeleteMetaData() noexcept { return EC_OK; }

size_t SimpleKVCacheBackend::GetMemUsage() const noexcept { return cache_ ? cache_->GetUsage() : 0; }

int64_t SimpleKVCacheBackend::GetOldestAccessTime() const noexcept {
    // 仅将 LRU 用作 KV 表，不维护访问顺序，LRU age 返回现有不可用值。
    return std::numeric_limits<int64_t>::max();
}

std::vector<ErrorCode> SimpleKVCacheBackend::Put(RequestContext * /*request_context*/,
                                                 const KeyTypeVec &keys,
                                                 const CacheLocationMapVector &locations,
                                                 const PropertyMapVector &properties) noexcept {
    return Write(keys, locations, properties, {}, WriteMode::Put);
}

std::vector<ErrorCode> SimpleKVCacheBackend::Put(RequestContext * /*request_context*/,
                                                 const KeyTypeVec &keys,
                                                 const CacheLocationMapVector &locations,
                                                 const PropertyMapVector &properties,
                                                 const std::vector<ErrorCode> &previous_error_codes) noexcept {
    if (previous_error_codes.size() != keys.size()) {
        return std::vector<ErrorCode>(keys.size(), EC_BADARGS);
    }
    return Write(keys, locations, properties, previous_error_codes, WriteMode::Put);
}

std::vector<ErrorCode> SimpleKVCacheBackend::PutIfAbsent(RequestContext * /*request_context*/,
                                                         const KeyTypeVec &keys,
                                                         const CacheLocationMapVector &locations,
                                                         const PropertyMapVector &properties) noexcept {
    return Write(keys, locations, properties, {}, WriteMode::PutIfAbsent);
}

std::vector<ErrorCode> SimpleKVCacheBackend::PutIfAbsent(RequestContext * /*request_context*/,
                                                         const KeyTypeVec &keys,
                                                         const CacheLocationMapVector &locations,
                                                         const PropertyMapVector &properties,
                                                         const std::vector<ErrorCode> &previous_error_codes) noexcept {
    if (previous_error_codes.size() != keys.size()) {
        return std::vector<ErrorCode>(keys.size(), EC_BADARGS);
    }
    return Write(keys, locations, properties, previous_error_codes, WriteMode::PutIfAbsent);
}

std::vector<ErrorCode> SimpleKVCacheBackend::Upsert(RequestContext * /*request_context*/,
                                                    const KeyTypeVec &keys,
                                                    const CacheLocationMapVector &locations,
                                                    const PropertyMapVector &properties) noexcept {
    return Write(keys, locations, properties, {}, WriteMode::Upsert);
}

std::vector<ErrorCode> SimpleKVCacheBackend::Upsert(RequestContext * /*request_context*/,
                                                    const KeyTypeVec &keys,
                                                    const CacheLocationMapVector &locations,
                                                    const PropertyMapVector &properties,
                                                    const std::vector<ErrorCode> &previous_error_codes) noexcept {
    if (previous_error_codes.size() != keys.size()) {
        return std::vector<ErrorCode>(keys.size(), EC_BADARGS);
    }
    return Write(keys, locations, properties, previous_error_codes, WriteMode::Upsert);
}

std::vector<ErrorCode> SimpleKVCacheBackend::Delete(RequestContext * /*request_context*/,
                                                    const KeyTypeVec &keys) noexcept {
    return DeleteKeys(keys, {});
}

std::vector<ErrorCode> SimpleKVCacheBackend::Delete(RequestContext * /*request_context*/,
                                                    const KeyTypeVec &keys,
                                                    const std::vector<ErrorCode> &previous_error_codes) noexcept {
    if (previous_error_codes.size() != keys.size()) {
        return std::vector<ErrorCode>(keys.size(), EC_BADARGS);
    }
    return DeleteKeys(keys, previous_error_codes);
}

std::vector<ErrorCode> SimpleKVCacheBackend::DeleteLocations(RequestContext * /*request_context*/,
                                                             const KeyTypeVec &keys,
                                                             const LocationIdsPerKey &location_ids) noexcept {
    return DeleteLocationBatch(keys, location_ids, {});
}

std::vector<ErrorCode>
SimpleKVCacheBackend::DeleteLocations(RequestContext * /*request_context*/,
                                      const KeyTypeVec &keys,
                                      const LocationIdsPerKey &location_ids,
                                      const std::vector<ErrorCode> &previous_error_codes) noexcept {
    if (previous_error_codes.size() != keys.size()) {
        return std::vector<ErrorCode>(keys.size(), EC_BADARGS);
    }
    return DeleteLocationBatch(keys, location_ids, previous_error_codes);
}

std::vector<ErrorCode> SimpleKVCacheBackend::DeleteLocationsForMaintenance(
    RequestContext * /*request_context*/, const KeyTypeVec &keys, const LocationIdsPerKey &location_ids) noexcept {
    return DeleteLocationBatch(keys, location_ids, {});
}

std::vector<ErrorCode>
SimpleKVCacheBackend::DeleteLocationsForMaintenance(RequestContext * /*request_context*/,
                                                    const KeyTypeVec &keys,
                                                    const LocationIdsPerKey &location_ids,
                                                    const std::vector<ErrorCode> &previous_error_codes) noexcept {
    if (previous_error_codes.size() != keys.size()) {
        return std::vector<ErrorCode>(keys.size(), EC_BADARGS);
    }
    return DeleteLocationBatch(keys, location_ids, previous_error_codes);
}

ErrorCode SimpleKVCacheBackend::RandomSample(RequestContext * /*request_context*/,
                                             const int64_t /*count*/,
                                             std::vector<KeyType> &out_keys) noexcept {
    out_keys.clear();
    return EC_UNIMPLEMENTED;
}

ErrorCode SimpleKVCacheBackend::SampleReclaimKeys(RequestContext * /*request_context*/,
                                                  const int64_t /*count*/,
                                                  std::vector<KeyType> &out_keys) noexcept {
    out_keys.clear();
    return EC_UNIMPLEMENTED;
}

ErrorCode SimpleKVCacheBackend::SampleReclaimCandidates(RequestContext * /*request_context*/,
                                                        int64_t /*count*/,
                                                        ReclaimCandidateVector &out_candidates,
                                                        bool /*require_read_success*/) noexcept {
    out_candidates.clear();
    return EC_UNIMPLEMENTED;
}

size_t SimpleKVCacheBackend::TouchKeysForMaintenance(const KeyTypeVec & /*keys*/) noexcept { return 0; }

size_t SimpleKVCacheBackend::GetLocalKeyCount() const noexcept { return cache_ ? cache_->GetOccupancyCount() : 0; }


} // namespace kv_cache_manager
