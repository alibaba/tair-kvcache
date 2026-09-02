#include "kv_cache_manager/meta/meta_local_backend.h"

#include <algorithm>
#include <climits>
#include <optional>
#include <random>
#include <unordered_set>
#include <utility>

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/standard_uri.h"
#include "kv_cache_manager/config/meta_storage_backend_config.h"

namespace kv_cache_manager {

namespace {

constexpr size_t kMaxRetainedCompactReadKeys = 16384;

struct CompactReadScratch {
    bool in_use = false;
    std::vector<std::string_view> key_views;
    std::vector<Cache::Handle *> handles;
    std::vector<int64_t> revisit_intervals;
    Cache::BatchOperationScratch cache_batch;
};

thread_local CompactReadScratch tls_compact_read_scratch;

class CompactReadScratchLease {
public:
    explicit CompactReadScratchLease(size_t key_count) {
        if (key_count <= kMaxRetainedCompactReadKeys && !tls_compact_read_scratch.in_use) {
            scratch_ = &tls_compact_read_scratch;
            scratch_->in_use = true;
            uses_thread_local_ = true;
        } else {
            local_.emplace();
            scratch_ = &*local_;
        }
    }

    ~CompactReadScratchLease() {
        if (uses_thread_local_) {
            scratch_->in_use = false;
        }
    }

    CompactReadScratch &get() noexcept { return *scratch_; }

private:
    std::optional<CompactReadScratch> local_;
    CompactReadScratch *scratch_ = nullptr;
    bool uses_thread_local_ = false;
};

} // namespace

SingleLocationRmwScratch::~SingleLocationRmwScratch() { ReleaseRetainedHandles(); }

void SingleLocationRmwScratch::ReleaseRetainedHandles() noexcept {
    if (!retained_handle_owner) {
        return;
    }
    retained_handle_owner->ReleaseBatchWithScratch(handles.data(), handles.size(), &cache_batch);
    std::fill(handles.begin(), handles.end(), nullptr);
    retained_handle_owner = nullptr;
}

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
    reclaim_sample_shard_cursor_.store(0, std::memory_order_relaxed);

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

ErrorCode MetaLocalBackend::UpdateHandleInPlace(Cache::Handle *handle,
                                                const CacheLocationMap &locations,
                                                const PropertyMap &properties) {
    assert(handle != nullptr);
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
                charge_delta += static_cast<ssize_t>(MetaMemCacheItem::EstimateLocationEntryUsage(loc_id, loc_ptr));
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
    return EC_OK;
}

ErrorCode MetaLocalBackend::UpdateHandleInPlaceSingleLocation(Cache::Handle *handle,
                                                              const LocationId &location_id,
                                                              CacheLocationConstPtr location,
                                                              CacheLocationVector *retired_locations) {
    assert(handle != nullptr);
    auto *existing = static_cast<MetaMemCacheItem *>(cache_->Value(handle));
    existing->TouchAccessTime();
    ssize_t charge_delta = 0;
    {
        std::unique_lock lock(existing->GetMutex());
        auto &existing_locations = existing->GetMutableLocations();
        auto it = existing_locations.end();
        if (existing_locations.size() == 1) {
            auto only = existing_locations.begin();
            if (only->first == location_id) {
                it = only;
            }
        } else if (!existing_locations.empty()) {
            it = existing_locations.find(location_id);
        }
        if (it != existing_locations.end()) {
            const ssize_t old_usage = it->second ? static_cast<ssize_t>(it->second->EstimateMemUsage()) : 0;
            const ssize_t new_usage = location ? static_cast<ssize_t>(location->EstimateMemUsage()) : 0;
            if (retired_locations) {
                assert(retired_locations->size() < retired_locations->capacity());
                retired_locations->push_back(std::move(it->second));
                it->second = std::move(location);
            } else {
                it->second = std::move(location);
            }
            charge_delta = new_usage - old_usage;
        } else {
            charge_delta = static_cast<ssize_t>(MetaMemCacheItem::EstimateLocationEntryUsage(location_id, location));
            existing_locations.emplace(location_id, std::move(location));
        }
    }
    if (charge_delta != 0) {
        cache_->AdjustCharge(handle, charge_delta);
    }
    return EC_OK;
}

ErrorCode MetaLocalBackend::UpdateInPlace(std::string_view key_sv,
                                          const CacheLocationMap &locations,
                                          const PropertyMap &properties) {
    Cache::Handle *handle = cache_->Lookup(key_sv);
    if (!handle) {
        return EC_NOENT;
    }
    const ErrorCode ec = UpdateHandleInPlace(handle, locations, properties);
    cache_->Release(handle);
    return ec;
}

ErrorCode MetaLocalBackend::CreateAndInsertSingleLocation(std::string_view key_sv,
                                                          const LocationId &location_id,
                                                          CacheLocationConstPtr location) {
    MetaMemCacheItem *item = MetaMemCacheItem::CreateSingleLocation(location_id, std::move(location));
    item->TouchAccessTime();
    const size_t charge = item->Size();
    Cache::Handle *handle = nullptr;
    const ErrorCode ec = cache_->Insert(key_sv, item, cache_item_helper_.get(), charge, &handle);
    if (ec != EC_OK) {
        MetaMemCacheItem::Deleter(item, nullptr);
    } else if (handle) {
        cache_->Release(handle);
    }
    return ec;
}

ErrorCode MetaLocalBackend::UpsertSingleLocationForOneKey(KeyType key,
                                                          const LocationId &location_id,
                                                          const CacheLocationConstPtr &location) {
    const std::string_view key_view = KeyToView(key);
    Cache::Handle *handle = cache_->Lookup(key_view);
    if (handle) {
        const ErrorCode ec = UpdateHandleInPlaceSingleLocation(handle, location_id, location);
        cache_->Release(handle);
        return ec;
    }
    return CreateAndInsertSingleLocation(key_view, location_id, location);
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
                charge_delta -=
                    static_cast<ssize_t>(MetaMemCacheItem::EstimateLocationEntryUsage(it->first, it->second));
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
    bool has_duplicate_keys = false;
    bool keys_are_sorted = true;
    for (size_t i = 1; i < keys.size(); ++i) {
        has_duplicate_keys = has_duplicate_keys || keys[i] == keys[i - 1];
        keys_are_sorted = keys_are_sorted && keys[i - 1] <= keys[i];
    }
    if (!has_duplicate_keys && !keys_are_sorted) {
        std::unordered_set<KeyType> seen_keys;
        seen_keys.reserve(keys.size());
        for (const KeyType key : keys) {
            if (!seen_keys.insert(key).second) {
                has_duplicate_keys = true;
                break;
            }
        }
    }
    if (has_duplicate_keys) {
        // Preserve the historical request-order merge semantics for the
        // general backend API. ReportEvent supplies sorted unique keys and
        // therefore stays on the batched-LRU fast path below.
        for (size_t i = 0; i < keys.size(); ++i) {
            results[i] = UpsertForOneKey(keys[i], locations[i], properties[i]);
        }
        return results;
    }

    std::vector<std::string_view> key_views(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        key_views[i] = KeyToView(keys[i]);
    }
    std::vector<Cache::Handle *> handles(keys.size(), nullptr);
    cache_->LookupBatch(key_views.data(), key_views.size(), handles.data());
    const size_t missing_count = static_cast<size_t>(std::count(handles.begin(), handles.end(), nullptr));
    if (missing_count > 0 && missing_count < handles.size()) {
        // Updating every hit before inserting every miss reorders a mixed
        // request. Besides partial-field merge semantics, order is observable
        // under strict capacity because in-place charge growth is admitted
        // differently from a new insertion. Preserve the original per-key
        // behavior for this uncommon shape. Existing handles were already
        // acquired in one batch, but are updated/released at their original
        // position so a preceding insert still observes the preceding cache
        // charge. All-hit updates and all-miss creates retain the fully
        // batched-LRU fast paths below.
        for (size_t i = 0; i < keys.size(); ++i) {
            if (handles[i]) {
                results[i] = UpdateHandleInPlace(handles[i], locations[i], properties[i]);
                cache_->Release(handles[i]);
            } else {
                results[i] = CreateAndInsert(key_views[i], locations[i], properties[i]);
            }
        }
        return results;
    }

    if (missing_count == 0) {
        for (size_t i = 0; i < keys.size(); ++i) {
            results[i] = UpdateHandleInPlace(handles[i], locations[i], properties[i]);
        }
        cache_->ReleaseBatch(handles.data(), handles.size());
        return results;
    }

    // The mixed shape returned above, so every handle is null here. Avoid a
    // pointless ReleaseBatch shard-group allocation and retain ordered insert
    // semantics for the all-new-key path.
    assert(missing_count == handles.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        results[i] = CreateAndInsert(key_views[i], locations[i], properties[i]);
    }
    return results;
}

std::vector<ErrorCode> MetaLocalBackend::UpsertSingleLocations(RequestContext * /*request_context*/,
                                                               const KeyTypeVec &keys,
                                                               const LocationIdRefVector &location_ids,
                                                               const CacheLocationVector &locations) noexcept {
    SingleLocationRmwScratch scratch;
    PrepareSingleLocationRmwScratch(keys.size(), scratch);
    std::vector<ErrorCode> results;
    results.reserve(keys.size());
    UpsertSingleLocationsInto(nullptr, keys, location_ids, locations, results, scratch);
    return results;
}

void MetaLocalBackend::PrepareSingleLocationRmwScratch(size_t max_count, SingleLocationRmwScratch &scratch) noexcept {
    scratch.ReleaseRetainedHandles();
    scratch.retired_locations.clear();
    scratch.key_views.reserve(max_count);
    scratch.handles.reserve(max_count);
    scratch.retired_locations.reserve(max_count);
    cache_->PrepareBatchOperationScratch(max_count, &scratch.cache_batch);
}

void MetaLocalBackend::UpsertSingleLocationsInto(RequestContext * /*request_context*/,
                                                 const KeyTypeVec &keys,
                                                 const LocationIdRefVector &location_ids,
                                                 const CacheLocationVector &locations,
                                                 std::vector<ErrorCode> &results,
                                                 SingleLocationRmwScratch &scratch) noexcept {
    if (keys.size() != location_ids.size() || keys.size() != locations.size()) {
        results.assign(keys.size(), EC_BADARGS);
        return;
    }
    if (keys.empty()) {
        results.clear();
        return;
    }
    results.assign(keys.size(), EC_OK);
    for (size_t i = 0; i < keys.size(); ++i) {
        if (location_ids[i] == nullptr || !locations[i] || locations[i]->id() != *location_ids[i]) {
            // The method is a batch write: do not report EC_OK for valid-looking
            // entries when a malformed sibling prevents the batch from being
            // applied. This also matches the generic backend adapter.
            results.assign(keys.size(), EC_BADARGS);
            return;
        }
    }

    bool has_duplicate_keys = false;
    bool keys_are_sorted = true;
    for (size_t i = 1; i < keys.size(); ++i) {
        has_duplicate_keys = has_duplicate_keys || keys[i] == keys[i - 1];
        keys_are_sorted = keys_are_sorted && keys[i - 1] <= keys[i];
    }
    if (!has_duplicate_keys && !keys_are_sorted) {
        std::unordered_set<KeyType> seen_keys;
        seen_keys.reserve(keys.size());
        for (const KeyType key : keys) {
            if (!seen_keys.insert(key).second) {
                has_duplicate_keys = true;
                break;
            }
        }
    }
    if (has_duplicate_keys) {
        for (size_t i = 0; i < keys.size(); ++i) {
            results[i] = UpsertSingleLocationForOneKey(keys[i], *location_ids[i], locations[i]);
        }
        return;
    }

    scratch.key_views.resize(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        scratch.key_views[i] = KeyToView(keys[i]);
    }
    scratch.handles.assign(keys.size(), nullptr);
    cache_->LookupBatchWithScratch(
        scratch.key_views.data(), scratch.key_views.size(), scratch.handles.data(), &scratch.cache_batch);
    const size_t missing_count =
        static_cast<size_t>(std::count(scratch.handles.begin(), scratch.handles.end(), nullptr));
    if (missing_count == 0) {
        for (size_t i = 0; i < keys.size(); ++i) {
            results[i] = UpdateHandleInPlaceSingleLocation(scratch.handles[i], *location_ids[i], locations[i]);
        }
        cache_->ReleaseBatchWithScratch(scratch.handles.data(), scratch.handles.size(), &scratch.cache_batch);
        return;
    }
    if (missing_count == scratch.handles.size()) {
        for (size_t i = 0; i < keys.size(); ++i) {
            results[i] = CreateAndInsertSingleLocation(scratch.key_views[i], *location_ids[i], locations[i]);
        }
        return;
    }

    // Preserve request order for the mixed hit/miss shape. Capacity admission
    // can observe whether an earlier item was inserted or grew in place.
    for (size_t i = 0; i < keys.size(); ++i) {
        if (scratch.handles[i]) {
            results[i] = UpdateHandleInPlaceSingleLocation(scratch.handles[i], *location_ids[i], locations[i]);
            cache_->Release(scratch.handles[i]);
        } else {
            results[i] = CreateAndInsertSingleLocation(scratch.key_views[i], *location_ids[i], locations[i]);
        }
    }
}

void MetaLocalBackend::UpsertSingleLocationsUsingRetainedHandlesInto(RequestContext * /*request_context*/,
                                                                     const KeyTypeVec &keys,
                                                                     const LocationIdRefVector &location_ids,
                                                                     CacheLocationVector &locations,
                                                                     const std::vector<size_t> &read_indices,
                                                                     std::vector<ErrorCode> &results,
                                                                     SingleLocationRmwScratch &scratch) noexcept {
    auto fail_and_release = [&results, &scratch, &keys](ErrorCode ec) {
        results.assign(keys.size(), ec);
        scratch.ReleaseRetainedHandles();
    };
    if (keys.size() != location_ids.size() || keys.size() != locations.size() || keys.size() != read_indices.size()) {
        fail_and_release(EC_BADARGS);
        return;
    }
    if (keys.empty()) {
        results.clear();
        scratch.ReleaseRetainedHandles();
        return;
    }
    if (scratch.retained_handle_owner != cache_.get()) {
        fail_and_release(EC_BADARGS);
        return;
    }

    results.assign(keys.size(), EC_OK);
    size_t previous_read_index = 0;
    for (size_t i = 0; i < keys.size(); ++i) {
        const size_t read_index = read_indices[i];
        if (read_index >= scratch.handles.size() || read_index >= scratch.key_views.size() ||
            (i > 0 && read_index <= previous_read_index) || location_ids[i] == nullptr || !locations[i] ||
            locations[i]->id() != *location_ids[i] || scratch.key_views[read_index] != KeyToView(keys[i])) {
            fail_and_release(EC_BADARGS);
            return;
        }
        previous_read_index = read_index;
    }

    // A modifier may skip some read hits. Release those before inserting new
    // keys so pinned, unrelated entries cannot change strict-capacity behavior.
    size_t selected_position = 0;
    for (size_t read_index = 0; read_index < scratch.handles.size(); ++read_index) {
        if (selected_position < read_indices.size() && read_indices[selected_position] == read_index) {
            ++selected_position;
            continue;
        }
        if (scratch.handles[read_index]) {
            cache_->Release(scratch.handles[read_index]);
            scratch.handles[read_index] = nullptr;
        }
    }

    for (size_t i = 0; i < keys.size(); ++i) {
        Cache::Handle *handle = scratch.handles[read_indices[i]];
        if (handle) {
            results[i] = UpdateHandleInPlaceSingleLocation(
                handle, *location_ids[i], std::move(locations[i]), &scratch.retired_locations);
        } else {
            results[i] = CreateAndInsertSingleLocation(
                scratch.key_views[read_indices[i]], *location_ids[i], std::move(locations[i]));
        }
    }
    scratch.ReleaseRetainedHandles();
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

ErrorCode MetaLocalBackend::GetForOneKeyForMaintenance(KeyType key,
                                                       CacheLocationMap *out_location_map,
                                                       std::vector<LocationId> *out_location_ids) {
    ErrorCode result = EC_OK;
    const bool found = cache_->ApplyToEntryNoTouch(
        KeyToView(key),
        [&](Cache::ObjectPtr value, size_t /*charge*/, const Cache::CacheItemHelper * /*helper*/) -> ssize_t {
            if (value == nullptr) {
                result = EC_ERROR;
                return 0;
            }
            const auto *item = static_cast<const MetaMemCacheItem *>(value);
            std::shared_lock lock(item->GetMutex());
            const auto &locations = item->GetLocations();
            if (out_location_map) {
                *out_location_map = locations;
            }
            if (out_location_ids) {
                out_location_ids->reserve(locations.size());
                for (const auto &[location_id, _] : locations) {
                    out_location_ids->push_back(location_id);
                }
            }
            return 0;
        });
    return found ? result : EC_NOENT;
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

std::vector<ErrorCode> MetaLocalBackend::GetLocationValues(RequestContext * /*request_context*/,
                                                           const KeyTypeVec &keys,
                                                           LocationsPerKey &out_locations) noexcept {
    std::vector<ErrorCode> results(keys.size(), EC_OK);
    out_locations.clear();
    out_locations.resize(keys.size());
    std::vector<int64_t> revisit_intervals;
    if (revisit_histogram_) {
        revisit_intervals.reserve(keys.size());
    }
    for (size_t i = 0; i < keys.size(); ++i) {
        std::string_view key_sv = KeyToView(keys[i]);
        Cache::Handle *handle = cache_->Lookup(key_sv);
        if (!handle) {
            results[i] = EC_NOENT;
            continue;
        }
        auto *item = static_cast<MetaMemCacheItem *>(cache_->Value(handle));
        const int64_t stored_time = item->GetLastAccessTime();
        if (revisit_histogram_ && stored_time > 0) {
            revisit_intervals.push_back(TimestampUtil::GetCurrentTimeUs() - stored_time);
        }
        item->TouchAccessTime();
        {
            std::shared_lock lock(item->GetMutex());
            const auto &locations = item->GetLocations();
            auto &values = out_locations[i];
            values.reserve(locations.size());
            for (const auto &[location_id, location] : locations) {
                (void)location_id;
                values.push_back(location);
            }
        }
        cache_->Release(handle);
    }
    if (revisit_histogram_) {
        revisit_histogram_->ObserveBatch(revisit_intervals);
    }
    return results;
}

std::vector<ErrorCode> MetaLocalBackend::GetLocationValuesCompact(RequestContext * /*request_context*/,
                                                                  const KeyType *keys,
                                                                  size_t key_count,
                                                                  CompactLocationsPerKey &out_locations) noexcept {
    std::vector<ErrorCode> results(key_count, EC_OK);
    out_locations.Clear(key_count, key_count);
    if (key_count != 0 && keys == nullptr) {
        results.assign(key_count, EC_BADARGS);
        for (size_t i = 0; i < key_count; ++i) {
            out_locations.FinishKey();
        }
        return results;
    }

    CompactReadScratchLease scratch_lease(key_count);
    auto &scratch = scratch_lease.get();
    auto &revisit_intervals = scratch.revisit_intervals;
    revisit_intervals.clear();
    if (revisit_histogram_) {
        revisit_intervals.reserve(key_count);
    }
    auto &key_views = scratch.key_views;
    key_views.resize(key_count);
    for (size_t i = 0; i < key_count; ++i) {
        key_views[i] = KeyToView(keys[i]);
    }
    auto &handles = scratch.handles;
    handles.assign(key_count, nullptr);
    cache_->PrepareBatchOperationScratch(key_count, &scratch.cache_batch);
    cache_->LookupBatchWithScratch(key_views.data(), key_count, handles.data(), &scratch.cache_batch);

    const int64_t access_time_us = TimestampUtil::GetCurrentTimeUs();
    for (size_t i = 0; i < key_count; ++i) {
        Cache::Handle *handle = handles[i];
        if (!handle) {
            results[i] = EC_NOENT;
            out_locations.FinishKey();
            continue;
        }

        auto *item = static_cast<MetaMemCacheItem *>(cache_->Value(handle));
        if (revisit_histogram_) {
            const int64_t stored_time = item->GetLastAccessTime();
            if (stored_time > 0 && stored_time <= access_time_us) {
                revisit_intervals.push_back(access_time_us - stored_time);
            }
        }
        item->TouchAccessTime(access_time_us);
        {
            std::shared_lock lock(item->GetMutex());
            for (const auto &[location_id, location] : item->GetLocations()) {
                (void)location_id;
                out_locations.values.push_back(location);
            }
        }
        out_locations.FinishKey();
    }
    cache_->ReleaseBatchWithScratch(handles.data(), handles.size(), &scratch.cache_batch);
    if (revisit_histogram_) {
        revisit_histogram_->ObserveBatch(revisit_intervals);
    }
    return results;
}

std::vector<std::vector<ErrorCode>> MetaLocalBackend::GetLocations(RequestContext *request_context,
                                                                   const KeyTypeVec &keys,
                                                                   const LocationIdsPerKey &location_ids,
                                                                   LocationsPerKey &out_locations) noexcept {
    std::vector<ErrorCode> ignored_key_error_codes;
    return GetLocationsWithKeyStatus(request_context, keys, location_ids, out_locations, ignored_key_error_codes);
}

std::vector<std::vector<ErrorCode>>
MetaLocalBackend::GetLocationsWithKeyStatus(RequestContext * /*request_context*/,
                                            const KeyTypeVec &keys,
                                            const LocationIdsPerKey &location_ids,
                                            LocationsPerKey &out_locations,
                                            std::vector<ErrorCode> &out_key_error_codes) noexcept {
    assert(keys.size() == location_ids.size());
    std::vector<std::vector<ErrorCode>> results(keys.size());
    out_key_error_codes.assign(keys.size(), EC_OK);
    out_locations.assign(keys.size(), CacheLocationVector{});

    // Targeted RMW commonly reads thousands of one-location keys at once.
    // Group the LRU lookups/releases by cache shard instead of taking the same
    // shard mutex once per key. Handles remain pinned while the immutable
    // CacheLocation pointers are projected below, matching the compact query
    // path's lifetime contract.
    std::vector<std::string_view> key_views(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        key_views[i] = KeyToView(keys[i]);
    }
    std::vector<Cache::Handle *> handles(keys.size(), nullptr);
    cache_->LookupBatch(key_views.data(), key_views.size(), handles.data());
    const int64_t access_time_us = TimestampUtil::GetCurrentTimeUs();
    for (size_t i = 0; i < keys.size(); ++i) {
        out_locations[i].resize(location_ids[i].size());

        Cache::Handle *handle = handles[i];
        if (!handle) {
            out_key_error_codes[i] = EC_NOENT;
            results[i].assign(location_ids[i].size(), EC_NOENT);
            continue;
        }
        auto *item = static_cast<MetaMemCacheItem *>(cache_->Value(handle));
        item->TouchAccessTime(access_time_us);
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
    }
    cache_->ReleaseBatch(handles.data(), handles.size());
    return results;
}

std::vector<ErrorCode>
MetaLocalBackend::GetSingleLocationsWithKeyStatus(RequestContext * /*request_context*/,
                                                  const KeyTypeVec &keys,
                                                  const LocationIdRefVector &location_ids,
                                                  CacheLocationVector &out_locations,
                                                  std::vector<ErrorCode> &out_key_error_codes) noexcept {
    SingleLocationRmwScratch scratch;
    PrepareSingleLocationRmwScratch(keys.size(), scratch);
    std::vector<ErrorCode> results;
    results.reserve(keys.size());
    GetSingleLocationsWithKeyStatusInto(
        nullptr, keys, location_ids, out_locations, out_key_error_codes, results, scratch);
    return results;
}

void MetaLocalBackend::GetSingleLocationsWithKeyStatusInto(RequestContext *request_context,
                                                           const KeyTypeVec &keys,
                                                           const LocationIdRefVector &location_ids,
                                                           CacheLocationVector &out_locations,
                                                           std::vector<ErrorCode> &out_key_error_codes,
                                                           std::vector<ErrorCode> &results,
                                                           SingleLocationRmwScratch &scratch,
                                                           bool retain_handles) noexcept {
    GetSingleLocationsWithKeyStatusIntoImpl(request_context,
                                            keys,
                                            location_ids,
                                            &out_locations,
                                            nullptr,
                                            out_key_error_codes,
                                            results,
                                            scratch,
                                            retain_handles);
}

void MetaLocalBackend::GetSingleLocationViewsWithKeyStatusInto(RequestContext *request_context,
                                                               const KeyTypeVec &keys,
                                                               const LocationIdRefVector &location_ids,
                                                               CacheLocationViewVector &out_locations,
                                                               std::vector<ErrorCode> &out_key_error_codes,
                                                               std::vector<ErrorCode> &results,
                                                               SingleLocationRmwScratch &scratch) noexcept {
    GetSingleLocationsWithKeyStatusIntoImpl(request_context,
                                            keys,
                                            location_ids,
                                            nullptr,
                                            &out_locations,
                                            out_key_error_codes,
                                            results,
                                            scratch,
                                            /*retain_handles=*/true);
}

void MetaLocalBackend::GetSingleLocationsWithKeyStatusIntoImpl(RequestContext * /*request_context*/,
                                                               const KeyTypeVec &keys,
                                                               const LocationIdRefVector &location_ids,
                                                               CacheLocationVector *out_owned_locations,
                                                               CacheLocationViewVector *out_borrowed_locations,
                                                               std::vector<ErrorCode> &out_key_error_codes,
                                                               std::vector<ErrorCode> &results,
                                                               SingleLocationRmwScratch &scratch,
                                                               bool retain_handles) noexcept {
    scratch.ReleaseRetainedHandles();
    assert((out_owned_locations == nullptr) != (out_borrowed_locations == nullptr));
    if (out_owned_locations) {
        out_owned_locations->assign(keys.size(), CacheLocationConstPtr{});
    }
    if (out_borrowed_locations) {
        out_borrowed_locations->assign(keys.size(), nullptr);
    }
    out_key_error_codes.assign(keys.size(), EC_OK);
    if (keys.size() != location_ids.size()) {
        out_key_error_codes.assign(keys.size(), EC_BADARGS);
        results.assign(keys.size(), EC_BADARGS);
        return;
    }
    if (keys.empty()) {
        results.clear();
        return;
    }
    results.assign(keys.size(), EC_NOENT);
    scratch.key_views.resize(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        if (location_ids[i] == nullptr) {
            out_key_error_codes.assign(keys.size(), EC_BADARGS);
            results.assign(keys.size(), EC_BADARGS);
            return;
        }
        scratch.key_views[i] = KeyToView(keys[i]);
    }
    scratch.handles.assign(keys.size(), nullptr);
    cache_->LookupBatchWithScratch(
        scratch.key_views.data(), scratch.key_views.size(), scratch.handles.data(), &scratch.cache_batch);
    const int64_t access_time_us = TimestampUtil::GetCurrentTimeUs();
    for (size_t i = 0; i < keys.size(); ++i) {
        Cache::Handle *handle = scratch.handles[i];
        if (!handle) {
            out_key_error_codes[i] = EC_NOENT;
            continue;
        }
        auto *item = static_cast<MetaMemCacheItem *>(cache_->Value(handle));
        item->TouchAccessTime(access_time_us);
        std::shared_lock lock(item->GetMutex());
        const auto &locations = item->GetLocations();
        auto it = locations.end();
        if (locations.size() == 1) {
            auto only = locations.begin();
            if (only->first == *location_ids[i]) {
                it = only;
            }
        } else if (!locations.empty()) {
            it = locations.find(*location_ids[i]);
        }
        if (it != locations.end()) {
            if (out_borrowed_locations) {
                (*out_borrowed_locations)[i] = it->second.get();
            } else {
                (*out_owned_locations)[i] = it->second;
            }
            results[i] = EC_OK;
        }
    }
    if (retain_handles) {
        scratch.retained_handle_owner = cache_.get();
    } else {
        cache_->ReleaseBatchWithScratch(scratch.handles.data(), scratch.handles.size(), &scratch.cache_batch);
    }
}

std::vector<std::vector<ErrorCode>>
MetaLocalBackend::GetLocationsForMaintenance(RequestContext * /*request_context*/,
                                             const KeyTypeVec &keys,
                                             const LocationIdsPerKey &location_ids,
                                             LocationsPerKey &out_locations) noexcept {
    if (keys.size() != location_ids.size()) {
        out_locations.clear();
        return {};
    }
    std::vector<std::vector<ErrorCode>> results(keys.size());
    out_locations.resize(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        results[i].assign(location_ids[i].size(), EC_NOENT);
        out_locations[i].resize(location_ids[i].size());
        const bool found = cache_->ApplyToEntryNoTouch(
            KeyToView(keys[i]),
            [&](Cache::ObjectPtr value, size_t /*charge*/, const Cache::CacheItemHelper * /*helper*/) -> ssize_t {
                if (value == nullptr) {
                    std::fill(results[i].begin(), results[i].end(), EC_ERROR);
                    return 0;
                }
                const auto *item = static_cast<const MetaMemCacheItem *>(value);
                std::shared_lock lock(item->GetMutex());
                const auto &locations = item->GetLocations();
                for (size_t j = 0; j < location_ids[i].size(); ++j) {
                    const auto it = locations.find(location_ids[i][j]);
                    if (it != locations.end()) {
                        results[i][j] = EC_OK;
                        out_locations[i][j] = it->second;
                    }
                }
                return 0;
            });
        if (!found) {
            std::fill(results[i].begin(), results[i].end(), EC_NOENT);
        }
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

std::vector<ErrorCode> MetaLocalBackend::GetLocationIdsForMaintenance(RequestContext * /*request_context*/,
                                                                      const KeyTypeVec &keys,
                                                                      LocationIdsPerKey &out_location_ids) noexcept {
    std::vector<ErrorCode> results(keys.size(), EC_OK);
    out_location_ids.resize(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        results[i] = GetForOneKeyForMaintenance(keys[i], nullptr, &out_location_ids[i]);
    }
    return results;
}

std::vector<ErrorCode> MetaLocalBackend::DeleteLocationsForMaintenance(RequestContext * /*request_context*/,
                                                                       const KeyTypeVec &keys,
                                                                       const LocationIdsPerKey &location_ids) noexcept {
    return DeleteLocationsForMaintenance(nullptr, keys, location_ids, std::vector<ErrorCode>(keys.size(), EC_OK));
}

std::vector<ErrorCode>
MetaLocalBackend::DeleteLocationsForMaintenance(RequestContext * /*request_context*/,
                                                const KeyTypeVec &keys,
                                                const LocationIdsPerKey &location_ids,
                                                const std::vector<ErrorCode> &previous_error_codes) noexcept {
    if (keys.size() != location_ids.size() || keys.size() != previous_error_codes.size()) {
        return {};
    }
    std::vector<ErrorCode> results(keys.size(), EC_OK);
    for (size_t i = 0; i < keys.size(); ++i) {
        if (previous_error_codes[i] != EC_OK && previous_error_codes[i] != EC_NOENT) {
            results[i] = previous_error_codes[i];
            continue;
        }
        if (location_ids[i].empty()) {
            continue;
        }
        const bool found = cache_->ApplyToEntryNoTouch(
            KeyToView(keys[i]),
            [&](Cache::ObjectPtr value, size_t /*charge*/, const Cache::CacheItemHelper * /*helper*/) -> ssize_t {
                if (value == nullptr) {
                    results[i] = EC_ERROR;
                    return 0;
                }
                auto *item = static_cast<MetaMemCacheItem *>(value);
                ssize_t charge_delta = 0;
                std::unique_lock lock(item->GetMutex());
                auto &locations = item->GetMutableLocations();
                for (const auto &location_id : location_ids[i]) {
                    const auto it = locations.find(location_id);
                    if (it == locations.end()) {
                        continue;
                    }
                    charge_delta -=
                        static_cast<ssize_t>(MetaMemCacheItem::EstimateLocationEntryUsage(it->first, it->second));
                    locations.erase(it);
                }
                return charge_delta;
            });
        if (!found) {
            results[i] = EC_NOENT;
        }
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

ErrorCode MetaLocalBackend::ScanLocationsForMaintenance(RequestContext * /*request_context*/,
                                                        const std::string &cursor,
                                                        const int64_t limit,
                                                        MaintenanceScanBatch &out) noexcept {
    out.Clear();

    int64_t start_shard = 0;
    if (!StringUtil::StrToInt64(cursor.c_str(), start_shard) || start_shard < 0 || start_shard > shard_mask_) {
        return EC_BADARGS;
    }

    const uint32_t num_shards = shard_mask_ + 1;
    int64_t collected = 0;
    for (uint32_t shard_id = static_cast<uint32_t>(start_shard); shard_id < num_shards; ++shard_id) {
        std::vector<KeyType> shard_keys;
        cache_->ApplyToSingleShard(shard_id,
                                   [&](const std::string_view &key,
                                       Cache::ObjectPtr /*value*/,
                                       size_t /*charge*/,
                                       const Cache::CacheItemHelper * /*helper*/) {
                                       if (key.size() != sizeof(KeyType)) {
                                           return;
                                       }
                                       shard_keys.push_back(ViewToKey(key));
                                   });

        // ApplyToSingleShard owns the LRU shard mutex. Copy each item only
        // after that iteration lock has been released; ApplyToEntryNoTouch
        // then pins the exact shard operation without promoting the entry.
        for (const KeyType key : shard_keys) {
            CacheLocationMap locations;
            const ErrorCode ec = GetForOneKeyForMaintenance(key, &locations, nullptr);
            out.keys.push_back(key);
            out.locations.emplace_back(std::move(locations));
            out.location_results.push_back(ec);
            ++collected;
        }

        if (collected >= limit) {
            const uint32_t next_shard = shard_id + 1;
            out.next_cursor = (next_shard >= num_shards) ? SCAN_BASE_CURSOR : std::to_string(next_shard);
            return EC_OK;
        }
    }

    out.next_cursor = SCAN_BASE_CURSOR;
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

ErrorCode MetaLocalBackend::SampleReclaimCandidates(RequestContext * /*request_context*/,
                                                    const int64_t count,
                                                    ReclaimCandidateVector &out_candidates) noexcept {
    out_candidates.clear();
    if (!cache_) {
        KVCM_LOG_ERROR("local backend not inited");
        return EC_ERROR;
    }
    if (count <= 0) {
        return EC_OK;
    }

    const size_t num_shards = shard_mask_ + 1;
    size_t num_rounds = std::min(sample_times_, num_shards);
    num_rounds = std::min(num_rounds, static_cast<size_t>(count));
    const int64_t per_round_count = (count + static_cast<int64_t>(num_rounds) - 1) / static_cast<int64_t>(num_rounds);
    std::vector<std::pair<int64_t, uint32_t>> shard_times;
    shard_times.reserve(num_shards);
    for (uint32_t shard_id = 0; shard_id < num_shards; ++shard_id) {
        const int64_t access_time = shard_oldest_access_time_[shard_id].load(std::memory_order_relaxed);
        if (access_time < INT64_MAX) {
            shard_times.emplace_back(access_time, shard_id);
        }
    }
    if (shard_times.empty()) {
        return EC_OK;
    }

    const size_t select_count = std::min(num_rounds, shard_times.size());
    std::sort(shard_times.begin(), shard_times.end());
    const size_t start =
        reclaim_sample_shard_cursor_.fetch_add(select_count, std::memory_order_relaxed) % shard_times.size();
    int64_t remaining = count;
    for (size_t i = 0; i < select_count && remaining > 0; ++i) {
        const size_t batch = static_cast<size_t>(std::min(per_round_count, remaining));
        const size_t shard_index = (start + i) % shard_times.size();
        const size_t collected =
            CollectNextReclaimCandidatesFromShard(shard_times[shard_index].second, batch, out_candidates);
        remaining -= static_cast<int64_t>(collected);
    }
    return EC_OK;
}

std::vector<ErrorCode>
MetaLocalBackend::GetLastAccessTimesForMaintenance(RequestContext * /*request_context*/,
                                                   const KeyTypeVec &keys,
                                                   std::vector<int64_t> &out_last_access_times) noexcept {
    out_last_access_times.assign(keys.size(), 0);
    std::vector<ErrorCode> results(keys.size(), EC_NOENT);
    if (!cache_) {
        std::fill(results.begin(), results.end(), EC_ERROR);
        return results;
    }

    for (size_t i = 0; i < keys.size(); ++i) {
        const bool found = cache_->ApplyToEntryNoTouch(
            KeyToView(keys[i]),
            [&results, &out_last_access_times, i](
                Cache::ObjectPtr value, size_t /*charge*/, const Cache::CacheItemHelper * /*helper*/) -> ssize_t {
                if (value == nullptr) {
                    results[i] = EC_ERROR;
                    return 0;
                }
                const auto *item = static_cast<const MetaMemCacheItem *>(value);
                out_last_access_times[i] = item->GetLastAccessTime();
                results[i] = EC_OK;
                return 0;
            });
        if (!found) {
            results[i] = EC_NOENT;
        }
    }
    return results;
}

// return OK to avoid error in MetaIndexer::PersistMetaData()
ErrorCode MetaLocalBackend::PutMetaData(const FieldMap & /*field_maps*/) noexcept { return EC_OK; }

ErrorCode MetaLocalBackend::GetMetaData(FieldMap & /*field_maps*/) noexcept { return EC_NOENT; }

size_t MetaLocalBackend::GetMemUsage() const noexcept { return cache_->GetUsage(); }

bool MetaLocalBackend::GetCacheHashSeed(uint32_t &out_hash_seed) const noexcept {
    if (!cache_) {
        return false;
    }
    out_hash_seed = cache_->GetHashSeed();
    return true;
}

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

size_t MetaLocalBackend::CollectNextReclaimCandidatesFromShard(const uint32_t shard_id,
                                                               const size_t count,
                                                               ReclaimCandidateVector &out_candidates) {
    const size_t initial_size = out_candidates.size();
    cache_->ApplyToNextOldestEntriesInShard(shard_id,
                                            count,
                                            [&out_candidates](const std::string_view &key,
                                                              Cache::ObjectPtr value,
                                                              size_t /*charge*/,
                                                              const Cache::CacheItemHelper * /*helper*/) {
                                                if (key.size() != sizeof(KeyType) || value == nullptr) {
                                                    return;
                                                }
                                                const auto *item = static_cast<const MetaMemCacheItem *>(value);
                                                out_candidates.push_back({ViewToKey(key), item->GetLastAccessTime()});
                                            });
    return out_candidates.size() - initial_size;
}

} // namespace kv_cache_manager
