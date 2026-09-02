//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#include "kv_cache_manager/common/cache/lru_cache.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>

// #include "cache/secondary_cache_adapter.h"

namespace kv_cache_manager {
namespace lru_cache {

namespace {

bool UseSparseShardGrouping(size_t item_count, size_t shard_count) { return item_count <= shard_count / 4; }

template <typename ShardForIndex>
void BuildSparseShardRuns(size_t item_count, ShardForIndex shard_for_index, Cache::BatchOperationScratch *scratch) {
    auto &ordered_indices = scratch->ordered_indices;
    std::sort(ordered_indices.begin(), ordered_indices.begin() + item_count, [&](size_t lhs, size_t rhs) {
        const size_t lhs_shard = shard_for_index(lhs);
        const size_t rhs_shard = shard_for_index(rhs);
        return lhs_shard != rhs_shard ? lhs_shard < rhs_shard : lhs < rhs;
    });

    auto &shard_offsets = scratch->shard_offsets;
    shard_offsets.clear();
    size_t previous_shard = 0;
    for (size_t i = 0; i < item_count; ++i) {
        const size_t shard = shard_for_index(ordered_indices[i]);
        if (i == 0 || shard != previous_shard) {
            shard_offsets.push_back(i);
            previous_shard = shard;
        }
    }
    shard_offsets.push_back(item_count);
}

} // namespace

LRUHandleTable::LRUHandleTable(int max_upper_hash_bits, MemoryAllocator *allocator)
    : length_bits_(/* historical starting size*/ 4)
    , list_(new LRUHandle *[size_t{1} << length_bits_] {})
    , elems_(0)
    , max_length_bits_(max_upper_hash_bits)
    , allocator_(allocator) {}

LRUHandleTable::~LRUHandleTable() {
    auto alloc = allocator_;
    ApplyToEntriesRange(
        [alloc](LRUHandle *h) {
            if (!h->HasRefs()) {
                h->Free(alloc);
            }
        },
        0,
        size_t{1} << length_bits_);
}

LRUHandle *LRUHandleTable::Lookup(const std::string_view &key, uint32_t hash) { return *FindPointer(key, hash); }

LRUHandle *LRUHandleTable::Insert(LRUHandle *h) {
    LRUHandle **ptr = FindPointer(h->key(), h->hash);
    LRUHandle *old = *ptr;
    h->next_hash = (old == nullptr ? nullptr : old->next_hash);
    *ptr = h;
    if (old == nullptr) {
        ++elems_;
        if ((elems_ >> length_bits_) > 0) { // elems_ >= length
            // Since each cache entry is fairly large, we aim for a small
            // average linked list length (<= 1).
            Resize();
        }
    }
    return old;
}

LRUHandle *LRUHandleTable::Remove(const std::string_view &key, uint32_t hash) {
    LRUHandle **ptr = FindPointer(key, hash);
    LRUHandle *result = *ptr;
    if (result != nullptr) {
        *ptr = result->next_hash;
        --elems_;
    }
    return result;
}

LRUHandle **LRUHandleTable::FindPointer(const std::string_view &key, uint32_t hash) {
    LRUHandle **ptr = &list_[hash >> (32 - length_bits_)];
    while (*ptr != nullptr && ((*ptr)->hash != hash || key != (*ptr)->key())) {
        ptr = &(*ptr)->next_hash;
    }
    return ptr;
}

void LRUHandleTable::Resize() {
    if (length_bits_ >= max_length_bits_) {
        // Due to reaching limit of hash information, if we made the table bigger,
        // we would allocate more addresses but only the same number would be used.
        return;
    }
    if (length_bits_ >= 31) {
        // Avoid undefined behavior shifting uint32_t by 32.
        return;
    }

    uint32_t old_length = uint32_t{1} << length_bits_;
    int new_length_bits = length_bits_ + 1;
    std::unique_ptr<LRUHandle *[]> new_list { new LRUHandle *[size_t{1} << new_length_bits] {} };
    [[maybe_unused]] uint32_t count = 0;
    for (uint32_t i = 0; i < old_length; i++) {
        LRUHandle *h = list_[i];
        while (h != nullptr) {
            LRUHandle *next = h->next_hash;
            uint32_t hash = h->hash;
            LRUHandle **ptr = &new_list[hash >> (32 - new_length_bits)];
            h->next_hash = *ptr;
            *ptr = h;
            h = next;
            count++;
        }
    }
    assert(elems_ == count);
    list_ = std::move(new_list);
    length_bits_ = new_length_bits;
}

LRUCacheShard::LRUCacheShard(size_t capacity,
                             bool strict_capacity_limit,
                             bool no_evict_on_insert,
                             double high_pri_pool_ratio,
                             double low_pri_pool_ratio,
                             bool use_adaptive_mutex,
                             CacheMetadataChargePolicy metadata_charge_policy,
                             int max_upper_hash_bits,
                             MemoryAllocator *allocator,
                             const Cache::EvictionCallback *eviction_callback)
    : CacheShardBase(metadata_charge_policy)
    , capacity_(0)
    , high_pri_pool_usage_(0)
    , low_pri_pool_usage_(0)
    , strict_capacity_limit_(strict_capacity_limit)
    , no_evict_on_insert_(no_evict_on_insert)
    , high_pri_pool_ratio_(high_pri_pool_ratio)
    , high_pri_pool_capacity_(0)
    , low_pri_pool_ratio_(low_pri_pool_ratio)
    , low_pri_pool_capacity_(0)
    , table_(max_upper_hash_bits, allocator)
    , usage_(0)
    , lru_usage_(0)
    , eviction_callback_(*eviction_callback) {
    // Make empty circular linked list.
    lru_.next = &lru_;
    lru_.prev = &lru_;
    lru_low_pri_ = &lru_;
    lru_bottom_pri_ = &lru_;
    SetCapacity(capacity);
}

void LRUCacheShard::EraseUnRefEntries() {
    autovector<LRUHandle *> last_reference_list;
    {
        std::lock_guard<std::mutex> l(mutex_);
        while (lru_.next != &lru_) {
            LRUHandle *old = lru_.next;
            // LRU list contains only elements which can be evicted.
            assert(old->InCache() && !old->HasRefs());
            LRU_Remove(old);
            table_.Remove(old->key(), old->hash);
            old->SetInCache(false);
            assert(usage_ >= old->total_charge);
            usage_ -= old->total_charge;
            last_reference_list.push_back(old);
        }
    }

    for (auto entry : last_reference_list) {
        entry->Free(table_.GetAllocator());
    }
}

void LRUCacheShard::ApplyToSomeEntries(
    const std::function<
        void(const std::string_view &key, Cache::ObjectPtr value, size_t charge, const Cache::CacheItemHelper *helper)>
        &callback,
    size_t average_entries_per_lock,
    size_t *state) {
    // The state is essentially going to be the starting hash, which works
    // nicely even if we resize between calls because we use upper-most
    // hash bits for table indexes.
    std::lock_guard<std::mutex> l(mutex_);
    int length_bits = table_.GetLengthBits();
    size_t length = size_t{1} << length_bits;

    assert(average_entries_per_lock > 0);
    // Assuming we are called with same average_entries_per_lock repeatedly,
    // this simplifies some logic (index_end will not overflow).
    assert(average_entries_per_lock < length || *state == 0);

    size_t index_begin = *state >> (sizeof(size_t) * 8u - length_bits);
    size_t index_end = index_begin + average_entries_per_lock;
    if (index_end >= length) {
        // Going to end
        index_end = length;
        *state = SIZE_MAX;
    } else {
        *state = index_end << (sizeof(size_t) * 8u - length_bits);
    }

    table_.ApplyToEntriesRange(
        [callback, metadata_charge_policy = metadata_charge_policy_](LRUHandle *h) {
            callback(h->key(), h->value, h->GetCharge(metadata_charge_policy), h->helper);
        },
        index_begin,
        index_end);
}

void LRUCacheShard::TEST_GetLRUList(LRUHandle **lru, LRUHandle **lru_low_pri, LRUHandle **lru_bottom_pri) {
    std::lock_guard<std::mutex> l(mutex_);
    *lru = &lru_;
    *lru_low_pri = lru_low_pri_;
    *lru_bottom_pri = lru_bottom_pri_;
}

size_t LRUCacheShard::TEST_GetLRUSize() {
    std::lock_guard<std::mutex> l(mutex_);
    LRUHandle *lru_handle = lru_.next;
    size_t lru_size = 0;
    while (lru_handle != &lru_) {
        lru_size++;
        lru_handle = lru_handle->next;
    }
    return lru_size;
}

double LRUCacheShard::GetHighPriPoolRatio() {
    std::lock_guard<std::mutex> l(mutex_);
    return high_pri_pool_ratio_;
}

double LRUCacheShard::GetLowPriPoolRatio() {
    std::lock_guard<std::mutex> l(mutex_);
    return low_pri_pool_ratio_;
}

void LRUCacheShard::LRU_Remove(LRUHandle *e) {
    assert(e->next != nullptr);
    assert(e->prev != nullptr);
    bool was_tail = (lru_.next == e);
    if (lru_low_pri_ == e) {
        lru_low_pri_ = e->prev;
    }
    if (lru_bottom_pri_ == e) {
        lru_bottom_pri_ = e->prev;
    }
    e->next->prev = e->prev;
    e->prev->next = e->next;
    e->prev = e->next = nullptr;
    assert(lru_usage_ >= e->total_charge);
    lru_usage_ -= e->total_charge;
    assert(!e->InHighPriPool() || !e->InLowPriPool());
    if (e->InHighPriPool()) {
        assert(high_pri_pool_usage_ >= e->total_charge);
        high_pri_pool_usage_ -= e->total_charge;
    } else if (e->InLowPriPool()) {
        assert(low_pri_pool_usage_ >= e->total_charge);
        low_pri_pool_usage_ -= e->total_charge;
    }
    if (was_tail) {
        NotifyTailChange();
    }
}

void LRUCacheShard::LRU_Insert(LRUHandle *e) {
    assert(e->next == nullptr);
    assert(e->prev == nullptr);
    LRUHandle *old_tail = lru_.next;
    if (high_pri_pool_ratio_ > 0 && (e->IsHighPri() || e->HasHit())) {
        // Inset "e" to head of LRU list.
        e->next = &lru_;
        e->prev = lru_.prev;
        e->prev->next = e;
        e->next->prev = e;
        e->SetInHighPriPool(true);
        e->SetInLowPriPool(false);
        high_pri_pool_usage_ += e->total_charge;
        MaintainPoolSize();
    } else if (low_pri_pool_ratio_ > 0 && (e->IsHighPri() || e->IsLowPri() || e->HasHit())) {
        // Insert "e" to the head of low-pri pool.
        e->next = lru_low_pri_->next;
        e->prev = lru_low_pri_;
        e->prev->next = e;
        e->next->prev = e;
        e->SetInHighPriPool(false);
        e->SetInLowPriPool(true);
        low_pri_pool_usage_ += e->total_charge;
        lru_low_pri_ = e;
        MaintainPoolSize();
    } else {
        // Insert "e" to the head of bottom-pri pool.
        e->next = lru_bottom_pri_->next;
        e->prev = lru_bottom_pri_;
        e->prev->next = e;
        e->next->prev = e;
        e->SetInHighPriPool(false);
        e->SetInLowPriPool(false);
        // if the low-pri pool is empty, lru_low_pri_ also needs to be updated.
        if (lru_bottom_pri_ == lru_low_pri_) {
            lru_low_pri_ = e;
        }
        lru_bottom_pri_ = e;
    }
    lru_usage_ += e->total_charge;
    if (lru_.next != old_tail) {
        NotifyTailChange();
    }
}

void LRUCacheShard::MaintainPoolSize() {
    while (high_pri_pool_usage_ > high_pri_pool_capacity_) {
        // Overflow last entry in high-pri pool to low-pri pool.
        lru_low_pri_ = lru_low_pri_->next;
        assert(lru_low_pri_ != &lru_);
        assert(lru_low_pri_->InHighPriPool());
        lru_low_pri_->SetInHighPriPool(false);
        lru_low_pri_->SetInLowPriPool(true);
        assert(high_pri_pool_usage_ >= lru_low_pri_->total_charge);
        high_pri_pool_usage_ -= lru_low_pri_->total_charge;
        low_pri_pool_usage_ += lru_low_pri_->total_charge;
    }

    while (low_pri_pool_usage_ > low_pri_pool_capacity_) {
        // Overflow last entry in low-pri pool to bottom-pri pool.
        lru_bottom_pri_ = lru_bottom_pri_->next;
        assert(lru_bottom_pri_ != &lru_);
        assert(lru_bottom_pri_->InLowPriPool());
        lru_bottom_pri_->SetInHighPriPool(false);
        lru_bottom_pri_->SetInLowPriPool(false);
        assert(low_pri_pool_usage_ >= lru_bottom_pri_->total_charge);
        low_pri_pool_usage_ -= lru_bottom_pri_->total_charge;
    }
}

void LRUCacheShard::EvictFromLRU(size_t charge, autovector<LRUHandle *> *deleted) {
    while ((usage_ + charge) > capacity_ && lru_.next != &lru_) {
        LRUHandle *old = lru_.next;
        // LRU list contains only elements which can be evicted.
        assert(old->InCache() && !old->HasRefs());
        LRU_Remove(old);
        table_.Remove(old->key(), old->hash);
        old->SetInCache(false);
        assert(usage_ >= old->total_charge);
        usage_ -= old->total_charge;
        deleted->push_back(old);
    }
}

void LRUCacheShard::NotifyEvicted(const autovector<LRUHandle *> &evicted_handles) {
    MemoryAllocator *alloc = table_.GetAllocator();
    for (LRUHandle *entry : evicted_handles) {
        if (eviction_callback_ &&
            eviction_callback_(entry->key(), static_cast<Cache::Handle *>(entry), entry->HasHit())) {
            // Callback took ownership of obj; just free handle
            free(entry);
        } else {
            // Free the entries here outside of mutex for performance reasons.
            entry->Free(alloc);
        }
    }
}

void LRUCacheShard::SetCapacity(size_t capacity) {
    autovector<LRUHandle *> last_reference_list;
    {
        std::lock_guard<std::mutex> l(mutex_);
        capacity_ = capacity;
        high_pri_pool_capacity_ = capacity_ * high_pri_pool_ratio_;
        low_pri_pool_capacity_ = capacity_ * low_pri_pool_ratio_;
        EvictFromLRU(0, &last_reference_list);
    }

    NotifyEvicted(last_reference_list);
}

void LRUCacheShard::SetStrictCapacityLimit(bool strict_capacity_limit) {
    std::lock_guard<std::mutex> l(mutex_);
    strict_capacity_limit_ = strict_capacity_limit;
}

ErrorCode
LRUCacheShard::DoInsertItemUnsafe(LRUHandle *e, LRUHandle **handle, autovector<LRUHandle *> *last_reference_list) {
    // Free the space following strict LRU policy until enough space
    // is freed or the lru list is empty.
    if (!no_evict_on_insert_) {
        EvictFromLRU(e->total_charge, last_reference_list);
    }

    if ((usage_ + e->total_charge) > capacity_ && (strict_capacity_limit_ || handle == nullptr)) {
        e->SetInCache(false);
        if (handle == nullptr) {
            // Don't insert the entry but still return ok, as if the entry inserted
            // into cache and get evicted immediately.
            last_reference_list->push_back(e);
        } else {
            free(e);
            e = nullptr;
            *handle = nullptr;
            return EC_NOSPC;
        }
    } else {
        // Insert into the cache. Note that the cache might get larger than its
        // capacity if not enough space was freed up.
        LRUHandle *old = table_.Insert(e);
        usage_ += e->total_charge;
        if (old != nullptr) {
            assert(old->InCache());
            old->SetInCache(false);
            if (!old->HasRefs()) {
                // old is on LRU because it's in cache and its reference count is 0.
                LRU_Remove(old);
                assert(usage_ >= old->total_charge);
                usage_ -= old->total_charge;
                last_reference_list->push_back(old);
            }
        }
        if (handle == nullptr) {
            LRU_Insert(e);
        } else {
            // If caller already holds a ref, no need to take one here.
            if (!e->HasRefs()) {
                e->Ref();
            }
            *handle = e;
        }
    }
    return EC_OK;
}

ErrorCode LRUCacheShard::InsertItem(LRUHandle *e, LRUHandle **handle) {
    ErrorCode s = EC_OK;
    autovector<LRUHandle *> last_reference_list;

    {
        std::lock_guard<std::mutex> l(mutex_);
        s = DoInsertItemUnsafe(e, handle, &last_reference_list);
    }

    NotifyEvicted(last_reference_list);
    return s;
}

ErrorCode LRUCacheShard::InsertItemIfAbsent(LRUHandle *e, LRUHandle **handle) {
    ErrorCode s = EC_OK;
    autovector<LRUHandle *> last_reference_list;

    {
        std::lock_guard<std::mutex> l(mutex_);

        // Atomically check if key already exists
        LRUHandle *existing = table_.Lookup(e->key(), e->hash);
        if (existing != nullptr) {
            // Key already exists; free the new entry and return EC_EXIST
            e->SetInCache(false);
            last_reference_list.push_back(e);
            s = EC_EXIST;
        } else {
            // Key does not exist; proceed with insertion
            s = DoInsertItemUnsafe(e, handle, &last_reference_list);
        }
    }

    NotifyEvicted(last_reference_list);
    return s;
}

LRUHandle *LRUCacheShard::Lookup(const std::string_view &key,
                                 uint32_t hash,
                                 const Cache::CacheItemHelper * /*helper*/,
                                 Cache::CreateContext * /*create_context*/,
                                 Cache::Priority /*priority*/,
                                 Statistics * /*stats*/) {
    std::lock_guard<std::mutex> l(mutex_);
    LRUHandle *e = table_.Lookup(key, hash);
    if (e != nullptr) {
        assert(e->InCache());
        if (!e->HasRefs()) {
            // The entry is in LRU since it's in hash and has no external
            // references.
            LRU_Remove(e);
        }
        e->Ref();
        e->SetHit();
    }
    return e;
}

void LRUCacheShard::LookupBatch(const std::string_view *keys,
                                const uint32_t *hashes,
                                const size_t *ordered_indices,
                                size_t count,
                                Cache::Handle **out_handles) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (size_t i = 0; i < count; ++i) {
        const size_t index = ordered_indices[i];
        LRUHandle *entry = table_.Lookup(keys[index], hashes[index]);
        if (entry != nullptr) {
            assert(entry->InCache());
            if (!entry->HasRefs()) {
                LRU_Remove(entry);
            }
            entry->Ref();
            entry->SetHit();
        }
        out_handles[index] = static_cast<Cache::Handle *>(entry);
    }
}

bool LRUCacheShard::ApplyToEntryNoTouch(
    const std::string_view &key,
    uint32_t hash,
    const std::function<ssize_t(Cache::ObjectPtr obj, size_t charge, const Cache::CacheItemHelper *helper)> &callback) {
    std::lock_guard<std::mutex> l(mutex_);
    LRUHandle *e = table_.Lookup(key, hash);
    if (e == nullptr) {
        return false;
    }
    assert(e->InCache());
    const bool in_lru = !e->HasRefs();
    const ssize_t delta = callback(e->value, e->total_charge, e->helper);
    if (delta > 0) {
        const size_t increase = static_cast<size_t>(delta);
        e->total_charge += increase;
        usage_ += increase;
        if (in_lru) {
            lru_usage_ += increase;
            if (e->InHighPriPool()) {
                high_pri_pool_usage_ += increase;
            } else if (e->InLowPriPool()) {
                low_pri_pool_usage_ += increase;
            }
            MaintainPoolSize();
        }
    } else if (delta < 0) {
        size_t decrease = static_cast<size_t>(-delta);
        decrease = std::min(decrease, e->total_charge);
        decrease = std::min(decrease, usage_);
        e->total_charge -= decrease;
        usage_ -= decrease;
        if (in_lru) {
            assert(lru_usage_ >= decrease);
            lru_usage_ -= decrease;
            if (e->InHighPriPool()) {
                assert(high_pri_pool_usage_ >= decrease);
                high_pri_pool_usage_ -= decrease;
            } else if (e->InLowPriPool()) {
                assert(low_pri_pool_usage_ >= decrease);
                low_pri_pool_usage_ -= decrease;
            }
        }
    }
    return true;
}

bool LRUCacheShard::Ref(LRUHandle *e) {
    std::lock_guard<std::mutex> l(mutex_);
    // To create another reference - entry must be already externally referenced.
    assert(e->HasRefs());
    e->Ref();
    return true;
}

void LRUCacheShard::AdjustCharge(LRUHandle *e, ssize_t delta) {
    std::lock_guard<std::mutex> l(mutex_);
    assert(e->HasRefs());
    assert(e->InCache());
    if (delta > 0) {
        e->total_charge += static_cast<size_t>(delta);
        usage_ += static_cast<size_t>(delta);
    } else if (delta < 0) {
        size_t decrease = static_cast<size_t>(-delta);
        decrease = std::min(decrease, e->total_charge);
        decrease = std::min(decrease, usage_);
        e->total_charge -= decrease;
        usage_ -= decrease;
    }
}

void LRUCacheShard::SetHighPriorityPoolRatio(double high_pri_pool_ratio) {
    std::lock_guard<std::mutex> l(mutex_);
    high_pri_pool_ratio_ = high_pri_pool_ratio;
    high_pri_pool_capacity_ = capacity_ * high_pri_pool_ratio_;
    MaintainPoolSize();
}

void LRUCacheShard::SetLowPriorityPoolRatio(double low_pri_pool_ratio) {
    std::lock_guard<std::mutex> l(mutex_);
    low_pri_pool_ratio_ = low_pri_pool_ratio;
    low_pri_pool_capacity_ = capacity_ * low_pri_pool_ratio_;
    MaintainPoolSize();
}

bool LRUCacheShard::Release(LRUHandle *e, bool /*useful*/, bool erase_if_last_ref) {
    if (e == nullptr) {
        return false;
    }
    bool must_free;
    bool was_in_cache;
    {
        std::lock_guard<std::mutex> l(mutex_);
        must_free = e->Unref();
        was_in_cache = e->InCache();
        if (must_free && was_in_cache) {
            // The item is still in cache, and nobody else holds a reference to it.
            if (erase_if_last_ref || (!no_evict_on_insert_ && usage_ > capacity_)) {
                // When no_evict_on_insert_ is false: usage > capacity means the
                // LRU list must be empty (EvictFromLRU drained it on Insert).
                // When no_evict_on_insert_ is true: skip opportunistic eviction
                // to avoid silently dropping entries whose charge grew in-place
                // (via AdjustCharge).
                assert(lru_.next == &lru_ || erase_if_last_ref);
                // Take this opportunity and remove the item.
                table_.Remove(e->key(), e->hash);
                e->SetInCache(false);
            } else {
                // Put the item back on the LRU list, and don't free it.
                LRU_Insert(e);
                must_free = false;
            }
        }
        // If about to be freed, then decrement the cache usage.
        if (must_free) {
            assert(usage_ >= e->total_charge);
            usage_ -= e->total_charge;
        }
    }

    // Free the entry here outside of mutex for performance reasons.
    if (must_free) {
        // Only call eviction callback if we're sure no one requested erasure
        // FIXME: disabled because of test churn
        if (false && was_in_cache && !erase_if_last_ref && eviction_callback_ &&
            eviction_callback_(e->key(), static_cast<Cache::Handle *>(e), e->HasHit())) {
            // Callback took ownership of obj; just free handle
            free(e);
        } else {
            e->Free(table_.GetAllocator());
        }
    }
    return must_free;
}

void LRUCacheShard::ReleaseBatch(Cache::Handle *const *handles, const size_t *ordered_indices, size_t count) {
    autovector<LRUHandle *> free_handles;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (size_t i = 0; i < count; ++i) {
            auto *entry = static_cast<LRUHandle *>(handles[ordered_indices[i]]);
            if (entry == nullptr) {
                continue;
            }
            bool must_free = entry->Unref();
            const bool was_in_cache = entry->InCache();
            if (must_free && was_in_cache) {
                if (!no_evict_on_insert_ && usage_ > capacity_) {
                    assert(lru_.next == &lru_);
                    table_.Remove(entry->key(), entry->hash);
                    entry->SetInCache(false);
                } else {
                    LRU_Insert(entry);
                    must_free = false;
                }
            }
            if (must_free) {
                assert(usage_ >= entry->total_charge);
                usage_ -= entry->total_charge;
                free_handles.push_back(entry);
            }
        }
    }
    for (auto *entry : free_handles) {
        entry->Free(table_.GetAllocator());
    }
}

LRUHandle *LRUCacheShard::CreateHandle(const std::string_view &key,
                                       uint32_t hash,
                                       Cache::ObjectPtr value,
                                       const Cache::CacheItemHelper *helper,
                                       size_t charge) {
    assert(helper);
    // value == nullptr is reserved for indicating failure in SecondaryCache
    assert(!(helper->IsSecondaryCacheCompatible() && value == nullptr));

    // Allocate the memory here outside of the mutex.
    // If the cache is full, we'll have to release it.
    // It shouldn't happen very often though.
    LRUHandle *e = static_cast<LRUHandle *>(malloc(sizeof(LRUHandle) - 1 + key.size()));

    e->value = value;
    e->m_flags = 0;
    e->im_flags = 0;
    e->helper = helper;
    e->key_length = key.size();
    e->hash = hash;
    e->refs = 0;
    e->next = e->prev = nullptr;
    memcpy(e->key_data, key.data(), key.size());
    e->CalcTotalCharge(charge, metadata_charge_policy_);

    return e;
}

ErrorCode LRUCacheShard::Insert(const std::string_view &key,
                                uint32_t hash,
                                Cache::ObjectPtr value,
                                const Cache::CacheItemHelper *helper,
                                size_t charge,
                                LRUHandle **handle,
                                Cache::Priority priority) {
    LRUHandle *e = CreateHandle(key, hash, value, helper, charge);
    e->SetPriority(priority);
    e->SetInCache(true);
    return InsertItem(e, handle);
}

ErrorCode LRUCacheShard::InsertIfAbsent(const std::string_view &key,
                                        uint32_t hash,
                                        Cache::ObjectPtr value,
                                        const Cache::CacheItemHelper *helper,
                                        size_t charge,
                                        LRUHandle **handle,
                                        Cache::Priority priority) {
    LRUHandle *e = CreateHandle(key, hash, value, helper, charge);
    e->SetPriority(priority);
    e->SetInCache(true);
    return InsertItemIfAbsent(e, handle);
}

LRUHandle *LRUCacheShard::CreateStandalone(const std::string_view &key,
                                           uint32_t hash,
                                           Cache::ObjectPtr value,
                                           const Cache::CacheItemHelper *helper,
                                           size_t charge,
                                           bool allow_uncharged) {
    LRUHandle *e = CreateHandle(key, hash, value, helper, charge);
    e->SetIsStandalone(true);
    e->Ref();

    autovector<LRUHandle *> last_reference_list;

    {
        std::lock_guard<std::mutex> l(mutex_);

        if (!no_evict_on_insert_) {
            EvictFromLRU(e->total_charge, &last_reference_list);
        }

        if (strict_capacity_limit_ && (usage_ + e->total_charge) > capacity_) {
            if (allow_uncharged) {
                e->total_charge = 0;
            } else {
                free(e);
                e = nullptr;
            }
        } else {
            usage_ += e->total_charge;
        }
    }

    NotifyEvicted(last_reference_list);
    return e;
}

bool LRUCacheShard::Erase(const std::string_view &key, uint32_t hash) {
    LRUHandle *e;
    bool last_reference = false;
    {
        std::lock_guard<std::mutex> l(mutex_);
        e = table_.Remove(key, hash);
        if (e != nullptr) {
            assert(e->InCache());
            e->SetInCache(false);
            if (!e->HasRefs()) {
                // The entry is in LRU since it's in hash and has no external references
                LRU_Remove(e);
                assert(usage_ >= e->total_charge);
                usage_ -= e->total_charge;
                last_reference = true;
            }
        }
    }

    // Free the entry here outside of mutex for performance reasons.
    // last_reference will only be true if e != nullptr.
    if (last_reference) {
        e->Free(table_.GetAllocator());
    }
    return e != nullptr;
}

bool LRUCacheShard::Exists(const std::string_view &key, uint32_t hash) {
    std::lock_guard<std::mutex> l(mutex_);
    return table_.Lookup(key, hash) != nullptr;
}

size_t LRUCacheShard::GetUsage() const {
    std::lock_guard<std::mutex> l(mutex_);
    return usage_;
}

size_t LRUCacheShard::GetPinnedUsage() const {
    std::lock_guard<std::mutex> l(mutex_);
    assert(usage_ >= lru_usage_);
    return usage_ - lru_usage_;
}

size_t LRUCacheShard::GetOccupancyCount() const {
    std::lock_guard<std::mutex> l(mutex_);
    return table_.GetOccupancyCount();
}

size_t LRUCacheShard::GetTableAddressCount() const {
    std::lock_guard<std::mutex> l(mutex_);
    return size_t{1} << table_.GetLengthBits();
}

size_t LRUCacheShard::GetOldestKeys(size_t count, std::vector<std::string> &out_keys) {
    std::lock_guard<std::mutex> l(mutex_);
    size_t collected = 0;
    LRUHandle *current = lru_.next;
    while (current != &lru_ && collected < count) {
        if (current->InCache()) {
            out_keys.emplace_back(current->key());
            ++collected;
        }
        current = current->next;
    }
    return collected;
}

size_t LRUCacheShard::ApplyToOldestEntries(
    size_t count,
    const std::function<
        void(const std::string_view &key, Cache::ObjectPtr value, size_t charge, const Cache::CacheItemHelper *helper)>
        &callback) {
    std::lock_guard<std::mutex> l(mutex_);
    size_t visited = 0;
    LRUHandle *current = lru_.next;
    while (current != &lru_ && visited < count) {
        if (current->InCache()) {
            callback(current->key(), current->value, current->GetCharge(metadata_charge_policy_), current->helper);
            ++visited;
        }
        current = current->next;
    }
    return visited;
}

void LRUCacheShard::SetTailChangeCallback(uint32_t shard_id, const Cache::TailChangeCallback &callback) {
    std::lock_guard<std::mutex> l(mutex_);
    shard_id_ = shard_id;
    tail_change_callback_ = callback;
    // Notify immediately with current tail state so the caller gets the
    // initial value.
    NotifyTailChange();
}

void LRUCacheShard::NotifyTailChange() {
    if (!tail_change_callback_) {
        return;
    }
    LRUHandle *tail = lru_.next;
    if (tail == &lru_) {
        // LRU list is empty.
        tail_change_callback_(shard_id_, nullptr);
    } else {
        tail_change_callback_(shard_id_, tail->value);
    }
}

void LRUCacheShard::AppendPrintableOptions(std::string &str) const {
    const int kBufferSize = 200;
    char buffer[kBufferSize];
    {
        std::lock_guard<std::mutex> l(mutex_);
        snprintf(buffer, kBufferSize, "    high_pri_pool_ratio: %.3lf\n", high_pri_pool_ratio_);
        snprintf(buffer + strlen(buffer),
                 kBufferSize - strlen(buffer),
                 "    low_pri_pool_ratio: %.3lf\n",
                 low_pri_pool_ratio_);
    }
    str.append(buffer);
}

LRUCache::LRUCache(const LRUCacheOptions &opts) : ShardedCache(opts) {
    size_t per_shard = GetPerShardCapacity();
    MemoryAllocator *alloc = memory_allocator();
    InitShards([&](LRUCacheShard *cs) {
        new (cs) LRUCacheShard(per_shard,
                               opts.strict_capacity_limit,
                               opts.no_evict_on_insert,
                               opts.high_pri_pool_ratio,
                               opts.low_pri_pool_ratio,
                               opts.use_adaptive_mutex,
                               opts.metadata_charge_policy,
                               /* max_upper_hash_bits */ 32 - opts.num_shard_bits,
                               alloc,
                               &eviction_callback_);
    });
}

Cache::ObjectPtr LRUCache::Value(Handle *handle) {
    auto h = static_cast<const LRUHandle *>(handle);
    return h->value;
}

size_t LRUCache::GetCharge(Handle *handle) const {
    return static_cast<const LRUHandle *>(handle)->GetCharge(GetShard(0).metadata_charge_policy_);
}

const Cache::CacheItemHelper *LRUCache::GetCacheItemHelper(Handle *handle) const {
    auto h = static_cast<const LRUHandle *>(handle);
    return h->helper;
}

void LRUCache::LookupBatch(const std::string_view *keys, size_t count, Handle **out_handles) {
    BatchOperationScratch scratch;
    if (count > 1) {
        PrepareBatchOperationScratch(count, &scratch);
    }
    LookupBatchWithScratch(keys, count, out_handles, &scratch);
}

void LRUCache::PrepareBatchOperationScratch(size_t max_count, BatchOperationScratch *scratch) {
    if (!scratch) {
        return;
    }
    if (max_count <= 1) {
        return;
    }
    scratch->hashes.reserve(max_count);
    scratch->ordered_indices.reserve(max_count);
    if (UseSparseShardGrouping(max_count, GetNumShards())) {
        scratch->shard_offsets.reserve(max_count + 1);
    } else {
        const size_t shard_slots = GetNumShards() + 1;
        scratch->shard_offsets.reserve(shard_slots);
        scratch->cursors.reserve(shard_slots);
    }
}

void LRUCache::LookupBatchWithScratch(const std::string_view *keys,
                                      size_t count,
                                      Handle **out_handles,
                                      BatchOperationScratch *scratch) {
    if (count == 0) {
        return;
    }
    if (!scratch) {
        LookupBatch(keys, count, out_handles);
        return;
    }
    scratch->lookup_layout_valid = false;
    std::fill(out_handles, out_handles + count, nullptr);

    if (count == 1) {
        out_handles[0] = Lookup(keys[0]);
        return;
    }

    const size_t shard_count = GetNumShards();
    scratch->hashes.resize(count);
    auto &hashes = scratch->hashes;
    for (size_t i = 0; i < count; ++i) {
        hashes[i] = LRUCacheShard::ComputeHash(keys[i], hash_seed_);
    }

    auto shard_for_index = [&](size_t index) {
        return LRUCacheShard::HashPieceForSharding(hashes[index]) & shard_mask_;
    };
    if (UseSparseShardGrouping(count, shard_count)) {
        scratch->ordered_indices.resize(count);
        std::iota(scratch->ordered_indices.begin(), scratch->ordered_indices.end(), 0);
        BuildSparseShardRuns(count, shard_for_index, scratch);
        for (size_t i = 0; i + 1 < scratch->shard_offsets.size(); ++i) {
            const size_t begin = scratch->shard_offsets[i];
            const size_t end = scratch->shard_offsets[i + 1];
            GetShard(hashes[scratch->ordered_indices[begin]])
                .LookupBatch(keys, hashes.data(), scratch->ordered_indices.data() + begin, end - begin, out_handles);
        }
        scratch->lookup_layout_valid = true;
        return;
    }

    scratch->shard_offsets.assign(shard_count + 1, 0);
    auto &shard_offsets = scratch->shard_offsets;
    for (size_t i = 0; i < count; ++i) {
        const size_t shard = shard_for_index(i);
        ++shard_offsets[shard + 1];
    }
    for (size_t shard = 0; shard < shard_count; ++shard) {
        shard_offsets[shard + 1] += shard_offsets[shard];
    }

    scratch->cursors.assign(shard_offsets.begin(), shard_offsets.end());
    scratch->ordered_indices.resize(count);
    auto &cursors = scratch->cursors;
    auto &ordered_indices = scratch->ordered_indices;
    for (size_t i = 0; i < count; ++i) {
        const size_t shard = LRUCacheShard::HashPieceForSharding(hashes[i]) & shard_mask_;
        ordered_indices[cursors[shard]++] = i;
    }
    for (size_t shard = 0; shard < shard_count; ++shard) {
        const size_t begin = shard_offsets[shard];
        const size_t end = shard_offsets[shard + 1];
        if (begin == end) {
            continue;
        }
        GetShard(hashes[ordered_indices[begin]])
            .LookupBatch(keys, hashes.data(), ordered_indices.data() + begin, end - begin, out_handles);
    }
    scratch->lookup_layout_valid = true;
}

void LRUCache::ReleaseBatch(Handle *const *handles, size_t count) {
    BatchOperationScratch scratch;
    if (count > 1) {
        PrepareBatchOperationScratch(count, &scratch);
    }
    ReleaseBatchWithScratch(handles, count, &scratch);
}

void LRUCache::ReleaseBatchWithScratch(Handle *const *handles, size_t count, BatchOperationScratch *scratch) {
    if (count == 0) {
        return;
    }
    if (!scratch) {
        ReleaseBatch(handles, count);
        return;
    }
    if (count == 1) {
        if (handles[0] != nullptr) {
            Release(handles[0]);
        }
        return;
    }

    const size_t shard_count = GetNumShards();
    // A caller may null or replace handles between lookup and release. Reuse
    // the lookup runs only while every live handle still belongs to its run.
    const bool has_lookup_layout = scratch->lookup_layout_valid;
    scratch->lookup_layout_valid = false;
    if (has_lookup_layout && scratch->hashes.size() == count && scratch->ordered_indices.size() == count) {
        bool layout_matches = true;
        for (size_t i = 0; i < count; ++i) {
            if (handles[i] == nullptr) {
                continue;
            }
            const auto *entry = static_cast<const LRUHandle *>(handles[i]);
            const size_t lookup_shard = LRUCacheShard::HashPieceForSharding(scratch->hashes[i]) & shard_mask_;
            const size_t handle_shard = LRUCacheShard::HashPieceForSharding(entry->GetHash()) & shard_mask_;
            if (lookup_shard != handle_shard) {
                layout_matches = false;
                break;
            }
        }
        if (layout_matches) {
            for (size_t i = 0; i + 1 < scratch->shard_offsets.size(); ++i) {
                const size_t begin = scratch->shard_offsets[i];
                const size_t end = scratch->shard_offsets[i + 1];
                const auto first_live = std::find_if(scratch->ordered_indices.begin() + begin,
                                                     scratch->ordered_indices.begin() + end,
                                                     [&](size_t index) { return handles[index] != nullptr; });
                if (first_live == scratch->ordered_indices.begin() + end) {
                    continue;
                }
                GetShard(scratch->hashes[*first_live])
                    .ReleaseBatch(handles, scratch->ordered_indices.data() + begin, end - begin);
            }
            return;
        }
    }
    if (!UseSparseShardGrouping(count, shard_count)) {
        scratch->shard_offsets.assign(shard_count + 1, 0);
        auto &shard_offsets = scratch->shard_offsets;
        size_t non_null_count = 0;
        for (size_t i = 0; i < count; ++i) {
            if (handles[i] == nullptr) {
                continue;
            }
            const auto *entry = static_cast<const LRUHandle *>(handles[i]);
            const size_t shard = LRUCacheShard::HashPieceForSharding(entry->GetHash()) & shard_mask_;
            ++shard_offsets[shard + 1];
            ++non_null_count;
        }
        for (size_t shard = 0; shard < shard_count; ++shard) {
            shard_offsets[shard + 1] += shard_offsets[shard];
        }

        scratch->cursors.assign(shard_offsets.begin(), shard_offsets.end());
        scratch->ordered_indices.resize(non_null_count);
        auto &cursors = scratch->cursors;
        auto &ordered_indices = scratch->ordered_indices;
        for (size_t i = 0; i < count; ++i) {
            if (handles[i] == nullptr) {
                continue;
            }
            const auto *entry = static_cast<const LRUHandle *>(handles[i]);
            const size_t shard = LRUCacheShard::HashPieceForSharding(entry->GetHash()) & shard_mask_;
            ordered_indices[cursors[shard]++] = i;
        }
        for (size_t shard = 0; shard < shard_count; ++shard) {
            const size_t begin = shard_offsets[shard];
            const size_t end = shard_offsets[shard + 1];
            if (begin == end) {
                continue;
            }
            const auto *entry = static_cast<const LRUHandle *>(handles[ordered_indices[begin]]);
            GetShard(entry->GetHash()).ReleaseBatch(handles, ordered_indices.data() + begin, end - begin);
        }
        return;
    }

    scratch->ordered_indices.clear();
    for (size_t i = 0; i < count; ++i) {
        if (handles[i] != nullptr) {
            scratch->ordered_indices.push_back(i);
        }
    }
    const size_t non_null_count = scratch->ordered_indices.size();
    if (non_null_count == 0) {
        scratch->shard_offsets.clear();
        return;
    }

    auto shard_for_index = [&](size_t index) {
        const auto *entry = static_cast<const LRUHandle *>(handles[index]);
        return LRUCacheShard::HashPieceForSharding(entry->GetHash()) & shard_mask_;
    };
    BuildSparseShardRuns(non_null_count, shard_for_index, scratch);
    for (size_t i = 0; i + 1 < scratch->shard_offsets.size(); ++i) {
        const size_t begin = scratch->shard_offsets[i];
        const size_t end = scratch->shard_offsets[i + 1];
        const auto *entry = static_cast<const LRUHandle *>(handles[scratch->ordered_indices[begin]]);
        GetShard(entry->GetHash()).ReleaseBatch(handles, scratch->ordered_indices.data() + begin, end - begin);
    }
}

void LRUCache::ApplyToHandle(
    Cache *cache,
    Handle *handle,
    const std::function<
        void(const std::string_view &key, ObjectPtr value, size_t charge, const CacheItemHelper *helper)> &callback) {
    auto cache_ptr = static_cast<LRUCache *>(cache);
    auto h = static_cast<const LRUHandle *>(handle);
    callback(h->key(), h->value, h->GetCharge(cache_ptr->GetShard(0).metadata_charge_policy_), h->helper);
}

size_t LRUCache::TEST_GetLRUSize() {
    return SumOverShards([](LRUCacheShard &cs) { return cs.TEST_GetLRUSize(); });
}

double LRUCache::GetHighPriPoolRatio() { return GetShard(0).GetHighPriPoolRatio(); }

size_t LRUCache::GetOldestKeysInShard(uint32_t shard_id, size_t count, std::vector<std::string> &out_keys) {
    uint32_t num_shards = GetNumShards();
    if (shard_id >= num_shards || count == 0) {
        return 0;
    }
    return GetShard(shard_id).GetOldestKeys(count, out_keys);
}

size_t LRUCache::ApplyToOldestEntriesInShard(
    uint32_t shard_id,
    size_t count,
    const std::function<
        void(const std::string_view &key, ObjectPtr value, size_t charge, const CacheItemHelper *helper)> &callback) {
    const uint32_t num_shards = GetNumShards();
    if (shard_id >= num_shards || count == 0 || !callback) {
        return 0;
    }
    return GetShard(shard_id).ApplyToOldestEntries(count, callback);
}

void LRUCache::SetTailChangeCallback(TailChangeCallback callback) {
    uint32_t num_shards = GetNumShards();
    for (uint32_t i = 0; i < num_shards; ++i) {
        GetShard(i).SetTailChangeCallback(i, callback);
    }
}

} // namespace lru_cache

std::shared_ptr<Cache> LRUCacheOptions::MakeSharedCache() const {
    if (num_shard_bits >= 20) {
        return nullptr; // The cache cannot be sharded into too many fine pieces.
    }
    if (high_pri_pool_ratio < 0.0 || high_pri_pool_ratio > 1.0) {
        // Invalid high_pri_pool_ratio
        return nullptr;
    }
    if (low_pri_pool_ratio < 0.0 || low_pri_pool_ratio > 1.0) {
        // Invalid low_pri_pool_ratio
        return nullptr;
    }
    if (low_pri_pool_ratio + high_pri_pool_ratio > 1.0) {
        // Invalid high_pri_pool_ratio and low_pri_pool_ratio combination
        return nullptr;
    }
    // For sanitized options
    LRUCacheOptions opts = *this;
    if (opts.num_shard_bits < 0) {
        opts.num_shard_bits = GetDefaultCacheShardBits(capacity);
    }
    std::shared_ptr<Cache> cache = std::make_shared<LRUCache>(opts);
    // if (secondary_cache) {
    //   cache = std::make_shared<CacheWithSecondaryAdapter>(cache, secondary_cache);
    // }
    return cache;
}

std::shared_ptr<RowCache> LRUCacheOptions::MakeSharedRowCache() const {
    if (secondary_cache) {
        // Not allowed for a RowCache
        return nullptr;
    }
    // Works while RowCache is an alias for Cache
    return MakeSharedCache();
}
} // namespace kv_cache_manager
