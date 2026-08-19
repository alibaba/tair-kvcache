#include "tools/kvcm_swarm/clients/v6d/local_cache.h"

#include <algorithm>

namespace kvcm_swarm {

LocalLease &LocalLease::operator=(LocalLease &&other) noexcept {
    if (this != &other) {
        Release();
        cache_ = other.cache_;
        object_id_ = std::move(other.object_id_);
        other.cache_ = nullptr;
    }
    return *this;
}

void LocalLease::Release() {
    if (cache_ != nullptr) {
        cache_->ReleaseLease(object_id_);
        cache_ = nullptr;
    }
}

LocalCache::LocalCache(SwarmExecutor &executor, uint64_t capacity_bytes)
    : executor_(executor), capacity_bytes_(capacity_bytes) {
    stats_.capacity_bytes = capacity_bytes;
}

std::optional<LocalLease> LocalCache::Acquire(const ObjectId &object_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = entries_.find(object_id);
    // An entry already marked evicting must not be handed to a new turn, so a
    // new use can never interleave with its cold write and delete.
    if (it == entries_.end() || it->second.state != CacheEntryState::kResident) {
        ++stats_.local_misses;
        return std::nullopt;
    }
    ++it->second.leases;
    TouchLocked(it->second);
    ++stats_.local_hits;
    return LocalLease(this, object_id);
}

Task<LocalLease>
LocalCache::ReserveAndInsert(GroupObject object, TimePoint deadline, StopToken stop, InsertOutcome *outcome) {
    InsertOutcome local_outcome = InsertOutcome::kInserted;
    if (outcome == nullptr) {
        outcome = &local_outcome;
    }
    const ObjectId object_id = object.object_key;
    const uint64_t size = object.object_size;
    if (size > capacity_bytes_) {
        std::lock_guard<std::mutex> lock(mutex_);
        ++stats_.insert_rejected_oversize;
        *outcome = InsertOutcome::kRejectedOversize;
        co_return LocalLease();
    }

    while (true) {
        std::shared_ptr<AsyncSlot<bool>> slot;
        bool need_trigger = false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            const auto existing = entries_.find(object_id);
            if (existing != entries_.end()) {
                if (existing->second.state == CacheEntryState::kResident) {
                    ++existing->second.leases;
                    TouchLocked(existing->second);
                    *outcome = InsertOutcome::kAlreadyResident;
                    co_return LocalLease(this, object_id);
                }
                // Being evicted: the same key must not have two concurrent
                // lifecycle operations, so this turn skips it. This is a benign
                // race, not capacity pressure.
                ++stats_.insert_skipped_evicting;
                *outcome = InsertOutcome::kSkippedEvicting;
                co_return LocalLease();
            }
            if (used_bytes_ + reserved_bytes_ + size <= capacity_bytes_) {
                reserved_bytes_ += size;
                break;
            }
            slot = std::make_shared<AsyncSlot<bool>>(executor_);
            waiters_.push_back(Waiter{slot, size});
            ++stats_.backpressure_waits;
            need_trigger = true;
        }
        if (need_trigger && trigger_) {
            // Only the bytes actually needed are requested; the pipeline picks
            // unleased LRU victims.
            trigger_();
        }
        const TimePoint wait_start = Now();
        executor_.ScheduleAt(deadline, [slot]() { slot->Complete(false); });
        StopCallbackGuard guard(stop, [slot]() { slot->Complete(false); });
        const bool granted = co_await *slot;
        const uint64_t waited_ns = static_cast<uint64_t>(std::max<int64_t>(0, (Now() - wait_start).count()));
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stats_.backpressure_wait_ns += waited_ns;
            stats_.backpressure_wait_ns_max = std::max(stats_.backpressure_wait_ns_max, waited_ns);
            if (!granted) {
                for (auto it = waiters_.begin(); it != waiters_.end(); ++it) {
                    if (it->slot == slot) {
                        waiters_.erase(it);
                        break;
                    }
                }
                if (stop.StopRequested()) {
                    ++stats_.insert_cancelled;
                } else {
                    ++stats_.backpressure_timeouts;
                }
            }
        }
        if (!granted) {
            *outcome = stop.StopRequested() ? InsertOutcome::kCancelled : InsertOutcome::kCapacityTimeout;
            co_return LocalLease();
        }
    }

    std::lock_guard<std::mutex> lock(mutex_);
    reserved_bytes_ -= size;
    Entry entry;
    entry.object = std::move(object);
    entry.state = CacheEntryState::kResident;
    entry.leases = 1;
    lru_.push_back(object_id);
    entry.lru = std::prev(lru_.end());
    used_bytes_ += size;
    ++stats_.inserts;
    stats_.used_bytes = used_bytes_;
    stats_.peak_used_bytes = std::max(stats_.peak_used_bytes, used_bytes_);
    entries_.emplace(object_id, std::move(entry));
    stats_.entries = entries_.size();
    stats_.peak_entries = std::max<uint64_t>(stats_.peak_entries, entries_.size());
    co_return LocalLease(this, object_id);
}

std::vector<GroupObject> LocalCache::SelectVictims(uint64_t bytes_needed, size_t max_batch) {
    std::vector<GroupObject> victims;
    std::lock_guard<std::mutex> lock(mutex_);
    uint64_t collected = 0;
    for (auto it = lru_.begin(); it != lru_.end() && victims.size() < max_batch;) {
        const ObjectId object_id = *it;
        ++it;
        auto entry_it = entries_.find(object_id);
        if (entry_it == entries_.end()) {
            continue;
        }
        Entry &entry = entry_it->second;
        if (entry.state != CacheEntryState::kResident || entry.leases > 0) {
            continue;
        }
        entry.state = CacheEntryState::kEvicting;
        collected += entry.object.object_size;
        victims.push_back(entry.object);
        ++stats_.victims_selected;
        if (collected >= bytes_needed) {
            break;
        }
    }
    if (victims.empty() && !waiters_.empty()) {
        ++stats_.no_victim_waits;
    }
    return victims;
}

std::vector<GroupObject> LocalCache::SelectAllEvictable(size_t max_batch) {
    return SelectVictims(UINT64_MAX, max_batch);
}

void LocalCache::MarkRemoved(const ObjectId &object_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = entries_.find(object_id);
    if (it == entries_.end()) {
        return;
    }
    used_bytes_ -= std::min(used_bytes_, it->second.object.object_size);
    lru_.erase(it->second.lru);
    entries_.erase(it);
    ++stats_.removed;
    stats_.used_bytes = used_bytes_;
    stats_.entries = entries_.size();
    WakeWaitersLocked();
}

void LocalCache::RestoreResident(const ObjectId &object_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = entries_.find(object_id);
    if (it == entries_.end()) {
        return;
    }
    if (it->second.state == CacheEntryState::kEvicting) {
        it->second.state = CacheEntryState::kResident;
        ++stats_.restored_resident;
    }
}

bool LocalCache::HasEvictable() const {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto &object_id : lru_) {
        const auto it = entries_.find(object_id);
        if (it != entries_.end() && it->second.state == CacheEntryState::kResident && it->second.leases == 0) {
            return true;
        }
    }
    return false;
}

uint64_t LocalCache::pending_wait_bytes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    uint64_t total = 0;
    for (const auto &waiter : waiters_) {
        total += waiter.bytes;
    }
    return total;
}

bool LocalCache::Contains(const ObjectId &object_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return entries_.find(object_id) != entries_.end();
}

bool LocalCache::IsResident(const ObjectId &object_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = entries_.find(object_id);
    return it != entries_.end() && it->second.state == CacheEntryState::kResident;
}

uint64_t LocalCache::used_bytes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return used_bytes_;
}

size_t LocalCache::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return entries_.size();
}

LocalCacheStats LocalCache::Stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    LocalCacheStats stats = stats_;
    stats.used_bytes = used_bytes_;
    stats.entries = entries_.size();
    return stats;
}

void LocalCache::ReleaseLease(const ObjectId &object_id) {
    bool became_evictable = false;
    bool has_waiters = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto it = entries_.find(object_id);
        if (it == entries_.end()) {
            return;
        }
        if (it->second.leases > 0) {
            --it->second.leases;
        }
        became_evictable = it->second.leases == 0 && it->second.state == CacheEntryState::kResident;
        has_waiters = !waiters_.empty();
    }
    // A capacity waiter may have found nothing evictable earlier; the entry that
    // just lost its last lease is now a valid victim, so the pipeline must be
    // woken explicitly. Without this the pipeline could sleep forever.
    if (became_evictable && has_waiters && trigger_) {
        trigger_();
    }
}

void LocalCache::WakeWaitersLocked() {
    // Grant waiters in FIFO order for whatever capacity is now free. The woken
    // waiter re-checks capacity itself, so nothing is reserved here.
    uint64_t free_bytes =
        capacity_bytes_ >= used_bytes_ + reserved_bytes_ ? capacity_bytes_ - used_bytes_ - reserved_bytes_ : 0;
    while (!waiters_.empty() && waiters_.front().bytes <= free_bytes) {
        const Waiter waiter = waiters_.front();
        waiters_.pop_front();
        if (waiter.slot->Complete(true)) {
            free_bytes -= waiter.bytes;
        }
    }
}

void LocalCache::TouchLocked(Entry &entry) {
    lru_.erase(entry.lru);
    lru_.push_back(entry.object.object_key);
    entry.lru = std::prev(lru_.end());
}

} // namespace kvcm_swarm
