// Per-process byte-accounted LRU cache.
//
// This is the only source of truth for local hits. A session never owns an
// entry: a turn holds a short lease for the duration of its use, and a leased
// object can never be selected for eviction. `used_bytes + reserved_bytes`
// never exceeds `capacity_bytes`, and waiting for capacity never blocks an
// Executor worker.
#pragma once

#include <cstdint>
#include <deque>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "tools/kvcm_swarm/clients/v6d/workload.h"
#include "tools/kvcm_swarm/runtime/executor.h"

namespace kvcm_swarm {

using ObjectId = std::string;

enum class CacheEntryState {
    kResident,
    kEvicting
};

// Why an insert did or did not take effect. The distinction matters: a capacity
// timeout is real backpressure, while hitting a key that is currently being
// evicted is a benign lifecycle race that must not be reported as pressure.
enum class InsertOutcome {
    kInserted,
    kAlreadyResident,
    kSkippedEvicting,
    kRejectedOversize,
    kCapacityTimeout,
    kCancelled,
};

class LocalCache;

// Move-only short-lived lease. Destroying it releases the reference.
class LocalLease {
public:
    LocalLease() = default;
    LocalLease(LocalCache *cache, ObjectId object_id) : cache_(cache), object_id_(std::move(object_id)) {}
    LocalLease(LocalLease &&other) noexcept : cache_(other.cache_), object_id_(std::move(other.object_id_)) {
        other.cache_ = nullptr;
    }
    LocalLease &operator=(LocalLease &&other) noexcept;
    LocalLease(const LocalLease &) = delete;
    LocalLease &operator=(const LocalLease &) = delete;
    ~LocalLease() { Release(); }

    bool valid() const { return cache_ != nullptr; }
    const ObjectId &object_id() const { return object_id_; }
    void Release();

private:
    LocalCache *cache_ = nullptr;
    ObjectId object_id_;
};

struct LocalCacheStats {
    uint64_t capacity_bytes = 0;
    uint64_t used_bytes = 0;
    uint64_t peak_used_bytes = 0;
    uint64_t entries = 0;
    uint64_t peak_entries = 0;
    uint64_t local_hits = 0;
    uint64_t local_misses = 0;
    uint64_t inserts = 0;
    uint64_t insert_rejected_oversize = 0;
    uint64_t removed = 0;
    uint64_t restored_resident = 0;
    uint64_t victims_selected = 0;
    uint64_t backpressure_waits = 0;
    uint64_t backpressure_timeouts = 0;
    uint64_t backpressure_wait_ns = 0;
    uint64_t backpressure_wait_ns_max = 0;
    uint64_t no_victim_waits = 0;
    uint64_t insert_skipped_evicting = 0;
    uint64_t insert_cancelled = 0;
};

class LocalCache {
public:
    LocalCache(SwarmExecutor &executor, uint64_t capacity_bytes);

    // Called by the process to wake its single eviction pipeline.
    void SetEvictionTrigger(std::function<void()> trigger) { trigger_ = std::move(trigger); }

    // Resident-only acquire; increments the short lease count.
    std::optional<LocalLease> Acquire(const ObjectId &object_id);

    // Waits only for the bytes actually needed, then inserts. Returns an
    // invalid lease when the deadline passed or the run is stopping.
    Task<LocalLease>
    ReserveAndInsert(GroupObject object, TimePoint deadline, StopToken stop, InsertOutcome *outcome = nullptr);

    // Marks unleased LRU-tail entries as evicting and returns them.
    std::vector<GroupObject> SelectVictims(uint64_t bytes_needed, size_t max_batch);
    // Shutdown flush: every unleased resident entry, oldest first.
    std::vector<GroupObject> SelectAllEvictable(size_t max_batch);

    // Cold write closed: drop the local object and release its capacity.
    void MarkRemoved(const ObjectId &object_id);
    // Eviction explicitly did not happen: the entry becomes resident again.
    void RestoreResident(const ObjectId &object_id);

    // True when at least one unleased resident entry could be evicted now.
    bool HasEvictable() const;
    // Total bytes that capacity waiters are currently blocked on.
    uint64_t pending_wait_bytes() const;

    bool Contains(const ObjectId &object_id) const;
    bool IsResident(const ObjectId &object_id) const;
    uint64_t used_bytes() const;
    size_t size() const;
    LocalCacheStats Stats() const;

private:
    friend class LocalLease;

    struct Entry {
        GroupObject object;
        CacheEntryState state = CacheEntryState::kResident;
        uint32_t leases = 0;
        std::list<ObjectId>::iterator lru;
    };

    struct Waiter {
        std::shared_ptr<AsyncSlot<bool>> slot;
        uint64_t bytes = 0;
    };

    void ReleaseLease(const ObjectId &object_id);
    void WakeWaitersLocked();
    void TouchLocked(Entry &entry);

    SwarmExecutor &executor_;
    const uint64_t capacity_bytes_;
    std::function<void()> trigger_;

    mutable std::mutex mutex_;
    std::unordered_map<ObjectId, Entry> entries_;
    std::list<ObjectId> lru_; // front = least recently used
    uint64_t used_bytes_ = 0;
    uint64_t reserved_bytes_ = 0;
    std::deque<Waiter> waiters_;
    LocalCacheStats stats_;
};

} // namespace kvcm_swarm
