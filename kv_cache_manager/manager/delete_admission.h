#pragma once

#include <chrono>
#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#include "kv_cache_manager/data_storage/pace_reclaim_free.h"
#include "kv_cache_manager/manager/delete_limits.h"
#include "kv_cache_manager/manager/free_busy.h"
#include "kv_cache_manager/metrics/metrics_registry.h"

namespace kv_cache_manager {

class DataStorageManager;
class DataStorageBackend;
class DeleteAdmission;

struct DeleteTargetCost {
    std::string storage;
    std::string bucket;
    DeleteSize size;
    uint64_t retry_bytes{0};
    std::weak_ptr<DataStorageBackend> provider_source{};
};

enum class DeleteBlockReason {
    Limit,
    Busy,
    Memory
};

struct DeleteDemand {
    DeleteSize size;
    uint64_t retry_bytes{0};
    std::map<std::string, DeleteSize> buckets;
    std::map<std::string, DeleteTargetCost> storages;
    void Add(const DeleteTargetCost &target);
};

class DeleteReservation {
public:
    ~DeleteReservation();
    void RetainStorageBatches(const std::set<std::string> &names);
    void ReleaseStorage(const std::string &name);
    void Finish();
    void Feedback(const std::string &storage, uint64_t busy_ms);
    bool Active() const;
    const DeleteSource source;
    const uint64_t generation;
    const std::shared_ptr<FreeBusyTask> protocol;

private:
    friend class DeleteAdmission;
    DeleteReservation(std::shared_ptr<DeleteAdmission> owner,
                      DeleteSource source,
                      DeleteDemand demand,
                      std::shared_ptr<FreeBusyTask> protocol);
    const std::shared_ptr<DeleteAdmission> owner_;
    DeleteDemand demand_;
    std::chrono::steady_clock::time_point submitted_at_;
    bool charged_{false};
};

// The executor owns one controller. A reservation survives delayed callbacks and
// returns the source request slot exactly once, even on cancellation.
class DeleteAdmission : public std::enable_shared_from_this<DeleteAdmission> {
public:
    DeleteAdmission(DeleteLimits limits,
                    std::shared_ptr<DataStorageManager> storage,
                    std::shared_ptr<MetricsRegistry> metrics);
    DeleteTargetCost Describe(const CacheLocation &location,
                              size_t instance_length,
                              DeleteSource source,
                              const std::string &group,
                              const DeleteDemand &selected) const;
    // On rejection, fits_alone tells the LRU selector if this Location can fit an empty window.
    bool TryAppend(DeleteSource source,
                   DeleteDemand &selected,
                   const DeleteTargetCost &target,
                   DeleteBlockReason *reason = nullptr,
                   bool *fits_alone = nullptr);
    uint64_t RemainingRequests(DeleteSource source) const;
    std::shared_ptr<DeleteReservation>
    TryReserve(DeleteSource source, const DeleteDemand &demand, DeleteBlockReason *reason = nullptr);
    // Select by the empty-window size, then wait for that entire prefix. A transient
    // shortage never fragments a page. Waiting takes place on the Trim caller.
    enum class TrimResult {
        Accepted,
        TooLarge,
        Cancelled
    };
    TrimResult ReserveTrim(const std::vector<DeleteTargetCost> &targets,
                           size_t begin,
                           uint64_t generation,
                           size_t &end,
                           std::shared_ptr<DeleteReservation> &reservation);
    void SetPaused(bool value);
    bool Accepting(uint64_t generation) const;
    uint64_t Generation() const { return resources->generation.load(); }
    const DeleteLimits limits;
    const std::shared_ptr<FreeBusyResources> resources;

private:
    friend class DeleteReservation;
    using Clock = std::chrono::steady_clock;
    struct SizeGauges {
        Gauge locations, bytes;
        void Set(DeleteSize size) {
            locations = static_cast<double>(size.locations);
            bytes = static_cast<double>(size.bytes);
        }
    };
    struct Bucket {
        DeleteSize used;
        PaceProviderCount providers;
        DeleteSize baseline;
        DeleteSize effective;
        Clock::time_point changed{}, cooldown{}, observe{};
        std::chrono::milliseconds interval{0};
        bool success{false};
        SizeGauges used_gauges, effective_gauges;
    };
    bool FitsSourceLocked(
        DeleteSource source, DeleteSize size, uint64_t retry_bytes, bool empty, DeleteBlockReason *reason) const;
    bool FitsBucketLocked(
        const std::string &key, DeleteSize size, bool empty, Clock::time_point now, DeleteBlockReason *reason);
    bool FitsNextLocked(DeleteSource source,
                        const DeleteDemand &selected,
                        const DeleteTargetCost &next,
                        bool empty,
                        Clock::time_point now,
                        DeleteBlockReason *reason);
    std::shared_ptr<DeleteReservation>
    TryReserveLocked(DeleteSource source, const DeleteDemand &demand, DeleteBlockReason *reason = nullptr);
    DeleteSize Hard(const std::string &bucket) const;
    DeleteSize Baseline(const std::string &bucket, PaceProviderCount providers, Clock::time_point now) const;
    Bucket &GetBucketLocked(const std::string &key, Clock::time_point now);
    void ObserveProvidersLocked(const DeleteTargetCost &target, Clock::time_point now);
    void RefreshWindow(const std::string &key, Bucket &bucket, Clock::time_point now);
    void ReleaseLocked(DeleteReservation &reservation, const DeleteSize &size, const std::string &bucket);
    std::shared_ptr<DataStorageManager> storage_;
    mutable std::mutex mutex_;
    std::condition_variable changed_;
    std::array<DeleteSourceLimit, 3> used_{};
    std::map<std::string, Bucket> buckets_;
    Clock::time_point next_idle_prune_{};
    Counter trim_wait_count_, trim_wait_us_;
    bool paused_{false};
};

} // namespace kv_cache_manager
