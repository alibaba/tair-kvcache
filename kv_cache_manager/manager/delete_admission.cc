#include "kv_cache_manager/manager/delete_admission.h"

#include <algorithm>
#include <limits>

#include "kv_cache_manager/data_storage/data_storage_manager.h"
#include "kv_cache_manager/data_storage/data_storage_uri.h"

namespace kv_cache_manager {
namespace {
using Clock = std::chrono::steady_clock;

uint64_t Sum(uint64_t a, uint64_t b) { return a + std::min(b, std::numeric_limits<uint64_t>::max() - a); }
void AddSize(DeleteSize &a, DeleteSize b) { a = {Sum(a.locations, b.locations), Sum(a.bytes, b.bytes)}; }
void SubSize(DeleteSize &a, DeleteSize b) {
    a.locations -= b.locations;
    a.bytes -= b.bytes;
}
bool Fits(uint64_t used, uint64_t next, uint64_t limit) {
    return limit == 0 || (used <= limit && next <= limit - used);
}
bool Fits(DeleteSize used, DeleteSize next, DeleteSize limit) {
    return Fits(used.locations, next.locations, limit.locations) && Fits(used.bytes, next.bytes, limit.bytes);
}
size_t Index(DeleteSource s) { return static_cast<size_t>(s); }
} // namespace

void DeleteDemand::Add(const DeleteTargetCost &target) {
    AddSize(size, target.size);
    retry_bytes = Sum(retry_bytes, target.retry_bytes);
    if (!target.bucket.empty()) {
        AddSize(buckets[target.bucket], target.size);
    }
    if (!target.storage.empty()) {
        auto &part = storages[target.storage];
        part.bucket = target.bucket;
        AddSize(part.size, target.size);
    }
}

DeleteAdmission::DeleteAdmission(DeleteLimits config,
                                 std::shared_ptr<DataStorageManager> storage,
                                 std::shared_ptr<MetricsRegistry> metrics)
    : limits(std::move(config))
    , resources(std::make_shared<FreeBusyResources>(limits.retry_memory_bytes, metrics))
    , storage_(std::move(storage)) {
    if (metrics) {
        trim_wait_count_ = metrics->GetCounter("delete_admission.trim_wait_count");
        trim_wait_us_ = metrics->GetCounter("delete_admission.trim_wait_us");
    }
}

DeleteTargetCost DeleteAdmission::Describe(const CacheLocation &loc,
                                           size_t instance_length,
                                           DeleteSource source,
                                           const std::string &group,
                                           const DeleteDemand &selected) const {
    DeleteTargetCost cost;
    cost.size.locations = 1;
    const auto type = ToBaseType(loc.type());
    const bool tair = IsTairMempoolStorageType(type);
    for (const auto &spec : loc.location_specs()) {
        DataStorageUri uri(spec.uri());
        uint64_t bytes = 0;
        if (uri.Valid()) {
            uri.GetParamAs<uint64_t>("size", bytes);
            cost.size.bytes = Sum(cost.size.bytes, bytes);
            if (tair && cost.storage.empty()) {
                cost.storage = uri.GetHostName();
            }
        }
    }
    if (!tair) {
        if (source == DeleteSource::Reclaimer) {
            cost.bucket = "group/" + std::to_string(static_cast<int>(type)) + "/" + group;
        }
        return cost;
    }
    cost.retry_bytes = limits.protocol ? FreeBusyTask::RecordBytes(loc, instance_length) : 0;
    auto known = selected.storages.find(cost.storage);
    if (known != selected.storages.end()) {
        cost.bucket = known->second.bucket;
        return cost;
    }
    cost.bucket = "storage/" + cost.storage;
    if (storage_) {
        const auto backend = storage_->GetDataStorageBackend(cost.storage);
        if (backend) {
            const auto spec =
                std::dynamic_pointer_cast<TairMemPoolStorageSpec>(backend->GetStorageConfig().storage_spec());
            if (spec) {
                const auto &endpoint =
                    spec->service_discovery_url().empty() ? spec->domain() : spec->service_discovery_url();
                if (!endpoint.empty()) {
                    cost.bucket = "endpoint/" + endpoint;
                }
            }
        }
    }
    return cost;
}

DeleteSize DeleteAdmission::Hard(const std::string &key) const {
    return key.compare(0, 6, "group/") == 0 ? limits.non_tair_reclaim : limits.cluster;
}

uint64_t DeleteAdmission::RemainingRequests(DeleteSource source) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto &limit = limits.sources[Index(source)];
    const auto &used = used_[Index(source)];
    if (paused_ || (limit.size.locations && used.size.locations >= limit.size.locations) ||
        (limit.size.bytes && used.size.bytes >= limit.size.bytes)) {
        return 0;
    }
    return limit.requests ? limit.requests - std::min(limit.requests, used.requests)
                          : std::numeric_limits<uint64_t>::max();
}

bool DeleteAdmission::FitsSourceLocked(
    DeleteSource source, DeleteSize size, uint64_t retry_bytes, bool empty, DeleteBlockReason *reason) const {
    if (reason) {
        *reason = DeleteBlockReason::Limit;
    }
    const auto &limit = limits.sources[Index(source)];
    const auto used = empty ? DeleteSourceLimit{} : used_[Index(source)];
    if (!Fits(used.requests, 1, limit.requests) || !Fits(used.size, size, limit.size)) {
        return false;
    }
    if (retry_bytes && retry_bytes > resources->memory_limit - (empty ? 0 : resources->reserved.load())) {
        if (reason) {
            *reason = DeleteBlockReason::Memory;
        }
        return false;
    }
    return true;
}

bool DeleteAdmission::FitsBucketLocked(
    const std::string &key, DeleteSize wanted, bool empty, Clock::time_point now, DeleteBlockReason *reason) {
    const auto hard = Hard(key);
    const auto it = buckets_.find(key);
    const DeleteSize used = empty || it == buckets_.end() ? DeleteSize{} : it->second.used;
    if (!Fits(used, wanted, hard)) {
        return false;
    }
    if (it == buckets_.end()) {
        return true;
    }
    auto &b = it->second;
    if (b.success && now >= b.observe) {
        b.effective = {std::min(hard.locations, Sum(b.effective.locations, b.effective.locations)),
                       std::min(hard.bytes, Sum(b.effective.bytes, b.effective.bytes))};
        b.effective_gauges.Set(b.effective);
        b.changed = now;
        b.success = false;
        if (b.effective.locations == hard.locations && b.effective.bytes == hard.bytes) {
            b.interval = std::chrono::milliseconds::zero();
        }
        auto room = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::time_point::max() - now);
        b.observe = now + std::min(b.interval, room);
    }
    // A single Location can exceed only the temporary byte ceiling, after drain.
    const bool single = used.locations == 0 && wanted.locations == 1;
    if ((!empty && now < b.cooldown) || (!single && !Fits(used, wanted, b.effective))) {
        if (reason) {
            *reason = DeleteBlockReason::Busy;
        }
        return false;
    }
    return true;
}

bool DeleteAdmission::FitsLocked(
    DeleteSource source, const DeleteDemand &demand, bool empty, Clock::time_point now, DeleteBlockReason *reason) {
    if (!FitsSourceLocked(source, demand.size, demand.retry_bytes, empty, reason)) {
        return false;
    }
    for (const auto &[key, amount] : demand.buckets) {
        if (!FitsBucketLocked(key, amount, empty, now, reason)) {
            return false;
        }
    }
    return true;
}

// Picking a candidate touches only one cluster. Recheck every cluster atomically
// at submission, since another source can consume capacity during a scan.
bool DeleteAdmission::FitsNextLocked(DeleteSource source,
                                     const DeleteDemand &selected,
                                     const DeleteTargetCost &next,
                                     bool empty,
                                     Clock::time_point now,
                                     DeleteBlockReason *reason) {
    auto total = selected.size;
    AddSize(total, next.size);
    if (!FitsSourceLocked(source, total, Sum(selected.retry_bytes, next.retry_bytes), empty, reason)) {
        return false;
    }
    if (next.bucket.empty()) {
        return true;
    }
    auto wanted = next.size;
    auto it = selected.buckets.find(next.bucket);
    if (it != selected.buckets.end()) {
        AddSize(wanted, it->second);
    }
    return FitsBucketLocked(next.bucket, wanted, empty, now, reason);
}

bool DeleteAdmission::TryAppend(DeleteSource source,
                                DeleteDemand &selected,
                                const DeleteTargetCost &cost,
                                DeleteBlockReason *reason,
                                bool *fits_alone) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (paused_ || !FitsNextLocked(source, selected, cost, false, Clock::now(), reason)) {
        if (fits_alone) {
            // A single Location may exceed the temporary byte ceiling once the bucket drains.
            *fits_alone = FitsSourceLocked(source, cost.size, cost.retry_bytes, true, nullptr) &&
                          (cost.bucket.empty() || Fits({}, cost.size, Hard(cost.bucket)));
        }
        return false;
    }
    selected.Add(cost);
    return true;
}

DeleteReservation::DeleteReservation(std::shared_ptr<DeleteAdmission> owner,
                                     DeleteSource src,
                                     DeleteDemand demand,
                                     std::shared_ptr<FreeBusyTask> task)
    : source(src)
    , generation(owner->Generation())
    , protocol(std::move(task))
    , owner_(std::move(owner))
    , demand_(std::move(demand))
    , submitted_at_(Clock::now()) {}

std::shared_ptr<DeleteReservation> DeleteAdmission::ReserveLocked(DeleteSource source, const DeleteDemand &demand) {
    auto task = demand.retry_bytes ? std::make_shared<FreeBusyTask>(resources, limits.initial_backoff_ms) : nullptr;
    auto lease = std::shared_ptr<DeleteReservation>(new DeleteReservation(shared_from_this(), source, demand, task));
    if (task) {
        task->on_feedback = [weak = std::weak_ptr<DeleteReservation>(lease)](const std::string &name,
                                                                             uint64_t busy_ms) {
            if (auto reservation = weak.lock()) {
                reservation->Feedback(name, busy_ms);
            }
        };
        if (!resources->Reserve(demand.retry_bytes)) {
            return {};
        }
        task->unassigned = demand.retry_bytes;
    }
    for (const auto &entry : demand.buckets) {
        const auto &key = entry.first;
        auto [it, inserted] = buckets_.try_emplace(key);
        if (inserted) {
            auto &bucket = it->second;
            bucket.effective = Hard(key);
            if (resources->metrics && key.compare(0, 6, "group/") != 0) {
                const auto gauges = [this](const MetricsTags &tags) {
                    return SizeGauges{resources->metrics->GetGauge("delete_admission.cluster_locations", tags),
                                      resources->metrics->GetGauge("delete_admission.cluster_bytes", tags)};
                };
                bucket.used_gauges = gauges({{"cluster", key}, {"quota", "used"}});
                bucket.effective_gauges = gauges({{"cluster", key}, {"quota", "effective"}});
                bucket.effective_gauges.Set(bucket.effective);
            }
        }
    }
    auto &used = used_[Index(source)];
    ++used.requests;
    AddSize(used.size, demand.size);
    for (const auto &[key, size] : demand.buckets) {
        auto &bucket = buckets_.at(key);
        AddSize(bucket.used, size);
        bucket.used_gauges.Set(bucket.used);
    }
    lease->charged_ = true;
    return lease;
}

std::shared_ptr<DeleteReservation>
DeleteAdmission::TryReserve(DeleteSource source, const DeleteDemand &demand, DeleteBlockReason *reason) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (paused_ || !FitsLocked(source, demand, false, Clock::now(), reason)) {
        return {};
    }
    auto reservation = ReserveLocked(source, demand);
    if (!reservation && reason) {
        *reason = DeleteBlockReason::Memory;
    }
    return reservation;
}

DeleteAdmission::TrimResult DeleteAdmission::ReserveTrim(const std::vector<DeleteTargetCost> &targets,
                                                         size_t begin,
                                                         uint64_t generation,
                                                         size_t &end,
                                                         std::shared_ptr<DeleteReservation> &lease) {
    std::unique_lock<std::mutex> lock(mutex_);
    bool waited = false;
    while (!paused_ && Generation() == generation) {
        const auto now = Clock::now();
        DeleteDemand prefix;
        end = begin;
        for (; end < targets.size(); ++end) {
            if (!FitsNextLocked(DeleteSource::Trim, prefix, targets[end], true, now, nullptr)) {
                break;
            }
            prefix.Add(targets[end]);
        }
        if (end == begin) {
            return TrimResult::TooLarge;
        }
        if (FitsLocked(DeleteSource::Trim, prefix, false, now)) {
            lease = ReserveLocked(DeleteSource::Trim, prefix);
            if (lease) {
                return TrimResult::Accepted;
            }
        }
        // Completions and lifecycle changes notify immediately. Also revisit cooldown
        // expiry and memory returned by a still-running multi-storage task.
        if (!waited) {
            ++trim_wait_count_;
            waited = true;
        }
        const auto wait_start = Clock::now();
        changed_.wait_for(lock, std::chrono::milliseconds(100));
        trim_wait_us_ += std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - wait_start).count();
    }
    return TrimResult::Cancelled;
}

bool DeleteAdmission::Accepting(uint64_t generation) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return !paused_ && generation == Generation();
}

void DeleteAdmission::SetPaused(bool value) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (value && !paused_) {
        ++resources->generation;
        for (auto it = buckets_.begin(); it != buckets_.end();) {
            auto &bucket = it->second;
            bucket.effective = Hard(it->first);
            bucket.effective_gauges.Set(bucket.effective);
            if (!bucket.used.locations) {
                it = buckets_.erase(it);
            } else {
                bucket.changed = bucket.cooldown = bucket.observe = {};
                bucket.interval = {};
                bucket.success = false;
                ++it;
            }
        }
    }
    paused_ = value;
    changed_.notify_all();
}

bool DeleteReservation::Active() const { return generation == owner_->Generation(); }
DeleteReservation::~DeleteReservation() {
    if (charged_) {
        Finish();
    }
}

void DeleteAdmission::ReleaseLocked(DeleteReservation &lease, const DeleteSize &size, const std::string &key) {
    auto it = buckets_.find(key);
    if (it != buckets_.end()) {
        auto &b = it->second;
        SubSize(b.used, size);
        b.used_gauges.Set(b.used);
        const auto hard = Hard(key);
        if (!b.used.locations && b.effective.locations == hard.locations && b.effective.bytes == hard.bytes) {
            buckets_.erase(it);
        }
    }
    SubSize(used_[Index(lease.source)].size, size);
}

// Ordinary targets have already finished physical deletion and CAD. Keep only
// the Tair storage batches that can still be outstanding after this worker exits.
void DeleteReservation::RetainStorageBatches(const std::set<std::string> &names) {
    std::lock_guard<std::mutex> lock(owner_->mutex_);
    if (!charged_) {
        return;
    }
    DeleteDemand remaining;
    for (const auto &name : names) {
        auto it = demand_.storages.find(name);
        if (it != demand_.storages.end()) {
            remaining.Add({name, it->second.bucket, it->second.size});
        }
    }
    DeleteSize released = demand_.size;
    SubSize(released, remaining.size);
    for (const auto &[key, total] : demand_.buckets) {
        auto amount = total;
        auto kept = remaining.buckets.find(key);
        if (kept != remaining.buckets.end()) {
            SubSize(amount, kept->second);
        }
        if (amount.locations) {
            owner_->ReleaseLocked(*this, amount, key);
            SubSize(released, amount);
        }
    }
    SubSize(owner_->used_[Index(source)].size, released);
    demand_ = std::move(remaining);
    owner_->changed_.notify_all();
}

void DeleteReservation::ReleaseStorage(const std::string &name) {
    std::lock_guard<std::mutex> lock(owner_->mutex_);
    auto it = demand_.storages.find(name);
    if (!charged_ || it == demand_.storages.end()) {
        return;
    }
    auto &part = it->second;
    owner_->ReleaseLocked(*this, part.size, part.bucket);
    SubSize(demand_.size, part.size);
    auto bucket = demand_.buckets.find(part.bucket);
    SubSize(bucket->second, part.size);
    if (!bucket->second.locations) {
        demand_.buckets.erase(bucket);
    }
    demand_.storages.erase(it);
    owner_->changed_.notify_all();
}

void DeleteReservation::Finish() {
    std::lock_guard<std::mutex> lock(owner_->mutex_);
    if (!charged_) {
        return;
    }
    if (protocol) {
        protocol->ReleaseUnused();
    }
    for (const auto &[key, size] : demand_.buckets) {
        owner_->ReleaseLocked(*this, size, key);
        SubSize(demand_.size, size);
    }
    auto &used = owner_->used_[Index(source)];
    SubSize(used.size, demand_.size);
    --used.requests;
    charged_ = false;
    owner_->changed_.notify_all();
}

void DeleteReservation::Feedback(const std::string &storage, uint64_t busy_ms) {
    std::lock_guard<std::mutex> lock(owner_->mutex_);
    auto part = demand_.storages.find(storage);
    if (!charged_ || !Active() || part == demand_.storages.end()) {
        return;
    }
    const auto &key = part->second.bucket;
    auto &bucket = owner_->buckets_.at(key);
    if (submitted_at_ <= bucket.changed) {
        return;
    }
    const auto now = Clock::now();
    if (busy_ms == 0) {
        if (now >= bucket.cooldown && bucket.interval.count() != 0) {
            bucket.success = true;
        }
    } else {
        bucket.effective = {std::max<uint64_t>(1, std::min(bucket.effective.locations, bucket.used.locations) / 2),
                            std::max<uint64_t>(1, std::min(bucket.effective.bytes, bucket.used.bytes) / 2)};
        bucket.effective_gauges.Set(bucket.effective);
        const auto room = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::time_point::max() - now).count();
        bucket.interval = std::chrono::milliseconds(std::min<uint64_t>(busy_ms, room / 2));
        bucket.changed = now;
        bucket.cooldown = now + bucket.interval;
        bucket.observe = bucket.cooldown + bucket.interval;
        bucket.success = false;
    }
    owner_->changed_.notify_all();
}

} // namespace kv_cache_manager
