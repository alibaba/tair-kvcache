#include <algorithm>
#include <map>
#include <random>

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/data_storage/data_storage_manager.h"
#include "kv_cache_manager/data_storage/data_storage_uri.h"
#include "kv_cache_manager/data_storage/reclaim_storage.h"
#include "kv_cache_manager/manager/free_busy.h"
#include "kv_cache_manager/manager/schedule_plan_executor.h"
#include "kv_cache_manager/meta/meta_indexer.h"
#include "kv_cache_manager/meta/meta_indexer_manager.h"
#include "kv_cache_manager/metrics/metrics_registry.h"

namespace kv_cache_manager {
namespace {
using Clock = FreeBusyTask::Clock;
using Millis = std::chrono::milliseconds;
// Operational limit for starting another attempt, measured from the first physical Free.
constexpr auto kRetryDeadline = std::chrono::seconds(30);

// Use fixed reason labels. Location IDs and exception details belong in logs.
void RecordRetained(const FreeBusyResources &resources, const char *reason, size_t count) {
    if (resources.metrics && count != 0) {
        try {
            resources.metrics->GetCounter("cache_reclaimer.free_busy_retained_location_count", {{"reason", reason}}) +=
                count;
        } catch (const std::exception &e) { KVCM_LOG_WARN("Failed to record retained Free locations: %s", e.what()); }
    }
}

uint64_t Jitter(uint64_t base, uint64_t max_ms) {
    thread_local std::mt19937 random(std::random_device{}());
    const auto low = base == max_ms ? std::max<uint64_t>(1, base * 4 / 5) : base;
    return std::uniform_int_distribution<uint64_t>(low, std::min(max_ms, base + base / 4))(random);
}

struct TaskCompletion {
    std::mutex mutex;
    size_t remaining = 0;
    PlanExecuteResult result;
    std::function<void(PlanExecuteResult)> done;

    void FinishBatch(bool ok) {
        std::lock_guard<std::mutex> lock(mutex);
        if (!ok && result.status == EC_OK) {
            result = {EC_PARTIAL_OK, "reclaim delete or CAD failed", true};
        }
        if (--remaining == 0) {
            done(std::move(result));
        }
    }
};

// The delayed closure owns only this batch. Location snapshots, parsed URIs,
// worker permits and strong backend/indexer references live on the call stack.
struct Batch : std::enable_shared_from_this<Batch> {
    Batch(SchedulePlanExecutor &executor,
          std::shared_ptr<DataStorageManager> storage,
          std::shared_ptr<MetaIndexerManager> meta,
          const std::shared_ptr<MetaIndexer> &indexer,
          const CacheLocationDelRequest &request,
          std::string name,
          std::unique_ptr<ReclaimStorageBatch> storage_batch,
          std::shared_ptr<TaskCompletion> completion)
        : executor(executor)
        , storage(std::move(storage))
        , meta(std::move(meta))
        , indexer(indexer)
        , task(request.free_busy)
        , reservation(request.reservation)
        , instance(request.instance_id)
        , name(std::move(name))
        , storage_batch(std::move(storage_batch))
        , completion(std::move(completion))
        , next_delay(task->initial_ms) {}

    SchedulePlanExecutor &executor;
    const std::shared_ptr<DataStorageManager> storage;
    const std::shared_ptr<MetaIndexerManager> meta;
    const std::weak_ptr<MetaIndexer> indexer;
    const std::shared_ptr<FreeBusyTask> task;
    const std::shared_ptr<DeleteReservation> reservation;
    const std::string instance, name;
    const std::unique_ptr<ReclaimStorageBatch> storage_batch;
    const std::shared_ptr<TaskCompletion> completion;
    struct SavedSpec {
        std::string uri;
        bool confirmed_missing;
    };
    std::map<std::pair<int64_t, std::string>, std::vector<SavedSpec>> records;
    uint64_t reserved = 0;
    uint64_t next_delay;
    Clock::time_point deadline;
    bool failed = false;
    bool holds_slot = false;
    bool retried_primary_read = false;

    void ReleaseSlot() {
        if (holds_slot) {
            --task->resources->executing_retries;
            holds_slot = false;
        }
    }

    ~Batch() { task->resources->reserved.fetch_sub(reserved); }

    void Finish(bool ok) {
        ReleaseSlot();
        records.clear();
        task->resources->reserved.fetch_sub(reserved);
        reserved = 0;
        if (reservation) {
            reservation->ReleaseStorage(name);
        }
        completion->FinishBatch(ok && !failed);
    }

    void Stop(const char *reason, const char *detail = nullptr) {
        RecordRetained(*task->resources, reason, records.size());
        for (const auto &[id, uris] : records) {
            KVCM_LOG_WARN("Reclaim delete stopped: %s, instance[%s] key[%ld] location[%s] storage[%s]",
                          detail ? detail : reason,
                          instance.c_str(),
                          id.first,
                          id.second.c_str(),
                          name.c_str());
        }
        Finish(false);
    }

    bool StopIfCancelledOrExpired(bool expired, const char *deadline_reason) {
        if (!task->Active()) {
            Stop("cancelled");
            return true;
        }
        if (expired) {
            Stop(deadline_reason);
            return true;
        }
        return false;
    }

    void Schedule(uint64_t wait_ms, const char *deadline_reason) {
        ReleaseSlot();
        const auto left = std::chrono::duration_cast<Millis>(deadline - Clock::now()).count();
        if (StopIfCancelledOrExpired(left <= 0 || wait_ms >= static_cast<uint64_t>(left), deadline_reason)) {
            return;
        }
        auto self = shared_from_this();
        if (!executor.SubmitTask(
                reservation && reservation->source == DeleteSource::Trim ? ScheduleTaskClass::kSystem
                                                                         : ScheduleTaskClass::kReclaim,
                [self, deadline_reason]() {
                    try {
                        self->Retry(deadline_reason);
                    } catch (const std::exception &e) {
                        self->Stop("local_error", e.what());
                    } catch (...) {
                        self->Stop("local_error", "retry exception");
                    }
                },
                Millis(wait_ms),
                [self]() { self->Stop("cancelled", "executor stopped"); })) {
            Stop("local_error", "retry submission rejected");
        }
    }

    enum class ReadResult {
        kOk,
        kRetryable,
        kFailed
    };

    ReadResult ReadTargets(const std::shared_ptr<MetaIndexer> &idx, ReclaimDeleteLocations &out) {
        KeyVector keys;
        for (const auto &[id, _] : records) {
            if (keys.empty() || keys.back() != id.first) {
                keys.push_back(id.first);
            }
        }
        RequestContext ctx("reclaim_free_retry");
        CacheLocationMapVector maps;
        auto read = idx->GetLocationsFromPrimary(&ctx, keys, maps);
        if (read.error_codes.size() != keys.size() || maps.size() != keys.size()) {
            return ReadResult::kFailed;
        }
        // Validate the entire read before dropping records or assembling a partial Free.
        bool retryable_error = false;
        for (auto ec : read.error_codes) {
            if (ec == EC_ERROR || ec == EC_TIMEOUT || ec == EC_IO_ERROR) {
                retryable_error = true;
            } else if (ec != EC_OK && ec != EC_NOENT) {
                return ReadResult::kFailed;
            }
        }
        if (retryable_error) {
            return ReadResult::kRetryable;
        }
        std::vector<ReclaimDeleteUri> specs;
        auto it = records.begin();
        for (size_t i = 0; i < keys.size(); ++i) {
            while (it != records.end() && it->first.first == keys[i]) {
                const auto found = maps[i].find(it->first.second);
                const auto loc = found == maps[i].end() ? nullptr : found->second;
                // The original URI list also pins the registered storage name.
                if (read.error_codes[i] == EC_NOENT || !loc || loc->status() != CacheLocationStatus::CLS_DELETING ||
                    !SupportsAdaptiveDelete(loc->type()) ||
                    !std::equal(it->second.begin(),
                                it->second.end(),
                                loc->location_specs().begin(),
                                loc->location_specs().end(),
                                [](const auto &saved, const auto &spec) { return saved.uri == spec.uri(); })) {
                    it = records.erase(it);
                    continue;
                }
                specs.clear();
                for (const auto &saved : it->second) {
                    specs.push_back({saved.uri, saved.confirmed_missing});
                }
                if (!storage_batch->AddLocation(specs)) {
                    RecordRetained(*task->resources, "invalid_uri", 1);
                    failed = true;
                    it = records.erase(it);
                    continue;
                }
                out.emplace_back(keys[i], loc);
                ++it;
            }
        }
        return ReadResult::kOk;
    }

    void Retry(const char *deadline_reason) {
        if (StopIfCancelledOrExpired(Clock::now() >= deadline, deadline_reason)) {
            return;
        }
        if (!task->resources->TryStartRetry(executor.GetWorkerCount())) {
            Schedule(Jitter(task->initial_ms, task->max_ms), "slot_deadline");
            return;
        }
        holds_slot = true;
        auto idx = indexer.lock();
        if (!idx || meta->GetMetaIndexer(instance) != idx) {
            Stop("instance_changed");
            return;
        }
        ReclaimDeleteLocations current;
        const auto read = ReadTargets(idx, current);
        if (read == ReadResult::kRetryable && !retried_primary_read) {
            retried_primary_read = true;
            Schedule(Jitter(task->initial_ms, task->max_ms), "primary_read");
            return;
        }
        if (read != ReadResult::kOk) {
            Stop("primary_read", "primary read failed");
            return;
        }
        if (current.empty()) {
            Finish(true);
            return;
        }
        Send(current, idx, true);
    }

    bool Cad(const ReclaimDeleteLocations &targets, const std::shared_ptr<MetaIndexer> &idx, bool retry) {
        KeyVector keys;
        LocationIdsPerKey ids;
        std::vector<std::vector<std::string>> expected;
        std::vector<std::vector<MetaSearcher::LocationCADTask>> statuses;
        for (const auto &[key, loc] : targets) {
            if (keys.empty() || keys.back() != key) {
                keys.push_back(key);
                if (retry) {
                    ids.emplace_back();
                    expected.emplace_back();
                } else {
                    statuses.emplace_back();
                }
            }
            if (retry) {
                ids.back().push_back(loc->id());
                expected.back().push_back(loc->ToJsonString());
            } else {
                statuses.back().push_back({loc->id(), CacheLocationStatus::CLS_DELETING});
            }
        }
        RequestContext ctx("reclaim_free_cad");
        MetaSearcher searcher(idx);
        std::vector<std::vector<ErrorCode>> results;
        const auto ec = retry ? searcher.BatchDeleteLocations(&ctx, keys, ids, results, expected)
                              : searcher.BatchCADLocationStatus(&ctx, keys, statuses, results);
        if (ec != EC_OK || results.size() != keys.size()) {
            return false;
        }
        for (size_t i = 0; i < results.size(); ++i) {
            if (results[i].size() != (retry ? ids[i].size() : statuses[i].size()) ||
                std::any_of(results[i].begin(), results[i].end(), [](auto code) { return code != EC_OK; })) {
                return false;
            }
        }
        return true;
    }

    void Send(const ReclaimDeleteLocations &targets,
              const std::shared_ptr<MetaIndexer> &idx,
              bool retry,
              const std::set<std::string> &skipped = {}) {
        uint64_t needed = 0;
        for (const auto &[key, loc] : targets) {
            needed += FreeBusyTask::RecordBytes(*loc, instance.size());
        }
        if (needed > reserved) {
            const auto extra = needed - reserved;
            if (!(retry ? task->resources->Reserve(extra) : task->Take(extra))) {
                // Without the original URI list a delayed callback cannot identify this allocation.
                Stop("budget", "insufficient budget for original Free targets");
                return;
            }
        } else if (needed < reserved) {
            // ReadTargets dropped locations. Their reservations are available to other batches now.
            task->resources->reserved.fetch_sub(reserved - needed);
        }
        reserved = needed;
        if (StopIfCancelledOrExpired(retry && Clock::now() >= deadline, "dispatch_deadline")) {
            return;
        }
        const auto stop_reason = [&]() -> const char * {
            if (!task->Active()) {
                return "cancelled";
            }
            return retry && Clock::now() >= deadline ? "dispatch_deadline" : nullptr;
        };
        const auto reply = storage_batch->Send(*storage, "location_del_task_trace", stop_reason);
        ReleaseSlot();
        if (retry) {
            task->resources->retry_attempts += reply.requests;
        }
        if (reply.busy_replies) {
            task->resources->busy_replies += reply.busy_replies;
        }
        const uint64_t wait = reply.busy_replies ? std::max(Jitter(next_delay, task->max_ms), reply.retry_after_ms) : 0;
        if (reply.busy_replies || reply.recovery_confirmed) {
            task->Record(name, wait);
        }
        if (reply.locations.size() != targets.size()) {
            Stop("local_error", "storage returned an inconsistent Location count");
            return;
        }
        for (const auto &[location_index, spec_index] : reply.failed_specs) {
            if (location_index >= targets.size() ||
                spec_index >= targets[location_index].second->location_specs().size()) {
                Stop("local_error", "storage returned an invalid failed spec");
                return;
            }
            const auto &[key, loc] = targets[location_index];
            KVCM_LOG_WARN("Reclaim delete failed, possible leak instance[%s] key[%ld] location[%s] uri[%s]",
                          instance.c_str(),
                          key,
                          loc->id().c_str(),
                          loc->location_specs()[spec_index].uri().c_str());
            failed = true;
        }
        ReclaimDeleteLocations completed;
        uint64_t waiting_bytes = 0;
        for (size_t i = 0; i < targets.size(); ++i) {
            const auto &[key, loc] = targets[i];
            const auto &outcome = reply.locations[i];
            if (outcome.action == ReclaimStorageResult::Action::Retry) {
                waiting_bytes += FreeBusyTask::RecordBytes(*loc, instance.size());
                if (!retry) {
                    auto &uris = records.at({key, loc->id()});
                    uris.reserve(loc->location_specs().size());
                    for (const auto &spec : loc->location_specs()) {
                        uris.push_back({spec.uri(), skipped.count(spec.uri()) != 0});
                    }
                }
                continue;
            }
            records.erase({key, loc->id()});
            if (outcome.action == ReclaimStorageResult::Action::Retain) {
                failed = true;
                RecordRetained(*task->resources, outcome.reason, 1);
                KVCM_LOG_WARN("Reclaim delete stopped: %s, instance[%s] key[%ld] location[%s] storage[%s]",
                              outcome.reason,
                              instance.c_str(),
                              key,
                              loc->id().c_str(),
                              name.c_str());
            } else {
                completed.emplace_back(key, loc);
            }
        }
        if (!completed.empty() && !Cad(completed, idx, retry)) {
            failed = true;
            KVCM_LOG_WARN("Reclaim CAD failed instance[%s] storage[%s] locations[%zu]",
                          instance.c_str(),
                          name.c_str(),
                          completed.size());
        }
        if (!reply.busy_replies) {
            Finish(true);
            return;
        }
        task->resources->reserved.fetch_sub(reserved - waiting_bytes);
        reserved = waiting_bytes;
        KVCM_LOG_DEBUG("Reclaim delete busy instance[%s] storage[%s] locations[%zu] retry[%d] wait_ms[%lu]",
                       instance.c_str(),
                       name.c_str(),
                       records.size(),
                       retry,
                       wait);
        next_delay = std::min(next_delay * 2, task->max_ms);
        Schedule(wait, "busy_deadline");
    }
};
} // namespace

void SchedulePlanExecutor::RunReclaimDeleteTask(const CacheLocationDelRequest &request,
                                                const std::shared_ptr<MetaIndexer> &indexer,
                                                std::function<void(PlanExecuteResult)> complete) {
    ReclaimDeleteLocations locations;
    auto result = DoLocationDelTask(request, indexer, &locations);
    struct StorageGroup {
        std::unique_ptr<ReclaimStorageBatch> storage_batch;
        ReclaimDeleteLocations locations;
        std::set<std::string> skipped;
    };
    std::map<std::string, StorageGroup> groups;
    std::vector<ReclaimDeleteUri> uris;
    for (const auto &[key, loc] : locations) {
        const auto &specs = loc->location_specs();
        const std::string name = specs.empty() ? std::string{} : DataStorageUri(specs.front().uri()).GetHostName();
        auto &group = groups[name];
        if (!group.storage_batch) {
            group.storage_batch = CreateReclaimStorageBatch(loc->type(), name);
        }
        const bool skip_missing = !request.confirmed_missing_uris.empty() && !specs.empty() &&
                                  data_storage_manager_->ShouldSkipConfirmedMissingDelete(name);
        uris.clear();
        for (const auto &spec : specs) {
            const bool skip =
                skip_missing && request.confirmed_missing_uris.count(DataStorageUri(spec.uri()).ToUriString());
            uris.push_back({spec.uri(), skip});
        }
        if (!group.storage_batch || !group.storage_batch->AddLocation(uris)) {
            KVCM_LOG_WARN("Invalid reclaim target instance[%s] key[%ld] storage[%s] location[%s]",
                          request.instance_id.c_str(),
                          key,
                          name.c_str(),
                          loc->ToJsonString().c_str());
            RecordRetained(*request.free_busy->resources, "invalid_uri", 1);
            result = {EC_PARTIAL_OK, "invalid reclaim Location", true};
            if (group.locations.empty()) {
                groups.erase(name);
            }
            continue;
        }
        group.locations.emplace_back(key, loc);
        for (const auto &spec : uris) {
            if (spec.skip) {
                group.skipped.insert(spec.uri);
            }
        }
    }
    if (request.reservation) {
        std::set<std::string> active;
        for (const auto &entry : groups) {
            active.insert(entry.first);
        }
        request.reservation->RetainStorageBatches(active);
    }
    if (groups.empty()) {
        request.free_busy->ReleaseUnused();
        complete(std::move(result));
        return;
    }
    auto completion = std::make_shared<TaskCompletion>();
    completion->done = std::move(complete);
    completion->result = std::move(result);
    // The extra reference keeps the Future pending through release of unused memory.
    completion->remaining = groups.size() + 1;
    std::vector<std::shared_ptr<Batch>> batches;
    for (auto &[name, group] : groups) {
        const auto &targets = group.locations;
        auto batch = std::make_shared<Batch>(*this,
                                             data_storage_manager_,
                                             meta_manager_,
                                             indexer,
                                             request,
                                             name,
                                             std::move(group.storage_batch),
                                             completion);
        if (!batch->storage_batch->HasRequests()) {
            batch->Finish(batch->Cad(targets, indexer, false));
            continue;
        }
        uint64_t bytes = 0;
        for (const auto &[key, loc] : targets) {
            bytes += FreeBusyTask::IdBytes(loc->id(), request.instance_id.size(), name.size());
        }
        if (!request.free_busy->Take(bytes)) {
            RecordRetained(*request.free_busy->resources, "budget", targets.size());
            for (const auto &[key, loc] : targets) {
                KVCM_LOG_WARN("No retry ID budget instance[%s] key[%ld] location[%s] storage[%s]",
                              request.instance_id.c_str(),
                              key,
                              loc->id().c_str(),
                              name.c_str());
            }
            batch->Finish(false);
            continue;
        }
        batch->reserved = bytes;
        for (const auto &[key, loc] : targets) {
            batch->records[{key, loc->id()}];
        }
        batches.push_back(std::move(batch));
    }
    for (auto &batch : batches) {
        batch->deadline = Clock::now() + kRetryDeadline;
        try {
            const auto &group = groups.at(batch->name);
            batch->Send(group.locations, indexer, false, group.skipped);
        } catch (const std::exception &e) { batch->Stop("local_error", e.what()); } catch (...) {
            batch->Stop("local_error", "initial Free exception");
        }
    }
    request.free_busy->ReleaseUnused();
    completion->FinishBatch(true);
}

} // namespace kv_cache_manager
