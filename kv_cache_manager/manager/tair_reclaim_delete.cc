#include "kv_cache_manager/manager/tair_reclaim_delete.h"

#include <map>
#include <random>

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/data_storage/data_storage_manager.h"
#include "kv_cache_manager/manager/free_busy.h"
#include "kv_cache_manager/meta/meta_indexer.h"
#include "kv_cache_manager/meta/meta_indexer_manager.h"

namespace kv_cache_manager {
namespace {
using Clock = FreeBusyTask::Clock;
using Millis = std::chrono::milliseconds;
// Operational limit for starting another attempt, measured from the first physical Free.
constexpr auto kRetryDeadline = std::chrono::seconds(30);
constexpr uint64_t kMaxLocalDelayMs = 5000;

uint64_t Jitter(uint64_t base) {
    thread_local std::mt19937 random(std::random_device{}());
    const auto low = base == kMaxLocalDelayMs ? base * 4 / 5 : base;
    return std::uniform_int_distribution<uint64_t>(low, std::min(kMaxLocalDelayMs, base + base / 4))(random);
}

// Keep every spec of a Location together. A malformed sibling keeps the
// Location in DELETING, while other Locations on the same storage can proceed.
bool AppendGAs(
    const std::string &instance, int64_t key, const CacheLocation &loc, std::string &name, std::vector<uint64_t> &gas) {
    const size_t start = gas.size();
    bool valid = !loc.location_specs().empty();
    for (const auto &spec : loc.location_specs()) {
        DataStorageUri uri(spec.uri());
        if (name.empty()) {
            name = uri.GetHostName();
        }
        uint64_t ga;
        if (uri.GetProtocol() != kTairMempoolUriScheme || name.empty() || uri.GetHostName() != name ||
            !ParseReclaimGA(uri.GetPath(), ga)) {
            valid = false;
            KVCM_LOG_WARN("Invalid reclaim GA instance[%s] key[%ld] location[%s] uri[%s]",
                          instance.c_str(),
                          key,
                          loc.id().c_str(),
                          spec.uri().c_str());
        } else {
            gas.push_back(ga);
        }
    }
    if (!valid) {
        gas.resize(start);
        KVCM_LOG_WARN(
            "Retain invalid Tair Location instance[%s] key[%ld] location[%s]", instance.c_str(), key, loc.id().c_str());
    }
    return valid;
}

struct TaskCompletion {
    std::mutex mutex;
    size_t remaining = 0;
    PlanExecuteResult result;
    std::function<void(PlanExecuteResult)> done;

    void FinishBatch(bool ok) {
        std::lock_guard<std::mutex> lock(mutex);
        if (!ok && result.status == EC_OK) {
            result = {EC_PARTIAL_OK, "Tair reclaim Free or CAD failed", true};
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
          std::shared_ptr<TaskCompletion> completion)
        : executor(executor)
        , storage(std::move(storage))
        , meta(std::move(meta))
        , indexer(indexer)
        , task(request.free_busy)
        , instance(request.instance_id)
        , name(std::move(name))
        , completion(std::move(completion))
        , next_delay(task->initial_ms) {}

    SchedulePlanExecutor &executor;
    const std::shared_ptr<DataStorageManager> storage;
    const std::shared_ptr<MetaIndexerManager> meta;
    const std::weak_ptr<MetaIndexer> indexer;
    const std::shared_ptr<FreeBusyTask> task;
    const std::string instance, name;
    const std::shared_ptr<TaskCompletion> completion;
    std::weak_ptr<DataStorageBackend> backend;
    std::map<std::pair<int64_t, std::string>, std::vector<std::string>> records;
    uint64_t reserved = 0;
    uint64_t next_delay;
    Clock::time_point deadline;
    bool was_busy = false;
    bool has_invalid_target = false;
    bool holds_slot = false;

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
        completion->FinishBatch(ok && !has_invalid_target);
    }

    void Stop(const char *reason) {
        for (const auto &[id, uris] : records) {
            KVCM_LOG_WARN("Tair reclaim Free stopped: %s, instance[%s] key[%ld] location[%s] storage[%s]",
                          reason,
                          instance.c_str(),
                          id.first,
                          id.second.c_str(),
                          name.c_str());
        }
        Finish(false);
    }

    void Schedule(uint64_t wait_ms) {
        ReleaseSlot();
        const auto left = std::chrono::duration_cast<Millis>(deadline - Clock::now()).count();
        if (!task->Active() || left <= 0 || wait_ms >= static_cast<uint64_t>(left)) {
            Stop("cancelled or retry deadline reached");
            return;
        }
        auto self = shared_from_this();
        if (!executor.SubmitTask(
                ScheduleTaskClass::kReclaim,
                [self]() {
                    try {
                        self->Retry();
                    } catch (const std::exception &e) { self->Stop(e.what()); } catch (...) {
                        self->Stop("retry exception");
                    }
                },
                Millis(wait_ms),
                [self]() { self->Stop("executor stopped"); })) {
            Stop("retry submission rejected");
        }
    }

    bool ReadTargets(const std::shared_ptr<MetaIndexer> &idx, ReclaimFreeLocations &out, std::vector<uint64_t> &gas) {
        std::string expected_storage = name;
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
            return false;
        }
        auto it = records.begin();
        for (size_t i = 0; i < keys.size(); ++i) {
            if (read.error_codes[i] != EC_NOENT && read.error_codes[i] != EC_OK) {
                return false;
            }
            while (it != records.end() && it->first.first == keys[i]) {
                const auto found = maps[i].find(it->first.second);
                const auto loc = found == maps[i].end() ? nullptr : found->second;
                if (read.error_codes[i] == EC_NOENT || !loc || loc->status() != CacheLocationStatus::CLS_DELETING ||
                    !IsTairMempoolStorageType(loc->type()) || loc->location_specs().empty() ||
                    DataStorageUri(loc->location_specs().front().uri()).GetHostName() != name ||
                    (was_busy && !std::equal(it->second.begin(),
                                             it->second.end(),
                                             loc->location_specs().begin(),
                                             loc->location_specs().end(),
                                             [](const auto &uri, const auto &spec) { return uri == spec.uri(); }))) {
                    it = records.erase(it);
                    continue;
                }
                if (!AppendGAs(instance, keys[i], *loc, expected_storage, gas)) {
                    has_invalid_target = true;
                    it = records.erase(it);
                    continue;
                }
                out.emplace_back(keys[i], loc);
                ++it;
            }
        }
        return true;
    }

    void Retry() {
        if (!task->Active() || Clock::now() >= deadline) {
            Stop("cancelled or retry deadline reached");
            return;
        }
        if (!task->resources->TryStartRetry(executor.GetWorkerCount())) {
            Schedule(Jitter(task->initial_ms));
            return;
        }
        holds_slot = true;
        auto idx = indexer.lock();
        if (!idx || meta->GetMetaIndexer(instance) != idx) {
            Stop("instance changed");
            return;
        }
        ReclaimFreeLocations current;
        std::vector<uint64_t> gas;
        if (!ReadTargets(idx, current, gas)) {
            Stop("primary read failed");
            return;
        }
        if (current.empty()) {
            Finish(true);
            return;
        }
        Send(current, gas, idx, true);
    }

    bool Cad(const ReclaimFreeLocations &targets, const std::shared_ptr<MetaIndexer> &idx, bool retry) {
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

    void Send(const ReclaimFreeLocations &targets,
              const std::vector<uint64_t> &gas,
              const std::shared_ptr<MetaIndexer> &idx,
              bool retry) {
        uint64_t needed = 0;
        for (const auto &[key, loc] : targets) {
            needed += FreeBusyTask::RecordBytes(*loc, instance.size());
        }
        if (needed > reserved) {
            const auto extra = needed - reserved;
            if (!(retry ? task->resources->Reserve(extra) : task->Take(extra))) {
                if (!retry) {
                    backend = storage->GetDataStorageBackend(name);
                }
                Schedule(Jitter(task->initial_ms));
                return;
            }
            reserved = needed;
        }
        if (gas.empty() || !task->Active() || (retry && Clock::now() >= deadline)) {
            Stop("empty, cancelled or expired before Free");
            return;
        }

        if (was_busy) {
            ++task->resources->retry_attempts;
        }
        auto can_send = [&] { return task->Active() && (!retry || Clock::now() < deadline); };
        const auto result =
            storage->DeleteTairForReclaim("location_del_task_trace", name, gas, backend, retry, can_send);
        ReleaseSlot();
        if (result.status == TairReclaimFreeResult::kNotSent) {
            Stop("cancelled or expired before backend dispatch");
            return;
        }
        if (result.status == TairReclaimFreeResult::kBusy) {
            ++task->resources->busy_replies;
            const uint64_t wait = std::max(Jitter(next_delay), result.retry_after_ms);
            next_delay = std::min(next_delay * 2, kMaxLocalDelayMs);
            std::set<DataStorageType> types;
            for (const auto &[key, loc] : targets) {
                types.insert(loc->type());
                if (!was_busy) {
                    auto &uris = records.at({key, loc->id()});
                    uris.reserve(loc->location_specs().size());
                    for (const auto &spec : loc->location_specs()) {
                        uris.push_back(spec.uri());
                    }
                }
            }
            for (auto type : types) {
                task->Record(type, wait);
            }
            KVCM_LOG_DEBUG("Tair Free busy instance[%s] storage[%s] locations[%zu] gas[%zu] retry[%d] wait_ms[%lu]",
                           instance.c_str(),
                           name.c_str(),
                           targets.size(),
                           gas.size(),
                           retry,
                           wait);
            was_busy = true;
            Schedule(wait);
            return;
        }
        if (result.status == TairReclaimFreeResult::kTooLarge ||
            result.status == TairReclaimFreeResult::kBackendChanged) {
            KVCM_LOG_WARN(
                "Tair Free not sent storage[%s] gas[%zu] max_gas[%lu]", name.c_str(), gas.size(), result.max_gas);
            Stop(result.status == TairReclaimFreeResult::kTooLarge ? "FREE_BATCH_TOO_LARGE" : "backend changed");
            return;
        }

        const bool success = result.status == TairReclaimFreeResult::kSuccess;
        if (success) {
            std::set<DataStorageType> types;
            for (const auto &target : targets) {
                types.insert(target.second->type());
            }
            for (auto type : types) {
                task->Record(type, 0);
            }
        } else {
            for (const auto &[key, loc] : targets) {
                for (const auto &spec : loc->location_specs()) {
                    KVCM_LOG_WARN("Tair Free failed, possible leak instance[%s] key[%ld] location[%s] uri[%s]",
                                  instance.c_str(),
                                  key,
                                  loc->id().c_str(),
                                  spec.uri().c_str());
                }
            }
        }
        const auto cad_ok = Cad(targets, idx, retry);
        if (!cad_ok) {
            KVCM_LOG_WARN("Tair reclaim CAD failed instance[%s] storage[%s] locations[%zu]",
                          instance.c_str(),
                          name.c_str(),
                          targets.size());
        }
        Finish(success && cad_ok);
    }
};
} // namespace

void RunTairReclaimDelete(SchedulePlanExecutor &executor,
                          const std::shared_ptr<DataStorageManager> &storage,
                          const std::shared_ptr<MetaIndexerManager> &meta,
                          const std::shared_ptr<MetaIndexer> &indexer,
                          const CacheLocationDelRequest &request,
                          const ReclaimFreeLocations &locations,
                          PlanExecuteResult result,
                          std::function<void(PlanExecuteResult)> complete) {
    struct StorageGroup {
        ReclaimFreeLocations locations;
        std::vector<uint64_t> gas;
    };
    std::map<std::string, StorageGroup> groups;
    std::vector<uint64_t> location_gas;
    for (const auto &[key, loc] : locations) {
        std::string name;
        location_gas.clear();
        if (!AppendGAs(request.instance_id, key, *loc, name, location_gas)) {
            result = {EC_PARTIAL_OK, "invalid Tair Location", true};
            continue;
        }
        auto &group = groups[name];
        group.locations.emplace_back(key, loc);
        group.gas.insert(group.gas.end(), location_gas.begin(), location_gas.end());
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
    for (const auto &[name, group] : groups) {
        const auto &targets = group.locations;
        auto batch = std::make_shared<Batch>(executor, storage, meta, indexer, request, name, completion);
        uint64_t bytes = 0;
        for (const auto &[key, loc] : targets) {
            bytes += FreeBusyTask::IdBytes(loc->id(), request.instance_id.size(), name.size());
        }
        if (!request.free_busy->Take(bytes)) {
            for (const auto &[key, loc] : targets) {
                KVCM_LOG_WARN("No retry ID budget instance[%s] key[%ld] location[%s] storage[%s]",
                              request.instance_id.c_str(),
                              key,
                              loc->id().c_str(),
                              name.c_str());
            }
            completion->FinishBatch(false);
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
            batch->Send(group.locations, group.gas, indexer, false);
        } catch (const std::exception &e) { batch->Stop(e.what()); } catch (...) {
            batch->Stop("initial Free exception");
        }
    }
    request.free_busy->ReleaseUnused();
    completion->FinishBatch(true);
}

} // namespace kv_cache_manager
