#include "tools/kvcm_swarm/clients/v6d/deployment.h"

#include <algorithm>
#include <set>

#include "tools/kvcm_swarm/scenario/config_node.h"
#include "tools/kvcm_swarm/scenario/config_reader.h"

namespace kvcm_swarm {
namespace {

constexpr size_t kMaxEventsPerReport = 128;

meta::StorageType StorageTypeFromName(const std::string &name) {
    if (name == "hf3fs")
        return meta::ST_3FS;
    if (name == "vcns_hf3fs")
        return meta::ST_VCNS_3FS;
    if (name == "mooncake")
        return meta::ST_MOONCAKE;
    if (name == "pace")
        return meta::ST_TAIRMEMPOOL;
    if (name == "pace_ssd")
        return meta::ST_TAIRMEMPOOL_SSD;
    if (name == "file")
        return meta::ST_NFS;
    return meta::ST_UNSPECIFIED;
}

} // namespace

V6dDeployment::V6dDeployment(BehaviorSpec spec, V6dConfig config, RuntimeServices services)
    : spec_(std::move(spec))
    , config_(std::move(config))
    , services_(services)
    , checks_(expected_, services.evidence, config_.instance_id) {
    group_stats_.assign(config_.groups.size(), GroupWorkloadStats());
    processes_.reserve(config_.process_count);
    for (uint32_t index = 0; index < config_.process_count; ++index) {
        processes_.push_back(
            std::make_unique<V6dProcess>(*this, MakeProcessIdentity(spec_.id, config_, index), spec_.transport));
    }
    sessions_ = std::make_unique<SessionManager>(*this, *this);
}

V6dDeployment::~V6dDeployment() = default;

void V6dDeployment::SetStorageConfigs(const std::string &json) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!storage_configs_json_.empty()) {
        return;
    }
    storage_configs_json_ = json;
    // Cold-tier backends are derived from the registration response instead of
    // being hard-coded, so the cold lookup matches the actual deployment.
    std::string error;
    ConfigNode root = ConfigNode::Parse(json.empty() ? "[]" : json, &error);
    std::set<int> seen;
    if (root.IsArray()) {
        for (const ConfigNode &item : root.Items()) {
            std::string type_name;
            if (!item.Get("type").AsString(&type_name)) {
                continue;
            }
            const meta::StorageType type = StorageTypeFromName(type_name);
            if (type == meta::ST_UNSPECIFIED) {
                continue;
            }
            if (seen.insert(static_cast<int>(type)).second) {
                cold_backends_.push_back(type);
            }
        }
    }
    std::sort(cold_backends_.begin(), cold_backends_.end());
}

std::vector<meta::StorageType> V6dDeployment::cold_backends() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return cold_backends_;
}

Task<bool> V6dDeployment::Initialize(TimePoint deadline) {
    const TimePoint start = Now();
    // process 0 starts at the beginning of initialize; every later process adds
    // one sampled interval. `0` means all processes start at the same moment.
    Rng startup_rng = services_.seeds.MakeRng("v6d/" + spec_.id + "/process_startup");
    std::vector<TimePoint> planned_starts;
    TimePoint planned = start;
    for (uint32_t index = 0; index < config_.process_count; ++index) {
        if (index > 0) {
            planned += Sample(config_.process_startup_interval, startup_rng);
        }
        planned_starts.push_back(planned);
    }

    // Registrations may run concurrently, but the ready barrier requires every
    // process to have registered and completed NODE_REGISTER + HEARTBEAT.
    auto results = std::make_shared<std::vector<int>>(config_.process_count, -1);
    auto pending = std::make_shared<std::atomic<uint32_t>>(config_.process_count);
    auto done = std::make_shared<AsyncSlot<bool>>(services_.executor);
    for (uint32_t index = 0; index < config_.process_count; ++index) {
        V6dProcess *process = processes_[index].get();
        auto register_task = [](V6dProcess *proc,
                                TimePoint planned_start,
                                TimePoint register_deadline,
                                std::shared_ptr<std::vector<int>> out,
                                uint32_t slot,
                                std::shared_ptr<std::atomic<uint32_t>> remaining,
                                std::shared_ptr<AsyncSlot<bool>> completion) -> Task<> {
            const bool ok = co_await proc->Register(planned_start, register_deadline);
            (*out)[slot] = ok ? 1 : 0;
            if (remaining->fetch_sub(1, std::memory_order_acq_rel) == 1) {
                completion->Complete(true);
            }
            co_return;
        };
        register_task(process, planned_starts[index], deadline, results, index, pending, done)
            .via(&services_.executor)
            .start([](auto &&) {});
    }
    services_.executor.ScheduleAt(deadline, [done]() { done->Complete(false); });
    co_await *done;

    bool all_ok = true;
    for (uint32_t index = 0; index < config_.process_count; ++index) {
        if ((*results)[index] != 1) {
            all_ok = false;
            ++register_failures_;
        }
    }
    initialize_duration_ = Now() - start;
    if (!all_ok) {
        // Any process failing to register fails initialize.
        co_return false;
    }
    for (auto &process : processes_) {
        process->StartMaintenance();
    }
    co_return true;
}

void V6dDeployment::StartTraffic() { sessions_->Start(); }

std::vector<uint32_t> V6dDeployment::ReadyProcesses() const {
    std::vector<uint32_t> ready;
    ready.reserve(processes_.size());
    for (uint32_t index = 0; index < processes_.size(); ++index) {
        if (processes_[index]->ready()) {
            ready.push_back(index);
        }
    }
    return ready;
}

Task<bool> V6dDeployment::RunHotLookup(V6dProcess &process,
                                       const CacheGroupSpec &group,
                                       const std::vector<GroupObject> &objects,
                                       const std::vector<bool> &local,
                                       std::vector<bool> *hot_remote,
                                       std::vector<bool> *cold,
                                       TimePoint deadline) {
    if (objects.empty()) {
        co_return true;
    }
    std::vector<LookupItem> items;
    items.reserve(objects.size());
    meta::GetCacheLocationsByBackendRequest request;
    request.set_trace_id("swarm-lookup-hot-" + process.identity().process_id + "-" + group.group_id + "-" +
                         std::to_string(objects.front().block_key));
    request.set_instance_id(config_.instance_id);
    request.set_query_type(meta::QT_BATCH_GET);
    auto *masks = request.mutable_block_mask()->mutable_bool_masks();
    for (size_t i = 0; i < objects.size(); ++i) {
        request.add_block_keys(objects[i].block_key);
        // block_keys and location_spec_names always have equal length and
        // correspond item by item.
        request.add_location_spec_names(objects[i].spec_name);
        masks->add_values(local[i]);
        LookupItem item;
        item.block_key = objects[i].block_key;
        item.spec_name = objects[i].spec_name;
        item.object_key = objects[i].object_key;
        item.masked = local[i];
        items.push_back(std::move(item));
    }
    // One batch uses exactly one explicit selector; PREFIX and COVERAGE are
    // never mixed in the same call.
    auto *selector = request.add_backend_selectors();
    selector->set_backend_type(meta::ST_EVENT_REPORT_L2);
    const FullSelector effective_selector =
        group.kind == CacheGroupKind::kMamba ? FullSelector::kCoverage : *group.lookup_selector;
    selector->set_strategy(effective_selector == FullSelector::kPrefix ? meta::LSS_V6D_PREFIX : meta::LSS_V6D_COVERAGE);

    const LookupExpectation expectation =
        checks_.BeforeLookup(LookupTier::kHot, items, effective_selector, process.reporter());
    meta::GetCacheLocationsByBackendResponse response;
    const TimePoint planned_at = Now();
    const RpcResult result = co_await process.Lookup(request, &response, planned_at, deadline, turn_stop_.Token());
    std::vector<std::string> hot_hosts;
    std::vector<std::string> cold_uris;
    checks_.OnLookupResult(items, expectation, response, process.reporter(), result.ok, &hot_hosts, &cold_uris);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        ++turn_stats_.hot_lookup_batches;
        turn_stats_.hot_lookup_keys += objects.size();
        if (!result.ok) {
            // A lookup cancelled by the drain deadline is not a lookup failure.
            if (result.transport_error == TransportError::kCancelled) {
                ++turn_stats_.lookups_cancelled_at_drain;
            } else {
                ++turn_stats_.hot_lookup_failures;
            }
        }
    }
    if (!result.ok) {
        co_return false;
    }
    for (size_t i = 0; i < objects.size(); ++i) {
        if (!hot_hosts[i].empty()) {
            (*hot_remote)[i] = true;
        } else if (!cold_uris[i].empty()) {
            (*cold)[i] = true;
        }
    }
    co_return true;
}

Task<bool> V6dDeployment::RunColdLookup(V6dProcess &process,
                                        const std::vector<GroupObject> &objects,
                                        const std::vector<size_t> &indices,
                                        std::vector<bool> *cold,
                                        TimePoint deadline) {
    const std::vector<meta::StorageType> backends = cold_backends();
    if (indices.empty() || backends.empty()) {
        co_return true;
    }
    std::vector<LookupItem> items;
    meta::GetCacheLocationsByBackendRequest request;
    request.set_trace_id("swarm-lookup-cold-" + process.identity().process_id + "-" +
                         std::to_string(objects[indices.front()].block_key));
    request.set_instance_id(config_.instance_id);
    request.set_query_type(meta::QT_BATCH_GET);
    request.mutable_block_mask()->set_offset(0);
    for (const size_t index : indices) {
        request.add_block_keys(objects[index].block_key);
        request.add_location_spec_names(objects[index].spec_name);
        LookupItem item;
        item.block_key = objects[index].block_key;
        item.spec_name = objects[index].spec_name;
        item.object_key = objects[index].object_key;
        item.masked = false;
        items.push_back(std::move(item));
    }
    // The cold tier is selected with WEIGHTED_RANDOM over the backends the
    // registration response actually advertised.
    for (const meta::StorageType type : backends) {
        auto *selector = request.add_backend_selectors();
        selector->set_backend_type(type);
        selector->set_strategy(meta::LSS_WEIGHTED_RANDOM);
    }

    const LookupExpectation expectation =
        checks_.BeforeLookup(LookupTier::kCold, items, FullSelector::kCoverage, process.reporter());
    meta::GetCacheLocationsByBackendResponse response;
    const TimePoint planned_at = Now();
    const RpcResult result = co_await process.Lookup(request, &response, planned_at, deadline, turn_stop_.Token());
    std::vector<std::string> hot_hosts;
    std::vector<std::string> cold_uris;
    checks_.OnLookupResult(items, expectation, response, process.reporter(), result.ok, &hot_hosts, &cold_uris);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        ++turn_stats_.cold_lookup_batches;
        turn_stats_.cold_lookup_keys += indices.size();
        if (!result.ok) {
            if (result.transport_error == TransportError::kCancelled) {
                ++turn_stats_.lookups_cancelled_at_drain;
            } else {
                ++turn_stats_.cold_lookup_failures;
            }
        }
    }
    if (!result.ok) {
        co_return false;
    }
    for (size_t slot = 0; slot < indices.size(); ++slot) {
        if (!cold_uris[slot].empty()) {
            (*cold)[indices[slot]] = true;
        }
    }
    co_return true;
}

Task<bool> V6dDeployment::RunMambaCoverageLookup(V6dProcess &process,
                                                 std::vector<GroupLookupState> &states,
                                                 const std::vector<size_t> &mamba_state_indices,
                                                 uint64_t full_boundary,
                                                 TimePoint deadline) {
    // All Mamba candidates that actually exist within the Full reuse bound are
    // merged into a single COVERAGE batch, because keys and specs correspond
    // item by item.
    std::vector<LookupItem> items;
    std::vector<std::pair<size_t, size_t>> origin; // (state index, object index)
    meta::GetCacheLocationsByBackendRequest request;
    request.set_instance_id(config_.instance_id);
    request.set_query_type(meta::QT_BATCH_GET);
    auto *masks = request.mutable_block_mask()->mutable_bool_masks();
    for (const size_t state_index : mamba_state_indices) {
        GroupLookupState &state = states[state_index];
        for (size_t i = 0; i < state.objects.size(); ++i) {
            if (state.objects[i].boundary_tokens > full_boundary) {
                continue;
            }
            request.add_block_keys(state.objects[i].block_key);
            request.add_location_spec_names(state.objects[i].spec_name);
            masks->add_values(state.local[i]);
            LookupItem item;
            item.block_key = state.objects[i].block_key;
            item.spec_name = state.objects[i].spec_name;
            item.object_key = state.objects[i].object_key;
            item.masked = state.local[i];
            items.push_back(std::move(item));
            origin.emplace_back(state_index, i);
        }
    }
    if (items.empty()) {
        co_return true;
    }
    request.set_trace_id("swarm-lookup-mamba-" + process.identity().process_id + "-" +
                         std::to_string(items.front().block_key));
    auto *selector = request.add_backend_selectors();
    selector->set_backend_type(meta::ST_EVENT_REPORT_L2);
    selector->set_strategy(meta::LSS_V6D_COVERAGE);

    const LookupExpectation expectation =
        checks_.BeforeLookup(LookupTier::kHot, items, FullSelector::kCoverage, process.reporter());
    meta::GetCacheLocationsByBackendResponse response;
    const TimePoint planned_at = Now();
    const RpcResult result = co_await process.Lookup(request, &response, planned_at, deadline, turn_stop_.Token());
    std::vector<std::string> hot_hosts;
    std::vector<std::string> cold_uris;
    checks_.OnLookupResult(items, expectation, response, process.reporter(), result.ok, &hot_hosts, &cold_uris);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        ++turn_stats_.mamba_coverage_batches;
        turn_stats_.mamba_candidates += items.size();
        ++turn_stats_.hot_lookup_batches;
        turn_stats_.hot_lookup_keys += items.size();
        if (!result.ok) {
            if (result.transport_error == TransportError::kCancelled) {
                ++turn_stats_.lookups_cancelled_at_drain;
            } else {
                ++turn_stats_.hot_lookup_failures;
            }
        }
    }
    if (!result.ok) {
        co_return false;
    }
    std::vector<size_t> unresolved_positions;
    for (size_t slot = 0; slot < origin.size(); ++slot) {
        GroupLookupState &state = states[origin[slot].first];
        const size_t object_index = origin[slot].second;
        if (!hot_hosts[slot].empty()) {
            state.hot_remote[object_index] = true;
        } else if (!cold_uris[slot].empty()) {
            state.cold[object_index] = true;
        } else if (!items[slot].masked) {
            unresolved_positions.push_back(slot);
        }
    }
    // Unresolved Mamba candidates fall back to the cold tier.
    if (!unresolved_positions.empty()) {
        std::vector<GroupObject> flattened;
        std::vector<size_t> flattened_indices;
        for (const size_t slot : unresolved_positions) {
            flattened.push_back(states[origin[slot].first].objects[origin[slot].second]);
            flattened_indices.push_back(flattened.size() - 1);
        }
        std::vector<bool> cold_flags(flattened.size(), false);
        co_await RunColdLookup(process, flattened, flattened_indices, &cold_flags, deadline);
        for (size_t i = 0; i < unresolved_positions.size(); ++i) {
            if (cold_flags[i]) {
                const auto &pair = origin[unresolved_positions[i]];
                states[pair.first].cold[pair.second] = true;
            }
        }
    }
    co_return true;
}

Task<bool>
V6dDeployment::RunTurn(SessionId session_id, uint32_t process_index, SessionWorkload &workload, TimePoint deadline) {
    active_turns_.fetch_add(1, std::memory_order_release);
    struct TurnGuard {
        std::atomic<uint32_t> *counter;
        ~TurnGuard() { counter->fetch_sub(1, std::memory_order_release); }
    } turn_guard{&active_turns_};

    V6dProcess &process = *processes_[process_index];
    // Reserve this turn's actual object bytes before taking any cache leases.
    // Concurrent turns use a simple byte sum; equal object keys are not
    // deducted because the first implementation deliberately avoids a
    // cross-turn sharing model.
    const uint64_t working_set_bytes = workload.WorkingSetBytes();
    AsyncCapacityBudget::Guard turn_capacity =
        co_await process.AcquireTurnCapacity(working_set_bytes, deadline, turn_stop_.Token());
    if (!turn_capacity.valid()) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (turn_stop_.StopRequested()) {
            ++turn_stats_.turns_cancelled_at_drain;
        } else {
            ++turn_stats_.turn_capacity_timeouts;
            services_.admission.MarkSaturated("process_turn_capacity_exhausted");
        }
        co_return true;
    }
    // Leases live only for this turn; the TurnContext destructor releases them.
    std::vector<LocalLease> leases;

    std::vector<GroupLookupState> states;
    states.reserve(workload.groups().size());
    uint64_t objects_considered = 0;
    uint64_t local_hits = 0;
    for (const auto &keyspace : workload.groups()) {
        GroupLookupState state;
        state.group = keyspace.group();
        state.objects = keyspace.objects();
        state.local.assign(state.objects.size(), false);
        state.hot_remote.assign(state.objects.size(), false);
        state.cold.assign(state.objects.size(), false);
        for (size_t i = 0; i < state.objects.size(); ++i) {
            // The process cache is the only source of truth for local hits.
            auto lease = process.cache().Acquire(state.objects[i].object_key);
            if (lease.has_value()) {
                state.local[i] = true;
                leases.push_back(std::move(*lease));
                ++local_hits;
            }
        }
        objects_considered += state.objects.size();
        states.push_back(std::move(state));
    }

    // ---- Full Attention groups: one independent batch per group ----
    std::vector<size_t> full_state_indices;
    std::vector<size_t> mamba_state_indices;
    for (size_t i = 0; i < states.size(); ++i) {
        if (states[i].group->kind == CacheGroupKind::kFullAttention) {
            full_state_indices.push_back(i);
        } else {
            mamba_state_indices.push_back(i);
        }
    }

    for (const size_t index : full_state_indices) {
        GroupLookupState &state = states[index];
        co_await RunHotLookup(
            process, *state.group, state.objects, state.local, &state.hot_remote, &state.cold, deadline);
        std::vector<size_t> unresolved;
        for (size_t i = 0; i < state.objects.size(); ++i) {
            if (!state.local[i] && !state.hot_remote[i] && !state.cold[i]) {
                unresolved.push_back(i);
            }
        }
        co_await RunColdLookup(process, state.objects, unresolved, &state.cold, deadline);
    }

    // Full reuse boundary: the contiguous prefix of available blocks, per
    // group, intersected across all Full groups.
    uint64_t full_boundary = UINT64_MAX;
    for (const size_t index : full_state_indices) {
        const GroupLookupState &state = states[index];
        uint64_t boundary = 0;
        uint32_t expected_block = 0;
        for (size_t i = 0; i < state.objects.size(); ++i) {
            if (state.objects[i].block_index != expected_block) {
                break;
            }
            if (!state.local[i] && !state.hot_remote[i] && !state.cold[i]) {
                break;
            }
            boundary = state.objects[i].boundary_tokens;
            ++expected_block;
        }
        full_boundary = std::min(full_boundary, boundary);
    }
    if (full_boundary == UINT64_MAX) {
        full_boundary = 0;
    }

    // ---- Mamba groups: a single merged COVERAGE batch within the bound ----
    if (!mamba_state_indices.empty() && full_boundary > 0) {
        co_await RunMambaCoverageLookup(process, states, mamba_state_indices, full_boundary, deadline);
    }

    // Final reusable boundary: the intersection of every group's result. For a
    // Mamba group only the last state block matters, so the group's boundary is
    // the largest available block boundary within the Full bound.
    uint64_t reusable_boundary = full_boundary;
    for (const size_t index : mamba_state_indices) {
        const GroupLookupState &state = states[index];
        uint64_t group_boundary = 0;
        for (size_t i = 0; i < state.objects.size(); ++i) {
            if (state.objects[i].boundary_tokens > full_boundary) {
                break;
            }
            if (state.local[i] || state.hot_remote[i] || state.cold[i]) {
                group_boundary = std::max(group_boundary, state.objects[i].boundary_tokens);
            }
        }
        reusable_boundary = std::min(reusable_boundary, group_boundary);
    }

    // ---- materialise reusable remote/cold hits, compute + seal the rest ----
    std::vector<GroupObject> materialized;
    std::vector<GroupObject> sealed;
    uint64_t remote_hot_hits = 0;
    uint64_t cold_hits = 0;
    for (size_t state_index = 0; state_index < states.size(); ++state_index) {
        GroupLookupState &state = states[state_index];
        for (size_t i = 0; i < state.objects.size(); ++i) {
            if (state.local[i]) {
                continue;
            }
            const bool within_bound = state.objects[i].boundary_tokens <= reusable_boundary;
            if (within_bound && (state.hot_remote[i] || state.cold[i])) {
                if (state.hot_remote[i]) {
                    ++remote_hot_hits;
                } else {
                    ++cold_hits;
                }
                materialized.push_back(state.objects[i]);
            } else {
                sealed.push_back(state.objects[i]);
            }
        }
    }

    // Insert materialised and sealed objects, waiting only for the capacity
    // actually needed. Waiting here is cache backpressure, not RPC latency.
    uint64_t insert_failed = 0;
    uint64_t insert_skipped_evicting = 0;
    uint64_t insert_cancelled = 0;
    std::vector<GroupObject> added;
    added.reserve(materialized.size() + sealed.size());
    for (const auto *batch : {&materialized, &sealed}) {
        for (const auto &object : *batch) {
            InsertOutcome outcome = InsertOutcome::kInserted;
            LocalLease lease =
                co_await process.cache().ReserveAndInsert(object, deadline, turn_stop_.Token(), &outcome);
            if (!lease.valid()) {
                if (outcome == InsertOutcome::kSkippedEvicting) {
                    ++insert_skipped_evicting;
                } else if (outcome == InsertOutcome::kCancelled) {
                    ++insert_cancelled;
                } else {
                    ++insert_failed;
                }
                continue;
            }
            leases.push_back(std::move(lease));
            added.push_back(object);
        }
    }

    // BLOCK_ADD for the objects that newly entered this process's cache. The
    // batch boundary is this turn's materialisation/seal batch, chunked only to
    // bound a single request; all keys are distinct so no single-key lifecycle
    // order is broken.
    uint64_t add_batches = 0;
    for (size_t offset = 0; offset < added.size(); offset += kMaxEventsPerReport) {
        const size_t end = std::min(added.size(), offset + kMaxEventsPerReport);
        std::vector<GroupObject> chunk(added.begin() + static_cast<long>(offset),
                                       added.begin() + static_cast<long>(end));
        co_await process.ReportBlockAdd(chunk);
        ++add_batches;
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        ++turn_stats_.turns;
        turn_stats_.objects_considered += objects_considered;
        turn_stats_.local_hits += local_hits;
        turn_stats_.remote_hot_hits += remote_hot_hits;
        turn_stats_.cold_hits += cold_hits;
        turn_stats_.materialized += materialized.size();
        turn_stats_.sealed += sealed.size();
        turn_stats_.insert_failed_backpressure += insert_failed;
        turn_stats_.insert_skipped_evicting += insert_skipped_evicting;
        turn_stats_.insert_cancelled_at_drain += insert_cancelled;
        turn_stats_.block_add_batches += add_batches;
        turn_stats_.reusable_tokens_total += reusable_boundary;
        turn_stats_.context_tokens_total += workload.token_count();
        for (size_t state_index = 0; state_index < states.size() && state_index < group_stats_.size(); ++state_index) {
            const GroupLookupState &state = states[state_index];
            GroupWorkloadStats &stats = group_stats_[state_index];
            stats.objects += state.objects.size();
            for (size_t i = 0; i < state.objects.size(); ++i) {
                if (state.local[i]) {
                    ++stats.local_hits;
                } else if (state.hot_remote[i]) {
                    ++stats.remote_hot_hits;
                } else if (state.cold[i]) {
                    ++stats.cold_hits;
                } else {
                    ++stats.sealed;
                }
            }
        }
    }
    (void)session_id;
    // Leases are released here, at the end of the turn, without exception.
    co_return true;
}

Task<> V6dDeployment::Drain(TimePoint deadline) {
    if (drained_.exchange(true)) {
        // Drain must be idempotent.
        co_return;
    }
    // 1. stop session admission and new turns
    sessions_->StopAdmission();
    // 2. bounded wait for the turns that already started
    const TimePoint now = Now();
    const TimePoint turn_window = now + (deadline > now ? (deadline - now) / 2 : Duration::zero());
    co_await sessions_->DrainTurns(turn_window);
    // Then cancel whatever is still running, so every short lease is released
    // before the shutdown flush selects victims.
    turn_stop_.RequestStop();
    co_await sessions_->DrainTurns(std::min(deadline, Now() + std::chrono::seconds(3)));
    // 3-5. finish started evictions, shutdown flush, then HOST_DOWN per process
    for (auto &process : processes_) {
        co_await process->Drain(deadline);
    }
    co_return;
}

bool V6dDeployment::Quiesced() const {
    if (active_turns_.load(std::memory_order_acquire) != 0) {
        return false;
    }
    if (!sessions_->quiesced()) {
        return false;
    }
    for (const auto &process : processes_) {
        if (!process->quiesced()) {
            return false;
        }
    }
    return true;
}

std::vector<InvariantObservation> V6dDeployment::Invariants() const {
    return checks_.Snapshot(std::string(TypeName()));
}

void V6dDeployment::WriteReport(JsonWriter &writer) const {
    TurnStats turns;
    std::vector<GroupWorkloadStats> groups;
    std::string storage_configs;
    std::vector<meta::StorageType> backends;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        turns = turn_stats_;
        groups = group_stats_;
        storage_configs = storage_configs_json_;
        backends = cold_backends_;
    }
    const SessionStats sessions = sessions_->Stats();
    const ExpectedLocationsStats expected = expected_.Stats();

    writer.BeginObject();
    writer.KeyString("type", std::string(TypeName()));
    writer.KeyString("instance_group", config_.instance_group);
    writer.KeyString("instance_id", config_.instance_id);
    writer.KeyUint("process_count", processes_.size());
    writer.KeyDouble("initialize_ms", ToMillis(initialize_duration_));
    writer.KeyUint("register_failures", register_failures_);

    writer.Key("sessions");
    writer.BeginObject();
    writer.KeyUint("planned_arrivals", sessions.planned_arrivals);
    writer.KeyUint("admitted", sessions.admitted);
    writer.KeyUint("rejected_admission", sessions.rejected_admission);
    writer.KeyUint("active_current", sessions.active_current);
    writer.KeyUint("active_peak", sessions.active_peak);
    writer.KeyUint("completed", sessions.completed);
    writer.KeyUint("aborted", sessions.aborted);
    writer.KeyUint("turns_started", sessions.turns_started);
    writer.KeyUint("turns_completed", sessions.turns_completed);
    writer.KeyUint("skipped_slots", sessions.skipped_slots);
    writer.KeyUint("migrations", sessions.migrations);
    writer.KeyUint("affinity_retained", sessions.affinity_retained);
    writer.KeyUint("shared_prefix_sessions", sessions.shared_prefix_sessions);
    writer.KeyUint("no_ready_process", sessions.no_ready_process);
    writer.Key("admitted_per_class");
    writer.BeginObject();
    for (size_t i = 0; i < sessions.admitted_per_class.size() && i < config_.session_classes.size(); ++i) {
        writer.KeyUint(config_.session_classes[i].name, sessions.admitted_per_class[i]);
    }
    writer.EndObject();
    writer.Key("turn_latency_ms");
    writer.BeginObject();
    writer.KeyUint("count", sessions.turn_latency_ms.count());
    writer.KeyDouble("mean", sessions.turn_latency_ms.mean_ms());
    writer.KeyDouble("p50", sessions.turn_latency_ms.Quantile(0.5));
    writer.KeyDouble("p99", sessions.turn_latency_ms.Quantile(0.99));
    writer.KeyDouble("max", sessions.turn_latency_ms.max_ms());
    writer.EndObject();
    writer.Key("turn_lag_ms");
    writer.BeginObject();
    writer.KeyUint("count", sessions.turn_lag_ms.count());
    writer.KeyDouble("mean", sessions.turn_lag_ms.mean_ms());
    writer.KeyDouble("p99", sessions.turn_lag_ms.Quantile(0.99));
    writer.KeyDouble("max", sessions.turn_lag_ms.max_ms());
    writer.EndObject();
    writer.Key("lifetime_ms");
    writer.BeginObject();
    writer.KeyUint("count", sessions.session_lifetime_ms.count());
    writer.KeyDouble("mean", sessions.session_lifetime_ms.mean_ms());
    writer.KeyDouble("max", sessions.session_lifetime_ms.max_ms());
    writer.EndObject();
    writer.EndObject();

    writer.Key("turns");
    writer.BeginObject();
    writer.KeyUint("turns", turns.turns);
    writer.KeyUint("objects_considered", turns.objects_considered);
    writer.KeyUint("local_hits", turns.local_hits);
    writer.KeyUint("remote_hot_hits", turns.remote_hot_hits);
    writer.KeyUint("cold_hits", turns.cold_hits);
    writer.KeyUint("materialized", turns.materialized);
    writer.KeyUint("sealed", turns.sealed);
    writer.KeyUint("insert_failed_backpressure", turns.insert_failed_backpressure);
    writer.KeyUint("insert_skipped_evicting", turns.insert_skipped_evicting);
    writer.KeyUint("turn_capacity_timeouts", turns.turn_capacity_timeouts);
    writer.KeyUint("insert_cancelled_at_drain", turns.insert_cancelled_at_drain);
    writer.KeyUint("lookups_cancelled_at_drain", turns.lookups_cancelled_at_drain);
    writer.KeyUint("turns_cancelled_at_drain", turns.turns_cancelled_at_drain);
    writer.KeyUint("hot_lookup_batches", turns.hot_lookup_batches);
    writer.KeyUint("hot_lookup_keys", turns.hot_lookup_keys);
    writer.KeyUint("hot_lookup_failures", turns.hot_lookup_failures);
    writer.KeyUint("cold_lookup_batches", turns.cold_lookup_batches);
    writer.KeyUint("cold_lookup_keys", turns.cold_lookup_keys);
    writer.KeyUint("cold_lookup_failures", turns.cold_lookup_failures);
    writer.KeyUint("mamba_coverage_batches", turns.mamba_coverage_batches);
    writer.KeyUint("mamba_candidates", turns.mamba_candidates);
    writer.KeyUint("block_add_batches", turns.block_add_batches);
    writer.KeyUint("reusable_tokens_total", turns.reusable_tokens_total);
    writer.KeyUint("context_tokens_total", turns.context_tokens_total);
    writer.EndObject();

    writer.Key("groups");
    writer.BeginArray();
    for (size_t i = 0; i < groups.size() && i < config_.groups.size(); ++i) {
        writer.BeginObject();
        writer.KeyString("id", config_.groups[i].group_id);
        writer.KeyString("kind", CacheGroupKindName(config_.groups[i].kind));
        writer.KeyUint("block_size", config_.groups[i].block_size_tokens);
        writer.KeyUint("object_size", config_.groups[i].object_size_bytes);
        writer.KeyString("spec_name", config_.groups[i].spec_name);
        writer.KeyString("lookup_selector",
                         config_.groups[i].kind == CacheGroupKind::kMamba
                             ? "coverage"
                             : FullSelectorName(*config_.groups[i].lookup_selector));
        writer.KeyUint("objects", groups[i].objects);
        writer.KeyUint("local_hits", groups[i].local_hits);
        writer.KeyUint("remote_hot_hits", groups[i].remote_hot_hits);
        writer.KeyUint("cold_hits", groups[i].cold_hits);
        writer.KeyUint("sealed", groups[i].sealed);
        writer.EndObject();
    }
    writer.EndArray();

    writer.Key("processes");
    writer.BeginArray();
    for (const auto &process : processes_) {
        process->WriteReport(writer);
    }
    writer.EndArray();

    writer.Key("expected_locations");
    writer.BeginObject();
    writer.KeyUint("hot_pending_create", expected.hot_pending_create);
    writer.KeyUint("hot_confirmed", expected.hot_confirmed);
    writer.KeyUint("hot_pending_delete", expected.hot_pending_delete);
    writer.KeyUint("hot_unknown", expected.hot_unknown);
    writer.KeyUint("hot_removed", expected.hot_removed);
    writer.KeyUint("cold_pending_create", expected.cold_pending_create);
    writer.KeyUint("cold_confirmed", expected.cold_confirmed);
    writer.KeyUint("cold_unknown", expected.cold_unknown);
    writer.KeyUint("cold_removed", expected.cold_removed);
    writer.KeyUint("cold_confirmed_bytes", expected.cold_confirmed_bytes);
    writer.Key("unresolved_preview");
    writer.BeginArray();
    for (const auto &entry : expected_.UnresolvedSummary(16)) {
        writer.String(entry);
    }
    writer.EndArray();
    writer.EndObject();

    writer.Key("cold_backends");
    writer.BeginArray();
    for (const meta::StorageType type : backends) {
        writer.String(meta::StorageType_Name(type));
    }
    writer.EndArray();
    writer.KeyString("storage_configs", storage_configs);
    writer.EndObject();
}

bool V6dDeployment::WriteCacheReport(JsonWriter &writer) const {
    uint64_t capacity = 0;
    uint64_t used = 0;
    uint64_t peak_used = 0;
    uint64_t entries = 0;
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t inserts = 0;
    uint64_t removed = 0;
    uint64_t victims = 0;
    uint64_t backpressure_waits = 0;
    uint64_t backpressure_timeouts = 0;
    double backpressure_ms = 0.0;
    double backpressure_ms_max = 0.0;
    uint64_t no_victim_waits = 0;
    uint64_t skipped_evicting = 0;
    uint64_t insert_cancelled = 0;
    uint64_t residual_objects = 0;
    uint64_t residual_bytes = 0;
    uint64_t cold_allocations = 0;
    uint64_t cold_bytes = 0;
    for (const auto &process : processes_) {
        const LocalCacheStats stats = process->cache().Stats();
        capacity += stats.capacity_bytes;
        used += stats.used_bytes;
        peak_used += stats.peak_used_bytes;
        entries += stats.entries;
        hits += stats.local_hits;
        misses += stats.local_misses;
        inserts += stats.inserts;
        removed += stats.removed;
        victims += stats.victims_selected;
        backpressure_waits += stats.backpressure_waits;
        backpressure_timeouts += stats.backpressure_timeouts;
        backpressure_ms += static_cast<double>(stats.backpressure_wait_ns) / 1e6;
        backpressure_ms_max = std::max(backpressure_ms_max, static_cast<double>(stats.backpressure_wait_ns_max) / 1e6);
        no_victim_waits += stats.no_victim_waits;
        skipped_evicting += stats.insert_skipped_evicting;
        insert_cancelled += stats.insert_cancelled;
        residual_objects += stats.entries;
        residual_bytes += stats.used_bytes;
        const EvictionStats eviction = process->eviction_stats();
        cold_allocations += eviction.cold_allocations_confirmed;
        cold_bytes += eviction.cold_allocation_bytes;
    }
    writer.BeginObject();
    writer.KeyUint("capacity_bytes_total", capacity);
    writer.KeyUint("used_bytes_total", used);
    writer.KeyUint("peak_used_bytes_sum", peak_used);
    writer.KeyUint("entries_total", entries);
    writer.KeyUint("local_hits", hits);
    writer.KeyUint("local_misses", misses);
    writer.KeyUint("inserts", inserts);
    writer.KeyUint("removed", removed);
    writer.KeyUint("victims_selected", victims);
    writer.KeyUint("backpressure_waits", backpressure_waits);
    writer.KeyUint("backpressure_timeouts", backpressure_timeouts);
    writer.KeyDouble("backpressure_wait_ms_total", backpressure_ms);
    writer.KeyDouble("backpressure_wait_ms_max", backpressure_ms_max);
    writer.KeyUint("no_victim_waits", no_victim_waits);
    writer.KeyUint("insert_skipped_evicting", skipped_evicting);
    writer.KeyUint("insert_cancelled", insert_cancelled);
    writer.KeyUint("residual_local_objects", residual_objects);
    writer.KeyUint("residual_local_bytes", residual_bytes);
    writer.KeyUint("cold_allocations_confirmed", cold_allocations);
    writer.KeyUint("cold_allocation_bytes", cold_bytes);
    writer.EndObject();
    return true;
}

bool V6dDeployment::WriteWorkloadShape(JsonWriter &writer) const {
    TurnStats turns;
    std::vector<GroupWorkloadStats> groups;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        turns = turn_stats_;
        groups = group_stats_;
    }
    const SessionStats sessions = sessions_->Stats();
    writer.BeginObject();
    writer.KeyUint("turns", turns.turns);
    writer.KeyDouble(
        "mean_context_tokens",
        turns.turns == 0 ? 0.0 : static_cast<double>(turns.context_tokens_total) / static_cast<double>(turns.turns));
    writer.KeyDouble(
        "mean_reusable_tokens",
        turns.turns == 0 ? 0.0 : static_cast<double>(turns.reusable_tokens_total) / static_cast<double>(turns.turns));
    writer.KeyDouble("mean_lookup_keys_per_batch",
                     turns.hot_lookup_batches == 0
                         ? 0.0
                         : static_cast<double>(turns.hot_lookup_keys) / static_cast<double>(turns.hot_lookup_batches));
    writer.KeyDouble("local_hit_ratio",
                     turns.objects_considered == 0
                         ? 0.0
                         : static_cast<double>(turns.local_hits) / static_cast<double>(turns.objects_considered));
    writer.KeyDouble("remote_hot_hit_ratio",
                     turns.objects_considered == 0
                         ? 0.0
                         : static_cast<double>(turns.remote_hot_hits) / static_cast<double>(turns.objects_considered));
    writer.KeyDouble("cold_hit_ratio",
                     turns.objects_considered == 0
                         ? 0.0
                         : static_cast<double>(turns.cold_hits) / static_cast<double>(turns.objects_considered));
    writer.KeyUint("shared_prefix_sessions", sessions.shared_prefix_sessions);
    writer.KeyUint("migrations", sessions.migrations);
    writer.KeyUint("worst_case_context_tokens", config_.WorstCaseContextTokens());
    writer.KeyUint("worst_case_turn_working_set_bytes", config_.WorstCaseTurnWorkingSetBytes());
    writer.KeyUint("local_cache_capacity_bytes", config_.local_cache_capacity_bytes);
    writer.Key("per_group");
    writer.BeginArray();
    for (size_t i = 0; i < groups.size() && i < config_.groups.size(); ++i) {
        writer.BeginObject();
        writer.KeyString("id", config_.groups[i].group_id);
        writer.KeyString("kind", CacheGroupKindName(config_.groups[i].kind));
        writer.KeyUint("objects", groups[i].objects);
        writer.KeyUint("local_hits", groups[i].local_hits);
        writer.KeyUint("remote_hot_hits", groups[i].remote_hot_hits);
        writer.KeyUint("cold_hits", groups[i].cold_hits);
        writer.KeyUint("sealed", groups[i].sealed);
        writer.EndObject();
    }
    writer.EndArray();
    writer.EndObject();
    return true;
}

bool V6dDeployment::WriteCleanupReport(JsonWriter &writer) const {
    uint64_t flush_batches = 0;
    uint64_t flush_objects = 0;
    uint64_t host_down_attempted = 0;
    uint64_t host_down_succeeded = 0;
    uint64_t unflushed_objects = 0;
    uint64_t unflushed_bytes = 0;
    uint64_t cold_allocations = 0;
    uint64_t cold_bytes = 0;
    uint64_t protected_uncertain = 0;
    for (const auto &process : processes_) {
        const EvictionStats eviction = process->eviction_stats();
        flush_batches += eviction.shutdown_flush_batches;
        flush_objects += eviction.shutdown_flush_objects;
        cold_allocations += eviction.cold_allocations_confirmed;
        cold_bytes += eviction.cold_allocation_bytes;
        protected_uncertain += eviction.protected_uncertain;
        const ReporterStats reporter = process->reporter_stats();
        host_down_attempted += reporter.host_down_attempted;
        host_down_succeeded += reporter.host_down_succeeded;
        const LocalCacheStats cache = process->cache().Stats();
        unflushed_objects += cache.entries;
        unflushed_bytes += cache.used_bytes;
    }
    writer.BeginObject();
    writer.KeyUint("shutdown_flush_batches", flush_batches);
    writer.KeyUint("shutdown_flush_objects", flush_objects);
    writer.KeyUint("unflushed_local_objects", unflushed_objects);
    writer.KeyUint("unflushed_local_bytes", unflushed_bytes);
    writer.KeyUint("host_down_attempted", host_down_attempted);
    writer.KeyUint("host_down_succeeded", host_down_succeeded);
    writer.KeyUint("confirmed_cold_allocations", cold_allocations);
    writer.KeyUint("confirmed_cold_allocation_bytes", cold_bytes);
    writer.KeyUint("protected_uncertain_operations", protected_uncertain);
    writer.KeyBool("remove_cache_called", false);
    writer.KeyString("cold_allocation_policy",
                     "normal shutdown never deletes cold allocations; they are left to KVCM/storage reclamation or "
                     "to the isolated test environment teardown");
    writer.EndObject();
    return true;
}

void V6dDeployment::WriteEffectiveConfig(JsonWriter &writer) const {
    writer.BeginObject();
    writer.KeyUint("process_count", config_.process_count);
    writer.Key("process_startup_interval");
    writer.BeginObject();
    writer.KeyString("min", FormatDuration(config_.process_startup_interval.min));
    writer.KeyString("max", FormatDuration(config_.process_startup_interval.max));
    writer.EndObject();
    writer.KeyString("instance_group", config_.instance_group);
    writer.KeyString("instance_id", config_.instance_id);
    writer.KeyString("process_host_ip", config_.process_host_ip);
    writer.KeyUint("process_port_base", config_.process_port_base);
    writer.Key("local_cache");
    writer.BeginObject();
    writer.KeyUint("capacity_bytes", config_.local_cache_capacity_bytes);
    writer.KeyUint("worst_case_turn_working_set_bytes", config_.WorstCaseTurnWorkingSetBytes());
    writer.EndObject();
    writer.Key("session_arrival");
    writer.BeginObject();
    writer.KeyDouble("rate", config_.session_arrival_rate);
    writer.KeyString("mode", ArrivalModeName(config_.arrival_mode));
    writer.EndObject();
    writer.KeyDouble("session_affinity", config_.session_affinity);
    writer.Key("limits");
    writer.BeginObject();
    writer.KeyUint("max_active_sessions", config_.max_active_sessions);
    writer.EndObject();
    writer.KeyString("heartbeat_interval", FormatDuration(config_.heartbeat_interval));
    writer.KeyInt("min_replica_count", config_.min_replica_count);
    writer.KeyString("leader_poll_interval", FormatDuration(config_.leader_poll_interval));
    writer.KeyString("write_timeout", FormatDuration(config_.write_timeout));
    writer.KeyString("turn_deadline", FormatDuration(config_.turn_deadline));
    writer.KeyString("rpc_timeout", FormatDuration(config_.rpc_timeout));
    writer.KeyString("host_down_timeout", FormatDuration(config_.host_down_timeout));
    writer.KeyUint("eviction_batch_size", config_.eviction_batch_size);
    writer.Key("shared_prefix_pool");
    writer.BeginObject();
    writer.KeyUint("root_count", config_.shared_prefix_pool.root_count);
    writer.Key("prefix_tokens");
    writer.BeginObject();
    writer.KeyUint("min", config_.shared_prefix_pool.prefix_tokens.min);
    writer.KeyUint("max", config_.shared_prefix_pool.prefix_tokens.max);
    writer.EndObject();
    writer.EndObject();
    writer.Key("groups");
    writer.BeginArray();
    for (const auto &group : config_.groups) {
        writer.BeginObject();
        writer.KeyString("id", group.group_id);
        writer.KeyString("kind", CacheGroupKindName(group.kind));
        writer.KeyUint("block_size", group.block_size_tokens);
        writer.KeyUint("object_size", group.object_size_bytes);
        writer.KeyString("spec_name", group.spec_name);
        if (group.kind == CacheGroupKind::kFullAttention) {
            writer.KeyString("lookup_selector", FullSelectorName(*group.lookup_selector));
        } else {
            writer.KeyDouble("key_presence_rate", group.key_presence_rate);
        }
        writer.EndObject();
    }
    writer.EndArray();
    writer.Key("session_classes");
    writer.BeginArray();
    for (const auto &session_class : config_.session_classes) {
        writer.BeginObject();
        writer.KeyString("name", session_class.name);
        writer.KeyDouble("weight", session_class.weight);
        writer.Key("turns");
        writer.BeginObject();
        writer.KeyUint("min", session_class.turns.min);
        writer.KeyUint("max", session_class.turns.max);
        writer.EndObject();
        writer.Key("turn_interval");
        writer.BeginObject();
        writer.KeyString("min", FormatDuration(session_class.turn_interval.min));
        writer.KeyString("max", FormatDuration(session_class.turn_interval.max));
        writer.EndObject();
        writer.Key("initial_tokens");
        writer.BeginObject();
        writer.KeyUint("min", session_class.initial_tokens.min);
        writer.KeyUint("max", session_class.initial_tokens.max);
        writer.EndObject();
        writer.Key("new_tokens_per_turn");
        writer.BeginObject();
        writer.KeyUint("min", session_class.new_tokens_per_turn.min);
        writer.KeyUint("max", session_class.new_tokens_per_turn.max);
        writer.EndObject();
        writer.Key("rewrite_tail_tokens");
        writer.BeginObject();
        writer.KeyUint("min", session_class.rewrite_tail_tokens.min);
        writer.KeyUint("max", session_class.rewrite_tail_tokens.max);
        writer.EndObject();
        writer.KeyDouble("shared_prefix_probability", session_class.shared_prefix_probability);
        writer.EndObject();
    }
    writer.EndArray();
    writer.EndObject();
}

namespace {

class V6dDeploymentFactory : public BehaviorFactory {
public:
    std::string_view TypeName() const override { return "v6d_deployment"; }

    ValidationResult Validate(const BehaviorSpec &spec) const override {
        ValidationResult result;
        V6dConfig config;
        std::vector<std::string> errors;
        ParseV6dConfig(spec, &config, &errors);
        for (auto &error : errors) {
            result.Fail("behaviors[" + spec.id + "]: " + error);
        }
        return result;
    }

    std::unique_ptr<ClientBehavior> Create(const BehaviorSpec &spec, RuntimeServices services) const override {
        V6dConfig config;
        std::vector<std::string> errors;
        if (!ParseV6dConfig(spec, &config, &errors)) {
            return nullptr;
        }
        return std::make_unique<V6dDeployment>(spec, std::move(config), services);
    }

    BehaviorIdentityClaims Claims(const BehaviorSpec &spec) const override {
        BehaviorIdentityClaims claims;
        V6dConfig config;
        std::vector<std::string> ignored;
        ParseV6dConfig(spec, &config, &ignored);
        if (!config.instance_id.empty()) {
            claims.exclusive_names.push_back("instance_id:" + config.instance_id);
        }
        if (!config.instance_group.empty()) {
            claims.required_instance_groups.push_back(config.instance_group);
        }
        for (uint32_t index = 0; index < config.process_count; ++index) {
            claims.exclusive_names.push_back("reporter_host_ip_port:" +
                                             MakeProcessIdentity(spec.id, config, index).host_ip_port);
        }
        return claims;
    }
};

} // namespace

std::unique_ptr<BehaviorFactory> MakeV6dDeploymentFactory() { return std::make_unique<V6dDeploymentFactory>(); }

} // namespace kvcm_swarm
