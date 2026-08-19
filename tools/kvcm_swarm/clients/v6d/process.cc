#include "tools/kvcm_swarm/clients/v6d/process.h"

#include <algorithm>

namespace kvcm_swarm {
namespace {

// Location ids and URIs must not contain '#'; the configuration validator
// already rejects such hosts, so this is a defensive assertion only.
bool IsSafeComponent(const std::string &value) { return !value.empty() && value.find('#') == std::string::npos; }

} // namespace

V6dProcess::V6dProcess(V6dDeploymentContext &deployment, V6dProcessIdentity identity, TransportKind transport_kind)
    : deployment_(deployment)
    , identity_(std::move(identity))
    , reporter_(ReporterIdentity{deployment.config().instance_id, identity_.host_ip_port})
    , self_uri_("vineyard://" + identity_.host_ip_port + "/mem")
    , transport_kind_(transport_kind)
    , cache_(deployment.services().executor, deployment.config().local_cache_capacity_bytes)
    , turn_capacity_(deployment.services().executor, deployment.config().local_cache_capacity_bytes) {
    ClientIdentity client_identity;
    client_identity.behavior_type = "v6d_deployment";
    client_identity.behavior_id = deployment.behavior_id();
    client_identity.process_id = identity_.process_id;
    transport_ = deployment.services().transports.CreateClientContext(client_identity, transport_kind_);
    cache_.SetEvictionTrigger([this]() { WakeEvictor(); });
}

V6dProcess::~V6dProcess() = default;

std::string V6dProcess::NextTraceId(const char *prefix) {
    const uint64_t sequence = trace_counter_.fetch_add(1, std::memory_order_relaxed);
    return std::string("swarm-") + prefix + "-" + identity_.process_id + "-" + std::to_string(sequence);
}

meta::ReportEventRequest V6dProcess::MakeReportEventRequest(const char *trace_prefix) {
    meta::ReportEventRequest request;
    request.set_trace_id(NextTraceId(trace_prefix));
    request.set_instance_id(deployment_.config().instance_id);
    request.set_host_ip_port(identity_.host_ip_port);
    // The confirmed compatibility mapping: the legacy ST_VINEYARD reporter
    // storage type is expressed as ST_EVENT_REPORT_L2.
    request.set_storage_type(meta::ST_EVENT_REPORT_L2);
    return request;
}

Task<bool> V6dProcess::Register(TimePoint planned_start, TimePoint deadline) {
    OperationGuard guard(this);
    if (planned_start > Now()) {
        co_await SleepUntil(deployment_.services().executor, planned_start, deployment_.services().stop);
    }
    if (!IsSafeComponent(identity_.host_ip_port)) {
        co_return false;
    }
    const V6dConfig &config = deployment_.config();

    meta::RegisterInstanceRequest request;
    request.set_trace_id(NextTraceId("register"));
    request.set_instance_group(config.instance_group);
    request.set_instance_id(config.instance_id);
    // The V6D protocol block_size is fixed at 1 and is unrelated to workload
    // group token block sizes.
    request.set_block_size(1);
    auto *deployment_info = request.mutable_model_deployment();
    deployment_info->set_model_name("v6d");
    deployment_info->set_dtype("bytes");
    deployment_info->set_use_mla(false);
    deployment_info->set_tp_size(1);
    deployment_info->set_dp_size(1);
    deployment_info->set_pp_size(1);
    deployment_info->set_extra("tiered_vineyard_tair_kvcm");
    // One single-member spec group per object size.
    std::vector<std::string> spec_names;
    for (const auto &group : config.groups) {
        if (std::find(spec_names.begin(), spec_names.end(), group.spec_name) != spec_names.end()) {
            continue;
        }
        spec_names.push_back(group.spec_name);
        auto *info = request.add_location_spec_infos();
        info->set_name(group.spec_name);
        info->set_size(static_cast<int64_t>(group.object_size_bytes));
        auto *spec_group = request.add_location_spec_groups();
        spec_group->set_name(group.spec_name);
        spec_group->add_spec_names(group.spec_name);
    }

    meta::RegisterInstanceResponse response;
    CallOptions options;
    options.lane = TrafficLane::kControl;
    options.deadline = std::min(deadline, Now() + config.rpc_timeout);
    options.stop = deployment_.services().stop;
    RpcResult result = co_await CallWithLeaderRefresh(Api::kRegisterInstance, request, &response, options);
    if (!result.ok) {
        co_return false;
    }
    deployment_.SetStorageConfigs(response.storage_configs());

    // First EventReport batch: NODE_REGISTER and HEARTBEAT in the same packet.
    meta::ReportEventRequest event_request = MakeReportEventRequest("node-register");
    auto *register_event = event_request.add_events();
    register_event->set_event_type(meta::EVENT_NODE_REGISTER);
    register_event->mutable_node_register()->add_mediums("mem");
    auto *heartbeat_event = event_request.add_events();
    heartbeat_event->set_event_type(meta::EVENT_HEARTBEAT);
    (*heartbeat_event->mutable_heartbeat()->mutable_system_status())["version"] = "v6d_1.0";
    (*heartbeat_event->mutable_heartbeat()->mutable_system_status())["heartbeat_seq"] = "0";

    meta::ReportEventResponse event_response;
    CallOptions event_options;
    event_options.lane = TrafficLane::kControl;
    event_options.deadline = std::min(deadline, Now() + config.rpc_timeout);
    event_options.stop = deployment_.services().stop;
    result = co_await CallWithLeaderRefresh(Api::kReportEvent, event_request, &event_response, event_options);
    deployment_.checks().CheckReportEventShape(static_cast<size_t>(event_request.events_size()),
                                               static_cast<size_t>(event_response.item_results_size()),
                                               "node_register+heartbeat");
    bool items_ok = true;
    for (const auto item : event_response.item_results()) {
        if (item != meta::OK) {
            items_ok = false;
        }
    }
    if (!result.ok || !items_ok) {
        co_return false;
    }
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        ++reporter_stats_.node_registers;
        ++reporter_stats_.generation;
        ++reporter_stats_.heartbeats_sent;
        reporter_stats_.registered = true;
    }
    deployment_.expected().SetReporterLive(reporter_);
    ready_.store(true, std::memory_order_release);
    co_return true;
}

void V6dProcess::StartMaintenance() {
    maintenance_loops_.fetch_add(2, std::memory_order_release);
    HeartbeatLoop().via(&deployment_.services().executor).start([](auto &&) {});
    LeaderDiscoveryLoop().via(&deployment_.services().executor).start([](auto &&) {});
    WakeEvictor();
}

Task<RpcResult> V6dProcess::CallWithLeaderRefresh(Api api,
                                                  const google::protobuf::Message &request,
                                                  google::protobuf::Message *response,
                                                  CallOptions options) {
    RpcResult result = co_await transport_->Call(api, request, response, options);
    if (result.service_status != kStatusServerNotLeader) {
        co_return result;
    }
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        ++reporter_stats_.not_leader_retries;
    }
    // Exactly one explicit endpoint refresh and retry. This is not a replay of
    // the enclosing turn, and it is never hidden inside the transport.
    const bool refreshed = co_await RefreshLeaderEndpoint(options.deadline);
    if (!refreshed) {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        ++reporter_stats_.not_leader_retry_failures;
        co_return result;
    }
    response->Clear();
    RpcResult retry = co_await transport_->Call(api, request, response, options);
    if (!retry.ok) {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        ++reporter_stats_.not_leader_retry_failures;
    }
    co_return retry;
}

Task<bool> V6dProcess::RefreshLeaderEndpoint(TimePoint deadline) {
    meta::GetClusterInfoRequest request;
    request.set_trace_id(NextTraceId("leader"));
    request.set_instance_id(deployment_.config().instance_id);
    meta::GetClusterInfoResponse response;
    CallOptions options;
    options.lane = TrafficLane::kControl;
    options.deadline = deadline.time_since_epoch().count() == 0
                           ? Now() + deployment_.config().rpc_timeout
                           : std::min(deadline, Now() + deployment_.config().rpc_timeout);
    options.stop = deployment_.services().stop;
    const RpcResult result = co_await transport_->Call(Api::kGetClusterInfo, request, &response, options);
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        ++reporter_stats_.leader_polls;
        if (!result.ok) {
            ++reporter_stats_.leader_poll_failures;
        }
    }
    if (!result.ok || !response.has_leader_endpoint()) {
        co_return false;
    }
    const auto &endpoint = response.leader_endpoint();
    if (endpoint.host().empty()) {
        co_return false;
    }
    std::string target;
    if (transport_kind_ == TransportKind::kHttp) {
        if (endpoint.meta_http_port() <= 0) {
            co_return false;
        }
        target = "http://" + endpoint.host() + ":" + std::to_string(endpoint.meta_http_port());
    } else {
        if (endpoint.meta_rpc_port() <= 0) {
            co_return false;
        }
        target = endpoint.host() + ":" + std::to_string(endpoint.meta_rpc_port());
    }
    if (target != transport_->MetaEndpoint()) {
        transport_->SetMetaEndpoint(target);
        std::lock_guard<std::mutex> lock(stats_mutex_);
        ++reporter_stats_.leader_endpoint_changes;
    }
    co_return true;
}

Task<> V6dProcess::HeartbeatLoop() {
    OperationGuard guard(this);
    const V6dConfig &config = deployment_.config();
    TimePoint planned = Now() + config.heartbeat_interval;
    while (!own_stop_.StopRequested() && !deployment_.services().stop.StopRequested()) {
        const bool slept = co_await SleepUntil(deployment_.services().executor, planned, own_stop_.Token());
        if (!slept) {
            break;
        }
        meta::ReportEventRequest request = MakeReportEventRequest("heartbeat");
        auto *event = request.add_events();
        event->set_event_type(meta::EVENT_HEARTBEAT);
        auto &status = *event->mutable_heartbeat()->mutable_system_status();
        uint64_t sequence = 0;
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            sequence = ++heartbeat_sequence_;
        }
        // Monotonic local counters only; KVCM treats system_status as opaque.
        status["version"] = "v6d_1.0";
        status["heartbeat_seq"] = std::to_string(sequence);
        status["cache_objects"] = std::to_string(cache_.size());
        status["cache_used_bytes"] = std::to_string(cache_.used_bytes());

        meta::ReportEventResponse response;
        CallOptions options;
        options.lane = TrafficLane::kControl;
        options.planned_at = planned;
        options.deadline = Now() + config.rpc_timeout;
        options.stop = deployment_.services().stop;
        const RpcResult result = co_await CallWithLeaderRefresh(Api::kReportEvent, request, &response, options);
        deployment_.checks().CheckReportEventShape(
            static_cast<size_t>(request.events_size()), static_cast<size_t>(response.item_results_size()), "heartbeat");
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            ++reporter_stats_.heartbeats_sent;
            if (!result.ok) {
                ++reporter_stats_.heartbeats_failed;
            }
        }
        planned += config.heartbeat_interval;
        const TimePoint now = Now();
        while (planned <= now) {
            planned += config.heartbeat_interval;
        }
    }
    maintenance_loops_.fetch_sub(1, std::memory_order_release);
    co_return;
}

Task<> V6dProcess::LeaderDiscoveryLoop() {
    OperationGuard guard(this);
    const V6dConfig &config = deployment_.config();
    TimePoint planned = Now() + config.leader_poll_interval;
    while (!own_stop_.StopRequested() && !deployment_.services().stop.StopRequested()) {
        const bool slept = co_await SleepUntil(deployment_.services().executor, planned, own_stop_.Token());
        if (!slept) {
            break;
        }
        co_await RefreshLeaderEndpoint(Now() + config.rpc_timeout);
        planned += config.leader_poll_interval;
        const TimePoint now = Now();
        while (planned <= now) {
            planned += config.leader_poll_interval;
        }
    }
    maintenance_loops_.fetch_sub(1, std::memory_order_release);
    co_return;
}

Task<AsyncCapacityBudget::Guard>
V6dProcess::AcquireTurnCapacity(uint64_t working_set_bytes, TimePoint deadline, StopToken stop) {
    return turn_capacity_.Acquire(working_set_bytes, deadline, std::move(stop));
}

void V6dProcess::WakeEvictor() {
    if (draining_.load(std::memory_order_acquire)) {
        // During drain the pipeline is driven directly by the shutdown flush.
        return;
    }
    std::shared_ptr<AsyncSlot<bool>> wake;
    {
        std::lock_guard<std::mutex> lock(evictor_mutex_);
        if (!evictor_running_) {
            evictor_running_ = true;
            evictor_wake_ = std::make_shared<AsyncSlot<bool>>(deployment_.services().executor);
            EvictionLoop().via(&deployment_.services().executor).start([](auto &&) {});
            return;
        }
        wake = evictor_wake_;
    }
    if (wake) {
        wake->Complete(true);
    }
}

Task<> V6dProcess::EvictionLoop() {
    OperationGuard guard(this);
    const V6dConfig &config = deployment_.config();
    while (!own_stop_.StopRequested() && !draining_.load(std::memory_order_acquire)) {
        // Eviction is driven purely by the capacity the waiting materialisations
        // actually requested: there is no periodic scan and no random spill.
        const uint64_t bytes_needed = cache_.pending_wait_bytes();
        if (bytes_needed > 0) {
            std::vector<GroupObject> batch = cache_.SelectVictims(bytes_needed, config.eviction_batch_size);
            if (!batch.empty()) {
                const bool progressed = co_await RunEvictionBatch(std::move(batch), false);
                if (!progressed) {
                    // No progress: back off briefly instead of spinning on a
                    // persistent failure.
                    co_await SleepFor(
                        deployment_.services().executor, std::chrono::milliseconds(20), deployment_.services().stop);
                }
                continue;
            }
        }
        if (deployment_.services().stop.StopRequested()) {
            break;
        }
        std::shared_ptr<AsyncSlot<bool>> wake;
        {
            std::lock_guard<std::mutex> lock(evictor_mutex_);
            evictor_wake_ = std::make_shared<AsyncSlot<bool>>(deployment_.services().executor);
            wake = evictor_wake_;
        }
        // Re-check after arming so a wake-up published between the selection and
        // the new slot is never lost.
        if (cache_.pending_wait_bytes() == 0 || !cache_.HasEvictable()) {
            StopCallbackGuard stop_guard(own_stop_.Token(), [wake]() { wake->Complete(false); });
            StopCallbackGuard global_guard(deployment_.services().stop, [wake]() { wake->Complete(false); });
            co_await *wake;
        }
    }
    {
        std::lock_guard<std::mutex> lock(evictor_mutex_);
        evictor_running_ = false;
    }
    co_return;
}

Task<bool> V6dProcess::RunEvictionBatch(std::vector<GroupObject> batch, bool shutdown_flush) {
    OperationGuard guard(this);
    eviction_batches_in_flight_.fetch_add(1, std::memory_order_release);
    struct BatchGuard {
        std::atomic<uint32_t> *counter;
        ~BatchGuard() { counter->fetch_sub(1, std::memory_order_release); }
    } batch_guard{&eviction_batches_in_flight_};
    const V6dConfig &config = deployment_.config();
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        ++eviction_stats_.batches;
        eviction_stats_.objects_selected += batch.size();
        if (shutdown_flush) {
            ++eviction_stats_.shutdown_flush_batches;
            eviction_stats_.shutdown_flush_objects += batch.size();
        }
    }

    meta::StartWriteCacheRequest request;
    request.set_trace_id(NextTraceId("start-evict-write"));
    request.set_instance_id(config.instance_id);
    for (const auto &object : batch) {
        request.add_block_keys(object.block_key);
        // One single-member spec group per object; the spec always comes from
        // the same object as the ADD, DELETE and lookup requests.
        request.add_location_spec_group_names(object.spec_name);
    }
    request.set_write_timeout_seconds(
        static_cast<int32_t>(std::chrono::duration_cast<std::chrono::seconds>(config.write_timeout).count()));
    request.set_min_replica_count(config.min_replica_count);

    meta::StartWriteCacheResponse response;
    CallOptions options;
    options.lane = shutdown_flush ? TrafficLane::kControl : TrafficLane::kBusiness;
    // Each RPC of an eviction gets its own bounded budget instead of inheriting
    // a possibly expired batch deadline: an operation that opens a write
    // session must be able to close it, and the BLOCK_DELETE that follows a
    // local removal must be able to reach the server.
    options.deadline = Now() + config.rpc_timeout;
    options.stop = shutdown_flush ? StopToken() : deployment_.services().stop;
    const RpcResult start_result = co_await CallWithLeaderRefresh(Api::kStartWriteCache, request, &response, options);

    if (!start_result.ok) {
        const bool uncertain = IsUncertain(start_result.transport_error);
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            if (uncertain) {
                ++eviction_stats_.start_write_unknown;
                ++eviction_stats_.protected_uncertain;
            } else {
                ++eviction_stats_.start_write_failed;
            }
        }
        if (!uncertain) {
            // Explicitly not executed: the objects stay resident and can be
            // retried on a later capacity request.
            for (const auto &object : batch) {
                cache_.RestoreResident(object.object_key);
            }
            std::lock_guard<std::mutex> restore_lock(stats_mutex_);
            eviction_stats_.restored_resident += batch.size();
        }
        // Uncertain: never release the local object or its capacity; the entry
        // stays protected and is reported as blocked.
        co_return false;
    }

    // Split the batch into writable and masked items. A masked item already
    // satisfies the replica threshold and must not create a new cold location.
    std::vector<size_t> writable_indices;
    for (size_t i = 0; i < batch.size(); ++i) {
        bool masked = false;
        if (response.block_mask().info_case() == meta::BlockMask::kOffset) {
            masked = static_cast<int32_t>(i) < response.block_mask().offset();
        } else if (response.block_mask().info_case() == meta::BlockMask::kBoolMasks) {
            const auto &values = response.block_mask().bool_masks().values();
            masked = static_cast<int>(i) < values.size() && values[static_cast<int>(i)];
        }
        if (!masked) {
            writable_indices.push_back(i);
        }
    }
    const size_t masked_count = batch.size() - writable_indices.size();
    deployment_.checks().CheckStartWriteShape(
        batch.size(), writable_indices.size(), static_cast<size_t>(response.locations_size()), "eviction_start_write");
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        ++eviction_stats_.start_write_ok;
        eviction_stats_.writable_items += writable_indices.size();
        eviction_stats_.masked_items += masked_count;
    }

    // Record the pending cold allocations before closing the session.
    std::vector<ColdLocationKey> cold_keys;
    const size_t usable_locations =
        std::min<size_t>(writable_indices.size(), static_cast<size_t>(response.locations_size()));
    for (size_t slot = 0; slot < usable_locations; ++slot) {
        const GroupObject &object = batch[writable_indices[slot]];
        const auto &location = response.locations(static_cast<int>(slot));
        std::string uri;
        for (const auto &spec : location.location_specs()) {
            if (spec.name() == object.spec_name) {
                uri = spec.uri();
                break;
            }
        }
        ColdLocationKey key;
        key.block_key = object.block_key;
        key.spec_name = object.spec_name;
        key.storage_uri = uri;
        cold_keys.push_back(key);
        deployment_.expected().ColdPendingCreate(key);
    }

    // metadata-only: no bytes are written; writable items declare success.
    meta::FinishWriteCacheRequest finish_request;
    finish_request.set_trace_id(NextTraceId("finish-evict-write"));
    finish_request.set_instance_id(config.instance_id);
    finish_request.set_write_session_id(response.write_session_id());
    if (writable_indices.empty()) {
        // Nothing writable: still close the returned write session.
        finish_request.mutable_success_blocks()->set_offset(0);
    } else {
        auto *masks = finish_request.mutable_success_blocks()->mutable_bool_masks();
        for (size_t i = 0; i < writable_indices.size(); ++i) {
            masks->add_values(true);
        }
    }
    meta::CommonResponse finish_response;
    CallOptions finish_options;
    finish_options.lane = options.lane;
    finish_options.deadline = Now() + config.rpc_timeout;
    // A write session that has been opened must always be closed, so the
    // Finish call is not cancelled by a drain stop.
    finish_options.stop = StopToken();
    const RpcResult finish_result =
        co_await CallWithLeaderRefresh(Api::kFinishWriteCache, finish_request, &finish_response, finish_options);

    if (!finish_result.ok) {
        const bool uncertain = IsUncertain(finish_result.transport_error);
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            if (uncertain) {
                ++eviction_stats_.finish_write_unknown;
                ++eviction_stats_.protected_uncertain;
            } else {
                ++eviction_stats_.finish_write_failed;
            }
        }
        for (const auto &key : cold_keys) {
            if (uncertain) {
                deployment_.expected().ColdUnknown(key);
            } else {
                deployment_.expected().ColdNotExecuted(key);
            }
        }
        if (!uncertain) {
            for (const auto &object : batch) {
                cache_.RestoreResident(object.object_key);
            }
            std::lock_guard<std::mutex> lock(stats_mutex_);
            eviction_stats_.restored_resident += batch.size();
        }
        // The write session did not close successfully, so the local hot
        // object must not be dropped and BLOCK_DELETE must not be sent.
        co_return false;
    }

    // The write session is now closed. Only from here may the local object be
    // removed, and only then may BLOCK_DELETE be sent.
    uint64_t confirmed_bytes = 0;
    for (size_t slot = 0; slot < cold_keys.size(); ++slot) {
        const GroupObject &object = batch[writable_indices[slot]];
        deployment_.expected().ColdConfirm(cold_keys[slot], object.object_size);
        confirmed_bytes += object.object_size;
    }
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        ++eviction_stats_.finish_write_ok;
        eviction_stats_.cold_allocations_confirmed += cold_keys.size();
        eviction_stats_.cold_allocation_bytes += confirmed_bytes;
    }

    for (const auto &object : batch) {
        cache_.MarkRemoved(object.object_key);
    }
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        eviction_stats_.local_removed += batch.size();
    }
    co_await ReportBlockDelete(batch);
    deployment_.checks().RecordCompletedEviction(
        writable_indices.size(), masked_count, cold_keys.size(), batch.size(), batch.size());
    co_return true;
}

Task<> V6dProcess::ReportBlockAdd(const std::vector<GroupObject> &objects) {
    if (objects.empty()) {
        co_return;
    }
    OperationGuard guard(this);
    const V6dConfig &config = deployment_.config();
    meta::ReportEventRequest request = MakeReportEventRequest("block-add");
    std::vector<HotLocationKey> keys;
    keys.reserve(objects.size());
    for (const auto &object : objects) {
        auto *event = request.add_events();
        event->set_event_type(meta::EVENT_BLOCK_ADD);
        auto *params = event->mutable_block_add();
        params->set_block_key(std::to_string(object.block_key));
        params->set_medium("mem");
        auto *spec = params->add_specs();
        spec->set_name(object.spec_name);
        spec->set_uri(self_uri_);
        HotLocationKey key;
        key.block_key = object.block_key;
        key.spec_name = object.spec_name;
        key.reporter = reporter_;
        keys.push_back(key);
        deployment_.expected().HotPendingCreate(key);
    }

    meta::ReportEventResponse response;
    CallOptions options;
    options.lane = TrafficLane::kBusiness;
    // The objects are already in this process's cache, so the announcement gets
    // its own bounded budget and must reach a definite outcome even when the
    // enclosing turn is already late.
    options.deadline = Now() + config.rpc_timeout;
    options.stop = deployment_.services().stop;
    const RpcResult result = co_await CallWithLeaderRefresh(Api::kReportEvent, request, &response, options);
    deployment_.checks().CheckReportEventShape(
        static_cast<size_t>(request.events_size()), static_cast<size_t>(response.item_results_size()), "block_add");

    uint64_t confirmed = 0;
    uint64_t failed = 0;
    uint64_t unknown = 0;
    const bool uncertain = IsUncertain(result.transport_error);
    for (size_t i = 0; i < keys.size(); ++i) {
        bool item_ok = result.ok;
        if (result.ok && static_cast<int>(i) < response.item_results_size()) {
            item_ok = response.item_results(static_cast<int>(i)) == meta::OK;
        }
        if (item_ok) {
            deployment_.expected().HotConfirm(keys[i]);
            ++confirmed;
        } else if (uncertain) {
            deployment_.expected().HotUnknown(keys[i], false);
            ++unknown;
        } else {
            deployment_.expected().HotNotExecuted(keys[i], false);
            ++failed;
        }
    }
    std::lock_guard<std::mutex> lock(stats_mutex_);
    ++reporter_stats_.block_add_batches;
    reporter_stats_.block_add_items += keys.size();
    reporter_stats_.block_add_confirmed += confirmed;
    reporter_stats_.block_add_failed += failed;
    reporter_stats_.block_add_unknown += unknown;
    co_return;
}

Task<> V6dProcess::ReportBlockDelete(const std::vector<GroupObject> &objects) {
    if (objects.empty()) {
        co_return;
    }
    meta::ReportEventRequest request = MakeReportEventRequest("block-delete");
    std::vector<HotLocationKey> keys;
    for (const auto &object : objects) {
        auto *event = request.add_events();
        event->set_event_type(meta::EVENT_BLOCK_DELETE);
        auto *params = event->mutable_block_delete();
        params->set_block_key(std::to_string(object.block_key));
        params->set_medium("mem");
        // The confirmed compatibility mapping: BLOCK_DELETE always carries the
        // non-empty spec_names of the object being removed.
        params->add_spec_names(object.spec_name);
        HotLocationKey key;
        key.block_key = object.block_key;
        key.spec_name = object.spec_name;
        key.reporter = reporter_;
        keys.push_back(key);
        deployment_.expected().HotPendingDelete(key);
    }

    meta::ReportEventResponse response;
    CallOptions options;
    options.lane = TrafficLane::kBusiness;
    // The delete happens after the local object is already gone, so it gets its
    // own bounded budget and is never cancelled: otherwise the hot location
    // would linger with no local object behind it.
    options.deadline = Now() + deployment_.config().rpc_timeout;
    options.stop = StopToken();
    const RpcResult result = co_await CallWithLeaderRefresh(Api::kReportEvent, request, &response, options);
    deployment_.checks().CheckReportEventShape(
        static_cast<size_t>(request.events_size()), static_cast<size_t>(response.item_results_size()), "block_delete");

    uint64_t confirmed = 0;
    uint64_t failed = 0;
    uint64_t unknown = 0;
    const bool uncertain = IsUncertain(result.transport_error);
    for (size_t i = 0; i < keys.size(); ++i) {
        bool item_ok = result.ok;
        if (result.ok && static_cast<int>(i) < response.item_results_size()) {
            item_ok = response.item_results(static_cast<int>(i)) == meta::OK;
        }
        if (item_ok) {
            deployment_.expected().HotRemove(keys[i]);
            ++confirmed;
        } else if (uncertain) {
            deployment_.expected().HotUnknown(keys[i], true);
            ++unknown;
        } else {
            deployment_.expected().HotNotExecuted(keys[i], true);
            ++failed;
        }
    }
    std::lock_guard<std::mutex> lock(stats_mutex_);
    ++reporter_stats_.block_delete_batches;
    reporter_stats_.block_delete_items += keys.size();
    reporter_stats_.block_delete_confirmed += confirmed;
    reporter_stats_.block_delete_failed += failed;
    reporter_stats_.block_delete_unknown += unknown;
    co_return;
}

Task<RpcResult> V6dProcess::Lookup(const meta::GetCacheLocationsByBackendRequest &request,
                                   meta::GetCacheLocationsByBackendResponse *response,
                                   TimePoint planned_at,
                                   TimePoint deadline,
                                   StopToken stop) {
    OperationGuard guard(this);
    CallOptions options;
    options.lane = TrafficLane::kBusiness;
    options.planned_at = planned_at;
    options.deadline = std::min(deadline, Now() + deployment_.config().rpc_timeout);
    options.stop = std::move(stop);
    RpcResult result = co_await CallWithLeaderRefresh(Api::kGetCacheLocationsByBackend, request, response, options);
    co_return result;
}

Task<> V6dProcess::Drain(TimePoint deadline) {
    draining_.store(true, std::memory_order_release);
    const V6dConfig &config = deployment_.config();
    // Let the background pipeline finish the eviction it already started: an
    // operation that already opened a write session must close it, and its
    // BLOCK_DELETE must be sent before this process reports HOST_DOWN.
    {
        std::shared_ptr<AsyncSlot<bool>> wake;
        {
            std::lock_guard<std::mutex> lock(evictor_mutex_);
            wake = evictor_wake_;
        }
        if (wake) {
            wake->Complete(false);
        }
    }
    while (Now() < deadline) {
        bool idle = eviction_batches_in_flight_.load(std::memory_order_acquire) == 0;
        {
            std::lock_guard<std::mutex> lock(evictor_mutex_);
            idle = idle && !evictor_running_;
        }
        if (idle) {
            break;
        }
        co_await SleepFor(deployment_.services().executor, std::chrono::milliseconds(5), StopToken());
    }
    // Shutdown flush: every remaining evictable resident object goes through
    // the same pipeline, in batches bounded by the eviction batch size.
    while (Now() < deadline) {
        std::vector<GroupObject> batch = cache_.SelectAllEvictable(config.eviction_batch_size);
        if (batch.empty()) {
            break;
        }
        co_await RunEvictionBatch(std::move(batch), true);
    }
    // Whatever is still resident after the deadline stays local and is reported
    // as an unflushed object rather than silently dropped.
    // Heartbeat and leader discovery run until this process goes down, which is
    // exactly now. They must stop before HOST_DOWN, otherwise a heartbeat that
    // arrives after the reporter is unregistered would legitimately be rejected
    // with NODE_NOT_REGISTERED.
    own_stop_.RequestStop();
    while (maintenance_loops_.load(std::memory_order_acquire) > 0 && Now() < deadline) {
        co_await SleepFor(deployment_.services().executor, std::chrono::milliseconds(2), StopToken());
    }
    co_await SendHostDown(std::min(deadline, Now() + config.host_down_timeout));
    co_return;
}

Task<> V6dProcess::SendHostDown(TimePoint deadline) {
    OperationGuard guard(this);
    meta::ReportEventRequest request = MakeReportEventRequest("host-down");
    auto *event = request.add_events();
    event->set_event_type(meta::EVENT_HOST_DOWN);
    event->mutable_host_down();
    meta::ReportEventResponse response;
    CallOptions options;
    options.lane = TrafficLane::kControl;
    options.deadline = deadline;
    options.stop = StopToken();
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        ++reporter_stats_.host_down_attempted;
    }
    const RpcResult result = co_await CallWithLeaderRefresh(Api::kReportEvent, request, &response, options);
    if (result.ok) {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        ++reporter_stats_.host_down_succeeded;
    }
    if (result.ok) {
        // A successful HOST_DOWN unregisters the reporter immediately, so every
        // hot location it owned retires now.
        deployment_.expected().RetireReporter(reporter_);
    } else {
        deployment_.expected().SetReporterUnavailable(reporter_, false);
    }
    co_return;
}

ReporterStats V6dProcess::reporter_stats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return reporter_stats_;
}

EvictionStats V6dProcess::eviction_stats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return eviction_stats_;
}

void V6dProcess::WriteReport(JsonWriter &writer) const {
    const ReporterStats reporter = reporter_stats();
    const EvictionStats eviction = eviction_stats();
    const LocalCacheStats cache = cache_.Stats();
    writer.BeginObject();
    writer.KeyString("process_id", identity_.process_id);
    writer.KeyString("host_ip_port", identity_.host_ip_port);
    writer.KeyString("reporter_identity", reporter_.instance_id + "|ST_EVENT_REPORT_L2|" + reporter_.host_ip_port);
    writer.KeyBool("registered", reporter.registered);
    writer.KeyUint("reporter_generation", reporter.generation);
    writer.Key("reporter");
    writer.BeginObject();
    writer.KeyUint("node_registers", reporter.node_registers);
    writer.KeyUint("heartbeats_sent", reporter.heartbeats_sent);
    writer.KeyUint("heartbeats_failed", reporter.heartbeats_failed);
    writer.KeyUint("leader_polls", reporter.leader_polls);
    writer.KeyUint("leader_poll_failures", reporter.leader_poll_failures);
    writer.KeyUint("leader_endpoint_changes", reporter.leader_endpoint_changes);
    writer.KeyUint("not_leader_retries", reporter.not_leader_retries);
    writer.KeyUint("not_leader_retry_failures", reporter.not_leader_retry_failures);
    writer.KeyUint("block_add_batches", reporter.block_add_batches);
    writer.KeyUint("block_add_items", reporter.block_add_items);
    writer.KeyUint("block_add_confirmed", reporter.block_add_confirmed);
    writer.KeyUint("block_add_failed", reporter.block_add_failed);
    writer.KeyUint("block_add_unknown", reporter.block_add_unknown);
    writer.KeyUint("block_delete_batches", reporter.block_delete_batches);
    writer.KeyUint("block_delete_items", reporter.block_delete_items);
    writer.KeyUint("block_delete_confirmed", reporter.block_delete_confirmed);
    writer.KeyUint("block_delete_failed", reporter.block_delete_failed);
    writer.KeyUint("block_delete_unknown", reporter.block_delete_unknown);
    writer.KeyUint("host_down_attempted", reporter.host_down_attempted);
    writer.KeyUint("host_down_succeeded", reporter.host_down_succeeded);
    writer.EndObject();
    writer.Key("turn_capacity");
    writer.BeginObject();
    writer.KeyUint("capacity_bytes", turn_capacity_.capacity());
    writer.KeyUint("in_use_bytes", turn_capacity_.in_use());
    writer.KeyUint("peak_in_use_bytes", turn_capacity_.peak_in_use());
    writer.KeyUint("waits", turn_capacity_.waits());
    writer.KeyUint("timeouts", turn_capacity_.timeouts());
    writer.KeyDouble("wait_ms_total", static_cast<double>(turn_capacity_.wait_ns_total()) / 1e6);
    writer.KeyDouble("wait_ms_max", static_cast<double>(turn_capacity_.wait_ns_max()) / 1e6);
    writer.EndObject();
    writer.Key("eviction");
    writer.BeginObject();
    writer.KeyUint("batches", eviction.batches);
    writer.KeyUint("objects_selected", eviction.objects_selected);
    writer.KeyUint("start_write_ok", eviction.start_write_ok);
    writer.KeyUint("start_write_failed", eviction.start_write_failed);
    writer.KeyUint("start_write_unknown", eviction.start_write_unknown);
    writer.KeyUint("writable_items", eviction.writable_items);
    writer.KeyUint("masked_items", eviction.masked_items);
    writer.KeyUint("finish_write_ok", eviction.finish_write_ok);
    writer.KeyUint("finish_write_failed", eviction.finish_write_failed);
    writer.KeyUint("finish_write_unknown", eviction.finish_write_unknown);
    writer.KeyUint("cold_allocations_confirmed", eviction.cold_allocations_confirmed);
    writer.KeyUint("cold_allocation_bytes", eviction.cold_allocation_bytes);
    writer.KeyUint("local_removed", eviction.local_removed);
    writer.KeyUint("restored_resident", eviction.restored_resident);
    writer.KeyUint("protected_uncertain", eviction.protected_uncertain);
    writer.KeyUint("shutdown_flush_batches", eviction.shutdown_flush_batches);
    writer.KeyUint("shutdown_flush_objects", eviction.shutdown_flush_objects);
    writer.EndObject();
    writer.Key("local_cache");
    writer.BeginObject();
    writer.KeyUint("capacity_bytes", cache.capacity_bytes);
    writer.KeyUint("used_bytes", cache.used_bytes);
    writer.KeyUint("peak_used_bytes", cache.peak_used_bytes);
    writer.KeyUint("entries", cache.entries);
    writer.KeyUint("peak_entries", cache.peak_entries);
    writer.KeyUint("local_hits", cache.local_hits);
    writer.KeyUint("local_misses", cache.local_misses);
    writer.KeyUint("inserts", cache.inserts);
    writer.KeyUint("removed", cache.removed);
    writer.KeyUint("restored_resident", cache.restored_resident);
    writer.KeyUint("victims_selected", cache.victims_selected);
    writer.KeyUint("insert_rejected_oversize", cache.insert_rejected_oversize);
    writer.KeyUint("backpressure_waits", cache.backpressure_waits);
    writer.KeyUint("backpressure_timeouts", cache.backpressure_timeouts);
    writer.KeyDouble("backpressure_wait_ms_total", static_cast<double>(cache.backpressure_wait_ns) / 1e6);
    writer.KeyDouble("backpressure_wait_ms_max", static_cast<double>(cache.backpressure_wait_ns_max) / 1e6);
    writer.KeyUint("no_victim_waits", cache.no_victim_waits);
    writer.EndObject();
    writer.EndObject();
}

} // namespace kvcm_swarm
