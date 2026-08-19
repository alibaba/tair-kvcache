#include "tools/kvcm_swarm/app/preflight.h"

#include <algorithm>

#include "tools/kvcm_swarm/clients/v6d/key_mapper.h"
#include "tools/kvcm_swarm/protocol/proto_alias.h"

namespace kvcm_swarm {
namespace {

constexpr const char *kPreflightHostIpPort = "10.255.255.254:65535";
constexpr const char *kPreflightSpecName = "v6d_4096";
constexpr int64_t kPreflightSpecSize = 4096;

std::string PreflightInstanceId(const ScenarioConfig &config) {
    return "kvcm-swarm-preflight-" + BlockHashHex(HashString(config.name + "/" + std::to_string(config.seed)));
}

// Chooses the instance group to validate against: the first group a behavior
// declared. Preflight never creates one.
std::string FirstInstanceGroup(const ScenarioConfig &config) {
    if (config.target.instance_groups.empty()) {
        return {};
    }
    return config.target.instance_groups.begin()->first;
}

} // namespace

Task<PreflightReport> PreflightRunner::Run(TimePoint deadline) {
    PreflightReport report;
    report.executed = true;
    report.temporary_instance_id = PreflightInstanceId(config_);
    report.temporary_host_ip_port = kPreflightHostIpPort;

    // The preflight uses the same protocol and transport as the real workload.
    TransportKind kind = config_.behaviors.empty() ? TransportKind::kHttp : config_.behaviors.front().transport;
    ClientIdentity identity;
    identity.behavior_type = "preflight";
    identity.behavior_id = "preflight";
    ClientTransportContext *context = transports_.CreateClientContext(identity, kind);

    const std::string instance_id = report.temporary_instance_id;
    const std::string instance_group = FirstInstanceGroup(config_);
    // Two independent temporary keys: the hot key validates BLOCK_ADD plus the
    // hot query, and the cold key validates the metadata-only cold write. Using
    // one key would let the freshly reported hot location satisfy the replica
    // threshold and silently skip the cold write.
    const std::string hot_object_key = "preflight-hot_" + BlockHashHex(HashString(instance_id)) + "_preflight";
    const std::string cold_object_key = "preflight-cold_" + BlockHashHex(HashString(instance_id)) + "_preflight";
    const int64_t block_key = ObjectKeyToBlockKey(hot_object_key);
    const int64_t cold_block_key = ObjectKeyToBlockKey(cold_object_key);

    auto make_options = [&](TimePoint call_deadline) {
        CallOptions options;
        options.lane = TrafficLane::kControl;
        options.deadline = std::min(call_deadline, deadline);
        return options;
    };
    auto fail = [&report](const char *stage, const std::string &detail) {
        report.passed = false;
        if (report.failure_stage.empty()) {
            report.failure_stage = stage;
            report.failure_detail = detail;
        }
    };

    // 1. admin and meta endpoints
    {
        admin::CheckHealthRequest request;
        request.set_trace_id("swarm-preflight-health");
        admin::CheckHealthResponse response;
        const RpcResult result = co_await context->Call(
            Api::kCheckHealth, request, &response, make_options(Now() + std::chrono::seconds(5)));
        const bool ok = result.ok && response.is_health();
        report.steps.emplace_back("admin_endpoint_check_health", ok);
        if (!ok) {
            fail("admin_endpoint_check_health",
                 result.raw_error.empty() ? "service reported unhealthy" : result.raw_error);
            co_return report;
        }
    }
    {
        meta::GetClusterInfoRequest request;
        request.set_trace_id("swarm-preflight-cluster");
        request.set_instance_id(instance_id);
        meta::GetClusterInfoResponse response;
        const RpcResult result = co_await context->Call(
            Api::kGetClusterInfo, request, &response, make_options(Now() + std::chrono::seconds(5)));
        report.steps.emplace_back("meta_endpoint_get_cluster_info", result.ok);
        if (!result.ok) {
            fail("meta_endpoint_get_cluster_info", result.raw_error);
            co_return report;
        }
    }

    if (instance_group.empty()) {
        report.cleanup_notes.push_back(
            "no instance group is declared in target.instance_groups, so registration, event reporting and the "
            "metadata-only cold write were not verified");
        report.passed = true;
        co_return report;
    }

    // 2. RegisterInstance with a temporary instance id
    {
        meta::RegisterInstanceRequest request;
        request.set_trace_id("swarm-preflight-register");
        request.set_instance_group(instance_group);
        request.set_instance_id(instance_id);
        request.set_block_size(1);
        auto *model = request.mutable_model_deployment();
        model->set_model_name("v6d");
        model->set_dtype("bytes");
        model->set_tp_size(1);
        model->set_dp_size(1);
        model->set_pp_size(1);
        model->set_extra("kvcm_swarm_preflight");
        auto *info = request.add_location_spec_infos();
        info->set_name(kPreflightSpecName);
        info->set_size(kPreflightSpecSize);
        auto *group = request.add_location_spec_groups();
        group->set_name(kPreflightSpecName);
        group->add_spec_names(kPreflightSpecName);
        meta::RegisterInstanceResponse response;
        const RpcResult result = co_await context->Call(
            Api::kRegisterInstance, request, &response, make_options(Now() + std::chrono::seconds(10)));
        report.steps.emplace_back("register_instance", result.ok);
        if (!result.ok) {
            fail("register_instance", result.raw_error);
            co_return report;
        }
    }

    auto make_event_request = [&](const char *trace) {
        meta::ReportEventRequest request;
        request.set_trace_id(trace);
        request.set_instance_id(instance_id);
        request.set_host_ip_port(kPreflightHostIpPort);
        request.set_storage_type(meta::ST_EVENT_REPORT_L2);
        return request;
    };

    bool hot_location_created = false;
    bool cold_key_created = false;

    // 3. NODE_REGISTER + HEARTBEAT
    {
        meta::ReportEventRequest request = make_event_request("swarm-preflight-node-register");
        auto *node = request.add_events();
        node->set_event_type(meta::EVENT_NODE_REGISTER);
        node->mutable_node_register()->add_mediums("mem");
        auto *heartbeat = request.add_events();
        heartbeat->set_event_type(meta::EVENT_HEARTBEAT);
        (*heartbeat->mutable_heartbeat()->mutable_system_status())["version"] = "v6d_1.0";
        meta::ReportEventResponse response;
        const RpcResult result = co_await context->Call(
            Api::kReportEvent, request, &response, make_options(Now() + std::chrono::seconds(5)));
        report.steps.emplace_back("node_register_and_heartbeat", result.ok);
        if (!result.ok) {
            fail("node_register_and_heartbeat", result.raw_error);
        }
    }

    // 4. BLOCK_ADD then hot query
    if (report.failure_stage.empty()) {
        meta::ReportEventRequest request = make_event_request("swarm-preflight-block-add");
        auto *event = request.add_events();
        event->set_event_type(meta::EVENT_BLOCK_ADD);
        auto *params = event->mutable_block_add();
        params->set_block_key(std::to_string(block_key));
        params->set_medium("mem");
        auto *spec = params->add_specs();
        spec->set_name(kPreflightSpecName);
        spec->set_uri(std::string("vineyard://") + kPreflightHostIpPort + "/mem");
        meta::ReportEventResponse response;
        const RpcResult result = co_await context->Call(
            Api::kReportEvent, request, &response, make_options(Now() + std::chrono::seconds(5)));
        bool ok = result.ok;
        for (const auto item : response.item_results()) {
            if (item != meta::OK) {
                ok = false;
            }
        }
        hot_location_created = ok;
        report.steps.emplace_back("block_add", ok);
        if (!ok) {
            fail("block_add", result.raw_error);
        }
    }
    if (report.failure_stage.empty()) {
        meta::GetCacheLocationsByBackendRequest request;
        request.set_trace_id("swarm-preflight-hot-query");
        request.set_instance_id(instance_id);
        request.set_query_type(meta::QT_BATCH_GET);
        request.add_block_keys(block_key);
        request.add_location_spec_names(kPreflightSpecName);
        request.mutable_block_mask()->set_offset(0);
        auto *selector = request.add_backend_selectors();
        selector->set_backend_type(meta::ST_EVENT_REPORT_L2);
        selector->set_strategy(meta::LSS_V6D_COVERAGE);
        meta::GetCacheLocationsByBackendResponse response;
        const RpcResult result = co_await context->Call(
            Api::kGetCacheLocationsByBackend, request, &response, make_options(Now() + std::chrono::seconds(5)));
        bool found = false;
        if (result.ok && response.key_locations_size() == 1) {
            for (const auto &location : response.key_locations(0).locations()) {
                if (location.type() == meta::ST_EVENT_REPORT_L2) {
                    found = true;
                }
            }
        }
        report.steps.emplace_back("hot_query", found);
        if (!found) {
            fail("hot_query", "the freshly reported hot location was not returned");
        }
    }

    // 5. metadata-only StartWriteCache -> FinishWriteCache -> cold query
    std::string write_session_id;
    if (report.failure_stage.empty()) {
        meta::StartWriteCacheRequest request;
        request.set_trace_id("swarm-preflight-start-write");
        request.set_instance_id(instance_id);
        request.add_block_keys(cold_block_key);
        request.add_location_spec_group_names(kPreflightSpecName);
        request.set_write_timeout_seconds(30);
        // A dedicated cold key with the default single-replica threshold, so
        // the cold allocation path is really exercised.
        request.set_min_replica_count(1);
        meta::StartWriteCacheResponse response;
        const RpcResult result = co_await context->Call(
            Api::kStartWriteCache, request, &response, make_options(Now() + std::chrono::seconds(10)));
        write_session_id = response.write_session_id();
        const bool ok = result.ok && !write_session_id.empty();
        report.steps.emplace_back("start_write_cache", ok);
        if (!ok) {
            fail("start_write_cache", result.raw_error);
        } else {
            cold_key_created = response.locations_size() > 0;
            meta::FinishWriteCacheRequest finish;
            finish.set_trace_id("swarm-preflight-finish-write");
            finish.set_instance_id(instance_id);
            finish.set_write_session_id(write_session_id);
            if (response.locations_size() > 0) {
                auto *masks = finish.mutable_success_blocks()->mutable_bool_masks();
                for (int i = 0; i < response.locations_size(); ++i) {
                    masks->add_values(true);
                }
            } else {
                finish.mutable_success_blocks()->set_offset(0);
            }
            meta::CommonResponse finish_response;
            const RpcResult finish_result = co_await context->Call(
                Api::kFinishWriteCache, finish, &finish_response, make_options(Now() + std::chrono::seconds(10)));
            report.steps.emplace_back("finish_write_cache", finish_result.ok);
            if (!finish_result.ok) {
                fail("finish_write_cache", finish_result.raw_error);
            }
        }
    }
    if (report.failure_stage.empty() && cold_key_created) {
        meta::GetCacheLocationsByBackendRequest request;
        request.set_trace_id("swarm-preflight-cold-query");
        request.set_instance_id(instance_id);
        request.set_query_type(meta::QT_BATCH_GET);
        request.add_block_keys(cold_block_key);
        request.add_location_spec_names(kPreflightSpecName);
        request.mutable_block_mask()->set_offset(0);
        for (const meta::StorageType type : {meta::ST_NFS,
                                             meta::ST_3FS,
                                             meta::ST_VCNS_3FS,
                                             meta::ST_MOONCAKE,
                                             meta::ST_TAIRMEMPOOL,
                                             meta::ST_TAIRMEMPOOL_SSD}) {
            auto *selector = request.add_backend_selectors();
            selector->set_backend_type(type);
            selector->set_strategy(meta::LSS_WEIGHTED_RANDOM);
        }
        meta::GetCacheLocationsByBackendResponse response;
        const RpcResult result = co_await context->Call(
            Api::kGetCacheLocationsByBackend, request, &response, make_options(Now() + std::chrono::seconds(5)));
        bool found = false;
        if (result.ok && response.key_locations_size() == 1) {
            for (const auto &location : response.key_locations(0).locations()) {
                if (location.type() != meta::ST_EVENT_REPORT_L2) {
                    found = true;
                }
            }
        }
        report.steps.emplace_back("cold_query", found);
        if (!found) {
            fail("cold_query", "the metadata-only cold allocation was not queryable");
        }
    }

    // ---- bounded cleanup, even when a step failed ----
    if (hot_location_created) {
        meta::ReportEventRequest request = make_event_request("swarm-preflight-block-delete");
        auto *event = request.add_events();
        event->set_event_type(meta::EVENT_BLOCK_DELETE);
        auto *params = event->mutable_block_delete();
        params->set_block_key(std::to_string(block_key));
        params->set_medium("mem");
        params->add_spec_names(kPreflightSpecName);
        meta::ReportEventResponse response;
        const RpcResult result = co_await context->Call(
            Api::kReportEvent, request, &response, make_options(Now() + std::chrono::seconds(5)));
        report.steps.emplace_back("cleanup_block_delete", result.ok);
    }
    if (cold_key_created) {
        // The only place RemoveCache is ever called: the tiny temporary cold
        // key preflight created itself. Never reused as workload cleanup.
        meta::RemoveCacheRequest request;
        request.set_trace_id("swarm-preflight-remove-cache");
        request.set_instance_id(instance_id);
        request.add_block_keys(cold_block_key);
        request.mutable_block_mask()->set_offset(0);
        meta::CommonResponse response;
        const RpcResult result = co_await context->Call(
            Api::kRemoveCache, request, &response, make_options(Now() + std::chrono::seconds(5)));
        ++report.remove_cache_calls;
        report.steps.emplace_back("cleanup_remove_cache", result.ok);
    }
    {
        meta::ReportEventRequest request = make_event_request("swarm-preflight-host-down");
        auto *event = request.add_events();
        event->set_event_type(meta::EVENT_HOST_DOWN);
        event->mutable_host_down();
        meta::ReportEventResponse response;
        const RpcResult result = co_await context->Call(
            Api::kReportEvent, request, &response, make_options(Now() + std::chrono::seconds(5)));
        report.steps.emplace_back("cleanup_host_down", result.ok);
    }
    report.cleanup_notes.push_back(
        "the temporary preflight instance registration itself is left in place: removing an instance is a "
        "deployment-management operation that belongs to the test fixture, not to the generator");

    report.passed = report.failure_stage.empty();
    co_return report;
}

} // namespace kvcm_swarm
