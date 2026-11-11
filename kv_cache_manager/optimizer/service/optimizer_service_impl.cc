#include "kv_cache_manager/optimizer/service/optimizer_service_impl.h"

#include <unordered_set>
#include <utility>
#include <vector>

#include "kv_cache_manager/common/error_code.h"
#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/event/event_manager.h"
#include "kv_cache_manager/event/spec_events/optimizer_query_hit_event.h"
#include "kv_cache_manager/optimizer/config/optimizer_instance_group.h"
#include "kv_cache_manager/optimizer/config/optimizer_instance_info.h"
#include "kv_cache_manager/optimizer/config/optimizer_registry_manager.h"
#include "kv_cache_manager/optimizer/manager/online_runtime/online_optimizer_manager.h"
#include "kv_cache_manager/optimizer/metrics/optimizer_metrics_collector.h"
#include "kv_cache_manager/optimizer/metrics/optimizer_metrics_reporter.h"
#include "kv_cache_manager/optimizer/service/online_optimizer_server_config.h"
#include "kv_cache_manager/optimizer/service/optimizer_call_guard.h"

namespace kv_cache_manager {

namespace {

void SetPbResponseHeader(proto::optimizer::CommonResponseHeader *header, ErrorCode ec) {
    auto *status = header->mutable_status();
    if (ec == EC_OK) {
        status->set_code(proto::optimizer::OK);
    } else {
        status->set_code(ToPbError<proto::optimizer::ErrorCode>(ec));
    }
}

OptimizerInstanceGroup ConvertProtoToInstanceGroup(const proto::optimizer::OptimizerInstanceGroupProto &pb) {
    OptimizerInstanceGroup group;
    group.set_name(pb.name());
    std::vector<double> caps(pb.capacity_gb().begin(), pb.capacity_gb().end());
    group.set_capacity_gb(caps);
    if (pb.eviction_policy() == proto::optimizer::OPTIMIZER_EVICTION_POLICY_LRU) {
        group.set_eviction_policy("lru");
    } else {
        group.set_eviction_policy("");
    }
    group.set_shared_group_quota(pb.shared_group_quota());
    group.set_enable_theoretical_max_cache(pb.enable_theoretical_max_cache());
    group.set_ttl_seconds_list({pb.ttl_seconds_list().begin(), pb.ttl_seconds_list().end()});
    group.set_enable_prefix_hash(pb.enable_prefix_hash());
    return group;
}

void ConvertInstanceGroupToProto(const OptimizerInstanceGroup &group,
                                 proto::optimizer::OptimizerInstanceGroupProto *pb) {
    pb->set_name(group.name());
    for (double cap : group.capacity_gb()) {
        pb->add_capacity_gb(cap);
    }
    pb->set_shared_group_quota(group.shared_group_quota());
    for (auto ttl : group.ttl_seconds_list()) {
        pb->add_ttl_seconds_list(ttl);
    }
    pb->set_enable_theoretical_max_cache(group.enable_theoretical_max_cache());
    pb->set_enable_prefix_hash(group.enable_prefix_hash());
    if (group.eviction_policy() == "lru") {
        pb->set_eviction_policy(proto::optimizer::OPTIMIZER_EVICTION_POLICY_LRU);
    } else {
        pb->set_eviction_policy(proto::optimizer::OPTIMIZER_EVICTION_POLICY_UNSPECIFIED);
    }
}

OptimizerInstanceInfo ConvertProtoToInstanceInfo(const proto::optimizer::OptimizerRegisterInstanceRequest &request) {
    std::vector<LocationSpecInfo> specs;
    specs.reserve(request.location_spec_infos_size());
    for (const auto &s : request.location_spec_infos()) {
        specs.emplace_back(s.name(), s.size());
    }

    std::vector<LocationSpecGroup> groups;
    groups.reserve(request.location_spec_groups_size());
    for (const auto &g : request.location_spec_groups()) {
        std::vector<std::string> spec_names(g.spec_names().begin(), g.spec_names().end());
        groups.emplace_back(g.name(), spec_names);
    }

    OptimizerStateInfo optimizer_state_info(request.optimizer_state_info().full_location_spec_group_name(),
                                            request.optimizer_state_info().linear_location_spec_group_name());

    return OptimizerInstanceInfo(request.instance_group(),
                                 request.instance_id(),
                                 request.block_size(),
                                 specs,
                                 groups,
                                 request.linear_step(),
                                 optimizer_state_info);
}

void SetErrorOnCollector(RequestContext *request_context, ErrorCode ec) {
    if (ec == EC_OK)
        return;
    auto *collector = dynamic_cast<OptimizerServiceMetricsCollector *>(request_context->metrics_collector());
    KVCM_METRICS_COLLECTOR_SET_METRICS(collector, service, error_code, static_cast<double>(ec));
}

} // namespace

OptimizerServiceImpl::OptimizerServiceImpl(std::shared_ptr<OnlineOptimizerManager> manager,
                                           std::shared_ptr<OptimizerMetricsReporter> metrics_reporter,
                                           std::shared_ptr<EventManager> event_manager)
    : manager_(std::move(manager))
    , metrics_reporter_(std::move(metrics_reporter))
    , event_manager_(std::move(event_manager)) {}

// InstanceGroup CRUD

void OptimizerServiceImpl::CreateInstanceGroup(RequestContext *request_context,
                                               const proto::optimizer::CreateInstanceGroupRequest *request,
                                               proto::optimizer::CommonResponse *response) {
    request_context->set_api_name("CreateInstanceGroup");
    OptimizerCallGuard guard(request_context, metrics_reporter_.get());

    auto group = ConvertProtoToInstanceGroup(request->instance_group());

    std::string invalid_fields;
    if (!group.ValidateRequiredFields(invalid_fields)) {
        KVCM_LOG_ERROR("CreateInstanceGroup: validation failed, invalid_fields[%s]", invalid_fields.c_str());
        SetPbResponseHeader(response->mutable_header(), EC_BADARGS);
        request_context->set_status_code(static_cast<int>(EC_BADARGS));
        SetErrorOnCollector(request_context, EC_BADARGS);
        return;
    }

    ErrorCode ec = manager_ ? manager_->CreateInstanceGroup(group) : EC_ERROR;

    SetPbResponseHeader(response->mutable_header(), ec);
    request_context->set_status_code(static_cast<int>(ec));
    SetErrorOnCollector(request_context, ec);
}

void OptimizerServiceImpl::UpdateInstanceGroup(RequestContext *request_context,
                                               const proto::optimizer::UpdateInstanceGroupRequest *request,
                                               proto::optimizer::CommonResponse *response) {
    request_context->set_api_name("UpdateInstanceGroup");
    OptimizerCallGuard guard(request_context, metrics_reporter_.get());

    auto group = ConvertProtoToInstanceGroup(request->instance_group());

    std::string invalid_fields;
    if (!group.ValidateRequiredFields(invalid_fields)) {
        KVCM_LOG_ERROR("UpdateInstanceGroup: validation failed, invalid_fields[%s]", invalid_fields.c_str());
        SetPbResponseHeader(response->mutable_header(), EC_BADARGS);
        request_context->set_status_code(static_cast<int>(EC_BADARGS));
        SetErrorOnCollector(request_context, EC_BADARGS);
        return;
    }

    ErrorCode ec = manager_ ? manager_->UpdateInstanceGroup(group) : EC_ERROR;

    SetPbResponseHeader(response->mutable_header(), ec);
    request_context->set_status_code(static_cast<int>(ec));
    SetErrorOnCollector(request_context, ec);
}

void OptimizerServiceImpl::RemoveInstanceGroup(RequestContext *request_context,
                                               const proto::optimizer::RemoveInstanceGroupRequest *request,
                                               proto::optimizer::CommonResponse *response) {
    request_context->set_api_name("RemoveInstanceGroup");
    OptimizerCallGuard guard(request_context, metrics_reporter_.get());

    ErrorCode ec = manager_ ? manager_->RemoveInstanceGroup(request->name()) : EC_ERROR;

    SetPbResponseHeader(response->mutable_header(), ec);
    request_context->set_status_code(static_cast<int>(ec));
    SetErrorOnCollector(request_context, ec);
}

void OptimizerServiceImpl::GetInstanceGroup(RequestContext *request_context,
                                            const proto::optimizer::GetInstanceGroupRequest *request,
                                            proto::optimizer::GetInstanceGroupResponse *response) {
    request_context->set_api_name("GetInstanceGroup");
    OptimizerCallGuard guard(request_context, metrics_reporter_.get());

    auto registry = manager_->registry_manager();
    if (!registry) {
        SetPbResponseHeader(response->mutable_header(), EC_ERROR);
        request_context->set_status_code(static_cast<int>(EC_ERROR));
        SetErrorOnCollector(request_context, EC_ERROR);
        return;
    }
    auto group = registry->GetInstanceGroup(request->name());
    if (!group) {
        SetPbResponseHeader(response->mutable_header(), EC_NOENT);
        request_context->set_status_code(static_cast<int>(EC_NOENT));
        SetErrorOnCollector(request_context, EC_NOENT);
        return;
    }
    ConvertInstanceGroupToProto(*group, response->mutable_instance_group());
    SetPbResponseHeader(response->mutable_header(), EC_OK);
    request_context->set_status_code(static_cast<int>(EC_OK));
}

void OptimizerServiceImpl::ListInstanceGroups(RequestContext *request_context,
                                              const proto::optimizer::ListInstanceGroupsRequest *,
                                              proto::optimizer::ListInstanceGroupsResponse *response) {
    request_context->set_api_name("ListInstanceGroups");
    OptimizerCallGuard guard(request_context, metrics_reporter_.get());

    auto registry = manager_->registry_manager();
    if (!registry) {
        SetPbResponseHeader(response->mutable_header(), EC_ERROR);
        request_context->set_status_code(static_cast<int>(EC_ERROR));
        SetErrorOnCollector(request_context, EC_ERROR);
        return;
    }
    auto groups = registry->ListInstanceGroups();
    for (const auto &g : groups) {
        ConvertInstanceGroupToProto(*g, response->add_instance_groups());
    }
    SetPbResponseHeader(response->mutable_header(), EC_OK);
    request_context->set_status_code(static_cast<int>(EC_OK));
}

// Instance management — call manager (which internally persists via registry)

void OptimizerServiceImpl::RegisterInstance(RequestContext *request_context,
                                            const proto::optimizer::OptimizerRegisterInstanceRequest *request,
                                            proto::optimizer::OptimizerRegisterInstanceResponse *response) {
    request_context->set_api_name("RegisterInstance");
    OptimizerCallGuard guard(request_context, metrics_reporter_.get());

    auto instance_info = ConvertProtoToInstanceInfo(*request);

    RegisterInstanceResult result;
    ErrorCode ec = manager_->RegisterInstance(instance_info, result);

    SetPbResponseHeader(response->mutable_header(), ec);
    request_context->set_status_code(static_cast<int>(ec));
    SetErrorOnCollector(request_context, ec);

    if (ec == EC_OK) {
        // Clear stale Prometheus series before the next ReportInterval writes
        // new labels (e.g. different capacity_gb tiers or age buckets).
        if (metrics_reporter_) {
            metrics_reporter_->RemoveInstanceMetrics(instance_info.instance_id());
        }
        for (int64_t cap : result.estimated_capacity_blocks) {
            response->add_estimated_capacity_blocks(cap);
        }
        response->set_size_full(result.size_full);
        response->set_size_full_linear(result.size_full_linear);
    }
}

void OptimizerServiceImpl::RemoveInstance(RequestContext *request_context,
                                          const proto::optimizer::OptimizerRemoveInstanceRequest *request,
                                          proto::optimizer::OptimizerRemoveInstanceResponse *response) {
    request_context->set_api_name("RemoveInstance");
    OptimizerCallGuard guard(request_context, metrics_reporter_.get());

    const auto &instance_id = request->instance_id();
    ErrorCode ec = manager_->RemoveInstance(instance_id);

    if (ec == EC_OK && metrics_reporter_) {
        metrics_reporter_->RemoveInstanceMetrics(instance_id);
    }

    SetPbResponseHeader(response->mutable_header(), ec);
    request_context->set_status_code(static_cast<int>(ec));
    SetErrorOnCollector(request_context, ec);
}

void OptimizerServiceImpl::GetInstance(RequestContext *request_context,
                                       const proto::optimizer::OptimizerGetInstanceRequest *request,
                                       proto::optimizer::OptimizerGetInstanceResponse *response) {
    request_context->set_api_name("GetInstance");
    OptimizerCallGuard guard(request_context, metrics_reporter_.get());

    ErrorCode ec = manager_->GetInstanceState(request->instance_id(), [&](const InstanceState &state) {
        const auto &info = *state.instance_info;
        response->set_instance_group(info.instance_group_name());
        response->set_instance_id(info.instance_id());
        response->set_block_size(info.block_size());
        for (const auto &spec : info.location_spec_infos()) {
            auto *pb_spec = response->add_location_spec_infos();
            pb_spec->set_name(spec.name());
            pb_spec->set_size(spec.size());
        }
        for (const auto &group : info.location_spec_groups()) {
            auto *pb_group = response->add_location_spec_groups();
            pb_group->set_name(group.name());
            for (const auto &spec_name : group.spec_names()) {
                pb_group->add_spec_names(spec_name);
            }
        }
        response->set_linear_step(info.linear_step());
        auto *state_info = response->mutable_optimizer_state_info();
        state_info->set_full_location_spec_group_name(info.optimizer_state_info().full_location_spec_group_name());
        state_info->set_linear_location_spec_group_name(info.optimizer_state_info().linear_location_spec_group_name());
    });

    SetPbResponseHeader(response->mutable_header(), ec);
    request_context->set_status_code(static_cast<int>(ec));
    if (ec != EC_OK) {
        SetErrorOnCollector(request_context, ec);
    }
}

ErrorCode OptimizerServiceImpl::ApplyKvcmConfiguration(const proto::optimizer::KvcmConfigurationResponse &configuration,
                                                       const KvcmEventSubscriptionConfig &config,
                                                       std::unordered_set<std::string> *unsupported_instance_ids,
                                                       std::unordered_map<std::string, int32_t> *registered_instances) {
    std::unordered_set<std::string> unsupported;
    std::unordered_map<std::string, int32_t> registered;
    if (!manager_ || !manager_->registry_manager()) {
        return EC_ERROR;
    }
    if (!config.Validate()) {
        return EC_BADARGS;
    }
    auto registry = manager_->registry_manager();
    for (const auto &source : configuration.instance_groups()) {
        OptimizerInstanceGroup group;
        group.set_name(source.name());
        if (!config.mock_cache_layout()) {
            group.set_capacity_gb({static_cast<double>(source.capacity_bytes()) / (1024.0 * 1024 * 1024)});
        } else {
            group.set_capacity_gb(config.capacity_gb());
        }
        group.set_ttl_seconds_list(config.ttl_seconds_list());
        group.set_eviction_policy("lru");
        group.set_enable_prefix_hash(true);
        group.set_enable_theoretical_max_cache(true);
        std::string invalid;
        if (!group.ValidateRequiredFields(invalid)) {
            return EC_BADARGS;
        }
        auto existing_group = registry->GetInstanceGroup(source.name());
        if (!existing_group) {
            const auto ec = manager_->CreateInstanceGroup(group);
            if (ec != EC_OK && ec != EC_DUPLICATE_ENTITY) {
                return ec;
            }
            existing_group = registry->GetInstanceGroup(source.name());
        }
        if (!existing_group || existing_group->ToJsonString() != group.ToJsonString()) {
            KVCM_LOG_ERROR("Conflicting auto-registration configuration for group[%s]", source.name().c_str());
            return EC_BADARGS;
        }
    }
    for (const auto &source : configuration.instances()) {
        const auto block_size = source.block_size();
        if (source.instance_id().empty() || block_size <= 0) {
            KVCM_LOG_ERROR("Ignore invalid KVCM instance[%s], block_size=%d", source.instance_id().c_str(), block_size);
            unsupported.insert(source.instance_id());
            continue;
        }
        const auto analysis_id = KvcmAnalysisInstanceId(source.instance_id(), block_size);
        bool matches = true;
        const auto exists = manager_->GetInstanceState(analysis_id, [&](const InstanceState &state) {
            matches = state.instance_info->block_size() == block_size &&
                      state.instance_info->instance_group_name() == source.instance_group_name() &&
                      state.linear_step == 0 && (!config.mock_cache_layout() || state.size_full == 1);
        });
        if (exists == EC_OK) {
            if (!matches) {
                KVCM_LOG_ERROR("KVCM configuration differs for registered instance[%s]", source.instance_id().c_str());
                unsupported.insert(source.instance_id());
                continue;
            }
            registered.emplace(source.instance_id(), block_size);
            continue;
        }
        std::vector<LocationSpecInfo> spec_infos;
        std::vector<LocationSpecGroup> spec_groups;
        if (!config.mock_cache_layout()) {
            // Keep the original KVCM spec mapping: a single group is full-attention.
            if (source.location_spec_groups_size() > 1) {
                KVCM_LOG_WARN("Ignore unsupported multi-group instance[%s]", source.instance_id().c_str());
                unsupported.insert(source.instance_id());
                continue;
            }
            for (const auto &spec : source.location_spec_infos()) {
                spec_infos.emplace_back(spec.name(), spec.size());
            }
            if (source.location_spec_groups().empty()) {
                std::vector<std::string> names;
                for (const auto &spec : spec_infos) {
                    names.push_back(spec.name());
                }
                spec_groups.emplace_back("full", names);
            } else {
                const auto &group = source.location_spec_groups(0);
                spec_groups.emplace_back(
                    group.name(), std::vector<std::string>(group.spec_names().begin(), group.spec_names().end()));
            }
        } else {
            spec_infos.emplace_back("block", 1);
            spec_groups.emplace_back("full", std::vector<std::string>{"block"});
        }
        OptimizerInstanceInfo instance(
            source.instance_group_name(), analysis_id, block_size, spec_infos, spec_groups, 0, OptimizerStateInfo());
        RegisterInstanceResult result;
        auto ec = manager_->RegisterInstance(instance, result);
        if (ec != EC_OK) {
            KVCM_LOG_ERROR("Ignore KVCM instance[%s]: registration failed, ec=%d", source.instance_id().c_str(), ec);
            unsupported.insert(source.instance_id());
            continue;
        }
        registered.emplace(source.instance_id(), block_size);
    }
    if (unsupported_instance_ids) {
        *unsupported_instance_ids = std::move(unsupported);
    }
    if (registered_instances) {
        *registered_instances = std::move(registered);
    }
    return EC_OK;
}

void OptimizerServiceImpl::TraceQuery(RequestContext *request_context,
                                      const proto::optimizer::TraceQueryRequest *request,
                                      proto::optimizer::TraceQueryResponse *response) {
    request_context->set_api_name("TraceQuery");
    OptimizerCallGuard guard(request_context, metrics_reporter_.get());

    const ErrorCode ec = ExecuteTraceQuery(*request, response);

    SetPbResponseHeader(response->mutable_header(), ec);
    request_context->set_status_code(static_cast<int>(ec));

    if (ec == EC_OK) {
        auto *collector = dynamic_cast<OptimizerServiceMetricsCollector *>(request_context->metrics_collector());
        if (collector) {
            collector->set_instance_id(request->instance_id());
            collector->set_total_blocks(response->total_blocks());
            collector->set_input_token_len(response->input_token_len());
            collector->SetTtlResults(*response);
        }
    } else {
        SetErrorOnCollector(request_context, ec);
    }
}

ErrorCode OptimizerServiceImpl::ExecuteTraceQuery(const proto::optimizer::TraceQueryRequest &request,
                                                  proto::optimizer::TraceQueryResponse *response) {
    std::vector<int64_t> block_keys(request.block_keys().begin(), request.block_keys().end());
    int64_t input_token_len = request.input_token_len();
    if (input_token_len == 0 && request.token_ids_size() > 0) {
        input_token_len = request.token_ids_size();
    }

    TraceQueryResult result;
    const ErrorCode ec =
        manager_->TraceQuery(request.instance_id(), block_keys, input_token_len, request.timestamp_ns(), result);
    if (ec != EC_OK || !response) {
        return ec;
    }

    response->set_total_blocks(result.total_blocks);
    response->set_input_token_len(result.input_token_len);
    response->clear_ttl_results();
    for (const auto &ttl_result : result.ttl_results) {
        const auto &result = ttl_result;
        auto *ttl_response = response->add_ttl_results();
        ttl_response->set_ttl_seconds(result.ttl_seconds);
        for (size_t i = 0; i < result.capacity_gb.size() && i < result.hit_count_per_capacity.size(); i++) {
            auto *capacity_result = ttl_response->add_capacity_results();
            capacity_result->set_capacity_gb(result.capacity_gb[i]);
            capacity_result->set_cache_hit_count(result.hit_count_per_capacity[i]);
            if (i < result.hit_rate_per_capacity.size()) {
                capacity_result->set_hit_rate(result.hit_rate_per_capacity[i]);
            }
            if (i < result.unique_keys_per_capacity.size()) {
                capacity_result->set_current_unique_keys(result.unique_keys_per_capacity[i]);
            }
        }
        ttl_response->mutable_theoretical_result()->set_max_hit_count(result.max_hit_count);
        ttl_response->mutable_theoretical_result()->set_current_unique_keys(result.theoretical_unique_keys);
        ttl_response->mutable_theoretical_result()->set_hit_rate(result.max_hit_rate);

        if (event_manager_) {
            auto event = std::make_shared<OptimizerQueryHitEvent>(
                TtlMetricsInstanceId(request.instance_id(), result.ttl_seconds));
            event->SetEventTriggerTime();
            event->SetAdditionalArgs(
                request.trace_id(), request.timestamp_ns(), response->input_token_len(), response->total_blocks());
            for (const auto &capacity_result : ttl_response->capacity_results()) {
                event->AddCapacityResult(capacity_result.capacity_gb(),
                                         capacity_result.cache_hit_count(),
                                         capacity_result.hit_rate(),
                                         capacity_result.current_unique_keys());
            }
            const auto &theoretical_result = ttl_response->theoretical_result();
            event->SetTheoreticalResult(theoretical_result.max_hit_count(),
                                        theoretical_result.hit_rate(),
                                        theoretical_result.current_unique_keys());
            event_manager_->Publish(event);
        }
    }

    return EC_OK;
}

void OptimizerServiceImpl::ListInstances(RequestContext *request_context,
                                         const proto::optimizer::OptimizerListInstancesRequest *request,
                                         proto::optimizer::OptimizerListInstancesResponse *response) {
    request_context->set_api_name("ListInstances");
    OptimizerCallGuard guard(request_context, metrics_reporter_.get());

    std::vector<InstanceSummary> summaries;
    ErrorCode ec = manager_->ListInstances(request->instance_group(), summaries);

    SetPbResponseHeader(response->mutable_header(), ec);
    request_context->set_status_code(static_cast<int>(ec));
    SetErrorOnCollector(request_context, ec);

    if (ec == EC_OK) {
        for (const auto &s : summaries) {
            auto *pb = response->add_instances();
            pb->set_instance_id(s.instance_id);
            pb->set_ttl_seconds(s.ttl_seconds);
            pb->set_instance_group(s.instance_group);
            pb->set_total_queries(s.total_queries);
            pb->set_total_blocks_queried(s.total_blocks_queried);
            pb->set_total_input_tokens(s.total_input_tokens);
            pb->mutable_theoretical_summary()->set_total_max_hits(s.total_max_hits);
            pb->mutable_theoretical_summary()->set_max_hit_rate(s.max_hit_rate);
            auto *debug = pb->mutable_debug_info();
            debug->set_block_size(s.block_size);
            debug->set_unique_keys(s.unique_keys);
            debug->set_linear_step(s.linear_step);
            debug->set_eviction_count(s.eviction_count);
            debug->set_memory_usage_bytes(s.memory_usage_bytes);
            debug->set_kv_cache_usage_bytes(s.kv_cache_usage_bytes);
            debug->set_ttl_eviction_count(s.ttl_eviction_count);
            for (const auto &cap : s.per_capacity_hit_rates) {
                auto *pb_cap = pb->add_capacity_summaries();
                pb_cap->set_capacity_gb(cap.capacity_gb);
                pb_cap->set_total_hits(cap.total_hits);
                pb_cap->set_hit_rate(cap.hit_rate);
            }
        }
    }
}

void OptimizerServiceImpl::ResetStats(RequestContext *request_context,
                                      const proto::optimizer::OptimizerResetStatsRequest *request,
                                      proto::optimizer::OptimizerResetStatsResponse *response) {
    request_context->set_api_name("ResetStats");
    OptimizerCallGuard guard(request_context, metrics_reporter_.get());

    const auto &instance_id = request->instance_id();
    ErrorCode ec = manager_->ResetStats(instance_id);

    if (ec == EC_OK && metrics_reporter_) {
        metrics_reporter_->RemoveInstanceMetrics(instance_id);
    }

    SetPbResponseHeader(response->mutable_header(), ec);
    request_context->set_status_code(static_cast<int>(ec));
    SetErrorOnCollector(request_context, ec);
}

} // namespace kv_cache_manager
