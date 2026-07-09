#include "kv_cache_manager/optimizer/service/optimizer_service_impl.h"

#include "kv_cache_manager/common/error_code.h"
#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/optimizer/config/optimizer_instance_group.h"
#include "kv_cache_manager/optimizer/config/optimizer_instance_info.h"
#include "kv_cache_manager/optimizer/config/optimizer_registry_manager.h"
#include "kv_cache_manager/optimizer/online_runtime/online_optimizer_manager.h"
#include "kv_cache_manager/optimizer/service/metrics/optimizer_metrics_collector.h"
#include "kv_cache_manager/optimizer/service/metrics/optimizer_metrics_reporter.h"
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
    group.set_ttl_seconds(pb.ttl_seconds());
    return group;
}

void ConvertInstanceGroupToProto(const OptimizerInstanceGroup &group,
                                 proto::optimizer::OptimizerInstanceGroupProto *pb) {
    pb->set_name(group.name());
    for (double cap : group.capacity_gb()) {
        pb->add_capacity_gb(cap);
    }
    pb->set_shared_group_quota(group.shared_group_quota());
    pb->set_ttl_seconds(group.ttl_seconds());
    pb->set_enable_theoretical_max_cache(group.enable_theoretical_max_cache());
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
                                           std::shared_ptr<OptimizerMetricsReporter> metrics_reporter)
    : manager_(std::move(manager)), metrics_reporter_(std::move(metrics_reporter)) {}

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
        response->set_size_full_only(result.size_full_only);
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

void OptimizerServiceImpl::TraceQuery(RequestContext *request_context,
                                      const proto::optimizer::TraceQueryRequest *request,
                                      proto::optimizer::TraceQueryResponse *response) {
    request_context->set_api_name("TraceQuery");
    OptimizerCallGuard guard(request_context, metrics_reporter_.get());

    std::vector<int64_t> block_keys(request->block_keys().begin(), request->block_keys().end());

    TraceQueryResult result;
    ErrorCode ec = manager_->TraceQuery(request->instance_id(), block_keys, result);

    SetPbResponseHeader(response->mutable_header(), ec);
    request_context->set_status_code(static_cast<int>(ec));

    if (ec == EC_OK) {
        response->set_total_blocks(result.total_blocks);
        for (size_t i = 0; i < result.capacity_gb.size() && i < result.hit_count_per_capacity.size(); i++) {
            auto *pb_cap = response->add_capacity_results();
            pb_cap->set_capacity_gb(result.capacity_gb[i]);
            pb_cap->set_cache_hit_count(result.hit_count_per_capacity[i]);
            pb_cap->set_hit_rate(result.total_blocks > 0 ? static_cast<double>(result.hit_count_per_capacity[i]) /
                                                               static_cast<double>(result.total_blocks)
                                                         : 0.0);
            if (i < result.unique_keys_per_capacity.size()) {
                pb_cap->set_current_unique_keys(result.unique_keys_per_capacity[i]);
            }
        }
        response->mutable_theoretical_result()->set_max_hit_count(result.max_hit_count);
        response->mutable_theoretical_result()->set_current_unique_keys(result.theoretical_unique_keys);

        auto *collector = dynamic_cast<OptimizerServiceMetricsCollector *>(request_context->metrics_collector());
        if (collector) {
            collector->set_instance_id(request->instance_id());
            collector->set_total_blocks(result.total_blocks);
            collector->set_cache_hit_count(result.cache_hit_count);
            std::vector<PerCapacityHitInfo> per_cap;
            for (size_t i = 0; i < result.capacity_gb.size() && i < result.hit_count_per_capacity.size(); i++) {
                per_cap.push_back({result.capacity_gb[i], result.hit_count_per_capacity[i]});
            }
            collector->set_per_capacity_hits(std::move(per_cap));
            collector->set_max_hit_count(result.max_hit_count);
            if (result.total_blocks > 0 && result.max_hit_count >= 0) {
                collector->set_max_hit_rate(static_cast<double>(result.max_hit_count) /
                                            static_cast<double>(result.total_blocks));
            }
        }
    } else {
        SetErrorOnCollector(request_context, ec);
    }
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
            pb->set_instance_group(s.instance_group);
            pb->set_total_queries(s.total_queries);
            pb->set_total_blocks_queried(s.total_blocks_queried);
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
