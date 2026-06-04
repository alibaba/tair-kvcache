#include "kv_cache_manager/online_optimizer/server/optimizer_service_impl.h"

#include "kv_cache_manager/common/error_code.h"
#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/online_optimizer/config/optimizer_instance_group.h"
#include "kv_cache_manager/online_optimizer/config/optimizer_instance_info.h"
#include "kv_cache_manager/online_optimizer/config/optimizer_registry_manager.h"
#include "kv_cache_manager/online_optimizer/manager/online_optimizer_manager.h"
#include "kv_cache_manager/online_optimizer/metrics/optimizer_metrics_collector.h"
#include "kv_cache_manager/online_optimizer/server/optimizer_call_guard.h"

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
    group.set_enabled(pb.enabled());
    std::vector<double> caps(pb.capacity_gb().begin(), pb.capacity_gb().end());
    group.set_capacity_gb(caps);
    group.set_primary_capacity_index(pb.primary_capacity_index());
    group.set_indexer_type(pb.indexer_type().empty() ? "fenwick_lru" : pb.indexer_type());
    group.set_max_key_count(pb.max_key_count());
    return group;
}

void ConvertInstanceGroupToProto(const OptimizerInstanceGroup &group,
                                  proto::optimizer::OptimizerInstanceGroupProto *pb) {
    pb->set_name(group.name());
    pb->set_enabled(group.enabled());
    for (double cap : group.capacity_gb()) {
        pb->add_capacity_gb(cap);
    }
    pb->set_primary_capacity_index(group.primary_capacity_index());
    pb->set_indexer_type(group.indexer_type());
    pb->set_max_key_count(group.max_key_count());
}

OptimizerInstanceInfo ConvertProtoToInstanceInfo(
    const proto::optimizer::OptimizerRegisterInstanceRequest &request) {
    std::vector<LocationSpecInfo> specs;
    specs.reserve(request.location_spec_infos().size());
    for (const auto &s : request.location_spec_infos()) {
        specs.emplace_back(s.name(), s.size());
    }

    std::vector<LocationSpecGroup> groups;
    groups.reserve(request.location_spec_groups().size());
    for (const auto &g : request.location_spec_groups()) {
        std::vector<std::string> names(g.spec_names().begin(), g.spec_names().end());
        groups.emplace_back(g.name(), names);
    }

    return OptimizerInstanceInfo(
        request.instance_group(), request.instance_id(),
        request.block_size(), specs, groups,
        request.linear_step(), request.full_group_name());
}

void SetErrorOnCollector(RequestContext *request_context, ErrorCode ec) {
    if (ec == EC_OK) return;
    auto *collector = dynamic_cast<OptimizerServiceMetricsCollector *>(request_context->metrics_collector());
    KVCM_METRICS_COLLECTOR_SET_METRICS(collector, service, error_code, static_cast<double>(ec));
}

} // namespace

OptimizerServiceImpl::OptimizerServiceImpl(std::shared_ptr<OnlineOptimizerManager> manager,
                                             std::shared_ptr<OptimizerMetricsReporter> metrics_reporter)
    : manager_(std::move(manager))
    , metrics_reporter_(std::move(metrics_reporter)) {}

// InstanceGroup CRUD — directly call registry_manager

void OptimizerServiceImpl::CreateInstanceGroup(RequestContext *request_context,
                                                const proto::optimizer::CreateInstanceGroupRequest *request,
                                                proto::optimizer::CommonResponse *response) {
    request_context->set_api_name("CreateInstanceGroup");
    OptimizerCallGuard guard(request_context, metrics_reporter_.get());

    auto registry = manager_->registry_manager();
    auto group = ConvertProtoToInstanceGroup(request->instance_group());
    ErrorCode ec = registry ? registry->CreateInstanceGroup(group) : EC_ERROR;

    SetPbResponseHeader(response->mutable_header(), ec);
    request_context->set_status_code(static_cast<int>(ec));
    SetErrorOnCollector(request_context, ec);
}

void OptimizerServiceImpl::UpdateInstanceGroup(RequestContext *request_context,
                                                const proto::optimizer::UpdateInstanceGroupRequest *request,
                                                proto::optimizer::CommonResponse *response) {
    request_context->set_api_name("UpdateInstanceGroup");
    OptimizerCallGuard guard(request_context, metrics_reporter_.get());

    auto registry = manager_->registry_manager();
    auto group = ConvertProtoToInstanceGroup(request->instance_group());
    ErrorCode ec = registry ? registry->UpdateInstanceGroup(group) : EC_ERROR;

    SetPbResponseHeader(response->mutable_header(), ec);
    request_context->set_status_code(static_cast<int>(ec));
    SetErrorOnCollector(request_context, ec);
}

void OptimizerServiceImpl::RemoveInstanceGroup(RequestContext *request_context,
                                                const proto::optimizer::RemoveInstanceGroupRequest *request,
                                                proto::optimizer::CommonResponse *response) {
    request_context->set_api_name("RemoveInstanceGroup");
    OptimizerCallGuard guard(request_context, metrics_reporter_.get());

    auto registry = manager_->registry_manager();
    ErrorCode ec = registry ? registry->RemoveInstanceGroup(request->name()) : EC_ERROR;

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

    auto registry = manager_->registry_manager();
    auto group_ptr = registry ? registry->GetInstanceGroup(request->instance_group()) : nullptr;
    if (!group_ptr) {
        SetPbResponseHeader(response->mutable_header(), EC_NOENT);
        request_context->set_status_code(static_cast<int>(EC_NOENT));
        SetErrorOnCollector(request_context, EC_NOENT);
        return;
    }

    RegisterInstanceResult result;
    ErrorCode ec = manager_->RegisterInstance(instance_info, *group_ptr, result);

    SetPbResponseHeader(response->mutable_header(), ec);
    request_context->set_status_code(static_cast<int>(ec));
    SetErrorOnCollector(request_context, ec);

    if (ec == EC_OK) {
        for (int64_t cap : result.capacity_blocks) {
            response->add_capacity_blocks(cap);
        }
        response->set_avg_bytes_per_block(result.avg_bytes_per_block);
        response->set_size_full_only(result.size_full_only);
        response->set_size_full_linear(result.size_full_linear);
    }
}

void OptimizerServiceImpl::RemoveInstance(RequestContext *request_context,
                                           const proto::optimizer::OptimizerRemoveInstanceRequest *request,
                                           proto::optimizer::OptimizerRemoveInstanceResponse *response) {
    request_context->set_api_name("RemoveInstance");
    OptimizerCallGuard guard(request_context, metrics_reporter_.get());

    ErrorCode ec = manager_->RemoveInstance(request->instance_id());

    SetPbResponseHeader(response->mutable_header(), ec);
    request_context->set_status_code(static_cast<int>(ec));
    SetErrorOnCollector(request_context, ec);
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
        response->set_cache_hit_count(result.cache_hit_count);
        response->set_total_blocks(result.total_blocks);
        response->set_current_unique_keys(result.current_unique_keys);

        auto *collector = dynamic_cast<OptimizerServiceMetricsCollector *>(
            request_context->metrics_collector());
        if (collector) {
            collector->set_instance_id(request->instance_id());
            collector->set_total_blocks(result.total_blocks);
            collector->set_cache_hit_count(result.cache_hit_count);
            std::vector<PerCapacityHitInfo> per_cap;
            for (size_t i = 0; i < result.capacity_gb.size() && i < result.hit_count_per_capacity.size(); i++) {
                per_cap.push_back({result.capacity_gb[i], result.hit_count_per_capacity[i]});
            }
            collector->set_per_capacity_hits(std::move(per_cap));
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
            pb->set_block_size(s.block_size);
            pb->set_total_queries(s.total_queries);
            pb->set_total_blocks_queried(s.total_blocks_queried);
            pb->set_total_hits(s.total_hits);
            pb->set_hit_rate(s.hit_rate);
            pb->set_unique_keys(s.unique_keys);
            pb->set_avg_bytes_per_block(s.avg_bytes_per_block);
            pb->set_linear_step(s.linear_step);
            pb->set_peak_unique_keys(s.peak_unique_keys);
        }
    }
}

void OptimizerServiceImpl::ResetStats(RequestContext *request_context,
                                       const proto::optimizer::OptimizerResetStatsRequest *request,
                                       proto::optimizer::OptimizerResetStatsResponse *response) {
    request_context->set_api_name("ResetStats");
    OptimizerCallGuard guard(request_context, metrics_reporter_.get());

    ErrorCode ec = manager_->ResetStats(request->instance_id());

    SetPbResponseHeader(response->mutable_header(), ec);
    request_context->set_status_code(static_cast<int>(ec));
    SetErrorOnCollector(request_context, ec);
}

} // namespace kv_cache_manager
