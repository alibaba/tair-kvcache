#include "kv_cache_manager/optimizer/service/optimizer_service_impl.h"

#include <algorithm>
#include <chrono>
#include <unordered_set>
#include <utility>
#include <vector>

#include "kv_cache_manager/common/error_code.h"
#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/event/event_manager.h"
#include "kv_cache_manager/event/spec_events/optimizer_query_hit_event.h"
#include "kv_cache_manager/metrics/metrics_registry.h"
#include "kv_cache_manager/optimizer/config/optimizer_instance_group.h"
#include "kv_cache_manager/optimizer/config/optimizer_instance_info.h"
#include "kv_cache_manager/optimizer/config/optimizer_registry_manager.h"
#include "kv_cache_manager/optimizer/manager/online_runtime/online_optimizer_manager.h"
#include "kv_cache_manager/optimizer/metrics/optimizer_metrics_collector.h"
#include "kv_cache_manager/optimizer/metrics/optimizer_metrics_reporter.h"
#include "kv_cache_manager/optimizer/quota_runtime/quota_plan.h"
#include "kv_cache_manager/optimizer/service/optimizer_call_guard.h"

namespace kv_cache_manager {

namespace {

constexpr long double kBytesPerGb = 1024.0L * 1024.0L * 1024.0L;
constexpr int64_t kKvcmAutoGroupTtlSeconds = 24 * 60 * 60;

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
    pb->set_ttl_seconds(group.ttl_seconds());
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
                                           std::shared_ptr<EventManager> event_manager,
                                           std::shared_ptr<InMemoryQuotaPlanStore> quota_plan_store,
                                           std::shared_ptr<MetricsRegistry> metrics_registry)
    : manager_(std::move(manager))
    , metrics_reporter_(std::move(metrics_reporter))
    , event_manager_(std::move(event_manager))
    , quota_plan_store_(std::move(quota_plan_store))
    , metrics_registry_(std::move(metrics_registry)) {}

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
                                                       std::unordered_set<std::string> &unsupported_instance_ids,
                                                       const std::vector<double> &capacity_gb_override) {
    unsupported_instance_ids.clear();
    if (!manager_) {
        KVCM_LOG_ERROR("ApplyKvcmConfiguration: optimizer manager is null");
        return EC_ERROR;
    }
    auto registry = manager_->registry_manager();
    if (!registry) {
        KVCM_LOG_ERROR("ApplyKvcmConfiguration: optimizer registry manager is null");
        return EC_ERROR;
    }

    // TODO: Reconcile existing Optimizer groups and instances when KVCM configuration changes.
    // The current synchronization only creates missing entries and does not update existing ones.
    std::unordered_set<std::string> available_groups;
    std::size_t created_groups = 0;
    std::size_t registered_instances = 0;
    for (const auto &source : configuration.instance_groups()) {
        if (source.name().empty() || source.capacity_bytes() <= 0) {
            KVCM_LOG_ERROR("ApplyKvcmConfiguration: invalid KVCM instance group[%s], capacity_bytes=%ld",
                           source.name().c_str(),
                           source.capacity_bytes());
            return EC_BADARGS;
        }
        if (!registry->GetInstanceGroup(source.name())) {
            OptimizerInstanceGroup group;
            group.set_name(source.name());
            if (capacity_gb_override.empty()) {
                group.set_capacity_gb(
                    {static_cast<double>(static_cast<long double>(source.capacity_bytes()) / kBytesPerGb)});
            } else {
                group.set_capacity_gb(capacity_gb_override);
            }
            group.set_eviction_policy("lru");
            group.set_enable_prefix_hash(true);
            group.set_enable_theoretical_max_cache(true);
            group.set_ttl_seconds(kKvcmAutoGroupTtlSeconds);

            std::string invalid_fields;
            if (!group.ValidateRequiredFields(invalid_fields)) {
                KVCM_LOG_ERROR("ApplyKvcmConfiguration: invalid mapped instance group[%s], invalid_fields[%s]",
                               source.name().c_str(),
                               invalid_fields.c_str());
                return EC_BADARGS;
            }

            const ErrorCode ec = manager_->CreateInstanceGroup(group);
            if (ec != EC_OK && ec != EC_DUPLICATE_ENTITY) {
                KVCM_LOG_ERROR("ApplyKvcmConfiguration: create instance group[%s] failed, ec=%d",
                               source.name().c_str(),
                               static_cast<int>(ec));
                return ec;
            }
            if (ec == EC_OK) {
                ++created_groups;
            }
        }
        available_groups.insert(source.name());
    }

    for (const auto &source : configuration.instances()) {
        if (source.instance_id().empty()) {
            KVCM_LOG_ERROR("ApplyKvcmConfiguration: empty KVCM instance id");
            return EC_BADARGS;
        }
        if (manager_->GetInstanceState(source.instance_id(), [](const InstanceState &) {}) == EC_OK) {
            continue;
        }
        if (source.location_spec_groups_size() > 1) {
            KVCM_LOG_WARN("ApplyKvcmConfiguration: ignore unsupported multi-group instance[%s], groups=%d",
                          source.instance_id().c_str(),
                          source.location_spec_groups_size());
            unsupported_instance_ids.insert(source.instance_id());
            continue;
        }
        if (available_groups.find(source.instance_group_name()) == available_groups.end()) {
            KVCM_LOG_ERROR("ApplyKvcmConfiguration: instance[%s] group[%s] is unavailable",
                           source.instance_id().c_str(),
                           source.instance_group_name().c_str());
            return EC_NOENT;
        }
        std::vector<LocationSpecInfo> spec_infos;
        spec_infos.reserve(source.location_spec_infos_size());
        for (const auto &spec : source.location_spec_infos()) {
            spec_infos.emplace_back(spec.name(), spec.size());
        }

        std::vector<LocationSpecGroup> spec_groups;
        spec_groups.reserve(source.location_spec_groups_size());
        for (const auto &source_group : source.location_spec_groups()) {
            std::vector<std::string> spec_names(source_group.spec_names().begin(), source_group.spec_names().end());
            spec_groups.emplace_back(source_group.name(), spec_names);
        }

        OptimizerInstanceInfo instance(source.instance_group_name(),
                                       source.instance_id(),
                                       source.block_size(),
                                       spec_infos,
                                       spec_groups,
                                       0,
                                       OptimizerStateInfo());
        RegisterInstanceResult result;
        const ErrorCode ec = manager_->RegisterInstance(instance, result);
        if (ec != EC_OK) {
            KVCM_LOG_ERROR("ApplyKvcmConfiguration: register instance[%s] failed, ec=%d",
                           source.instance_id().c_str(),
                           static_cast<int>(ec));
            return ec;
        }
        ++registered_instances;
    }

    KVCM_LOG_INFO("ApplyKvcmConfiguration: groups=%d instances=%d created_groups=%zu registered_instances=%zu "
                  "unsupported_instances=%zu",
                  configuration.instance_groups_size(),
                  configuration.instances_size(),
                  created_groups,
                  registered_instances,
                  unsupported_instance_ids.size());
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
            std::vector<PerCapacityHitInfo> per_cap;
            per_cap.reserve(response->capacity_results_size());
            for (const auto &capacity_result : response->capacity_results()) {
                per_cap.push_back(
                    {capacity_result.capacity_gb(), capacity_result.cache_hit_count(), capacity_result.hit_rate()});
            }
            collector->set_per_capacity_hits(std::move(per_cap));
            collector->set_max_hit_count(response->theoretical_result().max_hit_count());
            if (response->theoretical_result().max_hit_count() >= 0) {
                collector->set_max_hit_rate(response->theoretical_result().hit_rate());
            }
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
    for (size_t i = 0; i < result.capacity_gb.size() && i < result.hit_count_per_capacity.size(); i++) {
        auto *capacity_result = response->add_capacity_results();
        capacity_result->set_capacity_gb(result.capacity_gb[i]);
        capacity_result->set_cache_hit_count(result.hit_count_per_capacity[i]);
        if (i < result.hit_rate_per_capacity.size()) {
            capacity_result->set_hit_rate(result.hit_rate_per_capacity[i]);
        }
        if (i < result.unique_keys_per_capacity.size()) {
            capacity_result->set_current_unique_keys(result.unique_keys_per_capacity[i]);
        }
    }
    response->mutable_theoretical_result()->set_max_hit_count(result.max_hit_count);
    response->mutable_theoretical_result()->set_current_unique_keys(result.theoretical_unique_keys);
    response->mutable_theoretical_result()->set_hit_rate(result.max_hit_rate);

    if (event_manager_) {
        auto event = std::make_shared<OptimizerQueryHitEvent>(request.instance_id());
        event->SetEventTriggerTime();
        event->SetAdditionalArgs(
            request.trace_id(), request.timestamp_ns(), response->input_token_len(), response->total_blocks());
        for (const auto &capacity_result : response->capacity_results()) {
            event->AddCapacityResult(capacity_result.capacity_gb(),
                                     capacity_result.cache_hit_count(),
                                     capacity_result.hit_rate(),
                                     capacity_result.current_unique_keys());
        }
        const auto &theoretical_result = response->theoretical_result();
        event->SetTheoreticalResult(theoretical_result.max_hit_count(),
                                    theoretical_result.hit_rate(),
                                    theoretical_result.current_unique_keys());
        event_manager_->Publish(event);
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

void OptimizerServiceImpl::PullQuotaAllocation(RequestContext *request_context,
                                               const proto::optimizer::PullQuotaAllocationRequest *request,
                                               proto::optimizer::PullQuotaAllocationResponse *response) {
    request_context->set_api_name("PullQuotaAllocation");
    OptimizerCallGuard guard(request_context, metrics_reporter_.get());
    if (!quota_plan_store_ || request->pool_id().empty() || request->quota_target_id().empty()) {
        SetPbResponseHeader(response->mutable_header(), EC_BADARGS);
        request_context->set_status_code(static_cast<int>(EC_BADARGS));
        SetErrorOnCollector(request_context, EC_BADARGS);
        return;
    }
    const auto plan = quota_plan_store_->Get(request->pool_id());
    SetPbResponseHeader(response->mutable_header(), EC_OK);
    request_context->set_status_code(static_cast<int>(EC_OK));
    if (!plan) {
        response->set_pull_status(proto::optimizer::QUOTA_PULL_NO_PLAN);
        return;
    }
    response->set_plan_id(plan->plan_id);
    response->set_plan_hash(plan->plan_hash);
    response->set_pool_id(plan->pool_id);
    response->set_reason(plan->reason);
    response->set_leader_epoch(plan->leader_epoch);
    response->set_allocation_epoch(plan->allocation_epoch);
    response->set_valid_until_ns(plan->valid_until_ns);
    response->set_executable(plan->executable);
    response->set_execution_phase(plan->execution_phase);
    response->set_execution_revision(plan->execution_revision);
    response->set_release_deadline_ns(plan->release_deadline_ns);
    response->set_release_consecutive_samples(plan->release_consecutive_samples);
    response->set_writes_quota(plan->writes_quota);

    const int64_t now_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch())
            .count();
    if (plan->valid_until_ns <= now_ns) {
        std::string ignored_reason;
        quota_plan_store_->RecordResizeResult(QuotaResizeResult{plan->plan_id,
                                                                plan->plan_hash,
                                                                plan->pool_id,
                                                                request->quota_target_id(),
                                                                plan->leader_epoch,
                                                                plan->allocation_epoch,
                                                                plan->execution_revision,
                                                                "PLAN_TIMEOUT",
                                                                "plan_expired",
                                                                request->current_quota_bytes(),
                                                                request->current_used_bytes()},
                                              &ignored_reason);
        response->set_pull_status(proto::optimizer::QUOTA_PULL_FROZEN);
        response->set_reason("plan_expired");
        response->set_executable(false);
        return;
    }
    if (request->last_leader_epoch() == plan->leader_epoch &&
        request->last_allocation_epoch() == plan->allocation_epoch &&
        request->last_execution_revision() == plan->execution_revision) {
        response->set_pull_status(proto::optimizer::QUOTA_PULL_NOT_MODIFIED);
        return;
    }
    if (plan->status == "FROZEN" || plan->execution_phase == "FROZEN") {
        response->set_pull_status(proto::optimizer::QUOTA_PULL_FROZEN);
        return;
    }
    const auto allocation_it =
        std::find_if(plan->allocations.begin(), plan->allocations.end(), [&](const QuotaAllocation &allocation) {
            return allocation.quota_target_id == request->quota_target_id();
        });
    if (allocation_it == plan->allocations.end()) {
        response->set_pull_status(proto::optimizer::QUOTA_PULL_FROZEN);
        response->set_reason("quota_target_not_in_plan");
        return;
    }
    response->set_pull_status(proto::optimizer::QUOTA_PULL_PLAN);
    auto *allocation = response->mutable_allocation();
    allocation->set_quota_target_id(allocation_it->quota_target_id);
    allocation->set_instance_group(allocation_it->instance_group);
    allocation->set_current_quota_bytes(allocation_it->current_quota_bytes);
    allocation->set_target_quota_bytes(allocation_it->target_quota_bytes);
    allocation->set_min_quota_bytes(allocation_it->min_quota_bytes);
    allocation->set_max_quota_bytes(allocation_it->max_quota_bytes);
    if (metrics_registry_) {
        MetricsTags pool_tags{{"pool_id", plan->pool_id}};
        REPORT_DYNAMIC_GAUGE_(metrics_registry_,
                              "quota_plan.pool_allocatable_bytes",
                              pool_tags,
                              static_cast<double>(plan->pool_allocatable_bytes));
        REPORT_DYNAMIC_GAUGE_(
            metrics_registry_, "quota_plan.expected_hit_rate_gain_pp", pool_tags, plan->expected_hit_rate_gain_pp);
        REPORT_DYNAMIC_GAUGE_(
            metrics_registry_, "quota_plan.expected_net_gain_pp", pool_tags, plan->expected_net_gain_pp);
        REPORT_DYNAMIC_GAUGE_(
            metrics_registry_, "quota_plan.gain_pp_per_tib_moved", pool_tags, plan->gain_pp_per_tib_moved);
        REPORT_DYNAMIC_GAUGE_(metrics_registry_,
                              "quota_plan.quota_transfer_bytes",
                              pool_tags,
                              static_cast<double>(plan->quota_transfer_bytes));
        REPORT_DYNAMIC_GAUGE_(metrics_registry_,
                              "quota_plan.sla_capacity_saving_bytes",
                              pool_tags,
                              static_cast<double>(plan->sla_capacity_saving_bytes));
        MetricsTags target_tags{{"pool_id", plan->pool_id}, {"quota_target_id", allocation_it->quota_target_id}};
        REPORT_DYNAMIC_GAUGE_(metrics_registry_,
                              "quota_plan.target_quota_bytes",
                              target_tags,
                              static_cast<double>(allocation_it->target_quota_bytes));
    }
    if (plan->writes_quota) {
        if (plan->execution_phase == "RECONCILE") {
            response->set_execution_phase("HOLD");
            response->set_executable(false);
            return;
        }
        const bool donor = plan->release_required_targets.count(allocation_it->quota_target_id) != 0;
        const bool receiver = allocation_it->target_quota_bytes > allocation_it->current_quota_bytes;
        if ((plan->execution_phase == "DONOR_SHRINK" && !donor) ||
            (plan->execution_phase == "RECEIVER_GROW" && !receiver)) {
            response->set_execution_phase("HOLD");
            response->set_executable(false);
        }
    }
}

void OptimizerServiceImpl::ReportQuotaResizeResult(RequestContext *request_context,
                                                   const proto::optimizer::ReportQuotaResizeResultRequest *request,
                                                   proto::optimizer::ReportQuotaResizeResultResponse *response) {
    request_context->set_api_name("ReportQuotaResizeResult");
    OptimizerCallGuard guard(request_context, metrics_reporter_.get());
    ErrorCode ec = EC_BADARGS;
    std::string store_reason;
    if (quota_plan_store_ && !request->phase().empty() && !request->status().empty() &&
        quota_plan_store_->RecordResizeResult(QuotaResizeResult{request->plan_id(),
                                                                request->plan_hash(),
                                                                request->pool_id(),
                                                                request->quota_target_id(),
                                                                request->leader_epoch(),
                                                                request->allocation_epoch(),
                                                                request->execution_revision(),
                                                                request->status(),
                                                                request->reason(),
                                                                request->observed_quota_bytes(),
                                                                request->observed_used_bytes()},
                                              &store_reason)) {
        ec = EC_OK;
        if (request->observed_quota_bytes() > 0 && manager_) {
            const auto updated_plan = quota_plan_store_->Get(request->pool_id());
            if (updated_plan) {
                const auto allocation_it =
                    std::find_if(updated_plan->allocations.begin(),
                                 updated_plan->allocations.end(),
                                 [&](const QuotaAllocation &allocation) {
                                     return allocation.quota_target_id == request->quota_target_id();
                                 });
                if (allocation_it != updated_plan->allocations.end()) {
                    const auto shadow_ec =
                        manager_->SetEnforcedShadowQuota(allocation_it->source_id, request->observed_quota_bytes());
                    if (shadow_ec != EC_OK) {
                        KVCM_LOG_WARN("quota_decision_audit event=enforced_shadow_quota_ack_failed pool_id=%s "
                                      "quota_target_id=%s source_id=%s observed_quota_bytes=%ld ec=%d",
                                      request->pool_id().c_str(),
                                      request->quota_target_id().c_str(),
                                      allocation_it->source_id.c_str(),
                                      static_cast<long>(request->observed_quota_bytes()),
                                      static_cast<int>(shadow_ec));
                    }
                }
            }
        }
        KVCM_LOG_INFO("quota_decision_audit event=resize_result_received plan_id=%s plan_hash=%s pool_id=%s "
                      "quota_target_id=%s leader_epoch=%llu allocation_epoch=%llu phase=%s status=%s reason=%s "
                      "observed_quota_bytes=%ld observed_used_bytes=%ld instance_group_version=%ld",
                      request->plan_id().c_str(),
                      request->plan_hash().c_str(),
                      request->pool_id().c_str(),
                      request->quota_target_id().c_str(),
                      static_cast<unsigned long long>(request->leader_epoch()),
                      static_cast<unsigned long long>(request->allocation_epoch()),
                      request->phase().c_str(),
                      request->status().c_str(),
                      request->reason().c_str(),
                      static_cast<long>(request->observed_quota_bytes()),
                      static_cast<long>(request->observed_used_bytes()),
                      static_cast<long>(request->instance_group_version()));
        if (metrics_registry_) {
            MetricsTags tags{{"pool_id", request->pool_id()}, {"quota_target_id", request->quota_target_id()}};
            metrics_registry_->GetCounter("quota_plan.resize_results_total", tags) += 1;
            REPORT_DYNAMIC_GAUGE_(metrics_registry_,
                                  "quota_plan.observed_quota_bytes",
                                  tags,
                                  static_cast<double>(request->observed_quota_bytes()));
            REPORT_DYNAMIC_GAUGE_(metrics_registry_,
                                  "quota_plan.observed_used_bytes",
                                  tags,
                                  static_cast<double>(request->observed_used_bytes()));
            const auto updated_plan = quota_plan_store_->Get(request->pool_id());
            if (updated_plan) {
                REPORT_DYNAMIC_GAUGE_(metrics_registry_,
                                      "quota_plan.execution_revision",
                                      tags,
                                      static_cast<double>(updated_plan->execution_revision));
                MetricsTags phase_tags = tags;
                phase_tags["phase"] = updated_plan->execution_phase;
                metrics_registry_->GetCounter("quota_plan.phase_observations_total", phase_tags) += 1;
                if (updated_plan->execution_revision != request->execution_revision()) {
                    KVCM_LOG_INFO("quota_decision_audit event=execution_phase_transition plan_id=%s plan_hash=%s "
                                  "pool_id=%s previous_phase=%s next_phase=%s execution_revision=%llu status=%s "
                                  "reason=%s",
                                  updated_plan->plan_id.c_str(),
                                  updated_plan->plan_hash.c_str(),
                                  updated_plan->pool_id.c_str(),
                                  request->phase().c_str(),
                                  updated_plan->execution_phase.c_str(),
                                  static_cast<unsigned long long>(updated_plan->execution_revision),
                                  updated_plan->status.c_str(),
                                  updated_plan->reason.c_str());
                    metrics_registry_->GetCounter("quota_plan.phase_transitions_total", phase_tags) += 1;
                }
            }
        }
    } else if (!store_reason.empty()) {
        KVCM_LOG_WARN("quota_decision_audit event=resize_result_rejected pool_id=%s quota_target_id=%s reason=%s",
                      request->pool_id().c_str(),
                      request->quota_target_id().c_str(),
                      store_reason.c_str());
    }
    SetPbResponseHeader(response->mutable_header(), ec);
    request_context->set_status_code(static_cast<int>(ec));
    SetErrorOnCollector(request_context, ec);
}

} // namespace kv_cache_manager
