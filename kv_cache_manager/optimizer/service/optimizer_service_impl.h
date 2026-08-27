#pragma once

#include <memory>
#include <string>
#include <unordered_set>

#include "kv_cache_manager/common/error_code.h"
#include "kv_cache_manager/protocol/protobuf/optimizer_service.pb.h"

namespace kv_cache_manager {

class OnlineOptimizerManager;
class OptimizerMetricsReporter;
class OptimizerRegistryManager;
class InMemoryQuotaPlanStore;
class MetricsRegistry;
class RequestContext;

class OptimizerServiceImpl {
public:
    OptimizerServiceImpl(std::shared_ptr<OnlineOptimizerManager> manager,
                         std::shared_ptr<OptimizerMetricsReporter> metrics_reporter,
                         std::shared_ptr<InMemoryQuotaPlanStore> quota_plan_store = nullptr,
                         std::shared_ptr<MetricsRegistry> metrics_registry = nullptr);
    ~OptimizerServiceImpl() = default;

    OptimizerServiceImpl(const OptimizerServiceImpl &) = delete;
    OptimizerServiceImpl &operator=(const OptimizerServiceImpl &) = delete;

    // InstanceGroup CRUD
    void CreateInstanceGroup(RequestContext *request_context,
                             const proto::optimizer::CreateInstanceGroupRequest *request,
                             proto::optimizer::CommonResponse *response);

    void UpdateInstanceGroup(RequestContext *request_context,
                             const proto::optimizer::UpdateInstanceGroupRequest *request,
                             proto::optimizer::CommonResponse *response);

    void RemoveInstanceGroup(RequestContext *request_context,
                             const proto::optimizer::RemoveInstanceGroupRequest *request,
                             proto::optimizer::CommonResponse *response);

    void GetInstanceGroup(RequestContext *request_context,
                          const proto::optimizer::GetInstanceGroupRequest *request,
                          proto::optimizer::GetInstanceGroupResponse *response);

    void ListInstanceGroups(RequestContext *request_context,
                            const proto::optimizer::ListInstanceGroupsRequest *request,
                            proto::optimizer::ListInstanceGroupsResponse *response);

    // Instance management
    void RegisterInstance(RequestContext *request_context,
                          const proto::optimizer::OptimizerRegisterInstanceRequest *request,
                          proto::optimizer::OptimizerRegisterInstanceResponse *response);

    void RemoveInstance(RequestContext *request_context,
                        const proto::optimizer::OptimizerRemoveInstanceRequest *request,
                        proto::optimizer::OptimizerRemoveInstanceResponse *response);

    void GetInstance(RequestContext *request_context,
                     const proto::optimizer::OptimizerGetInstanceRequest *request,
                     proto::optimizer::OptimizerGetInstanceResponse *response);

    // KVCM ingress
    ErrorCode ApplyKvcmConfiguration(const proto::optimizer::KvcmConfigurationResponse &configuration,
                                     std::unordered_set<std::string> &unsupported_instance_ids);

    // TraceQuery
    ErrorCode ExecuteTraceQuery(const proto::optimizer::TraceQueryRequest &request,
                                proto::optimizer::TraceQueryResponse *response);

    void TraceQuery(RequestContext *request_context,
                    const proto::optimizer::TraceQueryRequest *request,
                    proto::optimizer::TraceQueryResponse *response);

    void ListInstances(RequestContext *request_context,
                       const proto::optimizer::OptimizerListInstancesRequest *request,
                       proto::optimizer::OptimizerListInstancesResponse *response);

    void ResetStats(RequestContext *request_context,
                    const proto::optimizer::OptimizerResetStatsRequest *request,
                    proto::optimizer::OptimizerResetStatsResponse *response);

    void PullQuotaAllocation(RequestContext *request_context,
                             const proto::optimizer::PullQuotaAllocationRequest *request,
                             proto::optimizer::PullQuotaAllocationResponse *response);

    void ReportQuotaResizeResult(RequestContext *request_context,
                                 const proto::optimizer::ReportQuotaResizeResultRequest *request,
                                 proto::optimizer::ReportQuotaResizeResultResponse *response);

private:
    std::shared_ptr<OnlineOptimizerManager> manager_;
    std::shared_ptr<OptimizerMetricsReporter> metrics_reporter_;
    std::shared_ptr<InMemoryQuotaPlanStore> quota_plan_store_;
    std::shared_ptr<MetricsRegistry> metrics_registry_;
};

} // namespace kv_cache_manager
