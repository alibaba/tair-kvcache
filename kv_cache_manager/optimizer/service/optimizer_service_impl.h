#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "kv_cache_manager/common/error_code.h"
#include "kv_cache_manager/protocol/protobuf/optimizer_service.pb.h"

namespace kv_cache_manager {

class OnlineOptimizerManager;
class EventManager;
class OptimizerMetricsReporter;
class OptimizerRegistryManager;
class RequestContext;

struct KvcmConfigurationApplyOptions {
    std::vector<double> capacity_gb;
    bool fanout_all_instances = false;
    std::vector<int32_t> linear_steps;
    std::string full_location_spec_group_name;
    std::string linear_location_spec_group_name;
};

class OptimizerServiceImpl {
public:
    OptimizerServiceImpl(std::shared_ptr<OnlineOptimizerManager> manager,
                         std::shared_ptr<OptimizerMetricsReporter> metrics_reporter,
                         std::shared_ptr<EventManager> event_manager = nullptr);
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
                                     std::unordered_set<std::string> &unsupported_instance_ids,
                                     const KvcmConfigurationApplyOptions &options = {});

    // Resolves the active Optimizer instances in the source instance's group.
    // All targets must use the source block size because online events already
    // carry block keys at that granularity and cannot be re-blocked losslessly.
    ErrorCode ListFanoutInstanceIds(const std::string &source_instance_id,
                                    std::vector<std::string> &instance_ids) const;

    // TraceQuery
    ErrorCode ExecuteTraceQuery(const proto::optimizer::TraceQueryRequest &request,
                                proto::optimizer::TraceQueryResponse *response);
    ErrorCode ExecuteTraceQueryForInstance(const proto::optimizer::TraceQueryRequest &request,
                                           const std::string &target_instance_id,
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

private:
    std::shared_ptr<OnlineOptimizerManager> manager_;
    std::shared_ptr<OptimizerMetricsReporter> metrics_reporter_;
    std::shared_ptr<EventManager> event_manager_;
};

} // namespace kv_cache_manager
