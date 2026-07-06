#pragma once

#include <memory>

#include "kv_cache_manager/metrics/metrics_registry.h"
#include "kv_cache_manager/protocol/protobuf/optimizer_service.pb.h"
#include "kv_cache_manager/service/http_service/coro_http_service.h"

namespace kv_cache_manager {

class OptimizerServiceImpl;

class OptimizerServiceHttp : public CoroHttpService {
public:
    OptimizerServiceHttp(std::shared_ptr<OptimizerServiceImpl> service_impl,
                         std::shared_ptr<MetricsRegistry> metrics_registry);

    void Init() override;
    void RegisterHandler() override;

    void RegisterPrometheusEndpoint(std::shared_ptr<MetricsRegistry> registry, const std::string &prefix);

    // InstanceGroup CRUD
    void CreateInstanceGroup(coro_http::coro_http_connection *http_conn,
                             proto::optimizer::CreateInstanceGroupRequest *request,
                             proto::optimizer::CommonResponse *response);

    void UpdateInstanceGroup(coro_http::coro_http_connection *http_conn,
                             proto::optimizer::UpdateInstanceGroupRequest *request,
                             proto::optimizer::CommonResponse *response);

    void RemoveInstanceGroup(coro_http::coro_http_connection *http_conn,
                             proto::optimizer::RemoveInstanceGroupRequest *request,
                             proto::optimizer::CommonResponse *response);

    void GetInstanceGroup(coro_http::coro_http_connection *http_conn,
                          proto::optimizer::GetInstanceGroupRequest *request,
                          proto::optimizer::GetInstanceGroupResponse *response);

    void ListInstanceGroups(coro_http::coro_http_connection *http_conn,
                            proto::optimizer::ListInstanceGroupsRequest *request,
                            proto::optimizer::ListInstanceGroupsResponse *response);

    // Instance management
    void RegisterInstance(coro_http::coro_http_connection *http_conn,
                          proto::optimizer::OptimizerRegisterInstanceRequest *request,
                          proto::optimizer::OptimizerRegisterInstanceResponse *response);

    void RemoveInstance(coro_http::coro_http_connection *http_conn,
                        proto::optimizer::OptimizerRemoveInstanceRequest *request,
                        proto::optimizer::OptimizerRemoveInstanceResponse *response);

    void GetInstance(coro_http::coro_http_connection *http_conn,
                     proto::optimizer::OptimizerGetInstanceRequest *request,
                     proto::optimizer::OptimizerGetInstanceResponse *response);

    // TraceQuery
    void TraceQuery(coro_http::coro_http_connection *http_conn,
                    proto::optimizer::TraceQueryRequest *request,
                    proto::optimizer::TraceQueryResponse *response);

    void ListInstances(coro_http::coro_http_connection *http_conn,
                       proto::optimizer::OptimizerListInstancesRequest *request,
                       proto::optimizer::OptimizerListInstancesResponse *response);

    void ResetStats(coro_http::coro_http_connection *http_conn,
                    proto::optimizer::OptimizerResetStatsRequest *request,
                    proto::optimizer::OptimizerResetStatsResponse *response);

private:
    std::shared_ptr<OptimizerServiceImpl> service_impl_;
    std::shared_ptr<MetricsRegistry> metrics_registry_;
};

} // namespace kv_cache_manager
