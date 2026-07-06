#include "kv_cache_manager/optimizer/service/http/optimizer_service_http.h"

#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/metrics/prometheus_exporter.h"
#include "kv_cache_manager/optimizer/service/metrics/optimizer_metrics_collector.h"
#include "kv_cache_manager/optimizer/service/optimizer_service_impl.h"

namespace kv_cache_manager {

namespace {

std::shared_ptr<OptimizerServiceMetricsCollector> MakeCollector(const std::shared_ptr<MetricsRegistry> &registry) {
    if (!registry) {
        return nullptr;
    }
    auto collector = std::make_shared<OptimizerServiceMetricsCollector>(registry);
    collector->Init();
    return collector;
}

} // namespace

#define REGISTER_HTTP_HANDLER_FOR_OPTIMIZER_SERVICE(req_type, path, req_name, resp_name, method)                       \
    do {                                                                                                               \
        Register##req_type##Handler(path,                                                                              \
                                    GetHandler<OptimizerServiceHttp,                                                   \
                                               proto::optimizer::req_name##Request,                                    \
                                               proto::optimizer::resp_name##Response>(&OptimizerServiceHttp::method)); \
    } while (0)

#define REGISTER_HTTP_HANDLER_COMMON_RESP(req_type, path, req_name, method)                                            \
    do {                                                                                                               \
        Register##req_type##Handler(                                                                                   \
            path,                                                                                                      \
            GetHandler<OptimizerServiceHttp, proto::optimizer::req_name##Request, proto::optimizer::CommonResponse>(   \
                &OptimizerServiceHttp::method));                                                                       \
    } while (0)

OptimizerServiceHttp::OptimizerServiceHttp(std::shared_ptr<OptimizerServiceImpl> service_impl,
                                           std::shared_ptr<MetricsRegistry> metrics_registry)
    : service_impl_(std::move(service_impl)), metrics_registry_(std::move(metrics_registry)) {}

void OptimizerServiceHttp::Init() {}

void OptimizerServiceHttp::RegisterHandler() {
    // InstanceGroup CRUD
    REGISTER_HTTP_HANDLER_COMMON_RESP(
        Post, "/api/optimizer/createInstanceGroup", CreateInstanceGroup, CreateInstanceGroup);
    REGISTER_HTTP_HANDLER_COMMON_RESP(
        Post, "/api/optimizer/updateInstanceGroup", UpdateInstanceGroup, UpdateInstanceGroup);
    REGISTER_HTTP_HANDLER_COMMON_RESP(
        Post, "/api/optimizer/removeInstanceGroup", RemoveInstanceGroup, RemoveInstanceGroup);
    REGISTER_HTTP_HANDLER_FOR_OPTIMIZER_SERVICE(
        Post, "/api/optimizer/getInstanceGroup", GetInstanceGroup, GetInstanceGroup, GetInstanceGroup);
    REGISTER_HTTP_HANDLER_FOR_OPTIMIZER_SERVICE(
        Post, "/api/optimizer/listInstanceGroups", ListInstanceGroups, ListInstanceGroups, ListInstanceGroups);

    // Instance management
    REGISTER_HTTP_HANDLER_FOR_OPTIMIZER_SERVICE(Post,
                                                "/api/optimizer/registerInstance",
                                                OptimizerRegisterInstance,
                                                OptimizerRegisterInstance,
                                                RegisterInstance);
    REGISTER_HTTP_HANDLER_FOR_OPTIMIZER_SERVICE(
        Post, "/api/optimizer/removeInstance", OptimizerRemoveInstance, OptimizerRemoveInstance, RemoveInstance);
    REGISTER_HTTP_HANDLER_FOR_OPTIMIZER_SERVICE(
        Post, "/api/optimizer/getInstance", OptimizerGetInstance, OptimizerGetInstance, GetInstance);

    // TraceQuery
    REGISTER_HTTP_HANDLER_FOR_OPTIMIZER_SERVICE(Post, "/api/optimizer/traceQuery", TraceQuery, TraceQuery, TraceQuery);
    REGISTER_HTTP_HANDLER_FOR_OPTIMIZER_SERVICE(
        Post, "/api/optimizer/listInstances", OptimizerListInstances, OptimizerListInstances, ListInstances);
    REGISTER_HTTP_HANDLER_FOR_OPTIMIZER_SERVICE(
        Post, "/api/optimizer/resetStats", OptimizerResetStats, OptimizerResetStats, ResetStats);
}

// InstanceGroup CRUD

void OptimizerServiceHttp::CreateInstanceGroup(coro_http::coro_http_connection *http_conn,
                                               proto::optimizer::CreateInstanceGroupRequest *request,
                                               proto::optimizer::CommonResponse *response) {
    RequestContext request_context(request->trace_id(), MakeCollector(metrics_registry_));
    request_context.set_client_ip(CoroHttpService::GetHttpClientIp(http_conn));
    service_impl_->CreateInstanceGroup(&request_context, request, response);
}

void OptimizerServiceHttp::UpdateInstanceGroup(coro_http::coro_http_connection *http_conn,
                                               proto::optimizer::UpdateInstanceGroupRequest *request,
                                               proto::optimizer::CommonResponse *response) {
    RequestContext request_context(request->trace_id(), MakeCollector(metrics_registry_));
    request_context.set_client_ip(CoroHttpService::GetHttpClientIp(http_conn));
    service_impl_->UpdateInstanceGroup(&request_context, request, response);
}

void OptimizerServiceHttp::RemoveInstanceGroup(coro_http::coro_http_connection *http_conn,
                                               proto::optimizer::RemoveInstanceGroupRequest *request,
                                               proto::optimizer::CommonResponse *response) {
    RequestContext request_context(request->trace_id(), MakeCollector(metrics_registry_));
    request_context.set_client_ip(CoroHttpService::GetHttpClientIp(http_conn));
    service_impl_->RemoveInstanceGroup(&request_context, request, response);
}

void OptimizerServiceHttp::GetInstanceGroup(coro_http::coro_http_connection *http_conn,
                                            proto::optimizer::GetInstanceGroupRequest *request,
                                            proto::optimizer::GetInstanceGroupResponse *response) {
    RequestContext request_context(request->trace_id(), MakeCollector(metrics_registry_));
    request_context.set_client_ip(CoroHttpService::GetHttpClientIp(http_conn));
    service_impl_->GetInstanceGroup(&request_context, request, response);
}

void OptimizerServiceHttp::ListInstanceGroups(coro_http::coro_http_connection *http_conn,
                                              proto::optimizer::ListInstanceGroupsRequest *request,
                                              proto::optimizer::ListInstanceGroupsResponse *response) {
    RequestContext request_context(request->trace_id(), MakeCollector(metrics_registry_));
    request_context.set_client_ip(CoroHttpService::GetHttpClientIp(http_conn));
    service_impl_->ListInstanceGroups(&request_context, request, response);
}

// Instance management

void OptimizerServiceHttp::RegisterInstance(coro_http::coro_http_connection *http_conn,
                                            proto::optimizer::OptimizerRegisterInstanceRequest *request,
                                            proto::optimizer::OptimizerRegisterInstanceResponse *response) {
    RequestContext request_context(request->trace_id(), MakeCollector(metrics_registry_));
    request_context.set_client_ip(CoroHttpService::GetHttpClientIp(http_conn));
    service_impl_->RegisterInstance(&request_context, request, response);
}

void OptimizerServiceHttp::RemoveInstance(coro_http::coro_http_connection *http_conn,
                                          proto::optimizer::OptimizerRemoveInstanceRequest *request,
                                          proto::optimizer::OptimizerRemoveInstanceResponse *response) {
    RequestContext request_context(request->trace_id(), MakeCollector(metrics_registry_));
    request_context.set_client_ip(CoroHttpService::GetHttpClientIp(http_conn));
    service_impl_->RemoveInstance(&request_context, request, response);
}

void OptimizerServiceHttp::GetInstance(coro_http::coro_http_connection *http_conn,
                                       proto::optimizer::OptimizerGetInstanceRequest *request,
                                       proto::optimizer::OptimizerGetInstanceResponse *response) {
    RequestContext request_context(request->trace_id(), MakeCollector(metrics_registry_));
    request_context.set_client_ip(CoroHttpService::GetHttpClientIp(http_conn));
    service_impl_->GetInstance(&request_context, request, response);
}

// TraceQuery

void OptimizerServiceHttp::TraceQuery(coro_http::coro_http_connection *http_conn,
                                      proto::optimizer::TraceQueryRequest *request,
                                      proto::optimizer::TraceQueryResponse *response) {
    RequestContext request_context(request->trace_id(), MakeCollector(metrics_registry_));
    request_context.set_client_ip(CoroHttpService::GetHttpClientIp(http_conn));
    service_impl_->TraceQuery(&request_context, request, response);
}

void OptimizerServiceHttp::ListInstances(coro_http::coro_http_connection *http_conn,
                                         proto::optimizer::OptimizerListInstancesRequest *request,
                                         proto::optimizer::OptimizerListInstancesResponse *response) {
    RequestContext request_context(request->trace_id(), MakeCollector(metrics_registry_));
    request_context.set_client_ip(CoroHttpService::GetHttpClientIp(http_conn));
    service_impl_->ListInstances(&request_context, request, response);
}

void OptimizerServiceHttp::ResetStats(coro_http::coro_http_connection *http_conn,
                                      proto::optimizer::OptimizerResetStatsRequest *request,
                                      proto::optimizer::OptimizerResetStatsResponse *response) {
    RequestContext request_context(request->trace_id(), MakeCollector(metrics_registry_));
    request_context.set_client_ip(CoroHttpService::GetHttpClientIp(http_conn));
    service_impl_->ResetStats(&request_context, request, response);
}

void OptimizerServiceHttp::RegisterPrometheusEndpoint(std::shared_ptr<MetricsRegistry> registry,
                                                      const std::string &prefix) {
    auto pfx = prefix;
    RegisterGetHandler("/metrics",
                       [registry, pfx](coro_http::coro_http_request &,
                                       coro_http::coro_http_response &res) -> async_simple::coro::Lazy<void> {
                           std::string body = PrometheusExporter::Expose(*registry, pfx);
                           res.add_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8");
                           res.set_status_and_content(coro_http::status_type::ok, std::move(body));
                           co_return;
                       });
}

} // namespace kv_cache_manager
