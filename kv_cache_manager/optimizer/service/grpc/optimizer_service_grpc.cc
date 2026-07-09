#include "kv_cache_manager/optimizer/service/grpc/optimizer_service_grpc.h"

#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/optimizer/service/metrics/optimizer_metrics_collector.h"
#include "kv_cache_manager/optimizer/service/optimizer_service_impl.h"
#include "kv_cache_manager/service/util/common.h"

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

OptimizerServiceGRpc::OptimizerServiceGRpc(std::shared_ptr<OptimizerServiceImpl> service_impl,
                                           std::shared_ptr<MetricsRegistry> metrics_registry)
    : service_impl_(std::move(service_impl)), metrics_registry_(std::move(metrics_registry)) {}

// InstanceGroup CRUD

grpc::Status OptimizerServiceGRpc::CreateInstanceGroup(grpc::ServerContext *context,
                                                       const proto::optimizer::CreateInstanceGroupRequest *request,
                                                       proto::optimizer::CommonResponse *response) {
    RequestContext request_context(request->trace_id(), MakeCollector(metrics_registry_));
    request_context.set_client_ip(ExtractIpFromPeer(context->peer()));
    service_impl_->CreateInstanceGroup(&request_context, request, response);
    return grpc::Status::OK;
}

grpc::Status OptimizerServiceGRpc::UpdateInstanceGroup(grpc::ServerContext *context,
                                                       const proto::optimizer::UpdateInstanceGroupRequest *request,
                                                       proto::optimizer::CommonResponse *response) {
    RequestContext request_context(request->trace_id(), MakeCollector(metrics_registry_));
    request_context.set_client_ip(ExtractIpFromPeer(context->peer()));
    service_impl_->UpdateInstanceGroup(&request_context, request, response);
    return grpc::Status::OK;
}

grpc::Status OptimizerServiceGRpc::RemoveInstanceGroup(grpc::ServerContext *context,
                                                       const proto::optimizer::RemoveInstanceGroupRequest *request,
                                                       proto::optimizer::CommonResponse *response) {
    RequestContext request_context(request->trace_id(), MakeCollector(metrics_registry_));
    request_context.set_client_ip(ExtractIpFromPeer(context->peer()));
    service_impl_->RemoveInstanceGroup(&request_context, request, response);
    return grpc::Status::OK;
}

grpc::Status OptimizerServiceGRpc::GetInstanceGroup(grpc::ServerContext *context,
                                                    const proto::optimizer::GetInstanceGroupRequest *request,
                                                    proto::optimizer::GetInstanceGroupResponse *response) {
    RequestContext request_context(request->trace_id(), MakeCollector(metrics_registry_));
    request_context.set_client_ip(ExtractIpFromPeer(context->peer()));
    service_impl_->GetInstanceGroup(&request_context, request, response);
    return grpc::Status::OK;
}

grpc::Status OptimizerServiceGRpc::ListInstanceGroups(grpc::ServerContext *context,
                                                      const proto::optimizer::ListInstanceGroupsRequest *request,
                                                      proto::optimizer::ListInstanceGroupsResponse *response) {
    RequestContext request_context(request->trace_id(), MakeCollector(metrics_registry_));
    request_context.set_client_ip(ExtractIpFromPeer(context->peer()));
    service_impl_->ListInstanceGroups(&request_context, request, response);
    return grpc::Status::OK;
}

// Instance management

grpc::Status OptimizerServiceGRpc::RegisterInstance(grpc::ServerContext *context,
                                                    const proto::optimizer::OptimizerRegisterInstanceRequest *request,
                                                    proto::optimizer::OptimizerRegisterInstanceResponse *response) {
    RequestContext request_context(request->trace_id(), MakeCollector(metrics_registry_));
    request_context.set_client_ip(ExtractIpFromPeer(context->peer()));
    service_impl_->RegisterInstance(&request_context, request, response);
    return grpc::Status::OK;
}

grpc::Status OptimizerServiceGRpc::RemoveInstance(grpc::ServerContext *context,
                                                  const proto::optimizer::OptimizerRemoveInstanceRequest *request,
                                                  proto::optimizer::OptimizerRemoveInstanceResponse *response) {
    RequestContext request_context(request->trace_id(), MakeCollector(metrics_registry_));
    request_context.set_client_ip(ExtractIpFromPeer(context->peer()));
    service_impl_->RemoveInstance(&request_context, request, response);
    return grpc::Status::OK;
}

grpc::Status OptimizerServiceGRpc::GetInstance(grpc::ServerContext *context,
                                               const proto::optimizer::OptimizerGetInstanceRequest *request,
                                               proto::optimizer::OptimizerGetInstanceResponse *response) {
    RequestContext request_context(request->trace_id(), MakeCollector(metrics_registry_));
    request_context.set_client_ip(ExtractIpFromPeer(context->peer()));
    service_impl_->GetInstance(&request_context, request, response);
    return grpc::Status::OK;
}

// TraceQuery

grpc::Status OptimizerServiceGRpc::TraceQuery(grpc::ServerContext *context,
                                              const proto::optimizer::TraceQueryRequest *request,
                                              proto::optimizer::TraceQueryResponse *response) {
    RequestContext request_context(request->trace_id(), MakeCollector(metrics_registry_));
    request_context.set_client_ip(ExtractIpFromPeer(context->peer()));
    service_impl_->TraceQuery(&request_context, request, response);
    return grpc::Status::OK;
}

grpc::Status OptimizerServiceGRpc::ListInstances(grpc::ServerContext *context,
                                                 const proto::optimizer::OptimizerListInstancesRequest *request,
                                                 proto::optimizer::OptimizerListInstancesResponse *response) {
    RequestContext request_context(request->trace_id(), MakeCollector(metrics_registry_));
    request_context.set_client_ip(ExtractIpFromPeer(context->peer()));
    service_impl_->ListInstances(&request_context, request, response);
    return grpc::Status::OK;
}

grpc::Status OptimizerServiceGRpc::ResetStats(grpc::ServerContext *context,
                                              const proto::optimizer::OptimizerResetStatsRequest *request,
                                              proto::optimizer::OptimizerResetStatsResponse *response) {
    RequestContext request_context(request->trace_id(), MakeCollector(metrics_registry_));
    request_context.set_client_ip(ExtractIpFromPeer(context->peer()));
    service_impl_->ResetStats(&request_context, request, response);
    return grpc::Status::OK;
}

} // namespace kv_cache_manager
