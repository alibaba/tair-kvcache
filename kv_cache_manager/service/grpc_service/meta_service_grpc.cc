#include "kv_cache_manager/service/grpc_service/meta_service_grpc.h"

#include <memory>
#include <string>
#include <utility>

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/metrics/metrics_registry.h"
#include "kv_cache_manager/protocol/protobuf/meta_service.grpc.pb.h"
#include "kv_cache_manager/service/meta_service_impl.h"
#include "kv_cache_manager/service/util/common.h"

namespace kv_cache_manager {

MetaServiceGRpc::MetaServiceGRpc(std::shared_ptr<MetricsRegistry> metrics_registry,
                                 std::shared_ptr<MetaServiceImpl> meta_service_impl,
                                 std::shared_ptr<RegistryManager> registry_manager)
    : MetaServiceMetricsBase(std::move(metrics_registry), registry_manager)
    , meta_service_impl_(std::move(meta_service_impl)) {}

void MetaServiceGRpc::Init() { MetaServiceMetricsBase::InitMetrics(); }

grpc::Status MetaServiceGRpc::RegisterInstance(grpc::ServerContext *context,
                                               const proto::meta::RegisterInstanceRequest *request,
                                               proto::meta::RegisterInstanceResponse *response) {
    API_CONTEXT_INIT_GRPC(RegisterInstance);
    meta_service_impl_->RegisterInstance(request_context, request, response);
    return grpc::Status::OK;
}

grpc::Status MetaServiceGRpc::GetInstanceInfo(grpc::ServerContext *context,
                                              const proto::meta::GetInstanceInfoRequest *request,
                                              proto::meta::GetInstanceInfoResponse *response) {
    API_CONTEXT_INIT_GRPC(GetInstanceInfo);
    meta_service_impl_->GetInstanceInfo(request_context, request, response);
    return grpc::Status::OK;
}

grpc::Status MetaServiceGRpc::GetCacheMeta(grpc::ServerContext *context,
                                           const proto::meta::GetCacheMetaRequest *request,
                                           proto::meta::GetCacheMetaResponse *response) {
    API_CONTEXT_GET_COLLECTOR_AND_INIT_GRPC(GetCacheMeta, grpc::Status::OK);
    meta_service_impl_->GetCacheMeta(request_context, request, response);
    return grpc::Status::OK;
}

grpc::Status MetaServiceGRpc::GetCacheLocation(grpc::ServerContext *context,
                                               const proto::meta::GetCacheLocationRequest *request,
                                               proto::meta::GetCacheLocationResponse *response) {
    API_CONTEXT_GET_COLLECTOR_AND_INIT_GRPC(GetCacheLocation, grpc::Status::OK);
    meta_service_impl_->GetCacheLocation(request_context, request, response);
    return grpc::Status::OK;
}

grpc::Status MetaServiceGRpc::GetCacheLocationsByBackend(grpc::ServerContext *context,
                                                         const proto::meta::GetCacheLocationsByBackendRequest *request,
                                                         proto::meta::GetCacheLocationsByBackendResponse *response) {
    API_CONTEXT_GET_COLLECTOR_AND_INIT_GRPC(GetCacheLocationsByBackend, grpc::Status::OK);
    meta_service_impl_->GetCacheLocationsByBackend(request_context, request, response);
    return grpc::Status::OK;
}

grpc::Status MetaServiceGRpc::GetCacheLocationLen(grpc::ServerContext *context,
                                                  const proto::meta::GetCacheLocationLenRequest *request,
                                                  proto::meta::GetCacheLocationLenResponse *response) {
    API_CONTEXT_GET_COLLECTOR_AND_INIT_GRPC(GetCacheLocationLen, grpc::Status::OK);
    meta_service_impl_->GetCacheLocationLen(request_context, request, response);
    return grpc::Status::OK;
}

grpc::Status MetaServiceGRpc::StartWriteCache(grpc::ServerContext *context,
                                              const proto::meta::StartWriteCacheRequest *request,
                                              proto::meta::StartWriteCacheResponse *response) {
    API_CONTEXT_GET_COLLECTOR_AND_INIT_GRPC(StartWriteCache, grpc::Status::OK);
    meta_service_impl_->StartWriteCache(request_context, request, response);
    return grpc::Status::OK;
}
grpc::Status MetaServiceGRpc::FinishWriteCache(grpc::ServerContext *context,
                                               const proto::meta::FinishWriteCacheRequest *request,
                                               proto::meta::CommonResponse *response) {
    API_CONTEXT_GET_COLLECTOR_AND_INIT_GRPC(FinishWriteCache, grpc::Status::OK);
    meta_service_impl_->FinishWriteCache(request_context, request, response);
    return grpc::Status::OK;
}
grpc::Status MetaServiceGRpc::RemoveCache(grpc::ServerContext *context,
                                          const proto::meta::RemoveCacheRequest *request,
                                          proto::meta::CommonResponse *response) {
    API_CONTEXT_GET_COLLECTOR_AND_INIT_GRPC(RemoveCache, grpc::Status::OK);
    meta_service_impl_->RemoveCache(request_context, request, response);
    return grpc::Status::OK;
}
grpc::Status MetaServiceGRpc::TrimCache(grpc::ServerContext *context,
                                        const proto::meta::TrimCacheRequest *request,
                                        proto::meta::CommonResponse *response) {
    API_CONTEXT_GET_COLLECTOR_AND_INIT_GRPC(TrimCache, grpc::Status::OK);
    meta_service_impl_->TrimCache(request_context, request, response);
    return grpc::Status::OK;
}

grpc::Status MetaServiceGRpc::GetClusterInfo(grpc::ServerContext *context,
                                             const proto::meta::GetClusterInfoRequest *request,
                                             proto::meta::GetClusterInfoResponse *response) {
    // instance_id 可能尚未注册（如 RegisterInstance 之前），此时 fallback 到全局 collector
    auto metrics_collector = get_metrics_collector_from_map_for_GetClusterInfo(request->instance_id());
    if (metrics_collector == nullptr) {
        metrics_collector = KVCM_METRICS_COLLECTOR_(GetClusterInfo);
    }
    API_CONTEXT_INIT(metrics_collector, ExtractIpFromPeer, context->peer())
    meta_service_impl_->GetClusterInfo(request_context, request, response);
    return grpc::Status::OK;
}

grpc::Status MetaServiceGRpc::ReportEvent(grpc::ServerContext *context,
                                          const proto::meta::ReportEventRequest *request,
                                          proto::meta::ReportEventResponse *response) {
    API_CONTEXT_GET_COLLECTOR_AND_INIT_GRPC(ReportEvent, grpc::Status::OK);
    bool has_block_add = false, has_block_delete = false;
    for (int i = 0; i < request->events_size(); ++i) {
        if (request->events(i).event_type() == proto::meta::EVENT_BLOCK_ADD)
            has_block_add = true;
        if (request->events(i).event_type() == proto::meta::EVENT_BLOCK_DELETE)
            has_block_delete = true;
    }
    if (has_block_add && !request->instance_id().empty()) {
        auto mc = get_metrics_collector_from_map_for_EventBlockAdd(request->instance_id());
        if (mc)
            request_context->GetMetricsCollectorsVehicle().AddMetricsCollector(mc);
    }
    if (has_block_delete && !request->instance_id().empty()) {
        auto mc = get_metrics_collector_from_map_for_EventBlockDelete(request->instance_id());
        if (mc)
            request_context->GetMetricsCollectorsVehicle().AddMetricsCollector(mc);
    }
    meta_service_impl_->ReportEvent(request_context, request, response);
    return grpc::Status::OK;
}

} // namespace kv_cache_manager
