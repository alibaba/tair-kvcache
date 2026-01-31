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

} // namespace kv_cache_manager
