#include "kv_cache_manager/service/grpc_service/kv_meta_service_grpc.h"

#include <memory>
#include <utility>

#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/metrics/metrics_registry.h"
#include "kv_cache_manager/service/kv_meta_service_impl.h"
#include "kv_cache_manager/service/util/common.h"

namespace kv_cache_manager {

#define MAKE_KV_META_SERVICE_METRICS_COLLECTOR(method)                                                               \
    KVCM_MAKE_METRICS_COLLECTOR_(                                                                                     \
        metrics_registry_, KvMeta##method, Service, (MetricsTags{{"api_name", "KvMeta." #method}}))

KvMetaServiceGRpc::KvMetaServiceGRpc(std::shared_ptr<MetricsRegistry> metrics_registry,
                                     std::shared_ptr<KvMetaServiceImpl> service_impl)
    : metrics_registry_(std::move(metrics_registry)), service_impl_(std::move(service_impl)) {}

void KvMetaServiceGRpc::Init() {
    MAKE_KV_META_SERVICE_METRICS_COLLECTOR(RegisterInstance);
    MAKE_KV_META_SERVICE_METRICS_COLLECTOR(GetInstanceInfo);
    MAKE_KV_META_SERVICE_METRICS_COLLECTOR(Get);
    MAKE_KV_META_SERVICE_METRICS_COLLECTOR(PutStart);
    MAKE_KV_META_SERVICE_METRICS_COLLECTOR(PutFinish);
    MAKE_KV_META_SERVICE_METRICS_COLLECTOR(Remove);
    MAKE_KV_META_SERVICE_METRICS_COLLECTOR(Trim);
}

grpc::Status KvMetaServiceGRpc::RegisterInstance(grpc::ServerContext *context,
                                                 const proto::kv_meta::RegisterInstanceRequest *request,
                                                 proto::kv_meta::RegisterInstanceResponse *response) {
    API_CONTEXT_INIT_GRPC(KvMetaRegisterInstance);
    service_impl_->RegisterInstance(request_context, request, response);
    return grpc::Status::OK;
}

grpc::Status KvMetaServiceGRpc::GetInstanceInfo(grpc::ServerContext *context,
                                                const proto::kv_meta::GetInstanceInfoRequest *request,
                                                proto::kv_meta::GetInstanceInfoResponse *response) {
    API_CONTEXT_INIT_GRPC(KvMetaGetInstanceInfo);
    service_impl_->GetInstanceInfo(request_context, request, response);
    return grpc::Status::OK;
}

grpc::Status KvMetaServiceGRpc::Get(grpc::ServerContext *context,
                                    const proto::kv_meta::GetRequest *request,
                                    proto::kv_meta::GetResponse *response) {
    API_CONTEXT_INIT_GRPC(KvMetaGet);
    service_impl_->Get(request_context, request, response);
    return grpc::Status::OK;
}

grpc::Status KvMetaServiceGRpc::PutStart(grpc::ServerContext *context,
                                         const proto::kv_meta::PutStartRequest *request,
                                         proto::kv_meta::PutStartResponse *response) {
    API_CONTEXT_INIT_GRPC(KvMetaPutStart);
    service_impl_->PutStart(request_context, request, response);
    return grpc::Status::OK;
}

grpc::Status KvMetaServiceGRpc::PutFinish(grpc::ServerContext *context,
                                          const proto::kv_meta::PutFinishRequest *request,
                                          proto::kv_meta::CommonResponse *response) {
    API_CONTEXT_INIT_GRPC(KvMetaPutFinish);
    service_impl_->PutFinish(request_context, request, response);
    return grpc::Status::OK;
}

grpc::Status KvMetaServiceGRpc::Remove(grpc::ServerContext *context,
                                       const proto::kv_meta::RemoveRequest *request,
                                       proto::kv_meta::CommonResponse *response) {
    API_CONTEXT_INIT_GRPC(KvMetaRemove);
    service_impl_->Remove(request_context, request, response);
    return grpc::Status::OK;
}

grpc::Status KvMetaServiceGRpc::Trim(grpc::ServerContext *context,
                                     const proto::kv_meta::TrimRequest *request,
                                     proto::kv_meta::CommonResponse *response) {
    API_CONTEXT_INIT_GRPC(KvMetaTrim);
    service_impl_->Trim(request_context, request, response);
    return grpc::Status::OK;
}

#undef MAKE_KV_META_SERVICE_METRICS_COLLECTOR

} // namespace kv_cache_manager
