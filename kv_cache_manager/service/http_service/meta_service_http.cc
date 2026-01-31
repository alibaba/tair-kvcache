#include "kv_cache_manager/service/http_service/meta_service_http.h"

#include <memory>
#include <string>
#include <utility>

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/metrics/metrics_registry.h"
#include "kv_cache_manager/protocol/protobuf/meta_service.pb.h"
#include "kv_cache_manager/service/meta_service_impl.h"
#include "kv_cache_manager/service/util/common.h"

namespace kv_cache_manager {

#define __NOTHING__

MetaServiceHttp::MetaServiceHttp(std::shared_ptr<MetricsRegistry> metrics_registry,
                                 std::shared_ptr<MetaServiceImpl> meta_service_impl,
                                 std::shared_ptr<RegistryManager> registry_manager)
    : MetaServiceMetricsBase(std::move(metrics_registry), registry_manager)
    , meta_service_impl_(std::move(meta_service_impl)) {}

void MetaServiceHttp::Init() { MetaServiceMetricsBase::InitMetrics(); }

void MetaServiceHttp::RegisterHandler() {
    REGISTER_HTTP_HANDLER_FOR_META_SERVICE(
        Post, registerInstance, RegisterInstance, RegisterInstance, RegisterInstance);
    REGISTER_HTTP_HANDLER_FOR_META_SERVICE(Post, getInstanceInfo, GetInstanceInfo, GetInstanceInfo, GetInstanceInfo);
    REGISTER_HTTP_HANDLER_FOR_META_SERVICE(Post, getCacheMeta, GetCacheMeta, GetCacheMeta, GetCacheMeta);
    REGISTER_HTTP_HANDLER_FOR_META_SERVICE(
        Post, getCacheLocation, GetCacheLocation, GetCacheLocation, GetCacheLocation);
    REGISTER_HTTP_HANDLER_FOR_META_SERVICE(Post, startWriteCache, StartWriteCache, StartWriteCache, StartWriteCache);
    REGISTER_HTTP_HANDLER_FOR_META_SERVICE(Post, finishWriteCache, FinishWriteCache, Common, FinishWriteCache);
    REGISTER_HTTP_HANDLER_FOR_META_SERVICE(Post, removeCache, RemoveCache, Common, RemoveCache);
    REGISTER_HTTP_HANDLER_FOR_META_SERVICE(Post, trimCache, TrimCache, Common, TrimCache);
}

void MetaServiceHttp::RegisterInstance(coro_http::coro_http_connection *http_conn,
                                       proto::meta::RegisterInstanceRequest *request,
                                       proto::meta::RegisterInstanceResponse *response) {
    API_CONTEXT_INIT_HTTP(RegisterInstance);
    KVCM_LOG_INFO("[traceId: %s] RegisterInstance called with instance id: %s, instance group: %s",
                  request->trace_id().c_str(),
                  request->instance_id().c_str(),
                  request->instance_group().c_str());
    KVCM_LOG_DEBUG("[traceId: %s] RegisterInstance request details: %s",
                   request->trace_id().c_str(),
                   request->model_deployment().ShortDebugString().c_str());
    meta_service_impl_->RegisterInstance(request_context, request, response);
}

void MetaServiceHttp::GetInstanceInfo(coro_http::coro_http_connection *http_conn,
                                      proto::meta::GetInstanceInfoRequest *request,
                                      proto::meta::GetInstanceInfoResponse *response) {
    API_CONTEXT_INIT_HTTP(GetInstanceInfo);
    KVCM_LOG_INFO("[traceId: %s] GetInstanceInfo called with instance id: %s",
                  request->trace_id().c_str(),
                  request->instance_id().c_str());
    meta_service_impl_->GetInstanceInfo(request_context, request, response);
}

void MetaServiceHttp::GetCacheLocation(coro_http::coro_http_connection *http_conn,
                                       proto::meta::GetCacheLocationRequest *request,
                                       proto::meta::GetCacheLocationResponse *response) {
    API_CONTEXT_GET_COLLECTOR_AND_INIT_HTTP(GetCacheLocation, __NOTHING__);
    meta_service_impl_->GetCacheLocation(request_context, request, response);
}

void MetaServiceHttp::GetCacheMeta(coro_http::coro_http_connection *http_conn,
                                   proto::meta::GetCacheMetaRequest *request,
                                   proto::meta::GetCacheMetaResponse *response) {
    API_CONTEXT_GET_COLLECTOR_AND_INIT_HTTP(GetCacheMeta, __NOTHING__);
    KVCM_LOG_INFO("[traceId: %s] GetCacheMeta called with instance id: %s, block keys count: %d, "
                  "token ids count: %d, detail level: %d",
                  request->trace_id().c_str(),
                  request->instance_id().c_str(),
                  request->block_keys_size(),
                  request->token_ids_size(),
                  request->detail_level());
    KVCM_LOG_DEBUG("[traceId: %s] GetCacheMeta request details: %s",
                   request->trace_id().c_str(),
                   request->ShortDebugString().c_str());
    meta_service_impl_->GetCacheMeta(request_context, request, response);
}

void MetaServiceHttp::StartWriteCache(coro_http::coro_http_connection *http_conn,
                                      proto::meta::StartWriteCacheRequest *request,
                                      proto::meta::StartWriteCacheResponse *response) {
    API_CONTEXT_GET_COLLECTOR_AND_INIT_HTTP(StartWriteCache, __NOTHING__);
    KVCM_LOG_DEBUG("[traceId: %s] StartWriteCache request details: %s",
                   request->trace_id().c_str(),
                   request->ShortDebugString().c_str());
    meta_service_impl_->StartWriteCache(request_context, request, response);
}

void MetaServiceHttp::FinishWriteCache(coro_http::coro_http_connection *http_conn,
                                       proto::meta::FinishWriteCacheRequest *request,
                                       proto::meta::CommonResponse *response) {
    API_CONTEXT_GET_COLLECTOR_AND_INIT_HTTP(FinishWriteCache, __NOTHING__);
    KVCM_LOG_INFO("[traceId: %s] FinishWriteCache called with instance id: %s, write session id: %s",
                  request->trace_id().c_str(),
                  request->instance_id().c_str(),
                  request->write_session_id().c_str());
    KVCM_LOG_DEBUG("[traceId: %s] FinishWriteCache request details: %s",
                   request->trace_id().c_str(),
                   request->ShortDebugString().c_str());
    meta_service_impl_->FinishWriteCache(request_context, request, response);
}

void MetaServiceHttp::RemoveCache(coro_http::coro_http_connection *http_conn,
                                  proto::meta::RemoveCacheRequest *request,
                                  proto::meta::CommonResponse *response) {
    API_CONTEXT_GET_COLLECTOR_AND_INIT_HTTP(RemoveCache, __NOTHING__);
    KVCM_LOG_INFO("[traceId: %s] RemoveCache called with instance id: %s, block keys count: %d, "
                  "token ids count: %d",
                  request->trace_id().c_str(),
                  request->instance_id().c_str(),
                  request->block_keys_size(),
                  request->token_ids_size());
    KVCM_LOG_DEBUG("[traceId: %s] RemoveCache request details: %s",
                   request->trace_id().c_str(),
                   request->ShortDebugString().c_str());
    meta_service_impl_->RemoveCache(request_context, request, response);
}

void MetaServiceHttp::TrimCache(coro_http::coro_http_connection *http_conn,
                                proto::meta::TrimCacheRequest *request,
                                proto::meta::CommonResponse *response) {
    API_CONTEXT_GET_COLLECTOR_AND_INIT_HTTP(TrimCache, __NOTHING__);
    KVCM_LOG_DEBUG("[traceId: %s] TrimCache request details: %s",
                   request->trace_id().c_str(),
                   request->ShortDebugString().c_str());
    meta_service_impl_->TrimCache(request_context, request, response);
}

} // namespace kv_cache_manager
