#include "kv_cache_manager/service/http_service/meta_service_http.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/metrics/metrics_registry.h"
#include "kv_cache_manager/protocol/protobuf/meta_service.pb.h"
#include "kv_cache_manager/service/meta_service_impl.h"
#include "kv_cache_manager/service/util/common.h"
#include "kv_cache_manager/service/util/report_event_json_parser.h"

namespace kv_cache_manager {

// Keep the existing per-instance early return before creating RequestContext.
// Only this HTTP coroutine variant needs co_return; synchronous adapters keep
// using the shared macros unchanged.
#define API_CONTEXT_GET_COLLECTOR_AND_INIT_HTTP_CORO(method)                                                           \
    auto metrics_collector = get_metrics_collector_from_map_for_##method(request->instance_id());                      \
    if (metrics_collector == nullptr) {                                                                                \
        KVCM_LOG_ERROR("get " #method " metrics collector failed, instance_id: %s", request->instance_id().c_str());   \
        auto *status = response->mutable_header()->mutable_status();                                                   \
        status->set_code(proto::meta::INSTANCE_NOT_EXIST);                                                             \
        status->set_message("get " #method " metrics collector failed");                                               \
        co_return HttpDispatchResult{};                                                                                \
    }                                                                                                                  \
    API_CONTEXT_INIT(metrics_collector, GetHttpClientIp, http_conn)

MetaServiceHttp::MetaServiceHttp(std::shared_ptr<MetricsRegistry> metrics_registry,
                                 std::shared_ptr<MetaServiceImpl> meta_service_impl,
                                 std::shared_ptr<RegistryManager> registry_manager,
                                 std::shared_ptr<ServiceRequestExecutor> executor,
                                 std::shared_ptr<MetricsLifecycle> metrics_lifecycle)
    : MetaServiceMetricsBase(std::move(metrics_registry), registry_manager, std::move(metrics_lifecycle))
    , meta_service_impl_(std::move(meta_service_impl))
    , executor_(std::move(executor)) {
    if (!executor_) {
        throw std::invalid_argument("MetaServiceHttp requires a service request executor");
    }
}

template <typename Request, typename Response>
async_simple::coro::Lazy<CoroHttpService::HttpDispatchResult>
MetaServiceHttp::DispatchRequest(RequestContext *context,
                                 Request *request,
                                 Response *response,
                                 void (MetaServiceImpl::*method)(RequestContext *, const Request *, Response *)) {
    // The caller's coroutine retains RequestContext, and its outer handler
    // retains both protobuf messages, until this dispatch has completed.
    return DispatchHttpRequest(
        *executor_,
        [impl = meta_service_impl_, context, request, response, method] {
            std::invoke(method, impl.get(), context, request, response);
            return context->TakeReusableResponseJson();
        },
        [context, response](const char *message) {
            auto *header = response->mutable_header();
            header->set_request_id(context->request_id());
            auto *status = header->mutable_status();
            status->set_code(proto::meta::SERVICE_NOT_READY);
            status->set_message(message);
        });
}

void MetaServiceHttp::Init() { MetaServiceMetricsBase::InitMetrics(); }

void MetaServiceHttp::RegisterHandler() {
    REGISTER_HTTP_HANDLER_FOR_META_SERVICE(
        Post, registerInstance, RegisterInstance, RegisterInstance, RegisterInstance);
    REGISTER_HTTP_HANDLER_FOR_META_SERVICE(Post, getInstanceInfo, GetInstanceInfo, GetInstanceInfo, GetInstanceInfo);
    REGISTER_HTTP_HANDLER_FOR_META_SERVICE(Post, getCacheMeta, GetCacheMeta, GetCacheMeta, GetCacheMeta);
    REGISTER_HTTP_HANDLER_FOR_META_SERVICE(
        Post, getCacheLocation, GetCacheLocation, GetCacheLocation, GetCacheLocation);
    REGISTER_HTTP_HANDLER_FOR_META_SERVICE(
        Post, getCacheLocationLen, GetCacheLocationLen, GetCacheLocationLen, GetCacheLocationLen);
    REGISTER_HTTP_HANDLER_FOR_META_SERVICE(Post,
                                           getCacheLocationsByBackend,
                                           GetCacheLocationsByBackend,
                                           GetCacheLocationsByBackend,
                                           GetCacheLocationsByBackend);
    REGISTER_HTTP_HANDLER_FOR_META_SERVICE(Post, startWriteCache, StartWriteCache, StartWriteCache, StartWriteCache);
    REGISTER_HTTP_HANDLER_FOR_META_SERVICE(Post, finishWriteCache, FinishWriteCache, Common, FinishWriteCache);
    REGISTER_HTTP_HANDLER_FOR_META_SERVICE(Post, removeCache, RemoveCache, Common, RemoveCache);
    REGISTER_HTTP_HANDLER_FOR_META_SERVICE(Post, trimCache, TrimCache, Common, TrimCache);
    REGISTER_HTTP_HANDLER_FOR_META_SERVICE(Post, getClusterInfo, GetClusterInfo, GetClusterInfo, GetClusterInfo);
    RegisterPostHandler(
        "/api/reportEvent",
        GetArenaHandler<MetaServiceHttp, proto::meta::ReportEventRequest, proto::meta::ReportEventResponse>(
            &MetaServiceHttp::ReportEvent, &ReportEventJsonParser::FromMutableNullTerminatedJson));
    REGISTER_HTTP_HANDLER_FOR_META_SERVICE(
        Post, getHostCacheState, GetHostCacheState, GetHostCacheState, GetHostCacheState);
}

async_simple::coro::Lazy<CoroHttpService::HttpDispatchResult>
MetaServiceHttp::RegisterInstance(coro_http::coro_http_connection *http_conn,
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
    co_return co_await DispatchRequest(request_context, request, response, &MetaServiceImpl::RegisterInstance);
}

async_simple::coro::Lazy<CoroHttpService::HttpDispatchResult>
MetaServiceHttp::GetInstanceInfo(coro_http::coro_http_connection *http_conn,
                                 proto::meta::GetInstanceInfoRequest *request,
                                 proto::meta::GetInstanceInfoResponse *response) {
    API_CONTEXT_INIT_HTTP(GetInstanceInfo);
    KVCM_LOG_INFO("[traceId: %s] GetInstanceInfo called with instance id: %s",
                  request->trace_id().c_str(),
                  request->instance_id().c_str());
    co_return co_await DispatchRequest(request_context, request, response, &MetaServiceImpl::GetInstanceInfo);
}

async_simple::coro::Lazy<CoroHttpService::HttpDispatchResult>
MetaServiceHttp::GetCacheLocation(coro_http::coro_http_connection *http_conn,
                                  proto::meta::GetCacheLocationRequest *request,
                                  proto::meta::GetCacheLocationResponse *response) {
    API_CONTEXT_GET_COLLECTOR_AND_INIT_HTTP_CORO(GetCacheLocation);
    co_return co_await DispatchRequest(request_context, request, response, &MetaServiceImpl::GetCacheLocation);
}

async_simple::coro::Lazy<CoroHttpService::HttpDispatchResult>
MetaServiceHttp::GetCacheLocationLen(coro_http::coro_http_connection *http_conn,
                                     proto::meta::GetCacheLocationLenRequest *request,
                                     proto::meta::GetCacheLocationLenResponse *response) {
    API_CONTEXT_GET_COLLECTOR_AND_INIT_HTTP_CORO(GetCacheLocationLen);
    co_return co_await DispatchRequest(request_context, request, response, &MetaServiceImpl::GetCacheLocationLen);
}

async_simple::coro::Lazy<CoroHttpService::HttpDispatchResult>
MetaServiceHttp::GetCacheLocationsByBackend(coro_http::coro_http_connection *http_conn,
                                            proto::meta::GetCacheLocationsByBackendRequest *request,
                                            proto::meta::GetCacheLocationsByBackendResponse *response) {
    API_CONTEXT_GET_COLLECTOR_AND_INIT_HTTP_CORO(GetCacheLocationsByBackend);
    co_return co_await DispatchRequest(
        request_context, request, response, &MetaServiceImpl::GetCacheLocationsByBackend);
}

async_simple::coro::Lazy<CoroHttpService::HttpDispatchResult>
MetaServiceHttp::GetCacheMeta(coro_http::coro_http_connection *http_conn,
                              proto::meta::GetCacheMetaRequest *request,
                              proto::meta::GetCacheMetaResponse *response) {
    API_CONTEXT_GET_COLLECTOR_AND_INIT_HTTP_CORO(GetCacheMeta);
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
    co_return co_await DispatchRequest(request_context, request, response, &MetaServiceImpl::GetCacheMeta);
}

async_simple::coro::Lazy<CoroHttpService::HttpDispatchResult>
MetaServiceHttp::StartWriteCache(coro_http::coro_http_connection *http_conn,
                                 proto::meta::StartWriteCacheRequest *request,
                                 proto::meta::StartWriteCacheResponse *response) {
    API_CONTEXT_GET_COLLECTOR_AND_INIT_HTTP_CORO(StartWriteCache);
    KVCM_LOG_DEBUG("[traceId: %s] StartWriteCache request details: %s",
                   request->trace_id().c_str(),
                   request->ShortDebugString().c_str());
    co_return co_await DispatchRequest(request_context, request, response, &MetaServiceImpl::StartWriteCache);
}

async_simple::coro::Lazy<CoroHttpService::HttpDispatchResult>
MetaServiceHttp::FinishWriteCache(coro_http::coro_http_connection *http_conn,
                                  proto::meta::FinishWriteCacheRequest *request,
                                  proto::meta::CommonResponse *response) {
    API_CONTEXT_GET_COLLECTOR_AND_INIT_HTTP_CORO(FinishWriteCache);
    KVCM_LOG_INFO("[traceId: %s] FinishWriteCache called with instance id: %s, write session id: %s",
                  request->trace_id().c_str(),
                  request->instance_id().c_str(),
                  request->write_session_id().c_str());
    KVCM_LOG_DEBUG("[traceId: %s] FinishWriteCache request details: %s",
                   request->trace_id().c_str(),
                   request->ShortDebugString().c_str());
    co_return co_await DispatchRequest(request_context, request, response, &MetaServiceImpl::FinishWriteCache);
}

async_simple::coro::Lazy<CoroHttpService::HttpDispatchResult>
MetaServiceHttp::RemoveCache(coro_http::coro_http_connection *http_conn,
                             proto::meta::RemoveCacheRequest *request,
                             proto::meta::CommonResponse *response) {
    API_CONTEXT_GET_COLLECTOR_AND_INIT_HTTP_CORO(RemoveCache);
    KVCM_LOG_INFO("[traceId: %s] RemoveCache called with instance id: %s, block keys count: %d, "
                  "token ids count: %d",
                  request->trace_id().c_str(),
                  request->instance_id().c_str(),
                  request->block_keys_size(),
                  request->token_ids_size());
    KVCM_LOG_DEBUG("[traceId: %s] RemoveCache request details: %s",
                   request->trace_id().c_str(),
                   request->ShortDebugString().c_str());
    co_return co_await DispatchRequest(request_context, request, response, &MetaServiceImpl::RemoveCache);
}

async_simple::coro::Lazy<CoroHttpService::HttpDispatchResult>
MetaServiceHttp::TrimCache(coro_http::coro_http_connection *http_conn,
                           proto::meta::TrimCacheRequest *request,
                           proto::meta::CommonResponse *response) {
    API_CONTEXT_GET_COLLECTOR_AND_INIT_HTTP_CORO(TrimCache);
    KVCM_LOG_DEBUG("[traceId: %s] TrimCache request details: %s",
                   request->trace_id().c_str(),
                   request->ShortDebugString().c_str());
    co_return co_await DispatchRequest(request_context, request, response, &MetaServiceImpl::TrimCache);
}

async_simple::coro::Lazy<CoroHttpService::HttpDispatchResult>
MetaServiceHttp::GetClusterInfo(coro_http::coro_http_connection *http_conn,
                                proto::meta::GetClusterInfoRequest *request,
                                proto::meta::GetClusterInfoResponse *response) {
    // instance_id 可能尚未注册（如 RegisterInstance 之前），此时 fallback 到全局 collector
    auto metrics_collector = get_metrics_collector_from_map_for_GetClusterInfo(request->instance_id());
    if (metrics_collector == nullptr) {
        metrics_collector = KVCM_METRICS_COLLECTOR_(GetClusterInfo);
    }
    API_CONTEXT_INIT(metrics_collector, GetHttpClientIp, http_conn)
    KVCM_LOG_DEBUG("[traceId: %s] GetClusterInfo request details: %s",
                   request->trace_id().c_str(),
                   request->ShortDebugString().c_str());
    co_return co_await DispatchRequest(request_context, request, response, &MetaServiceImpl::GetClusterInfo);
}

async_simple::coro::Lazy<CoroHttpService::HttpDispatchResult>
MetaServiceHttp::ReportEvent(coro_http::coro_http_connection *http_conn,
                             proto::meta::ReportEventRequest *request,
                             proto::meta::ReportEventResponse *response) {
    std::string metrics_type;
    auto metrics_collector = ResolveReportEventMetricsCollector(*request, metrics_type);
    API_CONTEXT_INIT(metrics_collector, GetHttpClientIp, http_conn)
    std::string first_event_type = "N/A";
    std::string first_block_key = "N/A";
    if (request->events_size() > 0) {
        const auto &first_event = request->events(0);
        first_event_type = proto::meta::ReportEventType_Name(first_event.event_type());
        if (first_event.has_block_add()) {
            first_block_key = first_event.block_add().block_key();
        } else if (first_event.has_block_delete()) {
            first_block_key = first_event.block_delete().block_key();
        } else if (first_event.has_block_snapshot()) {
            if (first_event.block_snapshot().blocks_size() > 0) {
                first_block_key = first_event.block_snapshot().blocks(0).block_key();
            }
        }
    }
    KVCM_LOG_DEBUG("[traceId: %s] ReportEvent called, instance_id: %s, host_ip_port: %s, event_count: %d, "
                   "first_event_type: %s, first_block_key: %s",
                   request->trace_id().c_str(),
                   request->instance_id().c_str(),
                   request->host_ip_port().c_str(),
                   request->events_size(),
                   first_event_type.c_str(),
                   first_block_key.c_str());
    AttachReportEventTypeMetricsCollectors(*request, metrics_type, request_context);
    co_return co_await DispatchRequest(request_context, request, response, &MetaServiceImpl::ReportEvent);
}

async_simple::coro::Lazy<CoroHttpService::HttpDispatchResult>
MetaServiceHttp::GetHostCacheState(coro_http::coro_http_connection *http_conn,
                                   proto::meta::GetHostCacheStateRequest *request,
                                   proto::meta::GetHostCacheStateResponse *response) {
    API_CONTEXT_GET_COLLECTOR_AND_INIT_HTTP_CORO(GetHostCacheState);
    KVCM_LOG_DEBUG("[traceId: %s] GetHostCacheState called, instance_id: %s, block_cache_keys_count: %d",
                   request->trace_id().c_str(),
                   request->instance_id().c_str(),
                   request->block_cache_keys_size());
    co_return co_await DispatchRequest(request_context, request, response, &MetaServiceImpl::GetHostCacheState);
}

#undef API_CONTEXT_GET_COLLECTOR_AND_INIT_HTTP_CORO

} // namespace kv_cache_manager
