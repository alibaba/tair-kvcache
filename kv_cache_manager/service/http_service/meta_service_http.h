#pragma once

#include <memory>

#include "kv_cache_manager/metrics/metrics_collector.h"
#include "kv_cache_manager/protocol/protobuf/meta_service.pb.h"
#include "kv_cache_manager/service/http_service/coro_http_service.h"
#include "kv_cache_manager/service/meta_service_metrics_base.h"

namespace kv_cache_manager {

class MetaServiceImpl;
class MetricsRegistry;
class RequestContext;
struct MetricsLifecycle;

class MetaServiceHttp : public CoroHttpService, public MetaServiceMetricsBase {
public:
    MetaServiceHttp(std::shared_ptr<MetricsRegistry> metrics_registry,
                    std::shared_ptr<MetaServiceImpl> meta_service_impl,
                    std::shared_ptr<RegistryManager> registry_manager,
                    std::shared_ptr<ServiceRequestExecutor> executor,
                    std::shared_ptr<MetricsLifecycle> metrics_lifecycle = nullptr);

    void Init() override;
    void RegisterHandler() override;

    async_simple::coro::Lazy<HttpDispatchResult> RegisterInstance(coro_http::coro_http_connection *http_conn,
                                                                  proto::meta::RegisterInstanceRequest *request,
                                                                  proto::meta::RegisterInstanceResponse *response);

    async_simple::coro::Lazy<HttpDispatchResult> GetInstanceInfo(coro_http::coro_http_connection *http_conn,
                                                                 proto::meta::GetInstanceInfoRequest *request,
                                                                 proto::meta::GetInstanceInfoResponse *response);
    async_simple::coro::Lazy<HttpDispatchResult> GetCacheMeta(coro_http::coro_http_connection *http_conn,
                                                              proto::meta::GetCacheMetaRequest *request,
                                                              proto::meta::GetCacheMetaResponse *response);
    async_simple::coro::Lazy<HttpDispatchResult> GetCacheLocation(coro_http::coro_http_connection *http_conn,
                                                                  proto::meta::GetCacheLocationRequest *request,
                                                                  proto::meta::GetCacheLocationResponse *response);
    async_simple::coro::Lazy<HttpDispatchResult>
    GetCacheLocationLen(coro_http::coro_http_connection *http_conn,
                        proto::meta::GetCacheLocationLenRequest *request,
                        proto::meta::GetCacheLocationLenResponse *response);
    async_simple::coro::Lazy<HttpDispatchResult>
    GetCacheLocationsByBackend(coro_http::coro_http_connection *http_conn,
                               proto::meta::GetCacheLocationsByBackendRequest *request,
                               proto::meta::GetCacheLocationsByBackendResponse *response);
    async_simple::coro::Lazy<HttpDispatchResult> StartWriteCache(coro_http::coro_http_connection *http_conn,
                                                                 proto::meta::StartWriteCacheRequest *request,
                                                                 proto::meta::StartWriteCacheResponse *response);

    async_simple::coro::Lazy<HttpDispatchResult> FinishWriteCache(coro_http::coro_http_connection *http_conn,
                                                                  proto::meta::FinishWriteCacheRequest *request,
                                                                  proto::meta::CommonResponse *response);
    async_simple::coro::Lazy<HttpDispatchResult> RemoveCache(coro_http::coro_http_connection *http_conn,
                                                             proto::meta::RemoveCacheRequest *request,
                                                             proto::meta::CommonResponse *response);
    async_simple::coro::Lazy<HttpDispatchResult> TrimCache(coro_http::coro_http_connection *http_conn,
                                                           proto::meta::TrimCacheRequest *request,
                                                           proto::meta::CommonResponse *response);
    async_simple::coro::Lazy<HttpDispatchResult> GetClusterInfo(coro_http::coro_http_connection *http_conn,
                                                                proto::meta::GetClusterInfoRequest *request,
                                                                proto::meta::GetClusterInfoResponse *response);

    async_simple::coro::Lazy<HttpDispatchResult> ReportEvent(coro_http::coro_http_connection *http_conn,
                                                             proto::meta::ReportEventRequest *request,
                                                             proto::meta::ReportEventResponse *response);

    async_simple::coro::Lazy<HttpDispatchResult> GetHostCacheState(coro_http::coro_http_connection *http_conn,
                                                                   proto::meta::GetHostCacheStateRequest *request,
                                                                   proto::meta::GetHostCacheStateResponse *response);

private:
    template <typename Request, typename Response>
    async_simple::coro::Lazy<HttpDispatchResult>
    DispatchRequest(RequestContext *context,
                    Request *request,
                    Response *response,
                    void (MetaServiceImpl::*method)(RequestContext *, const Request *, Response *));

    std::shared_ptr<MetaServiceImpl> meta_service_impl_;
    std::shared_ptr<ServiceRequestExecutor> executor_;
};

} // namespace kv_cache_manager
