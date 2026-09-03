#pragma once

#include <memory>

#include "kv_cache_manager/metrics/metrics_collector.h"
#include "kv_cache_manager/protocol/protobuf/meta_service.pb.h"
#include "kv_cache_manager/service/http_service/coro_http_service.h"
#include "kv_cache_manager/service/meta_service_metrics_base.h"

namespace kv_cache_manager {

class MetaServiceImpl;
class MetricsRegistry;
struct MetricsLifecycle;

class MetaServiceHttp : public CoroHttpService, public MetaServiceMetricsBase {
public:
    MetaServiceHttp(std::shared_ptr<MetricsRegistry> metrics_registry,
                    std::shared_ptr<MetaServiceImpl> meta_service_impl,
                    std::shared_ptr<RegistryManager> registry_manager,
                    std::shared_ptr<MetricsLifecycle> metrics_lifecycle = nullptr);

    void Init() override;
    void RegisterHandler() override;

    CachedJsonResponse RegisterInstance(coro_http::coro_http_connection *http_conn,
                                        proto::meta::RegisterInstanceRequest *request,
                                        proto::meta::RegisterInstanceResponse *response);

    CachedJsonResponse GetInstanceInfo(coro_http::coro_http_connection *http_conn,
                                       proto::meta::GetInstanceInfoRequest *request,
                                       proto::meta::GetInstanceInfoResponse *response);
    CachedJsonResponse GetCacheMeta(coro_http::coro_http_connection *http_conn,
                                    proto::meta::GetCacheMetaRequest *request,
                                    proto::meta::GetCacheMetaResponse *response);
    CachedJsonResponse GetCacheLocation(coro_http::coro_http_connection *http_conn,
                                        proto::meta::GetCacheLocationRequest *request,
                                        proto::meta::GetCacheLocationResponse *response);
    CachedJsonResponse GetCacheLocationLen(coro_http::coro_http_connection *http_conn,
                                           proto::meta::GetCacheLocationLenRequest *request,
                                           proto::meta::GetCacheLocationLenResponse *response);
    CachedJsonResponse GetCacheLocationsByBackend(coro_http::coro_http_connection *http_conn,
                                                  proto::meta::GetCacheLocationsByBackendRequest *request,
                                                  proto::meta::GetCacheLocationsByBackendResponse *response);
    CachedJsonResponse StartWriteCache(coro_http::coro_http_connection *http_conn,
                                       proto::meta::StartWriteCacheRequest *request,
                                       proto::meta::StartWriteCacheResponse *response);

    CachedJsonResponse FinishWriteCache(coro_http::coro_http_connection *http_conn,
                                        proto::meta::FinishWriteCacheRequest *request,
                                        proto::meta::CommonResponse *response);
    CachedJsonResponse RemoveCache(coro_http::coro_http_connection *http_conn,
                                   proto::meta::RemoveCacheRequest *request,
                                   proto::meta::CommonResponse *response);
    CachedJsonResponse TrimCache(coro_http::coro_http_connection *http_conn,
                                 proto::meta::TrimCacheRequest *request,
                                 proto::meta::CommonResponse *response);
    CachedJsonResponse GetClusterInfo(coro_http::coro_http_connection *http_conn,
                                      proto::meta::GetClusterInfoRequest *request,
                                      proto::meta::GetClusterInfoResponse *response);

    CachedJsonResponse ReportEvent(coro_http::coro_http_connection *http_conn,
                                   proto::meta::ReportEventRequest *request,
                                   proto::meta::ReportEventResponse *response);

    CachedJsonResponse GetHostCacheState(coro_http::coro_http_connection *http_conn,
                                         proto::meta::GetHostCacheStateRequest *request,
                                         proto::meta::GetHostCacheStateResponse *response);

private:
    std::shared_ptr<MetaServiceImpl> meta_service_impl_;
};

} // namespace kv_cache_manager
