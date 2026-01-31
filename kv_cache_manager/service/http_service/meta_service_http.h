#pragma once

#include <memory>

#include "kv_cache_manager/metrics/metrics_collector.h"
#include "kv_cache_manager/protocol/protobuf/meta_service.pb.h"
#include "kv_cache_manager/service/http_service/coro_http_service.h"
#include "kv_cache_manager/service/meta_service_metrics_base.h"

namespace kv_cache_manager {

class MetaServiceImpl;
class MetricsRegistry;

class MetaServiceHttp : public CoroHttpService, public MetaServiceMetricsBase {
public:
    MetaServiceHttp(std::shared_ptr<MetricsRegistry> metrics_registry,
                    std::shared_ptr<MetaServiceImpl> meta_service_impl,
                    std::shared_ptr<RegistryManager> registry_manager);

    void Init() override;
    void RegisterHandler() override;

    void RegisterInstance(coro_http::coro_http_connection *http_conn,
                          proto::meta::RegisterInstanceRequest *request,
                          proto::meta::RegisterInstanceResponse *response);

    void GetInstanceInfo(coro_http::coro_http_connection *http_conn,
                         proto::meta::GetInstanceInfoRequest *request,
                         proto::meta::GetInstanceInfoResponse *response);
    void GetCacheMeta(coro_http::coro_http_connection *http_conn,
                      proto::meta::GetCacheMetaRequest *request,
                      proto::meta::GetCacheMetaResponse *response);
    void GetCacheLocation(coro_http::coro_http_connection *http_conn,
                          proto::meta::GetCacheLocationRequest *request,
                          proto::meta::GetCacheLocationResponse *response);
    void StartWriteCache(coro_http::coro_http_connection *http_conn,
                         proto::meta::StartWriteCacheRequest *request,
                         proto::meta::StartWriteCacheResponse *response);

    void FinishWriteCache(coro_http::coro_http_connection *http_conn,
                          proto::meta::FinishWriteCacheRequest *request,
                          proto::meta::CommonResponse *response);
    void RemoveCache(coro_http::coro_http_connection *http_conn,
                     proto::meta::RemoveCacheRequest *request,
                     proto::meta::CommonResponse *response);
    void TrimCache(coro_http::coro_http_connection *http_conn,
                   proto::meta::TrimCacheRequest *request,
                   proto::meta::CommonResponse *response);

private:
    std::shared_ptr<MetaServiceImpl> meta_service_impl_;
};

} // namespace kv_cache_manager
