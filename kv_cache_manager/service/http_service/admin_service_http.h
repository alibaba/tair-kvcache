#pragma once

#include <memory>
#include <string>

#include "kv_cache_manager/metrics/metrics_collector.h"
#include "kv_cache_manager/protocol/protobuf/admin_service.pb.h"
#include "kv_cache_manager/service/http_service/coro_http_service.h"

namespace kv_cache_manager {

class AdminServiceImpl;
class MetricsRegistry;

class AdminServiceHttp : public CoroHttpService {
public:
    AdminServiceHttp(std::shared_ptr<MetricsRegistry> metrics_registry,
                     std::shared_ptr<AdminServiceImpl> admin_service_impl);
    AdminServiceHttp(std::shared_ptr<MetricsRegistry> metrics_registry,
                     std::shared_ptr<AdminServiceImpl> admin_service_impl,
                     bool enable_prometheus,
                     const std::string &prometheus_prefix);

    void Init() override;
    void RegisterHandler() override;

    CachedJsonResponse AddStorage(coro_http::coro_http_connection *http_conn,
                                  proto::admin::AddStorageRequest *request,
                                  proto::admin::CommonResponse *response);
    CachedJsonResponse EnableStorage(coro_http::coro_http_connection *http_conn,
                                     proto::admin::EnableStorageRequest *request,
                                     proto::admin::CommonResponse *response);
    CachedJsonResponse DisableStorage(coro_http::coro_http_connection *http_conn,
                                      proto::admin::DisableStorageRequest *request,
                                      proto::admin::CommonResponse *response);
    CachedJsonResponse RemoveStorage(coro_http::coro_http_connection *http_conn,
                                     proto::admin::RemoveStorageRequest *request,
                                     proto::admin::CommonResponse *response);
    CachedJsonResponse UpdateStorage(coro_http::coro_http_connection *http_conn,
                                     proto::admin::UpdateStorageRequest *request,
                                     proto::admin::CommonResponse *response);
    CachedJsonResponse ListStorage(coro_http::coro_http_connection *http_conn,
                                   proto::admin::ListStorageRequest *request,
                                   proto::admin::ListStorageResponse *response);

    CachedJsonResponse CreateInstanceGroup(coro_http::coro_http_connection *http_conn,
                                           proto::admin::CreateInstanceGroupRequest *request,
                                           proto::admin::CommonResponse *response);
    CachedJsonResponse UpdateInstanceGroup(coro_http::coro_http_connection *http_conn,
                                           proto::admin::UpdateInstanceGroupRequest *request,
                                           proto::admin::CommonResponse *response);
    CachedJsonResponse RemoveInstanceGroup(coro_http::coro_http_connection *http_conn,
                                           proto::admin::RemoveInstanceGroupRequest *request,
                                           proto::admin::CommonResponse *response);
    CachedJsonResponse GetInstanceGroup(coro_http::coro_http_connection *http_conn,
                                        proto::admin::GetInstanceGroupRequest *request,
                                        proto::admin::GetInstanceGroupResponse *response);
    CachedJsonResponse ListInstanceGroup(coro_http::coro_http_connection *http_conn,
                                         proto::admin::ListInstanceGroupRequest *request,
                                         proto::admin::ListInstanceGroupResponse *response);

    CachedJsonResponse GetCacheMeta(coro_http::coro_http_connection *http_conn,
                                    proto::admin::GetCacheMetaRequest *request,
                                    proto::admin::GetCacheMetaResponse *response);
    CachedJsonResponse RemoveCache(coro_http::coro_http_connection *http_conn,
                                   proto::admin::RemoveCacheRequest *request,
                                   proto::admin::CommonResponse *response);
    CachedJsonResponse MigrateCache(coro_http::coro_http_connection *http_conn,
                                    proto::admin::MigrateCacheRequest *request,
                                    proto::admin::MigrateCacheResponse *response);

    CachedJsonResponse RegisterInstance(coro_http::coro_http_connection *http_conn,
                                        proto::admin::RegisterInstanceRequest *request,
                                        proto::admin::CommonResponse *response);
    CachedJsonResponse RemoveInstance(coro_http::coro_http_connection *http_conn,
                                      proto::admin::RemoveInstanceRequest *request,
                                      proto::admin::CommonResponse *response);
    CachedJsonResponse GetInstanceInfo(coro_http::coro_http_connection *http_conn,
                                       proto::admin::GetInstanceInfoRequest *request,
                                       proto::admin::GetInstanceInfoResponse *response);
    CachedJsonResponse ListInstanceInfo(coro_http::coro_http_connection *http_conn,
                                        proto::admin::ListInstanceInfoRequest *request,
                                        proto::admin::ListInstanceInfoResponse *response);

    CachedJsonResponse AddAccount(coro_http::coro_http_connection *http_conn,
                                  proto::admin::AddAccountRequest *request,
                                  proto::admin::CommonResponse *response);
    CachedJsonResponse DeleteAccount(coro_http::coro_http_connection *http_conn,
                                     proto::admin::DeleteAccountRequest *request,
                                     proto::admin::CommonResponse *response);
    CachedJsonResponse ListAccount(coro_http::coro_http_connection *http_conn,
                                   proto::admin::ListAccountRequest *request,
                                   proto::admin::ListAccountResponse *response);

    CachedJsonResponse GenConfigSnapshot(coro_http::coro_http_connection *http_conn,
                                         proto::admin::GenConfigSnapshotRequest *request,
                                         proto::admin::ConfigSnapShotResponse *response);
    CachedJsonResponse LoadConfigSnapshot(coro_http::coro_http_connection *http_conn,
                                          proto::admin::LoadConfigSnapshotRequest *request,
                                          proto::admin::CommonResponse *response);

    CachedJsonResponse GetMetrics(coro_http::coro_http_connection *http_conn,
                                  proto::admin::GetMetricsRequest *request,
                                  proto::admin::GetMetricsResponse *response);

    // HA APIs
    CachedJsonResponse CheckHealth(coro_http::coro_http_connection *http_conn,
                                   proto::admin::CheckHealthRequest *request,
                                   proto::admin::CheckHealthResponse *response);
    CachedJsonResponse GetManagerClusterInfo(coro_http::coro_http_connection *http_conn,
                                             proto::admin::GetManagerClusterInfoRequest *request,
                                             proto::admin::GetManagerClusterInfoResponse *response);
    CachedJsonResponse LeaderDemote(coro_http::coro_http_connection *http_conn,
                                    proto::admin::LeaderDemoteRequest *request,
                                    proto::admin::CommonResponse *response);
    CachedJsonResponse GetLeaderElectorConfig(coro_http::coro_http_connection *http_conn,
                                              proto::admin::GetLeaderElectorConfigRequest *request,
                                              proto::admin::GetLeaderElectorConfigResponse *response);
    CachedJsonResponse UpdateLeaderElectorConfig(coro_http::coro_http_connection *http_conn,
                                                 proto::admin::UpdateLeaderElectorConfigRequest *request,
                                                 proto::admin::CommonResponse *response);
    CachedJsonResponse UpdateLogger(coro_http::coro_http_connection *http_conn,
                                    proto::admin::UpdateLoggerRequest *request,
                                    proto::admin::CommonResponse *response);

private:
    std::shared_ptr<MetricsRegistry> metrics_registry_;
    std::shared_ptr<AdminServiceImpl> admin_service_impl_;
    bool enable_prometheus_ = true;
    std::string prometheus_prefix_ = "kvcm";

    // for storage APIs
    KVCM_DECLARE_METRICS_COLLECTOR_(AddStorage);
    KVCM_DECLARE_METRICS_COLLECTOR_(EnableStorage);
    KVCM_DECLARE_METRICS_COLLECTOR_(DisableStorage);
    KVCM_DECLARE_METRICS_COLLECTOR_(RemoveStorage);
    KVCM_DECLARE_METRICS_COLLECTOR_(UpdateStorage);
    KVCM_DECLARE_METRICS_COLLECTOR_(ListStorage);

    // for instance group APIs
    KVCM_DECLARE_METRICS_COLLECTOR_(CreateInstanceGroup);
    KVCM_DECLARE_METRICS_COLLECTOR_(UpdateInstanceGroup);
    KVCM_DECLARE_METRICS_COLLECTOR_(RemoveInstanceGroup);
    KVCM_DECLARE_METRICS_COLLECTOR_(GetInstanceGroup);
    KVCM_DECLARE_METRICS_COLLECTOR_(ListInstanceGroup);

    // for cache APIs
    KVCM_DECLARE_METRICS_COLLECTOR_(GetCacheMeta);
    KVCM_DECLARE_METRICS_COLLECTOR_(RemoveCache);
    KVCM_DECLARE_METRICS_COLLECTOR_(MigrateCache);

    // for instance APIs
    KVCM_DECLARE_METRICS_COLLECTOR_(RegisterInstance);
    KVCM_DECLARE_METRICS_COLLECTOR_(RemoveInstance);
    KVCM_DECLARE_METRICS_COLLECTOR_(GetInstanceInfo);
    KVCM_DECLARE_METRICS_COLLECTOR_(ListInstanceInfo);

    // for account APIs
    KVCM_DECLARE_METRICS_COLLECTOR_(AddAccount);
    KVCM_DECLARE_METRICS_COLLECTOR_(DeleteAccount);
    KVCM_DECLARE_METRICS_COLLECTOR_(ListAccount);

    // for config snapshot APIs
    KVCM_DECLARE_METRICS_COLLECTOR_(GenConfigSnapshot);
    KVCM_DECLARE_METRICS_COLLECTOR_(LoadConfigSnapshot);

    // for metrics APIs
    KVCM_DECLARE_METRICS_COLLECTOR_(GetMetrics);

    // for HA APIs
    KVCM_DECLARE_METRICS_COLLECTOR_(CheckHealth);
    KVCM_DECLARE_METRICS_COLLECTOR_(GetManagerClusterInfo);
    KVCM_DECLARE_METRICS_COLLECTOR_(LeaderDemote);
    KVCM_DECLARE_METRICS_COLLECTOR_(GetLeaderElectorConfig);
    KVCM_DECLARE_METRICS_COLLECTOR_(UpdateLeaderElectorConfig);

    // for logging control APIs
    KVCM_DECLARE_METRICS_COLLECTOR_(UpdateLogger);
};

} // namespace kv_cache_manager
