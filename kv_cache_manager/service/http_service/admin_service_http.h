#pragma once

#include <memory>

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

    void Init() override;
    void RegisterHandler() override;

    void AddStorage(coro_http::coro_http_connection *http_conn,
                    proto::admin::AddStorageRequest *request,
                    proto::admin::CommonResponse *response);
    void EnableStorage(coro_http::coro_http_connection *http_conn,
                       proto::admin::EnableStorageRequest *request,
                       proto::admin::CommonResponse *response);
    void DisableStorage(coro_http::coro_http_connection *http_conn,
                        proto::admin::DisableStorageRequest *request,
                        proto::admin::CommonResponse *response);
    void RemoveStorage(coro_http::coro_http_connection *http_conn,
                       proto::admin::RemoveStorageRequest *request,
                       proto::admin::CommonResponse *response);
    void UpdateStorage(coro_http::coro_http_connection *http_conn,
                       proto::admin::UpdateStorageRequest *request,
                       proto::admin::CommonResponse *response);
    void ListStorage(coro_http::coro_http_connection *http_conn,
                     proto::admin::ListStorageRequest *request,
                     proto::admin::ListStorageResponse *response);

    void CreateInstanceGroup(coro_http::coro_http_connection *http_conn,
                             proto::admin::CreateInstanceGroupRequest *request,
                             proto::admin::CommonResponse *response);
    void UpdateInstanceGroup(coro_http::coro_http_connection *http_conn,
                             proto::admin::UpdateInstanceGroupRequest *request,
                             proto::admin::CommonResponse *response);
    void RemoveInstanceGroup(coro_http::coro_http_connection *http_conn,
                             proto::admin::RemoveInstanceGroupRequest *request,
                             proto::admin::CommonResponse *response);
    void GetInstanceGroup(coro_http::coro_http_connection *http_conn,
                          proto::admin::GetInstanceGroupRequest *request,
                          proto::admin::GetInstanceGroupResponse *response);

    void GetCacheMeta(coro_http::coro_http_connection *http_conn,
                      proto::admin::GetCacheMetaRequest *request,
                      proto::admin::GetCacheMetaResponse *response);
    void RemoveCache(coro_http::coro_http_connection *http_conn,
                     proto::admin::RemoveCacheRequest *request,
                     proto::admin::CommonResponse *response);

    void RegisterInstance(coro_http::coro_http_connection *http_conn,
                          proto::admin::RegisterInstanceRequest *request,
                          proto::admin::CommonResponse *response);
    void RemoveInstance(coro_http::coro_http_connection *http_conn,
                        proto::admin::RemoveInstanceRequest *request,
                        proto::admin::CommonResponse *response);
    void GetInstanceInfo(coro_http::coro_http_connection *http_conn,
                         proto::admin::GetInstanceInfoRequest *request,
                         proto::admin::GetInstanceInfoResponse *response);
    void ListInstanceInfo(coro_http::coro_http_connection *http_conn,
                          proto::admin::ListInstanceInfoRequest *request,
                          proto::admin::ListInstanceInfoResponse *response);

    void AddAccount(coro_http::coro_http_connection *http_conn,
                    proto::admin::AddAccountRequest *request,
                    proto::admin::CommonResponse *response);
    void DeleteAccount(coro_http::coro_http_connection *http_conn,
                       proto::admin::DeleteAccountRequest *request,
                       proto::admin::CommonResponse *response);
    void ListAccount(coro_http::coro_http_connection *http_conn,
                     proto::admin::ListAccountRequest *request,
                     proto::admin::ListAccountResponse *response);

    void GenConfigSnapshot(coro_http::coro_http_connection *http_conn,
                           proto::admin::GenConfigSnapshotRequest *request,
                           proto::admin::ConfigSnapShotResponse *response);
    void LoadConfigSnapshot(coro_http::coro_http_connection *http_conn,
                            proto::admin::LoadConfigSnapshotRequest *request,
                            proto::admin::CommonResponse *response);

    void GetMetrics(coro_http::coro_http_connection *http_conn,
                    proto::admin::GetMetricsRequest *request,
                    proto::admin::GetMetricsResponse *response);

    // HA APIs
    void CheckHealth(coro_http::coro_http_connection *http_conn,
                     proto::admin::CheckHealthRequest *request,
                     proto::admin::CheckHealthResponse *response);
    void GetManagerClusterInfo(coro_http::coro_http_connection *http_conn,
                               proto::admin::GetManagerClusterInfoRequest *request,
                               proto::admin::GetManagerClusterInfoResponse *response);
    void LeaderDemote(coro_http::coro_http_connection *http_conn,
                      proto::admin::LeaderDemoteRequest *request,
                      proto::admin::CommonResponse *response);
    void GetLeaderElectorConfig(coro_http::coro_http_connection *http_conn,
                                proto::admin::GetLeaderElectorConfigRequest *request,
                                proto::admin::GetLeaderElectorConfigResponse *response);
    void UpdateLeaderElectorConfig(coro_http::coro_http_connection *http_conn,
                                   proto::admin::UpdateLeaderElectorConfigRequest *request,
                                   proto::admin::CommonResponse *response);
    void UpdateLogger(coro_http::coro_http_connection *http_conn,
                      proto::admin::UpdateLoggerRequest *request,
                      proto::admin::CommonResponse *response);

private:
    std::shared_ptr<MetricsRegistry> metrics_registry_;
    std::shared_ptr<AdminServiceImpl> admin_service_impl_;

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

    // for cache APIs
    KVCM_DECLARE_METRICS_COLLECTOR_(GetCacheMeta);
    KVCM_DECLARE_METRICS_COLLECTOR_(RemoveCache);

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
