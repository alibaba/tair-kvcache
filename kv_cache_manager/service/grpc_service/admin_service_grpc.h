#pragma once

#include <grpcpp/grpcpp.h>
#include <memory>

#include "grpc++/grpc++.h"
#include "kv_cache_manager/metrics/metrics_collector.h"
#include "kv_cache_manager/protocol/protobuf/admin_service.grpc.pb.h"

namespace kv_cache_manager {

class AdminServiceImpl;
class MetricsRegistry;

class AdminServiceGRpc final : public proto::admin::AdminService::Service {
public:
    AdminServiceGRpc(std::shared_ptr<MetricsRegistry> metrics_registry,
                     std::shared_ptr<AdminServiceImpl> admin_service_impl);

    void Init();

    grpc::Status AddStorage(grpc::ServerContext *context,
                            const proto::admin::AddStorageRequest *request,
                            proto::admin::CommonResponse *response) override;
    grpc::Status EnableStorage(grpc::ServerContext *context,
                               const proto::admin::EnableStorageRequest *request,
                               proto::admin::CommonResponse *response) override;
    grpc::Status DisableStorage(grpc::ServerContext *context,
                                const proto::admin::DisableStorageRequest *request,
                                proto::admin::CommonResponse *response) override;
    grpc::Status RemoveStorage(grpc::ServerContext *context,
                               const proto::admin::RemoveStorageRequest *request,
                               proto::admin::CommonResponse *response) override;
    grpc::Status UpdateStorage(grpc::ServerContext *context,
                               const proto::admin::UpdateStorageRequest *request,
                               proto::admin::CommonResponse *response) override;
    grpc::Status ListStorage(grpc::ServerContext *context,
                             const proto::admin::ListStorageRequest *request,
                             proto::admin::ListStorageResponse *response) override;
    grpc::Status CreateInstanceGroup(grpc::ServerContext *context,
                                     const proto::admin::CreateInstanceGroupRequest *request,
                                     proto::admin::CommonResponse *response) override;
    grpc::Status UpdateInstanceGroup(grpc::ServerContext *context,
                                     const proto::admin::UpdateInstanceGroupRequest *request,
                                     proto::admin::CommonResponse *response) override;
    grpc::Status RemoveInstanceGroup(grpc::ServerContext *context,
                                     const proto::admin::RemoveInstanceGroupRequest *request,
                                     proto::admin::CommonResponse *response) override;
    grpc::Status GetInstanceGroup(grpc::ServerContext *context,
                                  const proto::admin::GetInstanceGroupRequest *request,
                                  proto::admin::GetInstanceGroupResponse *response) override;
    grpc::Status GetCacheMeta(grpc::ServerContext *context,
                              const proto::admin::GetCacheMetaRequest *request,
                              proto::admin::GetCacheMetaResponse *response) override;
    grpc::Status RemoveCache(grpc::ServerContext *context,
                             const proto::admin::RemoveCacheRequest *request,
                             proto::admin::CommonResponse *response) override;
    grpc::Status RegisterInstance(grpc::ServerContext *context,
                                  const proto::admin::RegisterInstanceRequest *request,
                                  proto::admin::CommonResponse *response) override;
    grpc::Status RemoveInstance(grpc::ServerContext *context,
                                const proto::admin::RemoveInstanceRequest *request,
                                proto::admin::CommonResponse *response) override;
    grpc::Status GetInstanceInfo(grpc::ServerContext *context,
                                 const proto::admin::GetInstanceInfoRequest *request,
                                 proto::admin::GetInstanceInfoResponse *response) override;
    grpc::Status ListInstanceInfo(grpc::ServerContext *context,
                                  const proto::admin::ListInstanceInfoRequest *request,
                                  proto::admin::ListInstanceInfoResponse *response) override;
    grpc::Status AddAccount(grpc::ServerContext *context,
                            const proto::admin::AddAccountRequest *request,
                            proto::admin::CommonResponse *response) override;
    grpc::Status DeleteAccount(grpc::ServerContext *context,
                               const proto::admin::DeleteAccountRequest *request,
                               proto::admin::CommonResponse *response) override;
    grpc::Status ListAccount(grpc::ServerContext *context,
                             const proto::admin::ListAccountRequest *request,
                             proto::admin::ListAccountResponse *response) override;
    grpc::Status GenConfigSnapshot(grpc::ServerContext *context,
                                   const proto::admin::GenConfigSnapshotRequest *request,
                                   proto::admin::ConfigSnapShotResponse *response) override;
    grpc::Status LoadConfigSnapshot(grpc::ServerContext *context,
                                    const proto::admin::LoadConfigSnapshotRequest *request,
                                    proto::admin::CommonResponse *response) override;
    grpc::Status GetMetrics(grpc::ServerContext *context,
                            const proto::admin::GetMetricsRequest *request,
                            proto::admin::GetMetricsResponse *response) override;

    // HA APIs
    grpc::Status CheckHealth(grpc::ServerContext *context,
                             const proto::admin::CheckHealthRequest *request,
                             proto::admin::CheckHealthResponse *response) override;
    grpc::Status GetManagerClusterInfo(grpc::ServerContext *context,
                                       const proto::admin::GetManagerClusterInfoRequest *request,
                                       proto::admin::GetManagerClusterInfoResponse *response) override;
    grpc::Status LeaderDemote(grpc::ServerContext *context,
                              const proto::admin::LeaderDemoteRequest *request,
                              proto::admin::CommonResponse *response) override;
    grpc::Status GetLeaderElectorConfig(grpc::ServerContext *context,
                                        const proto::admin::GetLeaderElectorConfigRequest *request,
                                        proto::admin::GetLeaderElectorConfigResponse *response) override;
    grpc::Status UpdateLeaderElectorConfig(grpc::ServerContext *context,
                                           const proto::admin::UpdateLeaderElectorConfigRequest *request,
                                           proto::admin::CommonResponse *response) override;
    grpc::Status UpdateLogger(grpc::ServerContext *context,
                              const proto::admin::UpdateLoggerRequest *request,
                              proto::admin::CommonResponse *response) override;

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
