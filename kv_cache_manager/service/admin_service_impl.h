#pragma once

#include <memory>

#include "kv_cache_manager/protocol/protobuf/admin_service.pb.h"
#include "kv_cache_manager/service/service_impl_base.h"

namespace kv_cache_manager {
class LeaderElector;

class CacheManager;
class MetricsReporter;
class MetricsRegistry;
class RegistryManager;
class RequestContext;

class AdminServiceImpl : public ServiceImplBase {
public:
    AdminServiceImpl(std::shared_ptr<CacheManager> cache_manager,
                     std::shared_ptr<MetricsReporter> metrics_reporter,
                     std::shared_ptr<MetricsRegistry> metrics_registry,
                     std::shared_ptr<RegistryManager> registry_manager,
                     std::shared_ptr<LeaderElector> leader_elector);
    ~AdminServiceImpl() override = default;

    // 实现所有ConfigService的接口方法
    void AddStorage(RequestContext *request_context,
                    const proto::admin::AddStorageRequest *request,
                    proto::admin::CommonResponse *response);
    void EnableStorage(RequestContext *request_context,
                       const proto::admin::EnableStorageRequest *request,
                       proto::admin::CommonResponse *response);
    void DisableStorage(RequestContext *request_context,
                        const proto::admin::DisableStorageRequest *request,
                        proto::admin::CommonResponse *response);
    void RemoveStorage(RequestContext *request_context,
                       const proto::admin::RemoveStorageRequest *request,
                       proto::admin::CommonResponse *response);
    void UpdateStorage(RequestContext *request_context,
                       const proto::admin::UpdateStorageRequest *request,
                       proto::admin::CommonResponse *response);
    void ListStorage(RequestContext *request_context,
                     const proto::admin::ListStorageRequest *request,
                     proto::admin::ListStorageResponse *response);

    void CreateInstanceGroup(RequestContext *request_context,
                             const proto::admin::CreateInstanceGroupRequest *request,
                             proto::admin::CommonResponse *response);
    void UpdateInstanceGroup(RequestContext *request_context,
                             const proto::admin::UpdateInstanceGroupRequest *request,
                             proto::admin::CommonResponse *response);
    void RemoveInstanceGroup(RequestContext *request_context,
                             const proto::admin::RemoveInstanceGroupRequest *request,
                             proto::admin::CommonResponse *response);
    void GetInstanceGroup(RequestContext *request_context,
                          const proto::admin::GetInstanceGroupRequest *request,
                          proto::admin::GetInstanceGroupResponse *response);

    void GetCacheMeta(RequestContext *request_context,
                      const proto::admin::GetCacheMetaRequest *request,
                      proto::admin::GetCacheMetaResponse *response);
    void RemoveCache(RequestContext *request_context,
                     const proto::admin::RemoveCacheRequest *request,
                     proto::admin::CommonResponse *response);

    void RegisterInstance(RequestContext *request_context,
                          const proto::admin::RegisterInstanceRequest *request,
                          proto::admin::CommonResponse *response);
    void RemoveInstance(RequestContext *request_context,
                        const proto::admin::RemoveInstanceRequest *request,
                        proto::admin::CommonResponse *response);
    void GetInstanceInfo(RequestContext *request_context,
                         const proto::admin::GetInstanceInfoRequest *request,
                         proto::admin::GetInstanceInfoResponse *response);
    void ListInstanceInfo(RequestContext *request_context,
                          const proto::admin::ListInstanceInfoRequest *request,
                          proto::admin::ListInstanceInfoResponse *response);

    void AddAccount(RequestContext *request_context,
                    const proto::admin::AddAccountRequest *request,
                    proto::admin::CommonResponse *response);
    void DeleteAccount(RequestContext *request_context,
                       const proto::admin::DeleteAccountRequest *request,
                       proto::admin::CommonResponse *response);
    void ListAccount(RequestContext *request_context,
                     const proto::admin::ListAccountRequest *request,
                     proto::admin::ListAccountResponse *response);

    void GenConfigSnapshot(RequestContext *request_context,
                           const proto::admin::GenConfigSnapshotRequest *request,
                           proto::admin::ConfigSnapShotResponse *response);
    void LoadConfigSnapshot(RequestContext *request_context,
                            const proto::admin::LoadConfigSnapshotRequest *request,
                            proto::admin::CommonResponse *response);

    void GetMetrics(RequestContext *request_context,
                    const proto::admin::GetMetricsRequest *request,
                    proto::admin::GetMetricsResponse *response);

    // 高可用相关接口
    void CheckHealth(RequestContext *request_context,
                     const proto::admin::CheckHealthRequest *request,
                     proto::admin::CheckHealthResponse *response);
    void GetManagerClusterInfo(RequestContext *request_context,
                               const proto::admin::GetManagerClusterInfoRequest *request,
                               proto::admin::GetManagerClusterInfoResponse *response);
    void LeaderDemote(RequestContext *request_context,
                      const proto::admin::LeaderDemoteRequest *request,
                      proto::admin::CommonResponse *response);
    void GetLeaderElectorConfig(RequestContext *request_context,
                                const proto::admin::GetLeaderElectorConfigRequest *request,
                                proto::admin::GetLeaderElectorConfigResponse *response);
    void UpdateLeaderElectorConfig(RequestContext *request_context,
                                   const proto::admin::UpdateLeaderElectorConfigRequest *request,
                                   proto::admin::CommonResponse *response);

private:
    std::shared_ptr<CacheManager> cache_manager_;
    std::shared_ptr<MetricsReporter> metrics_reporter_;
    std::shared_ptr<MetricsRegistry> metrics_registry_; // for the GetMetrics API
    std::shared_ptr<RegistryManager> registry_manager_;
    std::shared_ptr<LeaderElector> leader_elector_;
};

} // namespace kv_cache_manager
