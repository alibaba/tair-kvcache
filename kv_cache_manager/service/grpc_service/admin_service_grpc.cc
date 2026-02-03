#include "kv_cache_manager/service/grpc_service/admin_service_grpc.h"

#include <memory>
#include <string>
#include <utility>

#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/metrics/metrics_registry.h"
#include "kv_cache_manager/protocol/protobuf/admin_service.grpc.pb.h"
#include "kv_cache_manager/service/admin_service_impl.h"
#include "kv_cache_manager/service/util/common.h"

namespace kv_cache_manager {

AdminServiceGRpc::AdminServiceGRpc(std::shared_ptr<MetricsRegistry> metrics_registry,
                                   std::shared_ptr<AdminServiceImpl> admin_service_impl)
    : metrics_registry_(std::move(metrics_registry)), admin_service_impl_(std::move(admin_service_impl)) {}

void AdminServiceGRpc::Init() {
    // for storage APIs
    MAKE_SERVICE_METRICS_COLLECTOR(AddStorage);
    MAKE_SERVICE_METRICS_COLLECTOR(EnableStorage);
    MAKE_SERVICE_METRICS_COLLECTOR(DisableStorage);
    MAKE_SERVICE_METRICS_COLLECTOR(RemoveStorage);
    MAKE_SERVICE_METRICS_COLLECTOR(UpdateStorage);
    MAKE_SERVICE_METRICS_COLLECTOR(ListStorage);

    // for instance group APIs
    MAKE_SERVICE_METRICS_COLLECTOR(CreateInstanceGroup);
    MAKE_SERVICE_METRICS_COLLECTOR(UpdateInstanceGroup);
    MAKE_SERVICE_METRICS_COLLECTOR(RemoveInstanceGroup);
    MAKE_SERVICE_METRICS_COLLECTOR(GetInstanceGroup);

    // for cache APIs
    MAKE_SERVICE_METRICS_COLLECTOR(GetCacheMeta);
    MAKE_SERVICE_METRICS_COLLECTOR(RemoveCache);

    // for instance APIs
    MAKE_SERVICE_METRICS_COLLECTOR(RegisterInstance);
    MAKE_SERVICE_METRICS_COLLECTOR(RemoveInstance);
    MAKE_SERVICE_METRICS_COLLECTOR(GetInstanceInfo);
    MAKE_SERVICE_METRICS_COLLECTOR(ListInstanceInfo);

    // for account APIs
    MAKE_SERVICE_METRICS_COLLECTOR(AddAccount);
    MAKE_SERVICE_METRICS_COLLECTOR(DeleteAccount);
    MAKE_SERVICE_METRICS_COLLECTOR(ListAccount);

    // for config snapshot APIs
    MAKE_SERVICE_METRICS_COLLECTOR(GenConfigSnapshot);
    MAKE_SERVICE_METRICS_COLLECTOR(LoadConfigSnapshot);

    // for metrics APIs
    MAKE_SERVICE_METRICS_COLLECTOR(GetMetrics);

    // for high availability APIs
    MAKE_SERVICE_METRICS_COLLECTOR(CheckHealth);
    MAKE_SERVICE_METRICS_COLLECTOR(GetManagerClusterInfo);
    MAKE_SERVICE_METRICS_COLLECTOR(LeaderDemote);
    MAKE_SERVICE_METRICS_COLLECTOR(GetLeaderElectorConfig);
    MAKE_SERVICE_METRICS_COLLECTOR(UpdateLeaderElectorConfig);

    // for logging control APIs
    MAKE_SERVICE_METRICS_COLLECTOR(UpdateLogger);
}

grpc::Status AdminServiceGRpc::AddStorage(grpc::ServerContext *context,
                                          const proto::admin::AddStorageRequest *request,
                                          proto::admin::CommonResponse *response) {
    API_CONTEXT_INIT_GRPC(AddStorage)
    admin_service_impl_->AddStorage(request_context, request, response);
    return grpc::Status::OK;
}

grpc::Status AdminServiceGRpc::EnableStorage(grpc::ServerContext *context,
                                             const proto::admin::EnableStorageRequest *request,
                                             proto::admin::CommonResponse *response) {
    API_CONTEXT_INIT_GRPC(EnableStorage)
    admin_service_impl_->EnableStorage(request_context, request, response);
    return grpc::Status::OK;
}

grpc::Status AdminServiceGRpc::DisableStorage(grpc::ServerContext *context,
                                              const proto::admin::DisableStorageRequest *request,
                                              proto::admin::CommonResponse *response) {
    API_CONTEXT_INIT_GRPC(DisableStorage)
    admin_service_impl_->DisableStorage(request_context, request, response);
    return grpc::Status::OK;
}

grpc::Status AdminServiceGRpc::RemoveStorage(grpc::ServerContext *context,
                                             const proto::admin::RemoveStorageRequest *request,
                                             proto::admin::CommonResponse *response) {
    API_CONTEXT_INIT_GRPC(RemoveStorage)
    admin_service_impl_->RemoveStorage(request_context, request, response);
    return grpc::Status::OK;
}

grpc::Status AdminServiceGRpc::UpdateStorage(grpc::ServerContext *context,
                                             const proto::admin::UpdateStorageRequest *request,
                                             proto::admin::CommonResponse *response) {
    API_CONTEXT_INIT_GRPC(UpdateStorage)
    admin_service_impl_->UpdateStorage(request_context, request, response);
    return grpc::Status::OK;
}

grpc::Status AdminServiceGRpc::ListStorage(grpc::ServerContext *context,
                                           const proto::admin::ListStorageRequest *request,
                                           proto::admin::ListStorageResponse *response) {
    API_CONTEXT_INIT_GRPC(ListStorage)
    admin_service_impl_->ListStorage(request_context, request, response);
    return grpc::Status::OK;
}

grpc::Status AdminServiceGRpc::CreateInstanceGroup(grpc::ServerContext *context,
                                                   const proto::admin::CreateInstanceGroupRequest *request,
                                                   proto::admin::CommonResponse *response) {
    API_CONTEXT_INIT_GRPC(CreateInstanceGroup)
    admin_service_impl_->CreateInstanceGroup(request_context, request, response);
    return grpc::Status::OK;
}

grpc::Status AdminServiceGRpc::UpdateInstanceGroup(grpc::ServerContext *context,
                                                   const proto::admin::UpdateInstanceGroupRequest *request,
                                                   proto::admin::CommonResponse *response) {
    API_CONTEXT_INIT_GRPC(UpdateInstanceGroup)
    admin_service_impl_->UpdateInstanceGroup(request_context, request, response);
    return grpc::Status::OK;
}

grpc::Status AdminServiceGRpc::RemoveInstanceGroup(grpc::ServerContext *context,
                                                   const proto::admin::RemoveInstanceGroupRequest *request,
                                                   proto::admin::CommonResponse *response) {
    API_CONTEXT_INIT_GRPC(RemoveInstanceGroup)
    admin_service_impl_->RemoveInstanceGroup(request_context, request, response);
    return grpc::Status::OK;
}

grpc::Status AdminServiceGRpc::GetInstanceGroup(grpc::ServerContext *context,
                                                const proto::admin::GetInstanceGroupRequest *request,
                                                proto::admin::GetInstanceGroupResponse *response) {
    API_CONTEXT_INIT_GRPC(GetInstanceGroup)
    admin_service_impl_->GetInstanceGroup(request_context, request, response);
    return grpc::Status::OK;
}

grpc::Status AdminServiceGRpc::GetCacheMeta(grpc::ServerContext *context,
                                            const proto::admin::GetCacheMetaRequest *request,
                                            proto::admin::GetCacheMetaResponse *response) {
    API_CONTEXT_INIT_GRPC(GetCacheMeta)
    admin_service_impl_->GetCacheMeta(request_context, request, response);
    return grpc::Status::OK;
}

grpc::Status AdminServiceGRpc::RemoveCache(grpc::ServerContext *context,
                                           const proto::admin::RemoveCacheRequest *request,
                                           proto::admin::CommonResponse *response) {
    API_CONTEXT_INIT_GRPC(RemoveCache)
    admin_service_impl_->RemoveCache(request_context, request, response);
    return grpc::Status::OK;
}

grpc::Status AdminServiceGRpc::RegisterInstance(grpc::ServerContext *context,
                                                const proto::admin::RegisterInstanceRequest *request,
                                                proto::admin::CommonResponse *response) {
    API_CONTEXT_INIT_GRPC(RegisterInstance)
    admin_service_impl_->RegisterInstance(request_context, request, response);
    return grpc::Status::OK;
}

grpc::Status AdminServiceGRpc::RemoveInstance(grpc::ServerContext *context,
                                              const proto::admin::RemoveInstanceRequest *request,
                                              proto::admin::CommonResponse *response) {
    API_CONTEXT_INIT_GRPC(RemoveInstance)
    admin_service_impl_->RemoveInstance(request_context, request, response);
    return grpc::Status::OK;
}

grpc::Status AdminServiceGRpc::GetInstanceInfo(grpc::ServerContext *context,
                                               const proto::admin::GetInstanceInfoRequest *request,
                                               proto::admin::GetInstanceInfoResponse *response) {
    API_CONTEXT_INIT_GRPC(GetInstanceInfo)
    admin_service_impl_->GetInstanceInfo(request_context, request, response);
    return grpc::Status::OK;
}

grpc::Status AdminServiceGRpc::ListInstanceInfo(grpc::ServerContext *context,
                                                const proto::admin::ListInstanceInfoRequest *request,
                                                proto::admin::ListInstanceInfoResponse *response) {
    API_CONTEXT_INIT_GRPC(ListInstanceInfo)
    admin_service_impl_->ListInstanceInfo(request_context, request, response);
    return grpc::Status::OK;
}

grpc::Status AdminServiceGRpc::AddAccount(grpc::ServerContext *context,
                                          const proto::admin::AddAccountRequest *request,
                                          proto::admin::CommonResponse *response) {
    API_CONTEXT_INIT_GRPC(AddAccount)
    admin_service_impl_->AddAccount(request_context, request, response);
    return grpc::Status::OK;
}

grpc::Status AdminServiceGRpc::DeleteAccount(grpc::ServerContext *context,
                                             const proto::admin::DeleteAccountRequest *request,
                                             proto::admin::CommonResponse *response) {
    API_CONTEXT_INIT_GRPC(DeleteAccount)
    admin_service_impl_->DeleteAccount(request_context, request, response);
    return grpc::Status::OK;
}

grpc::Status AdminServiceGRpc::ListAccount(grpc::ServerContext *context,
                                           const proto::admin::ListAccountRequest *request,
                                           proto::admin::ListAccountResponse *response) {
    API_CONTEXT_INIT_GRPC(ListAccount)
    admin_service_impl_->ListAccount(request_context, request, response);
    return grpc::Status::OK;
}

grpc::Status AdminServiceGRpc::GenConfigSnapshot(grpc::ServerContext *context,
                                                 const proto::admin::GenConfigSnapshotRequest *request,
                                                 proto::admin::ConfigSnapShotResponse *response) {
    API_CONTEXT_INIT_GRPC(GenConfigSnapshot)
    admin_service_impl_->GenConfigSnapshot(request_context, request, response);
    return grpc::Status::OK;
}

grpc::Status AdminServiceGRpc::LoadConfigSnapshot(grpc::ServerContext *context,
                                                  const proto::admin::LoadConfigSnapshotRequest *request,
                                                  proto::admin::CommonResponse *response) {
    API_CONTEXT_INIT_GRPC(LoadConfigSnapshot)
    admin_service_impl_->LoadConfigSnapshot(request_context, request, response);
    return grpc::Status::OK;
}

grpc::Status AdminServiceGRpc::GetMetrics(grpc::ServerContext *context,
                                          const proto::admin::GetMetricsRequest *request,
                                          proto::admin::GetMetricsResponse *response) {
    API_CONTEXT_INIT_GRPC(GetMetrics)
    admin_service_impl_->GetMetrics(request_context, request, response);
    return grpc::Status::OK;
}

grpc::Status AdminServiceGRpc::CheckHealth(grpc::ServerContext *context,
                                           const proto::admin::CheckHealthRequest *request,
                                           proto::admin::CheckHealthResponse *response) {
    API_CONTEXT_INIT_GRPC(CheckHealth)
    admin_service_impl_->CheckHealth(request_context, request, response);
    return grpc::Status::OK;
}

grpc::Status AdminServiceGRpc::GetManagerClusterInfo(grpc::ServerContext *context,
                                                     const proto::admin::GetManagerClusterInfoRequest *request,
                                                     proto::admin::GetManagerClusterInfoResponse *response) {
    API_CONTEXT_INIT_GRPC(GetManagerClusterInfo)
    admin_service_impl_->GetManagerClusterInfo(request_context, request, response);
    return grpc::Status::OK;
}

grpc::Status AdminServiceGRpc::LeaderDemote(grpc::ServerContext *context,
                                            const proto::admin::LeaderDemoteRequest *request,
                                            proto::admin::CommonResponse *response) {
    API_CONTEXT_INIT_GRPC(LeaderDemote)
    admin_service_impl_->LeaderDemote(request_context, request, response);
    return grpc::Status::OK;
}

grpc::Status AdminServiceGRpc::GetLeaderElectorConfig(grpc::ServerContext *context,
                                                      const proto::admin::GetLeaderElectorConfigRequest *request,
                                                      proto::admin::GetLeaderElectorConfigResponse *response) {
    API_CONTEXT_INIT_GRPC(GetLeaderElectorConfig)
    admin_service_impl_->GetLeaderElectorConfig(request_context, request, response);
    return grpc::Status::OK;
}

grpc::Status AdminServiceGRpc::UpdateLeaderElectorConfig(grpc::ServerContext *context,
                                                         const proto::admin::UpdateLeaderElectorConfigRequest *request,
                                                         proto::admin::CommonResponse *response) {
    API_CONTEXT_INIT_GRPC(UpdateLeaderElectorConfig)
    admin_service_impl_->UpdateLeaderElectorConfig(request_context, request, response);
    return grpc::Status::OK;
}

grpc::Status AdminServiceGRpc::UpdateLogger(grpc::ServerContext *context,
                                            const proto::admin::UpdateLoggerRequest *request,
                                            proto::admin::CommonResponse *response) {
    API_CONTEXT_INIT_GRPC(UpdateLogger)
    admin_service_impl_->UpdateLogger(request_context, request, response);
    return grpc::Status::OK;
}

} // namespace kv_cache_manager
