#include "kv_cache_manager/service/http_service/admin_service_http.h"

#include <memory>
#include <string>
#include <utility>

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/metrics/metrics_registry.h"
#include "kv_cache_manager/protocol/protobuf/admin_service.pb.h"
#include "kv_cache_manager/service/admin_service_impl.h"
#include "kv_cache_manager/service/util/common.h"

namespace kv_cache_manager {

AdminServiceHttp::AdminServiceHttp(std::shared_ptr<MetricsRegistry> metrics_registry,
                                   std::shared_ptr<AdminServiceImpl> admin_service_impl)
    : metrics_registry_(std::move(metrics_registry)), admin_service_impl_(std::move(admin_service_impl)) {}

void AdminServiceHttp::Init() {
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
}

void AdminServiceHttp::RegisterHandler() {
    // Storage APIs
    REGISTER_HTTP_HANDLER_FOR_ADMIN_SERVICE(Post, addStorage, AddStorage, Common, AddStorage);
    REGISTER_HTTP_HANDLER_FOR_ADMIN_SERVICE(Post, enableStorage, EnableStorage, Common, EnableStorage);
    REGISTER_HTTP_HANDLER_FOR_ADMIN_SERVICE(Post, disableStorage, DisableStorage, Common, DisableStorage);
    REGISTER_HTTP_HANDLER_FOR_ADMIN_SERVICE(Post, removeStorage, RemoveStorage, Common, RemoveStorage);
    REGISTER_HTTP_HANDLER_FOR_ADMIN_SERVICE(Post, updateStorage, UpdateStorage, Common, UpdateStorage);
    REGISTER_HTTP_HANDLER_FOR_ADMIN_SERVICE(Post, listStorage, ListStorage, ListStorage, ListStorage);

    // Instance Group APIs
    REGISTER_HTTP_HANDLER_FOR_ADMIN_SERVICE(
        Post, createInstanceGroup, CreateInstanceGroup, Common, CreateInstanceGroup);
    REGISTER_HTTP_HANDLER_FOR_ADMIN_SERVICE(
        Post, updateInstanceGroup, UpdateInstanceGroup, Common, UpdateInstanceGroup);
    REGISTER_HTTP_HANDLER_FOR_ADMIN_SERVICE(
        Post, removeInstanceGroup, RemoveInstanceGroup, Common, RemoveInstanceGroup);
    REGISTER_HTTP_HANDLER_FOR_ADMIN_SERVICE(
        Post, getInstanceGroup, GetInstanceGroup, GetInstanceGroup, GetInstanceGroup);

    // Cache APIs
    REGISTER_HTTP_HANDLER_FOR_ADMIN_SERVICE(Post, getCacheMeta, GetCacheMeta, GetCacheMeta, GetCacheMeta);
    REGISTER_HTTP_HANDLER_FOR_ADMIN_SERVICE(Post, removeCache, RemoveCache, Common, RemoveCache);

    REGISTER_HTTP_HANDLER_FOR_ADMIN_SERVICE(Post, registerInstance, RegisterInstance, Common, RegisterInstance);
    REGISTER_HTTP_HANDLER_FOR_ADMIN_SERVICE(Post, removeInstance, RemoveInstance, Common, RemoveInstance);
    REGISTER_HTTP_HANDLER_FOR_ADMIN_SERVICE(Post, getInstanceInfo, GetInstanceInfo, GetInstanceInfo, GetInstanceInfo);
    REGISTER_HTTP_HANDLER_FOR_ADMIN_SERVICE(
        Post, listInstanceInfo, ListInstanceInfo, ListInstanceInfo, ListInstanceInfo);

    // Account APIs
    REGISTER_HTTP_HANDLER_FOR_ADMIN_SERVICE(Post, addAccount, AddAccount, Common, AddAccount);
    REGISTER_HTTP_HANDLER_FOR_ADMIN_SERVICE(Post, deleteAccount, DeleteAccount, Common, DeleteAccount);
    REGISTER_HTTP_HANDLER_FOR_ADMIN_SERVICE(Post, listAccount, ListAccount, ListAccount, ListAccount);

    // Config Snapshot APIs
    REGISTER_HTTP_HANDLER_FOR_ADMIN_SERVICE(
        Post, genConfigSnapshot, GenConfigSnapshot, ConfigSnapShot, GenConfigSnapshot);
    REGISTER_HTTP_HANDLER_FOR_ADMIN_SERVICE(Post, loadConfigSnapshot, LoadConfigSnapshot, Common, LoadConfigSnapshot);

    // Metrics API
    REGISTER_HTTP_HANDLER_FOR_ADMIN_SERVICE(Post, getMetrics, GetMetrics, GetMetrics, GetMetrics);

    // High Availability APIs
    REGISTER_HTTP_HANDLER_FOR_ADMIN_SERVICE(Post, checkHealth, CheckHealth, CheckHealth, CheckHealth);
    REGISTER_HTTP_HANDLER_FOR_ADMIN_SERVICE(
        Post, getManagerClusterInfo, GetManagerClusterInfo, GetManagerClusterInfo, GetManagerClusterInfo);
    REGISTER_HTTP_HANDLER_FOR_ADMIN_SERVICE(Post, leaderDemote, LeaderDemote, Common, LeaderDemote);
    REGISTER_HTTP_HANDLER_FOR_ADMIN_SERVICE(
        Post, getLeaderElectorConfig, GetLeaderElectorConfig, GetLeaderElectorConfig, GetLeaderElectorConfig);
    REGISTER_HTTP_HANDLER_FOR_ADMIN_SERVICE(
        Post, updateLeaderElectorConfig, UpdateLeaderElectorConfig, Common, UpdateLeaderElectorConfig);
}

void AdminServiceHttp::AddStorage(coro_http::coro_http_connection *http_conn,
                                  proto::admin::AddStorageRequest *request,
                                  proto::admin::CommonResponse *response) {
    API_CONTEXT_INIT_HTTP(AddStorage)
    KVCM_LOG_INFO("[traceId: %s] AddStorage [%s] called.",
                  request->trace_id().c_str(),
                  request->storage().global_unique_name().c_str());
    admin_service_impl_->AddStorage(request_context, request, response);
}

void AdminServiceHttp::EnableStorage(coro_http::coro_http_connection *http_conn,
                                     proto::admin::EnableStorageRequest *request,
                                     proto::admin::CommonResponse *response) {
    API_CONTEXT_INIT_HTTP(EnableStorage)
    KVCM_LOG_INFO("[traceId: %s] EnableStorage [%s] called.",
                  request->trace_id().c_str(),
                  request->storage_unique_name().c_str());
    admin_service_impl_->EnableStorage(request_context, request, response);
}

void AdminServiceHttp::DisableStorage(coro_http::coro_http_connection *http_conn,
                                      proto::admin::DisableStorageRequest *request,
                                      proto::admin::CommonResponse *response) {
    API_CONTEXT_INIT_HTTP(DisableStorage)
    KVCM_LOG_INFO("[traceId: %s] DisableStorage [%s] called.",
                  request->trace_id().c_str(),
                  request->storage_unique_name().c_str());
    admin_service_impl_->DisableStorage(request_context, request, response);
}

void AdminServiceHttp::RemoveStorage(coro_http::coro_http_connection *http_conn,
                                     proto::admin::RemoveStorageRequest *request,
                                     proto::admin::CommonResponse *response) {
    API_CONTEXT_INIT_HTTP(RemoveStorage)
    KVCM_LOG_INFO("[traceId: %s] RemoveStorage [%s] called.",
                  request->trace_id().c_str(),
                  request->storage_unique_name().c_str());
    admin_service_impl_->RemoveStorage(request_context, request, response);
}

void AdminServiceHttp::UpdateStorage(coro_http::coro_http_connection *http_conn,
                                     proto::admin::UpdateStorageRequest *request,
                                     proto::admin::CommonResponse *response) {
    API_CONTEXT_INIT_HTTP(UpdateStorage)
    KVCM_LOG_INFO("[traceId: %s] UpdateStorage [%s] called.",
                  request->trace_id().c_str(),
                  request->storage().global_unique_name().c_str());
    admin_service_impl_->UpdateStorage(request_context, request, response);
}

void AdminServiceHttp::ListStorage(coro_http::coro_http_connection *http_conn,
                                   proto::admin::ListStorageRequest *request,
                                   proto::admin::ListStorageResponse *response) {
    API_CONTEXT_INIT_HTTP(ListStorage)
    KVCM_LOG_INFO("[traceId: %s] ListStorage called.", request->trace_id().c_str());
    admin_service_impl_->ListStorage(request_context, request, response);
}

void AdminServiceHttp::CreateInstanceGroup(coro_http::coro_http_connection *http_conn,
                                           proto::admin::CreateInstanceGroupRequest *request,
                                           proto::admin::CommonResponse *response) {
    API_CONTEXT_INIT_HTTP(CreateInstanceGroup)
    KVCM_LOG_INFO("[traceId: %s] CreateInstanceGroup [%s] called.",
                  request->trace_id().c_str(),
                  request->instance_group().name().c_str());
    admin_service_impl_->CreateInstanceGroup(request_context, request, response);
}

void AdminServiceHttp::UpdateInstanceGroup(coro_http::coro_http_connection *http_conn,
                                           proto::admin::UpdateInstanceGroupRequest *request,
                                           proto::admin::CommonResponse *response) {
    API_CONTEXT_INIT_HTTP(UpdateInstanceGroup)
    KVCM_LOG_INFO("[traceId: %s] UpdateInstanceGroup [%s] called.",
                  request->trace_id().c_str(),
                  request->instance_group().name().c_str());
    admin_service_impl_->UpdateInstanceGroup(request_context, request, response);
}

void AdminServiceHttp::RemoveInstanceGroup(coro_http::coro_http_connection *http_conn,
                                           proto::admin::RemoveInstanceGroupRequest *request,
                                           proto::admin::CommonResponse *response) {
    API_CONTEXT_INIT_HTTP(RemoveInstanceGroup)
    KVCM_LOG_INFO(
        "[traceId: %s] RemoveInstanceGroup [%s] called.", request->trace_id().c_str(), request->name().c_str());
    admin_service_impl_->RemoveInstanceGroup(request_context, request, response);
}

void AdminServiceHttp::GetInstanceGroup(coro_http::coro_http_connection *http_conn,
                                        proto::admin::GetInstanceGroupRequest *request,
                                        proto::admin::GetInstanceGroupResponse *response) {
    API_CONTEXT_INIT_HTTP(GetInstanceGroup)
    KVCM_LOG_INFO("[traceId: %s] GetInstanceGroup [%s] called.", request->trace_id().c_str(), request->name().c_str());
    admin_service_impl_->GetInstanceGroup(request_context, request, response);
}

void AdminServiceHttp::GetCacheMeta(coro_http::coro_http_connection *http_conn,
                                    proto::admin::GetCacheMetaRequest *request,
                                    proto::admin::GetCacheMetaResponse *response) {
    API_CONTEXT_INIT_HTTP(GetCacheMeta)
    KVCM_LOG_INFO("[traceId: %s] GetCacheMeta for instance [%s] called.",
                  request->trace_id().c_str(),
                  request->instance_id().c_str());
    admin_service_impl_->GetCacheMeta(request_context, request, response);
}

void AdminServiceHttp::RemoveCache(coro_http::coro_http_connection *http_conn,
                                   proto::admin::RemoveCacheRequest *request,
                                   proto::admin::CommonResponse *response) {
    API_CONTEXT_INIT_HTTP(RemoveCache)
    KVCM_LOG_INFO("[traceId: %s] RemoveCache for instance [%s] called.",
                  request->trace_id().c_str(),
                  request->instance_id().c_str());

    admin_service_impl_->RemoveCache(request_context, request, response);
}

void AdminServiceHttp::RegisterInstance(coro_http::coro_http_connection *http_conn,
                                        proto::admin::RegisterInstanceRequest *request,
                                        proto::admin::CommonResponse *response) {
    API_CONTEXT_INIT_HTTP(RegisterInstance)
    KVCM_LOG_INFO("[traceId: %s] RegisterInstance for instance [%s] called.",
                  request->trace_id().c_str(),
                  request->instance_id().c_str());
    admin_service_impl_->RegisterInstance(request_context, request, response);
}

void AdminServiceHttp::RemoveInstance(coro_http::coro_http_connection *http_conn,
                                      proto::admin::RemoveInstanceRequest *request,
                                      proto::admin::CommonResponse *response) {
    API_CONTEXT_INIT_HTTP(RemoveInstance)
    KVCM_LOG_INFO("[traceId: %s] RemoveInstance for instance [%s] called.",
                  request->trace_id().c_str(),
                  request->instance_id().c_str());
    admin_service_impl_->RemoveInstance(request_context, request, response);
}

void AdminServiceHttp::GetInstanceInfo(coro_http::coro_http_connection *http_conn,
                                       proto::admin::GetInstanceInfoRequest *request,
                                       proto::admin::GetInstanceInfoResponse *response) {
    API_CONTEXT_INIT_HTTP(GetInstanceInfo)
    KVCM_LOG_INFO("[traceId: %s] GetInstanceInfo for instance [%s] called.",
                  request->trace_id().c_str(),
                  request->instance_id().c_str());
    admin_service_impl_->GetInstanceInfo(request_context, request, response);
}

void AdminServiceHttp::ListInstanceInfo(coro_http::coro_http_connection *http_conn,
                                        proto::admin::ListInstanceInfoRequest *request,
                                        proto::admin::ListInstanceInfoResponse *response) {
    API_CONTEXT_INIT_HTTP(ListInstanceInfo)
    KVCM_LOG_INFO("[traceId: %s] ListInstanceInfo for instance_group [%s] called.",
                  request->trace_id().c_str(),
                  request->instance_group_name().c_str());

    admin_service_impl_->ListInstanceInfo(request_context, request, response);
}

void AdminServiceHttp::AddAccount(coro_http::coro_http_connection *http_conn,
                                  proto::admin::AddAccountRequest *request,
                                  proto::admin::CommonResponse *response) {
    API_CONTEXT_INIT_HTTP(AddAccount)
    KVCM_LOG_INFO("[traceId: %s] AddAccount for user_name [%s] called.",
                  request->trace_id().c_str(),
                  request->user_name().c_str());
    admin_service_impl_->AddAccount(request_context, request, response);
}

void AdminServiceHttp::DeleteAccount(coro_http::coro_http_connection *http_conn,
                                     proto::admin::DeleteAccountRequest *request,
                                     proto::admin::CommonResponse *response) {
    API_CONTEXT_INIT_HTTP(DeleteAccount)
    KVCM_LOG_INFO("[traceId: %s] DeleteAccount for user_name [%s] called.",
                  request->trace_id().c_str(),
                  request->user_name().c_str());

    admin_service_impl_->DeleteAccount(request_context, request, response);
}

void AdminServiceHttp::ListAccount(coro_http::coro_http_connection *http_conn,
                                   proto::admin::ListAccountRequest *request,
                                   proto::admin::ListAccountResponse *response) {
    API_CONTEXT_INIT_HTTP(ListAccount)
    KVCM_LOG_INFO("[traceId: %s] ListAccount called.", request->trace_id().c_str());

    admin_service_impl_->ListAccount(request_context, request, response);
}

void AdminServiceHttp::GenConfigSnapshot(coro_http::coro_http_connection *http_conn,
                                         proto::admin::GenConfigSnapshotRequest *request,
                                         proto::admin::ConfigSnapShotResponse *response) {
    API_CONTEXT_INIT_HTTP(GenConfigSnapshot)
    KVCM_LOG_INFO("[traceId: %s] GenConfigSnapshot called.", request->trace_id().c_str());

    admin_service_impl_->GenConfigSnapshot(request_context, request, response);
}

void AdminServiceHttp::LoadConfigSnapshot(coro_http::coro_http_connection *http_conn,
                                          proto::admin::LoadConfigSnapshotRequest *request,
                                          proto::admin::CommonResponse *response) {
    API_CONTEXT_INIT_HTTP(LoadConfigSnapshot)
    KVCM_LOG_INFO("[traceId: %s] LoadConfigSnapshot called.", request->trace_id().c_str());
    admin_service_impl_->LoadConfigSnapshot(request_context, request, response);
}

void AdminServiceHttp::GetMetrics(coro_http::coro_http_connection *http_conn,
                                  proto::admin::GetMetricsRequest *request,
                                  proto::admin::GetMetricsResponse *response) {
    API_CONTEXT_INIT_HTTP(GetMetrics)
    KVCM_LOG_INFO("[traceId: %s] GetMetrics called.", request->trace_id().c_str());
    admin_service_impl_->GetMetrics(request_context, request, response);
}

void AdminServiceHttp::CheckHealth(coro_http::coro_http_connection *http_conn,
                                   proto::admin::CheckHealthRequest *request,
                                   proto::admin::CheckHealthResponse *response) {
    API_CONTEXT_INIT_HTTP(CheckHealth)
    KVCM_LOG_INFO("[traceId: %s] CheckHealth called.", request->trace_id().c_str());
    admin_service_impl_->CheckHealth(request_context, request, response);
}

void AdminServiceHttp::GetManagerClusterInfo(coro_http::coro_http_connection *http_conn,
                                             proto::admin::GetManagerClusterInfoRequest *request,
                                             proto::admin::GetManagerClusterInfoResponse *response) {
    API_CONTEXT_INIT_HTTP(GetManagerClusterInfo)
    KVCM_LOG_INFO("[traceId: %s] GetManagerClusterInfo called.", request->trace_id().c_str());
    admin_service_impl_->GetManagerClusterInfo(request_context, request, response);
}

void AdminServiceHttp::LeaderDemote(coro_http::coro_http_connection *http_conn,
                                    proto::admin::LeaderDemoteRequest *request,
                                    proto::admin::CommonResponse *response) {
    API_CONTEXT_INIT_HTTP(LeaderDemote)
    KVCM_LOG_INFO("[traceId: %s] LeaderDemote called.", request->trace_id().c_str());
    admin_service_impl_->LeaderDemote(request_context, request, response);
}

void AdminServiceHttp::GetLeaderElectorConfig(coro_http::coro_http_connection *http_conn,
                                              proto::admin::GetLeaderElectorConfigRequest *request,
                                              proto::admin::GetLeaderElectorConfigResponse *response) {
    API_CONTEXT_INIT_HTTP(GetLeaderElectorConfig)
    KVCM_LOG_INFO("[traceId: %s] GetLeaderElectorConfig called.", request->trace_id().c_str());
    admin_service_impl_->GetLeaderElectorConfig(request_context, request, response);
}

void AdminServiceHttp::UpdateLeaderElectorConfig(coro_http::coro_http_connection *http_conn,
                                                 proto::admin::UpdateLeaderElectorConfigRequest *request,
                                                 proto::admin::CommonResponse *response) {
    API_CONTEXT_INIT_HTTP(UpdateLeaderElectorConfig)
    KVCM_LOG_INFO("[traceId: %s] UpdateLeaderElectorConfig called.", request->trace_id().c_str());
    admin_service_impl_->UpdateLeaderElectorConfig(request_context, request, response);
}

} // namespace kv_cache_manager
