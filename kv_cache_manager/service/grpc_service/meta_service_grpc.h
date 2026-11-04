#pragma once

#include <grpcpp/grpcpp.h>
#include <memory>

#include "grpc++/grpc++.h"
#include "kv_cache_manager/metrics/metrics_collector.h"
#include "kv_cache_manager/protocol/protobuf/meta_service.grpc.pb.h"
#include "kv_cache_manager/service/meta_service_metrics_base.h"

namespace kv_cache_manager {

class MetaServiceImpl;
class ServiceRequestExecutor;
class MetricsRegistry;
struct MetricsLifecycle;

// A generated AsyncService may be registered with one server only. Construct
// a new adapter when restarting the server; its business implementation can be shared.
class MetaServiceGRpc final : public proto::meta::MetaService::AsyncService, public MetaServiceMetricsBase {
public:
    MetaServiceGRpc(std::shared_ptr<MetricsRegistry> metrics_registry,
                    std::shared_ptr<MetaServiceImpl> meta_service_impl,
                    std::shared_ptr<RegistryManager> registry_manager,
                    std::shared_ptr<ServiceRequestExecutor> executor,
                    std::shared_ptr<MetricsLifecycle> metrics_lifecycle = nullptr);

    ~MetaServiceGRpc() override;

    void Init();
    // Prepare before BuildAndStart; post application requests only after it succeeds.
    bool PrepareCompletionQueue(grpc::ServerBuilder &builder);
    void StartAccepting();
    void StopAccepting();
    // The gRPC server must be shut down while the CQ poller and executor still run.
    void WaitForCalls();
    void ShutdownCompletionQueueAndJoin();

private:
    bool PrepareRegisterInstance(RequestContext *request_context,
                                 const proto::meta::RegisterInstanceRequest *request,
                                 proto::meta::RegisterInstanceResponse *response);
    bool PrepareGetInstanceInfo(RequestContext *request_context,
                                const proto::meta::GetInstanceInfoRequest *request,
                                proto::meta::GetInstanceInfoResponse *response);
    bool PrepareGetCacheMeta(RequestContext *request_context,
                             const proto::meta::GetCacheMetaRequest *request,
                             proto::meta::GetCacheMetaResponse *response);

    bool PrepareGetCacheLocation(RequestContext *request_context,
                                 const proto::meta::GetCacheLocationRequest *request,
                                 proto::meta::GetCacheLocationResponse *response);

    bool PrepareGetCacheLocationsByBackend(RequestContext *request_context,
                                           const proto::meta::GetCacheLocationsByBackendRequest *request,
                                           proto::meta::GetCacheLocationsByBackendResponse *response);

    bool PrepareGetCacheLocationLen(RequestContext *request_context,
                                    const proto::meta::GetCacheLocationLenRequest *request,
                                    proto::meta::GetCacheLocationLenResponse *response);

    bool PrepareStartWriteCache(RequestContext *request_context,
                                const proto::meta::StartWriteCacheRequest *request,
                                proto::meta::StartWriteCacheResponse *response);
    bool PrepareFinishWriteCache(RequestContext *request_context,
                                 const proto::meta::FinishWriteCacheRequest *request,
                                 proto::meta::CommonResponse *response);
    bool PrepareRemoveCache(RequestContext *request_context,
                            const proto::meta::RemoveCacheRequest *request,
                            proto::meta::CommonResponse *response);
    bool PrepareTrimCache(RequestContext *request_context,
                          const proto::meta::TrimCacheRequest *request,
                          proto::meta::CommonResponse *response);
    bool PrepareGetClusterInfo(RequestContext *request_context,
                               const proto::meta::GetClusterInfoRequest *request,
                               proto::meta::GetClusterInfoResponse *response);

    bool PrepareReportEvent(RequestContext *request_context,
                            const proto::meta::ReportEventRequest *request,
                            proto::meta::ReportEventResponse *response);

    bool PrepareGetHostCacheState(RequestContext *request_context,
                                  const proto::meta::GetHostCacheStateRequest *request,
                                  proto::meta::GetHostCacheStateResponse *response);

private:
    class AsyncCalls;
    std::shared_ptr<MetaServiceImpl> meta_service_impl_;
    std::unique_ptr<AsyncCalls> async_calls_;
};

} // namespace kv_cache_manager
