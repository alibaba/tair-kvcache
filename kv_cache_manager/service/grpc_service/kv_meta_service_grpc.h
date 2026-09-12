#pragma once

#include <exception>
#include <grpcpp/grpcpp.h>
#include <memory>
#include <utility>

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/metrics/metrics_collector.h"
#include "kv_cache_manager/protocol/protobuf/kv_meta_service.grpc.pb.h"

namespace kv_cache_manager {

class KvMetaServiceImpl;
class MetricsRegistry;
class KvMetaServiceGRpcTestPeer;

class KvMetaServiceGRpc final : public proto::kv_meta::MetaService::Service {
public:
    KvMetaServiceGRpc(std::shared_ptr<MetricsRegistry> metrics_registry,
                      std::shared_ptr<KvMetaServiceImpl> service_impl);

    void Init();

    grpc::Status RegisterInstance(grpc::ServerContext *context,
                                  const proto::kv_meta::RegisterInstanceRequest *request,
                                  proto::kv_meta::RegisterInstanceResponse *response) override;
    grpc::Status GetInstanceInfo(grpc::ServerContext *context,
                                 const proto::kv_meta::GetInstanceInfoRequest *request,
                                 proto::kv_meta::GetInstanceInfoResponse *response) override;
    grpc::Status Get(grpc::ServerContext *context,
                     const proto::kv_meta::GetRequest *request,
                     proto::kv_meta::GetResponse *response) override;
    grpc::Status PutStart(grpc::ServerContext *context,
                          const proto::kv_meta::PutStartRequest *request,
                          proto::kv_meta::PutStartResponse *response) override;
    grpc::Status PutFinish(grpc::ServerContext *context,
                           const proto::kv_meta::PutFinishRequest *request,
                           proto::kv_meta::CommonResponse *response) override;
    grpc::Status Remove(grpc::ServerContext *context,
                        const proto::kv_meta::RemoveRequest *request,
                        proto::kv_meta::CommonResponse *response) override;
    grpc::Status Trim(grpc::ServerContext *context,
                      const proto::kv_meta::TrimRequest *request,
                      proto::kv_meta::CommonResponse *response) override;

private:
    friend class KvMetaServiceGRpcTestPeer;

    template <typename Response, typename Callback>
    static grpc::Status InvokeSafely(const char *operation, Response *response, Callback &&callback) {
        try {
            std::forward<Callback>(callback)();
            return grpc::Status::OK;
        } catch (const std::exception &) {
            // Never let service implementation, metrics, or tracing callback
            // exceptions unwind through a gRPC completion-queue thread. Do
            // not copy exception text into either logs or the wire response:
            // provider errors can contain object keys and endpoints.
            try {
                KVCM_LOG_ERROR("unexpected standard exception in KVMeta %s", operation);
            } catch (...) {}
        } catch (...) {
            try {
                KVCM_LOG_ERROR("unexpected non-standard exception in KVMeta %s", operation);
            } catch (...) {}
        }

        // The non-OK transport status suppresses response serialization, but
        // clear any partial response as a best effort for direct callers and
        // tests. Contain even a secondary protobuf allocation failure here.
        try {
            if (response != nullptr) {
                response->Clear();
                response->mutable_header()->mutable_status()->set_code(proto::kv_meta::INTERNAL_ERROR);
                response->mutable_header()->mutable_status()->set_message("KVMeta request failed unexpectedly");
            }
        } catch (...) {}
        // A mutation may already have reached storage. A non-OK transport
        // status forces clients to treat its outcome as unknown and perform
        // idempotent cleanup/reconciliation rather than trusting a normal
        // application-level response.
        return grpc::Status(grpc::StatusCode::INTERNAL, "KVMeta request failed unexpectedly");
    }

    std::shared_ptr<MetricsRegistry> metrics_registry_;
    std::shared_ptr<KvMetaServiceImpl> service_impl_;

    // Keep KVMeta metrics disjoint from the original Meta/Admin APIs. In
    // particular, RegisterInstance and GetInstanceInfo already exist there.
    KVCM_DECLARE_METRICS_COLLECTOR_(KvMetaRegisterInstance);
    KVCM_DECLARE_METRICS_COLLECTOR_(KvMetaGetInstanceInfo);
    KVCM_DECLARE_METRICS_COLLECTOR_(KvMetaGet);
    KVCM_DECLARE_METRICS_COLLECTOR_(KvMetaPutStart);
    KVCM_DECLARE_METRICS_COLLECTOR_(KvMetaPutFinish);
    KVCM_DECLARE_METRICS_COLLECTOR_(KvMetaRemove);
    KVCM_DECLARE_METRICS_COLLECTOR_(KvMetaTrim);
};

} // namespace kv_cache_manager
