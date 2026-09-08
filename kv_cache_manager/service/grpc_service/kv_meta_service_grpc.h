#pragma once

#include <grpcpp/grpcpp.h>

#include <memory>

#include "kv_cache_manager/metrics/metrics_collector.h"
#include "kv_cache_manager/protocol/protobuf/kv_meta_service.grpc.pb.h"

namespace kv_cache_manager {

class KvMetaServiceImpl;
class MetricsRegistry;

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
