#pragma once

#include <grpcpp/grpcpp.h>
#include <memory>

#include "grpc++/grpc++.h"
#include "kv_cache_manager/metrics/metrics_collector.h"
#include "kv_cache_manager/protocol/protobuf/meta_service.grpc.pb.h"
#include "kv_cache_manager/service/meta_service_metrics_base.h"

namespace kv_cache_manager {

class MetaServiceImpl;
class MetricsRegistry;

class MetaServiceGRpc final : public proto::meta::MetaService::Service, public MetaServiceMetricsBase {
public:
    MetaServiceGRpc(std::shared_ptr<MetricsRegistry> metrics_registry,
                    std::shared_ptr<MetaServiceImpl> meta_service_impl,
                    std::shared_ptr<RegistryManager> registry_manager);

    void Init();

    grpc::Status RegisterInstance(grpc::ServerContext *context,
                                  const proto::meta::RegisterInstanceRequest *request,
                                  proto::meta::RegisterInstanceResponse *response) override;
    grpc::Status GetInstanceInfo(grpc::ServerContext *context,
                                 const proto::meta::GetInstanceInfoRequest *request,
                                 proto::meta::GetInstanceInfoResponse *response) override;
    grpc::Status GetCacheMeta(grpc::ServerContext *context,
                              const proto::meta::GetCacheMetaRequest *request,
                              proto::meta::GetCacheMetaResponse *response) override;

    grpc::Status GetCacheLocation(grpc::ServerContext *context,
                                  const proto::meta::GetCacheLocationRequest *request,
                                  proto::meta::GetCacheLocationResponse *response) override;

    grpc::Status StartWriteCache(grpc::ServerContext *context,
                                 const proto::meta::StartWriteCacheRequest *request,
                                 proto::meta::StartWriteCacheResponse *response) override;
    grpc::Status FinishWriteCache(grpc::ServerContext *context,
                                  const proto::meta::FinishWriteCacheRequest *request,
                                  proto::meta::CommonResponse *response) override;
    grpc::Status RemoveCache(grpc::ServerContext *context,
                             const proto::meta::RemoveCacheRequest *request,
                             proto::meta::CommonResponse *response) override;
    grpc::Status TrimCache(grpc::ServerContext *context,
                           const proto::meta::TrimCacheRequest *request,
                           proto::meta::CommonResponse *response) override;

private:
    std::shared_ptr<MetaServiceImpl> meta_service_impl_;
};

} // namespace kv_cache_manager
