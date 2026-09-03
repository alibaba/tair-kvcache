#pragma once

#include <grpcpp/grpcpp.h>
#include <memory>

#include "kv_cache_manager/protocol/protobuf/optimizer_service.grpc.pb.h"

namespace kv_cache_manager {

class SubscriptionEventSink;
class RegistryManager;
class LeaderElector;
class CacheManager;

// gRPC service for optimizer event subscriptions.
class OptimizerEventServiceGRpc final : public proto::optimizer::OptimizerEventStreamService::Service {
public:
    OptimizerEventServiceGRpc(std::shared_ptr<SubscriptionEventSink> sink,
                              std::shared_ptr<RegistryManager> registry_manager,
                              std::shared_ptr<LeaderElector> leader_elector,
                              std::shared_ptr<CacheManager> cache_manager);

    void EnableSubscriptions();
    void DisableSubscriptions();

    grpc::Status GetConfiguration(grpc::ServerContext *context,
                                  const proto::optimizer::KvcmConfigurationRequest *request,
                                  proto::optimizer::KvcmConfigurationResponse *response) override;

    grpc::Status ReportOptimizerEvent(grpc::ServerContext *context,
                                      const proto::optimizer::TraceQueryRequest *request,
                                      proto::optimizer::CommonResponse *response) override;

    grpc::Status SubscribeEvents(grpc::ServerContext *context,
                                 const proto::optimizer::OptimizerEventSubscriptionRequest *request,
                                 grpc::ServerWriter<proto::optimizer::TraceQueryRequest> *writer) override;

private:
    bool IsAvailable() const;

    std::shared_ptr<SubscriptionEventSink> sink_;
    std::shared_ptr<RegistryManager> registry_manager_;
    std::shared_ptr<LeaderElector> leader_elector_;
    std::shared_ptr<CacheManager> cache_manager_;
};

} // namespace kv_cache_manager
