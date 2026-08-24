#pragma once

#include <grpcpp/grpcpp.h>
#include <memory>
#include <mutex>

#include "kv_cache_manager/protocol/protobuf/optimizer_service.grpc.pb.h"

namespace kv_cache_manager {

class SubscriptionEventSink;
class RegistryManager;
class LeaderElector;

// gRPC service for optimizer event subscriptions.
class OptimizerEventServiceGRpc final : public proto::optimizer::OptimizerEventStreamService::Service {
public:
    OptimizerEventServiceGRpc(std::shared_ptr<SubscriptionEventSink> sink,
                              std::shared_ptr<RegistryManager> registry_manager,
                              std::shared_ptr<LeaderElector> leader_elector);

    void EnableSubscriptions();
    void DisableSubscriptions();

    grpc::Status GetConfiguration(grpc::ServerContext *context,
                                  const proto::optimizer::KvcmConfigurationRequest *request,
                                  proto::optimizer::KvcmConfigurationResponse *response) override;

    grpc::Status ReportTraceBatch(grpc::ServerContext *context,
                                  const proto::optimizer::TraceObservationBatchRequest *request,
                                  proto::optimizer::TraceObservationBatchResponse *response) override;

    grpc::Status SubscribeEvents(grpc::ServerContext *context,
                                 const proto::optimizer::OptimizerEventSubscriptionRequest *request,
                                 grpc::ServerWriter<proto::optimizer::TraceQueryRequest> *writer) override;

private:
    bool IsAvailable() const;

    std::shared_ptr<SubscriptionEventSink> sink_;
    std::shared_ptr<RegistryManager> registry_manager_;
    std::shared_ptr<LeaderElector> leader_elector_;
    // Serializes batch admission so concurrent DashTrace producers cannot
    // interleave their observations inside a subscriber queue.
    std::mutex report_trace_mutex_;
};

} // namespace kv_cache_manager
