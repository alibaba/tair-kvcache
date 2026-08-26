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

    // Called by the KVCM HTTP adapter. DashTrace uses the public Manager
    // Python client and does not require a producer gRPC RPC.
    void ReportTraceBatch(const proto::optimizer::TraceObservationBatchRequest *request,
                          proto::optimizer::TraceObservationBatchResponse *response);

    grpc::Status SubscribeEvents(grpc::ServerContext *context,
                                 const proto::optimizer::OptimizerEventSubscriptionRequest *request,
                                 grpc::ServerWriter<proto::optimizer::TraceQueryRequest> *writer) override;

private:
    bool IsAvailable() const;

    std::shared_ptr<SubscriptionEventSink> sink_;
    std::shared_ptr<RegistryManager> registry_manager_;
    std::shared_ptr<LeaderElector> leader_elector_;
    std::mutex report_trace_mutex_;
};

} // namespace kv_cache_manager
