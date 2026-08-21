#pragma once

#include <grpcpp/grpcpp.h>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>

#include "kv_cache_manager/protocol/protobuf/optimizer_service.grpc.pb.h"

namespace kv_cache_manager {

class SubscriptionEventSink;
class RegistryManager;
class MetricsRegistry;

// gRPC service for optimizer event subscriptions.
class OptimizerEventServiceGRpc final : public proto::optimizer::OptimizerEventStreamService::Service {
public:
    OptimizerEventServiceGRpc(std::shared_ptr<SubscriptionEventSink> sink,
                              std::shared_ptr<RegistryManager> registry_manager,
                              std::shared_ptr<MetricsRegistry> metrics_registry = nullptr);

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
    struct IngressCounters {
        std::uint64_t batches = 0;
        std::uint64_t observations = 0;
        std::uint64_t tokens = 0;
    };

    void ReportIngressMetrics(const std::map<std::string, IngressCounters> &batch_counters);
    void ReportStreamMetrics();

    std::shared_ptr<SubscriptionEventSink> sink_;
    std::shared_ptr<RegistryManager> registry_manager_;
    std::shared_ptr<MetricsRegistry> metrics_registry_;
    std::mutex report_trace_mutex_;
    std::map<std::string, IngressCounters> ingress_counters_;
};

} // namespace kv_cache_manager
