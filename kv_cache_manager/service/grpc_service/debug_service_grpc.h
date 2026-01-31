#pragma once

#include <grpcpp/grpcpp.h>
#include <memory>

#include "grpc++/grpc++.h"
#include "kv_cache_manager/protocol/protobuf/debug_service.grpc.pb.h"

namespace kv_cache_manager {

class DebugServiceImpl;
class MetricsRegistry;

class DebugServiceGRpc final : public proto::debug::DebugService::Service {
public:
    DebugServiceGRpc(std::shared_ptr<MetricsRegistry> metrics_registry,
                     std::shared_ptr<DebugServiceImpl> debug_service_impl);

    void Init() { /* currently no metrics registration for debug service */
    }

    grpc::Status InjectFault(grpc::ServerContext *context,
                             const proto::debug::InjectFaultRequest *request,
                             proto::debug::CommonResponse *response) override;

    grpc::Status RemoveFault(grpc::ServerContext *context,
                             const proto::debug::RemoveFaultRequest *request,
                             proto::debug::CommonResponse *response) override;

    grpc::Status ClearFaults(grpc::ServerContext *context,
                             const proto::debug::ClearFaultsRequest *request,
                             proto::debug::CommonResponse *response) override;

private:
    std::shared_ptr<MetricsRegistry> metrics_registry_;
    std::shared_ptr<DebugServiceImpl> debug_service_impl_;
};

} // namespace kv_cache_manager
