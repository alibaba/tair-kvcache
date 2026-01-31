#include "kv_cache_manager/service/grpc_service/debug_service_grpc.h"

#include <memory>
#include <utility>

#include "kv_cache_manager/protocol/protobuf/debug_service.grpc.pb.h"
#include "kv_cache_manager/service/debug_service_impl.h"

namespace kv_cache_manager {

DebugServiceGRpc::DebugServiceGRpc(std::shared_ptr<MetricsRegistry> metrics_registry,
                                   std::shared_ptr<DebugServiceImpl> debug_service_impl)
    : metrics_registry_(std::move(metrics_registry)), debug_service_impl_(std::move(debug_service_impl)) {}

grpc::Status DebugServiceGRpc::InjectFault(grpc::ServerContext *context,
                                           const proto::debug::InjectFaultRequest *request,
                                           proto::debug::CommonResponse *response) {
    debug_service_impl_->InjectFault(request, response);
    return grpc::Status::OK;
}

grpc::Status DebugServiceGRpc::RemoveFault(grpc::ServerContext *context,
                                           const proto::debug::RemoveFaultRequest *request,
                                           proto::debug::CommonResponse *response) {
    debug_service_impl_->RemoveFault(request, response);
    return grpc::Status::OK;
}

grpc::Status DebugServiceGRpc::ClearFaults(grpc::ServerContext *context,
                                           const proto::debug::ClearFaultsRequest *request,
                                           proto::debug::CommonResponse *response) {
    debug_service_impl_->ClearFaults(request, response);
    return grpc::Status::OK;
}

} // namespace kv_cache_manager
