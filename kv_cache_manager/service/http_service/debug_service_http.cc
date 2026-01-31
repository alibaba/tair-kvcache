#include "kv_cache_manager/service/http_service/debug_service_http.h"

#include <memory>
#include <utility>

#include "kv_cache_manager/protocol/protobuf/debug_service.pb.h"
#include "kv_cache_manager/service/debug_service_impl.h"

namespace kv_cache_manager {

DebugServiceHttp::DebugServiceHttp(std::shared_ptr<MetricsRegistry> metrics_registry,
                                   std::shared_ptr<DebugServiceImpl> debug_service_impl)
    : metrics_registry_(std::move(metrics_registry)), debug_service_impl_(std::move(debug_service_impl)) {}

void DebugServiceHttp::RegisterHandler() {
    RegisterPostHandler("/api/injectFault",
                        GetHandler<DebugServiceHttp, proto::debug::InjectFaultRequest, proto::debug::CommonResponse>(
                            &DebugServiceHttp::InjectFault));
    RegisterPostHandler("/api/removeFault",
                        GetHandler<DebugServiceHttp, proto::debug::RemoveFaultRequest, proto::debug::CommonResponse>(
                            &DebugServiceHttp::RemoveFault));
    RegisterPostHandler("/api/clearFaults",
                        GetHandler<DebugServiceHttp, proto::debug::ClearFaultsRequest, proto::debug::CommonResponse>(
                            &DebugServiceHttp::ClearFaults));
}

void DebugServiceHttp::InjectFault(coro_http::coro_http_connection *http_conn,
                                   const proto::debug::InjectFaultRequest *request,
                                   proto::debug::CommonResponse *response) {
    debug_service_impl_->InjectFault(request, response);
}

void DebugServiceHttp::RemoveFault(coro_http::coro_http_connection *http_conn,
                                   const proto::debug::RemoveFaultRequest *request,
                                   proto::debug::CommonResponse *response) {
    debug_service_impl_->RemoveFault(request, response);
}

void DebugServiceHttp::ClearFaults(coro_http::coro_http_connection *http_conn,
                                   const proto::debug::ClearFaultsRequest *request,
                                   proto::debug::CommonResponse *response) {
    debug_service_impl_->ClearFaults(request, response);
}

} // namespace kv_cache_manager
