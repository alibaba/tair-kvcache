#pragma once

#include <memory>

#include "kv_cache_manager/protocol/protobuf/debug_service.pb.h"
#include "kv_cache_manager/service/http_service/coro_http_service.h"

namespace kv_cache_manager {

class DebugServiceImpl;
class MetricsRegistry;

class DebugServiceHttp : public CoroHttpService {
public:
    DebugServiceHttp(std::shared_ptr<MetricsRegistry> metrics_registry,
                     std::shared_ptr<DebugServiceImpl> debug_service_impl);

    void Init() override { /* currently no metrics registration for debug service */
    }

    void RegisterHandler() override;

    void InjectFault(coro_http::coro_http_connection *http_conn,
                     const proto::debug::InjectFaultRequest *request,
                     proto::debug::CommonResponse *response);

    void RemoveFault(coro_http::coro_http_connection *http_conn,
                     const proto::debug::RemoveFaultRequest *request,
                     proto::debug::CommonResponse *response);

    void ClearFaults(coro_http::coro_http_connection *http_conn,
                     const proto::debug::ClearFaultsRequest *request,
                     proto::debug::CommonResponse *response);

private:
    std::shared_ptr<MetricsRegistry> metrics_registry_;
    std::shared_ptr<DebugServiceImpl> debug_service_impl_;
};

} // namespace kv_cache_manager
