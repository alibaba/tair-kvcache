#pragma once

#include <string>

#include "kv_cache_manager/metrics/metrics_collector.h"

namespace kv_cache_manager {

std::string ExtractIpFromPeer(const std::string &peer);

#ifndef MAKE_SERVICE_METRICS_COLLECTOR
#define MAKE_SERVICE_METRICS_COLLECTOR(method)                                                                         \
    KVCM_MAKE_METRICS_COLLECTOR_(metrics_registry_, method, Service, (MetricsTags{{"api_name", #method}}))
#endif

#ifndef API_CONTEXT_INIT
#define API_CONTEXT_INIT(metrics_collector, extract_ip, ...)                                                           \
    std::shared_ptr<RequestContext> request_context_ptr =                                                              \
        std::make_shared<RequestContext>(request->trace_id(), metrics_collector);                                      \
    RequestContext *request_context = request_context_ptr.get();                                                       \
    request_context->set_client_ip(extract_ip(__VA_ARGS__));
#endif

#ifndef API_CONTEXT_INIT_GRPC
#define API_CONTEXT_INIT_GRPC(method)                                                                                  \
    API_CONTEXT_INIT(KVCM_METRICS_COLLECTOR_(method), ExtractIpFromPeer, context->peer())
#endif

#ifndef API_CONTEXT_INIT_HTTP
#define API_CONTEXT_INIT_HTTP(method) API_CONTEXT_INIT(KVCM_METRICS_COLLECTOR_(method), GetHttpClientIp, http_conn)
#endif

#ifndef API_CONTEXT_GET_AND_INIT_COLLECTOR
#define API_CONTEXT_GET_AND_INIT_COLLECTOR(method, return_value)                                                       \
    auto metrics_collector = get_metrics_collector_from_map_for_##method(request->instance_id());                      \
    if (metrics_collector == nullptr) {                                                                                \
        KVCM_LOG_ERROR("get " #method " metrics collector failed");                                                    \
        auto *header = response->mutable_header();                                                                     \
        auto *status = header->mutable_status();                                                                       \
        status->set_code(proto::meta::INSTANCE_NOT_EXIST);                                                             \
        status->set_message("get " #method " metrics collector failed");                                               \
        return return_value;                                                                                           \
    }
#endif

#ifndef API_CONTEXT_GET_COLLECTOR_AND_INIT_GRPC
#define API_CONTEXT_GET_COLLECTOR_AND_INIT_GRPC(method, return_value)                                                  \
    API_CONTEXT_GET_AND_INIT_COLLECTOR(method, return_value)                                                           \
    API_CONTEXT_INIT(metrics_collector, ExtractIpFromPeer, context->peer())
#endif

#ifndef API_CONTEXT_GET_COLLECTOR_AND_INIT_HTTP
#define API_CONTEXT_GET_COLLECTOR_AND_INIT_HTTP(method, return_value)                                                  \
    API_CONTEXT_GET_AND_INIT_COLLECTOR(method, return_value)                                                           \
    API_CONTEXT_INIT(metrics_collector, GetHttpClientIp, http_conn)
#endif

#ifndef REGISTER_HTTP_HANDLER_FOR_ADMIN_SERVICE
#define REGISTER_HTTP_HANDLER_FOR_ADMIN_SERVICE(req_type, name, req_name, resp_name, method)                           \
    do {                                                                                                               \
        Register##req_type##Handler(                                                                                   \
            "/api/" #name,                                                                                             \
            GetHandler<AdminServiceHttp, proto::admin::req_name##Request, proto::admin::resp_name##Response>(          \
                &AdminServiceHttp::method));                                                                           \
    } while (0)
#endif

#ifndef REGISTER_HTTP_HANDLER_FOR_META_SERVICE
#define REGISTER_HTTP_HANDLER_FOR_META_SERVICE(req_type, name, req_name, resp_name, method)                            \
    do {                                                                                                               \
        Register##req_type##Handler(                                                                                   \
            "/api/" #name,                                                                                             \
            GetHandler<MetaServiceHttp, proto::meta::req_name##Request, proto::meta::resp_name##Response>(             \
                &MetaServiceHttp::method));                                                                            \
    } while (0)
#endif

} // namespace kv_cache_manager
