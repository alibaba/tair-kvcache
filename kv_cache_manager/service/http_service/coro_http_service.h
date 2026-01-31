#pragma once

#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>

#include "google/protobuf/message.h"
#include "kv_cache_manager/service/util/proto_message_json_util.h"
#include "ylt/coro_http/coro_http_server.hpp"

namespace kv_cache_manager {

class MetricsCollector;
class MetricsRegistry;

class CoroHttpService {
public:
    using HandlerType =
        std::function<async_simple::coro::Lazy<void>(coro_http::coro_http_request &, coro_http::coro_http_response &)>;

    CoroHttpService() = default;
    virtual ~CoroHttpService() = default;

    virtual void Init() = 0;
    virtual void RegisterHandler() = 0;

    bool Start(int32_t port, size_t thread_num = std::thread::hardware_concurrency());
    void Stop();

    static std::string GetHttpClientIp(const coro_http::coro_http_connection *http_conn);

protected:
    void RegisterGetHandler(const std::string &api, HandlerType handler);
    void RegisterPostHandler(const std::string &api, HandlerType handler);
    HandlerType WrapWithLogger(const std::string &api, HandlerType handler);

    template <typename ServiceType, typename PbRequestMessage, typename PbResponseMessage>
    HandlerType GetHandler(
        std::function<std::enable_if_t<std::is_base_of_v<CoroHttpService, ServiceType>>(
            ServiceType *, coro_http::coro_http_connection *, PbRequestMessage *, PbResponseMessage *)> callback);

private:
    std::unordered_map<std::string, HandlerType> get_handlers_{};
    std::unordered_map<std::string, HandlerType> post_handlers_{};
    std::unique_ptr<coro_http::coro_http_server> server_{};
};

template <typename ServiceType, typename PbRequestMessage, typename PbResponseMessage>
CoroHttpService::HandlerType CoroHttpService::GetHandler(
    std::function<std::enable_if_t<std::is_base_of_v<CoroHttpService, ServiceType>>(
        ServiceType *, coro_http::coro_http_connection *, PbRequestMessage *, PbResponseMessage *)> callback) {
    return [this, callback](coro_http::coro_http_request &req,
                            coro_http::coro_http_response &res) -> async_simple::coro::Lazy<void> {
        PbRequestMessage pb_req;
        PbResponseMessage pb_res;

        std::string json_res;

        if (!ProtoMessageJsonUtil::FromJson(std::string(req.get_body()), &pb_req)) {
            json_res = "{}";
            res.set_status_and_content(coro_http::status_type::bad_request, json_res);
            co_return;
        }

        callback(static_cast<ServiceType *>(this), req.get_conn(), &pb_req, &pb_res);

        if (!ProtoMessageJsonUtil::ToJson(&pb_res, json_res)) {
            json_res = "{}";
            res.set_status_and_content(coro_http::status_type::internal_server_error, json_res);
            co_return;
        }
        res.add_header("Content-Type", "application/json");

        res.set_status_and_content(coro_http::status_type::ok, json_res);
        co_return;
    };
}

} // namespace kv_cache_manager
