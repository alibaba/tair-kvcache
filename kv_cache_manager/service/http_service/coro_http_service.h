#pragma once

#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <unordered_map>

#include "google/protobuf/arena.h"
#include "google/protobuf/message.h"
#include "kv_cache_manager/common/timestamp_util.h"
#include "kv_cache_manager/metrics/metrics_collector.h"
#include "kv_cache_manager/service/util/proto_message_json_util.h"
#include "ylt/coro_http/coro_http_server.hpp"

namespace kv_cache_manager {

struct HttpRequestMetricsSample {
    std::shared_ptr<ServiceMetricsCollector> collector;
    HttpRequestLatency latency;
};

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

    void MergeFrom(const CoroHttpService &other);

    static std::string GetHttpClientIp(const coro_http::coro_http_connection *http_conn);

protected:
    void RegisterGetHandler(const std::string &api, HandlerType handler);
    void RegisterPostHandler(const std::string &api, HandlerType handler);
    HandlerType WrapWithLogger(const std::string &api, HandlerType handler);

    template <typename ServiceType, typename PbRequestMessage, typename PbResponseMessage>
    HandlerType GetHandler(
        std::function<std::enable_if_t<std::is_base_of_v<CoroHttpService, ServiceType>>(
            ServiceType *, coro_http::coro_http_connection *, PbRequestMessage *, PbResponseMessage *)> callback);
    template <typename ServiceType, typename PbRequestMessage, typename PbResponseMessage>
    HandlerType GetArenaHandler(
        std::function<std::enable_if_t<std::is_base_of_v<CoroHttpService, ServiceType>>(
            ServiceType *,
            coro_http::coro_http_connection *,
            PbRequestMessage *,
            PbResponseMessage *,
            HttpRequestMetricsSample *)> callback,
        bool (*request_parser)(char *, size_t, PbRequestMessage *) = nullptr);

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

        if (!ProtoMessageJsonUtil::FromJson(req.get_body(), &pb_req)) {
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

template <typename ServiceType, typename PbRequestMessage, typename PbResponseMessage>
CoroHttpService::HandlerType CoroHttpService::GetArenaHandler(
    std::function<std::enable_if_t<std::is_base_of_v<CoroHttpService, ServiceType>>(
        ServiceType *,
        coro_http::coro_http_connection *,
        PbRequestMessage *,
        PbResponseMessage *,
        HttpRequestMetricsSample *)> callback,
    bool (*request_parser)(char *, size_t, PbRequestMessage *)) {
    return [this, callback, request_parser](coro_http::coro_http_request &req,
                                            coro_http::coro_http_response &res) -> async_simple::coro::Lazy<void> {
        const auto handler_begin_us = TimestampUtil::GetSteadyTimeUs();
        // Large ReportEvent requests contain tens of thousands of nested
        // protobuf messages. Keeping them on a request-scoped arena avoids a
        // separate malloc/free for every EventItem/spec and releases all
        // parser-owned objects in one pass when the synchronous handler
        // returns. Neither message escapes this coroutine.
        const std::string_view body = req.get_body();
        google::protobuf::ArenaOptions arena_options;
        if (body.size() >= 32 * 1024) {
            // Protobuf 3.13 defaults to 256-byte/8-KiB arena blocks. A multi-MiB
            // ReportEvent would otherwise allocate and free hundreds of tiny
            // blocks. Keep small heartbeats on the defaults while allowing
            // large requests to grow geometrically to 1 MiB blocks.
            arena_options.start_block_size = 64 * 1024;
            arena_options.max_block_size = 1024 * 1024;
        }
        google::protobuf::Arena arena(arena_options);
        auto *pb_req = google::protobuf::Arena::CreateMessage<PbRequestMessage>(&arena);
        auto *pb_res = google::protobuf::Arena::CreateMessage<PbResponseMessage>(&arena);
        std::string json_res;

        // cinatra 0.5.5 backs get_body() with the connection's mutable
        // std::string. The handler is synchronous with respect to that body,
        // and neither the request nor a DOM view escapes this coroutine, so a
        // specialized parser may safely decode it in place. Generic handlers
        // continue to receive the immutable length-aware view.
        const bool parsed =
            request_parser
                ? request_parser(body.empty() ? nullptr : const_cast<char *>(body.data()), body.size(), pb_req)
                : ProtoMessageJsonUtil::FromJson(body, pb_req);
        const auto request_parse_end_us = TimestampUtil::GetSteadyTimeUs();
        if (!parsed) {
            json_res = "{}";
            res.set_status_and_content(coro_http::status_type::bad_request, json_res);
            co_return;
        }

        HttpRequestMetricsSample metrics_sample;
        callback(static_cast<ServiceType *>(this),
                 req.get_conn(),
                 pb_req,
                 pb_res,
                 &metrics_sample);
        const auto service_callback_end_us = TimestampUtil::GetSteadyTimeUs();

        json_res.reserve(512);
        if (!ProtoMessageJsonUtil::ToJson(pb_res, json_res)) {
            json_res = "{}";
            res.set_status_and_content(coro_http::status_type::internal_server_error, json_res);
        } else {
            res.add_header("Content-Type", "application/json");
            res.set_status_and_content(coro_http::status_type::ok, json_res);
        }
        const auto response_serialize_end_us = TimestampUtil::GetSteadyTimeUs();
        arena.Reset();
        const auto handler_end_us = TimestampUtil::GetSteadyTimeUs();
        metrics_sample.latency.request_parse_time_us = request_parse_end_us - handler_begin_us;
        metrics_sample.latency.service_callback_time_us = service_callback_end_us - request_parse_end_us;
        metrics_sample.latency.response_serialize_time_us = response_serialize_end_us - service_callback_end_us;
        metrics_sample.latency.handler_time_us = handler_end_us - handler_begin_us;
        metrics_sample.collector->RecordHttpRequestLatency(metrics_sample.latency);
        co_return;
    };
}

} // namespace kv_cache_manager
