#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>

#include "google/protobuf/descriptor.h"
#include "google/protobuf/message.h"
#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/timestamp_util.h"
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
        const std::string request_type = pb_req.GetDescriptor() ? pb_req.GetDescriptor()->full_name() : "unknown";
        const bool should_log_http_stage = request_type.find(".ReportEventRequest") != std::string::npos ||
                                           request_type.find(".GetHostCacheStateRequest") != std::string::npos;

        const int64_t t0 = TimestampUtil::GetSteadyTimeUs();
        std::string body(req.get_body());
        const int64_t t1 = TimestampUtil::GetSteadyTimeUs();

        if (!ProtoMessageJsonUtil::FromJson(body, &pb_req)) {
            const int64_t t2 = TimestampUtil::GetSteadyTimeUs();
            json_res = "{}";
            res.set_status_and_content(coro_http::status_type::bad_request, json_res);
            const int64_t t3 = TimestampUtil::GetSteadyTimeUs();
            if (should_log_http_stage) {
                KVCM_LOG_INFO("http json stage parse_failed url=%.*s request_type=%s body_size=%zu body_copy_us=%ld "
                              "from_json_us=%ld set_response_us=%ld handler_total_us=%ld",
                              static_cast<int>(req.get_url().size()),
                              req.get_url().data(),
                              request_type.c_str(),
                              body.size(),
                              t1 - t0,
                              t2 - t1,
                              t3 - t2,
                              t3 - t0);
            }
            co_return;
        }
        const int64_t t2 = TimestampUtil::GetSteadyTimeUs();

        std::string trace_id;
        if (pb_req.GetDescriptor() && pb_req.GetReflection()) {
            const auto *trace_field = pb_req.GetDescriptor()->FindFieldByName("trace_id");
            if (trace_field && trace_field->cpp_type() == google::protobuf::FieldDescriptor::CPPTYPE_STRING &&
                !trace_field->is_repeated()) {
                trace_id = pb_req.GetReflection()->GetString(pb_req, trace_field);
            }
        }

        callback(static_cast<ServiceType *>(this), req.get_conn(), &pb_req, &pb_res);
        const int64_t t3 = TimestampUtil::GetSteadyTimeUs();

        if (!ProtoMessageJsonUtil::ToJson(&pb_res, json_res)) {
            const int64_t t4 = TimestampUtil::GetSteadyTimeUs();
            json_res = "{}";
            res.set_status_and_content(coro_http::status_type::internal_server_error, json_res);
            const int64_t t5 = TimestampUtil::GetSteadyTimeUs();
            if (should_log_http_stage) {
                const std::string response_type =
                    pb_res.GetDescriptor() ? pb_res.GetDescriptor()->full_name() : "unknown";
                KVCM_LOG_INFO("http json stage response_json_failed url=%.*s request_type=%s response_type=%s "
                              "trace_id=%s body_size=%zu response_size=%zu body_copy_us=%ld from_json_us=%ld "
                              "callback_us=%ld to_json_us=%ld set_response_us=%ld handler_total_us=%ld",
                              static_cast<int>(req.get_url().size()),
                              req.get_url().data(),
                              request_type.c_str(),
                              response_type.c_str(),
                              trace_id.c_str(),
                              body.size(),
                              json_res.size(),
                              t1 - t0,
                              t2 - t1,
                              t3 - t2,
                              t4 - t3,
                              t5 - t4,
                              t5 - t0);
            }
            co_return;
        }
        const int64_t t4 = TimestampUtil::GetSteadyTimeUs();
        res.add_header("Content-Type", "application/json");

        res.set_status_and_content(coro_http::status_type::ok, json_res);
        const int64_t t5 = TimestampUtil::GetSteadyTimeUs();

        if (should_log_http_stage) {
            const std::string response_type = pb_res.GetDescriptor() ? pb_res.GetDescriptor()->full_name() : "unknown";
            KVCM_LOG_INFO("http json stage url=%.*s request_type=%s response_type=%s trace_id=%s body_size=%zu "
                          "response_size=%zu body_copy_us=%ld from_json_us=%ld callback_us=%ld to_json_us=%ld "
                          "set_response_us=%ld handler_total_us=%ld",
                          static_cast<int>(req.get_url().size()),
                          req.get_url().data(),
                          request_type.c_str(),
                          response_type.c_str(),
                          trace_id.c_str(),
                          body.size(),
                          json_res.size(),
                          t1 - t0,
                          t2 - t1,
                          t3 - t2,
                          t4 - t3,
                          t5 - t4,
                          t5 - t0);
        }
        co_return;
    };
}

} // namespace kv_cache_manager
