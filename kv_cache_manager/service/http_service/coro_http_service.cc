#include "kv_cache_manager/service/http_service/coro_http_service.h"

#include <memory>
#include <string>
#include <utility>

#include "kv_cache_manager/common/logger.h"
#include "ylt/coro_http/coro_http_server.hpp"

namespace kv_cache_manager {

namespace {

void LogDebugHttpRequest(coro_http::coro_http_request &req) {
    KVCM_LOG_DEBUG("[REQ] method=%.*s path=%.*s body=%.*s",
                   (int)req.get_method().size(),
                   req.get_method().data(),
                   (int)req.get_url().size(),
                   req.get_url().data(),
                   (int)req.get_body().size(),
                   req.get_body().data());
}

void LogDebugHttpResponse(coro_http::coro_http_response &res) {
    KVCM_LOG_DEBUG(
        "[RES] status=%d body=%.*s", static_cast<int>(res.status()), (int)res.content().size(), res.content().data());
}

} // namespace

// Implementation is now in the header file
void CoroHttpService::RegisterGetHandler(const std::string &api, HandlerType handler) {
    get_handlers_[api] = std::move(handler);
}
void CoroHttpService::RegisterPostHandler(const std::string &api, HandlerType handler) {
    post_handlers_[api] = std::move(handler);
}

bool CoroHttpService::Start(int32_t port, size_t thread_num) {
    server_ = std::make_unique<coro_http::coro_http_server>(thread_num, static_cast<unsigned short>(port), "0.0.0.0");

    // 注册所有 GET/POST handler
    for (const auto &[path, handler] : get_handlers_) {
        server_->set_http_handler<coro_http::GET>(path, WrapWithLogger(path, handler));
    }
    for (const auto &[path, handler] : post_handlers_) {
        server_->set_http_handler<coro_http::POST>(path, WrapWithLogger(path, handler));
    }

    auto ec = server_->async_start().get(); // 注意这里用 get()
    if (ec) {
        KVCM_LOG_ERROR("HTTP server start failed on port [%d]: %s", port, ec.message().c_str());
        return false;
    }

    KVCM_LOG_INFO("http server exit on port [%d]", port);

    return true;
}

void CoroHttpService::Stop() {
    if (server_) {
        server_->stop();
        KVCM_LOG_INFO("http server stopped.");
    }
}

std::string CoroHttpService::GetHttpClientIp(const coro_http::coro_http_connection *http_conn) {
    if (http_conn) {
        coro_http::coro_http_connection *non_const_conn = const_cast<coro_http::coro_http_connection *>(http_conn);
        std::string remote_addr = non_const_conn->remote_address();
        if (!remote_addr.empty()) {
            // 处理 IPv6 地址 [IP]:port 格式
            if (remote_addr.front() == '[') {
                auto end_bracket = remote_addr.find(']');
                if (end_bracket != std::string::npos) {
                    return remote_addr.substr(1, end_bracket - 1);
                }
            }

            // 处理 IPv4 地址 IP:port 格式
            auto pos = remote_addr.find(':');
            if (pos != std::string::npos) {
                return remote_addr.substr(0, pos); // 取 IP 部分
            }
            return remote_addr;
        }
    }

    return "0.0.0.0";
}

CoroHttpService::HandlerType CoroHttpService::WrapWithLogger(const std::string &api, HandlerType handler) {
    return [api, handler](coro_http::coro_http_request &req,
                          coro_http::coro_http_response &res) -> async_simple::coro::Lazy<void> {
        LogDebugHttpRequest(req);

        co_await handler(req, res);

        LogDebugHttpResponse(res);
    };
}

} // namespace kv_cache_manager
