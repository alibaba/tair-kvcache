#include <atomic>
#include <chrono>
#include <future>
#include <string>
#include <thread>

#include "kv_cache_manager/common/unittest.h"
#include "ylt/coro_http/coro_http_client.hpp"
#include "ylt/coro_http/coro_http_server.hpp"

using namespace std::chrono_literals;

namespace kv_cache_manager {

#ifdef CINATRA_HAS_HTTP_IO_METRICS
TEST(CoroHttpIoMetricsTest, ReportsReceiveLoopLagResponseBuildAndSocketWrite) {
    coro_http::coro_http_server server(1, 0);
    server.enable_http_io_metrics();

    std::atomic<int> request_count{0};
    std::promise<coro_http::http_io_metrics> metrics_promise;
    auto metrics_future = metrics_promise.get_future();
    server.set_http_handler<coro_http::POST>(
        "/io-metrics",
        [&](coro_http::coro_http_request &request,
            coro_http::coro_http_response &response) -> async_simple::coro::Lazy<void> {
            const int sequence = ++request_count;
            if (sequence == 1) {
                std::this_thread::sleep_for(50ms);
            } else if (sequence == 3) {
                request.set_http_io_metrics_callback(
                    [&](const coro_http::http_io_metrics &metrics) { metrics_promise.set_value(metrics); });
            }
            for (int i = 0; i < 64; ++i) {
                response.add_header("X-Io-Metrics-" + std::to_string(i), std::string(64, 'v'));
            }
            response.set_status_and_content(coro_http::status_type::ok, std::string(512 * 1024, 'r'));
            co_return;
        });
    server.async_start();
    ASSERT_GT(server.port(), 0);

    {
        coro_http::coro_http_client client;
        const std::string url = "http://127.0.0.1:" + std::to_string(server.port()) + "/io-metrics";
        const std::string request_body(512 * 1024, 'q');
        auto first = client.post(url, request_body, coro_http::req_content_type::text);
        ASSERT_EQ(first.status, 200);
        auto second = client.post("/io-metrics", request_body, coro_http::req_content_type::text);
        ASSERT_EQ(second.status, 200);
        auto third = client.post("/io-metrics", request_body, coro_http::req_content_type::text);
        ASSERT_EQ(third.status, 200);
    }

    ASSERT_EQ(metrics_future.wait_for(2s), std::future_status::ready);
    const auto metrics = metrics_future.get();
    EXPECT_GT(metrics.request_receive_wait_time_us, 0);
    EXPECT_GT(metrics.io_event_loop_lag_us, 10 * 1000);
    EXPECT_GT(metrics.response_build_time_us, 0);
    EXPECT_GT(metrics.socket_write_time_us, 0);

    server.stop();
}
#endif

} // namespace kv_cache_manager
