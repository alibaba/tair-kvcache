#include <array>
#include <sstream>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/service/http_service/coro_http_service.h"

namespace kv_cache_manager {
namespace {

class BodyTestService : public CoroHttpService {
public:
    void Init() override {}
    void RegisterHandler() override {
        RegisterPostHandler("/body", [](auto &req, auto &res) -> async_simple::coro::Lazy<void> {
            auto *conn = req.get_conn();
            // Observe on the IO thread, after the preceding request's cleanup.
            const auto body = req.get_body();
            res.set_status_and_content(
                body.starts_with("invalid") ? coro_http::status_type::bad_request : coro_http::status_type::ok,
                std::to_string(conn->body_.capacity()) + " " + std::to_string(conn->need_shrink_every_time_));
            co_return;
        });
    }
};

class HttpBodyShrinkTest : public TESTBASE {
protected:
    void SetUp() override {
        service_.RegisterHandler();
        service_.server_ = std::make_unique<coro_http::coro_http_server>(1, 0, "127.0.0.1");
        for (const auto &[path, handler] : service_.post_handlers_) {
            service_.server_->set_http_handler<coro_http::POST>(path, service_.WrapWithLogger(path, handler));
        }
        service_.server_->async_start();
        ASSERT_FALSE(service_.server_->errc_);
        socket_.connect({asio::ip::make_address("127.0.0.1"), service_.server_->acceptor_.local_endpoint().port()});
    }
    void TearDown() override {
        asio::error_code ec;
        socket_.close(ec);
        service_.Stop();
    }
    std::pair<size_t, bool> Send(const std::string &body, int expected_status = 200) {
        const std::string header =
            "POST /body HTTP/1.1\r\nHost: localhost\r\nConnection: keep-alive\r\nContent-Length: " +
            std::to_string(body.size()) + "\r\n\r\n";
        asio::write(socket_, std::array<asio::const_buffer, 2>{asio::buffer(header), asio::buffer(body)});
        asio::streambuf response;
        asio::read_until(socket_, response, "\r\n\r\n");
        std::istream input(&response);
        std::string line;
        std::getline(input, line);
        EXPECT_NE(std::string::npos, line.find(std::to_string(expected_status)));
        size_t length = 0;
        while (std::getline(input, line) && line != "\r") {
            if (line.starts_with("Content-Length:")) {
                length = std::stoull(line.substr(15));
            }
        }
        if (response.size() < length) {
            asio::read(socket_, response, asio::transfer_exactly(length - response.size()));
        }
        size_t capacity = 0;
        bool shrink = false;
        input >> capacity >> shrink;
        EXPECT_FALSE(input.fail());
        return {capacity, shrink};
    }
    asio::io_context io_;
    asio::ip::tcp::socket socket_{io_};
    BodyTestService service_;
};

TEST_F(HttpBodyShrinkTest, LargeBodyReleasesCapacityAndSmallRequestsReuseBuffer) {
    constexpr size_t threshold = 8 * 1024 * 1024;
    auto [large_capacity, large_shrink] = Send(std::string(threshold, 'x'));
    EXPECT_GE(large_capacity, threshold);
    EXPECT_TRUE(large_shrink);
    auto [small_capacity, small_shrink] = Send(std::string(4096, 'x'));
    EXPECT_LT(small_capacity, threshold);
    EXPECT_FALSE(small_shrink);
    auto [reused_capacity, reused_shrink] = Send("small");
    EXPECT_EQ(small_capacity, reused_capacity);
    EXPECT_FALSE(reused_shrink);
}

TEST_F(HttpBodyShrinkTest, BelowThresholdRetainsCapacity) {
    auto [capacity, shrink] = Send(std::string(8 * 1024 * 1024 - 1, 'x'));
    EXPECT_FALSE(shrink);
    EXPECT_EQ(capacity, Send("small").first);
}

TEST_F(HttpBodyShrinkTest, ErrorResponseStillReleasesLargeBody) {
    std::string body(8 * 1024 * 1024, 'x');
    body.replace(0, 7, "invalid");
    EXPECT_TRUE(Send(body, 400).second);
    EXPECT_LT(Send("small").first, body.size());
}

} // namespace
} // namespace kv_cache_manager
