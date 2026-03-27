#include "client_c.h"

#include <gtest/gtest.h>

#include <chrono>
#include <memory>
#include <optional>
#include <netinet/in.h>
#include <string>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>

#include <ylt/coro_http/coro_http_server.hpp>

namespace mooncake {
class Client;
} // namespace mooncake

namespace {

using ResolveMetricsHostFn = ErrorCode_t (*)(void *, std::string *);
using DestroyMetricsResolverStateFn = void (*)(void *);

struct MetricsResolverVTable {
    ResolveMetricsHostFn resolve_host = nullptr;
    DestroyMetricsResolverStateFn destroy_state = nullptr;
};

struct FakeMetricsResolverState {
    std::string host;
    ErrorCode_t error_code = MOONCAKE_ERROR_OK;
};

ErrorCode_t ResolveFakeMetricsHost(void *opaque, std::string *host) {
    auto *state = reinterpret_cast<FakeMetricsResolverState *>(opaque);
    if (state == nullptr || host == nullptr) {
        return MOONCAKE_ERROR_INVALID_PARAMS;
    }
    if (state->error_code != MOONCAKE_ERROR_OK) {
        return state->error_code;
    }
    *host = state->host;
    return MOONCAKE_ERROR_OK;
}

struct FakeClientSession {
    std::shared_ptr<mooncake::Client> client;
    coro_http::coro_http_client http_client;
    uint16_t metrics_port = 9003;
    void *resolver_state = nullptr;
    MetricsResolverVTable resolver_vtable{};
};

int GetFreeTcpPort() {
    int sock = ::socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
        return -1;
    }

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = htons(0);
    if (::bind(sock, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) != 0) {
        ::close(sock);
        return -1;
    }

    socklen_t len = sizeof(addr);
    if (::getsockname(sock, reinterpret_cast<sockaddr *>(&addr), &len) != 0) {
        ::close(sock);
        return -1;
    }

    const int port = ntohs(addr.sin_port);
    ::close(sock);
    return port;
}

class TestHttpServer {
public:
    explicit TestHttpServer(int port)
        : server_(std::make_unique<coro_http::coro_http_server>(
              1, static_cast<unsigned short>(port), "127.0.0.1")) {}

    void SetHealthResponse(int status_code, std::string body) {
        health_status_code_ = status_code;
        health_body_ = std::move(body);
    }

    void SetMetricsResponse(int status_code, std::string body) {
        metrics_status_code_ = status_code;
        metrics_body_ = std::move(body);
    }

    void Start() {
        using coro_http::GET;

        server_->set_http_handler<GET>(
            "/health",
            [this](coro_http::coro_http_request &,
                   coro_http::coro_http_response &resp)
                -> async_simple::coro::Lazy<void> {
                resp.set_status_and_content(ToStatusType(health_status_code_),
                                            health_body_);
                co_return;
            });
        server_->set_http_handler<GET>(
            "/metrics",
            [this](coro_http::coro_http_request &,
                   coro_http::coro_http_response &resp)
                -> async_simple::coro::Lazy<void> {
                resp.set_status_and_content(ToStatusType(metrics_status_code_),
                                            metrics_body_);
                co_return;
            });

        auto ec = server_->async_start();
        ASSERT_FALSE(ec.hasResult());
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

    void Stop() {
        if (server_) {
            server_->stop();
        }
    }

private:
    static coro_http::status_type ToStatusType(int code) {
        return static_cast<coro_http::status_type>(code);
    }

private:
    std::unique_ptr<coro_http::coro_http_server> server_;
    int health_status_code_ = 200;
    std::string health_body_ = "OK";
    int metrics_status_code_ = 200;
    std::string metrics_body_;
};

class MooncakeClientCTest : public ::testing::Test {
public:
    void SetUp() override {
        port_ = GetFreeTcpPort();
        ASSERT_GT(port_, 0);
        server_ = std::make_unique<TestHttpServer>(port_);
        session_ = std::make_unique<FakeClientSession>();
        session_->metrics_port = static_cast<uint16_t>(port_);
        resolver_state_ = std::make_unique<FakeMetricsResolverState>();
        session_->resolver_state = resolver_state_.get();
        session_->resolver_vtable = MetricsResolverVTable{
            .resolve_host = ResolveFakeMetricsHost,
            .destroy_state = nullptr,
        };
        client_ = reinterpret_cast<client_t>(session_.get());
    }

    void TearDown() override {
        if (server_) {
            server_->Stop();
        }
    }

protected:
    int port_ = -1;
    client_t client_ = nullptr;
    std::unique_ptr<FakeClientSession> session_;
    std::unique_ptr<FakeMetricsResolverState> resolver_state_;
    std::unique_ptr<TestHttpServer> server_;
};

TEST_F(MooncakeClientCTest, GetStoreStatusParsesHealthAndWaterLevel) {
    resolver_state_->host = "127.0.0.1";
    server_->SetHealthResponse(200, "OK");
    server_->SetMetricsResponse(
        200,
        "# HELP master_allocated_bytes allocated bytes\n"
        "master_allocated_bytes 256\n"
        "# HELP master_total_capacity_bytes total bytes\n"
        "master_total_capacity_bytes 1024\n");
    server_->Start();

    MooncakeStoreStatus_t status{};
    const auto ec = mooncake_client_get_store_status(client_, &status);

    EXPECT_EQ(MOONCAKE_ERROR_OK, ec);
    EXPECT_TRUE(status.healthy);
    EXPECT_EQ(200, status.health_status_code);
    EXPECT_EQ(256u, status.allocated_bytes);
    EXPECT_EQ(1024u, status.total_capacity_bytes);
    EXPECT_DOUBLE_EQ(0.25, status.used_ratio);
}

TEST_F(MooncakeClientCTest, GetStoreStatusTreatsNon200HealthAsUnhealthy) {
    resolver_state_->host = "127.0.0.1";
    server_->SetHealthResponse(503, "NOT_OK");
    server_->SetMetricsResponse(200,
                                "master_allocated_bytes 128\n"
                                "master_total_capacity_bytes 512\n");
    server_->Start();

    MooncakeStoreStatus_t status{};
    const auto ec = mooncake_client_get_store_status(client_, &status);

    EXPECT_EQ(MOONCAKE_ERROR_OK, ec);
    EXPECT_FALSE(status.healthy);
    EXPECT_EQ(503, status.health_status_code);
    EXPECT_EQ(128u, status.allocated_bytes);
    EXPECT_EQ(512u, status.total_capacity_bytes);
    EXPECT_DOUBLE_EQ(0.25, status.used_ratio);
}

TEST_F(MooncakeClientCTest, GetStoreStatusFailsWhenMetricsAreIncomplete) {
    resolver_state_->host = "127.0.0.1";
    server_->SetHealthResponse(200, "OK");
    server_->SetMetricsResponse(200, "master_allocated_bytes 256\n");
    server_->Start();

    MooncakeStoreStatus_t status{};
    const auto ec = mooncake_client_get_store_status(client_, &status);

    EXPECT_EQ(MOONCAKE_ERROR_INTERNAL_ERROR, ec);
}

TEST_F(MooncakeClientCTest, GetStoreStatusUsesResolvedHostFromSession) {
    resolver_state_->host = "127.0.0.1";
    server_->SetHealthResponse(200, "OK");
    server_->SetMetricsResponse(
        200,
        "master_allocated_bytes 64\n"
        "master_total_capacity_bytes 256\n");
    server_->Start();

    MooncakeStoreStatus_t status{};
    const auto ec = mooncake_client_get_store_status(client_, &status);

    EXPECT_EQ(MOONCAKE_ERROR_OK, ec);
    EXPECT_TRUE(status.healthy);
    EXPECT_EQ(64u, status.allocated_bytes);
    EXPECT_EQ(256u, status.total_capacity_bytes);
    EXPECT_DOUBLE_EQ(0.25, status.used_ratio);
}

TEST_F(MooncakeClientCTest, GetStoreStatusFailsWhenHostResolutionFails) {
    resolver_state_->error_code = MOONCAKE_ERROR_UNAVAILABLE_IN_CURRENT_STATUS;

    MooncakeStoreStatus_t status{};
    const auto ec = mooncake_client_get_store_status(client_, &status);

    EXPECT_EQ(MOONCAKE_ERROR_UNAVAILABLE_IN_CURRENT_STATUS, ec);
}

} // namespace
