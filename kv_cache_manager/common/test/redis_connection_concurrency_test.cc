#include <algorithm>
#include <arpa/inet.h>
#include <atomic>
#include <autil/Thread.h>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <poll.h>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>

#include "kv_cache_manager/common/env_util.h"
#include "kv_cache_manager/common/redis_client.h"
#include "kv_cache_manager/common/unittest.h"

namespace kv_cache_manager {
namespace {

class HandshakeRedisClient : public RedisClient {
public:
    explicit HandshakeRedisClient(std::function<bool()> connect,
                                  int64_t timeout_ms = 2000,
                                  ConnectionType type = ConnectionType::kDefault)
        : RedisClient(
              StandardUri::FromUri("redis://localhost:1/?retry_count=1&timeout_ms=" + std::to_string(timeout_ms)), type)
        , connect_(std::move(connect)) {}

    bool ReconnectForTest() { return Reconnect(); }

protected:
    bool DoConnect() override { return connect_(); }

private:
    std::function<bool()> connect_;
};

struct HandshakeGate {
    bool Enter() {
        std::unique_lock<std::mutex> lock(mutex);
        ++active;
        ++calls;
        peak = std::max(peak, active);
        changed.notify_all();
        changed.wait(lock, [&] { return release; });
        --active;
        return true;
    }

    bool WaitForActive(size_t count) {
        std::unique_lock<std::mutex> lock(mutex);
        return changed.wait_for(lock, std::chrono::seconds(5), [&] { return active >= count; });
    }

    void Release() {
        std::lock_guard<std::mutex> lock(mutex);
        release = true;
        changed.notify_all();
    }

    std::mutex mutex;
    std::condition_variable changed;
    size_t active = 0;
    size_t calls = 0;
    size_t peak = 0;
    bool release = false;
};

// Minimal PING endpoint; replies can be disabled to test handshake timeouts.
class PingServer {
public:
    explicit PingServer(bool reply) : reply_(reply) {}

    bool Start() {
        listener_ = socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK, 0);
        if (listener_ < 0) {
            return false;
        }
        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        if (bind(listener_, reinterpret_cast<sockaddr *>(&address), sizeof(address)) != 0 ||
            listen(listener_, 1) != 0) {
            close(listener_);
            listener_ = -1;
            return false;
        }
        socklen_t length = sizeof(address);
        if (getsockname(listener_, reinterpret_cast<sockaddr *>(&address), &length) != 0) {
            close(listener_);
            listener_ = -1;
            return false;
        }
        port_ = ntohs(address.sin_port);
        worker_ = autil::Thread::createThread([this] { Run(); });
        if (!worker_) {
            close(listener_);
            listener_ = -1;
            return false;
        }
        return true;
    }

    ~PingServer() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_.store(true);
        }
        changed_.notify_all();
        if (worker_) {
            worker_->join();
        }
        if (listener_ >= 0) {
            close(listener_);
        }
    }

    int Port() const { return port_; }

private:
    void Run() {
        pollfd incoming{listener_, POLLIN, 0};
        while (!stop_.load() && poll(&incoming, 1, 20) <= 0) {}
        if (stop_.load()) {
            return;
        }
        const int client = accept(listener_, nullptr, nullptr);
        if (client < 0) {
            return;
        }
        pollfd command{client, POLLIN, 0};
        while (!stop_.load() && poll(&command, 1, 20) <= 0) {}
        if (!stop_.load()) {
            char buffer[256];
            if (recv(client, buffer, sizeof(buffer), MSG_DONTWAIT) > 0 && reply_) {
                const char reply[] = "+PONG\r\n";
                [[maybe_unused]] const auto sent = send(client, reply, sizeof(reply) - 1, MSG_NOSIGNAL);
            }
            std::unique_lock<std::mutex> lock(mutex_);
            changed_.wait(lock, [&] { return stop_.load(); });
        }
        close(client);
    }

    int listener_ = -1;
    int port_ = 0;
    bool reply_;
    std::atomic<bool> stop_{false};
    std::mutex mutex_;
    std::condition_variable changed_;
    autil::ThreadPtr worker_;
};

class RedisConnectionConcurrencyTest : public TESTBASE {
protected:
    // Fix the process-wide limit before its first use.
    ScopedEnv limit_{"kvcm.redis_client.max_concurrent_connects", "2"};
};

TEST_F(RedisConnectionConcurrencyTest, BoundsOpenAndReconnectAcrossClients) {
    HandshakeGate gate;
    std::vector<std::thread> workers;
    std::atomic<int> successful{0};
    for (int i = 0; i < 8; ++i) {
        workers.emplace_back([&, i] {
            HandshakeRedisClient client([&] { return gate.Enter(); });
            if (i % 2 ? client.Open() : client.ReconnectForTest()) {
                ++successful;
            }
        });
    }
    const bool overlapped = gate.WaitForActive(2);
    gate.Release();
    for (auto &worker : workers) {
        worker.join();
    }
    EXPECT_TRUE(overlapped);
    EXPECT_EQ(8, successful.load());
    EXPECT_EQ(8, gate.calls);
    EXPECT_EQ(2, gate.peak);
}

TEST_F(RedisConnectionConcurrencyTest, CoordinationConnectsWhileDefaultBudgetIsSaturated) {
    HandshakeGate gate;
    std::thread first([&] { HandshakeRedisClient([&] { return gate.Enter(); }).Open(); });
    std::thread second([&] { HandshakeRedisClient([&] { return gate.Enter(); }).Open(); });
    const bool full = gate.WaitForActive(2);
    size_t default_calls = 0;
    HandshakeRedisClient waiting(
        [&] {
            ++default_calls;
            return true;
        },
        20);
    const bool timed_out = !waiting.Open();

    size_t coordination_calls = 0;
    HandshakeRedisClient coordination(
        [&] {
            ++coordination_calls;
            return true;
        },
        20,
        RedisClient::ConnectionType::kCoordination);
    const bool opened = coordination.Open();
    const bool reconnected = coordination.ReconnectForTest();

    gate.Release();
    first.join();
    second.join();

    EXPECT_TRUE(full);
    EXPECT_TRUE(opened);
    EXPECT_TRUE(reconnected);
    EXPECT_EQ(2, coordination_calls);
    EXPECT_TRUE(timed_out);
    EXPECT_EQ(0, default_calls);
    EXPECT_TRUE(waiting.Open());
    EXPECT_EQ(1, default_calls);
}

TEST_F(RedisConnectionConcurrencyTest, FailureReleasesPermits) {
    HandshakeRedisClient failed([] { return false; });
    EXPECT_FALSE(failed.Open());
    HandshakeGate gate;
    std::thread first([&] { HandshakeRedisClient([&] { return gate.Enter(); }).Open(); });
    std::thread second([&] { HandshakeRedisClient([&] { return gate.Enter(); }).Open(); });
    const bool full = gate.WaitForActive(2);
    gate.Release();
    first.join();
    second.join();
    EXPECT_TRUE(full);
    EXPECT_EQ(2, gate.peak);
}

TEST_F(RedisConnectionConcurrencyTest, RealHandshakeTimesOutWithoutPingReply) {
    PingServer server(false);
    ASSERT_TRUE(server.Start());
    RedisClient client(
        StandardUri::FromUri("redis://127.0.0.1:" + std::to_string(server.Port()) + "/?retry_count=1&timeout_ms=40"));
    const auto begin = std::chrono::steady_clock::now();
    EXPECT_FALSE(client.Open());
    EXPECT_LT(std::chrono::steady_clock::now() - begin, std::chrono::seconds(2));
}

TEST_F(RedisConnectionConcurrencyTest, RealHandshakeRestoresSteadyStateSocketTimeout) {
    PingServer server(true);
    ASSERT_TRUE(server.Start());
    RedisClient client(
        StandardUri::FromUri("redis://127.0.0.1:" + std::to_string(server.Port()) + "/?retry_count=1&timeout_ms=1000"));
    ASSERT_TRUE(client.Open());
    timeval timeout{};
    socklen_t size = sizeof(timeout);
    ASSERT_EQ(0, getsockopt(client.context_->fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, &size));
    EXPECT_EQ(0, timeout.tv_sec);
    EXPECT_EQ(0, timeout.tv_usec);
    ASSERT_EQ(0, getsockopt(client.context_->fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, &size));
    EXPECT_EQ(0, timeout.tv_sec);
    EXPECT_EQ(0, timeout.tv_usec);
}

} // namespace
} // namespace kv_cache_manager
