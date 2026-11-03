#include "kv_cache_manager/metrics/redis_health_metrics.h"

#include <arpa/inet.h>
#include <hiredis.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <stdexcept>
#include <thread>
#include <vector>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/metrics/metrics_lifecycle.h"
#include "kv_cache_manager/metrics/metrics_registry.h"
#include "kv_cache_manager/metrics/prometheus_exporter.h"

using namespace kv_cache_manager;

namespace {
// A real TCP/RESP peer, including accepted connections that never reply.
class ScriptedRedis {
public:
    explicit ScriptedRedis(std::vector<std::string> replies, bool stall = false) {
        fd_ = socket(AF_INET, SOCK_STREAM, 0);
        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        if (fd_ < 0 || bind(fd_, reinterpret_cast<sockaddr *>(&address), sizeof(address)) || listen(fd_, 1)) {
            throw std::runtime_error("cannot listen on loopback");
        }
        socklen_t size = sizeof(address);
        getsockname(fd_, reinterpret_cast<sockaddr *>(&address), &size);
        port_ = ntohs(address.sin_port);
        thread_ = std::thread([this, replies = std::move(replies), stall]() {
            pollfd ready{fd_, POLLIN, 0};
            if (poll(&ready, 1, 2000) <= 0) {
                return;
            }
            const int client = accept(fd_, nullptr, nullptr);
            if (client < 0) {
                return;
            }
            const timeval timeout{1, 0};
            setsockopt(client, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
            if (stall) {
                std::this_thread::sleep_for(std::chrono::milliseconds(250));
                close(client);
                return;
            }
            auto *reader = redisReaderCreate();
            for (const auto &response : replies) {
                void *raw = nullptr;
                while (!raw) {
                    char buffer[1024];
                    const auto n = recv(client, buffer, sizeof(buffer), 0);
                    if (n <= 0 || redisReaderFeed(reader, buffer, n) != REDIS_OK ||
                        redisReaderGetReply(reader, &raw) != REDIS_OK) {
                        redisReaderFree(reader);
                        close(client);
                        return;
                    }
                }
                auto *reply = static_cast<redisReply *>(raw);
                std::vector<std::string> command;
                for (size_t i = 0; i < reply->elements; ++i) {
                    command.emplace_back(reply->element[i]->str, reply->element[i]->len);
                }
                commands.push_back(std::move(command));
                freeReplyObject(reply);
                send(client, response.data(), response.size(), MSG_NOSIGNAL);
            }
            redisReaderFree(reader);
            close(client);
        });
    }
    ~ScriptedRedis() {
        Finish();
        close(fd_);
    }
    void Finish() {
        if (thread_.joinable()) {
            thread_.join();
        }
    }
    std::string Uri(const std::string &auth = "", const std::string &query = "") const {
        return "redis://" + (auth.empty() ? "" : auth + "@") + "127.0.0.1:" + std::to_string(port_) + "/?" + query;
    }
    std::vector<std::vector<std::string>> commands;

private:
    int fd_;
    int port_;
    std::thread thread_;
};
} // namespace

class RedisHealthMetricsTest : public TESTBASE {};

TEST_F(RedisHealthMetricsTest, SelectRedisDependencies) {
    EXPECT_EQ("uri", RedisHealthMetrics::RedisUri("redis", "uri"));
    EXPECT_EQ("uri", RedisHealthMetrics::RedisUri("async_redis", "uri"));
    EXPECT_TRUE(RedisHealthMetrics::RedisUri("cached", "redis://host:6379"));
    EXPECT_TRUE(RedisHealthMetrics::RedisUri("cached", "redis://host:6379/?persistent_type=async_redis"));
    EXPECT_FALSE(RedisHealthMetrics::RedisUri("local", "redis://host:6379"));
    EXPECT_FALSE(RedisHealthMetrics::RedisUri("cached", "redis://host:6379/?persistent_type=local"));
}

TEST_F(RedisHealthMetricsTest, ProbePingAuthAndDatabase) {
    ScriptedRedis server({"-NOAUTH Authentication required\r\n", "+OK\r\n", "+PONG\r\n", "+OK\r\n"});
    ASSERT_TRUE(RedisHealthMetrics::ProbeRedis(server.Uri("probe-test-token", "db=3")));
    server.Finish();
    EXPECT_EQ((std::vector<std::vector<std::string>>{{"PING"}, {"AUTH", "probe-test-token"}, {"PING"}, {"SELECT", "3"}}),
              server.commands);
}

TEST_F(RedisHealthMetricsTest, ProbeFailuresAndTimeout) {
    for (const auto &replies : std::vector<std::vector<std::string>>{
             {"-ERR unavailable\r\n"},
             {"+OK\r\n"}, // PING must return PONG
             {"-NOAUTH required\r\n", "-WRONGPASS invalid\r\n"},
             {"+PONG\r\n", "-ERR invalid DB index\r\n"},
             {"-NOAUTH required\r\n", "+OK\r\n", "-ERR unavailable\r\n"}}) {
        ScriptedRedis server(replies);
        EXPECT_FALSE(RedisHealthMetrics::ProbeRedis(server.Uri("probe-test-token", "db=3")));
    }
    ScriptedRedis stalled({}, true);
    const auto start = std::chrono::steady_clock::now();
    EXPECT_FALSE(RedisHealthMetrics::ProbeRedis(stalled.Uri(), 50));
    EXPECT_LT(std::chrono::steady_clock::now() - start, std::chrono::seconds(1));
    EXPECT_FALSE(RedisHealthMetrics::ProbeRedis("redis://127.0.0.1:6379/?db=invalid"));
    EXPECT_FALSE(RedisHealthMetrics::ProbeRedis("redis://127.0.0.1:6379/?db=-1"));
    EXPECT_FALSE(RedisHealthMetrics::ProbeRedis("not-a-uri"));
}

TEST_F(RedisHealthMetricsTest, ConnectionRefused) {
    const int fd = socket(AF_INET, SOCK_STREAM, 0);
    ASSERT_GE(fd, 0);
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    ASSERT_EQ(0, bind(fd, reinterpret_cast<sockaddr *>(&address), sizeof(address)));
    socklen_t size = sizeof(address);
    ASSERT_EQ(0, getsockname(fd, reinterpret_cast<sockaddr *>(&address), &size));
    // Reserve the port without listening, so no unrelated service is contacted.
    EXPECT_FALSE(RedisHealthMetrics::ProbeRedis("redis://127.0.0.1:" + std::to_string(ntohs(address.sin_port)), 50));
    close(fd);
}

TEST_F(RedisHealthMetricsTest, FreshConnectionRecoversAndChecksSelectedDatabase) {
    ScriptedRedis failed({"-ERR unavailable\r\n"});
    EXPECT_FALSE(RedisHealthMetrics::ProbeRedis(failed.Uri()));
    ScriptedRedis recovered({"+PONG\r\n", "+OK\r\n"});
    EXPECT_TRUE(RedisHealthMetrics::ProbeRedis(recovered.Uri("", "db=2")));
    recovered.Finish();
    EXPECT_EQ((std::vector<std::vector<std::string>>{{"PING"}, {"SELECT", "2"}}), recovered.commands);
}

TEST_F(RedisHealthMetricsTest, PublishesFailureRecoveryAndDeduplicatesGroups) {
    auto registry = std::make_shared<MetricsRegistry>();
    auto lifecycle = std::make_shared<MetricsLifecycle>();
    RedisHealthMetrics::Targets targets{{"group_a", "redis://secret@host:6379"},
                                       {"group_b", "redis://secret@host:6379"},
                                       {"group_c", "redis://other:6379"}};
    int calls = 0;
    bool healthy = true;
    RedisHealthMetrics metrics(registry, lifecycle, [&]() { return targets; }, [&](const std::string &uri) {
        ++calls;
        return uri == "redis://other:6379" || healthy;
    });
    for (bool state : {true, false, true}) {
        healthy = state;
        calls = 0;
        metrics.Report();
        EXPECT_EQ(2, calls);
        EXPECT_DOUBLE_EQ(state ? 1 : 0, registry->GetGauge(RedisHealthMetrics::kMetricName, {{"host", "host"}, {"instance_group", "group_a"}, {"port", "6379"}}).Get());
        EXPECT_DOUBLE_EQ(state ? 1 : 0, registry->GetGauge(RedisHealthMetrics::kMetricName, {{"host", "host"}, {"instance_group", "group_b"}, {"port", "6379"}}).Get());
        EXPECT_DOUBLE_EQ(1, registry->GetGauge(RedisHealthMetrics::kMetricName, {{"host", "other"}, {"instance_group", "group_c"}, {"port", "6379"}}).Get());
    }
    const auto output = PrometheusExporter::Expose(*registry);
    EXPECT_NE(std::string::npos, output.find("kvcm_meta_storage_redis_healthy_status{host=\"host\",instance_group=\"group_a\",port=\"6379\"} 1"));
    EXPECT_EQ(std::string::npos, output.find("secret"));
    EXPECT_EQ(std::string::npos, output.find("host:6379"));
    targets.erase("group_b");
    metrics.Report();
    EXPECT_EQ(std::string::npos, PrometheusExporter::Expose(*registry).find("group_b"));
    targets.clear(); // demotion or removal of the last Redis group
    metrics.Report();
    EXPECT_EQ(std::string::npos, PrometheusExporter::Expose(*registry).find("redis_healthy_status"));
    targets["group_a"] = "redis://host:6379"; // promotion / group recreated
    healthy = false;
    metrics.Report();
    EXPECT_NE(std::string::npos, PrometheusExporter::Expose(*registry).find("instance_group=\"group_a\",port=\"6379\"} 0"));
}

TEST_F(RedisHealthMetricsTest, DoesNotRepublishObsoleteProbes) {
    for (bool demotion : {false, true}) {
        auto registry = std::make_shared<MetricsRegistry>();
        auto lifecycle = std::make_shared<MetricsLifecycle>();
        RedisHealthMetrics::Targets targets{{"group", "old"}};
        bool mutate = false;
        RedisHealthMetrics metrics(registry, lifecycle, [&]() { return targets; }, [&](const std::string &) {
            if (mutate) {
                // Network work must not hold the lifecycle lock.
                std::unique_lock lock(lifecycle->mut_);
                targets = demotion ? RedisHealthMetrics::Targets{} : RedisHealthMetrics::Targets{{"group", "new"}};
            }
            return true;
        });
        metrics.Report();
        mutate = true;
        metrics.Report();
        EXPECT_EQ(std::string::npos, PrometheusExporter::Expose(*registry).find("redis_healthy_status"));
    }
}

TEST_F(RedisHealthMetricsTest, ReplacesHostAndPortLabelsWithoutLeakingCredentials) {
    auto registry = std::make_shared<MetricsRegistry>();
    auto lifecycle = std::make_shared<MetricsLifecycle>();
    RedisHealthMetrics::Targets targets{{"group", "redis://old-password@redis-a.example:6379/?db=3"}};
    bool healthy = true;
    RedisHealthMetrics metrics(registry, lifecycle, [&]() { return targets; }, [&](const std::string &) {
        return healthy;
    });
    metrics.Report();
    EXPECT_NE(std::string::npos, PrometheusExporter::Expose(*registry).find(
                                    "{host=\"redis-a.example\",instance_group=\"group\",port=\"6379\"} 1"));
    targets["group"] = "redis://new-password@10.0.0.8:6380/?db=4";
    healthy = false;
    metrics.Report();
    auto output = PrometheusExporter::Expose(*registry);
    EXPECT_EQ(std::string::npos, output.find("redis-a.example"));
    EXPECT_EQ(std::string::npos, output.find("6379"));
    EXPECT_NE(std::string::npos, output.find("{host=\"10.0.0.8\",instance_group=\"group\",port=\"6380\"} 0"));
    EXPECT_EQ(std::string::npos, output.find("password"));
    EXPECT_EQ(std::string::npos, output.find("db="));
    EXPECT_EQ(std::string::npos, output.find("redis://"));
    targets.clear();
    metrics.Report();
    EXPECT_EQ(std::string::npos, PrometheusExporter::Expose(*registry).find("redis_healthy_status"));
}
