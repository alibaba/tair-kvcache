#include <arpa/inet.h>
#include <chrono>
#include <future>
#include <hiredis.h>
#include <poll.h>
#include <stdexcept>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <vector>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/config/instance_group.h"
#include "kv_cache_manager/config/instance_info.h"
#include "kv_cache_manager/config/registry_manager.h"
#include "kv_cache_manager/meta/meta_indexer.h"
#include "kv_cache_manager/meta/meta_indexer_manager.h"
#include "kv_cache_manager/metrics/metrics_lifecycle.h"
#include "kv_cache_manager/metrics/metrics_registry.h"
#include "kv_cache_manager/metrics/prometheus_exporter.h"
#include "kv_cache_manager/metrics/redis_health_metrics.h"

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

class RedisHealthMetricsTest : public TESTBASE {
public:
    void SetUp() override {
        registry_manager_ = std::make_shared<RegistryManager>("", std::make_shared<MetricsRegistry>());
        indexer_manager_ = std::make_shared<MetaIndexerManager>();
    }
    // Populate real runtime snapshots without opening business connections.
    void SetTargets(const RedisHealthMetrics::Targets &targets) {
        std::scoped_lock lock(registry_manager_->mutex_, indexer_manager_->mutex_);
        registry_manager_->instance_infos_.clear();
        indexer_manager_->meta_indexers_.clear();
        size_t id = 0;
        for (const auto &[group, uri] : targets) {
            const auto name = std::to_string(id++);
            auto info = std::make_shared<InstanceInfo>();
            info->set_instance_group_name(group);
            registry_manager_->instance_infos_[name] = info;
            auto indexer = std::make_shared<MetaIndexer>();
            indexer->storage_backend_config_ = {"redis", uri};
            indexer_manager_->meta_indexers_[name] = indexer;
        }
    }
    std::shared_ptr<RegistryManager> registry_manager_;
    std::shared_ptr<MetaIndexerManager> indexer_manager_;
};

TEST_F(RedisHealthMetricsTest, CoordinationReportsFailureRecoveryWithoutExposingCredentials) {
    auto registry = std::make_shared<MetricsRegistry>();
    const std::string uri = "redis://private-token@coord.example:6380/?db=2&cluster_name=private-cluster";
    bool healthy = true;
    int calls = 0;
    CoordinationRedisHealthMetrics metrics(registry, uri, [&](const std::string &target) {
        EXPECT_EQ(uri, target);
        ++calls;
        return healthy;
    });
    ASSERT_TRUE(metrics.IsEnabled());
    EXPECT_TRUE(PrometheusExporter::Expose(*registry).empty()); // no invented initial result
    for (const bool result : {true, false, true}) {
        healthy = result;
        metrics.Report();
        const auto output = PrometheusExporter::Expose(*registry);
        const std::string line = "kvcm_coordination_redis_healthy_status{host=\"coord.example\",port=\"6380\"} ";
        EXPECT_NE(std::string::npos, output.find(line + (result ? "1\n" : "0\n")));
        for (const auto *secret : {"private-token", "private-cluster", "db=", "instance_group", "meta_storage"}) {
            EXPECT_EQ(std::string::npos, output.find(secret));
        }
    }
    metrics.RequestStop();
    metrics.Report();
    EXPECT_EQ(3, calls);
}

TEST_F(RedisHealthMetricsTest, CoordinationSkipsNonRedisBackends) {
    auto registry = std::make_shared<MetricsRegistry>();
    int calls = 0;
    for (const auto *uri : {"", "memory://", "file:///tmp/coordination", "unsupported://host:6379"}) {
        CoordinationRedisHealthMetrics metrics(registry, uri, [&](const std::string &) {
            ++calls;
            return false;
        });
        EXPECT_FALSE(metrics.IsEnabled());
        metrics.Report();
    }
    EXPECT_EQ(0, calls);
    EXPECT_TRUE(PrometheusExporter::Expose(*registry).empty());
}

TEST_F(RedisHealthMetricsTest, CoordinationUsesReadOnlyProbeAndReportsTimeout) {
    for (const bool stall : {false, true}) {
        ScriptedRedis server({"-NOAUTH required\r\n", "+OK\r\n", "+PONG\r\n", "+OK\r\n"}, stall);
        auto registry = std::make_shared<MetricsRegistry>();
        CoordinationRedisHealthMetrics metrics(registry, server.Uri("test-token", "db=3"), [](const std::string &uri) {
            return RedisHealthMetrics::ProbeRedis(uri, 50);
        });
        metrics.Report();
        const auto output = PrometheusExporter::Expose(*registry);
        EXPECT_NE(std::string::npos, output.find(stall ? "} 0\n" : "} 1\n"));
        server.Finish();
        if (!stall) {
            EXPECT_EQ(
                (std::vector<std::vector<std::string>>{{"PING"}, {"AUTH", "test-token"}, {"PING"}, {"SELECT", "3"}}),
                server.commands);
        }
    }
}

TEST_F(RedisHealthMetricsTest, CoordinationIsIndependentOfBlockedMetadataProbe) {
    auto registry = std::make_shared<MetricsRegistry>();
    auto lifecycle = std::make_shared<MetricsLifecycle>();
    std::promise<void> entered, release;
    auto released = release.get_future().share();
    SetTargets({{"group", "redis://meta.example:6379"}});
    RedisHealthMetrics metadata(
        registry,
        lifecycle,
        registry_manager_, indexer_manager_, [] { return true; },
        [&](const std::string &) {
            entered.set_value();
            released.wait();
            return false;
        });
    auto pending = std::async(std::launch::async, [&] { metadata.Report(); });
    EXPECT_EQ(std::future_status::ready, entered.get_future().wait_for(std::chrono::seconds(2)));
    // A recovery/cleanup lock and a stalled metadata connection cannot suppress
    // coordination observations, nor can a coordination failure affect metadata health.
    {
        std::unique_lock writer(lifecycle->mut_);
        CoordinationRedisHealthMetrics coordination(
            registry, "redis://coord.example:6379", [](const std::string &) { return false; });
        coordination.Report();
        EXPECT_NE(std::string::npos,
                  PrometheusExporter::Expose(*registry).find(
                      "kvcm_coordination_redis_healthy_status{host=\"coord.example\",port=\"6379\"} 0\n"));
    }
    release.set_value();
    pending.get();
}

TEST_F(RedisHealthMetricsTest, CoordinationStopDiscardsInFlightResult) {
    auto registry = std::make_shared<MetricsRegistry>();
    std::promise<void> entered, release;
    auto released = release.get_future().share();
    CoordinationRedisHealthMetrics metrics(registry, "redis://coord.example:6379", [&](const std::string &) {
        entered.set_value();
        released.wait();
        return true;
    });
    auto pending = std::async(std::launch::async, [&] { metrics.Report(); });
    EXPECT_EQ(std::future_status::ready, entered.get_future().wait_for(std::chrono::seconds(2)));
    metrics.RequestStop(); // must not wait for network I/O
    release.set_value();
    pending.get();
    metrics.Report();
    EXPECT_TRUE(PrometheusExporter::Expose(*registry).empty());
}

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
    RedisHealthMetrics metrics(registry, lifecycle, registry_manager_, indexer_manager_, [] { return true; }, [&](const std::string &uri) {
        ++calls;
        return uri == "redis://other:6379" || healthy;
    });
    for (bool state : {true, false, true}) {
        healthy = state;
        calls = 0;
        SetTargets(targets);
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
    targets.erase({"group_b", "redis://secret@host:6379"});
    SetTargets(targets);
    metrics.Report();
    EXPECT_EQ(std::string::npos, PrometheusExporter::Expose(*registry).find("group_b"));
    targets.clear(); // demotion or removal of the last Redis group
    SetTargets(targets);
    metrics.Report();
    EXPECT_EQ(std::string::npos, PrometheusExporter::Expose(*registry).find("redis_healthy_status"));
    targets.emplace("group_a", "redis://host:6379"); // promotion / group recreated
    healthy = false;
    SetTargets(targets);
    metrics.Report();
    EXPECT_NE(std::string::npos, PrometheusExporter::Expose(*registry).find("instance_group=\"group_a\",port=\"6379\"} 0"));
}

TEST_F(RedisHealthMetricsTest, DoesNotRepublishObsoleteProbes) {
    for (bool demotion : {false, true}) {
        auto registry = std::make_shared<MetricsRegistry>();
        auto lifecycle = std::make_shared<MetricsLifecycle>();
        RedisHealthMetrics::Targets targets{{"group", "old"}};
        bool mutate = false;
        RedisHealthMetrics metrics(registry, lifecycle, registry_manager_, indexer_manager_, [] { return true; }, [&](const std::string &) {
            if (mutate) {
                // Network work must not hold the lifecycle lock.
                std::unique_lock lock(lifecycle->mut_);
                targets = demotion ? RedisHealthMetrics::Targets{} : RedisHealthMetrics::Targets{{"group", "new"}};
                SetTargets(targets);
            }
            return true;
        });
        SetTargets(targets);
        metrics.Report();
        mutate = true;
        SetTargets(targets);
        metrics.Report();
        EXPECT_EQ(std::string::npos, PrometheusExporter::Expose(*registry).find("redis_healthy_status"));
    }
}

TEST_F(RedisHealthMetricsTest, ReplacesHostAndPortLabelsWithoutLeakingCredentials) {
    auto registry = std::make_shared<MetricsRegistry>();
    auto lifecycle = std::make_shared<MetricsLifecycle>();
    RedisHealthMetrics::Targets targets{{"group", "redis://old-password@redis-a.example:6379/?db=3"}};
    bool healthy = true;
    RedisHealthMetrics metrics(registry, lifecycle, registry_manager_, indexer_manager_, [] { return true; }, [&](const std::string &) {
        return healthy;
    });
    SetTargets(targets);
    metrics.Report();
    EXPECT_NE(std::string::npos, PrometheusExporter::Expose(*registry).find(
                                    "{host=\"redis-a.example\",instance_group=\"group\",port=\"6379\"} 1"));
    targets = {{"group", "redis://new-password@10.0.0.8:6380/?db=4"}};
    healthy = false;
    SetTargets(targets);
    metrics.Report();
    auto output = PrometheusExporter::Expose(*registry);
    EXPECT_EQ(std::string::npos, output.find("redis-a.example"));
    EXPECT_EQ(std::string::npos, output.find("6379"));
    EXPECT_NE(std::string::npos, output.find("{host=\"10.0.0.8\",instance_group=\"group\",port=\"6380\"} 0"));
    EXPECT_EQ(std::string::npos, output.find("password"));
    EXPECT_EQ(std::string::npos, output.find("db="));
    EXPECT_EQ(std::string::npos, output.find("redis://"));
    targets.clear();
    SetTargets(targets);
    metrics.Report();
    EXPECT_EQ(std::string::npos, PrometheusExporter::Expose(*registry).find("redis_healthy_status"));
}

TEST_F(RedisHealthMetricsTest, CollectsRuntimeAndConfiguredRedisAfterGroupUpdate) {
    auto metrics = std::make_shared<MetricsRegistry>();
    auto registry = std::make_shared<RegistryManager>("", metrics);
    auto indexers = std::make_shared<MetaIndexerManager>();
    auto group = std::make_shared<InstanceGroup>();
    group->set_name("group");
    auto backend = std::make_shared<MetaStorageBackendConfig>();
    backend->SetStorageType("redis");
    backend->SetStorageUri("redis://new:6380");
    auto config = std::make_shared<MetaIndexerConfig>();
    config->SetMetaStorageBackendConfig(backend);
    group->set_cache_config(std::make_shared<CacheConfig>(CachePreferStrategy::CPS_UNSPECIFIED, nullptr, config));
    registry->instance_group_configs_[group->name()] = group;
    auto info = std::make_shared<InstanceInfo>();
    info->set_instance_group_name(group->name());
    registry->instance_infos_["instance"] = info;
    auto indexer = std::make_shared<MetaIndexer>();
    indexer->storage_backend_config_ = {"redis", "redis://old:6379"};
    indexers->meta_indexers_["instance"] = indexer;

    auto targets = RedisHealthMetrics::CollectTargets(*registry, *indexers);
    ASSERT_TRUE(targets);
    EXPECT_EQ((RedisHealthMetrics::Targets{{"group", "redis://old:6379"}, {"group", "redis://new:6380"}}), *targets);
    auto lifecycle = std::make_shared<MetricsLifecycle>();
    RedisHealthMetrics reporter(
        metrics,
        lifecycle,
        registry, indexers, [] { return true; },
        [](const std::string &uri) { return uri != "redis://old:6379"; });
    reporter.Report();
    EXPECT_DOUBLE_EQ(0,
                     metrics
                         ->GetGauge(RedisHealthMetrics::kMetricName,
                                    {{"host", "old"}, {"port", "6379"}, {"instance_group", "group"}})
                         .Get());
    EXPECT_DOUBLE_EQ(1,
                     metrics
                         ->GetGauge(RedisHealthMetrics::kMetricName,
                                    {{"host", "new"}, {"port", "6380"}, {"instance_group", "group"}})
                         .Get());
    // Switching the group to local must not hide the existing instance's Redis.
    backend->SetStorageType("local");
    reporter.Report();
    EXPECT_EQ((RedisHealthMetrics::Targets{{"group", "redis://old:6379"}}),
              *RedisHealthMetrics::CollectTargets(*registry, *indexers));
    EXPECT_EQ(std::string::npos, PrometheusExporter::Expose(*metrics).find("host=\"new\""));
    // Once the instance is actually removed, its old endpoint can be removed.
    indexers->meta_indexers_.clear();
    registry->instance_infos_.clear();
    reporter.Report();
    EXPECT_EQ(std::string::npos, PrometheusExporter::Expose(*metrics).find("redis_healthy_status"));
}

TEST_F(RedisHealthMetricsTest, SameEndpointCombinesAllLiveDatabaseAndAuthSettings) {
    auto registry = std::make_shared<MetricsRegistry>();
    auto lifecycle = std::make_shared<MetricsLifecycle>();
    RedisHealthMetrics::Targets targets{{"group", "redis://host:6379/?db=1"}, {"group", "redis://host:6379/?db=2"}};
    bool healthy = false;
    RedisHealthMetrics reporter(
        registry,
        lifecycle,
        registry_manager_, indexer_manager_, [] { return true; },
        [&](const std::string &uri) { return uri.find("db=1") != std::string::npos || healthy; });
    for (bool state : {false, true, false}) {
        healthy = state;
        SetTargets(targets);
        reporter.Report();
        EXPECT_DOUBLE_EQ(state ? 1 : 0,
                         registry
                             ->GetGauge(RedisHealthMetrics::kMetricName,
                                        {{"host", "host"}, {"port", "6379"}, {"instance_group", "group"}})
                             .Get());
    }
    // Removing one dependency with the same public labels must retain the other.
    targets.erase({"group", "redis://host:6379/?db=2"});
    SetTargets(targets);
    reporter.Report();
    EXPECT_DOUBLE_EQ(1,
                     registry
                         ->GetGauge(RedisHealthMetrics::kMetricName,
                                    {{"host", "host"}, {"port", "6379"}, {"instance_group", "group"}})
                         .Get());
}

TEST_F(RedisHealthMetricsTest, BusySnapshotPreservesPublishedStateAndSkipsProbes) {
    auto registry = std::make_shared<MetricsRegistry>();
    auto lifecycle = std::make_shared<MetricsLifecycle>();
    RedisHealthMetrics::Targets targets = RedisHealthMetrics::Targets{{"group", "redis://host:6379"}};
    int calls = 0;
    RedisHealthMetrics reporter(
        registry,
        lifecycle,
        registry_manager_, indexer_manager_, [] { return true; },
        [&](const std::string &) {
            ++calls;
            return true;
        });
    SetTargets(targets);
    reporter.Report();
    const auto before = PrometheusExporter::Expose(*registry);
    {
        std::unique_lock lock(registry_manager_->mutex_);
        auto skipped = std::async(std::launch::async, [&]() { reporter.Report(); });
        const auto status = skipped.wait_for(std::chrono::seconds(1));
        lock.unlock();
        skipped.get();
        EXPECT_EQ(std::future_status::ready, status);
    }
    EXPECT_EQ(1, calls);
    EXPECT_EQ(before, PrometheusExporter::Expose(*registry));
    // A writer holding the lifecycle lock must not be waited on by the probe.
    std::unique_lock lock(lifecycle->mut_);
    auto skipped = std::async(std::launch::async, [&]() { reporter.Report(); });
    const auto status = skipped.wait_for(std::chrono::seconds(1));
    lock.unlock();
    skipped.get();
    EXPECT_EQ(std::future_status::ready, status);
    EXPECT_EQ(1, calls);
    EXPECT_EQ(before, PrometheusExporter::Expose(*registry));
}

TEST_F(RedisHealthMetricsTest, BusyBusinessLocksDoNotBlockTargetCollection) {
    auto metrics = std::make_shared<MetricsRegistry>();
    RegistryManager registry("", metrics);
    MetaIndexerManager indexers;
    for (auto *mutex : {&registry.mutex_, &indexers.mutex_}) {
        std::unique_lock lock(*mutex);
        auto collection =
            std::async(std::launch::async, [&]() { return RedisHealthMetrics::CollectTargets(registry, indexers); });
        const auto status = collection.wait_for(std::chrono::seconds(1));
        lock.unlock();
        const auto result = collection.get();
        EXPECT_EQ(std::future_status::ready, status);
        EXPECT_FALSE(result);
    }
}

TEST_F(RedisHealthMetricsTest, CancellationDoesNotWaitForProbeOrStartMoreTargets) {
    auto registry = std::make_shared<MetricsRegistry>();
    auto lifecycle = std::make_shared<MetricsLifecycle>();
    RedisHealthMetrics::Targets targets{{"a", "redis://host:6379"}, {"b", "redis://host:6380"}};
    std::promise<void> started, release;
    auto released = release.get_future();
    int calls = 0;
    RedisHealthMetrics reporter(
        registry,
        lifecycle,
        registry_manager_, indexer_manager_, [] { return true; },
        [&](const std::string &) {
            ++calls;
            started.set_value();
            released.wait();
            return true;
        });
    SetTargets(targets);
    auto report = std::async(std::launch::async, [&]() { reporter.Report(); });
    started.get_future().wait();
    // Simulate removal/demotion while network I/O is still stalled.
    std::unique_lock writer(lifecycle->mut_, std::try_to_lock);
    EXPECT_TRUE(writer.owns_lock());
    if (writer.owns_lock()) {
        writer.unlock();
    }
    reporter.RequestStop();
    EXPECT_EQ(std::string::npos, PrometheusExporter::Expose(*registry).find("redis_healthy_status"));
    release.set_value();
    report.get();
    SetTargets(targets);
    reporter.Report();
    EXPECT_EQ(1, calls);
    EXPECT_EQ(std::string::npos, PrometheusExporter::Expose(*registry).find("redis_healthy_status"));
}

TEST_F(RedisHealthMetricsTest, RuntimeSnapshotKeepsInitializationValues) {
    MetaIndexerManager indexers;
    auto config = std::make_shared<MetaIndexerConfig>();
    config->GetMetaStorageBackendConfig()->SetStorageType("local");
    ASSERT_EQ(EC_OK, indexers.CreateMetaIndexer("instance", config));
    config->GetMetaStorageBackendConfig()->SetStorageType("redis");
    config->GetMetaStorageBackendConfig()->SetStorageUri("redis://new:6379");
    const auto snapshot = indexers.TryGetStorageBackendConfigs();
    ASSERT_TRUE(snapshot);
    ASSERT_EQ(1, snapshot->size());
    EXPECT_EQ("local", snapshot->at("instance").first);
    EXPECT_TRUE(snapshot->at("instance").second.empty());
    indexers.DoCleanup();
    EXPECT_TRUE(indexers.TryGetStorageBackendConfigs()->empty());
}

TEST_F(RedisHealthMetricsTest, DemotionCancelsRemainingProbesAndRemovesSeries) {
    auto registry = std::make_shared<MetricsRegistry>();
    auto lifecycle = std::make_shared<MetricsLifecycle>();
    RedisHealthMetrics::Targets targets{
        {"a", "redis://host:6379"}, {"b", "redis://host:6380"}, {"c", "redis://host:6381"}};
    int probes = 0;
    bool active = true;
    bool demote = false;
    RedisHealthMetrics reporter(
        registry,
        lifecycle,
        registry_manager_, indexer_manager_, [&]() { return active; },
        [&](const std::string &) {
            ++probes;
            if (demote) {
                active = false;
            }
            return true;
        });
    SetTargets(targets);
    reporter.Report();
    EXPECT_EQ(3, probes);
    probes = 0;
    demote = true;
    SetTargets(targets);
    reporter.Report();
    EXPECT_EQ(1, probes);
    EXPECT_EQ(std::string::npos, PrometheusExporter::Expose(*registry).find("redis_healthy_status"));
}

TEST_F(RedisHealthMetricsTest, ConcreteSourcesAndDefaultProbeFollowLeadership) {
    auto registry = std::make_shared<MetricsRegistry>();
    auto lifecycle = std::make_shared<MetricsLifecycle>();
    ScriptedRedis server({"+PONG\r\n"});
    SetTargets({{"group", server.Uri()}});
    bool leader = false;
    RedisHealthMetrics reporter(registry, lifecycle, registry_manager_, indexer_manager_, [&] { return leader; });
    reporter.Report();
    EXPECT_EQ(std::string::npos, PrometheusExporter::Expose(*registry).find("redis_healthy_status"));
    leader = true;
    reporter.Report();
    server.Finish();
    EXPECT_EQ((std::vector<std::vector<std::string>>{{"PING"}}), server.commands);
    EXPECT_NE(std::string::npos, PrometheusExporter::Expose(*registry).find("instance_group=\"group\""));
    leader = false;
    reporter.Report();
    EXPECT_EQ(std::string::npos, PrometheusExporter::Expose(*registry).find("redis_healthy_status"));
}
