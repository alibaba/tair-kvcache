#include <arpa/inet.h>
#include <atomic>
#include <chrono>
#include <poll.h>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/metrics/metrics_registry.h"
#include "kv_cache_manager/metrics/prometheus_exporter.h"
#include "kv_cache_manager/mrc/kvcm_event_stream_client.h"
#include "kv_cache_manager/mrc/online_mrc_fact_registry.h"
#include "kv_cache_manager/optimizer/config/optimizer_registry_manager.h"
#include "kv_cache_manager/optimizer/liteHit/hit_curve.h"
#include "kv_cache_manager/optimizer/liteHit/lite_hit.h"
#include "kv_cache_manager/optimizer/liteHit/request_preprocess.h"
#include "kv_cache_manager/optimizer/online_runtime/online_optimizer_manager.h"

namespace kv_cache_manager {
namespace {

OnlineMrcConfig MakeConfig() {
    OnlineMrcConfig config;
    config.enable = true;
    config.capacity_gb_grid = {2};
    config.max_instances = 1;
    config.receiver_queue_max_batches = 4;
    config.discovery_refresh_interval_ms = 20;
    config.connect_timeout_ms = 100;
    config.reconnect_interval_ms = 20;
    config.max_frame_bytes = 1024 * 1024;
    return config;
}

std::vector<OptimizerInstanceGroup> MakeFormalGroups() {
    std::vector<OptimizerInstanceGroup> groups;
    for (const std::string &name : {"group_a", "group_b"}) {
        OptimizerInstanceGroup group;
        group.set_name(name);
        group.set_capacity_gb({2});
        group.set_enable_prefix_hash(true);
        groups.push_back(std::move(group));
    }
    return groups;
}

std::shared_ptr<OnlineOptimizerManager> MakeManager() {
    auto formal_registry = std::make_shared<OptimizerRegistryManager>("");
    if (!formal_registry->Init()) {
        return nullptr;
    }
    return std::make_shared<OnlineOptimizerManager>(formal_registry);
}

proto::optimizer::CacheEventBatch MakeBatch(int32_t block_size = 2) {
    proto::optimizer::CacheEventBatch batch;
    batch.set_instance_id("instance_a");
    batch.set_instance_group("group_a");
    auto *meta = batch.mutable_instance_meta();
    meta->set_block_size(block_size);
    auto *spec = meta->add_location_spec_infos();
    spec->set_name("full_spec");
    spec->set_size(1024LL * 1024 * 1024);
    auto *group = meta->add_location_spec_groups();
    group->set_name("full_group");
    group->add_spec_names("full_spec");
    meta->mutable_optimizer_state_info()->set_full_location_spec_group_name("full_group");

    auto *event = batch.add_events();
    event->set_event_id("event_a");
    event->set_timestamp_ns(1);
    event->set_input_token_len(static_cast<int64_t>(block_size) * 3);
    event->add_block_keys(1);
    event->add_block_keys(2);
    event->add_block_keys(3);
    return batch;
}

bool ReadAll(int fd, void *data, size_t size) {
    auto *bytes = static_cast<uint8_t *>(data);
    size_t received = 0;
    while (received < size) {
        const ssize_t rc = recv(fd, bytes + received, size - received, 0);
        if (rc <= 0) {
            return false;
        }
        received += static_cast<size_t>(rc);
    }
    return true;
}

bool WriteAll(int fd, const void *data, size_t size) {
    const auto *bytes = static_cast<const uint8_t *>(data);
    size_t written = 0;
    while (written < size) {
        const ssize_t rc = send(fd, bytes + written, size - written, 0);
        if (rc <= 0) {
            return false;
        }
        written += static_cast<size_t>(rc);
    }
    return true;
}

int CreateLoopbackListener(int &port) {
    const int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        return -1;
    }
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = 0;
    if (bind(fd, reinterpret_cast<sockaddr *>(&address), sizeof(address)) != 0 || listen(fd, 4) != 0) {
        close(fd);
        return -1;
    }
    socklen_t address_length = sizeof(address);
    if (getsockname(fd, reinterpret_cast<sockaddr *>(&address), &address_length) != 0) {
        close(fd);
        return -1;
    }
    port = ntohs(address.sin_port);
    return fd;
}

int AcceptWithTimeout(int listen_fd, int timeout_ms = 3000) {
    pollfd poll_fd{listen_fd, POLLIN, 0};
    if (poll(&poll_fd, 1, timeout_ms) <= 0 || !(poll_fd.revents & POLLIN)) {
        return -1;
    }
    return accept(listen_fd, nullptr, nullptr);
}

bool ReadHelloFrame(int fd) {
    uint32_t network_length = 0;
    if (!ReadAll(fd, &network_length, sizeof(network_length))) {
        return false;
    }
    std::vector<uint8_t> frame(ntohl(network_length));
    if (!ReadAll(fd, frame.data(), frame.size()) || frame.empty() || frame.front() != 1) {
        return false;
    }
    proto::optimizer::OptimizerHello hello;
    return hello.ParseFromArray(frame.data() + 1, static_cast<int>(frame.size() - 1)) &&
           hello.protocol_version() == 1 && !hello.optimizer_id().empty();
}

bool WriteBatchFrame(int fd, const proto::optimizer::CacheEventBatch &batch) {
    std::string payload;
    if (!batch.SerializeToString(&payload)) {
        return false;
    }
    const uint32_t event_length = htonl(static_cast<uint32_t>(payload.size() + 1));
    const uint8_t event_type = 2;
    return WriteAll(fd, &event_length, sizeof(event_length)) && WriteAll(fd, &event_type, sizeof(event_type)) &&
           WriteAll(fd, payload.data(), payload.size());
}

} // namespace

TEST(OnlineMrcFactRegistryTest, KeepsAllFactsAndSeparatesMetadataGenerations) {
    auto metrics = std::make_shared<MetricsRegistry>();
    auto manager = MakeManager();
    OnlineMrcFactRegistry registry(MakeConfig(), MakeFormalGroups(), metrics, manager);
    ASSERT_TRUE(registry.Init());

    ASSERT_TRUE(registry.Observe(MakeBatch()));
    ASSERT_TRUE(registry.Observe(MakeBatch()));
    ASSERT_TRUE(registry.Observe(MakeBatch()));
    EXPECT_EQ(3u, registry.FactCount("instance_a"));

    registry.ReportMetrics();
    auto hit_data = metrics->GetMetricsData("online_mrc.theoretical_hit_rate");
    ASSERT_NE(nullptr, hit_data);
    MetricsTags tags{
        {"instance_group", "group_a"}, {"instance_id", "instance_a"}, {"meta_generation", "1"}, {"capacity_gb", "2"}};
    auto hit = hit_data->GetGauge(tags);
    ASSERT_TRUE(hit.has_value());
    EXPECT_DOUBLE_EQ(4.0 / 9.0, hit->Get());

    MetricsTags instance_tags{
        {"instance_group", "group_a"}, {"instance_id", "instance_a"}, {"meta_generation", "1"}};
    auto fact_memory = metrics->GetMetricsData("online_mrc.fact_memory_bytes")->GetGauge(instance_tags);
    auto lite_hit_memory = metrics->GetMetricsData("online_mrc.lite_hit_memory_bytes")->GetGauge(instance_tags);
    auto instance_memory = metrics->GetMetricsData("online_mrc.instance_memory_bytes")->GetGauge(instance_tags);
    ASSERT_TRUE(fact_memory.has_value());
    ASSERT_TRUE(lite_hit_memory.has_value());
    ASSERT_TRUE(instance_memory.has_value());
    EXPECT_GT(fact_memory->Get(), 0.0);
    EXPECT_GT(lite_hit_memory->Get(), 0.0);
    EXPECT_DOUBLE_EQ(fact_memory->Get() + lite_hit_memory->Get(), instance_memory->Get());

    auto total_memory = metrics->GetMetricsData("online_mrc.total_memory_bytes")->GetGauge({});
    auto projection_duration = metrics->GetMetricsData("online_mrc.projection_duration_us")->GetGauge({});
    auto projection_scans = metrics->GetMetricsData("online_mrc.projection_fact_scans")->GetGauge({});
    ASSERT_TRUE(total_memory.has_value());
    ASSERT_TRUE(projection_duration.has_value());
    ASSERT_TRUE(projection_scans.has_value());
    EXPECT_DOUBLE_EQ(instance_memory->Get(), total_memory->Get());
    EXPECT_GE(projection_duration->Get(), 0.0);
    EXPECT_DOUBLE_EQ(3.0, projection_scans->Get());

    const std::string prometheus = PrometheusExporter::Expose(*metrics, "kvcm_optimizer");
    EXPECT_NE(std::string::npos, prometheus.find("kvcm_optimizer_online_mrc_fact_memory_bytes"));
    EXPECT_NE(std::string::npos, prometheus.find("kvcm_optimizer_online_mrc_lite_hit_memory_bytes"));
    EXPECT_NE(std::string::npos, prometheus.find("kvcm_optimizer_online_mrc_total_memory_bytes"));
    EXPECT_NE(std::string::npos, prometheus.find("kvcm_optimizer_online_mrc_projection_duration_us"));

    ASSERT_TRUE(registry.Observe(MakeBatch(/*block_size=*/4)));
    EXPECT_EQ(2u, registry.MetaGeneration("instance_a"));
    EXPECT_EQ(1u, registry.FactCount("instance_a"));
}

TEST(OnlineMrcFactRegistryTest, DropsOutOfOrderEventsWithinOneInstance) {
    auto metrics = std::make_shared<MetricsRegistry>();
    auto manager = MakeManager();
    OnlineMrcFactRegistry registry(MakeConfig(), MakeFormalGroups(), metrics, manager);
    ASSERT_TRUE(registry.Init());

    auto newest = MakeBatch();
    newest.mutable_events(0)->set_timestamp_ns(20);
    ASSERT_TRUE(registry.Observe(newest));

    auto stale = MakeBatch();
    stale.mutable_events(0)->set_timestamp_ns(19);
    ASSERT_TRUE(registry.Observe(stale));
    EXPECT_EQ(1u, registry.FactCount("instance_a"));

    registry.ReportMetrics();
    auto order_data = metrics->GetMetricsData("online_mrc.out_of_order_events");
    ASSERT_NE(nullptr, order_data);
    MetricsTags tags{{"instance_group", "group_a"}, {"instance_id", "instance_a"}, {"meta_generation", "1"}};
    auto out_of_order = order_data->GetGauge(tags);
    ASSERT_TRUE(out_of_order.has_value());
    EXPECT_DOUBLE_EQ(1.0, out_of_order->Get());
}

TEST(OnlineMrcFactRegistryTest, PrefixHashesKeysBeforeUpdatingLiteHit) {
    auto config = MakeConfig();
    config.capacity_gb_grid = {4};
    auto metrics = std::make_shared<MetricsRegistry>();
    auto manager = MakeManager();
    OnlineMrcFactRegistry registry(config, MakeFormalGroups(), metrics, manager);
    ASSERT_TRUE(registry.Init());

    const auto observe = [&](int64_t timestamp_ns, std::initializer_list<int64_t> raw_keys) {
        auto batch = MakeBatch();
        auto *event = batch.mutable_events(0);
        event->set_timestamp_ns(timestamp_ns);
        event->set_input_token_len(4);
        event->clear_block_keys();
        for (const int64_t key : raw_keys) {
            event->add_block_keys(key);
        }
        ASSERT_TRUE(registry.Observe(batch));
    };

    observe(1, {1, 2});
    observe(2, {3, 2});
    observe(3, {1, 4});
    observe(4, {1, 2});

    registry.ReportMetrics();
    auto hit_data = metrics->GetMetricsData("online_mrc.theoretical_hit_rate");
    ASSERT_NE(nullptr, hit_data);
    MetricsTags tags{
        {"instance_group", "group_a"}, {"instance_id", "instance_a"}, {"meta_generation", "1"}, {"capacity_gb", "4"}};
    auto hit = hit_data->GetGauge(tags);
    ASSERT_TRUE(hit.has_value());
    // Re-hashing the raw key sequence makes the second key depend on its
    // prefix. Without normalization, the second request would refresh key 2
    // under the wrong prefix and this value would be 3/8 instead of 2/8.
    EXPECT_DOUBLE_EQ(0.25, hit->Get());
}

TEST(OnlineMrcFactRegistryTest, MatchesFormalOfflineLiteHitOnTheSameCompleteTrace) {
    auto config = MakeConfig();
    auto metrics = std::make_shared<MetricsRegistry>();
    auto manager = MakeManager();
    OnlineMrcFactRegistry online(config, MakeFormalGroups(), metrics, manager);
    ASSERT_TRUE(online.Init());

    LiteHit offline;
    uint64_t offline_total_tokens = 0;
    uint64_t offline_hit_tokens = 0;
    const std::vector<std::vector<int64_t>> requests{{1, 2, 3}, {1, 2, 4}, {5, 2, 3}, {1, 2, 3}};
    int64_t timestamp_ns = 1;
    for (const auto &raw_keys : requests) {
        auto batch = MakeBatch();
        auto *event = batch.mutable_events(0);
        event->set_timestamp_ns(timestamp_ns++);
        event->clear_block_keys();
        for (const int64_t key : raw_keys) {
            event->add_block_keys(key);
        }
        ASSERT_TRUE(online.Observe(batch));

        const NormalizedRequest normalized = NormalizeRequest(raw_keys, 6, 2, true);
        const RequestFact fact = offline.ProcessRequest(normalized.block_keys);
        const uint64_t hit_blocks = HitCurveProjector::ProjectBytes(fact, 2ULL * 1024 * 1024 * 1024, 1024ULL * 1024 * 1024);
        offline_total_tokens += normalized.input_token_len;
        offline_hit_tokens += std::min<uint64_t>(hit_blocks * 2, normalized.input_token_len);
    }

    online.ReportMetrics();
    const auto hit_data = metrics->GetMetricsData("online_mrc.theoretical_hit_rate");
    ASSERT_NE(nullptr, hit_data);
    const MetricsTags tags{
        {"instance_group", "group_a"}, {"instance_id", "instance_a"}, {"meta_generation", "1"}, {"capacity_gb", "2"}};
    const auto online_hit = hit_data->GetGauge(tags);
    ASSERT_TRUE(online_hit.has_value());
    EXPECT_DOUBLE_EQ(static_cast<double>(offline_hit_tokens) / offline_total_tokens, online_hit->Get());
}

TEST(OnlineMrcFactRegistryTest, AutoCreatesGroupsAndMovesInstanceMembership) {
    auto manager = MakeManager();
    OnlineMrcFactRegistry registry(MakeConfig(), MakeFormalGroups(), nullptr, manager);
    ASSERT_TRUE(registry.Init());

    ASSERT_TRUE(registry.Observe(MakeBatch()));
    EXPECT_EQ(2u, registry.GroupCount());
    EXPECT_EQ(1u, registry.GroupInstanceCount("group_a"));
    const auto formal_group = manager->registry_manager()->GetInstanceGroup("group_a");
    ASSERT_NE(nullptr, formal_group);
    EXPECT_TRUE(formal_group->enable_prefix_hash());
    OptimizerInstanceGroup round_trip_group;
    ASSERT_TRUE(round_trip_group.FromJsonString(formal_group->ToJsonString()));
    EXPECT_TRUE(round_trip_group.enable_prefix_hash());
    EXPECT_EQ(EC_OK, manager->GetInstanceState("instance_a", [](const InstanceState &) {}));

    auto moved = MakeBatch();
    moved.set_instance_group("group_b");
    moved.mutable_events(0)->set_timestamp_ns(2);
    ASSERT_TRUE(registry.Observe(moved));
    EXPECT_EQ(2u, registry.GroupCount());
    EXPECT_EQ(0u, registry.GroupInstanceCount("group_a"));
    EXPECT_EQ(1u, registry.GroupInstanceCount("group_b"));
    EXPECT_EQ(2u, registry.MetaGeneration("instance_a"));
    EXPECT_EQ(1u, registry.FactCount("instance_a"));
}

TEST(OnlineMrcFactRegistryTest, CreatesFormalGroupsAtInitAndRejectsUnknownGroup) {
    auto manager = MakeManager();
    OnlineMrcFactRegistry registry(MakeConfig(), MakeFormalGroups(), nullptr, manager);
    ASSERT_TRUE(registry.Init());
    ASSERT_NE(nullptr, manager->registry_manager()->GetInstanceGroup("group_a"));
    ASSERT_NE(nullptr, manager->registry_manager()->GetInstanceGroup("group_b"));

    auto unknown = MakeBatch();
    unknown.set_instance_group("not_configured");
    EXPECT_FALSE(registry.Observe(unknown));
    EXPECT_EQ(nullptr, manager->registry_manager()->GetInstanceInfo("instance_a"));
}

TEST(OnlineMrcFactRegistryTest, RefusesToModifyAnExistingDifferentFormalGroup) {
    auto manager = MakeManager();
    OptimizerInstanceGroup existing;
    existing.set_name("group_a");
    existing.set_capacity_gb({99});
    existing.set_enable_prefix_hash(false);
    ASSERT_EQ(EC_OK, manager->CreateInstanceGroup(existing));

    OnlineMrcFactRegistry registry(MakeConfig(), MakeFormalGroups(), nullptr, manager);
    EXPECT_FALSE(registry.Init());
    const auto unchanged = manager->registry_manager()->GetInstanceGroup("group_a");
    ASSERT_NE(nullptr, unchanged);
    EXPECT_EQ((std::vector<double>{99}), unchanged->capacity_gb());
    EXPECT_FALSE(unchanged->enable_prefix_hash());
    EXPECT_EQ(nullptr, manager->registry_manager()->GetInstanceGroup("group_b"));
}

TEST(OnlineMrcFactRegistryTest, HotUpdatesCapacityGridWithoutResettingFacts) {
    auto metrics = std::make_shared<MetricsRegistry>();
    auto manager = MakeManager();
    OnlineMrcFactRegistry registry(MakeConfig(), MakeFormalGroups(), metrics, manager);
    ASSERT_TRUE(registry.Init());
    ASSERT_TRUE(registry.Observe(MakeBatch()));
    registry.ReportMetrics();

    auto hit_data = metrics->GetMetricsData("online_mrc.theoretical_hit_rate");
    ASSERT_NE(nullptr, hit_data);
    MetricsTags old_tags{
        {"instance_group", "group_a"}, {"instance_id", "instance_a"}, {"meta_generation", "1"}, {"capacity_gb", "2"}};
    ASSERT_TRUE(hit_data->GetGauge(old_tags).has_value());

    ASSERT_TRUE(registry.UpdateCapacityGrid({1, 4}));
    EXPECT_EQ((std::vector<double>{1, 4}), registry.CapacityGrid());
    EXPECT_EQ(2u, registry.ProjectionGeneration());
    EXPECT_EQ(1u, registry.FactCount("instance_a"));
    EXPECT_EQ(1u, registry.MetaGeneration("instance_a"));
    EXPECT_FALSE(hit_data->GetGauge(old_tags).has_value());

    registry.ReportMetrics();
    MetricsTags new_tags = old_tags;
    new_tags["capacity_gb"] = "4";
    EXPECT_TRUE(hit_data->GetGauge(new_tags).has_value());

    EXPECT_FALSE(registry.UpdateCapacityGrid({4, 1}));
    EXPECT_FALSE(registry.UpdateCapacityGrid({}));
    EXPECT_EQ(2u, registry.ProjectionGeneration());
    EXPECT_EQ((std::vector<double>{1, 4}), registry.CapacityGrid());
}

TEST(KvcmEventStreamClientTest, OptimizerDiscoversConnectsSendsHelloAndReceivesBatch) {
    const int listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    ASSERT_GE(listen_fd, 0);
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = 0;
    ASSERT_EQ(0, bind(listen_fd, reinterpret_cast<sockaddr *>(&address), sizeof(address)));
    ASSERT_EQ(0, listen(listen_fd, 1));
    socklen_t address_length = sizeof(address);
    ASSERT_EQ(0, getsockname(listen_fd, reinterpret_cast<sockaddr *>(&address), &address_length));
    const int port = ntohs(address.sin_port);

    std::atomic<bool> received_hello{false};
    std::thread server([&]() {
        const int client_fd = accept(listen_fd, nullptr, nullptr);
        if (client_fd < 0) {
            return;
        }
        uint32_t network_length = 0;
        if (!ReadAll(client_fd, &network_length, sizeof(network_length))) {
            close(client_fd);
            return;
        }
        std::vector<uint8_t> hello_frame(ntohl(network_length));
        if (!ReadAll(client_fd, hello_frame.data(), hello_frame.size()) || hello_frame.empty() ||
            hello_frame.front() != 1) {
            close(client_fd);
            return;
        }
        proto::optimizer::OptimizerHello hello;
        received_hello = hello.ParseFromArray(hello_frame.data() + 1, static_cast<int>(hello_frame.size() - 1)) &&
                         hello.protocol_version() == 1 && !hello.optimizer_id().empty();

        std::string payload;
        MakeBatch().SerializeToString(&payload);
        const uint32_t event_length = htonl(static_cast<uint32_t>(payload.size() + 1));
        const uint8_t event_type = 2;
        // Deliberately fragment the frame header to exercise partial reads.
        WriteAll(client_fd, &event_length, 2);
        WriteAll(client_fd, reinterpret_cast<const uint8_t *>(&event_length) + 2, 2);
        WriteAll(client_fd, &event_type, 1);
        WriteAll(client_fd, payload.data(), payload.size());
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        close(client_fd);
    });

    auto config = MakeConfig();
    config.kvcm_service_discovery_url = "static://127.0.0.1:" + std::to_string(port);
    auto metrics = std::make_shared<MetricsRegistry>();
    auto manager = MakeManager();
    auto facts = std::make_shared<OnlineMrcFactRegistry>(config, MakeFormalGroups(), metrics, manager);
    ASSERT_TRUE(facts->Init());
    KvcmEventStreamClient client(config, facts, metrics);
    ASSERT_TRUE(client.Init());
    ASSERT_TRUE(client.Start());
    for (int i = 0; i < 200 && facts->FactCount("instance_a") == 0; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    EXPECT_TRUE(received_hello.load());
    EXPECT_EQ(1u, facts->FactCount("instance_a"));
    client.ReportMetrics();
    const MetricsTags empty_tags;
    auto queue_bytes = metrics->GetMetricsData("online_mrc.receiver_queue_bytes")->GetGauge(empty_tags);
    auto queue_capacity =
        metrics->GetMetricsData("online_mrc.receiver_queue_capacity_batches")->GetGauge(empty_tags);
    ASSERT_TRUE(queue_bytes.has_value());
    ASSERT_TRUE(queue_capacity.has_value());
    EXPECT_DOUBLE_EQ(0.0, queue_bytes->Get());
    EXPECT_DOUBLE_EQ(4.0, queue_capacity->Get());
    client.Stop();
    server.join();
    close(listen_fd);
}

TEST(KvcmEventStreamClientTest, OneEventLoopConnectsToAllDiscoveredNodes) {
    int port_a = 0;
    int port_b = 0;
    const int listen_a = CreateLoopbackListener(port_a);
    const int listen_b = CreateLoopbackListener(port_b);
    ASSERT_GE(listen_a, 0);
    ASSERT_GE(listen_b, 0);

    std::atomic<int> hello_count{0};
    const auto serve = [&](int listen_fd, const std::string &event_id) {
        const int client_fd = AcceptWithTimeout(listen_fd);
        if (client_fd < 0) {
            return;
        }
        if (ReadHelloFrame(client_fd)) {
            hello_count.fetch_add(1);
            auto batch = MakeBatch();
            batch.mutable_events(0)->set_event_id(event_id);
            WriteBatchFrame(client_fd, batch);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        close(client_fd);
    };
    std::thread server_a(serve, listen_a, "event_a");
    std::thread server_b(serve, listen_b, "event_b");

    auto config = MakeConfig();
    config.kvcm_service_discovery_url = "static://127.0.0.1:" + std::to_string(port_a) + ",127.0.0.1:" +
                                        std::to_string(port_b);
    auto metrics = std::make_shared<MetricsRegistry>();
    auto manager = MakeManager();
    auto facts = std::make_shared<OnlineMrcFactRegistry>(config, MakeFormalGroups(), metrics, manager);
    ASSERT_TRUE(facts->Init());
    KvcmEventStreamClient client(config, facts, metrics);
    ASSERT_TRUE(client.Init());
    ASSERT_TRUE(client.Start());
    for (int i = 0; i < 500 && facts->FactCount("instance_a") < 2; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    EXPECT_EQ(2, hello_count.load());
    EXPECT_EQ(2u, facts->FactCount("instance_a"));
    EXPECT_EQ(2u, client.ConnectionCount());

    client.Stop();
    server_a.join();
    server_b.join();
    close(listen_a);
    close(listen_b);
}

TEST(KvcmEventStreamClientTest, EventLoopReconnectsWithoutPerEndpointThreads) {
    int port = 0;
    const int listen_fd = CreateLoopbackListener(port);
    ASSERT_GE(listen_fd, 0);

    std::atomic<int> hello_count{0};
    std::thread server([&]() {
        const int first_fd = AcceptWithTimeout(listen_fd);
        if (first_fd < 0) {
            return;
        }
        if (ReadHelloFrame(first_fd)) {
            hello_count.fetch_add(1);
        }
        close(first_fd);

        const int second_fd = AcceptWithTimeout(listen_fd);
        if (second_fd < 0) {
            return;
        }
        if (ReadHelloFrame(second_fd)) {
            hello_count.fetch_add(1);
            WriteBatchFrame(second_fd, MakeBatch());
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        close(second_fd);
    });

    auto config = MakeConfig();
    config.kvcm_service_discovery_url = "static://127.0.0.1:" + std::to_string(port);
    auto metrics = std::make_shared<MetricsRegistry>();
    auto manager = MakeManager();
    auto facts = std::make_shared<OnlineMrcFactRegistry>(config, MakeFormalGroups(), metrics, manager);
    ASSERT_TRUE(facts->Init());
    KvcmEventStreamClient client(config, facts, metrics);
    ASSERT_TRUE(client.Init());
    ASSERT_TRUE(client.Start());
    for (int i = 0; i < 1000 && facts->FactCount("instance_a") == 0; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    EXPECT_EQ(2, hello_count.load());
    EXPECT_EQ(1u, facts->FactCount("instance_a"));

    client.ReportMetrics();
    auto reconnects = metrics->GetMetricsData("online_mrc.reconnects")->GetGauge({});
    ASSERT_TRUE(reconnects.has_value());
    EXPECT_GE(reconnects->Get(), 1.0);

    client.Stop();
    server.join();
    close(listen_fd);
}

} // namespace kv_cache_manager
