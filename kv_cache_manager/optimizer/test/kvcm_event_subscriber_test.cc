#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <grpcpp/grpcpp.h>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "kv_cache_manager/common/static_service_discovery.h"
#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/metrics/metrics_registry.h"
#include "kv_cache_manager/optimizer/config/optimizer_registry_manager.h"
#include "kv_cache_manager/optimizer/manager/online_runtime/online_optimizer_manager.h"
#include "kv_cache_manager/optimizer/metrics/optimizer_metrics_reporter.h"
#include "kv_cache_manager/optimizer/service/event_subscriber/kvcm_event_subscriber.h"
#include "kv_cache_manager/optimizer/service/online_optimizer_server_config.h"
#include "kv_cache_manager/optimizer/service/optimizer_service_impl.h"
#include "kv_cache_manager/protocol/protobuf/meta_service.grpc.pb.h"
#include "kv_cache_manager/protocol/protobuf/optimizer_service.grpc.pb.h"

namespace kv_cache_manager {

namespace {

template <typename Predicate>
bool WaitUntil(Predicate predicate, int timeout_ms = 3000) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (std::chrono::steady_clock::now() < deadline) {
        if (predicate()) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    return predicate();
}

proto::optimizer::TraceQueryRequest
MakeEvent(const std::string &instance_id, const std::string &trace_id, int64_t key) {
    proto::optimizer::TraceQueryRequest event;
    event.set_instance_id(instance_id);
    event.set_block_size(4);
    event.set_trace_id(trace_id);
    event.set_input_token_len(4);
    event.set_timestamp_ns(1000LL * 1000000000);
    event.add_block_keys(key);
    return event;
}

proto::optimizer::KvcmConfigurationResponse MakeConfiguration(const std::vector<std::string> &instance_ids) {
    proto::optimizer::KvcmConfigurationResponse response;
    auto *group = response.add_instance_groups();
    group->set_name("g1");
    group->set_capacity_bytes(2LL * 1024 * 1024 * 1024);
    for (const auto &instance_id : instance_ids) {
        auto *instance = response.add_instances();
        instance->set_instance_group_name("g1");
        instance->set_instance_id(instance_id);
        instance->set_block_size(4);
        auto *spec = instance->add_location_spec_infos();
        spec->set_name("tp0");
        spec->set_size(16);
        auto *spec_group = instance->add_location_spec_groups();
        spec_group->set_name("full_cache");
        spec_group->add_spec_names("tp0");
    }
    return response;
}

class TestMetaService final : public proto::meta::MetaService::Service {
public:
    void SetLeader(const std::string &host, int port) {
        std::lock_guard<std::mutex> lock(mutex_);
        leader_host_ = host;
        leader_port_ = port;
    }

    grpc::Status GetClusterInfo(grpc::ServerContext *,
                                const proto::meta::GetClusterInfoRequest *,
                                proto::meta::GetClusterInfoResponse *response) override {
        std::lock_guard<std::mutex> lock(mutex_);
        response->mutable_header()->mutable_status()->set_code(proto::meta::OK);
        response->set_self_node_id("seed");
        response->set_leader_node_id("leader");
        auto *endpoint = response->mutable_leader_endpoint();
        endpoint->set_node_id("leader");
        endpoint->set_host(leader_host_);
        endpoint->set_meta_rpc_port(leader_port_);
        return grpc::Status::OK;
    }

private:
    std::mutex mutex_;
    std::string leader_host_;
    int leader_port_ = 0;
};

class TestEventService final : public proto::optimizer::OptimizerEventStreamService::Service {
public:
    explicit TestEventService(bool keep_open) : keep_open_(keep_open) {}

    void SetConfiguration(const proto::optimizer::KvcmConfigurationResponse &configuration) {
        std::lock_guard<std::mutex> lock(mutex_);
        configuration_.CopyFrom(configuration);
    }

    void SetConfigurationAvailable(bool available) { configuration_available_.store(available); }

    void Publish(const proto::optimizer::TraceQueryRequest &event) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            events_.push_back(event);
        }
        cv_.notify_all();
    }

    grpc::Status GetConfiguration(grpc::ServerContext *,
                                  const proto::optimizer::KvcmConfigurationRequest *,
                                  proto::optimizer::KvcmConfigurationResponse *response) override {
        configuration_count_.fetch_add(1);
        std::lock_guard<std::mutex> lock(mutex_);
        response->CopyFrom(configuration_);
        response->mutable_header()->mutable_status()->set_code(
            configuration_available_.load() ? proto::optimizer::OK : proto::optimizer::SERVICE_NOT_READY);
        return grpc::Status::OK;
    }

    grpc::Status SubscribeEvents(grpc::ServerContext *context,
                                 const proto::optimizer::OptimizerEventSubscriptionRequest *request,
                                 grpc::ServerWriter<proto::optimizer::TraceQueryRequest> *writer) override {
        received_expected_consumer_id_.store(request->consumer_id() == "subscriber-test");
        subscribe_count_.fetch_add(1);
        if (!keep_open_) {
            return grpc::Status::OK;
        }

        active_subscribers_.fetch_add(1);
        while (!context->IsCancelled()) {
            proto::optimizer::TraceQueryRequest event;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait_for(lock, std::chrono::milliseconds(5), [this] { return !events_.empty(); });
                if (events_.empty()) {
                    continue;
                }
                event = std::move(events_.front());
                events_.pop_front();
            }
            if (!writer->Write(event)) {
                break;
            }
            delivered_count_.fetch_add(1);
        }
        active_subscribers_.fetch_sub(1);
        return grpc::Status::OK;
    }

    std::atomic<int> configuration_count_{0};
    std::atomic<int> subscribe_count_{0};
    std::atomic<int> active_subscribers_{0};
    std::atomic<int> delivered_count_{0};
    std::atomic<bool> received_expected_consumer_id_{false};
    std::atomic<bool> configuration_available_{true};

private:
    const bool keep_open_;
    std::mutex mutex_;
    std::condition_variable cv_;
    proto::optimizer::KvcmConfigurationResponse configuration_;
    std::deque<proto::optimizer::TraceQueryRequest> events_;
};

class TestKvcmServer {
public:
    explicit TestKvcmServer(bool keep_stream_open = true) : event_service_(keep_stream_open) {
        static std::atomic<int> next_seed_id{1};
        seed_ip_ = "127.0.10." + std::to_string(next_seed_id.fetch_add(1));
        grpc::ServerBuilder builder;
        builder.AddListeningPort(seed_ip_ + ":6381", grpc::InsecureServerCredentials());
        builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port_);
        builder.RegisterService(&meta_service_);
        builder.RegisterService(&event_service_);
        server_ = builder.BuildAndStart();
        EXPECT_NE(nullptr, server_);
        meta_service_.SetLeader("127.0.0.1", port_);
    }

    ~TestKvcmServer() {
        if (server_) {
            server_->Shutdown();
        }
    }

    // Advertise an HTTP port, while GetClusterInfo returns the actual dynamic Leader RPC port.
    std::string endpoint() const { return seed_ip_ + ":6382"; }
    std::string rpc_endpoint() const { return "127.0.0.1:" + std::to_string(port_); }
    int port() const { return port_; }

    TestMetaService meta_service_;
    TestEventService event_service_;

private:
    std::string seed_ip_;
    int port_ = 0;
    std::unique_ptr<grpc::Server> server_;
};

KvcmEventSubscriptionConfig MakeConfig(const std::string &seed_endpoint, int refresh_interval_ms = 50) {
    OnlineOptimizerServerConfig server_config;
    EXPECT_TRUE(server_config.FromJsonString(
        std::string(R"({"kvcm_event_subscriptions":[{"ttl_seconds_list":[86400],"service_discovery_url":"static://)") +
        seed_endpoint + R"(","consumer_id":"subscriber-test","discovery_refresh_interval_ms":)" +
        std::to_string(refresh_interval_ms) + "}]}"));
    return server_config.kvcm_event_subscriptions().front();
}

KvcmEventSubscriptionConfig MakeMultiCapacityConfig(const std::string &seed_endpoint) {
    OnlineOptimizerServerConfig server_config;
    EXPECT_TRUE(server_config.FromJsonString(
        std::string(R"({"kvcm_event_subscriptions":[{"ttl_seconds_list":[86400],"service_discovery_url":"static://)") +
        seed_endpoint +
        R"(","consumer_id":"subscriber-test","discovery_refresh_interval_ms":50,"capacity_gb":[1.0,2.0,4.0]}]})"));
    return server_config.kvcm_event_subscriptions().front();
}

} // namespace

class KvcmEventSubscriberTest : public TESTBASE {
protected:
    void SetUp() override {
        registry_ = std::make_shared<OptimizerRegistryManager>("");
        ASSERT_TRUE(registry_->Init());
        manager_ = std::make_shared<OnlineOptimizerManager>(registry_);
        metrics_registry_ = std::make_shared<MetricsRegistry>();
        metrics_reporter_ = std::make_shared<OptimizerMetricsReporter>(manager_, metrics_registry_);
        optimizer_service_ = std::make_shared<OptimizerServiceImpl>(manager_, metrics_reporter_);
    }

    bool IsRegistered(const std::string &instance_id, int32_t block_size = 4) const {
        return manager_->GetInstanceState(OptimizerServiceImpl::KvcmAnalysisInstanceId(instance_id, block_size),
                                          [](const InstanceState &) {}) == EC_OK;
    }

    int64_t TotalQueries(const std::string &instance_id, int32_t block_size = 4) const {
        std::vector<InstanceSummary> summaries;
        EXPECT_EQ(EC_OK, manager_->ListInstances("", summaries));
        for (const auto &summary : summaries) {
            if (summary.instance_id == OptimizerServiceImpl::KvcmAnalysisInstanceId(instance_id, block_size)) {
                return summary.total_queries;
            }
        }
        return -1;
    }

    std::shared_ptr<OptimizerRegistryManager> registry_;
    std::shared_ptr<OnlineOptimizerManager> manager_;
    std::shared_ptr<MetricsRegistry> metrics_registry_;
    std::shared_ptr<OptimizerMetricsReporter> metrics_reporter_;
    std::shared_ptr<OptimizerServiceImpl> optimizer_service_;
};

TEST_F(KvcmEventSubscriberTest, DiscoversLeaderViaFixedRpcPortWhenDiscoveryAdvertisesHttp) {
    TestKvcmServer seed;
    TestKvcmServer leader;
    seed.meta_service_.SetLeader("127.0.0.1", leader.port());
    leader.event_service_.SetConfiguration(MakeConfiguration({"known"}));

    KvcmEventSubscriber subscriber(
        MakeConfig(seed.endpoint()), optimizer_service_, metrics_registry_, metrics_reporter_);
    ASSERT_TRUE(subscriber.Init());
    ASSERT_TRUE(subscriber.Start());

    ASSERT_TRUE(WaitUntil([this] { return IsRegistered("known"); }));
    ASSERT_TRUE(WaitUntil([&leader] { return leader.event_service_.active_subscribers_.load() == 1; }));
    EXPECT_EQ(0, seed.event_service_.subscribe_count_.load());
    EXPECT_TRUE(leader.event_service_.received_expected_consumer_id_.load());

    auto group = registry_->GetInstanceGroup("g1");
    ASSERT_NE(nullptr, group);
    EXPECT_TRUE(group->capacity_gb().empty());
    EXPECT_TRUE(group->enable_prefix_hash());
    EXPECT_TRUE(group->enable_theoretical_max_cache());
    EXPECT_EQ(24 * 60 * 60, group->ttl_seconds_list().front());

    leader.event_service_.Publish(MakeEvent("known", "normal", 1));
    auto short_prompt = MakeEvent("known", "short", 2);
    short_prompt.clear_block_keys();
    short_prompt.set_input_token_len(3);
    leader.event_service_.Publish(short_prompt);
    ASSERT_TRUE(WaitUntil([this] { return TotalQueries("known") == 2; }));

    subscriber.Stop();
    EXPECT_FALSE(subscriber.running_);
    EXPECT_EQ(nullptr, subscriber.worker_);
}

TEST_F(KvcmEventSubscriberTest, BlockSizeChangeRetiresStateAndRejectsMismatchedEvents) {
    TestKvcmServer leader;
    leader.event_service_.SetConfiguration(MakeConfiguration({"known"}));
    KvcmEventSubscriber subscriber(
        MakeConfig(leader.endpoint()), optimizer_service_, metrics_registry_, metrics_reporter_);
    ASSERT_TRUE(subscriber.Init());
    ASSERT_TRUE(subscriber.SyncConfiguration(leader.rpc_endpoint()));
    auto event = MakeEvent("known", "old", 1);
    subscriber.ProcessEvent(event, "127.0.0.1");
    subscriber.ProcessEvent(event, "127.0.0.1");
    EXPECT_EQ(2, TotalQueries("known"));
    const MetricsTags old_tags = {{"instance_group", "g1"}, {"instance_id", "known_bs4"}};
    EXPECT_EQ(2u, metrics_registry_->GetCounter("service.query_counter", old_tags).Get());

    auto next_event = event;
    next_event.set_block_size(8);
    next_event.set_input_token_len(8);
    subscriber.ProcessEvent(next_event, "127.0.0.1");
    EXPECT_TRUE(subscriber.configuration_refresh_requested_);
    EXPECT_EQ(2, TotalQueries("known"));
    auto configuration = MakeConfiguration({"known"});
    configuration.mutable_instances(0)->set_block_size(8);
    leader.event_service_.SetConfiguration(configuration);
    ASSERT_TRUE(subscriber.SyncConfiguration(leader.rpc_endpoint()));
    EXPECT_FALSE(IsRegistered("known", 4));
    ASSERT_TRUE(IsRegistered("known", 8));
    EXPECT_EQ(0u, metrics_registry_->GetCounter("service.query_counter", old_tags).Get());
    subscriber.ProcessEvent(next_event, "127.0.0.1");
    EXPECT_EQ(1, TotalQueries("known", 8));
    // Delayed events from the previous layout must not be replayed into the new state.
    subscriber.ProcessEvent(event, "127.0.0.1");
    EXPECT_EQ(1, TotalQueries("known", 8));
    EXPECT_EQ(0u, metrics_registry_->GetCounter("service.query_counter", old_tags).Get());
    next_event.clear_block_size();
    subscriber.ProcessEvent(next_event, "127.0.0.1");
    EXPECT_EQ(1, TotalQueries("known", 8));
    const MetricsTags hit_tags = {
        {"instance_group", "g1"}, {"instance_id", "known_bs8"}, {"ttl", "86400"}, {"client_ip", "127.0.0.1"}};
    EXPECT_DOUBLE_EQ(0.0, metrics_registry_->GetGauge("query_max_hit_count", hit_tags).Get());
    leader.event_service_.SetConfiguration(MakeConfiguration({"known"}));
    ASSERT_TRUE(subscriber.SyncConfiguration(leader.rpc_endpoint()));
    EXPECT_FALSE(IsRegistered("known", 8));
    EXPECT_EQ(0, TotalQueries("known", 4));
    subscriber.ProcessEvent(event, "127.0.0.1");
    EXPECT_EQ(1, TotalQueries("known", 4));
}

TEST_F(KvcmEventSubscriberTest, InstanceConflictDoesNotBlockOtherInstancesOrLeaderMigration) {
    TestKvcmServer first;
    TestKvcmServer second;
    first.event_service_.SetConfiguration(MakeConfiguration({"conflict", "healthy"}));
    KvcmEventSubscriber subscriber(
        MakeConfig(first.endpoint()), optimizer_service_, metrics_registry_, metrics_reporter_);
    ASSERT_TRUE(subscriber.Init());
    subscriber.running_ = true;
    subscriber.RefreshLeader();
    ASSERT_TRUE(WaitUntil([&] { return first.event_service_.active_subscribers_ == 1; }));
    auto configuration = MakeConfiguration({"conflict", "invalid", "healthy", "new"});
    configuration.add_instance_groups()->set_name("other");
    configuration.mutable_instances(0)->set_instance_group_name("other");
    configuration.mutable_instances(1)->set_block_size(0);
    second.event_service_.SetConfiguration(configuration);
    first.meta_service_.SetLeader("127.0.0.1", second.port());
    subscriber.RefreshLeader();
    ASSERT_TRUE(WaitUntil([&] { return second.event_service_.active_subscribers_ == 1; }));
    EXPECT_EQ(0, first.event_service_.active_subscribers_);
    EXPECT_TRUE(IsRegistered("new"));
    subscriber.ProcessEvent(MakeEvent("conflict", "blocked", 1), "127.0.0.1");
    EXPECT_EQ(0, TotalQueries("conflict"));
    EXPECT_FALSE(subscriber.configuration_refresh_requested_);
    subscriber.ProcessEvent(MakeEvent("healthy", "ok", 1), "127.0.0.1");
    subscriber.ProcessEvent(MakeEvent("new", "ok", 1), "127.0.0.1");
    EXPECT_EQ(1, TotalQueries("healthy"));
    EXPECT_EQ(1, TotalQueries("new"));
    subscriber.Stop();
}

TEST_F(KvcmEventSubscriberTest, DiscoveryRequiresHealthyIpButNotAdvertisedHost) {
    TestKvcmServer seed;
    KvcmEventSubscriber subscriber(
        MakeConfig(seed.endpoint()), optimizer_service_, metrics_registry_, metrics_reporter_);
    std::vector<ServiceEndpoint> endpoints;
    ASSERT_TRUE(StaticServiceDiscovery::ParseHostPortList(seed.endpoint(), endpoints));
    const auto healthy_endpoint = endpoints.front();
    auto discovery = std::make_unique<StaticServiceDiscovery>();
    auto *discovery_ptr = discovery.get();
    subscriber.service_discovery_ = std::move(discovery);
    std::string leader_endpoint;

    endpoints.front().healthy = false;
    ASSERT_TRUE(discovery_ptr->InitWithEndpoints(endpoints));
    EXPECT_FALSE(subscriber.DiscoverLeader(leader_endpoint));

    endpoints.front() = healthy_endpoint;
    endpoints.front().ip.clear();
    ASSERT_TRUE(discovery_ptr->InitWithEndpoints(endpoints));
    EXPECT_FALSE(subscriber.DiscoverLeader(leader_endpoint));

    endpoints.front() = healthy_endpoint;
    endpoints.front().host.clear();
    ASSERT_TRUE(discovery_ptr->InitWithEndpoints(endpoints));
    ASSERT_TRUE(subscriber.DiscoverLeader(leader_endpoint));
    EXPECT_EQ("127.0.0.1:" + std::to_string(seed.port()), leader_endpoint);
}

TEST_F(KvcmEventSubscriberTest, AppliesConfiguredCapacityTiers) {
    TestKvcmServer leader;
    leader.event_service_.SetConfiguration(MakeConfiguration({"known"}));

    KvcmEventSubscriber subscriber(
        MakeMultiCapacityConfig(leader.endpoint()), optimizer_service_, metrics_registry_, metrics_reporter_);
    ASSERT_TRUE(subscriber.Init());
    ASSERT_TRUE(subscriber.Start());

    ASSERT_TRUE(WaitUntil([this] { return IsRegistered("known"); }));
    auto group = registry_->GetInstanceGroup("g1");
    ASSERT_NE(nullptr, group);
    EXPECT_EQ((std::vector<double>{1.0, 2.0, 4.0}), group->capacity_gb());

    leader.event_service_.Publish(MakeEvent("known", "multi-capacity", 1));
    ASSERT_TRUE(WaitUntil([this] { return TotalQueries("known") == 1; }));
    subscriber.Stop();
}

TEST_F(KvcmEventSubscriberTest, UnknownInstanceTriggersImmediateConfigurationRefresh) {
    TestKvcmServer leader;
    leader.event_service_.SetConfiguration(MakeConfiguration({"known"}));
    KvcmEventSubscriber subscriber(
        MakeConfig(leader.endpoint(), 5000), optimizer_service_, metrics_registry_, metrics_reporter_);
    ASSERT_TRUE(subscriber.Init());
    ASSERT_TRUE(subscriber.Start());
    ASSERT_TRUE(WaitUntil([this] { return IsRegistered("known"); }));
    ASSERT_TRUE(WaitUntil([&leader] { return leader.event_service_.active_subscribers_.load() == 1; }));
    const int initial_refreshes = leader.event_service_.configuration_count_.load();

    leader.event_service_.SetConfiguration(MakeConfiguration({"known", "new-instance"}));
    leader.event_service_.Publish(MakeEvent("new-instance", "triggers-refresh", 1));
    ASSERT_TRUE(WaitUntil([&leader, initial_refreshes] {
        return leader.event_service_.configuration_count_.load() > initial_refreshes;
    }));
    ASSERT_TRUE(WaitUntil([this] { return IsRegistered("new-instance"); }));

    leader.event_service_.Publish(MakeEvent("new-instance", "accepted", 2));
    ASSERT_TRUE(WaitUntil([this] { return TotalQueries("new-instance") == 1; }));
    const MetricsTags unknown_instance_tags = {{"instance_group", ""}, {"instance_id", "new-instance_bs4"}};
    const MetricsTags known_instance_tags = {{"instance_group", "g1"}, {"instance_id", "new-instance_bs4"}};
    ASSERT_TRUE(WaitUntil([this, &unknown_instance_tags, &known_instance_tags] {
        return metrics_registry_->GetCounter("service.query_counter").Get() == 2u &&
               metrics_registry_->GetCounter("service.error_counter").Get() == 1u &&
               metrics_registry_->GetCounter("service.query_counter", unknown_instance_tags).Get() == 1u &&
               metrics_registry_->GetCounter("service.error_counter", unknown_instance_tags).Get() == 1u &&
               metrics_registry_->GetCounter("service.query_counter", known_instance_tags).Get() == 1u &&
               metrics_registry_->GetCounter("service.error_counter", known_instance_tags).Get() == 0u;
    }));
    subscriber.Stop();
}

TEST_F(KvcmEventSubscriberTest, ReportsQueryMetricsForStreamEvents) {
    TestKvcmServer leader;
    leader.event_service_.SetConfiguration(MakeConfiguration({"known"}));
    KvcmEventSubscriber subscriber(
        MakeMultiCapacityConfig(leader.endpoint()), optimizer_service_, metrics_registry_, metrics_reporter_);
    ASSERT_TRUE(subscriber.Init());
    ASSERT_TRUE(subscriber.Start());
    ASSERT_TRUE(WaitUntil([this] { return IsRegistered("known"); }));
    ASSERT_TRUE(WaitUntil([&leader] { return leader.event_service_.active_subscribers_.load() == 1; }));

    const MetricsTags base_tags = {
        {"instance_group", "g1"}, {"instance_id", "known_bs4"}, {"ttl", "86400"}, {"client_ip", "127.0.0.1"}};
    const MetricsTags capacity_tags = {{"instance_group", "g1"},
                                       {"instance_id", "known_bs4"},
                                       {"ttl", "86400"},
                                       {"client_ip", "127.0.0.1"},
                                       {"capacity_gb", std::to_string(2.0)}};

    leader.event_service_.Publish(MakeEvent("known", "miss", 1));
    ASSERT_TRUE(WaitUntil(
        [this, &base_tags] { return metrics_registry_->GetGauge("query_total_blocks", base_tags).Get() == 1.0; }));
    EXPECT_DOUBLE_EQ(0.0, metrics_registry_->GetGauge("query_hit_count", capacity_tags).Get());
    EXPECT_DOUBLE_EQ(0.0, metrics_registry_->GetGauge("query_hit_rate", capacity_tags).Get());

    leader.event_service_.Publish(MakeEvent("known", "hit", 1));
    ASSERT_TRUE(WaitUntil(
        [this, &capacity_tags] { return metrics_registry_->GetGauge("query_hit_count", capacity_tags).Get() == 1.0; }));
    EXPECT_DOUBLE_EQ(1.0, metrics_registry_->GetGauge("query_hit_rate", capacity_tags).Get());
    const MetricsTags service_instance_tags = {{"instance_group", "g1"}, {"instance_id", "known_bs4"}};
    EXPECT_EQ(2u, metrics_registry_->GetCounter("service.query_counter", service_instance_tags).Get());
    EXPECT_EQ(0u, metrics_registry_->GetCounter("service.error_counter", service_instance_tags).Get());
    EXPECT_GE(metrics_registry_->GetGauge("service.query_rt_us", service_instance_tags).Get(), 0.0);

    subscriber.Stop();
}

TEST_F(KvcmEventSubscriberTest, MovesTheOnlyStreamWhenLeaderChanges) {
    TestKvcmServer first;
    TestKvcmServer second;
    first.event_service_.SetConfiguration(MakeConfiguration({"known"}));
    second.event_service_.SetConfiguration(MakeConfiguration({"known"}));

    KvcmEventSubscriber subscriber(
        MakeConfig(first.endpoint()), optimizer_service_, metrics_registry_, metrics_reporter_);
    ASSERT_TRUE(subscriber.Init());
    ASSERT_TRUE(subscriber.Start());
    ASSERT_TRUE(WaitUntil([&first] { return first.event_service_.active_subscribers_.load() == 1; }));
    first.event_service_.Publish(MakeEvent("known", "first-leader", 1));
    ASSERT_TRUE(WaitUntil([this] { return TotalQueries("known") == 1; }));

    first.meta_service_.SetLeader("127.0.0.1", second.port());
    ASSERT_TRUE(WaitUntil([&first, &second] {
        return first.event_service_.active_subscribers_.load() == 0 &&
               second.event_service_.active_subscribers_.load() == 1;
    }));
    EXPECT_EQ(1, first.event_service_.subscribe_count_.load());
    second.event_service_.Publish(MakeEvent("known", "second-leader", 2));
    ASSERT_TRUE(WaitUntil([this] { return TotalQueries("known") == 2; }));
    subscriber.Stop();
}

TEST_F(KvcmEventSubscriberTest, KeepsOldStreamUntilNewLeaderConfigurationSucceeds) {
    TestKvcmServer first;
    TestKvcmServer second;
    first.event_service_.SetConfiguration(MakeConfiguration({"known"}));
    second.event_service_.SetConfiguration(MakeConfiguration({"known"}));
    second.event_service_.SetConfigurationAvailable(false);

    KvcmEventSubscriber subscriber(
        MakeConfig(first.endpoint()), optimizer_service_, metrics_registry_, metrics_reporter_);
    ASSERT_TRUE(subscriber.Init());
    ASSERT_TRUE(subscriber.Start());
    ASSERT_TRUE(WaitUntil([&first] { return first.event_service_.active_subscribers_.load() == 1; }));

    first.meta_service_.SetLeader("127.0.0.1", second.port());
    ASSERT_TRUE(WaitUntil([&second] { return second.event_service_.configuration_count_.load() > 0; }));
    EXPECT_EQ(1, first.event_service_.active_subscribers_.load());
    EXPECT_EQ(0, second.event_service_.subscribe_count_.load());

    first.event_service_.Publish(MakeEvent("known", "old-leader-still-active", 1));
    ASSERT_TRUE(WaitUntil([this] { return TotalQueries("known") == 1; }));

    second.event_service_.SetConfigurationAvailable(true);
    ASSERT_TRUE(WaitUntil([&first, &second] {
        return first.event_service_.active_subscribers_.load() == 0 &&
               second.event_service_.active_subscribers_.load() == 1;
    }));
    subscriber.Stop();
}

TEST_F(KvcmEventSubscriberTest, AutoRegistersWithoutSpecInterpretation) {
    TestKvcmServer leader;
    auto configuration = MakeConfiguration({});
    auto *instance = configuration.add_instances();
    instance->set_instance_group_name("g1");
    instance->set_instance_id("ambiguous");
    instance->set_block_size(4);
    for (const auto *name : {"state-a", "state-b"}) {
        auto *spec = instance->add_location_spec_infos();
        spec->set_name(name);
        spec->set_size(16);
        auto *spec_group = instance->add_location_spec_groups();
        spec_group->set_name(std::string("group-") + name);
        spec_group->add_spec_names(name);
    }
    leader.event_service_.SetConfiguration(configuration);

    KvcmEventSubscriber subscriber(
        MakeConfig(leader.endpoint(), 5000), optimizer_service_, metrics_registry_, metrics_reporter_);
    ASSERT_TRUE(subscriber.Init());
    ASSERT_TRUE(subscriber.Start());
    ASSERT_TRUE(WaitUntil([&leader] { return leader.event_service_.active_subscribers_.load() == 1; }));
    EXPECT_TRUE(IsRegistered("ambiguous"));
    const int configuration_count = leader.event_service_.configuration_count_.load();

    leader.event_service_.Publish(MakeEvent("ambiguous", "unsupported", 1));
    ASSERT_TRUE(WaitUntil([&leader] { return leader.event_service_.delivered_count_.load() == 1; }));
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    EXPECT_EQ(configuration_count, leader.event_service_.configuration_count_.load());
    subscriber.Stop();
}

TEST_F(KvcmEventSubscriberTest, UnsupportedInstancesDoNotRequestRefreshAndRecoverAfterConfigurationChange) {
    TestKvcmServer leader;
    auto configuration = MakeConfiguration({"ambiguous"});
    auto *extra_group = configuration.mutable_instances(0)->add_location_spec_groups();
    extra_group->set_name("linear");
    extra_group->add_spec_names("tp0");
    leader.event_service_.SetConfiguration(configuration);

    KvcmEventSubscriptionConfig config;
    ASSERT_TRUE(config.FromJsonString(std::string(R"({"mock_cache_layout":false,"service_discovery_url":"static://)") +
                                      leader.endpoint() + R"(","discovery_refresh_interval_ms":5000})"));
    KvcmEventSubscriber subscriber(config, optimizer_service_, metrics_registry_, metrics_reporter_);
    ASSERT_TRUE(subscriber.Init());
    // Drive sync and event consumption synchronously so the supervisor cannot
    // clear the refresh flag before the assertion.
    ASSERT_TRUE(subscriber.SyncConfiguration(leader.rpc_endpoint()));
    EXPECT_FALSE(IsRegistered("ambiguous"));
    for (int i = 0; i < 100; ++i) {
        subscriber.ProcessEvent(MakeEvent("ambiguous", "unsupported", i), "127.0.0.1");
        EXPECT_FALSE(subscriber.configuration_refresh_requested_);
    }
    EXPECT_EQ(1, leader.event_service_.configuration_count_.load());

    // A failed sync must not discard the last successful unsupported set.
    leader.event_service_.SetConfigurationAvailable(false);
    EXPECT_FALSE(subscriber.SyncConfiguration(leader.rpc_endpoint()));
    subscriber.ProcessEvent(MakeEvent("ambiguous", "failed-sync", 1), "127.0.0.1");
    EXPECT_FALSE(subscriber.configuration_refresh_requested_);
    leader.event_service_.SetConfigurationAvailable(true);

    // A successful replacement snapshot removes stale exclusions.
    leader.event_service_.SetConfiguration(MakeConfiguration({}));
    ASSERT_TRUE(subscriber.SyncConfiguration(leader.rpc_endpoint()));
    subscriber.ProcessEvent(MakeEvent("ambiguous", "now-unknown", 1), "127.0.0.1");
    EXPECT_TRUE(subscriber.configuration_refresh_requested_);

    leader.event_service_.SetConfiguration(MakeConfiguration({"ambiguous"}));
    ASSERT_TRUE(subscriber.SyncConfiguration(leader.rpc_endpoint()));
    ASSERT_TRUE(IsRegistered("ambiguous"));
    subscriber.ProcessEvent(MakeEvent("ambiguous", "now-supported", 1), "127.0.0.1");
    EXPECT_EQ(1, TotalQueries("ambiguous"));
}

TEST_F(KvcmEventSubscriberTest, ReconnectsClosedLeaderStream) {
    TestKvcmServer leader(false);
    leader.event_service_.SetConfiguration(MakeConfiguration({}));
    KvcmEventSubscriber subscriber(
        MakeConfig(leader.endpoint()), optimizer_service_, metrics_registry_, metrics_reporter_);
    ASSERT_TRUE(subscriber.Init());
    ASSERT_TRUE(subscriber.Start());

    ASSERT_TRUE(WaitUntil([&leader] { return leader.event_service_.subscribe_count_.load() >= 2; }));
    subscriber.Stop();
}

TEST_F(KvcmEventSubscriberTest, ReconnectDelayUsesCappedExponentialBackoffWithJitter) {
    constexpr int64_t kMaxDelayMs = 30000;
    const std::vector<int64_t> nominal_delays_ms = {500, 1000, 2000, 4000, 8000, 16000, 30000, 30000};
    for (uint32_t attempt = 0; attempt < nominal_delays_ms.size(); ++attempt) {
        const int64_t nominal_ms = nominal_delays_ms[attempt];
        const auto delay = KvcmEventSubscriber::ComputeReconnectDelay(attempt);
        EXPECT_GE(delay.count(), nominal_ms * 80 / 100);
        EXPECT_LE(delay.count(), std::min(nominal_ms * 120 / 100, kMaxDelayMs));
    }
}

TEST_F(KvcmEventSubscriberTest, FormatsLeaderEndpointForIpv4AndIpv6) {
    EXPECT_EQ("127.0.0.1:6381", KvcmEventSubscriber::FormatEndpoint("127.0.0.1", 6381));
    EXPECT_EQ("kvcm.example:6381", KvcmEventSubscriber::FormatEndpoint("kvcm.example", 6381));
    EXPECT_EQ("[2001:db8::1]:6381", KvcmEventSubscriber::FormatEndpoint("2001:db8::1", 6381));
    EXPECT_EQ("[2001:db8::1]:6381", KvcmEventSubscriber::FormatEndpoint("[2001:db8::1]", 6381));
}

TEST_F(KvcmEventSubscriberTest, ManualAnalysisConfigurationStillDiscoversInstances) {
    TestKvcmServer leader;
    leader.event_service_.SetConfiguration(MakeConfiguration({"known"}));
    KvcmEventSubscriptionConfig config;
    ASSERT_TRUE(config.FromJsonString(
        std::string(R"({"mock_cache_layout":true,"ttl_seconds_list":[10,30],"service_discovery_url":"static://)") +
        leader.endpoint() + R"(","discovery_refresh_interval_ms":50})"));
    KvcmEventSubscriber subscriber(config, optimizer_service_, metrics_registry_, metrics_reporter_);
    ASSERT_TRUE(subscriber.Init());
    ASSERT_TRUE(subscriber.Start());
    ASSERT_TRUE(WaitUntil([&leader] { return leader.event_service_.active_subscribers_.load() == 1; }));
    leader.event_service_.Publish(MakeEvent("known", "manual", 1));
    ASSERT_TRUE(WaitUntil([this] { return TotalQueries("known") == 1; }));
    EXPECT_GT(leader.event_service_.configuration_count_.load(), 0);
    subscriber.Stop();
}

} // namespace kv_cache_manager
