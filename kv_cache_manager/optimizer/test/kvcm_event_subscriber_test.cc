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
        grpc::ServerBuilder builder;
        builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port_);
        builder.RegisterService(&meta_service_);
        builder.RegisterService(&event_service_);
        server_ = builder.BuildAndStart();
        meta_service_.SetLeader("127.0.0.1", port_);
    }

    ~TestKvcmServer() {
        if (server_) {
            server_->Shutdown();
        }
    }

    std::string endpoint() const { return "127.0.0.1:" + std::to_string(port_); }
    int port() const { return port_; }

    TestMetaService meta_service_;
    TestEventService event_service_;

private:
    int port_ = 0;
    std::unique_ptr<grpc::Server> server_;
};

KvcmEventSubscriptionConfig MakeConfig(const std::string &seed_endpoint, int refresh_interval_ms = 50) {
    OnlineOptimizerServerConfig server_config;
    EXPECT_TRUE(server_config.FromJsonString(
        std::string(R"({"kvcm_event_subscription":{"enable":true,"service_discovery_url":"static://)") + seed_endpoint +
        R"(","consumer_id":"subscriber-test","discovery_refresh_interval_ms":)" + std::to_string(refresh_interval_ms) +
        "}}"));
    return server_config.kvcm_event_subscription();
}

} // namespace

class KvcmEventSubscriberTest : public TESTBASE {
protected:
    void SetUp() override {
        registry_ = std::make_shared<OptimizerRegistryManager>("");
        ASSERT_TRUE(registry_->Init());
        manager_ = std::make_shared<OnlineOptimizerManager>(registry_);
        metrics_registry_ = std::make_shared<MetricsRegistry>();
        metrics_reporter_ = std::make_shared<OptimizerMetricsReporter>(manager_, metrics_registry_, "test_opt");
        optimizer_service_ = std::make_shared<OptimizerServiceImpl>(manager_, metrics_reporter_);
    }

    bool IsRegistered(const std::string &instance_id) const {
        return manager_->GetInstanceState(instance_id, [](const InstanceState &) {}) == EC_OK;
    }

    int64_t TotalQueries(const std::string &instance_id) const {
        std::vector<InstanceSummary> summaries;
        EXPECT_EQ(EC_OK, manager_->ListInstances("", summaries));
        for (const auto &summary : summaries) {
            if (summary.instance_id == instance_id) {
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

TEST_F(KvcmEventSubscriberTest, DiscoversLeaderAndRegistersConfigurationBeforeConsuming) {
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
    ASSERT_EQ(1u, group->capacity_gb().size());
    EXPECT_DOUBLE_EQ(2.0, group->capacity_gb()[0]);
    EXPECT_TRUE(group->enable_prefix_hash());
    EXPECT_TRUE(group->enable_theoretical_max_cache());
    EXPECT_EQ(24 * 60 * 60, group->ttl_seconds());

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
    ASSERT_TRUE(WaitUntil([this] {
        return metrics_registry_->GetCounter("service.query_counter").Get() == 2u &&
               metrics_registry_->GetCounter("service.error_counter").Get() == 1u;
    }));
    subscriber.Stop();
}

TEST_F(KvcmEventSubscriberTest, ReportsQueryMetricsForStreamEvents) {
    TestKvcmServer leader;
    leader.event_service_.SetConfiguration(MakeConfiguration({"known"}));
    KvcmEventSubscriber subscriber(
        MakeConfig(leader.endpoint()), optimizer_service_, metrics_registry_, metrics_reporter_);
    ASSERT_TRUE(subscriber.Init());
    ASSERT_TRUE(subscriber.Start());
    ASSERT_TRUE(WaitUntil([this] { return IsRegistered("known"); }));
    ASSERT_TRUE(WaitUntil([&leader] { return leader.event_service_.active_subscribers_.load() == 1; }));

    const MetricsTags base_tags = {{"instance_id", "known"}, {"client_ip", "127.0.0.1"}};
    const MetricsTags capacity_tags = {
        {"instance_id", "known"}, {"client_ip", "127.0.0.1"}, {"capacity_gb", std::to_string(2.0)}};

    leader.event_service_.Publish(MakeEvent("known", "miss", 1));
    ASSERT_TRUE(WaitUntil(
        [this, &base_tags] { return metrics_registry_->GetGauge("query_total_blocks", base_tags).Get() == 1.0; }));
    EXPECT_DOUBLE_EQ(0.0, metrics_registry_->GetGauge("query_hit_count", capacity_tags).Get());
    EXPECT_DOUBLE_EQ(0.0, metrics_registry_->GetGauge("query_hit_rate", capacity_tags).Get());

    leader.event_service_.Publish(MakeEvent("known", "hit", 1));
    ASSERT_TRUE(WaitUntil(
        [this, &capacity_tags] { return metrics_registry_->GetGauge("query_hit_count", capacity_tags).Get() == 1.0; }));
    EXPECT_DOUBLE_EQ(1.0, metrics_registry_->GetGauge("query_hit_rate", capacity_tags).Get());
    EXPECT_EQ(2u, metrics_registry_->GetCounter("service.query_counter").Get());
    EXPECT_EQ(0u, metrics_registry_->GetCounter("service.error_counter").Get());
    EXPECT_GE(metrics_registry_->GetGauge("service.query_rt_us").Get(), 0.0);

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

TEST_F(KvcmEventSubscriberTest, SubscribesAndDoesNotRefreshForUnsupportedInstance) {
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
    EXPECT_FALSE(IsRegistered("ambiguous"));
    const int configuration_count = leader.event_service_.configuration_count_.load();

    leader.event_service_.Publish(MakeEvent("ambiguous", "unsupported", 1));
    ASSERT_TRUE(WaitUntil([&leader] { return leader.event_service_.delivered_count_.load() == 1; }));
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    EXPECT_EQ(configuration_count, leader.event_service_.configuration_count_.load());
    subscriber.Stop();
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

} // namespace kv_cache_manager
