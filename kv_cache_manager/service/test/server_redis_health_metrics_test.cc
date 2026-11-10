#include <atomic>
#include <chrono>
#include <future>
#include <memory>
#include <shared_mutex>
#include <string>
#include <thread>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/config/leader_elector.h"
#include "kv_cache_manager/config/registry_manager.h"
#include "kv_cache_manager/manager/cache_manager.h"
#include "kv_cache_manager/metrics/metrics_lifecycle.h"
#include "kv_cache_manager/metrics/metrics_reporter.h"
#include "kv_cache_manager/metrics/prometheus_exporter.h"
#include "kv_cache_manager/metrics/redis_health_metrics.h"
#include "kv_cache_manager/service/server.h"

using namespace kv_cache_manager;

namespace {
class BlockingReporter : public MetricsReporter {
public:
    BlockingReporter() : release_(released_.get_future().share()) {}
    bool Init(std::shared_ptr<CacheManager>,
              std::shared_ptr<MetricsRegistry>,
              const std::string &,
              const MetricsTags & = {}) override {
        return true;
    }
    void ReportPerQuery(MetricsCollector *, std::optional<bool> = std::nullopt) override {}
    void ReportInterval() override {
        if (!entered_.exchange(true)) {
            started.set_value();
        }
        release_.wait();
    }
    std::shared_ptr<MetricsRegistry> GetMetricsRegistry() override { return nullptr; }
    void Release() { released_.set_value(); }
    std::promise<void> started;

private:
    std::atomic<bool> entered_{false};
    std::promise<void> released_;
    std::shared_future<void> release_;
};
} // namespace

class ServerRedisHealthMetricsTest : public TESTBASE {
public:
    void SetUp() override {
        server_.metrics_registry_ = std::make_shared<MetricsRegistry>();
        server_.metrics_lifecycle_ = std::make_shared<MetricsLifecycle>();
        server_.registry_manager_ = std::make_shared<RegistryManager>("", server_.metrics_registry_);
        server_.cache_manager_ = std::make_shared<CacheManager>(server_.metrics_registry_, server_.registry_manager_);
        server_.leader_elector_ = std::make_shared<LeaderElector>(nullptr, "key", "node");
        reporter_ = std::make_shared<BlockingReporter>();
        server_.metrics_reporter_ = reporter_;
        server_.config_.metrics_report_interval_ms_ = 10;
    }
    void TearDown() override {
        reporter_->Release();
        server_.Stop();
    }
    bool WaitForLine(const std::string &line) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
        while (std::chrono::steady_clock::now() < deadline) {
            if (PrometheusExporter::Expose(*server_.metrics_registry_).find(line) != std::string::npos) {
                return true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        return false;
    }
    Server server_;
    std::shared_ptr<BlockingReporter> reporter_;
};

TEST_F(ServerRedisHealthMetricsTest, CoordinationFailureReportsInEveryPhaseDespiteBlockedReporterAndLifecycle) {
    // Exercise the legacy URI option as well. Port 0 is deliberately invalid:
    // this test must not connect to any existing Redis on the developer machine.
    ASSERT_TRUE(server_.config_.Parse(
        "", {{"kvcm.distributed_lock.uri", "redis://127.0.0.1:0"}, {"kvcm.metrics.report_interval_ms", "10"}}));
    std::unique_lock writer(server_.metrics_lifecycle_->mut_);
    ASSERT_TRUE(server_.StartMetricsReportThread());
    ASSERT_TRUE(server_.coordination_redis_health_thread_);
    ASSERT_EQ(std::future_status::ready, reporter_->started.get_future().wait_for(std::chrono::seconds(2)));
    for (const auto phase : {RoleState::FOLLOWER, RoleState::PROMOTING, RoleState::LEADER, RoleState::DEMOTING}) {
        server_.leader_elector_->role_state_.store(phase);
        // Clear the previous result so each phase requires a new observation.
        server_.metrics_registry_->GetGauge(CoordinationRedisHealthMetrics::kMetricName,
                                            {{"host", "127.0.0.1"}, {"port", "0"}}) = 1;
        EXPECT_TRUE(WaitForLine("kvcm_coordination_redis_healthy_status{host=\"127.0.0.1\",port=\"0\"} 0\n"));
        EXPECT_FALSE(server_.leader_elector_->IsLeader()); // the probe cannot acquire/revoke a lease
    }
    // The ordinary reporter is still blocked, yet all four samples were published.
    EXPECT_FALSE(server_.stop_.load());
}

TEST_F(ServerRedisHealthMetricsTest, DisabledProbeCreatesNoProbeObjectsOrThreads) {
    ASSERT_TRUE(server_.config_.Parse(
        "", {{"kvcm.coordination.uri", "redis://127.0.0.1:0"},
             {"kvcm.metrics.report_interval_ms", "10"},
             {"kvcm.metrics.redis_health_probe_enabled", "false"}}));
    ASSERT_TRUE(server_.StartMetricsReportThread());
    EXPECT_TRUE(server_.metrics_report_thread_);
    EXPECT_FALSE(server_.redis_health_metrics_);
    EXPECT_FALSE(server_.redis_health_thread_);
    EXPECT_FALSE(server_.coordination_redis_health_metrics_);
    EXPECT_FALSE(server_.coordination_redis_health_thread_);
    ASSERT_EQ(std::future_status::ready, reporter_->started.get_future().wait_for(std::chrono::seconds(2)));
    EXPECT_EQ(std::string::npos, PrometheusExporter::Expose(*server_.metrics_registry_).find("redis_healthy_status"));
}

TEST_F(ServerRedisHealthMetricsTest, EnabledProbeWithNonRedisCoordinationCreatesOnlyMetadataThread) {
    ASSERT_TRUE(server_.config_.Parse(
        "", {{"kvcm.coordination.uri", "memory://"},
             {"kvcm.metrics.report_interval_ms", "10"},
             {"kvcm.metrics.redis_health_probe_enabled", "true"}}));
    ASSERT_TRUE(server_.StartMetricsReportThread());
    EXPECT_TRUE(server_.redis_health_metrics_);
    EXPECT_TRUE(server_.redis_health_thread_);
    EXPECT_FALSE(server_.coordination_redis_health_thread_);
}
