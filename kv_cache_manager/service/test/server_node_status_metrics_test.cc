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

class ServerNodeStatusMetricsTest : public TESTBASE {
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
    bool WaitForPhase(const char *phase) {
        return WaitForLine(std::string("kvcm_role_state{state=\"") + phase + "\"} 1\n");
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

TEST_F(ServerNodeStatusMetricsTest, PhaseCollectionContinuesWhileReporterAndLifecycleAreBlocked) {
    // Recovery/removal can hold this writer lock while the ordinary reporter is
    // stalled. The phase observer must not depend on either one.
    std::unique_lock writer(server_.metrics_lifecycle_->mut_);
    ASSERT_TRUE(server_.StartMetricsReportThread());
    ASSERT_EQ(std::future_status::ready, reporter_->started.get_future().wait_for(std::chrono::seconds(2)));
    EXPECT_TRUE(WaitForPhase("follower"));
    server_.leader_elector_->role_state_.store(RoleState::PROMOTING);
    EXPECT_TRUE(WaitForPhase("promoting"));
    server_.leader_elector_->role_state_.store(RoleState::LEADER);
    EXPECT_TRUE(WaitForPhase("leader"));
    server_.leader_elector_->role_state_.store(RoleState::DEMOTING);
    EXPECT_TRUE(WaitForPhase("demoting"));
    server_.leader_elector_->role_state_.store(RoleState::FOLLOWER);
    EXPECT_TRUE(WaitForPhase("follower"));
    EXPECT_FALSE(server_.leader_elector_->IsLeader()); // observing phases never acquires a lease
    EXPECT_EQ(std::string::npos, PrometheusExporter::Expose(*server_.metrics_registry_).find("kvcm_access_mode"));
}
