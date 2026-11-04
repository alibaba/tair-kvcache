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
class CapacityObserver : public MetricsReporter {
public:
    bool Init(std::shared_ptr<CacheManager>, std::shared_ptr<MetricsRegistry>,
              const std::string &, const MetricsTags & = {}) override { return true; }
    void ReportPerQuery(MetricsCollector *, std::optional<bool> = std::nullopt) override {}
    void ReportInterval() override {
        const auto &provider = GetStorageUsageProvider();
        enabled.store(!provider || provider());
    }
    std::shared_ptr<MetricsRegistry> GetMetricsRegistry() override { return nullptr; }
    std::atomic<int> enabled{-1};
};
} // namespace

class ServerStorageUsageMetricsTest : public TESTBASE {};

TEST_F(ServerStorageUsageMetricsTest, CapacityEligibilityFollowsLeaseWithoutChangingIt) {
    Server server;
    auto reporter = std::make_shared<CapacityObserver>();
    server.metrics_reporter_ = reporter;
    server.leader_elector_ = std::make_shared<LeaderElector>(nullptr, "key", "node");
    server.config_.metrics_report_interval_ms_ = 10;
    ASSERT_TRUE(server.StartMetricsReportThread());
    for (bool leader : {false, true, false, true, false}) {
        server.leader_elector_->lock_acquired_.store(leader);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
        while (reporter->enabled.load() != int(leader) && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        EXPECT_EQ(int(leader), reporter->enabled.load());
        EXPECT_EQ(leader, server.leader_elector_->IsLeader());
    }
    server.Stop();
}
