#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <vector>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/metrics/metrics_reporter.h"
#include "kv_cache_manager/metrics/prometheus_exporter.h"
#include "kv_cache_manager/service/server.h"
#include "kv_cache_manager/service/service_request_executor.h"

namespace kv_cache_manager {
namespace {

using namespace std::chrono_literals;
using Executor = ServiceRequestExecutor;
using Token = Executor::CompletionToken;
using Result = Executor::SubmitResult;

struct ReportedState {
    double queued;
    double running;
    uint64_t rejected;
    uint64_t discarded;
};

class RecordingReporter final : public MetricsReporter {
public:
    explicit RecordingReporter(std::shared_ptr<MetricsRegistry> registry) : registry_(std::move(registry)) {}

    bool Init(std::shared_ptr<CacheManager>,
              std::shared_ptr<MetricsRegistry>,
              const std::string &,
              const MetricsTags &) override {
        return true;
    }
    void ReportPerQuery(MetricsCollector *, std::optional<bool>) override {}
    std::shared_ptr<MetricsRegistry> GetMetricsRegistry() override { return registry_; }

    void ReportInterval() override {
        std::lock_guard<std::mutex> lock(mutex_);
        reports_.push_back({registry_->GetGauge("service.request_queue_length").Get(),
                            registry_->GetGauge("service.request_running").Get(),
                            registry_->GetCounter("service.request_rejected").Get(),
                            registry_->GetCounter("service.request_discarded").Get()});
        changed_.notify_all();
    }

    bool WaitForReport() {
        std::unique_lock<std::mutex> lock(mutex_);
        return changed_.wait_for(lock, 3s, [&] { return !reports_.empty(); });
    }

    std::vector<ReportedState> Reports() {
        std::lock_guard<std::mutex> lock(mutex_);
        return reports_;
    }

private:
    std::shared_ptr<MetricsRegistry> registry_;
    std::mutex mutex_;
    std::condition_variable changed_;
    std::vector<ReportedState> reports_;
};

class WorkerGate {
public:
    void Enter() {
        std::unique_lock<std::mutex> lock(mutex_);
        entered_ = true;
        changed_.notify_all();
        changed_.wait(lock, [&] { return open_; });
    }

    bool WaitForEntry() {
        std::unique_lock<std::mutex> lock(mutex_);
        return changed_.wait_for(lock, 3s, [&] { return entered_; });
    }

    void Open() {
        std::lock_guard<std::mutex> lock(mutex_);
        open_ = true;
        changed_.notify_all();
    }

private:
    std::mutex mutex_;
    std::condition_variable changed_;
    bool entered_ = false;
    bool open_ = false;
};

Executor::Task UnexpectedTask() {
    return {[](Token) { ADD_FAILURE() << "rejected task ran"; },
            [](Token) { ADD_FAILURE() << "rejected task was discarded"; }};
}

class ServerMetricsTest : public TESTBASE {
public:
    void SetUp() override {
        server_.metrics_registry_ = std::make_shared<MetricsRegistry>();
        server_.request_executor_ = std::make_shared<Executor>(1, 1);
        reporter_ = std::make_shared<RecordingReporter>(server_.metrics_registry_);
        server_.metrics_reporter_ = reporter_;
        server_.config_.metrics_report_interval_ms_ = 60000;
    }

    void TearDown() override {
        // Always release a blocked worker, including after a fatal assertion.
        gate_.Open();
        server_.Stop();
    }

    void BlockWorker() {
        ASSERT_EQ(Result::kAccepted,
                  server_.request_executor_->TrySubmit(
                      {[this](Token) { gate_.Enter(); }, [](Token) { ADD_FAILURE() << "blocker was discarded"; }}));
        ASSERT_TRUE(gate_.WaitForEntry());
    }

    Server server_;
    WorkerGate gate_;
    std::shared_ptr<RecordingReporter> reporter_;
};

TEST_F(ServerMetricsTest, StopReportsDiscardsBeforeNextIntervalAndIsIdempotent) {
    ASSERT_NO_FATAL_FAILURE(BlockWorker());
    ASSERT_EQ(Result::kAccepted,
              server_.request_executor_->TrySubmit(
                  {[](Token) { ADD_FAILURE() << "pending task ran during shutdown"; },
                   [this](Token) {
                       // A rejection after the periodic sample must also appear in the final report.
                       EXPECT_EQ(Result::kClosed, server_.request_executor_->TrySubmit(UnexpectedTask()));
                       gate_.Open();
                   }}));
    EXPECT_EQ(Result::kFull, server_.request_executor_->TrySubmit(UnexpectedTask()));
    ASSERT_TRUE(server_.StartMetricsReportThread());
    ASSERT_TRUE(reporter_->WaitForReport());

    // Closing admission discards the queued task and opens the worker gate.
    // Stop must sample again without waiting for the next 60-second interval.
    server_.Stop();
    const auto reports = reporter_->Reports();
    ASSERT_EQ(2u, reports.size());
    EXPECT_EQ(1, reports.front().queued);
    EXPECT_EQ(1, reports.front().running);
    EXPECT_EQ(1u, reports.front().rejected);
    EXPECT_EQ(0u, reports.front().discarded);
    EXPECT_EQ(0, reports.back().queued);
    EXPECT_EQ(0, reports.back().running);
    EXPECT_EQ(2u, reports.back().rejected);
    EXPECT_EQ(1u, reports.back().discarded);
    EXPECT_EQ(1u, server_.metrics_registry_->GetCounter("service.request_discarded").Get());
    const auto wait = server_.request_executor_->GetSnapshot().queue_wait;
    EXPECT_EQ(1u, wait.count); // Rejection and pending discard do not add wait samples.
    EXPECT_EQ(wait.count, server_.metrics_registry_->GetCounter("service.request_queue_wait_seconds_count").Get());
    EXPECT_EQ(wait.sum_us, server_.metrics_registry_->GetCounter("service.request_queue_wait_seconds_sum").Get());

    server_.Stop();
    EXPECT_EQ(2u, reporter_->Reports().size());
}

TEST_F(ServerMetricsTest, QueueWaitHistogramExportsSecondsAndDoesNotCountRepeatedSnapshots) {
    server_.ReportMetrics();
    EXPECT_EQ(0u, server_.metrics_registry_->GetCounter("service.request_queue_wait_seconds_count").Get());
    ASSERT_NO_FATAL_FAILURE(BlockWorker());
    const auto wait = server_.request_executor_->GetSnapshot().queue_wait;
    server_.ReportMetrics();
    server_.ReportMetrics();
    const auto output = PrometheusExporter::Expose(*server_.metrics_registry_, "kvcm", {});
    EXPECT_NE(std::string::npos, output.find("# TYPE kvcm_service_request_queue_wait_seconds histogram\n"));
    EXPECT_NE(std::string::npos, output.find("kvcm_service_request_queue_wait_seconds_count 1\n"));
    EXPECT_NE(std::string::npos, output.find("kvcm_service_request_queue_wait_seconds_bucket{le=\"+Inf\"} 1\n"));
    const std::string prefix = "kvcm_service_request_queue_wait_seconds_sum ";
    const auto offset = output.find(prefix);
    ASSERT_NE(std::string::npos, offset);
    const auto expected_seconds = static_cast<double>(wait.sum_us) / 1e6;
    EXPECT_NEAR(expected_seconds, std::stod(output.substr(offset + prefix.size())), expected_seconds * 1e-5 + 1e-12);
    EXPECT_EQ(wait.sum_us, server_.metrics_registry_->GetCounter("service.request_queue_wait_seconds_sum").Get());
    gate_.Open();
    server_.Stop();
    EXPECT_EQ(1u, server_.metrics_registry_->GetCounter("service.request_queue_wait_seconds_count").Get());
}

TEST_F(ServerMetricsTest, RepeatedSamplesAccumulateOnlyNewCounts) {
    ASSERT_NO_FATAL_FAILURE(BlockWorker());
    ASSERT_EQ(Result::kAccepted,
              server_.request_executor_->TrySubmit({[](Token) { ADD_FAILURE() << "pending task ran during shutdown"; },
                                                    [this](Token) { gate_.Open(); }}));
    EXPECT_EQ(Result::kFull, server_.request_executor_->TrySubmit(UnexpectedTask()));
    server_.ReportMetrics();
    server_.ReportMetrics();
    EXPECT_EQ(1u, server_.metrics_registry_->GetCounter("service.request_rejected").Get());

    server_.request_executor_->CloseAdmission();
    EXPECT_EQ(Result::kClosed, server_.request_executor_->TrySubmit(UnexpectedTask()));
    server_.request_executor_->Drain();
    server_.ReportMetrics();
    server_.ReportMetrics();
    server_.Stop();
    const auto reports = reporter_->Reports();
    ASSERT_EQ(5u, reports.size());
    for (size_t i = 2; i < reports.size(); ++i) {
        EXPECT_EQ(0, reports[i].queued);
        EXPECT_EQ(0, reports[i].running);
        EXPECT_EQ(2u, reports[i].rejected);
        EXPECT_EQ(1u, reports[i].discarded);
    }
    EXPECT_EQ(2u, server_.metrics_registry_->GetCounter("service.request_rejected").Get());
    EXPECT_EQ(1u, server_.metrics_registry_->GetCounter("service.request_discarded").Get());
}

TEST(ServerMetricsPartialStartupTest, StopWithoutReporterOrExecutorIsSafe) {
    Server empty_server;
    empty_server.Stop();
    empty_server.Stop();

    Server executor_only_server;
    executor_only_server.request_executor_ = std::make_shared<Executor>(1);
    executor_only_server.Stop();
    executor_only_server.Stop();

    Server reporter_only_server;
    reporter_only_server.metrics_registry_ = std::make_shared<MetricsRegistry>();
    auto reporter = std::make_shared<RecordingReporter>(reporter_only_server.metrics_registry_);
    reporter_only_server.metrics_reporter_ = reporter;
    reporter_only_server.Stop();
    reporter_only_server.Stop();
    EXPECT_EQ(1u, reporter->Reports().size());
}

} // namespace
} // namespace kv_cache_manager
