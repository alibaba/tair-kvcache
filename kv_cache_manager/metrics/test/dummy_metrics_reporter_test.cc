#include <memory>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/manager/cache_manager.h"
#include "kv_cache_manager/metrics/dummy_metrics_reporter.h"
#include "kv_cache_manager/metrics/metrics_collector.h"

using namespace kv_cache_manager;

class DummyMetricsReporterTest : public TESTBASE {
public:
    void SetUp() override { reporter_ = std::make_unique<DummyMetricsReporter>(); }

    void TearDown() override { reporter_.reset(); }

    std::unique_ptr<DummyMetricsReporter> reporter_;
};

TEST_F(DummyMetricsReporterTest, TestConstructor) {
    EXPECT_NE(reporter_, nullptr);
}

TEST_F(DummyMetricsReporterTest, TestInitWithNull) {
    bool result = reporter_->Init(nullptr, nullptr, "");
    EXPECT_TRUE(result);
}

TEST_F(DummyMetricsReporterTest, TestReportPerQuery) {
    EXPECT_NO_THROW(reporter_->ReportPerQuery(nullptr));
    DummyMetricsCollector collector;
    EXPECT_NO_THROW(reporter_->ReportPerQuery(&collector));
}

TEST_F(DummyMetricsReporterTest, TestReportInterval) {
    EXPECT_NO_THROW(reporter_->ReportInterval());
}

TEST_F(DummyMetricsReporterTest, TestGetMetricsRegistry) {
    const auto registry = reporter_->GetMetricsRegistry();
    EXPECT_EQ(nullptr, registry);
}
