#include <cstdint>
#include <memory>
#include <set>
#include <vector>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/metrics/metrics_registry.h"
#include "kv_cache_manager/metrics/revisit_interval_histogram.h"

using namespace kv_cache_manager;

class RevisitIntervalHistogramTest : public TESTBASE {
public:
    void SetUp() override { registry_ = std::make_shared<MetricsRegistry>(); }
    void TearDown() override {}

    std::shared_ptr<MetricsRegistry> registry_;
};

// 测试 Init 参数校验
TEST_F(RevisitIntervalHistogramTest, InitWithEmptyBoundariesFails) {
    RevisitIntervalHistogram hist;
    std::vector<double> empty;
    EXPECT_FALSE(hist.Init(registry_, empty, "test_instance"));
}

TEST_F(RevisitIntervalHistogramTest, InitWithNonPositiveBoundaryFails) {
    RevisitIntervalHistogram hist;
    std::vector<double> boundaries = {0.0, 1.0, 5.0};
    EXPECT_FALSE(hist.Init(registry_, boundaries, "test_instance"));
}

TEST_F(RevisitIntervalHistogramTest, InitWithNegativeBoundaryFails) {
    RevisitIntervalHistogram hist;
    std::vector<double> boundaries = {-1.0, 1.0, 5.0};
    EXPECT_FALSE(hist.Init(registry_, boundaries, "test_instance"));
}

TEST_F(RevisitIntervalHistogramTest, InitWithUnsortedBoundariesFails) {
    RevisitIntervalHistogram hist;
    std::vector<double> boundaries = {5.0, 1.0, 10.0};
    EXPECT_FALSE(hist.Init(registry_, boundaries, "test_instance"));
}

TEST_F(RevisitIntervalHistogramTest, InitWithDuplicateBoundariesFails) {
    RevisitIntervalHistogram hist;
    std::vector<double> boundaries = {1.0, 5.0, 5.0, 10.0};
    EXPECT_FALSE(hist.Init(registry_, boundaries, "test_instance"));
}

TEST_F(RevisitIntervalHistogramTest, InitWithValidBoundariesSucceeds) {
    RevisitIntervalHistogram hist;
    std::vector<double> boundaries = {1.0, 5.0, 10.0};
    EXPECT_TRUE(hist.Init(registry_, boundaries, "test_instance"));
    EXPECT_EQ(hist.GetBoundaries().size(), 3);
}

// 测试 Observe 基本功能
TEST_F(RevisitIntervalHistogramTest, ObserveZeroOrNegativeIsIgnored) {
    RevisitIntervalHistogram hist;
    std::vector<double> boundaries = {1.0, 5.0, 10.0};
    ASSERT_TRUE(hist.Init(registry_, boundaries, "test_instance"));

    hist.Observe(0);
    hist.Observe(-1000000); // -1 second

    EXPECT_EQ(hist.GetCount(), 0);
    EXPECT_EQ(hist.GetSum(), 0);
}

TEST_F(RevisitIntervalHistogramTest, ObserveSingleValue) {
    RevisitIntervalHistogram hist;
    std::vector<double> boundaries = {1.0, 5.0, 10.0};
    ASSERT_TRUE(hist.Init(registry_, boundaries, "test_instance"));

    // Observe 3 seconds (3,000,000 microseconds)
    hist.Observe(3000000);

    // Count should be 1
    EXPECT_EQ(hist.GetCount(), 1);
    // Sum should be 3,000,000 microseconds
    EXPECT_EQ(hist.GetSum(), 3000000);

    // Bucket counts: [0, 1, 1, 1] for boundaries [1.0, 5.0, 10.0, +Inf]
    // 3 seconds <= 5.0, so buckets 5.0, 10.0, +Inf are incremented
    auto counts = hist.GetBucketCounts();
    ASSERT_EQ(counts.size(), 4);
    EXPECT_EQ(counts[0], 0); // <= 1.0: no
    EXPECT_EQ(counts[1], 1); // <= 5.0: yes
    EXPECT_EQ(counts[2], 1); // <= 10.0: yes (cumulative)
    EXPECT_EQ(counts[3], 1); // <= +Inf: yes (cumulative)
}

TEST_F(RevisitIntervalHistogramTest, ObserveExactBoundaryValue) {
    RevisitIntervalHistogram hist;
    std::vector<double> boundaries = {1.0, 5.0, 10.0};
    ASSERT_TRUE(hist.Init(registry_, boundaries, "test_instance"));

    // Observe exactly 5 seconds
    hist.Observe(5000000);

    auto counts = hist.GetBucketCounts();
    ASSERT_EQ(counts.size(), 4);
    EXPECT_EQ(counts[0], 0); // <= 1.0: no
    EXPECT_EQ(counts[1], 1); // <= 5.0: yes (exact match)
    EXPECT_EQ(counts[2], 1); // <= 10.0: yes
    EXPECT_EQ(counts[3], 1); // <= +Inf: yes
}

TEST_F(RevisitIntervalHistogramTest, ObserveValueLargerThanAllBoundaries) {
    RevisitIntervalHistogram hist;
    std::vector<double> boundaries = {1.0, 5.0, 10.0};
    ASSERT_TRUE(hist.Init(registry_, boundaries, "test_instance"));

    // Observe 100 seconds
    hist.Observe(100000000);

    auto counts = hist.GetBucketCounts();
    ASSERT_EQ(counts.size(), 4);
    EXPECT_EQ(counts[0], 0); // <= 1.0: no
    EXPECT_EQ(counts[1], 0); // <= 5.0: no
    EXPECT_EQ(counts[2], 0); // <= 10.0: no
    EXPECT_EQ(counts[3], 1); // <= +Inf: yes
}

// 测试累积性质
TEST_F(RevisitIntervalHistogramTest, CumulativeProperty) {
    RevisitIntervalHistogram hist;
    std::vector<double> boundaries = {1.0, 5.0, 10.0};
    ASSERT_TRUE(hist.Init(registry_, boundaries, "test_instance"));

    // Observe multiple values
    hist.Observe(500000);   // 0.5s -> <= 1.0, 5.0, 10.0, +Inf
    hist.Observe(3000000);  // 3.0s -> <= 5.0, 10.0, +Inf
    hist.Observe(7000000);  // 7.0s -> <= 10.0, +Inf
    hist.Observe(50000000); // 50.0s -> <= +Inf

    EXPECT_EQ(hist.GetCount(), 4);

    auto counts = hist.GetBucketCounts();
    ASSERT_EQ(counts.size(), 4);
    EXPECT_EQ(counts[0], 1); // <= 1.0: only 0.5s
    EXPECT_EQ(counts[1], 2); // <= 5.0: 0.5s and 3.0s
    EXPECT_EQ(counts[2], 3); // <= 10.0: 0.5s, 3.0s, 7.0s
    EXPECT_EQ(counts[3], 4); // <= +Inf: all 4 values
}

// 测试 Sum 计算
TEST_F(RevisitIntervalHistogramTest, SumCalculation) {
    RevisitIntervalHistogram hist;
    std::vector<double> boundaries = {1.0, 5.0, 10.0};
    ASSERT_TRUE(hist.Init(registry_, boundaries, "test_instance"));

    hist.Observe(1000000); // 1s = 1,000,000 us
    hist.Observe(2500000); // 2.5s = 2,500,000 us
    hist.Observe(100000);  // 0.1s = 100,000 us

    EXPECT_EQ(hist.GetSum(), 3600000); // sum stored in microseconds
}

// 测试 Per-instance 隔离
TEST_F(RevisitIntervalHistogramTest, PerInstanceIsolation) {
    RevisitIntervalHistogram hist_a;
    RevisitIntervalHistogram hist_b;
    std::vector<double> boundaries = {1.0, 5.0, 10.0};

    ASSERT_TRUE(hist_a.Init(registry_, boundaries, "instance_a"));
    ASSERT_TRUE(hist_b.Init(registry_, boundaries, "instance_b"));

    hist_a.Observe(2000000); // 2s
    hist_a.Observe(3000000); // 3s

    hist_b.Observe(7000000); // 7s

    EXPECT_EQ(hist_a.GetCount(), 2);
    EXPECT_EQ(hist_b.GetCount(), 1);

    auto counts_a = hist_a.GetBucketCounts();
    auto counts_b = hist_b.GetBucketCounts();

    // instance_a: 2 observations (2s and 3s), both <= 5.0
    EXPECT_EQ(counts_a[1], 2); // <= 5.0

    // instance_b: 1 observation (7s), <= 10.0 but not <= 5.0
    EXPECT_EQ(counts_b[1], 0); // <= 5.0
    EXPECT_EQ(counts_b[2], 1); // <= 10.0
}

// 测试默认桶边界
TEST_F(RevisitIntervalHistogramTest, DefaultBoundaries) {
    RevisitIntervalHistogram hist;
    std::vector<double> boundaries = {1, 5, 30, 60, 120, 180, 300, 600, 900, 1800, 3600, 21600, 86400};
    ASSERT_TRUE(hist.Init(registry_, boundaries, "test_instance"));

    EXPECT_EQ(hist.GetBoundaries().size(), 13);
    EXPECT_EQ(hist.GetBucketCounts().size(), 14); // 13 boundaries + 1 for +Inf
}

// 测试大量观测
TEST_F(RevisitIntervalHistogramTest, ManyObservations) {
    RevisitIntervalHistogram hist;
    std::vector<double> boundaries = {1.0, 5.0, 10.0};
    ASSERT_TRUE(hist.Init(registry_, boundaries, "test_instance"));

    // Observe 1000 values of 3 seconds each
    for (int i = 0; i < 1000; ++i) {
        hist.Observe(3000000);
    }

    EXPECT_EQ(hist.GetCount(), 1000);
    EXPECT_EQ(hist.GetSum(), 3000000000); // 1000 * 3,000,000 us

    auto counts = hist.GetBucketCounts();
    EXPECT_EQ(counts[0], 0);    // <= 1.0
    EXPECT_EQ(counts[1], 1000); // <= 5.0
    EXPECT_EQ(counts[2], 1000); // <= 10.0
    EXPECT_EQ(counts[3], 1000); // <= +Inf
}

// 测试 le 标签格式化（整数边界不含多余零）
TEST_F(RevisitIntervalHistogramTest, LeLabelFormatting) {
    RevisitIntervalHistogram hist;
    // Mix integer and non-integer boundaries
    std::vector<double> boundaries = {1.0, 2.5, 60.0, 300.0, 3600.0};
    ASSERT_TRUE(hist.Init(registry_, boundaries, "test_instance"));

    std::vector<MetricsRegistry::metrics_tuple_t> all_metrics;
    registry_->GetAllMetrics(all_metrics);

    // Collect le values from bucket metrics
    std::set<std::string> le_values;
    for (const auto &[name, tags, val] : all_metrics) {
        if (name == "revisit_interval_seconds_bucket") {
            auto it = tags.find("le");
            if (it != tags.end()) {
                le_values.insert(it->second);
            }
        }
    }

    // Should have 6 buckets: 5 boundaries + +Inf
    ASSERT_EQ(le_values.size(), 6);

    // Integer boundaries must not have trailing zeros like "1.000000"
    EXPECT_EQ(le_values.count("1"), 1);        // not "1.000000"
    EXPECT_EQ(le_values.count("2.500000"), 1); // non-integer keeps decimals (std::to_string behavior)
    EXPECT_EQ(le_values.count("60"), 1);       // not "60.000000"
    EXPECT_EQ(le_values.count("300"), 1);      // not "300.000000"
    EXPECT_EQ(le_values.count("3600"), 1);     // not "3600.000000"
    EXPECT_EQ(le_values.count("+Inf"), 1);
}
