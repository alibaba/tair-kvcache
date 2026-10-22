#include <thread>
#include <vector>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/meta/meta_store_health_tracker.h"

namespace kv_cache_manager {
namespace {

using Config = MetaStoreHealthTracker::Config;

// The tracker owns std::atomic members, so it is neither copyable nor movable:
// tests declare it by value and configure it in place.
void ConfigureTracker(MetaStoreHealthTracker &tracker, int64_t failure_threshold, int64_t probe_interval_ms) {
    tracker.Configure("test", Config{failure_threshold, probe_interval_ms});
}

TEST(MetaStoreHealthTrackerTest, StartsAvailable) {
    MetaStoreHealthTracker tracker;
    EXPECT_TRUE(tracker.IsAvailable());
    EXPECT_EQ(0, tracker.ConsecutiveFailures());
    EXPECT_FALSE(tracker.ShouldProbe());
}

TEST(MetaStoreHealthTrackerTest, ConfigureClampsInvalidValues) {
    MetaStoreHealthTracker tracker;
    ConfigureTracker(tracker, /*failure_threshold*/ 0, /*probe_interval_ms*/ -5);
    tracker.RecordFailure();
    EXPECT_FALSE(tracker.IsAvailable());
    EXPECT_EQ(1, tracker.ConsecutiveFailures());
}

TEST(MetaStoreHealthTrackerTest, DegradesOnlyAfterThresholdConsecutiveFailures) {
    MetaStoreHealthTracker tracker;
    ConfigureTracker(tracker, /*failure_threshold*/ 3, /*probe_interval_ms*/ 1000);
    tracker.RecordFailure();
    EXPECT_TRUE(tracker.IsAvailable());
    tracker.RecordFailure();
    EXPECT_TRUE(tracker.IsAvailable());
    tracker.RecordFailure();
    EXPECT_FALSE(tracker.IsAvailable());
    EXPECT_EQ(3, tracker.ConsecutiveFailures());
}

TEST(MetaStoreHealthTrackerTest, SuccessBelowThresholdResetsTheRun) {
    MetaStoreHealthTracker tracker;
    ConfigureTracker(tracker, /*failure_threshold*/ 3, /*probe_interval_ms*/ 1000);
    tracker.RecordFailure();
    tracker.RecordFailure();
    tracker.RecordSuccess();
    EXPECT_EQ(0, tracker.ConsecutiveFailures());
    tracker.RecordFailure();
    tracker.RecordFailure();
    EXPECT_TRUE(tracker.IsAvailable());
}

TEST(MetaStoreHealthTrackerTest, SingleSuccessRecoversImmediately) {
    MetaStoreHealthTracker tracker;
    ConfigureTracker(tracker, /*failure_threshold*/ 1, /*probe_interval_ms*/ 1000);
    tracker.RecordFailure();
    ASSERT_FALSE(tracker.IsAvailable());
    tracker.RecordSuccess();
    EXPECT_TRUE(tracker.IsAvailable());
    EXPECT_EQ(0, tracker.ConsecutiveFailures());
}

TEST(MetaStoreHealthTrackerTest, FailureCounterSaturatesAtThreshold) {
    MetaStoreHealthTracker tracker;
    ConfigureTracker(tracker, /*failure_threshold*/ 3, /*probe_interval_ms*/ 1000);
    for (int i = 0; i < 100; ++i) {
        tracker.RecordFailure();
    }
    EXPECT_EQ(3, tracker.ConsecutiveFailures());
    EXPECT_FALSE(tracker.IsAvailable());
    // One success is still enough to recover from a long outage.
    tracker.RecordSuccess();
    EXPECT_TRUE(tracker.IsAvailable());
}

// The heartbeat is what closes the allocation gate on an instance that was idle
// when Redis went down: with no operation in flight there is nothing else to
// report the outage, and a purely failure driven tracker would keep serving the
// last known-good verdict forever.
TEST(MetaStoreHealthTrackerTest, HealthyTrackerStillProbesAsHeartbeat) {
    MetaStoreHealthTracker tracker;
    ConfigureTracker(tracker, /*failure_threshold*/ 3, /*probe_interval_ms*/ 50);
    // Configure starts the probe clock: Open() has just proven the store
    // reachable, so there is no reason to probe again straight away.
    EXPECT_FALSE(tracker.ShouldProbe());

    std::this_thread::sleep_for(std::chrono::milliseconds(60));
    EXPECT_TRUE(tracker.ShouldProbe());
    // Only one caller wins per interval, so a fan-out of consumers issues a
    // single probe instead of one per thread.
    EXPECT_FALSE(tracker.ShouldProbe());
    EXPECT_TRUE(tracker.IsAvailable());
}

TEST(MetaStoreHealthTrackerTest, ProbeIsRateLimitedWhileDegraded) {
    MetaStoreHealthTracker tracker;
    ConfigureTracker(tracker, /*failure_threshold*/ 1, /*probe_interval_ms*/ 50);
    tracker.RecordFailure();
    ASSERT_FALSE(tracker.IsAvailable());

    std::this_thread::sleep_for(std::chrono::milliseconds(60));
    EXPECT_TRUE(tracker.ShouldProbe());
    EXPECT_FALSE(tracker.ShouldProbe());
    EXPECT_FALSE(tracker.ShouldProbe());

    std::this_thread::sleep_for(std::chrono::milliseconds(60));
    EXPECT_TRUE(tracker.ShouldProbe());
    EXPECT_FALSE(tracker.ShouldProbe());
}

// A probe is issued exactly because there is no recent verdict to trust, and it
// cannot fail for command level reasons, so one failed probe has to be enough.
// Counting it towards the threshold would stretch detection over
// threshold * probe_interval and let that many allocations through first.
TEST(MetaStoreHealthTrackerTest, ProbeFailureIsConclusive) {
    MetaStoreHealthTracker tracker;
    ConfigureTracker(tracker, /*failure_threshold*/ 5, /*probe_interval_ms*/ 1000);
    tracker.RecordFailure();
    tracker.RecordFailure();
    ASSERT_TRUE(tracker.IsAvailable());

    tracker.RecordProbeFailure();
    EXPECT_FALSE(tracker.IsAvailable());
    EXPECT_EQ(5, tracker.ConsecutiveFailures());

    // Recovery stays a single success away.
    tracker.RecordSuccess();
    EXPECT_TRUE(tracker.IsAvailable());
    EXPECT_EQ(0, tracker.ConsecutiveFailures());
}

TEST(MetaStoreHealthTrackerTest, ConcurrentFailuresStillDegrade) {
    MetaStoreHealthTracker tracker;
    ConfigureTracker(tracker, /*failure_threshold*/ 8, /*probe_interval_ms*/ 1000);
    std::vector<std::thread> workers;
    workers.reserve(8);
    for (int i = 0; i < 8; ++i) {
        workers.emplace_back([&tracker]() { tracker.RecordFailure(); });
    }
    for (auto &worker : workers) {
        worker.join();
    }
    EXPECT_FALSE(tracker.IsAvailable());
    EXPECT_EQ(8, tracker.ConsecutiveFailures());
}

} // namespace
} // namespace kv_cache_manager
