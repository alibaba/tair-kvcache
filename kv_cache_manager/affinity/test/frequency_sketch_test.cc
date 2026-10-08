// affinity v1 F5: FrequencySketch（per-(caller, key) LRU counter）单测

#include "kv_cache_manager/affinity/frequency_sketch.h"
#include "kv_cache_manager/common/unittest.h"

namespace kv_cache_manager {

class FrequencySketchTest : public TESTBASE {};

TEST_F(FrequencySketchTest, EmptyReturnsZero) {
    FrequencySketch s;
    EXPECT_EQ(0u, s.RemoteCount("caller_a", 123));
    EXPECT_EQ(0u, s.Size());
}

TEST_F(FrequencySketchTest, ObserveIncrementsCounter) {
    FrequencySketch s;
    s.Observe("caller_a", 100);
    EXPECT_EQ(1u, s.RemoteCount("caller_a", 100));
    s.Observe("caller_a", 100);
    s.Observe("caller_a", 100);
    EXPECT_EQ(3u, s.RemoteCount("caller_a", 100));
}

TEST_F(FrequencySketchTest, DifferentCallerIsolated) {
    FrequencySketch s;
    s.Observe("caller_a", 100);
    s.Observe("caller_b", 100);
    s.Observe("caller_a", 100);
    EXPECT_EQ(2u, s.RemoteCount("caller_a", 100));
    EXPECT_EQ(1u, s.RemoteCount("caller_b", 100));
}

TEST_F(FrequencySketchTest, EmptyCallerIgnored) {
    FrequencySketch s;
    s.Observe("", 100);
    EXPECT_EQ(0u, s.RemoteCount("", 100));
    EXPECT_EQ(0u, s.Size());
}

TEST_F(FrequencySketchTest, ResetRemovesEntry) {
    FrequencySketch s;
    s.Observe("caller_a", 100);
    s.Observe("caller_a", 100);
    EXPECT_EQ(2u, s.RemoteCount("caller_a", 100));
    s.Reset("caller_a", 100);
    EXPECT_EQ(0u, s.RemoteCount("caller_a", 100));
    EXPECT_EQ(0u, s.Size());
}

TEST_F(FrequencySketchTest, LRUEvictsOldestEntryWhenFull) {
    FrequencySketch s(3); // 容量 3
    s.Observe("c", 1);
    s.Observe("c", 2);
    s.Observe("c", 3);
    EXPECT_EQ(3u, s.Size());
    s.Observe("c", 4);    // 触发淘汰
    EXPECT_EQ(3u, s.Size());
    EXPECT_EQ(0u, s.RemoteCount("c", 1)); // 最老的被淘汰
    EXPECT_EQ(1u, s.RemoteCount("c", 4));
}

TEST_F(FrequencySketchTest, MRUMovesEntryToFrontKeepingItAlive) {
    FrequencySketch s(3);
    s.Observe("c", 1);
    s.Observe("c", 2);
    s.Observe("c", 3);
    s.Observe("c", 1);    // 提升 (c,1) 到 MRU
    s.Observe("c", 4);    // 淘汰 (c,2)
    EXPECT_EQ(2u, s.RemoteCount("c", 1));
    EXPECT_EQ(0u, s.RemoteCount("c", 2));
    EXPECT_EQ(1u, s.RemoteCount("c", 3));
    EXPECT_EQ(1u, s.RemoteCount("c", 4));
}

TEST_F(FrequencySketchTest, InstanceCountersAndResetAreIndependent) {
    FrequencySketch sketch(4);
    sketch.Observe("reader", 42, "instance:a");
    sketch.Observe("reader", 42, "instance:a");
    sketch.Observe("reader", 42, "instance:b");
    EXPECT_EQ(2u, sketch.RemoteCount("reader", 42, "instance:a"));
    EXPECT_EQ(1u, sketch.RemoteCount("reader", 42, "instance:b"));
    EXPECT_EQ(0u, sketch.RemoteCount("reader", 42));
    sketch.Reset("reader", 42, "instance:a");
    EXPECT_EQ(0u, sketch.RemoteCount("reader", 42, "instance:a"));
    EXPECT_EQ(1u, sketch.RemoteCount("reader", 42, "instance:b"));
    // Delimiters in either identity cannot alias a different tuple.
    sketch.Observe("b:c", 7, "a");
    EXPECT_EQ(0u, sketch.RemoteCount("c", 7, "a:b"));
}

TEST_F(FrequencySketchTest, DecaysAcrossIdlePeriodsAndDoesNotRefreshOnRead) {
    int64_t now = 1000;
    FrequencySketch sketch(10, [&] { return now; });
    for (int i = 0; i < 8; ++i) sketch.Observe("a", 1, "instance", 100);
    now += 100;
    EXPECT_EQ(4u, sketch.RemoteCount("a", 1, "instance", 100));
    EXPECT_EQ(4u, sketch.RemoteCount("a", 1, "instance", 100));
    now += 100;
    sketch.Observe("a", 1, "instance", 100);
    EXPECT_EQ(3u, sketch.RemoteCount("a", 1, "instance", 100));
    now += 3200;
    EXPECT_EQ(0u, sketch.RemoteCount("a", 1, "instance", 100));
    sketch.Observe("a", 1, "instance", 100);
    EXPECT_EQ(1u, sketch.RemoteCount("a", 1, "instance", 100));
    EXPECT_EQ(0u, sketch.RemoteCount("a", 1, "other", 100));
    EXPECT_EQ(0u, sketch.RemoteCount("a", 1, "instance", 200));
    sketch.Observe("a", 1, "instance", 200);
    EXPECT_EQ(1u, sketch.RemoteCount("a", 1, "instance", 200));
}

} // namespace kv_cache_manager
