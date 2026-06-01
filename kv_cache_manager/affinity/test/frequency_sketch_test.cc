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

} // namespace kv_cache_manager
