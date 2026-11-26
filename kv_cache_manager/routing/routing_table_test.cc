#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/routing/routing_table.h"

namespace kv_cache_manager {
class RoutingTableTest : public TESTBASE {
public:
    RoutingTable table{std::make_shared<InstanceBucketPolicy>()};
    RoutingTable::Clock::time_point now{};
    const std::chrono::seconds ttl{10};
};

TEST_F(RoutingTableTest, VersionedAssignmentAndPartitionIsolation) {
    ASSERT_FALSE(table.Resolve({"Get", "a"}, now, ttl));
    ASSERT_FALSE(table.ApplyAssignment({0, {0, 1}}));
    ASSERT_TRUE(table.ApplyAssignment({1, {0, 1}}));
    ASSERT_TRUE(table.ApplyAssignment({1, {0, 1}}));
    ASSERT_FALSE(table.ApplyAssignment({1, {1, 0}}));
    ASSERT_FALSE(table.Report({2, "a", 1, 1, "unused", true}, now));
    for (PartitionId p : {0u, 1u}) {
        ASSERT_TRUE(table.Report({p, "replica", 1, 1, std::to_string(p), true}, now));
    }
    const auto start = table.Resolve({"StartWriteCache", "instance"}, now, ttl);
    ASSERT_TRUE(start);
    const auto finish = table.Resolve({"FinishWriteCache", "instance"}, now, ttl);
    ASSERT_TRUE(finish);
    EXPECT_EQ(start->endpoint, finish->endpoint);
    ASSERT_FALSE(table.Resolve({"Get", ""}, now, ttl));
    ASSERT_FALSE(table.ApplyAssignment({2, {1}}));
    ASSERT_TRUE(table.ApplyAssignment({2, {1, 0}}));
    ASSERT_FALSE(table.ApplyAssignment({1, {0, 1}}));
    ASSERT_FALSE(table.Report({0, "replica", 1, 2, "old", true}, now));
    ASSERT_FALSE(table.Resolve({"Get", "instance"}, now, ttl));
}

TEST_F(RoutingTableTest, ConflictingLeadersAndOutOfOrderHeartbeats) {
    ASSERT_TRUE(table.ApplyAssignment({12, {9}}));
    ASSERT_TRUE(table.Report({9, "first", 12, 1, "A", true}, now));
    ASSERT_TRUE(table.Report({9, "second", 12, 1, "B", true}, now));
    EXPECT_FALSE(table.Resolve({"Get", "key"}, now, ttl));
    ASSERT_TRUE(table.Report({9, "first", 12, 2, "A", false}, now));
    EXPECT_FALSE(table.Report({9, "first", 12, 1, "A", true}, now));
    const auto active = table.Resolve({"Get", "key"}, now, ttl);
    ASSERT_TRUE(active);
    EXPECT_EQ("B", active->endpoint);
    EXPECT_FALSE(table.Resolve({"Get", "key"}, now + ttl, ttl));
    table.Expire(now + ttl, ttl);
    EXPECT_FALSE(table.Resolve({"Get", "key"}, now + ttl, ttl));
}
} // namespace kv_cache_manager
