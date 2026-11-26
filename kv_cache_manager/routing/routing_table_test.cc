#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/routing/routing_table.h"

namespace kv_cache_manager {
class RoutingTableTest : public TESTBASE {
public:
    RoutingTable table{std::make_shared<HashBucketPolicy>()};
    RoutingTable::Clock::time_point now{};
    const std::chrono::seconds ttl{10};
};

TEST_F(RoutingTableTest, VersionedAssignmentAndPartitionIsolation) {
    ASSERT_TRUE(table.Resolve({Placement::Hash, "a"}, now, ttl).empty());
    ASSERT_FALSE(table.Configure(0, 2));
    ASSERT_TRUE(table.Configure(1, 2));
    ASSERT_TRUE(table.Configure(1, 2));
    ASSERT_FALSE(table.Configure(1, 3));
    ASSERT_FALSE(table.Report({2, "a", 1, 1, "unused", true}, now));
    for (PartitionId p : {0u, 1u}) {
        ASSERT_TRUE(table.Report({p, "replica", 1, 1, std::to_string(p), true}, now));
    }
    ASSERT_EQ(2u, table.Resolve({Placement::All, {}}, now, ttl).size());
    ASSERT_EQ(0u, table.Resolve({Placement::RoleZero, {}}, now, ttl).at(0).partition);
    ASSERT_TRUE(table.Report({1, "replica", 1, 2, "1", false}, now));
    ASSERT_TRUE(table.Resolve({Placement::All, {}}, now, ttl).empty());
    ASSERT_EQ(0u, table.Resolve(RouteKey::PaceAllocate(), now, ttl).at(0).partition);
    ASSERT_TRUE(table.Report({1, "replica", 1, 3, "1", true}, now));
    const auto start = table.Resolve(RouteKey::Session("conversation"), now, ttl);
    ASSERT_EQ(1u, start.size());
    const auto finish = table.Resolve(RouteKey::Session("conversation"), now, ttl);
    ASSERT_EQ(1u, finish.size());
    EXPECT_EQ(start.front().endpoint, finish.front().endpoint);
    EXPECT_EQ(start.front().partition, table.Resolve(RouteKey::Embedding("conversation"), now, ttl).at(0).partition);
    ASSERT_TRUE(table.Resolve({Placement::Hash, ""}, now, ttl).empty());
    ASSERT_FALSE(table.Configure(2, 0));
    ASSERT_TRUE(table.Configure(2, 2));
    ASSERT_FALSE(table.Configure(1, 2));
    ASSERT_FALSE(table.Report({0, "replica", 1, 2, "old", true}, now));
    ASSERT_TRUE(table.Resolve(RouteKey::Session("conversation"), now, ttl).empty());
}

TEST_F(RoutingTableTest, ConflictingLeadersAndOutOfOrderHeartbeats) {
    ASSERT_TRUE(table.Configure(12, 1));
    ASSERT_TRUE(table.Report({0, "first", 12, 1, "A", true}, now));
    ASSERT_TRUE(table.Report({0, "second", 12, 1, "B", true}, now));
    EXPECT_TRUE(table.Resolve({Placement::Hash, "key"}, now, ttl).empty());
    ASSERT_TRUE(table.Report({0, "first", 12, 2, "A", false}, now));
    EXPECT_FALSE(table.Report({0, "first", 12, 1, "A", true}, now));
    const auto active = table.Resolve({Placement::Hash, "key"}, now, ttl);
    ASSERT_EQ(1u, active.size());
    EXPECT_EQ("B", active.front().endpoint);
    EXPECT_TRUE(table.Resolve({Placement::Hash, "key"}, now + ttl, ttl).empty());
    table.Expire(now + ttl, ttl);
    EXPECT_TRUE(table.Resolve({Placement::Hash, "key"}, now + ttl, ttl).empty());
}
} // namespace kv_cache_manager
