// affinity v1 F1: CacheAffinityManager.GetStrategy 优先级链 + JSON 解析单测
//
// 验证：
//   1. 三层都空 → 返回内置默认 Noop（未配置即零行为）
//   2. 仅 process 配 noop → 返回 Noop
//   3. instance 配 noop 覆盖 process local_replica → 返回 Noop
//   4. 解析失败时按层级 fall through

#include <memory>
#include <string>

#include "kv_cache_manager/affinity/affinity_strategy.h"
#include "kv_cache_manager/affinity/cache_affinity_manager.h"
#include "kv_cache_manager/common/unittest.h"

namespace kv_cache_manager {

class CacheAffinityManagerStrategyTest : public TESTBASE {};

TEST_F(CacheAffinityManagerStrategyTest, EmptyAllLayersFallsBackToNoop) {
    CacheAffinityManager m;
    auto s = m.GetStrategy("", "");
    ASSERT_NE(nullptr, s);
    EXPECT_EQ("noop", s->Name());
}

TEST_F(CacheAffinityManagerStrategyTest, ProcessLevelNoopAppliesWhenOverridesAreEmpty) {
    CacheAffinityManager m;
    std::string err;
    ASSERT_TRUE(m.LoadProcessStrategyFromJsonString(R"({"type":"noop"})", &err));
    auto s = m.GetStrategy("", "");
    ASSERT_NE(nullptr, s);
    EXPECT_EQ("noop", s->Name());
}

TEST_F(CacheAffinityManagerStrategyTest, InstanceLevelOverridesProcess) {
    CacheAffinityManager m;
    ASSERT_TRUE(m.LoadProcessStrategyFromJsonString(R"({"type":"local_replica"})"));
    auto s = m.GetStrategy(/*instance=*/R"({"type":"noop"})", /*group=*/"");
    ASSERT_NE(nullptr, s);
    EXPECT_EQ("noop", s->Name());
}

TEST_F(CacheAffinityManagerStrategyTest, InstanceFallsThroughToGroupWhenInstanceParseFails) {
    CacheAffinityManager m;
    ASSERT_TRUE(m.LoadProcessStrategyFromJsonString(R"({"type":"noop"})"));
    auto s = m.GetStrategy(/*instance=*/"not_json", /*group=*/R"({"type":"local_replica"})");
    ASSERT_NE(nullptr, s);
    EXPECT_EQ("local_replica", s->Name());
}

TEST_F(CacheAffinityManagerStrategyTest, GroupFallsThroughToProcessWhenGroupParseFails) {
    CacheAffinityManager m;
    ASSERT_TRUE(m.LoadProcessStrategyFromJsonString(R"({"type":"noop"})"));
    auto s = m.GetStrategy(/*instance=*/"", /*group=*/"not_json");
    ASSERT_NE(nullptr, s);
    EXPECT_EQ("noop", s->Name()); // process 命中
}

TEST_F(CacheAffinityManagerStrategyTest, RepeatedGetStrategyReturnsSameSharedPtr) {
    CacheAffinityManager m;
    ASSERT_TRUE(m.LoadProcessStrategyFromJsonString(R"({"type":"noop"})"));
    auto s1 = m.GetStrategy("", "");
    auto s2 = m.GetStrategy("", "");
    EXPECT_EQ(s1.get(), s2.get()); // memoize 生效
}

TEST_F(CacheAffinityManagerStrategyTest, InvalidOverridesCannotMaskLowerPriorityStrategies) {
    CacheAffinityManager m;
    ASSERT_TRUE(m.LoadProcessStrategyFromJsonString(R"({"type":"noop"})"));
    const auto process = m.GetStrategy("", "");
    const std::string group_json = R"({"type":"local_replica","enabled_aspects":{"eviction":false}})";
    const auto group = m.GetStrategy("", group_json);
    ASSERT_NE(group.get(), process.get());
    const char *invalid_overrides[] = {
        R"({"type":"local_replica","enabled_aspects":{"eviction":"false"}})",
        R"({"type":"local_replica","read":{"on_miss":{"enabled":0}}})",
        R"({"type":"local_replica","read":{"on_miss":{"suppression_window_ms":-1}}})",
        R"({"type":"local_replica","read":{"on_miss":{"caller_capacity_threshold":0.1,"caller_capacity_buffer":0.2}}})",
        R"({"type":"local_replica","write":{"ops":{"limit":0}}})",
        R"({"type":"local_replica","write":{"ops":[]}})",
        R"({"type":"local_replica","eviction":{"ops":[{"op":"node_water_level","low":0.9}]}})",
        R"({"type":"local_replica","replica_limits":{"min_retained_replicas":2,"max_replicas_per_key":1}})",
    };
    for (const auto *invalid : invalid_overrides) {
        SCOPED_TRACE(invalid);
        for (int retry = 0; retry < 2; ++retry) {
            EXPECT_EQ(group.get(), m.GetStrategy(invalid, group_json).get());
            EXPECT_EQ(process.get(), m.GetStrategy("", invalid).get());
            EXPECT_EQ(process.get(), m.GetStrategy(invalid, invalid).get());
        }
        std::string err;
        EXPECT_FALSE(m.LoadProcessStrategyFromJsonString(invalid, &err));
        EXPECT_FALSE(err.empty());
        EXPECT_EQ(process.get(), m.GetStrategy("", "").get()); // Failed replacement retains the current policy.
    }
}

// affinity v1 C9: SetGloballyDisabled(true) ⇒ 强制返回 Noop，无视任何配置
TEST_F(CacheAffinityManagerStrategyTest, GloballyDisabledForcesNoop) {
    CacheAffinityManager m;
    // 配置 LocalReplica，但 globally_disabled 后应该返回 Noop
    ASSERT_TRUE(m.LoadProcessStrategyFromJsonString(R"({"type":"local_replica"})"));
    {
        auto s = m.GetStrategy("", "");
        EXPECT_EQ("local_replica", s->Name());
    }
    m.SetGloballyDisabled(true);
    {
        auto s = m.GetStrategy("", "");
        EXPECT_EQ("noop", s->Name()); // noop 无任何亲和行为
    }
    // 关闭后恢复
    m.SetGloballyDisabled(false);
    {
        auto s = m.GetStrategy("", "");
        EXPECT_EQ("local_replica", s->Name());
    }
}

} // namespace kv_cache_manager
