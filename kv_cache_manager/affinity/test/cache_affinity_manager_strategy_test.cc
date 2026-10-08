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
