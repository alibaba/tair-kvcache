// affinity v1 F0: 框架骨架的最小可执行证据
//
//   1. NoopAffinityStrategy 4 个 toggle 全 false（任何插入点 short-circuit）
//   2. NoopAffinityStrategy 所有方法可被调用且返回 default 值（保险）
//   3. StrategyFactory 解析 "noop" → NoopAffinityStrategy
//   4. StrategyFactory 对空 / 错 JSON / 未知 type 安全返回 nullptr

#include <memory>
#include <string>

#include "kv_cache_manager/affinity/affinity_strategy.h"
#include "kv_cache_manager/affinity/local_replica_strategy.h"
#include "kv_cache_manager/affinity/noop_strategy.h"
#include "kv_cache_manager/affinity/strategy_factory.h"
#include "kv_cache_manager/common/unittest.h"

namespace kv_cache_manager {

class StrategyFrameworkTest : public TESTBASE {};

namespace {
// toggle 现在是算法内部细节，不在公共接口暴露；测试通过 downcast 到具体算法
// 的 params() 验证 JSON enabled_aspects 解析。
const LocalReplicaAffinityStrategy::Params *LocalReplicaParams(const std::shared_ptr<AffinityStrategy> &s) {
    auto *lr = dynamic_cast<const LocalReplicaAffinityStrategy *>(s.get());
    return lr != nullptr ? &lr->params() : nullptr;
}
} // namespace

TEST_F(StrategyFrameworkTest, NoopMethodsReturnSafeDefaults) {
    NoopAffinityStrategy s;
    EXPECT_EQ("noop", s.Name());
    StrategyContext ctx;

    WriteDecision wdec = s.ResolveWrite({"node_a", "node_b"}, ctx);
    EXPECT_EQ(AffinityStatus::kOk, wdec.status);
    EXPECT_TRUE(wdec.hints.preferred_node_ids.empty());

    ReadRequest req;
    ReadDecision dec = s.ResolveRead(req, ctx);
    EXPECT_TRUE(dec.picked_specs.empty());
    EXPECT_TRUE(dec.side_effects.empty());

    EXPECT_TRUE(s.ResolveEviction(ctx).empty());
}

TEST_F(StrategyFrameworkTest, FactoryParsesNoop) {
    std::string err;
    auto s = StrategyFactory::ParseJsonString(R"({"type":"noop"})", nullptr, &err);
    ASSERT_NE(nullptr, s);
    EXPECT_EQ("noop", s->Name());
    EXPECT_TRUE(err.empty());
}

TEST_F(StrategyFrameworkTest, FactoryAcceptsWrappedNoop) {
    auto s = StrategyFactory::ParseJsonString(R"({"strategy":{"type":"noop"}})");
    ASSERT_NE(nullptr, s);
    EXPECT_EQ("noop", s->Name());
}

TEST_F(StrategyFrameworkTest, FactoryReturnsNullOnEmpty) {
    std::string err;
    auto s = StrategyFactory::ParseJsonString("", nullptr, &err);
    EXPECT_EQ(nullptr, s);
    EXPECT_FALSE(err.empty());
}

TEST_F(StrategyFrameworkTest, FactoryReturnsNullOnBadJson) {
    std::string err;
    auto s = StrategyFactory::ParseJsonString("not_json_at_all", nullptr, &err);
    EXPECT_EQ(nullptr, s);
    EXPECT_FALSE(err.empty());
}

TEST_F(StrategyFrameworkTest, FactoryReturnsNullOnUnknownType) {
    std::string err;
    auto s = StrategyFactory::ParseJsonString(R"({"type":"unicorn"})", nullptr, &err);
    EXPECT_EQ(nullptr, s);
    EXPECT_NE(std::string::npos, err.find("unknown"));
}

TEST_F(StrategyFrameworkTest, FactoryParsesLocalReplicaWithToggles) {
    auto s = StrategyFactory::ParseJsonString(R"({
        "type": "local_replica",
        "enabled_aspects": {
            "write": false,
            "read": true,
            "eviction": false
        }
    })");
    ASSERT_NE(nullptr, s);
    EXPECT_EQ("local_replica", s->Name());
    const auto *p = LocalReplicaParams(s);
    ASSERT_NE(nullptr, p);
    EXPECT_FALSE(p->enable_write);
    EXPECT_TRUE(p->enable_read);
    EXPECT_FALSE(p->enable_eviction);
}

// 新 JSON 层级（write.ops + read.on_miss + eviction.ops）解析
TEST_F(StrategyFrameworkTest, FactoryParsesPerAspectOps) {
    auto s = StrategyFactory::ParseJsonString(R"({
        "type": "local_replica",
        "write": {
            "ops": { "prefer_local": {"on_miss": "passthrough"} }
        },
        "read": {
            "on_miss": {
                "enabled": true,
                "replication_hot_threshold": 7,
                "caller_capacity_threshold": 0.9
            }
        },
        "eviction": {
            "ops": [
                {"op": "node_water_level", "threshold": 0.7, "critical": 0.92}
            ]
        }
    })");
    ASSERT_NE(nullptr, s);
    EXPECT_EQ("local_replica", s->Name());
    const auto *p = LocalReplicaParams(s);
    ASSERT_NE(nullptr, p);
    EXPECT_TRUE(p->enable_write);
    EXPECT_TRUE(p->enable_read);
    EXPECT_TRUE(p->enable_eviction);
}

// read.on_miss.enabled = false 关闭复制触发但保留 read.pick 本地优先
TEST_F(StrategyFrameworkTest, FactoryParsesOnMissDisabled) {
    auto s = StrategyFactory::ParseJsonString(R"({
        "type": "local_replica",
        "read": { "on_miss": { "enabled": false } }
    })");
    ASSERT_NE(nullptr, s);
    EXPECT_EQ("local_replica", s->Name());
    const auto *p = LocalReplicaParams(s);
    ASSERT_NE(nullptr, p);
    EXPECT_TRUE(p->enable_read); // 一级仍 on
    EXPECT_FALSE(p->enable_on_miss);
    // 子开关在 Params 内，行为另见 LocalReplicaStrategyTest
}

} // namespace kv_cache_manager
