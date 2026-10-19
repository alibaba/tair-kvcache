// affinity v1 F0: 框架骨架的最小可执行证据
//
//   1. NoopAffinityStrategy 4 个 toggle 全 false（任何插入点 short-circuit）
//   2. NoopAffinityStrategy 所有方法可被调用且返回 default 值（保险）
//   3. StrategyFactory 解析 "noop" → NoopAffinityStrategy
//   4. StrategyFactory 对空 / 错 JSON / 未知 type 安全返回 nullptr

#include <memory>
#include <string>
#include <vector>

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
    auto s = StrategyFactory::ParseJsonString(R"({"type":"noop"})", nullptr, nullptr, &err);
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
    auto s = StrategyFactory::ParseJsonString("", nullptr, nullptr, &err);
    EXPECT_EQ(nullptr, s);
    EXPECT_FALSE(err.empty());
}

TEST_F(StrategyFrameworkTest, FactoryReturnsNullOnBadJson) {
    std::string err;
    auto s = StrategyFactory::ParseJsonString("not_json_at_all", nullptr, nullptr, &err);
    EXPECT_EQ(nullptr, s);
    EXPECT_FALSE(err.empty());
}

TEST_F(StrategyFrameworkTest, FactoryReturnsNullOnUnknownType) {
    std::string err;
    auto s = StrategyFactory::ParseJsonString(R"({"type":"unicorn"})", nullptr, nullptr, &err);
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
                {"op": "node_water_level", "threshold": 0.7}
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

TEST_F(StrategyFrameworkTest, FactoryParsesSuppressionWindow) {
    {
        auto s = StrategyFactory::ParseJsonString(R"({"type":"local_replica"})");
        ASSERT_NE(nullptr, s);
        const auto *p = LocalReplicaParams(s);
        ASSERT_NE(nullptr, p);
        EXPECT_EQ(60000u, p->suppression_window_ms);
    }
    {
        auto s = StrategyFactory::ParseJsonString(R"({
            "type": "local_replica",
            "read": { "on_miss": { "suppression_window_ms": 30000 } }
        })");
        ASSERT_NE(nullptr, s);
        const auto *p = LocalReplicaParams(s);
        ASSERT_NE(nullptr, p);
        EXPECT_EQ(30000u, p->suppression_window_ms);
    }
    {
        auto s = StrategyFactory::ParseJsonString(R"({
            "type": "local_replica",
            "read": { "on_miss": { "suppression_window_ms": 0 } }
        })");
        ASSERT_NE(nullptr, s);
        const auto *p = LocalReplicaParams(s);
        ASSERT_NE(nullptr, p);
        EXPECT_EQ(0u, p->suppression_window_ms);
    }
}

TEST_F(StrategyFrameworkTest, RejectsMalformedConfigurationBlocksAndPipeline) {
    const char *invalid_fields[] = {
        R"("strategy":false)",
        R"("enabled_aspects":[])",
        R"("replica_limits":null)",
        R"("write":false)",
        R"("read":[])",
        R"("eviction":true)",
        R"("write":{"ops":[]})",
        R"("write":{"ops":{"limit":0}})",
        R"("write":{"ops":{"prefer_local":{"on_miss":"ignore"}}})",
        R"("write":{"ops":{"prefer_local":{"same_supernode":1}}})",
        R"("read":{"on_miss":false})",
        R"("eviction":{"ops":{}})",
        R"("eviction":{"ops":[null]})",
        R"("eviction":{"ops":[{}]})",
        R"("eviction":{"ops":[{"op":3}]})",
        R"("eviction":{"ops":[{"op":"unknown"}]})",
    };
    for (const auto *fields : invalid_fields) {
        SCOPED_TRACE(fields);
        std::string err;
        EXPECT_EQ(nullptr,
                  StrategyFactory::ParseJsonString(
                      std::string(R"({"type":"local_replica",)") + fields + "}", nullptr, nullptr, &err));
        EXPECT_FALSE(err.empty());
    }
    std::string err;
    EXPECT_EQ(nullptr,
              StrategyFactory::ParseJsonString(
                  R"({"type":"local_replica","write":{"ops":{"limit":0}}})", nullptr, nullptr, &err));
    EXPECT_NE(std::string::npos, err.find("limit")); // Preserve the nested parser's diagnostic.
}

TEST_F(StrategyFrameworkTest, RejectsWrongTypesAndOutOfRangeValues) {
    struct FieldCase {
        const char *before;
        const char *after;
        std::vector<const char *> invalid_values;
    };
    const FieldCase cases[] = {
        {R"("enabled_aspects":{"write":)", "}", {"0", "\"false\"", "null"}},
        {R"("enabled_aspects":{"read":)", "}", {"1", "\"false\""}},
        {R"("enabled_aspects":{"eviction":)", "}", {"1", "\"false\""}},
        {R"("read":{"on_miss":{"enabled":)", "}}", {"0", "\"false\""}},
        {R"("read":{"on_miss":{"replication_hot_threshold":)", "}}", {"-1", "1.5", "4294967296", "true"}},
        {R"("read":{"on_miss":{"heat_half_life_ms":)", "}}", {"-1", "1.5", "4294967296"}},
        {R"("read":{"on_miss":{"suppression_window_ms":)", "}}", {"-1", "1.5", "4294967296"}},
        {R"("read":{"on_miss":{"max_replication_bytes":)", "}}", {"-1", "1.5", "18446744073709551616"}},
        {R"("read":{"on_miss":{"min_benefit_ratio":)", "}}", {"-1", "\"2\"", "true", "1e400"}},
        {R"("read":{"on_miss":{"prefix_bonus":)", "}}", {"-1", "\"2\"", "null"}},
        {R"("read":{"on_miss":{"caller_capacity_threshold":)", "}}", {"-0.1", "1.01", "0.04", "\"0.9\""}},
        {R"("read":{"on_miss":{"caller_capacity_buffer":)", "}}", {"-0.1", "0.86", "true"}},
        {R"("replica_limits":{"max_replicas_per_key":)", "}", {"-1", "1.5", "4294967296"}},
        {R"("replica_limits":{"min_retained_replicas":)", "}", {"-1", "1.5", "4294967296"}},
        {R"("replica_limits":{"max_instance_bytes":)", "}", {"-1", "1.5", "18446744073709551616"}},
        {R"("eviction":{"ops":[{"op":"node_water_level","threshold":)", "}]}", {"-0.1", "1.01", "0.69", "true"}},
        {R"("eviction":{"ops":[{"op":"node_water_level","low":)", "}]}", {"-0.1", "0.86", "\"0.7\""}},
        {R"("replica_limits":{"max_replicas_per_key":2,"min_retained_replicas":)", "}", {"3"}},
    };
    for (const auto &test : cases) {
        for (const auto *value : test.invalid_values) {
            const auto json = std::string(R"({"type":"local_replica",)") + test.before + value + test.after + "}";
            SCOPED_TRACE(json);
            std::string err;
            EXPECT_EQ(nullptr, StrategyFactory::ParseJsonString(json, nullptr, nullptr, &err));
            EXPECT_FALSE(err.empty());
        }
    }
}

TEST_F(StrategyFrameworkTest, AcceptsZeroAndMaximumIntegerBoundaries) {
    for (const auto *value : {"0", "4294967295"}) {
        const auto json = std::string(R"({"type":"local_replica","read":{"on_miss":{)") + R"("heat_half_life_ms":)" +
                          value + R"(,"replication_hot_threshold":)" + value + R"(,"suppression_window_ms":)" + value +
                          "}}}";
        SCOPED_TRACE(json);
        EXPECT_NE(nullptr, StrategyFactory::ParseJsonString(json));
    }
    auto strategy = StrategyFactory::ParseJsonString(R"({
        "type":"local_replica",
        "replica_limits":{"max_replicas_per_key":0,"min_retained_replicas":0,"max_instance_bytes":18446744073709551615},
        "read":{"on_miss":{"max_replication_bytes":18446744073709551615,"caller_capacity_threshold":0,
                           "caller_capacity_buffer":0,"min_benefit_ratio":0,"prefix_bonus":0}},
        "eviction":{"ops":[{"op":"node_water_level","threshold":1,"low":1}]}
    })");
    ASSERT_NE(nullptr, strategy);
    const auto *p = LocalReplicaParams(strategy);
    ASSERT_NE(nullptr, p);
    EXPECT_EQ(UINT64_MAX, p->max_replication_bytes);
    EXPECT_EQ(UINT64_MAX, p->replica_limits.max_instance_bytes);
    EXPECT_EQ(0u, p->replica_limits.min_retained_replicas);
    EXPECT_DOUBLE_EQ(0, p->caller_capacity_threshold - p->caller_capacity_buffer);
    EXPECT_DOUBLE_EQ(1, p->node_water_low);
}

TEST_F(StrategyFrameworkTest, LegacyCriticalIsIgnoredWithoutChangingEvictionPolicy) {
    for (const auto *critical : {"0.1", "0.99", "null", "\"legacy\""}) {
        SCOPED_TRACE(critical);
        const auto json = std::string(R"({"type":"local_replica","eviction":{"ops":[{)") +
                          R"("op":"node_water_level","threshold":0.85,"low":0.70,"critical":)" + critical + "}]}}";
        auto strategy = StrategyFactory::ParseJsonString(json);
        ASSERT_NE(nullptr, strategy);
        StrategyContext ctx;
        NodeMetrics node;
        node.node_id = "node";
        node.load_ratio = 0.9;
        ctx.all_nodes.push_back(node);
        EXPECT_EQ(1u, strategy->ResolveEviction(ctx).count("node"));
        ctx.all_nodes.front().load_ratio = 0.8;
        EXPECT_TRUE(strategy->ResolveEviction(ctx).empty());
    }
}

} // namespace kv_cache_manager
