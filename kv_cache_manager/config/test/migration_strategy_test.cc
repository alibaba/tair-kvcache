#include <limits>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/config/cache_config.h"
#include "kv_cache_manager/config/migration_strategy.h"

using namespace kv_cache_manager;

class MigrationStrategyTest : public TESTBASE {
public:
    void SetUp() override {}
    void TearDown() override {}
};

// 解析一条完整的迁移规则，校验各字段（含嵌套 methods 与枚举 retention）
TEST_F(MigrationStrategyTest, TestParseFull) {
    MigrationStrategy strategy;
    std::string json = R"({
        "source_storage_name": "pace_mempool_01",
        "target_storage_name": "pace_ssd_01",
        "trigger_threshold": 0.70,
        "methods": {
            "copy": { "enabled": true },
            "mark": { "enabled": true, "timeout_ms": 60000 }
        },
        "retention": 1
    })";
    ASSERT_TRUE(strategy.FromJsonString(json));
    ASSERT_EQ("pace_mempool_01", strategy.source_storage_name());
    ASSERT_EQ("pace_ssd_01", strategy.target_storage_name());
    ASSERT_DOUBLE_EQ(0.70, strategy.trigger_threshold());
    ASSERT_TRUE(strategy.methods().copy().enabled());
    ASSERT_TRUE(strategy.methods().mark().enabled());
    ASSERT_EQ(60000, strategy.methods().mark().timeout_ms());
    ASSERT_EQ(MigrationRetention::MIGRATION_RETENTION_DELETE_SOURCE, strategy.retention());

    std::string invalid_fields;
    ASSERT_TRUE(strategy.ValidateRequiredFields(invalid_fields)) << invalid_fields;
}

// ToJsonString -> FromJsonString 往返一致
TEST_F(MigrationStrategyTest, TestRoundTrip) {
    MigrationStrategy strategy;
    strategy.set_source_storage_name("hot");
    strategy.set_target_storage_name("cold");
    strategy.set_trigger_threshold(0.6);
    MigrationMethods methods;
    methods.mutable_copy().set_enabled(false);
    methods.mutable_mark().set_enabled(true);
    methods.mutable_mark().set_timeout_ms(12345);
    strategy.set_methods(methods);
    strategy.set_retention(MigrationRetention::MIGRATION_RETENTION_KEEP_BOTH);
    MigrationAdmissionConfig admission;
    admission.set_mode(MigrationAdmissionMode::SHADOW);
    auto policy = std::make_shared<MigrationAdmissionPolicyConfig>();
    policy->set_recent_access(std::make_shared<RecentAccessAdmissionConfig>(3600));
    admission.set_policies({policy});
    strategy.set_admission(admission);

    MigrationStrategy parsed;
    ASSERT_TRUE(parsed.FromJsonString(strategy.ToJsonString()));
    ASSERT_EQ("hot", parsed.source_storage_name());
    ASSERT_EQ("cold", parsed.target_storage_name());
    ASSERT_DOUBLE_EQ(0.6, parsed.trigger_threshold());
    ASSERT_FALSE(parsed.methods().copy().enabled());
    ASSERT_TRUE(parsed.methods().mark().enabled());
    ASSERT_EQ(12345, parsed.methods().mark().timeout_ms());
    ASSERT_EQ(MigrationRetention::MIGRATION_RETENTION_KEEP_BOTH, parsed.retention());
    ASSERT_EQ(MigrationAdmissionMode::SHADOW, parsed.admission().mode());
    ASSERT_EQ(1u, parsed.admission().policies().size());
    ASSERT_EQ(3600, parsed.admission().policies()[0]->recent_access()->window_seconds());
}

TEST_F(MigrationStrategyTest, TestAdmissionDefaultsAndValidation) {
    MigrationStrategy disabled;
    disabled.set_source_storage_name("hot");
    disabled.set_target_storage_name("cold");
    disabled.set_trigger_threshold(0.7);
    disabled.mutable_methods().mutable_mark().set_enabled(true);
    std::string invalid_fields;
    ASSERT_TRUE(disabled.ValidateRequiredFields(invalid_fields)) << invalid_fields;
    ASSERT_EQ(MigrationAdmissionMode::DISABLED, disabled.admission().mode());
    ASSERT_TRUE(disabled.admission().policies().empty());

    MigrationAdmissionConfig enabled_without_policy;
    enabled_without_policy.set_mode(MigrationAdmissionMode::ENFORCE);
    invalid_fields.clear();
    EXPECT_FALSE(enabled_without_policy.ValidateRequiredFields(invalid_fields));
    EXPECT_NE(std::string::npos, invalid_fields.find("policies_count"));

    MigrationAdmissionConfig invalid_window;
    invalid_window.set_mode(MigrationAdmissionMode::SHADOW);
    auto policy = std::make_shared<MigrationAdmissionPolicyConfig>();
    policy->set_recent_access(std::make_shared<RecentAccessAdmissionConfig>(0));
    invalid_window.set_policies({policy});
    invalid_fields.clear();
    EXPECT_FALSE(invalid_window.ValidateRequiredFields(invalid_fields));
    EXPECT_NE(std::string::npos, invalid_fields.find("window_seconds"));

    MigrationAdmissionConfig valid;
    valid.set_mode(MigrationAdmissionMode::ENFORCE);
    policy->set_recent_access(std::make_shared<RecentAccessAdmissionConfig>(60));
    valid.set_policies({policy});
    invalid_fields.clear();
    EXPECT_TRUE(valid.ValidateRequiredFields(invalid_fields)) << invalid_fields;
}

// 校验：各类非法配置都应被拒绝
TEST_F(MigrationStrategyTest, TestValidation) {
    auto make_valid = []() {
        MigrationStrategy s;
        s.set_source_storage_name("hot");
        s.set_target_storage_name("cold");
        s.set_trigger_threshold(0.7);
        MigrationMethods m;
        m.mutable_copy().set_enabled(true);
        s.set_methods(m);
        s.set_retention(MigrationRetention::MIGRATION_RETENTION_DELETE_SOURCE);
        return s;
    };

    {
        auto s = make_valid();
        std::string f;
        ASSERT_TRUE(s.ValidateRequiredFields(f)) << f;
    }
    { // 源 storage 为空
        auto s = make_valid();
        s.set_source_storage_name("");
        std::string f;
        ASSERT_FALSE(s.ValidateRequiredFields(f));
    }
    { // 目标 storage 为空
        auto s = make_valid();
        s.set_target_storage_name("");
        std::string f;
        ASSERT_FALSE(s.ValidateRequiredFields(f));
    }
    { // 源与目标相同
        auto s = make_valid();
        s.set_target_storage_name("hot");
        std::string f;
        ASSERT_FALSE(s.ValidateRequiredFields(f));
    }
    { // 阈值越界（<=0）
        auto s = make_valid();
        s.set_trigger_threshold(0.0);
        std::string f;
        ASSERT_FALSE(s.ValidateRequiredFields(f));
    }
    { // 阈值越界（>=1）
        auto s = make_valid();
        s.set_trigger_threshold(1.0);
        std::string f;
        ASSERT_FALSE(s.ValidateRequiredFields(f));
    }
    { // 阈值必须是有限数，NaN / inf 都应拒绝
        auto s = make_valid();
        s.set_trigger_threshold(std::numeric_limits<double>::quiet_NaN());
        std::string f;
        ASSERT_FALSE(s.ValidateRequiredFields(f));

        s = make_valid();
        s.set_trigger_threshold(std::numeric_limits<double>::infinity());
        f.clear();
        ASSERT_FALSE(s.ValidateRequiredFields(f));
    }
    { // 没有启用任何执行方式
        auto s = make_valid();
        MigrationMethods m; // copy/mark 均 false
        s.set_methods(m);
        std::string f;
        ASSERT_FALSE(s.ValidateRequiredFields(f));
    }
    { // 启用 copy 但未指定 retention
        auto s = make_valid();
        s.set_retention(MigrationRetention::MIGRATION_RETENTION_UNSPECIFIED);
        std::string f;
        ASSERT_FALSE(s.ValidateRequiredFields(f));
    }
    { // 启用 copy 但 retention 枚举值非法
        auto s = make_valid();
        s.set_retention(static_cast<MigrationRetention>(999));
        std::string f;
        ASSERT_FALSE(s.ValidateRequiredFields(f));
    }
    { // 仅 mark：未指定 retention 也合法（mark 不删源端）
        MigrationStrategy s;
        s.set_source_storage_name("hot");
        s.set_target_storage_name("cold");
        s.set_trigger_threshold(0.7);
        MigrationMethods m;
        m.mutable_mark().set_enabled(true);
        s.set_methods(m);
        std::string f;
        ASSERT_TRUE(s.ValidateRequiredFields(f)) << f;
    }
    { // 仅 mark：未指定 retention 合法，但非法枚举仍应拒绝
        MigrationStrategy s;
        s.set_source_storage_name("hot");
        s.set_target_storage_name("cold");
        s.set_trigger_threshold(0.7);
        MigrationMethods m;
        m.mutable_mark().set_enabled(true);
        s.set_methods(m);
        s.set_retention(static_cast<MigrationRetention>(999));
        std::string f;
        ASSERT_FALSE(s.ValidateRequiredFields(f));
    }
    { // mark timeout 必须为正数
        auto s = make_valid();
        s.mutable_methods().mutable_mark().set_enabled(true);
        s.mutable_methods().mutable_mark().set_timeout_ms(0);
        std::string f;
        ASSERT_FALSE(s.ValidateRequiredFields(f));
    }
}

// CacheConfig 内 migration_config 列表的解析与往返
TEST_F(MigrationStrategyTest, TestInCacheConfig) {
    CacheConfig cache_config;
    std::string json = R"({
        "cache_prefer_strategy": 5,
        "reclaim_strategy": { "source_storage_name": "pace_mempool_01" },
        "meta_indexer_config": { "meta_storage_backend_config": { "storage_type": "local" } },
        "migration_config": {
            "copy_max_concurrency": 6,
            "mark_clear_policy": 1,
            "strategies": [
                {
                    "source_storage_name": "pace_mempool_01",
                    "target_storage_name": "pace_ssd_01",
                    "trigger_threshold": 0.70,
                    "methods": { "copy": { "enabled": true }, "mark": { "enabled": false } },
                    "retention": 1
                },
                {
                    "source_storage_name": "pace_mempool_02",
                    "target_storage_name": "pace_ssd_02",
                    "trigger_threshold": 0.60,
                    "methods": { "copy": { "enabled": true }, "mark": { "enabled": true, "timeout_ms": 120000 } },
                    "retention": 2
                }
            ]
        }
    })";
    ASSERT_TRUE(cache_config.FromJsonString(json));
    ASSERT_EQ(6, cache_config.migration_copy_max_concurrency());
    ASSERT_EQ(MigrationMarkClearPolicy::CLEAR_ON_FULL_BLOCK_COVERED, cache_config.migration_mark_clear_policy());
    ASSERT_EQ(2u, cache_config.migration_strategies().size());
    ASSERT_EQ("pace_mempool_01", cache_config.migration_strategies()[0]->source_storage_name());
    ASSERT_EQ("pace_ssd_02", cache_config.migration_strategies()[1]->target_storage_name());
    ASSERT_EQ(120000, cache_config.migration_strategies()[1]->methods().mark().timeout_ms());

    // 往返后列表仍然存在
    CacheConfig parsed;
    ASSERT_TRUE(parsed.FromJsonString(cache_config.ToJsonString()));
    ASSERT_EQ(6, parsed.migration_copy_max_concurrency());
    ASSERT_EQ(MigrationMarkClearPolicy::CLEAR_ON_FULL_BLOCK_COVERED, parsed.migration_mark_clear_policy());
    ASSERT_EQ(2u, parsed.migration_strategies().size());
    ASSERT_DOUBLE_EQ(0.70, parsed.migration_strategies()[0]->trigger_threshold());
    ASSERT_EQ(120000, parsed.migration_strategies()[1]->methods().mark().timeout_ms());

    // 缺省（不配 migration_config）应为空列表，且不影响其它字段
    CacheConfig no_migration;
    std::string json2 = R"({
        "cache_prefer_strategy": 5,
        "reclaim_strategy": { "source_storage_name": "pace_mempool_01" },
        "meta_indexer_config": { "meta_storage_backend_config": { "storage_type": "local" } }
    })";
    ASSERT_TRUE(no_migration.FromJsonString(json2));
    ASSERT_EQ(CacheConfig::kDefaultMigrationCopyMaxConcurrency, no_migration.migration_copy_max_concurrency());
    ASSERT_EQ(MigrationMarkClearPolicy::CLEAR_ON_NEXT_WRITE_SUCCESS, no_migration.migration_mark_clear_policy());
    ASSERT_TRUE(no_migration.migration_strategies().empty());
}

TEST_F(MigrationStrategyTest, TestCacheConfigRejectsInvalidMigrationCopyConcurrency) {
    CacheConfig cache_config;
    cache_config.set_cache_prefer_strategy(CachePreferStrategy::CPS_ALWAYS_TAIR_MEMPOOL);
    cache_config.set_reclaim_strategy(std::make_shared<CacheReclaimStrategy>());
    cache_config.reclaim_strategy()->set_storage_unique_name("hot");
    cache_config.set_meta_indexer_config(std::make_shared<MetaIndexerConfig>());
    cache_config.set_migration_copy_max_concurrency(0);

    std::string invalid_fields;
    ASSERT_FALSE(cache_config.ValidateRequiredFields(invalid_fields));
    ASSERT_NE(std::string::npos, invalid_fields.find("copy_max_concurrency"));
}

TEST_F(MigrationStrategyTest, TestCacheConfigRejectsInvalidMigrationMarkClearPolicy) {
    CacheConfig cache_config;
    cache_config.set_cache_prefer_strategy(CachePreferStrategy::CPS_ALWAYS_TAIR_MEMPOOL);
    cache_config.set_reclaim_strategy(std::make_shared<CacheReclaimStrategy>());
    cache_config.reclaim_strategy()->set_storage_unique_name("hot");
    cache_config.set_meta_indexer_config(std::make_shared<MetaIndexerConfig>());
    cache_config.set_migration_mark_clear_policy(static_cast<MigrationMarkClearPolicy>(999));

    std::string invalid_fields;
    ASSERT_FALSE(cache_config.ValidateRequiredFields(invalid_fields));
    ASSERT_NE(std::string::npos, invalid_fields.find("mark_clear_policy"));
}

TEST_F(MigrationStrategyTest, TestMigrationConfigRequiresUniqueRoutes) {
    const auto make_strategy = [](const std::string &source,
                                  const std::string &target,
                                  double threshold,
                                  bool copy,
                                  bool mark) {
        auto strategy = std::make_shared<MigrationStrategy>();
        strategy->set_source_storage_name(source);
        strategy->set_target_storage_name(target);
        strategy->set_trigger_threshold(threshold);
        strategy->mutable_methods().mutable_copy().set_enabled(copy);
        strategy->mutable_methods().mutable_mark().set_enabled(mark);
        if (copy) {
            strategy->set_retention(MigrationRetention::MIGRATION_RETENTION_DELETE_SOURCE);
        }
        return strategy;
    };

    MigrationConfig config;
    config.set_strategies({make_strategy("hot", "cold", 0.9, true, false),
                           make_strategy("hot", "cold", 0.5, false, true)});
    std::string invalid_fields;
    EXPECT_FALSE(config.ValidateRequiredFields(invalid_fields));
    EXPECT_NE(std::string::npos, invalid_fields.find("duplicate_route"));

    // Only an identical source/target pair is forbidden. Fan-out, fan-in and tier chains remain valid.
    config.set_strategies({make_strategy("hot", "warm", 0.5, true, false),
                           make_strategy("hot", "cold", 0.7, true, false),
                           make_strategy("warm", "cold", 0.8, true, false)});
    invalid_fields.clear();
    EXPECT_TRUE(config.ValidateRequiredFields(invalid_fields)) << invalid_fields;
}
