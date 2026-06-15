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
        "storage_unique_name": "pace_mempool_01",
        "target_storage": "pace_ssd_01",
        "trigger_threshold": 0.70,
        "methods": {
            "copy": { "enabled": true, "max_concurrency": 4 },
            "mark": { "enabled": true, "mark_timeout_ms": 60000 }
        },
        "retention": 1
    })";
    ASSERT_TRUE(strategy.FromJsonString(json));
    ASSERT_EQ("pace_mempool_01", strategy.storage_unique_name());
    ASSERT_EQ("pace_ssd_01", strategy.target_storage());
    ASSERT_DOUBLE_EQ(0.70, strategy.trigger_threshold());
    ASSERT_TRUE(strategy.methods().copy().enabled());
    ASSERT_EQ(4, strategy.methods().copy().max_concurrency());
    ASSERT_TRUE(strategy.methods().mark().enabled());
    ASSERT_EQ(60000, strategy.methods().mark().mark_timeout_ms());
    ASSERT_EQ(MigrationRetention::MIGRATION_RETENTION_DELETE_SOURCE, strategy.retention());

    std::string invalid_fields;
    ASSERT_TRUE(strategy.ValidateRequiredFields(invalid_fields)) << invalid_fields;
}

// ToJsonString -> FromJsonString 往返一致
TEST_F(MigrationStrategyTest, TestRoundTrip) {
    MigrationStrategy strategy;
    strategy.set_storage_unique_name("hot");
    strategy.set_target_storage("cold");
    strategy.set_trigger_threshold(0.6);
    MigrationMethods methods;
    methods.mutable_copy().set_enabled(false);
    methods.mutable_copy().set_max_concurrency(8);
    methods.mutable_mark().set_enabled(true);
    methods.mutable_mark().set_mark_timeout_ms(30000);
    strategy.set_methods(methods);
    strategy.set_retention(MigrationRetention::MIGRATION_RETENTION_KEEP_BOTH);

    MigrationStrategy parsed;
    ASSERT_TRUE(parsed.FromJsonString(strategy.ToJsonString()));
    ASSERT_EQ("hot", parsed.storage_unique_name());
    ASSERT_EQ("cold", parsed.target_storage());
    ASSERT_DOUBLE_EQ(0.6, parsed.trigger_threshold());
    ASSERT_FALSE(parsed.methods().copy().enabled());
    ASSERT_EQ(8, parsed.methods().copy().max_concurrency());
    ASSERT_TRUE(parsed.methods().mark().enabled());
    ASSERT_EQ(30000, parsed.methods().mark().mark_timeout_ms());
    ASSERT_EQ(MigrationRetention::MIGRATION_RETENTION_KEEP_BOTH, parsed.retention());
}

// 校验：各类非法配置都应被拒绝
TEST_F(MigrationStrategyTest, TestValidation) {
    auto make_valid = []() {
        MigrationStrategy s;
        s.set_storage_unique_name("hot");
        s.set_target_storage("cold");
        s.set_trigger_threshold(0.7);
        MigrationMethods m;
        m.mutable_copy().set_enabled(true);
        m.mutable_copy().set_max_concurrency(3);
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
        s.set_storage_unique_name("");
        std::string f;
        ASSERT_FALSE(s.ValidateRequiredFields(f));
    }
    { // 目标 storage 为空
        auto s = make_valid();
        s.set_target_storage("");
        std::string f;
        ASSERT_FALSE(s.ValidateRequiredFields(f));
    }
    { // 源与目标相同
        auto s = make_valid();
        s.set_target_storage("hot");
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
    { // 启用 copy 但并发上限非法
        auto s = make_valid();
        s.mutable_methods().mutable_copy().set_max_concurrency(0);
        std::string f;
        ASSERT_FALSE(s.ValidateRequiredFields(f));
    }
    { // 仅 mark：未指定 retention 也合法（mark 不删源端）
        MigrationStrategy s;
        s.set_storage_unique_name("hot");
        s.set_target_storage("cold");
        s.set_trigger_threshold(0.7);
        MigrationMethods m;
        m.mutable_mark().set_enabled(true);
        s.set_methods(m);
        std::string f;
        ASSERT_TRUE(s.ValidateRequiredFields(f)) << f;
    }
}

// CacheConfig 内 migration_strategies 列表的解析与往返
TEST_F(MigrationStrategyTest, TestInCacheConfig) {
    CacheConfig cache_config;
    std::string json = R"({
        "cache_prefer_strategy": 5,
        "reclaim_strategy": { "storage_unique_name": "pace_mempool_01" },
        "meta_indexer_config": { "meta_storage_backend_config": { "storage_type": "local" } },
        "migration_strategies": [
            {
                "storage_unique_name": "pace_mempool_01",
                "target_storage": "pace_ssd_01",
                "trigger_threshold": 0.70,
                "methods": { "copy": { "enabled": true, "max_concurrency": 2 }, "mark": { "enabled": false } },
                "retention": 1
            },
            {
                "storage_unique_name": "pace_mempool_02",
                "target_storage": "pace_ssd_02",
                "trigger_threshold": 0.60,
                "methods": { "copy": { "enabled": true, "max_concurrency": 6 }, "mark": { "enabled": true, "mark_timeout_ms": 5000 } },
                "retention": 2
            }
        ]
    })";
    ASSERT_TRUE(cache_config.FromJsonString(json));
    ASSERT_EQ(2u, cache_config.migration_strategies().size());
    ASSERT_EQ("pace_mempool_01", cache_config.migration_strategies()[0]->storage_unique_name());
    ASSERT_EQ(2, cache_config.migration_strategies()[0]->methods().copy().max_concurrency());
    ASSERT_EQ("pace_ssd_02", cache_config.migration_strategies()[1]->target_storage());
    ASSERT_EQ(6, cache_config.migration_strategies()[1]->methods().copy().max_concurrency());
    ASSERT_EQ(5000, cache_config.migration_strategies()[1]->methods().mark().mark_timeout_ms());

    // 往返后列表仍然存在
    CacheConfig parsed;
    ASSERT_TRUE(parsed.FromJsonString(cache_config.ToJsonString()));
    ASSERT_EQ(2u, parsed.migration_strategies().size());
    ASSERT_DOUBLE_EQ(0.70, parsed.migration_strategies()[0]->trigger_threshold());
    ASSERT_EQ(2, parsed.migration_strategies()[0]->methods().copy().max_concurrency());

    // 缺省（不配 migration_strategies）应为空列表，且不影响其它字段
    CacheConfig no_migration;
    std::string json2 = R"({
        "cache_prefer_strategy": 5,
        "reclaim_strategy": { "storage_unique_name": "pace_mempool_01" },
        "meta_indexer_config": { "meta_storage_backend_config": { "storage_type": "local" } }
    })";
    ASSERT_TRUE(no_migration.FromJsonString(json2));
    ASSERT_TRUE(no_migration.migration_strategies().empty());
}
