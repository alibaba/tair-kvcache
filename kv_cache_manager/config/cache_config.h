#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "kv_cache_manager/common/jsonizable.h"
#include "kv_cache_manager/config/cache_reclaim_strategy.h"
#include "kv_cache_manager/config/data_storage_strategy.h"
#include "kv_cache_manager/config/meta_indexer_config.h"
#include "kv_cache_manager/config/migration_strategy.h"

namespace kv_cache_manager {

namespace proto {
namespace admin {

class CacheConfig;

} // namespace admin
} // namespace proto

enum class CachePreferStrategy {
    CPS_UNSPECIFIED = 0,
    CPS_ALWAYS_3FS = 1,
    CPS_PREFER_3FS = 2,
    CPS_ALWAYS_MOONCAKE = 3,
    CPS_PREFER_MOONCAKE = 4,
    CPS_ALWAYS_TAIR_MEMPOOL = 5,
    CPS_PREFER_TAIR_MEMPOOL = 6,
    CPS_ALWAYS_VCNS_3FS = 7,
    CPS_PREFER_VCNS_3FS = 8,
    CPS_ALWAYS_TAIR_MEMPOOL_SSD = 9,
    CPS_PREFER_TAIR_MEMPOOL_SSD = 10,
};

/*
 * 按照instance_group级别组织配置配置
 */
class CacheConfig : public Jsonizable {
public:
    static constexpr int64_t kDefaultMigrationCopyMaxConcurrency = MigrationConfig::kDefaultCopyMaxConcurrency;

    CacheConfig() = default;
    CacheConfig(CachePreferStrategy cache_prefer_strategy,
                const std::shared_ptr<CacheReclaimStrategy> &reclaim_strategy,
                const std::shared_ptr<MetaIndexerConfig> &meta_indexer_config)
        : cache_prefer_strategy_(cache_prefer_strategy)
        , reclaim_strategy_(reclaim_strategy)
        , meta_indexer_config_(meta_indexer_config) {}

    ~CacheConfig() override;
    bool ValidateRequiredFields(std::string &invalid_fields) const;
    const std::shared_ptr<CacheReclaimStrategy> &reclaim_strategy() const { return reclaim_strategy_; }

    CachePreferStrategy cache_prefer_strategy() const { return cache_prefer_strategy_; }

    const std::shared_ptr<MetaIndexerConfig> &meta_indexer_config() const { return meta_indexer_config_; }

    const std::vector<std::shared_ptr<MigrationStrategy>> &migration_strategies() const {
        return migration_config_.strategies();
    }
    const MigrationConfig &migration_config() const { return migration_config_; }
    int64_t migration_copy_max_concurrency() const { return migration_config_.copy_max_concurrency(); }
    MigrationMarkClearPolicy migration_mark_clear_policy() const { return migration_config_.mark_clear_policy(); }
    MigrationCopyExecutionMode migration_copy_execution_mode() const {
        return migration_config_.copy_execution_mode();
    }
    uint64_t migration_copy_max_inflight_bytes() const { return migration_config_.copy_max_inflight_bytes(); }
    int64_t migration_copy_max_quarantine_operations() const {
        return migration_config_.copy_max_quarantine_operations();
    }
    uint64_t migration_copy_max_quarantine_bytes() const { return migration_config_.copy_max_quarantine_bytes(); }
    int64_t migration_copy_operation_deadline_ms() const {
        return migration_config_.copy_operation_deadline_ms();
    }
    int64_t migration_copy_poll_initial_interval_ms() const {
        return migration_config_.copy_poll_initial_interval_ms();
    }
    int64_t migration_copy_poll_max_interval_ms() const {
        return migration_config_.copy_poll_max_interval_ms();
    }
    int64_t migration_copy_connect_timeout_ms() const { return migration_config_.copy_connect_timeout_ms(); }
    int64_t migration_copy_submit_timeout_ms() const { return migration_config_.copy_submit_timeout_ms(); }
    int64_t migration_copy_query_timeout_ms() const { return migration_config_.copy_query_timeout_ms(); }
    // Setters
    void set_reclaim_strategy(const std::shared_ptr<CacheReclaimStrategy> &reclaim_strategy) {
        reclaim_strategy_ = reclaim_strategy;
    }
    void set_cache_prefer_strategy(CachePreferStrategy cache_prefer_strategy) {
        cache_prefer_strategy_ = cache_prefer_strategy;
    }
    void set_meta_indexer_config(const std::shared_ptr<MetaIndexerConfig> &meta_indexer_config) {
        meta_indexer_config_ = meta_indexer_config;
    }
    void set_migration_strategies(const std::vector<std::shared_ptr<MigrationStrategy>> &migration_strategies) {
        migration_config_.set_strategies(migration_strategies);
    }
    void set_migration_copy_max_concurrency(int64_t migration_copy_max_concurrency) {
        migration_config_.set_copy_max_concurrency(migration_copy_max_concurrency);
    }
    void set_migration_mark_clear_policy(MigrationMarkClearPolicy migration_mark_clear_policy) {
        migration_config_.set_mark_clear_policy(migration_mark_clear_policy);
    }
    void set_migration_copy_execution_mode(MigrationCopyExecutionMode value) {
        migration_config_.set_copy_execution_mode(value);
    }
    void set_migration_copy_max_inflight_bytes(uint64_t value) {
        migration_config_.set_copy_max_inflight_bytes(value);
    }
    void set_migration_copy_max_quarantine_operations(int64_t value) {
        migration_config_.set_copy_max_quarantine_operations(value);
    }
    void set_migration_copy_max_quarantine_bytes(uint64_t value) {
        migration_config_.set_copy_max_quarantine_bytes(value);
    }
    void set_migration_copy_operation_deadline_ms(int64_t value) {
        migration_config_.set_copy_operation_deadline_ms(value);
    }
    void set_migration_copy_poll_initial_interval_ms(int64_t value) {
        migration_config_.set_copy_poll_initial_interval_ms(value);
    }
    void set_migration_copy_poll_max_interval_ms(int64_t value) {
        migration_config_.set_copy_poll_max_interval_ms(value);
    }
    void set_migration_copy_connect_timeout_ms(int64_t value) {
        migration_config_.set_copy_connect_timeout_ms(value);
    }
    void set_migration_copy_submit_timeout_ms(int64_t value) {
        migration_config_.set_copy_submit_timeout_ms(value);
    }
    void set_migration_copy_query_timeout_ms(int64_t value) {
        migration_config_.set_copy_query_timeout_ms(value);
    }
    void set_migration_config(const MigrationConfig &migration_config) {
        migration_config_ = migration_config;
    }

public:
    bool FromRapidValue(const rapidjson::Value &rapid_value) override;

    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override;

    void FromProtoMessage(proto::admin::CacheConfig *message);

private:
    CachePreferStrategy cache_prefer_strategy_;
    std::shared_ptr<CacheReclaimStrategy> reclaim_strategy_;
    std::shared_ptr<MetaIndexerConfig> meta_indexer_config_;
    MigrationConfig migration_config_;
};

using CacheConfigConstPtr = std::shared_ptr<const CacheConfig>;

} // namespace kv_cache_manager
