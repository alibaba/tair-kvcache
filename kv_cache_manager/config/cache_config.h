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
};

/*
 * 按照instance_group级别组织配置配置
 */
class CacheConfig : public Jsonizable {
public:
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

public:
    bool FromRapidValue(const rapidjson::Value &rapid_value) override;

    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override;

    void FromProtoMessage(proto::admin::CacheConfig *message);

private:
    CachePreferStrategy cache_prefer_strategy_;
    std::shared_ptr<CacheReclaimStrategy> reclaim_strategy_;
    std::shared_ptr<MetaIndexerConfig> meta_indexer_config_;
};

using CacheConfigConstPtr = std::shared_ptr<const CacheConfig>;

} // namespace kv_cache_manager
