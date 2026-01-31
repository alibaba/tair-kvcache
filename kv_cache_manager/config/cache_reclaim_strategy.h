#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "kv_cache_manager/common/jsonizable.h"
#include "kv_cache_manager/config/trigger_strategy.h"

namespace kv_cache_manager {

enum class ReclaimPolicy {
    POLICY_UNSPECIFIED = 0,
    POLICY_LRU = 1,
    POLICY_LFU = 2,
    POLICY_TTL = 3,
};

/**
 * Cache整理策略相关的配置，包括淘汰策略
 */
class CacheReclaimStrategy : public Jsonizable {
public:
    CacheReclaimStrategy() = default;
    CacheReclaimStrategy(const std::string &storage_unique_name,
                         ReclaimPolicy reclaim_policy,
                         const TriggerStrategy &trigger_strategy,
                         int32_t trigger_period_seconds,
                         int32_t reclaim_step_size,
                         int32_t reclaim_step_percentage,
                         int32_t delay_before_delete_ms = 0)
        : storage_unique_name_(storage_unique_name)
        , reclaim_policy_(reclaim_policy)
        , trigger_strategy_(trigger_strategy)
        , trigger_period_seconds_(trigger_period_seconds)
        , reclaim_step_size_(reclaim_step_size)
        , reclaim_step_percentage_(reclaim_step_percentage)
        , delay_before_delete_ms_(delay_before_delete_ms) {}

    ~CacheReclaimStrategy() override;

    const std::string GetAlgorithm() const { return "LRU"; }

    void GetTTL() {}
    // Getters
    const std::string &storage_unique_name() const { return storage_unique_name_; }
    ReclaimPolicy reclaim_policy() const { return reclaim_policy_; }
    const TriggerStrategy &trigger_strategy() const { return trigger_strategy_; }
    int32_t trigger_period_seconds() const { return trigger_period_seconds_; }
    int32_t reclaim_step_size() const { return reclaim_step_size_; }
    int32_t reclaim_step_percentage() const { return reclaim_step_percentage_; }
    int32_t delay_before_delete_ms() const { return delay_before_delete_ms_; }

    // Setters
    void set_storage_unique_name(const std::string &storage_unique_name) { storage_unique_name_ = storage_unique_name; }
    void set_reclaim_policy(ReclaimPolicy reclaim_policy) { reclaim_policy_ = reclaim_policy; }
    void set_trigger_strategy(const TriggerStrategy &trigger_strategy) { trigger_strategy_ = trigger_strategy; }
    void set_trigger_period_seconds(int32_t trigger_period_seconds) {
        trigger_period_seconds_ = trigger_period_seconds;
    }
    void set_reclaim_step_size(int32_t reclaim_step_size) { reclaim_step_size_ = reclaim_step_size; }
    void set_reclaim_step_percentage(int32_t reclaim_step_percentage) {
        reclaim_step_percentage_ = reclaim_step_percentage;
    }
    void set_delay_before_delete_ms(int32_t delay_before_delete_ms) {
        delay_before_delete_ms_ = delay_before_delete_ms;
    }

public:
    bool FromRapidValue(const rapidjson::Value &rapid_value) override;

    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override;
    bool ValidateRequiredFields(std::string &invalid_fields) const;

private:
    std::string storage_unique_name_;
    ReclaimPolicy reclaim_policy_;
    TriggerStrategy trigger_strategy_;
    int32_t trigger_period_seconds_;
    int32_t reclaim_step_size_;
    int32_t reclaim_step_percentage_;
    int32_t delay_before_delete_ms_;
};

} // namespace kv_cache_manager