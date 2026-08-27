#pragma once

#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#include "kv_cache_manager/common/jsonizable.h"
#include "kv_cache_manager/optimizer/quota_runtime/quota_mrc_snapshot.h"

namespace kv_cache_manager {

class QuotaPoolMemberConfig : public Jsonizable {
public:
    bool FromRapidValue(const rapidjson::Value &value) override;
    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override;
    bool Check(std::string &reason) const;

    std::string quota_target_id;
    std::string source_id;
    std::string instance_group;
    std::string quota_scope;
    int64_t current_quota_bytes = 0;
    int64_t min_quota_bytes = 0;
    int64_t configured_max_quota_bytes = 0;
    int64_t hardware_max_quota_bytes = 0;
    std::string configured_max_source;
    std::string hardware_max_source;
};

class QuotaPoolConfig : public Jsonizable {
public:
    bool FromRapidValue(const rapidjson::Value &value) override;
    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override;
    bool Check(std::string &reason) const;

    std::string pool_id;
    std::string quota_scope;
    int64_t allocatable_bytes = 0;
    std::string allocatable_source;
    int64_t max_mrc_freshness_seconds = 300;
    int64_t candidate_step_bytes = int64_t(8) * 1024 * 1024 * 1024;
    std::vector<QuotaPoolMemberConfig> members;
};

struct QuotaPlannerRuntimeConfig {
    bool enable = false;
    // Hard quota writes are intentionally guarded by an independent switch.
    bool enable_hard_resize = false;
    int64_t period_seconds = 300;
    int64_t plan_ttl_seconds = 900;
    int64_t release_timeout_seconds = 1800;
    int64_t release_consecutive_samples = 3;
    std::vector<QuotaPoolConfig> pools;
};

struct QuotaAllocation {
    std::string quota_target_id;
    std::string source_id;
    std::string instance_group;
    int64_t current_quota_bytes = 0;
    int64_t target_quota_bytes = 0;
    int64_t min_quota_bytes = 0;
    int64_t max_quota_bytes = 0;
    uint64_t input_tokens = 0;
    uint64_t hit_tokens = 0;
};

struct PoolQuotaPlan {
    std::string plan_id;
    std::string plan_hash;
    std::string pool_id;
    std::string quota_scope;
    std::string algorithm = "fixed_budget_mrc_dp_v1";
    std::string status = "FROZEN";
    std::string reason;
    uint64_t leader_epoch = 0;
    uint64_t allocation_epoch = 0;
    uint64_t mrc_snapshot_id = 0;
    int64_t created_at_ns = 0;
    int64_t valid_until_ns = 0;
    int64_t pool_allocatable_bytes = 0;
    bool executable = false;
    bool writes_quota = false;
    std::string execution_phase = "SHADOW";
    uint64_t execution_revision = 1;
    int64_t release_deadline_ns = 0;
    int64_t release_consecutive_samples = 3;
    std::vector<QuotaAllocation> allocations;
    std::set<std::string> reconciled_targets;
    std::set<std::string> release_required_targets;
    std::set<std::string> release_confirmed_targets;
    std::set<std::string> grow_applied_targets;
    std::map<std::string, int64_t> observed_quota_bytes;
    std::map<std::string, int64_t> observed_used_bytes;
};

struct QuotaResizeResult {
    std::string plan_id;
    std::string plan_hash;
    std::string pool_id;
    std::string quota_target_id;
    uint64_t leader_epoch = 0;
    uint64_t allocation_epoch = 0;
    uint64_t execution_revision = 0;
    std::string status;
    std::string reason;
    int64_t observed_quota_bytes = -1;
    int64_t observed_used_bytes = -1;
};

class InMemoryQuotaPlanStore {
public:
    using ObservedQuotaMap = std::map<std::string, std::map<std::string, int64_t>>;
    // Returns false while a previous hard-resize plan for the pool is active.
    bool Publish(std::shared_ptr<const PoolQuotaPlan> plan);
    std::shared_ptr<const PoolQuotaPlan> Get(const std::string &pool_id) const;
    bool RecordResizeResult(const QuotaResizeResult &result, std::string *reason = nullptr);
    ObservedQuotaMap GetObservedQuotas() const;

private:
    mutable std::mutex mutex_;
    std::map<std::string, std::shared_ptr<const PoolQuotaPlan>> plans_;
    ObservedQuotaMap observed_quotas_;
};

class ShadowQuotaPlanner {
public:
    explicit ShadowQuotaPlanner(QuotaPlannerRuntimeConfig config);

    std::vector<std::shared_ptr<const PoolQuotaPlan>>
    BuildPlans(const OnlineMrcDecisionSnapshot &snapshot,
               const InMemoryQuotaPlanStore::ObservedQuotaMap &observed_quotas = {});
    uint64_t leader_epoch() const { return leader_epoch_; }

private:
    std::shared_ptr<const PoolQuotaPlan> BuildPoolPlan(const QuotaPoolConfig &pool,
                                                       const OnlineMrcDecisionSnapshot &snapshot,
                                                       const std::map<std::string, int64_t> &observed_quotas);
    static std::string HashPlan(const PoolQuotaPlan &plan);

    QuotaPlannerRuntimeConfig config_;
    uint64_t leader_epoch_ = 0;
    std::atomic<uint64_t> allocation_epoch_{0};
};

} // namespace kv_cache_manager
