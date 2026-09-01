#pragma once

#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <utility>
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
    // All benefit thresholds are disabled by default for upgrade compatibility.
    // Hard-resize deployments should set them explicitly after Shadow validation.
    double min_expected_hit_rate_gain_pp = 0.0;
    double min_gain_pp_per_tib_moved = 0.0;
    double movement_penalty_pp_per_tib = 0.0;
    int64_t min_quota_transfer_bytes = 0;
    int64_t stability_required_plans = 1;
    int64_t stability_tolerance_bytes = 0;
    double capacity_saving_sla_ratio = 0.99;
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
    uint64_t baseline_hit_tokens = 0;
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
    double baseline_hit_rate_pp = 0.0;
    double target_hit_rate_pp = 0.0;
    double expected_hit_rate_gain_pp = 0.0;
    bool enforced_shadow_baseline_valid = false;
    uint64_t enforced_shadow_baseline_input_tokens = 0;
    uint64_t enforced_shadow_baseline_hit_tokens = 0;
    double enforced_shadow_baseline_hit_rate_pp = 0.0;
    double movement_penalty_pp = 0.0;
    double expected_net_gain_pp = 0.0;
    double gain_pp_per_tib_moved = 0.0;
    uint64_t quota_change_bytes = 0;
    uint64_t quota_transfer_bytes = 0;
    int64_t stability_confirmed_plans = 0;
    int64_t stability_required_plans = 0;
    double capacity_saving_sla_ratio = 0.0;
    int64_t sla_required_capacity_bytes = 0;
    int64_t sla_capacity_saving_bytes = 0;
    int64_t sla_capacity_deficit_bytes = 0;
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

// Returns the theoretical token hit-rate improvement of the selected allocation
// over the quota observed at decision time, expressed in percentage points.
double ExpectedHitRateGainPercentagePoints(const PoolQuotaPlan &plan);

struct QuotaRealizedHitRateGain {
    uint64_t snapshot_id = 0;
    uint64_t input_tokens = 0;
    double before_hit_rate_pp = 0.0;
    double after_hit_rate_pp = 0.0;
    double gain_pp = 0.0;
};

// Replays the applied plan's before/after quotas on the same post-resize MRC
// snapshot, so traffic-window drift is not counted as realized quota benefit.
bool ComputeRealizedHitRateGain(const PoolQuotaPlan &applied_plan,
                                const OnlineMrcDecisionSnapshot &snapshot,
                                QuotaRealizedHitRateGain *result,
                                std::string *reason = nullptr);

struct QuotaEnforcedShadowHitRateGain {
    uint64_t snapshot_id = 0;
    uint64_t before_input_tokens = 0;
    uint64_t after_input_tokens = 0;
    double before_hit_rate_pp = 0.0;
    double after_hit_rate_pp = 0.0;
    double gain_pp = 0.0;
};

// Compares two actual online shadow windows. The baseline was measured at the
// quotas observed when the plan was built; the post window is accepted only
// when every source was measured at the plan's applied target quota.
bool ComputeEnforcedShadowHitRateGain(const PoolQuotaPlan &applied_plan,
                                      const OnlineMrcDecisionSnapshot &snapshot,
                                      QuotaEnforcedShadowHitRateGain *result,
                                      std::string *reason = nullptr);

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
    struct StabilityState {
        std::vector<std::pair<std::string, int64_t>> targets;
        int64_t consecutive_plans = 0;
    };

    std::shared_ptr<const PoolQuotaPlan> BuildPoolPlan(const QuotaPoolConfig &pool,
                                                       const OnlineMrcDecisionSnapshot &snapshot,
                                                       const std::map<std::string, int64_t> &observed_quotas);
    static std::string HashPlan(const PoolQuotaPlan &plan);

    QuotaPlannerRuntimeConfig config_;
    uint64_t leader_epoch_ = 0;
    std::atomic<uint64_t> allocation_epoch_{0};
    std::map<std::string, StabilityState> stability_states_;
};

} // namespace kv_cache_manager
