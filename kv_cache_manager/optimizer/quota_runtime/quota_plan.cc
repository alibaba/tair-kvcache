#include "kv_cache_manager/optimizer/quota_runtime/quota_plan.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <set>
#include <sstream>
#include <unordered_map>

namespace kv_cache_manager {
namespace {

constexpr int64_t kNanosPerSecond = 1000000000LL;

int64_t EffectiveMax(const QuotaPoolMemberConfig &member) {
    return std::min(member.configured_max_quota_bytes, member.hardware_max_quota_bytes);
}

const OnlineMrcSourceSnapshot *FindSource(const OnlineMrcDecisionSnapshot &snapshot, const std::string &source_id) {
    for (const auto &source : snapshot.sources) {
        if (source.source_id == source_id) {
            return &source;
        }
    }
    return nullptr;
}

void Freeze(PoolQuotaPlan *plan, std::string reason) {
    plan->status = "FROZEN";
    plan->reason = std::move(reason);
    plan->execution_phase = "FROZEN";
    plan->executable = false;
    ++plan->execution_revision;
}

} // namespace

bool QuotaPoolMemberConfig::FromRapidValue(const rapidjson::Value &value) {
    KVCM_JSON_GET_MACRO(value, "quota_target_id", quota_target_id);
    KVCM_JSON_GET_MACRO(value, "source_id", source_id);
    KVCM_JSON_GET_MACRO(value, "instance_group", instance_group);
    KVCM_JSON_GET_MACRO(value, "quota_scope", quota_scope);
    KVCM_JSON_GET_MACRO(value, "current_quota_bytes", current_quota_bytes);
    KVCM_JSON_GET_MACRO(value, "min_quota_bytes", min_quota_bytes);
    KVCM_JSON_GET_MACRO(value, "configured_max_quota_bytes", configured_max_quota_bytes);
    KVCM_JSON_GET_MACRO(value, "hardware_max_quota_bytes", hardware_max_quota_bytes);
    KVCM_JSON_GET_MACRO(value, "configured_max_source", configured_max_source);
    KVCM_JSON_GET_MACRO(value, "hardware_max_source", hardware_max_source);
    return true;
}

void QuotaPoolMemberConfig::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept {
    Put(writer, "quota_target_id", quota_target_id);
    Put(writer, "source_id", source_id);
    Put(writer, "instance_group", instance_group);
    Put(writer, "quota_scope", quota_scope);
    Put(writer, "current_quota_bytes", current_quota_bytes);
    Put(writer, "min_quota_bytes", min_quota_bytes);
    Put(writer, "configured_max_quota_bytes", configured_max_quota_bytes);
    Put(writer, "hardware_max_quota_bytes", hardware_max_quota_bytes);
    Put(writer, "configured_max_source", configured_max_source);
    Put(writer, "hardware_max_source", hardware_max_source);
}

bool QuotaPoolMemberConfig::Check(std::string &reason) const {
    if (quota_target_id.empty() || source_id.empty() || instance_group.empty() || quota_scope.empty()) {
        reason = "member identity and quota_scope are required";
        return false;
    }
    if (min_quota_bytes <= 0 || current_quota_bytes <= 0 || configured_max_quota_bytes <= 0 ||
        hardware_max_quota_bytes <= 0 || configured_max_source.empty() || hardware_max_source.empty()) {
        reason = "member quota bounds require positive authoritative config and hardware sources";
        return false;
    }
    const int64_t maximum = EffectiveMax(*this);
    if (min_quota_bytes > maximum || current_quota_bytes < min_quota_bytes || current_quota_bytes > maximum) {
        reason = "member current/min/max quota bounds are inconsistent";
        return false;
    }
    return true;
}

bool QuotaPoolConfig::FromRapidValue(const rapidjson::Value &value) {
    KVCM_JSON_GET_MACRO(value, "pool_id", pool_id);
    KVCM_JSON_GET_MACRO(value, "quota_scope", quota_scope);
    KVCM_JSON_GET_MACRO(value, "allocatable_bytes", allocatable_bytes);
    KVCM_JSON_GET_MACRO(value, "allocatable_source", allocatable_source);
    KVCM_JSON_GET_DEFAULT_MACRO(value, "max_mrc_freshness_seconds", max_mrc_freshness_seconds, int64_t(300));
    KVCM_JSON_GET_DEFAULT_MACRO(value, "candidate_step_bytes", candidate_step_bytes, int64_t(8) * 1024 * 1024 * 1024);
    KVCM_JSON_GET_MACRO(value, "members", members);
    return true;
}

void QuotaPoolConfig::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept {
    Put(writer, "pool_id", pool_id);
    Put(writer, "quota_scope", quota_scope);
    Put(writer, "allocatable_bytes", allocatable_bytes);
    Put(writer, "allocatable_source", allocatable_source);
    Put(writer, "max_mrc_freshness_seconds", max_mrc_freshness_seconds);
    Put(writer, "candidate_step_bytes", candidate_step_bytes);
    Put(writer, "members", members);
}

bool QuotaPoolConfig::Check(std::string &reason) const {
    if (pool_id.empty() || quota_scope.empty() || allocatable_bytes <= 0 || allocatable_source.empty() ||
        members.empty() || max_mrc_freshness_seconds <= 0 || candidate_step_bytes <= 0) {
        reason = "pool identity, authoritative capacity, members, and freshness are required";
        return false;
    }
    std::set<std::string> targets;
    std::set<std::string> sources;
    int64_t floor_sum = 0;
    int64_t current_sum = 0;
    for (const auto &member : members) {
        if (!member.Check(reason)) {
            return false;
        }
        if (member.quota_scope != quota_scope) {
            reason = "member quota_scope does not match pool quota_scope";
            return false;
        }
        if (!targets.emplace(member.quota_target_id).second || !sources.emplace(member.source_id).second) {
            reason = "pool member target and source ids must be unique";
            return false;
        }
        if (floor_sum > std::numeric_limits<int64_t>::max() - member.min_quota_bytes) {
            reason = "pool quota floor sum overflow";
            return false;
        }
        floor_sum += member.min_quota_bytes;
        if (current_sum > std::numeric_limits<int64_t>::max() - member.current_quota_bytes) {
            reason = "pool current quota sum overflow";
            return false;
        }
        current_sum += member.current_quota_bytes;
    }
    if (floor_sum > allocatable_bytes) {
        reason = "pool allocatable capacity is below member quota floors";
        return false;
    }
    if (current_sum > allocatable_bytes) {
        reason = "pool current quota sum exceeds authoritative allocatable capacity";
        return false;
    }
    return true;
}

bool InMemoryQuotaPlanStore::Publish(std::shared_ptr<const PoolQuotaPlan> plan) {
    if (!plan) {
        return false;
    }
    std::lock_guard<std::mutex> guard(mutex_);
    const auto current = plans_.find(plan->pool_id);
    if (current != plans_.end() && current->second->writes_quota) {
        if (current->second->execution_phase == "FROZEN" ||
            (current->second->executable && current->second->execution_phase != "COMPLETE")) {
            return false;
        }
    }
    plans_[plan->pool_id] = std::move(plan);
    return true;
}

std::shared_ptr<const PoolQuotaPlan> InMemoryQuotaPlanStore::Get(const std::string &pool_id) const {
    std::lock_guard<std::mutex> guard(mutex_);
    const auto it = plans_.find(pool_id);
    return it == plans_.end() ? nullptr : it->second;
}

bool InMemoryQuotaPlanStore::RecordResizeResult(const QuotaResizeResult &result, std::string *reason) {
    std::lock_guard<std::mutex> guard(mutex_);
    const auto current = plans_.find(result.pool_id);
    if (current == plans_.end()) {
        if (reason)
            *reason = "plan_not_found";
        return false;
    }
    const auto &existing = *current->second;
    if (existing.plan_id != result.plan_id || existing.plan_hash != result.plan_hash ||
        existing.leader_epoch != result.leader_epoch || existing.allocation_epoch != result.allocation_epoch ||
        existing.execution_revision != result.execution_revision) {
        if (reason)
            *reason = "plan_or_execution_revision_mismatch";
        return false;
    }
    const auto allocation_it =
        std::find_if(existing.allocations.begin(), existing.allocations.end(), [&](const QuotaAllocation &allocation) {
            return allocation.quota_target_id == result.quota_target_id;
        });
    if (allocation_it == existing.allocations.end()) {
        if (reason)
            *reason = "quota_target_not_in_plan";
        return false;
    }
    if (result.observed_quota_bytes > 0) {
        observed_quotas_[result.pool_id][result.quota_target_id] = result.observed_quota_bytes;
    }
    auto updated = std::make_shared<PoolQuotaPlan>(existing);
    if (result.status == "HOLD_ACKNOWLEDGED" && existing.execution_phase == "RECONCILE") {
        if (result.observed_quota_bytes <= 0 || result.observed_used_bytes < 0) {
            Freeze(updated.get(), "reconcile_observation_unavailable:" + result.quota_target_id);
            current->second = std::move(updated);
            return true;
        }
        updated->reconciled_targets.insert(result.quota_target_id);
        updated->observed_quota_bytes[result.quota_target_id] = result.observed_quota_bytes;
        updated->observed_used_bytes[result.quota_target_id] = result.observed_used_bytes;
        if (updated->reconciled_targets.size() == updated->allocations.size()) {
            int64_t quota_sum = 0;
            bool has_receiver = false;
            for (auto &allocation : updated->allocations) {
                const int64_t observed_quota = updated->observed_quota_bytes[allocation.quota_target_id];
                const int64_t observed_used = updated->observed_used_bytes[allocation.quota_target_id];
                if (observed_quota < allocation.min_quota_bytes || observed_quota > allocation.max_quota_bytes ||
                    quota_sum > updated->pool_allocatable_bytes - observed_quota) {
                    Freeze(updated.get(), "reconciled_quota_out_of_pool_bounds:" + allocation.quota_target_id);
                    current->second = std::move(updated);
                    return true;
                }
                allocation.current_quota_bytes = observed_quota;
                quota_sum += observed_quota;
                if (allocation.target_quota_bytes < observed_quota || observed_used > allocation.target_quota_bytes) {
                    updated->release_required_targets.insert(allocation.quota_target_id);
                }
                has_receiver = has_receiver || allocation.target_quota_bytes > observed_quota;
            }
            if (!updated->release_required_targets.empty()) {
                updated->execution_phase = "DONOR_SHRINK";
            } else if (has_receiver) {
                updated->execution_phase = "RECEIVER_GROW";
            } else {
                updated->execution_phase = "COMPLETE";
                updated->status = "APPLIED";
                updated->reason = "quota_unchanged_after_reconcile";
                updated->executable = false;
            }
            ++updated->execution_revision;
        }
        current->second = std::move(updated);
        return true;
    }
    const bool terminal_failure = result.status.find("FAILED") != std::string::npos ||
                                  result.status.find("TIMEOUT") != std::string::npos ||
                                  result.status.find("REJECTED") != std::string::npos;
    if (terminal_failure) {
        Freeze(updated.get(), result.status + ":" + result.reason);
        current->second = std::move(updated);
        return true;
    }
    if (result.status == "DONOR_RELEASE_CONFIRMED") {
        if (existing.execution_phase != "DONOR_SHRINK" ||
            existing.release_required_targets.count(result.quota_target_id) == 0) {
            if (reason)
                *reason = "unexpected_donor_release_confirmation";
            return false;
        }
        updated->release_confirmed_targets.insert(result.quota_target_id);
        bool all_donors_confirmed = true;
        bool has_receiver = false;
        for (const auto &allocation : updated->allocations) {
            if (updated->release_required_targets.count(allocation.quota_target_id) != 0 &&
                updated->release_confirmed_targets.count(allocation.quota_target_id) == 0) {
                all_donors_confirmed = false;
            }
            has_receiver = has_receiver || allocation.target_quota_bytes > allocation.current_quota_bytes;
        }
        if (all_donors_confirmed) {
            updated->execution_phase = has_receiver ? "RECEIVER_GROW" : "COMPLETE";
            updated->status = has_receiver ? "EXECUTING" : "APPLIED";
            updated->executable = has_receiver;
            ++updated->execution_revision;
        }
    } else if (result.status == "RECEIVER_GROW_APPLIED") {
        if (existing.execution_phase != "RECEIVER_GROW" ||
            allocation_it->target_quota_bytes <= allocation_it->current_quota_bytes) {
            if (reason)
                *reason = "unexpected_receiver_grow_confirmation";
            return false;
        }
        updated->grow_applied_targets.insert(result.quota_target_id);
        bool all_receivers_applied = true;
        for (const auto &allocation : updated->allocations) {
            if (allocation.target_quota_bytes > allocation.current_quota_bytes &&
                updated->grow_applied_targets.count(allocation.quota_target_id) == 0) {
                all_receivers_applied = false;
            }
        }
        if (all_receivers_applied) {
            updated->execution_phase = "COMPLETE";
            updated->status = "APPLIED";
            updated->reason = "two_phase_hard_resize_complete";
            updated->executable = false;
            ++updated->execution_revision;
        }
    }
    current->second = std::move(updated);
    return true;
}

InMemoryQuotaPlanStore::ObservedQuotaMap InMemoryQuotaPlanStore::GetObservedQuotas() const {
    std::lock_guard<std::mutex> guard(mutex_);
    return observed_quotas_;
}

ShadowQuotaPlanner::ShadowQuotaPlanner(QuotaPlannerRuntimeConfig config) : config_(std::move(config)) {
    leader_epoch_ = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch())
            .count());
}

std::vector<std::shared_ptr<const PoolQuotaPlan>>
ShadowQuotaPlanner::BuildPlans(const OnlineMrcDecisionSnapshot &snapshot,
                               const InMemoryQuotaPlanStore::ObservedQuotaMap &observed_quotas) {
    std::vector<std::shared_ptr<const PoolQuotaPlan>> plans;
    plans.reserve(config_.pools.size());
    for (const auto &pool : config_.pools) {
        const auto it = observed_quotas.find(pool.pool_id);
        plans.push_back(
            BuildPoolPlan(pool, snapshot, it == observed_quotas.end() ? std::map<std::string, int64_t>() : it->second));
    }
    return plans;
}

std::shared_ptr<const PoolQuotaPlan>
ShadowQuotaPlanner::BuildPoolPlan(const QuotaPoolConfig &pool,
                                  const OnlineMrcDecisionSnapshot &snapshot,
                                  const std::map<std::string, int64_t> &observed_quotas) {
    auto plan = std::make_shared<PoolQuotaPlan>();
    plan->pool_id = pool.pool_id;
    plan->quota_scope = pool.quota_scope;
    plan->leader_epoch = leader_epoch_;
    plan->allocation_epoch = allocation_epoch_.fetch_add(1, std::memory_order_relaxed) + 1;
    plan->mrc_snapshot_id = snapshot.snapshot_id;
    plan->created_at_ns = snapshot.created_at_ns;
    plan->valid_until_ns = snapshot.created_at_ns + config_.plan_ttl_seconds * kNanosPerSecond;
    plan->pool_allocatable_bytes = pool.allocatable_bytes;
    plan->release_consecutive_samples = config_.release_consecutive_samples;
    plan->release_deadline_ns = snapshot.created_at_ns + config_.release_timeout_seconds * kNanosPerSecond;
    plan->plan_id = std::to_string(plan->leader_epoch) + "-" + std::to_string(plan->allocation_epoch) + "-" +
                    std::to_string(snapshot.snapshot_id);

    std::string config_reason;
    if (!pool.Check(config_reason)) {
        plan->reason = "invalid_pool_config:" + config_reason;
        plan->plan_hash = HashPlan(*plan);
        return plan;
    }
    std::vector<int64_t> current_quotas;
    current_quotas.reserve(pool.members.size());
    int64_t current_sum = 0;
    for (const auto &member : pool.members) {
        const auto observed = observed_quotas.find(member.quota_target_id);
        const int64_t current = observed == observed_quotas.end() ? member.current_quota_bytes : observed->second;
        if (current < member.min_quota_bytes || current > EffectiveMax(member) ||
            current_sum > pool.allocatable_bytes - current) {
            plan->reason = "observed_current_quota_out_of_pool_bounds:" + member.quota_target_id;
            plan->plan_hash = HashPlan(*plan);
            return plan;
        }
        current_quotas.push_back(current);
        current_sum += current;
    }

    struct Candidate {
        int64_t quota_bytes = 0;
        uint64_t input_tokens = 0;
        uint64_t hit_tokens = 0;
    };
    std::vector<std::vector<Candidate>> candidates;
    candidates.reserve(pool.members.size());
    for (const auto &member : pool.members) {
        const auto *source = FindSource(snapshot, member.source_id);
        if (!source) {
            plan->reason = "missing_mrc_source:" + member.source_id;
            plan->plan_hash = HashPlan(*plan);
            return plan;
        }
        if (source->accepted_facts == 0) {
            plan->reason = "empty_mrc_source:" + member.source_id;
            plan->plan_hash = HashPlan(*plan);
            return plan;
        }
        const int64_t freshness_ns = snapshot.created_at_ns - source->newest_event_time_ns;
        if (source->newest_event_time_ns <= 0 || freshness_ns < 0 ||
            freshness_ns > pool.max_mrc_freshness_seconds * kNanosPerSecond) {
            plan->reason = "stale_mrc_source:" + member.source_id;
            plan->plan_hash = HashPlan(*plan);
            return plan;
        }
        const int64_t maximum = EffectiveMax(member);
        std::vector<Candidate> member_candidates;
        for (const auto &point : source->curve) {
            if (point.capacity_bytes >= static_cast<uint64_t>(member.min_quota_bytes) &&
                point.capacity_bytes <= static_cast<uint64_t>(maximum) && point.input_tokens > 0) {
                member_candidates.push_back(
                    Candidate{static_cast<int64_t>(point.capacity_bytes), point.input_tokens, point.hit_tokens});
            }
        }
        if (member_candidates.empty()) {
            plan->reason = "no_feasible_mrc_point:" + member.source_id;
            plan->plan_hash = HashPlan(*plan);
            return plan;
        }
        candidates.push_back(std::move(member_candidates));
    }

    struct State {
        uint64_t hit_tokens = 0;
        uint64_t movement_bytes = 0;
        std::vector<size_t> selections;
    };
    std::map<int64_t, State> states{{0, State{}}};
    for (size_t member_index = 0; member_index < candidates.size(); ++member_index) {
        std::map<int64_t, State> next;
        for (const auto &[used, state] : states) {
            for (size_t candidate_index = 0; candidate_index < candidates[member_index].size(); ++candidate_index) {
                const auto &candidate = candidates[member_index][candidate_index];
                if (candidate.quota_bytes > pool.allocatable_bytes - used) {
                    continue;
                }
                const int64_t new_used = used + candidate.quota_bytes;
                State proposed = state;
                proposed.hit_tokens += candidate.hit_tokens;
                proposed.movement_bytes +=
                    static_cast<uint64_t>(std::llabs(candidate.quota_bytes - current_quotas[member_index]));
                proposed.selections.push_back(candidate_index);
                const auto existing = next.find(new_used);
                if (existing == next.end() || proposed.hit_tokens > existing->second.hit_tokens ||
                    (proposed.hit_tokens == existing->second.hit_tokens &&
                     proposed.movement_bytes < existing->second.movement_bytes)) {
                    next[new_used] = std::move(proposed);
                }
            }
        }
        states = std::move(next);
    }
    if (states.empty()) {
        plan->reason = "pool_budget_has_no_feasible_allocation";
        plan->plan_hash = HashPlan(*plan);
        return plan;
    }
    const State *best = nullptr;
    for (const auto &[_, state] : states) {
        if (!best || state.hit_tokens > best->hit_tokens ||
            (state.hit_tokens == best->hit_tokens && state.movement_bytes < best->movement_bytes)) {
            best = &state;
        }
    }
    for (size_t i = 0; i < pool.members.size(); ++i) {
        const auto &member = pool.members[i];
        const auto &candidate = candidates[i][best->selections[i]];
        plan->allocations.push_back(QuotaAllocation{member.quota_target_id,
                                                    member.source_id,
                                                    member.instance_group,
                                                    current_quotas[i],
                                                    candidate.quota_bytes,
                                                    member.min_quota_bytes,
                                                    EffectiveMax(member),
                                                    candidate.input_tokens,
                                                    candidate.hit_tokens});
    }
    if (config_.enable_hard_resize) {
        plan->status = "EXECUTING";
        plan->reason = "reconcile_before_two_phase_hard_resize";
        plan->executable = true;
        plan->writes_quota = true;
        plan->execution_phase = "RECONCILE";
    } else {
        plan->status = "SHADOW_READY";
        plan->reason = "writes_quota=false";
        plan->execution_phase = "SHADOW";
    }
    plan->plan_hash = HashPlan(*plan);
    return plan;
}

std::string ShadowQuotaPlanner::HashPlan(const PoolQuotaPlan &plan) {
    std::ostringstream input;
    input << plan.pool_id << '|' << plan.leader_epoch << '|' << plan.allocation_epoch << '|' << plan.mrc_snapshot_id
          << '|' << plan.status;
    for (const auto &allocation : plan.allocations) {
        input << '|' << allocation.quota_target_id << ':' << allocation.target_quota_bytes;
    }
    uint64_t hash = 1469598103934665603ULL;
    for (const unsigned char byte : input.str()) {
        hash ^= byte;
        hash *= 1099511628211ULL;
    }
    char buffer[17];
    std::snprintf(buffer, sizeof(buffer), "%016llx", static_cast<unsigned long long>(hash));
    return buffer;
}

} // namespace kv_cache_manager
