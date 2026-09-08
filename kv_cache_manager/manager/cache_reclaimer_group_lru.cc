#include <algorithm>
#include <chrono>
#include <limits>
#include <unordered_set>

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/common/string_util.h"
#include "kv_cache_manager/common/timestamp_util.h"
#include "kv_cache_manager/config/cache_config.h"
#include "kv_cache_manager/config/cache_reclaim_strategy.h"
#include "kv_cache_manager/config/instance_group.h"
#include "kv_cache_manager/config/instance_info.h"
#include "kv_cache_manager/manager/cache_reclaimer.h"
#include "kv_cache_manager/meta/common.h"
#include "kv_cache_manager/meta/meta_indexer.h"
#include "kv_cache_manager/meta/meta_indexer_manager.h"

namespace kv_cache_manager {

std::future<CacheReclaimer::SamplingResult> CacheReclaimer::SubmitSamplingTask(std::function<SamplingResult()> sample) {
    auto promise = std::make_shared<std::promise<SamplingResult>>();
    auto future = promise->get_future();
    in_flight_sampling_tasks_.fetch_add(1);
    SubmitTask([this, sample = std::move(sample), promise]() {
        SamplingResult result;
        try {
            result = sample();
        } catch (...) {
            result = SamplingResult{};
            KVCM_LOG_WARN("key sampling task threw an exception");
        }
        in_flight_sampling_tasks_.fetch_sub(1);
        promise->set_value(std::move(result));
    });
    return future;
}

bool CacheReclaimer::SameReclaimScope(const WaterLevelExceed &initial, const WaterLevelExceed &current) noexcept {
    if (initial.CheckStorageTypeWaterLevelExceed()) {
        if (!current.CheckStorageTypeWaterLevelExceed()) {
            return false;
        }
        for (std::size_t i = 1; i < static_cast<std::size_t>(DataStorageType::COUNT); ++i) {
            const auto type = static_cast<DataStorageType>(i);
            if (initial.GetWaterLevelExceedByType(type) != current.GetWaterLevelExceedByType(type)) {
                return false;
            }
        }
        return true;
    }
    if (current.CheckStorageTypeWaterLevelExceed()) {
        return false;
    }
    if (initial.GetGroupBytesWaterLevelExceed()) {
        return current.GetGroupBytesWaterLevelExceed();
    }
    return initial.GetGroupKeysWaterLevelExceed() && current.GetGroupKeysWaterLevelExceed() &&
           !current.GetGroupBytesWaterLevelExceed();
}

bool CacheReclaimer::BuildGroupLruPlan(const RequestContext *request_context,
                                       const std::string &instance_group,
                                       const WaterLevelExceed &scope,
                                       const std::vector<std::shared_ptr<const InstanceInfo>> &instance_infos,
                                       std::size_t configured_sampling_size,
                                       std::size_t configured_batch_size,
                                       GroupLruPlan &out_plan) noexcept {
    out_plan = GroupLruPlan{};
    if (configured_sampling_size == 0 || configured_batch_size == 0 || group_lru_config_.max_sampling_size == 0 ||
        group_lru_config_.max_delete_requests_per_round == 0 || !scope.CheckGroupWaterLevelExceed()) {
        return false;
    }
    out_plan.configured_batch_size = configured_batch_size;
    out_plan.normalized_sampling_size = std::max(configured_sampling_size, configured_batch_size);

    struct EligibleInstance {
        std::shared_ptr<const InstanceInfo> info;
        std::uint64_t key_count;
    };
    std::map<std::string, EligibleInstance> eligible;
    for (const auto &info : instance_infos) {
        if (!info) {
            continue;
        }
        const auto indexer = meta_indexer_manager_->GetMetaIndexer(info->instance_id());
        if (!indexer) {
            continue;
        }
        const auto key_count = static_cast<std::uint64_t>(indexer->GetKeyCount());
        std::uint64_t usage = 0;
        if (!scope.CheckStorageTypeWaterLevelExceed() && !scope.GetGroupBytesWaterLevelExceed()) {
            usage = key_count;
        } else {
            for (std::size_t i = 1; i < static_cast<std::size_t>(DataStorageType::COUNT); ++i) {
                const auto type = static_cast<DataStorageType>(i);
                if (type == DataStorageType::DATA_STORAGE_TYPE_VCNS_HF3FS || IsEventReportStorageType(type) ||
                    (scope.CheckStorageTypeWaterLevelExceed() && !scope.GetWaterLevelExceedByType(type))) {
                    continue;
                }
                usage = SaturatingAdd(usage, indexer->GetStorageUsageByType(type));
            }
        }
        if (usage > 0) {
            eligible.emplace(info->instance_id(), EligibleInstance{info, key_count});
        }
    }
    out_plan.eligible_instance_count = eligible.size();
    if (eligible.empty()) {
        return false;
    }
    const auto n = eligible.size();
    if (configured_batch_size > std::numeric_limits<std::size_t>::max() / n ||
        out_plan.normalized_sampling_size > std::numeric_limits<std::size_t>::max() / n) {
        KVCM_LOG_ERROR("trace_id [%s] group [%s] Group LRU budget overflow",
                       request_context->trace_id().c_str(),
                       instance_group.c_str());
        return false;
    }
    out_plan.theoretical_batch_size = configured_batch_size * n;
    const auto sampling_plan = std::min(out_plan.normalized_sampling_size * n, group_lru_config_.max_sampling_size);

    auto &queue = group_lru_rotation_by_group_[instance_group].instance_ids;
    std::deque<std::string> next_queue;
    std::unordered_set<std::string> present;
    for (const auto &id : queue) {
        if (eligible.count(id) && present.insert(id).second) {
            next_queue.push_back(id);
        }
    }
    for (const auto &[id, _] : eligible) {
        if (present.insert(id).second) {
            next_queue.push_back(id);
        }
    }
    queue.swap(next_queue);

    const std::size_t covered_count = std::min(n, sampling_plan);
    out_plan.partial = covered_count < n;
    std::vector<std::string> ids;
    std::vector<std::uint64_t> weights;
    ids.reserve(covered_count);
    weights.reserve(covered_count);
    for (std::size_t i = 0; i < covered_count; ++i) {
        ids.push_back(queue[i]);
        weights.push_back(eligible.at(queue[i]).key_count);
    }
    if (std::all_of(weights.begin(), weights.end(), [](auto weight) { return weight == 0; })) {
        std::fill(weights.begin(), weights.end(), 1);
    }
    const auto base_pool = std::max(covered_count, sampling_plan / 2 + sampling_plan % 2);
    std::vector<std::size_t> base, extra;
    if (!AllocateFairBudget(base_pool, std::vector<std::uint64_t>(covered_count, 1), ids, base) ||
        !AllocateFairBudget(sampling_plan - base_pool, weights, ids, extra)) {
        return false;
    }
    out_plan.items.reserve(covered_count);
    for (std::size_t i = 0; i < covered_count; ++i) {
        const auto sample_size = std::min(base[i] + extra[i], kSizeLimit - 1);
        out_plan.items.push_back({eligible.at(ids[i]).info, sample_size});
        out_plan.sampling_size += sample_size;
    }
    return !out_plan.items.empty();
}

std::size_t CacheReclaimer::GroupLruBatchSize(const GroupLruPlan &plan, std::size_t successful_sampling_size) noexcept {
    if (successful_sampling_size == 0 || plan.configured_batch_size == 0 || plan.normalized_sampling_size == 0) {
        return 0;
    }
    const auto scaled = static_cast<unsigned __int128>(successful_sampling_size) * plan.configured_batch_size /
                        plan.normalized_sampling_size;
    return std::min(plan.theoretical_batch_size, std::max(std::size_t{1}, static_cast<std::size_t>(scaled)));
}

bool CacheReclaimer::FilterGroupLruCandidates(const std::shared_ptr<RequestContext> &request_context,
                                              const WaterLevelExceed &scope,
                                              const GroupLruPlan &plan,
                                              std::size_t instance_index,
                                              const std::map<std::int64_t, std::int64_t> &times,
                                              std::chrono::steady_clock::time_point deadline,
                                              std::vector<GroupLruCandidate> &out_candidates) noexcept {
    // Do not retain Location maps for the whole sample pool. Publish this
    // Instance only if every bounded eligibility read completed successfully.
    constexpr std::size_t kLocationReadBatch = 256;
    std::vector<GroupLruCandidate> candidates;
    auto it = times.begin();
    while (it != times.end()) {
        if (!IsRunning() || IsPaused() || std::chrono::steady_clock::now() >= deadline) {
            return false;
        }
        std::vector<std::int64_t> keys;
        keys.reserve(std::min(kLocationReadBatch, times.size()));
        for (; it != times.end() && keys.size() < kLocationReadBatch; ++it) {
            keys.push_back(it->first);
        }
        std::vector<std::vector<std::string>> location_ids;
        BytesByStorageType bytes{};
        CountsByStorageType counts{};
        std::uint64_t predicted_keys = 0;
        AgeStats ages;
        if (!FilterLocIDImpl(request_context.get(),
                             plan.items[instance_index].instance_info,
                             keys,
                             scope,
                             location_ids,
                             bytes,
                             counts,
                             predicted_keys,
                             ages,
                             true,
                             true) ||
            !IsRunning() || IsPaused() || std::chrono::steady_clock::now() >= deadline) {
            return false;
        }
        for (std::size_t i = 0; i < keys.size(); ++i) {
            if (!location_ids[i].empty()) {
                candidates.push_back({instance_index, keys[i], times.at(keys[i])});
            }
        }
    }
    if (!IsRunning() || IsPaused() || std::chrono::steady_clock::now() >= deadline) {
        return false;
    }
    out_candidates.insert(out_candidates.end(), candidates.begin(), candidates.end());
    return true;
}

bool CacheReclaimer::CollectGroupLruCandidates(const std::shared_ptr<RequestContext> &request_context,
                                               const WaterLevelExceed &scope,
                                               GroupLruPlan &plan,
                                               std::vector<GroupLruCandidate> &out_candidates,
                                               std::size_t &out_successful_sampling_size) noexcept {
    out_candidates.clear();
    out_successful_sampling_size = 0;
    if (plan.items.empty()) {
        return false;
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(future_timeout_ms_.load());
    // As in the existing sampler, zero disables per-Instance task splitting.
    const auto per_task = sampling_size_per_task_.load();
    struct State {
        std::size_t remaining{0};
        std::size_t outstanding{0};
        bool started{false};
        bool failed{false};
        bool collected{false};
        std::shared_ptr<std::atomic<bool>> cancelled{std::make_shared<std::atomic<bool>>(false)};
        std::map<std::int64_t, std::int64_t> times;
    };
    struct Task {
        std::size_t instance_index;
        std::future<SamplingResult> future;
    };
    std::vector<State> states(plan.items.size());
    std::deque<std::size_t> ready;
    for (std::size_t i = 0; i < states.size(); ++i) {
        states[i].remaining = plan.items[i].sampling_size;
        ready.push_back(i);
    }
    auto &queue = group_lru_rotation_by_group_[plan.items.front().instance_info->instance_group_name()].instance_ids;
    while (!ready.empty() && IsRunning() && !IsPaused() && std::chrono::steady_clock::now() < deadline) {
        const auto in_flight = in_flight_sampling_tasks_.load();
        if (in_flight >= workers_.size()) {
            break;
        }
        const auto available = workers_.size() - in_flight;
        std::vector<Task> wave;
        wave.reserve(available);
        while (wave.size() < available && !ready.empty() && IsRunning() && !IsPaused() &&
               std::chrono::steady_clock::now() < deadline) {
            const auto index = ready.front();
            ready.pop_front();
            auto &state = states[index];
            if (state.failed) {
                continue;
            }
            const auto &id = plan.items[index].instance_info->instance_id();
            if (!state.started) {
                state.started = true;
                // First attempts always follow the current queue prefix.
                // Move even a failed attempt behind never-attempted Instances.
                if (!queue.empty() && queue.front() == id) {
                    queue.push_back(std::move(queue.front()));
                    queue.pop_front();
                }
            }
            const auto indexer = meta_indexer_manager_->GetMetaIndexer(id);
            if (!indexer) {
                state.failed = true;
                continue;
            }
            const auto count = per_task == 0 ? state.remaining : std::min(per_task, state.remaining);
            state.remaining -= count;
            ++state.outstanding;
            if (state.remaining > 0) {
                ready.push_back(index);
            }
            // A request-local collector must not be shared by parallel workers.
            auto task_context = std::make_shared<RequestContext>(request_context->trace_id());
            auto cancelled = state.cancelled;
            wave.push_back({index, SubmitSamplingTask([this, indexer, task_context, cancelled, count, deadline]() {
                                SamplingResult result;
                                const auto active = [&]() {
                                    return !cancelled->load(std::memory_order_relaxed) && IsRunning() && !IsPaused() &&
                                           std::chrono::steady_clock::now() < deadline;
                                };
                                if (!active()) {
                                    return result;
                                }
                                result.ec =
                                    indexer->SampleReclaimKeysForMaintenance(task_context.get(), count, result.keys);
                                if (result.ec != ErrorCode::EC_OK) {
                                    return result;
                                }
                                if (result.keys.size() > count || !active()) {
                                    return SamplingResult{};
                                }
                                if (result.keys.empty()) {
                                    return result;
                                }
                                const auto properties = indexer->GetPropertiesForMaintenance(
                                    task_context.get(), result.keys, {PROPERTY_LRU_TIME}, result.maps);
                                if (properties.error_codes.size() != result.keys.size() ||
                                    result.maps.size() != result.keys.size()) {
                                    return SamplingResult{};
                                }
                                std::size_t kept = 0;
                                for (std::size_t i = 0; i < result.keys.size(); ++i) {
                                    const auto ec = properties.error_codes[i];
                                    if (ec == ErrorCode::EC_NOENT) {
                                        continue;
                                    }
                                    if (ec != ErrorCode::EC_OK) {
                                        return SamplingResult{};
                                    }
                                    if (kept != i) {
                                        result.keys[kept] = result.keys[i];
                                        result.maps[kept] = std::move(result.maps[i]);
                                    }
                                    ++kept;
                                }
                                result.keys.resize(kept);
                                result.maps.resize(kept);
                                return active() ? std::move(result) : SamplingResult{};
                            })});
        }
        std::size_t waiting = wave.size();
        while (waiting > 0 && IsRunning() && !IsPaused() && std::chrono::steady_clock::now() < deadline) {
            bool received = false;
            for (auto &task : wave) {
                if (!task.future.valid() ||
                    task.future.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready) {
                    continue;
                }
                auto sampled = task.future.get();
                received = true;
                --waiting;
                auto &state = states[task.instance_index];
                --state.outstanding;
                if (sampled.ec != ErrorCode::EC_OK) {
                    state.failed = true;
                    state.cancelled->store(true, std::memory_order_relaxed);
                }
                if (state.failed) {
                    state.times.clear();
                    continue;
                }
                METRICS_(cache_reclaimer, group_lru_sampled_key_count) += sampled.keys.size();
                for (std::size_t i = 0; i < sampled.keys.size(); ++i) {
                    std::int64_t time = 0;
                    const auto prop = sampled.maps[i].find(PROPERTY_LRU_TIME);
                    if (prop == sampled.maps[i].end() || !StringUtil::StrToInt64(prop->second.c_str(), time) ||
                        time <= 0) {
                        time = 0;
                        METRICS_(cache_reclaimer, group_lru_invalid_time_count) += 1;
                    }
                    auto [it, inserted] = state.times.emplace(sampled.keys[i], time);
                    if (!inserted) {
                        it->second = std::max(it->second, time);
                    }
                }
                if (state.remaining == 0 && state.outstanding == 0) {
                    state.collected = FilterGroupLruCandidates(
                        request_context, scope, plan, task.instance_index, state.times, deadline, out_candidates);
                    state.failed = !state.collected;
                    state.times.clear();
                    if (state.collected) {
                        out_successful_sampling_size += plan.items[task.instance_index].sampling_size;
                    }
                }
            }
            if (!received) {
                // Poll all ready results before waiting, so one slow Instance
                // cannot prevent healthy results from being filtered in time.
                for (auto &task : wave) {
                    if (task.future.valid()) {
                        const auto remaining = deadline - std::chrono::steady_clock::now();
                        if (remaining > std::chrono::steady_clock::duration::zero()) {
                            task.future.wait_for(
                                std::min(remaining,
                                         std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                                             std::chrono::milliseconds(1))));
                        }
                        break;
                    }
                }
            }
        }
        if (waiting > 0) {
            break;
        }
    }
    std::size_t started = 0, collected = 0, failed = 0;
    for (auto &state : states) {
        state.cancelled->store(true, std::memory_order_relaxed);
        started += state.started;
        collected += state.collected;
        failed += state.failed;
    }
    plan.partial = collected < plan.eligible_instance_count;
    METRICS_(cache_reclaimer, group_lru_started_instance_count) += started;
    METRICS_(cache_reclaimer, group_lru_collected_instance_count) += collected;
    METRICS_(cache_reclaimer, group_lru_failed_instance_count) += failed;
    METRICS_(cache_reclaimer, group_lru_skipped_instance_count) += plan.eligible_instance_count - started;
    if (std::chrono::steady_clock::now() >= deadline && collected < plan.items.size()) {
        METRICS_(cache_reclaimer, group_lru_deadline_count) += 1;
    }
    return collected > 0 && IsRunning() && !IsPaused();
}

CacheReclaimer::ReclaimResult
CacheReclaimer::TryReclaimOnGroupLru(const std::shared_ptr<RequestContext> &request_context,
                                     const std::shared_ptr<const InstanceGroup> &instance_group,
                                     const std::shared_ptr<CacheReclaimStrategy> &reclaim_strategy,
                                     const std::vector<std::shared_ptr<const InstanceInfo>> &instance_infos) noexcept {
    ReclaimResult result;
    if (!IsRunning() || IsPaused()) {
        return result;
    }
    const auto policy = reclaim_strategy->reclaim_policy();
    if (policy != ReclaimPolicy::POLICY_UNSPECIFIED && policy != ReclaimPolicy::POLICY_LRU) {
        KVCM_LOG_WARN("group [%s] Group LRU requires an LRU reclaim policy", instance_group->name().c_str());
        return result;
    }
    const auto delay = reclaim_strategy->delay_before_delete_ms();
    const auto read_water_level = [&]() {
        return GetWaterLevelExceed(
            request_context.get(), instance_group->name(), instance_group->quota(), reclaim_strategy, instance_infos);
    };
    const auto initial = read_water_level();
    if (!IsTriggerReclaiming(initial)) {
        return result;
    }
    result.water_level_exceeded = true;
    const auto process_full = [&]() {
        return pending_delete_handler_count_ >= async_delete_config_.pending_delete_handler_limit ||
               pending_delete_bytes_ >= async_delete_config_.pending_bytes_limit;
    };
    if (process_full()) {
        METRICS_(cache_reclaimer, group_lru_backpressure_stop_count) += 1;
        return result;
    }
    GroupLruPlan plan;
    if (!BuildGroupLruPlan(request_context.get(),
                           instance_group->name(),
                           *initial,
                           instance_infos,
                           sampling_size_.load(),
                           batching_size_.load(),
                           plan)) {
        METRICS_(cache_reclaimer, group_lru_plan_failure_count) += 1;
        return result;
    }
    METRICS_(cache_reclaimer, group_lru_plan_count) += 1;
    METRICS_(cache_reclaimer, group_lru_eligible_instance_count) += plan.eligible_instance_count;
    std::vector<GroupLruCandidate> candidates;
    std::size_t successful_sampling_size = 0;
    const auto collect_begin = TimestampUtil::GetSteadyTimeUs();
    const bool collected =
        CollectGroupLruCandidates(request_context, *initial, plan, candidates, successful_sampling_size);
    METRICS_(cache_reclaimer, group_lru_collect_duration_us) = TimestampUtil::GetSteadyTimeUs() - collect_begin;
    if (plan.partial) {
        METRICS_(cache_reclaimer, group_lru_partial_plan_count) += 1;
    }
    if (!collected) {
        return result;
    }
    METRICS_(cache_reclaimer, group_lru_candidate_count) += candidates.size();
    const auto sort_begin = TimestampUtil::GetSteadyTimeUs();
    std::sort(candidates.begin(), candidates.end(), [&](const auto &a, const auto &b) {
        return std::tie(a.lru_time_us, plan.items[a.instance_index].instance_info->instance_id(), a.block_key) <
               std::tie(b.lru_time_us, plan.items[b.instance_index].instance_info->instance_id(), b.block_key);
    });
    candidates.resize(std::min(candidates.size(), GroupLruBatchSize(plan, successful_sampling_size)));
    METRICS_(cache_reclaimer, group_lru_sort_duration_us) = TimestampUtil::GetSteadyTimeUs() - sort_begin;
    METRICS_(cache_reclaimer, group_lru_selected_block_count) += candidates.size();

    const auto scope_active = [&]() {
        const auto current = read_water_level();
        if (!current) {
            return false;
        }
        if (!IsTriggerReclaiming(current)) {
            METRICS_(cache_reclaimer, group_lru_watermark_stop_count) += 1;
            return false;
        }
        if (!SameReclaimScope(*initial, *current)) {
            METRICS_(cache_reclaimer, group_lru_scope_change_stop_count) += 1;
            return false;
        }
        return true;
    };
    const auto request_limit = std::min(plan.configured_batch_size, kSizeLimit - 1);
    const auto submit_begin = TimestampUtil::GetSteadyTimeUs();
    std::size_t attempted_requests = 0;
    for (std::size_t position = 0; position < candidates.size();) {
        if (!IsRunning() || IsPaused() || !scope_active()) {
            break;
        }
        if (process_full()) {
            METRICS_(cache_reclaimer, group_lru_backpressure_stop_count) += 1;
            break;
        }
        if (attempted_requests >= group_lru_config_.max_delete_requests_per_round) {
            METRICS_(cache_reclaimer, group_lru_request_limit_count) += 1;
            break;
        }
        const auto index = candidates[position].instance_index;
        const auto &info = plan.items[index].instance_info;
        CacheLocationDelRequest request;
        request.instance_id = info->instance_id();
        request.delay = std::chrono::milliseconds(delay);
        while (position < candidates.size() && candidates[position].instance_index == index &&
               request.block_keys.size() < request_limit) {
            request.block_keys.push_back(candidates[position++].block_key);
        }
        BytesByStorageType bytes{};
        CountsByStorageType counts{};
        std::uint64_t predicted_keys = 0;
        AgeStats ages;
        if (!FilterLocIDImpl(request_context.get(),
                             info,
                             request.block_keys,
                             *initial,
                             request.location_ids,
                             bytes,
                             counts,
                             predicted_keys,
                             ages,
                             false,
                             true)) {
            continue;
        }
        const auto nonempty_blocks = std::count_if(request.location_ids.begin(),
                                                   request.location_ids.end(),
                                                   [](const auto &locations) { return !locations.empty(); });
        if (nonempty_blocks == 0) {
            continue;
        }
        if (!IsRunning() || IsPaused() || !scope_active()) {
            break;
        }
        ++attempted_requests;
        METRICS_(cache_reclaimer, group_lru_delete_request_count) += 1;
        if (SubmitDelReq(request_context, info, request, bytes, counts, predicted_keys)) {
            result.made_progress = true;
            METRICS_(cache_reclaimer, reclaim_job_count) += 1;
            METRICS_(cache_reclaimer, group_lru_submitted_block_count) += nonempty_blocks;
            if (!scope_active()) {
                break;
            }
        }
    }
    METRICS_(cache_reclaimer, group_lru_submit_duration_us) = TimestampUtil::GetSteadyTimeUs() - submit_begin;
    KVCM_LOG_DEBUG("trace_id [%s] group [%s] Group LRU: eligible [%zu] covered [%zu] partial [%d] "
                   "sample budget [%zu] successful [%zu] top blocks [%zu] request attempts [%zu]",
                   request_context->trace_id().c_str(),
                   instance_group->name().c_str(),
                   plan.eligible_instance_count,
                   plan.items.size(),
                   plan.partial,
                   plan.sampling_size,
                   successful_sampling_size,
                   candidates.size(),
                   attempted_requests);
    return result;
}

} // namespace kv_cache_manager
