#include "kv_cache_manager/manager/cache_reclaimer.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cinttypes>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <future>
#include <iomanip>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

#include "kv_cache_manager/common/error_code.h"
#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/common/string_util.h"
#include "kv_cache_manager/config/cache_config.h"
#include "kv_cache_manager/config/cache_reclaim_strategy.h"
#include "kv_cache_manager/config/instance_group.h"
#include "kv_cache_manager/config/instance_group_quota.h"
#include "kv_cache_manager/config/instance_info.h"
#include "kv_cache_manager/config/registry_manager.h"
#include "kv_cache_manager/event/event_manager.h"
#include "kv_cache_manager/event/spec_events/cache_reclaim_event.h"
#include "kv_cache_manager/manager/cache_location.h"
#include "kv_cache_manager/manager/meta_searcher.h"
#include "kv_cache_manager/manager/meta_searcher_manager.h"
#include "kv_cache_manager/manager/schedule_plan_executor.h"
#include "kv_cache_manager/meta/common.h"
#include "kv_cache_manager/meta/meta_indexer.h"
#include "kv_cache_manager/meta/meta_indexer_manager.h"
#include "kv_cache_manager/metrics/metrics_registry.h"

namespace kv_cache_manager {

#define LOG_WITH_TRACE(LEVEL, format, args...)                                                                         \
    do {                                                                                                               \
        KVCM_LOG_##LEVEL("trace_id [%s] | " format, request_context->trace_id().c_str(), ##args);                      \
    } while (0)

#define LOG_WITH_GR(LEVEL, format, args...)                                                                            \
    do {                                                                                                               \
        KVCM_LOG_##LEVEL("trace_id [%s] | instance_group [%s] | " format,                                              \
                         request_context->trace_id().c_str(),                                                          \
                         ins_gr.c_str(),                                                                               \
                         ##args);                                                                                      \
    } while (0)

#define LOG_WITH_ID(LEVEL, format, args...)                                                                            \
    do {                                                                                                               \
        KVCM_LOG_##LEVEL("trace_id [%s] | instance_id [%s] | instance_group [%s] | " format,                           \
                         request_context->trace_id().c_str(),                                                          \
                         ins_id.c_str(),                                                                               \
                         ins_gr.c_str(),                                                                               \
                         ##args);                                                                                      \
    } while (0)

#define DEFINE_METRICS_NAME_FOR_CACHE_RECLAIMER(name) DEFINE_METRICS_NAME_(CacheReclaimer, cache_reclaimer, name)

#define REGISTER_METRICS_FOR_CACHE_RECLAIMER(name) REGISTER_METRICS_COUNTER_(metrics_registry_, cache_reclaimer, name)

DEFINE_METRICS_NAME_FOR_CACHE_RECLAIMER(block_submit_count);
DEFINE_METRICS_NAME_FOR_CACHE_RECLAIMER(location_submit_count);
DEFINE_METRICS_NAME_FOR_CACHE_RECLAIMER(block_del_count);
DEFINE_METRICS_NAME_FOR_CACHE_RECLAIMER(location_del_count);

const std::string CacheReclaimer::kTraceIDPrefix{"cache_reclaimer_internal_trace_"};

inline std::string CacheReclaimer::GenTraceID() {
    // generate a random 64-bit unsigned integer
    static std::random_device rd;
    static std::mt19937_64 rng(rd());
    static std::uniform_int_distribution<std::uint64_t> dis;

    const std::uint64_t rand_val = dis(rng);

    // convert to hexadecimal string representation
    std::stringstream ss;
    ss << kTraceIDPrefix << std::right << std::setfill('0') << std::setw(16) << std::hex << std::noshowbase << rand_val;

    return ss.str();
}

CacheReclaimer::CacheReclaimer(std::shared_ptr<RegistryManager> registry_manager,
                               std::shared_ptr<MetaIndexerManager> meta_indexer_manager,
                               std::shared_ptr<MetaSearcherManager> meta_searcher_manager,
                               std::shared_ptr<SchedulePlanExecutor> sched_plan_executor,
                               std::shared_ptr<MetricsRegistry> metrics_registry,
                               std::shared_ptr<EventManager> event_manager)
    : registry_manager_(std::move(registry_manager))
    , meta_indexer_manager_(std::move(meta_indexer_manager))
    , meta_searcher_manager_(std::move(meta_searcher_manager))
    , sched_plan_executor_(std::move(sched_plan_executor))
    , metrics_registry_(std::move(metrics_registry))
    , event_manager_(std::move(event_manager))
    , job_state_flag_(false)
    , pause_flag_(false)
    , sampling_size_(1000)
    , batching_size_(100)
    , sleep_interval_ms_(100) {}

CacheReclaimer::~CacheReclaimer() { this->Stop(); }

ErrorCode CacheReclaimer::Start() noexcept {
    if (registry_manager_ == nullptr) {
        KVCM_LOG_ERROR("registry manager is nullptr");
        return ErrorCode::EC_ERROR;
    }

    if (meta_indexer_manager_ == nullptr) {
        KVCM_LOG_ERROR("meta indexer manager is nullptr");
        return ErrorCode::EC_ERROR;
    }

    if (meta_searcher_manager_ == nullptr) {
        KVCM_LOG_ERROR("meta searcher manager is nullptr");
        return ErrorCode::EC_ERROR;
    }

    if (sched_plan_executor_ == nullptr) {
        KVCM_LOG_ERROR("schedule plan executor is nullptr");
        return ErrorCode::EC_ERROR;
    }

    if (metrics_registry_ == nullptr) {
        KVCM_LOG_ERROR("metrics registry is nullptr");
        return ErrorCode::EC_ERROR;
    }

    // allow event_manager_ to be nullptr

    REGISTER_METRICS_FOR_CACHE_RECLAIMER(block_submit_count);
    REGISTER_METRICS_FOR_CACHE_RECLAIMER(location_submit_count);
    REGISTER_METRICS_FOR_CACHE_RECLAIMER(block_del_count);
    REGISTER_METRICS_FOR_CACHE_RECLAIMER(location_del_count);

    {
        std::unique_lock<std::mutex> lock(cv_mutex_);
        if (job_state_flag_
            // there is no need to check on reclaimer_.joinable(), because
            // under our usage model assumption, there will be no parallel
            // calls to this->Start() and this->Stop(); any previous
            // this->Stop() call would block the calling thread to wait for
            // its finishing, and synchronise with the completion of the
            // working thread
        ) {
            KVCM_LOG_ERROR("cannot start new reclaiming job; there is already a running one");
            return ErrorCode::EC_EXIST;
        }
        job_state_flag_ = true;
    }

    reclaimer_ = std::thread([this]() -> void { this->ReclaimCron(); });
    KVCM_LOG_INFO("cache reclaimer start OK");
    return ErrorCode::EC_OK;
}

void CacheReclaimer::Stop() noexcept {
    {
        std::unique_lock<std::mutex> lock(cv_mutex_);
        if (!job_state_flag_) {
            return;
        }
        job_state_flag_ = false;
        cv_job_state_.notify_one();
    }

    if (reclaimer_.joinable()) {
        reclaimer_.join();
    }
    KVCM_LOG_DEBUG("cache reclaimer stop OK");
}

bool CacheReclaimer::IsRunning() noexcept {
    std::unique_lock<std::mutex> lock(cv_mutex_);
    return job_state_flag_;
}

void CacheReclaimer::Pause() noexcept { pause_flag_.store(true); }

void CacheReclaimer::Resume() noexcept { pause_flag_.store(false); }

bool CacheReclaimer::IsPaused() const noexcept { return pause_flag_.load(); }

std::size_t CacheReclaimer::GetSamplingSize(const RequestContext *request_context) const noexcept {
    const std::size_t sampling_size = sampling_size_.load();
    LOG_WITH_TRACE(DEBUG, "sampling size is [%zu]", sampling_size);
    return sampling_size;
}

ErrorCode CacheReclaimer::SetSamplingSize(const RequestContext *request_context,
                                          const std::size_t sampling_size) noexcept {
    if (sampling_size >= kSizeLimit) {
        LOG_WITH_TRACE(
            ERROR, "set sampling size failed: sampling_size [%zu] >= kSizeLimit [%zu]", sampling_size, kSizeLimit);
        return ErrorCode::EC_OUT_OF_RANGE;
    }
    sampling_size_.store(sampling_size);
    LOG_WITH_TRACE(DEBUG, "set sampling size [%zu]", sampling_size);
    return ErrorCode::EC_OK;
}

std::size_t CacheReclaimer::GetBatchingSize(const RequestContext *request_context) const noexcept {
    const std::size_t batching_size = batching_size_.load();
    LOG_WITH_TRACE(DEBUG, "batching size is [%zu]", batching_size);
    return batching_size;
}

ErrorCode CacheReclaimer::SetBatchingSize(const RequestContext *request_context,
                                          const std::size_t batching_size) noexcept {
    if (batching_size >= kSizeLimit) {
        LOG_WITH_TRACE(
            ERROR, "set batching size failed: batching_size [%zu] >= kSizeLimit [%zu]", batching_size, kSizeLimit);
        return ErrorCode::EC_OUT_OF_RANGE;
    }
    batching_size_.store(batching_size);
    LOG_WITH_TRACE(DEBUG, "set batching size [%zu]", batching_size);
    return ErrorCode::EC_OK;
}

std::uint32_t CacheReclaimer::GetSleepIntervalMs(const RequestContext *request_context) const noexcept {
    const std::uint32_t sleep_interval_ms = sleep_interval_ms_.load();
    LOG_WITH_TRACE(DEBUG, "sleep interval is [%u] ms", sleep_interval_ms);
    return sleep_interval_ms;
}

void CacheReclaimer::SetSleepIntervalMs(const RequestContext *request_context,
                                        const std::uint32_t sleep_interval_ms) noexcept {
    sleep_interval_ms_.store(sleep_interval_ms);
    LOG_WITH_TRACE(DEBUG, "set sleep interval to [%u] ms", sleep_interval_ms);
}

bool CacheReclaimer::IsTriggerReclaiming(const RequestContext *request_context,
                                         const std::string &ins_gr,
                                         const InstanceGroupQuota &instance_group_quota,
                                         const std::shared_ptr<CacheReclaimStrategy> &reclaim_strategy,
                                         const std::vector<std::shared_ptr<const InstanceInfo>> &instance_infos,
                                         std::array<bool, 5> &out_water_level_exceed_results) noexcept {
    if (!IsRunning() || IsPaused()) {
        // fast exiting in the middle of one job round
        return false;
    }

    // TODO (rui): per storage backend reclaiming is not supported due
    //             to lacking of capacity and usage info

    // NOTE: reclaim_strategy->storage_unique_name() is ignored
    //       reclaim_strategy->trigger_strategy().used_size() is ignored

    // NOTE: the trigger detecting strategy is based on:
    //       1. the entire instance group usage and capacity quota
    //       2. storage type usage and capacity quota for this group

    auto get_trigger_result = [&request_context, &ins_gr, &out_water_level_exceed_results]() -> bool {
        if (std::any_of(out_water_level_exceed_results.cbegin(),
                        out_water_level_exceed_results.cend(),
                        [](const bool v) -> bool { return v; })) {
            LOG_WITH_GR(DEBUG, "instance group trigger reclaiming");
            return true;
        }
        LOG_WITH_GR(DEBUG, "instance group does not trigger reclaiming");
        return false;
    };

    // 1. calculate the key count and used byte size of this group
    const GroupUsageData data = GetGroupUsageData(request_context, instance_infos);

    // 2. generate the result water level exceeding array
    const double threshold_used_percentage = reclaim_strategy->trigger_strategy().used_percentage();

    // 2.1. results for each storage type
    for (const auto &storage_quota : instance_group_quota.quota_config()) {
        const auto &type = storage_quota.storage_spec();
        if (type == DataStorageType::DATA_STORAGE_TYPE_VCNS_HF3FS) {
            // skip vcns_hf3fs, because it is treated as hf3fs
            continue;
        }

        try {
            if (storage_quota.capacity() <= 0) {
                LOG_WITH_GR(DEBUG,
                            "instance group storage type [%d] capacity quota used percentage [inf] "
                            "has reached or exceeded the threshold percentage [%f]",
                            static_cast<std::uint8_t>(type),
                            threshold_used_percentage);
                out_water_level_exceed_results.at(static_cast<std::uint8_t>(type)) = true;
                continue;
            }
            if (const double storage_type_wl =
                    static_cast<double>(data.grp_storage_usage_array_.at(static_cast<std::uint8_t>(type))) /
                    static_cast<double>(storage_quota.capacity());
                storage_type_wl + kEpsilon > threshold_used_percentage) {
                LOG_WITH_GR(DEBUG,
                            "instance group storage type [%d] capacity quota used percentage [%f] "
                            "has reached or exceeded the threshold percentage [%f]",
                            static_cast<std::uint8_t>(type),
                            storage_type_wl,
                            threshold_used_percentage);
                out_water_level_exceed_results.at(static_cast<std::uint8_t>(type)) = true;
            }
        } catch (const std::out_of_range &e) {
            LOG_WITH_GR(WARN, "data storage type out of range: %d", static_cast<std::uint8_t>(type));
        }
    }

    // 2.2. result for the entire instance group
    if (data.grp_used_key_cnt_ == 0 || data.grp_used_byte_sz_ == 0) {
        out_water_level_exceed_results[0] = false;
        return get_trigger_result();
    }

    // 2.2.1. trigger_strategy:used_percent for instance group quota capacity
    if (instance_group_quota.capacity() <= 0) {
        // proceed as group quota capacity is 0
        LOG_WITH_GR(DEBUG,
                    "instance group capacity quota used percentage [inf] "
                    "has reached or exceeded the threshold percentage [%f]",
                    threshold_used_percentage);
        out_water_level_exceed_results[0] = true;
        return get_trigger_result();
    }
    if (const double group_used_percentage =
            static_cast<double>(data.grp_used_byte_sz_) / static_cast<double>(instance_group_quota.capacity());
        group_used_percentage + kEpsilon > threshold_used_percentage) {
        LOG_WITH_GR(DEBUG,
                    "instance group capacity quota used percentage [%f] "
                    "has reached or exceeded the threshold percentage [%f]",
                    group_used_percentage,
                    threshold_used_percentage);
        out_water_level_exceed_results[0] = true;
        return get_trigger_result();
    }

    // 2.2.2. trigger_strategy:used_percent for group key count
    if (data.grp_max_key_cnt_ == 0) {
        LOG_WITH_GR(DEBUG,
                    "instance group total key count used percentage [inf] "
                    "has reached or exceeded the threshold percentage [%f]",
                    threshold_used_percentage);
        out_water_level_exceed_results[0] = true;
        return get_trigger_result();
    }
    if (const double group_used_percentage =
            static_cast<double>(data.grp_used_key_cnt_) / static_cast<double>(data.grp_max_key_cnt_);
        group_used_percentage + kEpsilon > threshold_used_percentage) {
        LOG_WITH_GR(DEBUG,
                    "instance group total key count used percentage [%f] "
                    "has reached or exceeded the threshold percentage [%f]",
                    group_used_percentage,
                    threshold_used_percentage);
        out_water_level_exceed_results[0] = true;
        return get_trigger_result();
    }

    // 2.2.3. instance group do not trigger reclaiming
    out_water_level_exceed_results[0] = false;
    return get_trigger_result();
}

void CacheReclaimer::ReclaimByLRU(const std::shared_ptr<RequestContext> &request_context,
                                  const InstanceInfoConstPtr &instance_info,
                                  const std::array<bool, 5> &water_level_exceed_results,
                                  const std::int32_t delay_before_delete_ms) noexcept {
    if (!IsRunning() || IsPaused()) {
        // fast exiting in the middle of one job round
        return;
    }

    if (instance_info == nullptr) {
        LOG_WITH_TRACE(WARN, "instance is nullptr");
        return;
    }

    const std::string &ins_id = instance_info->instance_id();
    const std::string &ins_gr = instance_info->instance_group_name();

    std::vector<std::int64_t> keys;
    std::vector<std::map<std::string, std::string>> maps;

    // 1. get the sampled block keys and the LRU timestamp info from
    // the meta indexer
    if (!DoKeySampling(request_context.get(), instance_info, keys, maps)) {
        LOG_WITH_ID(DEBUG, "key sampling failed");
        return;
    }
    LOG_WITH_ID(DEBUG, "[%zu] key(s) sampled", keys.size());

    // init the deleting request with content to be filled later
    // here the cache location form of deleting request is used to
    // permit the cache location status aware deleting control
    CacheLocationDelRequest request;
    request.instance_id = ins_id;
    request.delay = std::chrono::milliseconds(delay_before_delete_ms);

    // 2. constitute the batch based on the LRU timestamp info
    if (!MakeBatchByLRU(request_context.get(), instance_info, keys, maps, request.block_keys)) {
        LOG_WITH_ID(DEBUG, "make batch failed");
        return;
    }
    LOG_WITH_ID(DEBUG, "batch is made with size [%zu]", request.block_keys.size());
    if (request.block_keys.empty()) {
        return;
    }

    // 3. inspect the cache location status for every blocks so that
    // only the cache locations with the "CLS_SERVING" status are
    // submitted to be deleted
    if (!FilterLocID(request_context.get(),
                     instance_info,
                     request.block_keys,
                     water_level_exceed_results,
                     request.location_ids)) {
        LOG_WITH_ID(DEBUG, "filter location ID failed");
        return;
    }

    // 4. submit the final deleting request to the executor
    SubmitDelReq(request_context, instance_info, request);
}

void CacheReclaimer::ReclaimByLFU(const std::shared_ptr<RequestContext> &request_context,
                                  const InstanceInfoConstPtr &instance_info,
                                  const std::array<bool, 5> &water_level_exceed_results,
                                  const std::int32_t delay_before_delete_ms) noexcept {
    if (!IsRunning() || IsPaused()) {
        // fast exiting in the middle of one job round
        return;
    }

    // TODO: impl LFU policy
    LOG_WITH_TRACE(WARN, "LFU reclaim policy not supported yet; fall back to LRU policy");
    ReclaimByLRU(request_context, instance_info, water_level_exceed_results, delay_before_delete_ms);
}

void CacheReclaimer::ReclaimByTTL(const std::shared_ptr<RequestContext> &request_context,
                                  const InstanceInfoConstPtr &instance_info,
                                  const std::array<bool, 5> &water_level_exceed_results,
                                  const std::int32_t delay_before_delete_ms) noexcept {
    if (!IsRunning() || IsPaused()) {
        // fast exiting in the middle of one job round
        return;
    }

    // TODO: impl TTL policy
    LOG_WITH_TRACE(WARN, "TTL reclaim policy not supported yet; fall back to LRU policy");
    ReclaimByLRU(request_context, instance_info, water_level_exceed_results, delay_before_delete_ms);
}

void CacheReclaimer::ReclaimCron() noexcept {
    std::uint32_t sleep_interval_ms = sleep_interval_ms_.load();
    while (true) {
        {
            std::unique_lock<std::mutex> lock(cv_mutex_);
            if (!job_state_flag_) {
                // prevent unnecessary sleeping
                break;
            }

            cv_job_state_.wait_for(lock, std::chrono::milliseconds(sleep_interval_ms));
            if (!job_state_flag_) {
                break;
            }
        }

        if (IsPaused()) {
            continue;
        }

        const auto request_context = std::make_shared<RequestContext>(GenTraceID());

        const auto [ec, instance_groups] = registry_manager_->ListInstanceGroup(request_context.get());
        if (ec != ErrorCode::EC_OK) {
            LOG_WITH_TRACE(WARN, "list instance group failed, error code: [%d]", static_cast<std::int32_t>(ec));
            continue;
        }

        bool triggered = false;
        for (const auto &instance_group : instance_groups) {
            if (TryReclaimOnGroup(request_context, instance_group)) {
                triggered = true;
            }
        }

        HandleDelRes();

        if (triggered) {
            sleep_interval_ms = 0;
        } else {
            sleep_interval_ms = sleep_interval_ms_.load();
        }
    }
}

bool CacheReclaimer::DoKeySampling(RequestContext *request_context,
                                   const std::shared_ptr<const InstanceInfo> &instance_info,
                                   std::vector<std::int64_t> &out_keys,
                                   std::vector<std::map<std::string, std::string>> &out_maps) const noexcept {
    const std::string &ins_id = instance_info->instance_id();
    const std::string &ins_gr = instance_info->instance_group_name();

    const auto meta_indexer = meta_indexer_manager_->GetMetaIndexer(ins_id);
    if (meta_indexer == nullptr) {
        LOG_WITH_ID(WARN, "meta indexer is nullptr");
        return false;
    }

    if (const auto ec = meta_indexer->RandomSample(sampling_size_.load(), out_keys); ec != ErrorCode::EC_OK) {
        LOG_WITH_ID(WARN, "random sample failed, error code: [%d]", static_cast<std::int32_t>(ec));
        return false;
    }

    if (out_keys.empty()) {
        LOG_WITH_ID(DEBUG, "random sample got empty keys");
        return false;
    }

    if (const auto res = meta_indexer->GetProperties(request_context, out_keys, {PROPERTY_LRU_TIME}, out_maps);
        res.ec != ErrorCode::EC_OK) {
        LOG_WITH_ID(WARN, "get properties failed, error code: [%d]", static_cast<std::int32_t>(res.ec));
        return false;
    }

    if (out_keys.size() != out_maps.size()) {
        LOG_WITH_ID(
            WARN, "num of sampled keys [%zu] and property maps [%zu] do not match", out_keys.size(), out_maps.size());
        return false;
    }

    return true;
}

bool CacheReclaimer::MakeBatchByLRU(const RequestContext *request_context,
                                    const std::shared_ptr<const InstanceInfo> &instance_info,
                                    const std::vector<std::int64_t> &sampled_keys,
                                    const std::vector<std::map<std::string, std::string>> &property_maps,
                                    std::vector<std::int64_t> &out_batch) const noexcept {
    // the sampled_keys' and property_maps' size are guaranteed to be
    // equal and the content of them are guaranteed to be matched by
    // index

    const std::string &ins_id = instance_info->instance_id();
    const std::string &ins_gr = instance_info->instance_group_name();

    // the PROPERTY_LRU_TIME value is represented as an int64_t type
    // timepoint string; parse them into integers
    std::vector<std::int64_t> lru_ts_vec;
    lru_ts_vec.reserve(sampled_keys.size());

    for (const auto &m : property_maps) {
        if (const auto it = m.find(PROPERTY_LRU_TIME); it != m.end()) {
            const auto &lru_ts_str = it->second;
            std::int64_t lru_ts;
            if (!StringUtil::StrToInt64(lru_ts_str.c_str(), lru_ts)) {
                LOG_WITH_ID(WARN, "lru_time str [%s] to int64 failed", lru_ts_str.c_str());
                return false;
            }
            lru_ts_vec.emplace_back(lru_ts);
        } else {
            LOG_WITH_ID(WARN, "PROPERTY_LRU_TIME not found");
            return false;
        }
    }

    if (lru_ts_vec.size() != sampled_keys.size()) {
        LOG_WITH_ID(WARN,
                    "LRU timestamp vec size [%zu] and key sample size [%zu] not match",
                    lru_ts_vec.size(),
                    sampled_keys.size());
        return false;
    }

    const std::size_t batching_size = batching_size_.load();
    if (batching_size == 0) {
        return true;
    }

    // assume M random keys are sampled every round and
    // N is the target batching size
    // the (N - 1)th timestamp should be chosen as the threshold
    std::int64_t ts_threshold;
    std::vector<std::int64_t> lru_ts_vec_sorted(lru_ts_vec);
    std::sort(lru_ts_vec_sorted.begin(), lru_ts_vec_sorted.end());
    if (lru_ts_vec_sorted.size() < batching_size) {
        // handle the M < N case: can not constitute a full-sized batch;
        // select the last one as the threshold so all the timestamps
        // can get picked
        ts_threshold = lru_ts_vec_sorted.back();
    } else {
        ts_threshold = lru_ts_vec_sorted[batching_size - 1];
    }

    // constitute the batch to be submitted for deleting
    // the first N timestamp would be picked out
    out_batch.reserve(batching_size);
    for (std::size_t i = 0; i != sampled_keys.size(); ++i) {
        if (lru_ts_vec[i] <= ts_threshold) {
            out_batch.emplace_back(sampled_keys[i]);
        }
        if (out_batch.size() == batching_size) {
            // the batch is successfully constituted
            return true;
        }
    }

    // the only reason possible to reach here is the M < N case, e.g.,
    // the number of sampled keys is not enough to constitute a
    // full-sized batch; the batch size shall be equal to the size of
    // sampled keys
    if (out_batch.size() != sampled_keys.size()) {
        // this is unusual
        LOG_WITH_ID(
            WARN,
            "wrong batch size, final batch size: [%zu], sampled keys size: [%zu], intended batching size: [%zu]",
            out_batch.size(),
            sampled_keys.size(),
            batching_size);
        return false;
    }

    // permit a no-full-sized batch
    return true;
}

bool CacheReclaimer::FilterLocID(RequestContext *request_context,
                                 const std::shared_ptr<const InstanceInfo> &instance_info,
                                 const std::vector<std::int64_t> &batch,
                                 const std::array<bool, 5> &water_level_exceed_results,
                                 std::vector<std::vector<std::string>> &out_loc_ids) const noexcept {
    const std::string &ins_id = instance_info->instance_id();
    const std::string &ins_gr = instance_info->instance_group_name();

    const auto meta_searcher = meta_searcher_manager_->GetMetaSearcher(ins_id);
    if (meta_searcher == nullptr) {
        LOG_WITH_ID(WARN, "meta searcher is nullptr");
        return false;
    }

    // get the location map of each block in the batch
    std::vector<CacheLocationMap> loc_maps;
    const BlockMask blk_mask(std::in_place_type<BlockMaskVector>, batch.size(), false);
    assert(std::holds_alternative<BlockMaskVector>(blk_mask));
    if (const auto ec = meta_searcher->BatchGetLocation(request_context, batch, blk_mask, loc_maps);
        ec != ErrorCode::EC_OK) {
        LOG_WITH_ID(WARN, "get cache location maps failed, error code: [%d]", static_cast<std::int32_t>(ec));
        return false;
    }

    if (loc_maps.size() != batch.size()) {
        LOG_WITH_ID(WARN,
                    "get cache location maps failed: result vec size [%zu] not match batch size [%zu]",
                    loc_maps.size(),
                    batch.size());
        return false;
    }

    // detect if any storage type water level exceeded
    const bool any_st_wl_exceeded = std::any_of(water_level_exceed_results.cbegin() + 1,
                                                water_level_exceed_results.cend(),
                                                [](const bool v) -> bool { return v; });

    // inspect the cache location status of each block and get the
    // filtered location ID vecs
    out_loc_ids.reserve(loc_maps.size());
    for (const auto &loc_map : loc_maps) {
        std::vector<std::string> loc_id_vec;
        for (const auto &[_, loc] : loc_map) {
            // pick out only the cache location with "serving" status
            if (loc.status() == CacheLocationStatus::CLS_SERVING) {
                if (any_st_wl_exceeded) {
                    // some storage type water level exceeded; only
                    // collect the location with matched type but
                    // fairness is ignored
                    // TODO (rui): implement the fair eviction
                    auto type = loc.type();
                    if (type == DataStorageType::DATA_STORAGE_TYPE_VCNS_HF3FS) {
                        // treat vcns_hf3fs as hf3fs
                        type = DataStorageType::DATA_STORAGE_TYPE_HF3FS;
                    }
                    if (const auto t = static_cast<std::uint8_t>(type); t > 0 && t < 5) {
                        if (water_level_exceed_results[t]) {
                            loc_id_vec.emplace_back(loc.id());
                        }
                    } else {
                        LOG_WITH_ID(WARN, "invalid storage type: [%d]", t);
                    }
                } else {
                    // there's no storage type water level exceeded
                    // and since the reclaiming is triggered, the total
                    // usage water level must be exceeded; ignore the
                    // type detection
                    loc_id_vec.emplace_back(loc.id());
                }
            }
        }
        out_loc_ids.emplace_back(std::move(loc_id_vec));
    }
    return true;
}

void CacheReclaimer::SubmitDelReq(const std::shared_ptr<RequestContext> &request_context,
                                  const std::shared_ptr<const InstanceInfo> &instance_info,
                                  const CacheLocationDelRequest &req) noexcept {
    if (!IsRunning() || IsPaused()) {
        // fast exiting in the middle of one job round
        return;
    }

    assert(req.block_keys.size() == req.location_ids.size());

    const std::string &ins_id = instance_info->instance_id();
    const std::string &ins_gr = instance_info->instance_group_name();

    std::uint64_t blk_count = 0;
    std::uint64_t loc_count = 0;
    for (std::size_t i = 0; i != req.block_keys.size(); ++i) {
        if (!req.location_ids[i].empty()) {
            ++blk_count;
            loc_count += req.location_ids[i].size();
        }
    }

    auto fut = sched_plan_executor_->Submit(req);
    delete_handlers_.emplace_front(request_context, ins_id, ins_gr, blk_count, loc_count, std::move(fut));

    METRICS_(cache_reclaimer, block_submit_count) += blk_count;
    METRICS_(cache_reclaimer, location_submit_count) += loc_count;

    if (event_manager_ != nullptr && blk_count != 0 && loc_count != 0) {
        auto reclaim_submit_event = std::make_shared<CacheReclaimSubmitEvent>(ins_id);
        reclaim_submit_event->SetEventTriggerTime();
        reclaim_submit_event->SetAdditionalArgs(req.block_keys, req.location_ids, req.delay.count());
        event_manager_->Publish(reclaim_submit_event);
    }

    LOG_WITH_ID(DEBUG,
                "submit reclaim request to schedule plan executor OK, "
                "with effective cache block count: [%" PRIu64 "], "
                "cache location count: [%" PRIu64 "]",
                blk_count,
                loc_count);
}

CacheReclaimer::GroupUsageData CacheReclaimer::GetGroupUsageData(
    const RequestContext *request_context,
    const std::vector<std::shared_ptr<const InstanceInfo>> &instance_infos) const noexcept {
    GroupUsageData data;
    for (const auto &instance_info : instance_infos) {
        if (instance_info == nullptr) {
            LOG_WITH_TRACE(WARN, "instance is nullptr");
            continue;
        }

        const std::string &ins_id = instance_info->instance_id();
        const std::string &ins_gr = instance_info->instance_group_name();
        const auto meta_indexer = meta_indexer_manager_->GetMetaIndexer(ins_id);
        if (meta_indexer == nullptr) {
            LOG_WITH_ID(WARN, "meta indexer is nullptr");
            continue;
        }

        meta_indexer->PersistMetaData();
        const std::size_t ins_used_key_cnt = meta_indexer->GetKeyCount();
        const std::size_t ins_max_key_cnt = meta_indexer->GetMaxKeyCount();

        // calc the total usage of this group
        std::size_t byte_size_per_key = 0;
        for (auto &location_spec_info : instance_info->location_spec_infos()) {
            byte_size_per_key += location_spec_info.size();
        }
        const std::size_t ins_used_byte_size = byte_size_per_key * ins_used_key_cnt;

        data.grp_used_key_cnt_ += ins_used_key_cnt;
        data.grp_max_key_cnt_ += ins_max_key_cnt;
        data.grp_used_byte_sz_ += ins_used_byte_size;

        // calc the usage size of each storage type of this group
        auto calc_sz = [&request_context, &ins_gr, &ins_id, &meta_indexer, &data](const DataStorageType &type) -> void {
            try {
                const std::uint64_t sz = meta_indexer->storage_usage_array_.at(static_cast<std::uint8_t>(type)).load();
                data.grp_storage_usage_array_.at(static_cast<std::uint8_t>(type)) += sz;
            } catch (const std::out_of_range &e) {
                LOG_WITH_ID(WARN, "data storage type out of range: %d", static_cast<std::uint8_t>(type));
            }
        };
        calc_sz(DataStorageType::DATA_STORAGE_TYPE_HF3FS);
        calc_sz(DataStorageType::DATA_STORAGE_TYPE_MOONCAKE);
        calc_sz(DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL);
        calc_sz(DataStorageType::DATA_STORAGE_TYPE_NFS);
        // vcns_hf3fs is merged into hf3fs, no need to calc the size
    }
    return data;
}

void CacheReclaimer::HandleDelRes() noexcept {
    auto it_pre = delete_handlers_.before_begin();
    auto it = delete_handlers_.begin();
    while (it != delete_handlers_.end()) {
        const auto &request_context = it->req_ctx_;
        const std::string &ins_id = it->ins_id_;
        const std::string &ins_gr = it->ins_gr_;

        if (!it->fut_.valid()) {
            LOG_WITH_ID(WARN, "reclaim request got invalid future");
            it = delete_handlers_.erase_after(it_pre);
        } else if (const auto fs = it->fut_.wait_for(std::chrono::seconds::zero()); fs == std::future_status::ready) {
            try {
                if (const auto [ec, err_msg] = it->fut_.get(); ec != ErrorCode::EC_OK) {
                    LOG_WITH_ID(WARN,
                                "reclaim request execute failed, error_code: [%d], error message: [%s]",
                                static_cast<std::int32_t>(ec),
                                err_msg.c_str());
                } else {
                    METRICS_(cache_reclaimer, block_del_count) += it->blk_count_;
                    METRICS_(cache_reclaimer, location_del_count) += it->loc_count_;
                    LOG_WITH_ID(DEBUG,
                                "reclaim request execute finished successfully, "
                                "with effective cache block count: [%" PRIu64 "], "
                                "cache location count: [%" PRIu64 "]",
                                it->blk_count_,
                                it->loc_count_);
                }
            } catch (const std::exception &e) {
                LOG_WITH_ID(WARN, "reclaim request finished with exception: [%s]", e.what());
            } catch (...) {
                // make sure no exception can possibly escape
            }

            // eliminate this handler anyway, by erasing and updating
            // the iterator ``it''; ``it_pre'' remains unchanged
            it = delete_handlers_.erase_after(it_pre);
        } else {
            // handle other std::future_status possible:
            // - std::future_status:deferred
            // - std::future_status:timeout
            ++it_pre, ++it;
        }
    }
}

bool CacheReclaimer::TryReclaimOnGroup(const std::shared_ptr<RequestContext> &request_context,
                                       const std::shared_ptr<const InstanceGroup> &instance_group) noexcept {
    if (!IsRunning() || IsPaused()) {
        // fast exiting in the middle of one job round
        return false;
    }

    if (instance_group == nullptr) {
        LOG_WITH_TRACE(WARN, "instance group is nullptr");
        return false;
    }

    const std::string &ins_gr = instance_group->name();

    const auto cache_config = instance_group->cache_config();
    if (cache_config == nullptr) {
        LOG_WITH_GR(WARN, "cache config is nullptr");
        return false;
    }

    const auto &reclaim_strategy = cache_config->reclaim_strategy();
    if (reclaim_strategy == nullptr) {
        LOG_WITH_GR(WARN, "reclaim strategy is nullptr");
        return false;
    }
    const std::int32_t delay_before_delete_ms = reclaim_strategy->delay_before_delete_ms();

    // retrieve all the instances in this group
    const auto [ec, instance_infos] = registry_manager_->ListInstanceInfo(request_context.get(), ins_gr);
    if (ec != ErrorCode::EC_OK) {
        LOG_WITH_GR(WARN, "list instances info failed, error code: [%d]", static_cast<std::int32_t>(ec));
        return false;
    }

    // do we need to reclaim the storage for this instance group?
    // <total, hf3fs, mooncake, pace, nfs>
    std::array<bool, 5> water_level_exceed_results{false, false, false, false, false};
    if (!IsTriggerReclaiming(request_context.get(),
                             ins_gr,
                             instance_group->quota(), // TODO (rui): validate the quota is valid
                             reclaim_strategy,
                             instance_infos,
                             water_level_exceed_results)) {
        return false;
    }

    // run the reclaiming algorithm with the chosen policy
    switch (reclaim_strategy->reclaim_policy()) {
    case ReclaimPolicy::POLICY_LRU:
        LOG_WITH_GR(DEBUG, "start to run the LRU reclaim policy");
        for (const auto &instance_info : instance_infos) {
            ReclaimByLRU(request_context, instance_info, water_level_exceed_results, delay_before_delete_ms);
        }
        break;

    case ReclaimPolicy::POLICY_LFU:
        LOG_WITH_GR(DEBUG, "start to run the LFU reclaim policy");
        for (const auto &instance_info : instance_infos) {
            ReclaimByLFU(request_context, instance_info, water_level_exceed_results, delay_before_delete_ms);
        }
        break;

    case ReclaimPolicy::POLICY_TTL:
        LOG_WITH_GR(DEBUG, "start to run the TTL reclaim policy");
        for (const auto &instance_info : instance_infos) {
            ReclaimByTTL(request_context, instance_info, water_level_exceed_results, delay_before_delete_ms);
        }
        break;

    case ReclaimPolicy::POLICY_UNSPECIFIED: // default to LRU
        LOG_WITH_GR(DEBUG, "reclaim policy not specified; default to LRU policy");
        // break; skipped intentionally

    default:
        for (const auto &instance_info : instance_infos) {
            ReclaimByLRU(request_context, instance_info, water_level_exceed_results, delay_before_delete_ms);
        }
        break;
    }

    return true;
}

CacheReclaimer::DeleteHandler::DeleteHandler(std::shared_ptr<RequestContext> req_ctx,
                                             std::string ins_id,
                                             std::string ins_gr,
                                             const std::uint64_t blk_count,
                                             const std::uint64_t loc_count,
                                             std::future<PlanExecuteResult> fut)
    : req_ctx_(std::move(req_ctx))
    , ins_id_(std::move(ins_id))
    , ins_gr_(std::move(ins_gr))
    , blk_count_(blk_count)
    , loc_count_(loc_count)
    , fut_(std::move(fut)) {}

} // namespace kv_cache_manager
