#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <forward_list>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "kv_cache_manager/common/error_code.h"
#include "kv_cache_manager/metrics/metrics_registry.h"

namespace kv_cache_manager {

#ifndef KVCM_METRICS_FOR_CACHE_RECLAIMER
#define KVCM_METRICS_FOR_CACHE_RECLAIMER(name)                                                                         \
public:                                                                                                                \
    DECLARE_METRICS_NAME_(cache_reclaimer, name);                                                                      \
    DEFINE_GET_METRICS_COUNTER_(cache_reclaimer, name)                                                                 \
                                                                                                                       \
private:                                                                                                               \
    DECLARE_METRICS_COUNTER_(cache_reclaimer, name);
#endif

class CacheReclaimStrategy;
class EventManager;
class InstanceGroup;
class InstanceGroupQuota;
class InstanceInfo;
class MetaIndexerManager;
class MetaSearcherManager;
class RegistryManager;
class RequestContext;
class SchedulePlanExecutor;
struct CacheLocationDelRequest;
struct PlanExecuteResult;

/**
 * @brief Manages cache reclamation operations to free up memory by
 * removing the least valuable entries
 *
 * The CacheReclaimer is responsible for periodically checking and
 * removing expired or less frequently used cache entries based on
 * configured policies.  It operates in its own dedicated thread and
 * uses various strategies like LRU, LFU, and TTL for determining which
 * entries to reclaim.
 *
 * The reclaimer works by:
 * 1. Running a periodic cron job in a separate thread
 * 2. Checking if reclamation is needed based on quota usage and time
 *    thresholds
 * 3. Sampling keys from instances and identifying the least valuable
 *    ones
 * 4. Submitting deletion requests for locations in CLS_SERVING status
 *    only for those keys
 *
 * @note This class is not thread-safe and should only be accessed by
 * the cache-manager thread.
 */
class CacheReclaimer {
public:
    /**
     * @brief Delete default constructor
     */
    CacheReclaimer() = delete;

    /**
     * @brief Construct a new CacheReclaimer object
     *
     * @param registry_manager Shared pointer to RegistryManager for
     * retrieving instance groups and instances
     * @param meta_indexer_manager Shared pointer to MetaIndexerManager
     * for getting keys of an instance
     * @param meta_searcher_manager Shared pointer to MetaSearcherManager
     * for cache location info query
     * @param sched_plan_executor Shared pointer to SchedulePlanExecutor
     * for submitting deletion tasks
     * @param metrics_registry Shared pointer to MetricsRegistry
     * for cache-reclaimer related metrics data management
     * @param event_manager Shared pointer to EventManager for event
     * publish
     */
    CacheReclaimer(std::shared_ptr<RegistryManager> registry_manager,
                   std::shared_ptr<MetaIndexerManager> meta_indexer_manager,
                   std::shared_ptr<MetaSearcherManager> meta_searcher_manager,
                   std::shared_ptr<SchedulePlanExecutor> sched_plan_executor,
                   std::shared_ptr<MetricsRegistry> metrics_registry,
                   std::shared_ptr<EventManager> event_manager);

    /**
     * @brief Delete copy constructor
     */
    CacheReclaimer(const CacheReclaimer &) = delete;

    /**
     * @brief Delete move constructor
     */
    CacheReclaimer(CacheReclaimer &&) = delete;

    /**
     * @brief Delete copy assignment operator
     */
    CacheReclaimer &operator=(const CacheReclaimer &) = delete;

    /**
     * @brief Delete move assignment operator
     */
    CacheReclaimer &operator=(CacheReclaimer &&) = delete;

    /**
     * @brief Destroy the CacheReclaimer object and clean up resources
     *
     * This will stop the reclaimer thread if it is running.
     */
    ~CacheReclaimer();

    /**
     * @brief Start the execution of the working thread that performs
     *        the reclaiming cron job
     *
     * There can only be 0 or 1 job existing at any given time.
     *
     * @return ErrorCode indicating success or failure of the operation
     * - EC_OK: Job started successfully
     * - EC_EXIST: A job is already running
     * - EC_ERROR: Error occurred, e.g., required dependencies
     *   are null
     */
    ErrorCode Start() noexcept;

    /**
     * @brief Stop the reclaiming cron job and wait for its finishing
     *
     * This method will block until the reclaimer thread has finished
     * executing.
     */
    void Stop() noexcept;

    /**
     * @brief Check if the reclaiming job is running
     *
     * @return true if the reclaiming job is running
     * @return false if the reclaiming job is stopped
     */
    [[nodiscard]] bool IsRunning() noexcept;

    /**
     * @brief Pause or resume the reclaiming job
     *
     * When paused, the reclaiming cron job will be idling
     */
    void Pause() noexcept;
    void Resume() noexcept;

    /**
     * @brief Check if the reclaiming job is paused/resumed
     *
     * @return true if the reclaiming job is paused
     * @return false if the reclaiming job is resumed
     */
    [[nodiscard]] bool IsPaused() const noexcept;

    /**
     * @brief Get the current sampling size
     *
     * @param request_context The context of the request
     * @return std::size_t Number of keys sampled per instance in a
     * single reclaiming round
     */
    [[nodiscard]] std::size_t GetSamplingSize(const RequestContext *request_context) const noexcept;

    /**
     * @brief Set the sampling size
     *
     * @param request_context The context of the request
     * @param sampling_size Number of keys to sample per instance in a
     * single reclaiming round.  Value must be in range [0, kSizeLimit)
     * @return ErrorCode indicating success or failure of the operation
     * - EC_OK: Value set successfully
     * - EC_OUT_OF_RANGE: Value is outside the valid range
     */
    [[nodiscard]] ErrorCode SetSamplingSize(const RequestContext *request_context, std::size_t sampling_size) noexcept;

    /**
     * @brief Get the current batching size
     *
     * @param request_context The context of the request
     * @return std::size_t Maximum allowed key size of one deletion
     * request submitted to the executor
     */
    [[nodiscard]] std::size_t GetBatchingSize(const RequestContext *request_context) const noexcept;

    /**
     * @brief Set the batching size
     *
     * @param request_context The context of the request
     * @param batching_size Maximum allowed key size of one deletion
     * request submitted to the executor.  Value must be in range
     * [0, kSizeLimit)
     * @return ErrorCode indicating success or failure of the operation
     * - EC_OK: Value set successfully
     * - EC_OUT_OF_RANGE: Value is outside the valid range
     */
    [[nodiscard]] ErrorCode SetBatchingSize(const RequestContext *request_context, std::size_t batching_size) noexcept;

    /**
     * @brief Get the sleep interval between cron jobs
     *
     * @param request_context The context of the request
     * @return std::uint32_t Current sleep interval of the working
     * thread between two cron jobs, in milliseconds
     */
    [[nodiscard]] std::uint32_t GetSleepIntervalMs(const RequestContext *request_context) const noexcept;

    /**
     * @brief Set the sleep interval between cron jobs
     *
     * @param request_context The context of the request
     * @param sleep_interval_ms Maximum sleep interval of the working
     * thread between two cron jobs, in milliseconds
     */
    void SetSleepIntervalMs(const RequestContext *request_context, std::uint32_t sleep_interval_ms) noexcept;

private:
    static constexpr double kEpsilon = 1e-9;
    static constexpr std::size_t kSizeLimit = 1 << 16;
    static const std::string kTraceIDPrefix;
    static std::string GenTraceID();

    // to retrieve the instance_group and instance list
    const std::shared_ptr<RegistryManager> registry_manager_;
    // to get the keys of an instance
    const std::shared_ptr<MetaIndexerManager> meta_indexer_manager_;
    // to query the location status
    const std::shared_ptr<MetaSearcherManager> meta_searcher_manager_;
    // for submitting the deleting tasks
    const std::shared_ptr<SchedulePlanExecutor> sched_plan_executor_;
    // for metrics
    const std::shared_ptr<MetricsRegistry> metrics_registry_;
    // for event publish
    const std::shared_ptr<EventManager> event_manager_;

    // represents the object of the associated working thread
    std::thread reclaimer_;

    // communicates the job status with the working thread
    std::mutex cv_mutex_; // the mutex to protect the condition
    std::condition_variable cv_job_state_;
    bool job_state_flag_; // the condition, true: job running, false: job stopped

    // communicates the pause/resume status with the working thread
    std::atomic<bool> pause_flag_;

    // controls how many keys would be sampled per instance in a single
    // reclaiming round
    // range: [0, kSizeLimit)
    // default to 1000
    std::atomic<std::size_t> sampling_size_;

    // controls the maximum allowed key size of one del request that
    // would be submitted to the executor
    // range: [0, kSizeLimit)
    // default to 100
    std::atomic<std::size_t> batching_size_;

    // controls the maximum sleep interval of the working thread between
    // two cron jobs, in milliseconds
    // default to 100
    std::atomic<std::uint32_t> sleep_interval_ms_;

    // a singly-linked list to help inspect the deleting result in a
    // non-blocking way
    struct DeleteHandler {
        const std::shared_ptr<RequestContext> req_ctx_;
        const std::string ins_id_;
        const std::string ins_gr_;
        const std::uint64_t blk_count_;
        const std::uint64_t loc_count_;
        std::future<PlanExecuteResult> fut_;

        DeleteHandler(std::shared_ptr<RequestContext> req_ctx,
                      std::string ins_id,
                      std::string ins_gr,
                      std::uint64_t blk_count,
                      std::uint64_t loc_count,
                      std::future<PlanExecuteResult> fut);
    };
    std::forward_list<DeleteHandler> delete_handlers_;

    /**
     * @brief Determine if a reclamation should be triggered for an
     * instance group
     *
     * A reclamation is triggered based on multiple criteria:
     * 1. Group total used percentage exceeds quota threshold
     * 2. Group total used percentage of key count exceeds threshold
     * 3. A storage type used percentage exceeds it's quota threshold
     *
     * @param request_context The context of the request
     * @param instance_group_name Name of the instance group to check
     * @param instance_group_quota Quota info for the instance group
     * @param reclaim_strategy Strategy to use for reclamation decisions
     * @param instance_infos Vector of instance information
     * @param out_water_level_exceed_results the detailed trigger result
     * @return true if reclamation should be triggered
     * @return false if reclamation should not be triggered
     */
    [[nodiscard]] bool IsTriggerReclaiming(const RequestContext *request_context,
                                           const std::string &instance_group_name,
                                           const InstanceGroupQuota &instance_group_quota,
                                           const std::shared_ptr<CacheReclaimStrategy> &reclaim_strategy,
                                           const std::vector<std::shared_ptr<const InstanceInfo>> &instance_infos,
                                           std::array<bool, 5> &out_water_level_exceed_results) noexcept;

    /**
     * @brief Reclaim cache entries using LRU (Least Recently Used)
     * strategy
     *
     * This method samples keys from each instance, retrieves their LRU
     * timestamps, sorts them, and deletes the oldest keys up to the
     * batching size limit.
     *
     * @param request_context A shared_ptr that holds the context of the
     * request
     * @param instance_info Instance information to process
     * @param delay_before_delete_ms delay milliseconds for executor
     */
    void ReclaimByLRU(const std::shared_ptr<RequestContext> &request_context,
                      const std::shared_ptr<const InstanceInfo> &instance_info,
                      const std::array<bool, 5> &water_level_exceed_results,
                      std::int32_t delay_before_delete_ms) noexcept;

    /**
     * @brief Reclaim cache entries using LFU (Least Frequently Used)
     * strategy
     *
     * Currently not implemented and falls back to LRU strategy.
     *
     * @param request_context A shared_ptr that holds the context of the
     * request
     * @param instance_info Instance information to process
     * @param delay_before_delete_ms delay milliseconds for executor
     */
    void ReclaimByLFU(const std::shared_ptr<RequestContext> &request_context,
                      const std::shared_ptr<const InstanceInfo> &instance_info,
                      const std::array<bool, 5> &water_level_exceed_results,
                      std::int32_t delay_before_delete_ms) noexcept;

    /**
     * @brief Reclaim cache entries using TTL (Time To Live) strategy
     *
     * Currently not implemented and falls back to LRU strategy.
     *
     * @param request_context A shared_ptr that holds the context of the
     * request
     * @param instance_info Instance information to process
     * @param delay_before_delete_ms delay milliseconds for executor
     */
    void ReclaimByTTL(const std::shared_ptr<RequestContext> &request_context,
                      const std::shared_ptr<const InstanceInfo> &instance_info,
                      const std::array<bool, 5> &water_level_exceed_results,
                      std::int32_t delay_before_delete_ms) noexcept;

    bool TryReclaimOnGroup(const std::shared_ptr<RequestContext> &request_context,
                           const std::shared_ptr<const InstanceGroup> &instance_group) noexcept;

    void HandleDelRes() noexcept;

    /**
     * @brief Main cron job function that runs in the reclaimer thread
     *
     * This function periodically:
     * 1. Sleeps for the configured interval
     * 2. Lists all instance groups
     * 3. For each group, checks if reclamation should be triggered
     * 4. If triggered, runs the appropriate reclamation strategy
     */
    void ReclaimCron() noexcept;

    // below are helper routines for internal usage
    bool DoKeySampling(RequestContext *request_context,
                       const std::shared_ptr<const InstanceInfo> &instance_info,
                       std::vector<std::int64_t> &out_keys,
                       std::vector<std::map<std::string, std::string>> &out_maps) const noexcept;

    bool MakeBatchByLRU(const RequestContext *request_context,
                        const std::shared_ptr<const InstanceInfo> &instance_info,
                        const std::vector<std::int64_t> &sampled_keys,
                        const std::vector<std::map<std::string, std::string>> &property_maps,
                        std::vector<std::int64_t> &out_batch) const noexcept;

    bool FilterLocID(RequestContext *request_context,
                     const std::shared_ptr<const InstanceInfo> &instance_info,
                     const std::vector<std::int64_t> &batch,
                     const std::array<bool, 5> &water_level_exceed_results,
                     std::vector<std::vector<std::string>> &out_loc_ids) const noexcept;

    void SubmitDelReq(const std::shared_ptr<RequestContext> &request_context,
                      const std::shared_ptr<const InstanceInfo> &instance_info,
                      const CacheLocationDelRequest &req) noexcept;

    struct GroupUsageData {
        std::size_t grp_used_key_cnt_ = 0;
        std::size_t grp_max_key_cnt_ = 0;
        std::size_t grp_used_byte_sz_ = 0;
        std::array<std::uint64_t, 5> grp_storage_usage_array_{0, 0, 0, 0, 0};
    };

    [[nodiscard]] GroupUsageData
    GetGroupUsageData(const RequestContext *request_context,
                      const std::vector<std::shared_ptr<const InstanceInfo>> &instance_infos) const noexcept;

    KVCM_METRICS_FOR_CACHE_RECLAIMER(block_submit_count)
    KVCM_METRICS_FOR_CACHE_RECLAIMER(location_submit_count)
    KVCM_METRICS_FOR_CACHE_RECLAIMER(block_del_count)
    KVCM_METRICS_FOR_CACHE_RECLAIMER(location_del_count)
};

#undef KVCM_METRICS_FOR_CACHE_RECLAIMER

} // namespace kv_cache_manager
