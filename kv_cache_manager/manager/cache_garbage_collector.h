#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include "kv_cache_manager/common/error_code.h"
#include "kv_cache_manager/metrics/metrics_registry.h"

namespace kv_cache_manager {

#ifndef KVCM_COUNTER_METRICS_FOR_CACHE_GC
#define KVCM_COUNTER_METRICS_FOR_CACHE_GC(name)                                                                        \
public:                                                                                                                \
    DECLARE_METRICS_NAME_(cache_gc, name);                                                                             \
    DEFINE_GET_METRICS_COUNTER_(cache_gc, name)                                                                        \
                                                                                                                       \
private:                                                                                                               \
    DECLARE_METRICS_COUNTER_(cache_gc, name);
#endif

#ifndef KVCM_GAUGE_METRICS_FOR_CACHE_GC
#define KVCM_GAUGE_METRICS_FOR_CACHE_GC(name)                                                                          \
public:                                                                                                                \
    DECLARE_METRICS_NAME_(cache_gc, name);                                                                             \
    DEFINE_GET_METRICS_GAUGE_(cache_gc, name)                                                                          \
                                                                                                                       \
private:                                                                                                               \
    DECLARE_METRICS_GAUGE_(cache_gc, name);
#endif

class CacheLocation;
class DataStorageManager;
class EventManager;
class MetaIndexer;
class MetaIndexerManager;
class RegistryManager;
class RequestContext;
class SchedulePlanExecutor;
class WriteLocationManager;

class CacheGarbageCollector {
public:
    struct Config {
        std::size_t scan_batch_size = 256;
        std::uint32_t inter_batch_sleep_ms = 50;
        std::uint32_t inter_round_sleep_ms = 60000;
        std::int64_t writing_orphan_grace_period_us = 600 * 1000000LL;
        bool check_serving_data_exist = true;
        std::size_t max_deletions_per_batch = 64;
        bool enabled = true;
    };

    CacheGarbageCollector() = delete;

    CacheGarbageCollector(Config config,
                          std::shared_ptr<RegistryManager> registry_manager,
                          std::shared_ptr<MetaIndexerManager> meta_indexer_manager,
                          std::shared_ptr<SchedulePlanExecutor> sched_plan_executor,
                          std::shared_ptr<MetricsRegistry> metrics_registry,
                          std::shared_ptr<EventManager> event_manager,
                          std::shared_ptr<WriteLocationManager> write_location_manager);

    CacheGarbageCollector(const CacheGarbageCollector &) = delete;
    CacheGarbageCollector(CacheGarbageCollector &&) = delete;
    CacheGarbageCollector &operator=(const CacheGarbageCollector &) = delete;
    CacheGarbageCollector &operator=(CacheGarbageCollector &&) = delete;

    ~CacheGarbageCollector();

    ErrorCode Start() noexcept;
    void Stop() noexcept;
    [[nodiscard]] bool IsRunning() const noexcept;
    void Pause() noexcept;
    void Resume() noexcept;
    [[nodiscard]] bool IsPaused() const noexcept;

private:
    static const std::string kTraceIDPrefix;
    static std::string GenTraceID();

    void GCScanLoop() noexcept;
    void ScanInstance(RequestContext *request_context,
                     const std::string &instance_id,
                     const std::shared_ptr<MetaIndexer> &meta_indexer) noexcept;
    bool IsOrphanedWriting(const CacheLocation &loc) const noexcept;
    bool IsStaleServing(const std::string &instance_id, const CacheLocation &loc) const noexcept;

    const Config config_;
    const std::shared_ptr<RegistryManager> registry_manager_;
    const std::shared_ptr<MetaIndexerManager> meta_indexer_manager_;
    const std::shared_ptr<SchedulePlanExecutor> sched_plan_executor_;
    const std::shared_ptr<MetricsRegistry> metrics_registry_;
    const std::shared_ptr<EventManager> event_manager_;
    const std::shared_ptr<WriteLocationManager> write_location_manager_;

    std::thread gc_thread_;
    std::mutex state_mutex_;
    std::condition_variable cv_state_;
    bool running_ = false;
    std::atomic<bool> paused_{false};

    KVCM_COUNTER_METRICS_FOR_CACHE_GC(scan_round_count)
    KVCM_COUNTER_METRICS_FOR_CACHE_GC(orphaned_writing_count)
    KVCM_COUNTER_METRICS_FOR_CACHE_GC(stale_serving_count)
    KVCM_COUNTER_METRICS_FOR_CACHE_GC(location_submit_count)

    KVCM_GAUGE_METRICS_FOR_CACHE_GC(scan_batch_duration_us)
};

#undef KVCM_COUNTER_METRICS_FOR_CACHE_GC
#undef KVCM_GAUGE_METRICS_FOR_CACHE_GC

} // namespace kv_cache_manager
