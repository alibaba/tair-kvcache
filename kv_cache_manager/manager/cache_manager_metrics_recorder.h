#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <thread>
#include <unordered_map>

#include "kv_cache_manager/common/error_code.h"

namespace kv_cache_manager {

class MetaIndexerManager;
class WriteLocationManager;
class RegistryManager;
struct MetricsLifecycle;

class CacheManagerMetricsRecorder {
public:
    struct InstanceMetric {
        size_t key_count = 0;
        size_t byte_size = 0;
        int64_t async_queue_max_size = 0;
        int64_t async_queue_avg_size = 0;
        int64_t async_flush_key_count = 0;
        int64_t async_batch_flush_time_us = 0;
        int64_t async_pipeline_error_count = 0;
        int64_t max_lru_age_us = 0;
    };
    using GroupUsageRatioMap = std::map<std::string, double>;
    using GroupInstanceIdMetricMap = std::map<std::string, std::unordered_map<std::string, InstanceMetric>>;

    CacheManagerMetricsRecorder(std::shared_ptr<MetaIndexerManager> meta_indexer_manager,
                                std::shared_ptr<WriteLocationManager> write_location_manager,
                                std::shared_ptr<RegistryManager> registry_manager,
                                std::shared_ptr<MetricsLifecycle> metrics_lifecycle);
    ~CacheManagerMetricsRecorder();
    void Start();
    void Stop();
    void DoCleanup();
    void RemoveInstance(const std::string &instance_id);
    void RemoveGroup(const std::string &group_name);
    size_t write_location_expire_size() const;
    GroupUsageRatioMap group_usage_ratio_map() const;
    GroupInstanceIdMetricMap group_instance_id_metric_map() const;

private:
    void RecorderLoop();

    // instance_group_name -> usage_ratio
    GroupUsageRatioMap group_usage_ratio_map_;
    // instance_group_name -> instance_id -> InstanceMetric
    GroupInstanceIdMetricMap group_instance_id_metric_map_;

    std::thread recorder_thread_;
    std::atomic_bool stop_ = false;
    std::mutex stop_mutex_;
    std::condition_variable stop_cv_;
    mutable std::shared_mutex mutex_;

    std::shared_ptr<MetaIndexerManager> meta_indexer_manager_;
    std::shared_ptr<WriteLocationManager> write_location_manager_;
    std::shared_ptr<RegistryManager> registry_manager_;
    // shared between this recorder, CacheManager, AdminServiceImpl and
    // the metrics reporters; the recorder loop holds a shared lock
    // around the registry-read+publish span so removals cannot slip
    // between sampling and publishing a stale snapshot
    std::shared_ptr<MetricsLifecycle> metrics_lifecycle_;
};

} // namespace kv_cache_manager
