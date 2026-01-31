#pragma once

#include <atomic>
#include <map>
#include <memory>
#include <shared_mutex>
#include <thread>
#include <unordered_map>

#include "kv_cache_manager/common/error_code.h"

namespace kv_cache_manager {

class MetaIndexerManager;
class WriteLocationManager;
class RegistryManager;

class CacheManagerMetricsRecorder {
public:
    struct InstanceMetric {
        size_t key_count = 0;
        size_t byte_size = 0;
    };
    using GroupUsageRatioMap = std::map<std::string, double>;
    using GroupInstanceIdMetricMap = std::map<std::string, std::unordered_map<std::string, InstanceMetric>>;

    CacheManagerMetricsRecorder(std::shared_ptr<MetaIndexerManager> meta_indexer_manager,
                                std::shared_ptr<WriteLocationManager> write_location_manager,
                                std::shared_ptr<RegistryManager> registry_manager);
    ~CacheManagerMetricsRecorder();
    void Start();
    void Stop();
    void DoCleanup();
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
    mutable std::shared_mutex mutex_;

    std::shared_ptr<MetaIndexerManager> meta_indexer_manager_;
    std::shared_ptr<WriteLocationManager> write_location_manager_;
    std::shared_ptr<RegistryManager> registry_manager_;
};

} // namespace kv_cache_manager