#include "kv_cache_manager/manager/cache_manager_metrics_recorder.h"

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/config/instance_group.h"
#include "kv_cache_manager/config/instance_info.h"
#include "kv_cache_manager/config/registry_manager.h"
#include "kv_cache_manager/manager/write_location_manager.h"
#include "kv_cache_manager/meta/meta_indexer.h"
#include "kv_cache_manager/meta/meta_indexer_manager.h"

namespace kv_cache_manager {

namespace {
constexpr static int kDefaultRecorderLoopSleepTime = 5; // seconds
constexpr static const char *kTraceId = "CacheManagerMetricsRecorderTraceId";

class SleepHelper {
public:
    ~SleepHelper() { std::this_thread::sleep_for(std::chrono::seconds(kDefaultRecorderLoopSleepTime)); }
};

} // namespace

CacheManagerMetricsRecorder::CacheManagerMetricsRecorder(std::shared_ptr<MetaIndexerManager> meta_indexer_manager,
                                                         std::shared_ptr<WriteLocationManager> write_location_manager,
                                                         std::shared_ptr<RegistryManager> registry_manager)
    : meta_indexer_manager_(meta_indexer_manager)
    , write_location_manager_(write_location_manager)
    , registry_manager_(registry_manager) {}

CacheManagerMetricsRecorder::~CacheManagerMetricsRecorder() { Stop(); }

void CacheManagerMetricsRecorder::Start() {
    recorder_thread_ = std::thread([this]() { this->RecorderLoop(); });
}

void CacheManagerMetricsRecorder::Stop() {
    stop_.store(true, std::memory_order_relaxed);
    if (recorder_thread_.joinable()) {
        recorder_thread_.join();
    }
}
void CacheManagerMetricsRecorder::DoCleanup() {
    // members will update by RecorderLoop, so no need to clean up.
    return;
}

void CacheManagerMetricsRecorder::RecorderLoop() {
    KVCM_LOG_INFO("RecorderLoop started");
    while (!stop_.load(std::memory_order_relaxed)) {
        SleepHelper sleep_helper;
        const auto request_context = std::make_shared<RequestContext>(kTraceId);
        const auto [ec, instance_groups] = registry_manager_->ListInstanceGroup(request_context.get());
        if (ec != ErrorCode::EC_OK) {
            KVCM_LOG_WARN("list instance group failed, error code: [%d]", static_cast<std::int32_t>(ec));
            continue;
        }
        GroupUsageRatioMap group_usage_ratio_map;
        GroupInstanceIdMetricMap group_instance_id_metric_map;
        for (const auto &instance_group : instance_groups) {
            const std::string &instance_group_name = instance_group->name();
            const auto [ec, instance_infos] =
                registry_manager_->ListInstanceInfo(request_context.get(), instance_group_name);
            if (ec != ErrorCode::EC_OK) {
                KVCM_LOG_WARN("list instances info failed, error code: [%d]", static_cast<std::int32_t>(ec));
                continue;
            }
            std::size_t group_byte_size = 0;
            for (const auto &instance_info : instance_infos) {
                if (instance_info == nullptr) {
                    KVCM_LOG_WARN("instance is nullptr");
                    continue;
                }
                const std::string &instance_id = instance_info->instance_id();
                const auto meta_indexer = meta_indexer_manager_->GetMetaIndexer(instance_id);
                if (meta_indexer == nullptr) {
                    KVCM_LOG_WARN("meta indexer is nullptr");
                    continue;
                }
                meta_indexer->PersistMetaData();
                const std::size_t key_cnt = meta_indexer->GetKeyCount();
                std::size_t byte_size_per_key = 0;
                for (auto &location_spec_info : instance_info->location_spec_infos()) {
                    byte_size_per_key += location_spec_info.size();
                }
                const std::size_t byte_size = byte_size_per_key * key_cnt;
                group_byte_size += byte_size;
                group_instance_id_metric_map[instance_group_name][instance_id] = InstanceMetric({key_cnt, byte_size});
            }
            int64_t capacity = instance_group->quota().capacity();
            group_usage_ratio_map[instance_group_name] =
                capacity > 0 ? (static_cast<double>(group_byte_size) / capacity) : 0;
        }
        {
            std::scoped_lock write_guard(mutex_);
            group_usage_ratio_map_ = std::move(group_usage_ratio_map);
            group_instance_id_metric_map_ = std::move(group_instance_id_metric_map);
        }
    }
}

size_t CacheManagerMetricsRecorder::write_location_expire_size() const { return write_location_manager_->ExpireSize(); }

CacheManagerMetricsRecorder::GroupUsageRatioMap CacheManagerMetricsRecorder::group_usage_ratio_map() const {
    std::shared_lock read_guard(mutex_);
    return group_usage_ratio_map_;
}

CacheManagerMetricsRecorder::GroupInstanceIdMetricMap
CacheManagerMetricsRecorder::group_instance_id_metric_map() const {
    std::shared_lock read_guard(mutex_);
    return group_instance_id_metric_map_;
}

} // namespace kv_cache_manager