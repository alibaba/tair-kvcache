#include "kv_cache_manager/meta/meta_store_health_tracker.h"

#include <algorithm>

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/timestamp_util.h"

namespace kv_cache_manager {

MetaStoreHealthTracker::MetaStoreHealthTracker(std::string name, Config config) : name_(std::move(name)) {
    Configure(name_, config);
}

void MetaStoreHealthTracker::Configure(std::string name, Config config) noexcept {
    name_ = std::move(name);
    config_ = config;
    config_.failure_threshold = std::max<int64_t>(1, config_.failure_threshold);
    config_.probe_interval_ms = std::max<int64_t>(1, config_.probe_interval_ms);
    // Open() has just proven the store reachable (or the backend would not be
    // serving at all), so start the probe clock instead of probing immediately.
    last_probe_time_ms_.store(TimestampUtil::GetSteadyTimeMs(), std::memory_order_relaxed);
}

void MetaStoreHealthTracker::RecordSuccess() noexcept {
    consecutive_failures_.store(0, std::memory_order_relaxed);
    if (!available_.exchange(true, std::memory_order_acq_rel)) {
        ReportTransition(true);
    }
}

void MetaStoreHealthTracker::RecordFailure() noexcept {
    // Saturate at the threshold: the counter only has to distinguish "below
    // threshold" from "at threshold", and an unbounded increment would keep
    // growing for the whole duration of a long outage.
    int64_t failures = consecutive_failures_.load(std::memory_order_relaxed);
    while (failures < config_.failure_threshold) {
        if (consecutive_failures_.compare_exchange_weak(failures, failures + 1, std::memory_order_acq_rel)) {
            ++failures;
            break;
        }
    }
    if (failures < config_.failure_threshold) {
        return;
    }
    if (available_.exchange(false, std::memory_order_acq_rel)) {
        ReportTransition(false);
    }
}

void MetaStoreHealthTracker::RecordProbeFailure() noexcept {
    consecutive_failures_.store(config_.failure_threshold, std::memory_order_relaxed);
    if (available_.exchange(false, std::memory_order_acq_rel)) {
        ReportTransition(false);
    }
}

bool MetaStoreHealthTracker::ShouldProbe() noexcept {
    const int64_t now_ms = TimestampUtil::GetSteadyTimeMs();
    int64_t last_probe_ms = last_probe_time_ms_.load(std::memory_order_relaxed);
    if (now_ms - last_probe_ms < config_.probe_interval_ms) {
        return false;
    }
    return last_probe_time_ms_.compare_exchange_strong(last_probe_ms, now_ms, std::memory_order_relaxed);
}

void MetaStoreHealthTracker::ReportTransition(bool available) noexcept {
    if (available) {
        KVCM_LOG_INFO("meta store[%s] recovered, durable metadata writes and reclaim resume", name_.c_str());
        return;
    }
    KVCM_LOG_ERROR("meta store[%s] unavailable, metadata can no longer be persisted durably; new location allocation "
                   "is rejected and location deletion is deferred until it recovers",
                   name_.c_str());
}

} // namespace kv_cache_manager
