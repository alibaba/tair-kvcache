#include "kv_cache_manager/manager/write_location_manager.h"

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/timestamp_util.h"
#include "kv_cache_manager/metrics/metrics_registry.h"
namespace kv_cache_manager {

namespace {
constexpr int kDefaultExpireLoopSleepTimeUs = 5 * 1000 * 1000; // us
};

// caller must hold mux_
void WriteLocationManager::SessionIdMap::AddToLocationIndexUnsafe(const std::vector<std::string> &location_ids) {
    for (const auto &id : location_ids) {
        ++location_id_index_[id];
    }
}

// caller must hold mux_
void WriteLocationManager::SessionIdMap::RemoveFromLocationIndexUnsafe(const std::vector<std::string> &location_ids) {
    for (const auto &id : location_ids) {
        if (auto it = location_id_index_.find(id); it != location_id_index_.end() && --it->second == 0) {
            location_id_index_.erase(it);
        }
    }
}

size_t WriteLocationManager::SessionIdMap::Size() const {
    std::unique_lock lock(mux_);
    return unit_map_.size();
}

bool WriteLocationManager::SessionIdMap::Empty() const {
    std::unique_lock lock(mux_);
    return unit_map_.empty();
}

WriteLocationManager::SessionStats WriteLocationManager::SessionIdMap::GetStats(int64_t now_us) const {
    std::unique_lock lock(mux_);
    SessionStats stats;
    int64_t oldest_created_at_us = 0;
    for (const auto &[_, unit] : unit_map_) {
        if (!unit) {
            continue;
        }
        stats.inflight_blocks += unit->write_location_info.keys.size();
        if (oldest_created_at_us == 0 || unit->created_at_us < oldest_created_at_us) {
            oldest_created_at_us = unit->created_at_us;
        }
    }
    if (oldest_created_at_us > 0 && now_us > oldest_created_at_us) {
        stats.oldest_age_seconds = static_cast<double>(now_us - oldest_created_at_us) / 1000000.0;
    }
    return stats;
}

int64_t WriteLocationManager::SessionIdMap::DropByExpirePoint(int64_t cur_point) {
    std::vector<ExpireUnitPtr> prepare_to_expire_units;
    {
        std::unique_lock lock(mux_);
        for (auto it = unit_map_.begin(); (it != unit_map_.end()) && (it->first <= cur_point);) {
            RemoveFromLocationIndexUnsafe(it->second->write_location_info.location_ids);
            session_id_map_impl_.erase(it->second->write_session_id);
            prepare_to_expire_units.push_back(it->second);
            it = unit_map_.erase(it);
        }
        if (prepare_to_expire_units.empty()) {
            return 0;
        }
    }
    for (auto &unit : prepare_to_expire_units) {
        KVCM_LOG_DEBUG("Expiring write_session [%s]", unit->write_session_id.c_str());
        std::unique_ptr<WriteLocationInfo> write_location_info = std::make_unique<WriteLocationInfo>();
        *write_location_info = std::move(unit->write_location_info);
        write_location_info->expired = true;
        unit->callback(std::move(write_location_info));
    }
    {
        std::unique_lock lock(mux_);
        if (unit_map_.empty()) {
            return 0;
        }
        return unit_map_.begin()->first;
    }
}

void WriteLocationManager::SessionIdMap::DropAll() {
    std::vector<ExpireUnitPtr> prepare_to_expire_units;
    {
        std::unique_lock lock(mux_);
        for (auto it = unit_map_.begin(); it != unit_map_.end();) {
            session_id_map_impl_.erase(it->second->write_session_id);
            prepare_to_expire_units.push_back(it->second);
            it = unit_map_.erase(it);
        }
        location_id_index_.clear();
    }
    for (auto &unit : prepare_to_expire_units) {
        KVCM_LOG_DEBUG("Expiring write_session [%s]", unit->write_session_id.c_str());
        std::unique_ptr<WriteLocationInfo> write_location_info = std::make_unique<WriteLocationInfo>();
        *write_location_info = std::move(unit->write_location_info);
        unit->callback(std::move(write_location_info));
    }
}

void WriteLocationManager::SessionIdMap::Put(ExpireUnitPtr unit) {
    std::unique_lock lock(mux_);
    while (unit_map_.find(unit->expire_point) != unit_map_.end()) {
        unit->expire_point++;
    }
    AddToLocationIndexUnsafe(unit->write_location_info.location_ids);
    unit_map_[unit->expire_point] = unit;
    session_id_map_impl_[unit->write_session_id] = unit->expire_point;
}

bool WriteLocationManager::SessionIdMap::GetAndDelete(const std::string &write_session_id,
                                                      WriteLocationInfo &location_info) {
    std::unique_lock lock(mux_);
    auto it_s = session_id_map_impl_.find(write_session_id);
    if (it_s == session_id_map_impl_.end()) {
        return false;
    }
    auto it_u = unit_map_.find(it_s->second);
    assert(it_u != unit_map_.end());
    RemoveFromLocationIndexUnsafe(it_u->second->write_location_info.location_ids);
    location_info = std::move(it_u->second->write_location_info);
    unit_map_.erase(it_u);
    session_id_map_impl_.erase(it_s);
    return true;
}

WriteLocationManager::WriteLocationManager(std::shared_ptr<MetricsRegistry> metrics_registry)
    : metrics_registry_(std::move(metrics_registry)) {
    next_sleep_time_us_.store(kDefaultExpireLoopSleepTimeUs, std::memory_order_relaxed);
    KVCM_LOG_DEBUG("WriteLocationManager constructed");
}

WriteLocationManager::SessionStats WriteLocationManager::GetSessionStats() const {
    return session_id_map_.GetStats(TimestampUtil::GetSteadyTimeUs());
}

WriteLocationManager::~WriteLocationManager() { Stop(); }

void WriteLocationManager::Start() {
    expire_thread_ = std::thread([this]() { this->ExpireLoop(); });
}

void WriteLocationManager::Stop() {
    {
        std::lock_guard<std::mutex> lock(stop_mutex_);
        stop_.store(true, std::memory_order_relaxed);
    }
    stop_cond_.notify_all();

    if (expire_thread_.joinable()) {
        expire_thread_.join();
    }
}

void WriteLocationManager::DoCleanup() {
    KVCM_LOG_DEBUG("Cleaning up all write sessions");
    session_id_map_.DropAll();
    next_sleep_time_us_.store(kDefaultExpireLoopSleepTimeUs, std::memory_order_relaxed);
}

void WriteLocationManager::StoreMinNextSleepTimeUs(int64_t next_sleep_time_us) {
    int64_t expected = next_sleep_time_us_.load(std::memory_order_relaxed);
    int64_t desired = std::min(expected, next_sleep_time_us);
    while (!next_sleep_time_us_.compare_exchange_weak(expected, desired, std::memory_order_relaxed)) {
        desired = std::min(expected, desired);
    }
}

void WriteLocationManager::ExpireLoop() {
    KVCM_LOG_INFO("ExpireLoop started");
    while (!stop_.load(std::memory_order_relaxed)) {
        ExpireUnitPtr unit_ptr_to_expire;
        {
            {
                std::unique_lock lock(stop_mutex_);
                stop_cond_.wait_for(lock, std::chrono::microseconds(next_sleep_time_us_), [this]() {
                    return stop_.load(std::memory_order_relaxed);
                });
            }

            if (session_id_map_.Empty()) {
                KVCM_INTERVAL_LOG_DEBUG(100, "expire queue empty");
                continue;
            }
            int64_t cur_point = TimestampUtil::GetSteadyTimeUs();
            if (int64_t next_point = session_id_map_.DropByExpirePoint(cur_point); next_point > 0) {
                StoreMinNextSleepTimeUs(next_point - cur_point);
            } else {
                next_sleep_time_us_.store(kDefaultExpireLoopSleepTimeUs, std::memory_order_relaxed);
            }
        }
    }
}

void WriteLocationManager::Put(const std::string &write_session_id,
                               std::vector<int64_t> &&keys,
                               std::vector<std::string> &&location_ids,
                               int64_t write_timeout_seconds,
                               CallBack callback) {
    KVCM_LOG_DEBUG("Putting session %s with %zu keys and %zu location_ids, timeout: %lld seconds",
                   write_session_id.c_str(),
                   keys.size(),
                   location_ids.size(),
                   static_cast<long long>(write_timeout_seconds));

    ExpireUnitPtr unit_ptr = std::make_shared<ExpireUnit>();
    unit_ptr->write_session_id = write_session_id;
    unit_ptr->created_at_us = TimestampUtil::GetSteadyTimeUs();
    unit_ptr->expire_point = unit_ptr->created_at_us + write_timeout_seconds * 1000 * 1000;
    unit_ptr->callback = std::move(callback);
    unit_ptr->write_location_info.keys = std::move(keys);
    unit_ptr->write_location_info.location_ids = std::move(location_ids);
    if (metrics_registry_ && !unit_ptr->write_location_info.keys.empty()) {
        try {
            metrics_registry_->GetCounter("write_session.blocks_total", {{"result", "started"}}) +=
                unit_ptr->write_location_info.keys.size();
        } catch (...) {
            // Monitoring must never prevent a write session from being stored.
        }
    }
    session_id_map_.Put(unit_ptr);
    StoreMinNextSleepTimeUs(write_timeout_seconds * 1000 * 1000);
}

bool WriteLocationManager::GetAndDelete(const std::string &write_session_id, WriteLocationInfo &location_info) {
    return session_id_map_.GetAndDelete(write_session_id, location_info);
}

bool WriteLocationManager::SessionIdMap::HasLocationId(const std::string &location_id) const {
    std::unique_lock lock(mux_);
    return location_id_index_.find(location_id) != location_id_index_.end();
}

bool WriteLocationManager::HasLocationId(const std::string &location_id) const {
    return session_id_map_.HasLocationId(location_id);
}

} // namespace kv_cache_manager
