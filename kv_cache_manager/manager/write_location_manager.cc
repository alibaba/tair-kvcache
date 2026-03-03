#include "kv_cache_manager/manager/write_location_manager.h"

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/timestamp_util.h"
namespace kv_cache_manager {

namespace {
constexpr static int kDefaultExpireLoopSleepTime = 5 * 1000 * 1000; // us
};

size_t WriteLocationManager::SessionIdMap::Size() const {
    std::unique_lock lock(mux_);
    return unit_map_.size();
}

bool WriteLocationManager::SessionIdMap::Empty() const {
    std::unique_lock lock(mux_);
    return unit_map_.empty();
}

int64_t WriteLocationManager::SessionIdMap::DropByExpirePoint(int64_t cur_point) {
    std::vector<ExpireUnitPtr> prepare_to_expire_units;
    {
        std::unique_lock lock(mux_);
        for (auto it = unit_map_.begin(); (it != unit_map_.end()) && (it->first <= cur_point);) {
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
        unit->callback();
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
    }
    for (auto &unit : prepare_to_expire_units) {
        KVCM_LOG_DEBUG("Expiring write_session [%s]", unit->write_session_id.c_str());
        unit->callback();
    }
}

void WriteLocationManager::SessionIdMap::Put(ExpireUnitPtr unit) {
    std::unique_lock lock(mux_);
    while (unit_map_.find(unit->expire_point) != unit_map_.end()) {
        unit->expire_point++;
    }
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
    location_info = std::move(it_u->second->write_location_info);
    unit_map_.erase(it_u);
    session_id_map_impl_.erase(it_s);
    return true;
}

WriteLocationManager::WriteLocationManager() {
    next_sleep_time_.store(kDefaultExpireLoopSleepTime, std::memory_order_relaxed);
    KVCM_LOG_DEBUG("WriteLocationManager constructed");
}

WriteLocationManager::~WriteLocationManager() { Stop(); }

void WriteLocationManager::Start() {
    expire_thread_ = std::thread([this]() { this->ExpireLoop(); });
}

void WriteLocationManager::Stop() {
    stop_.store(true, std::memory_order_relaxed);
    if (expire_thread_.joinable()) {
        expire_thread_.join();
    }
}

void WriteLocationManager::DoCleanup() {
    KVCM_LOG_DEBUG("Cleaning up all write sessions");
    session_id_map_.DropAll();
    next_sleep_time_.store(kDefaultExpireLoopSleepTime, std::memory_order_relaxed);
}

void WriteLocationManager::StoreMinNextSleepTime(int64_t next_sleep_time) {
    int64_t expected = next_sleep_time_.load(std::memory_order_relaxed);
    int64_t desired = std::min(expected, next_sleep_time);
    while (!next_sleep_time_.compare_exchange_weak(expected, desired, std::memory_order_relaxed)) {
        desired = std::min(expected, desired);
    }
}

void WriteLocationManager::ExpireLoop() {
    KVCM_LOG_INFO("ExpireLoop started");
    while (!stop_.load(std::memory_order_relaxed)) {
        ExpireUnitPtr unit_ptr_to_expire;
        {
            std::this_thread::sleep_for(std::chrono::microseconds(next_sleep_time_));

            if (session_id_map_.Empty()) {
                KVCM_INTERVAL_LOG_DEBUG(100, "expire queue empty");
                continue;
            }
            int64_t cur_point = TimestampUtil::GetSteadyTimeUs();
            if (int64_t next_point = session_id_map_.DropByExpirePoint(cur_point); next_point > 0) {
                StoreMinNextSleepTime(next_point - cur_point);
            } else {
                next_sleep_time_.store(kDefaultExpireLoopSleepTime, std::memory_order_relaxed);
            }
        }
    }
}

void WriteLocationManager::Put(const std::string &write_session_id,
                               std::vector<int64_t> &&keys,
                               std::vector<std::string> &&location_ids,
                               int64_t write_timeout_seconds,
                               std::function<void()> callback) {
    KVCM_LOG_DEBUG("Putting session %s with %zu keys and %zu location_ids, timeout: %ld seconds",
                   write_session_id.c_str(),
                   keys.size(),
                   location_ids.size(),
                   write_timeout_seconds);

    ExpireUnitPtr unit_ptr = std::make_shared<ExpireUnit>();
    unit_ptr->write_session_id = write_session_id;
    unit_ptr->expire_point = TimestampUtil::GetSteadyTimeUs() + write_timeout_seconds * 1000 * 1000;
    unit_ptr->callback = std::move(callback);
    unit_ptr->write_location_info.keys = std::move(keys);
    unit_ptr->write_location_info.location_ids = std::move(location_ids);
    session_id_map_.Put(unit_ptr);
    StoreMinNextSleepTime(write_timeout_seconds);
}

bool WriteLocationManager::GetAndDelete(const std::string &write_session_id, WriteLocationInfo &location_info) {
    return session_id_map_.GetAndDelete(write_session_id, location_info);
}

} // namespace kv_cache_manager