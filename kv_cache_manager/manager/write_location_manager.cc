#include "kv_cache_manager/manager/write_location_manager.h"

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/timestamp_util.h"
namespace kv_cache_manager {

namespace {
constexpr static int kDefaultExpireLoopSleepTime = 5; // seconds
};

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
    std::vector<ExpireUnitPtr> pending_units;
    {
        std::unique_lock<std::mutex> lock(do_expire_mutex_);

        ExpireUnitPtr unit_ptr;
        while (expire_queue_.Pop(&unit_ptr)) {
            if (session_id_map_.Contains(unit_ptr->write_session_id)) {
                pending_units.push_back(unit_ptr);
            }
        }
        next_sleep_time_.store(kDefaultExpireLoopSleepTime, std::memory_order_relaxed);
        // callback will clean session_id_map_, so we ignore it here.
    }

    for (const auto &unit : pending_units) {
        KVCM_LOG_INFO("Cleaning up abandoned write session: %s", unit->write_session_id.c_str());
        unit->callback();
    }
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
            //
            std::unique_lock<std::mutex> lock(do_expire_mutex_);
            cond_.wait_for(lock, std::chrono::seconds(next_sleep_time_));

            if (expire_queue_.Empty()) {
                KVCM_INTERVAL_LOG_DEBUG(100, "expire queue empty");
                continue;
            }
            ExpireUnitPtr unit_ptr;
            if (expire_queue_.Pop(&unit_ptr)) {
                if (!session_id_map_.Contains(unit_ptr->write_session_id)) {
                    KVCM_LOG_DEBUG("write_session_id [%s] has been consumed", unit_ptr->write_session_id.c_str());
                } else if (int64_t current_point = TimestampUtil::GetSteadyTimeSec();
                           current_point >= unit_ptr->expire_point) {
                    unit_ptr_to_expire = unit_ptr;
                } else {
                    int64_t next_sleep_time = unit_ptr->expire_point - current_point;
                    KVCM_LOG_DEBUG("Not expiring session %s, next_sleep_time [%lds]; requeue it",
                                   unit_ptr->write_session_id.c_str(),
                                   next_sleep_time);
                    StoreMinNextSleepTime(next_sleep_time);
                    expire_queue_.Push(unit_ptr);
                }
            }
            if (expire_queue_.Empty()) {
                next_sleep_time_.store(kDefaultExpireLoopSleepTime, std::memory_order_relaxed);
            }
        }

        if (unit_ptr_to_expire) {
            KVCM_LOG_DEBUG("Expiring session %s", unit_ptr_to_expire->write_session_id.c_str());
            // callback will remove the expired write_session_id
            unit_ptr_to_expire->callback();
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

    session_id_map_[write_session_id] = {std::move(keys), std::move(location_ids)};
    ExpireUnitPtr unit_ptr = std::make_shared<ExpireUnit>();
    unit_ptr->write_session_id = write_session_id;
    unit_ptr->expire_point = TimestampUtil::GetSteadyTimeSec() + write_timeout_seconds;
    unit_ptr->callback = std::move(callback);
    expire_queue_.Push(unit_ptr);
    StoreMinNextSleepTime(write_timeout_seconds);
    cond_.notify_one();
}

bool WriteLocationManager::GetAndDelete(const std::string &write_session_id, WriteLocationInfo &location_info) {
    if (session_id_map_.Contains(write_session_id)) {
        location_info = session_id_map_.Find(write_session_id)->second;
        KVCM_LOG_DEBUG("Retrieved and deleted session %s with %zu keys and %zu location_ids",
                       write_session_id.c_str(),
                       location_info.keys.size(),
                       location_info.location_ids.size());

        SessionIdMap::SizeType erase_count = session_id_map_.Erase(write_session_id);
        return (erase_count > 0);
    }
    return false;
}

} // namespace kv_cache_manager