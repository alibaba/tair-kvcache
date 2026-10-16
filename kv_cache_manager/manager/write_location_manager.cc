#include "kv_cache_manager/manager/write_location_manager.h"

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/timestamp_util.h"
namespace kv_cache_manager {

namespace {
constexpr int kDefaultExpireLoopSleepTimeUs = 5 * 1000 * 1000; // us
// 已完成会话应用结果的容量上限：只服务于丢应答后的幂等重试（秒级窗口），
// 无限增长才需要更大的表。超限丢最旧的一半并出声，被丢会话的重试退化为
// “会话不存在”的失败答复（客户端保守地当作未提交，保住字节）。
constexpr size_t kFinishedMaskLimit = 4096;
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

WriteLocationManager::WriteLocationManager() {
    next_sleep_time_us_.store(kDefaultExpireLoopSleepTimeUs, std::memory_order_relaxed);
    KVCM_LOG_DEBUG("WriteLocationManager constructed");
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
    StoreMinNextSleepTimeUs(write_timeout_seconds * 1000 * 1000);
}

bool WriteLocationManager::GetAndDelete(const std::string &write_session_id, WriteLocationInfo &location_info) {
    // 首个 finish 在删会话之前登记 in-progress：并发的幂等重试若观察到会话
    // 已删而结果未记（GetAndDelete 之后、PutFinishedMask 之前），凭标记等待
    // 首次应用落定，而不是被误判为“会话不存在”而耗尽重试。标记先于删除、
    // 删除在锁下原子生效，故该顺序对任何观察者成立。
    MarkFinishInProgress(write_session_id);
    if (!session_id_map_.GetAndDelete(write_session_id, location_info)) {
        UnmarkFinishInProgress(write_session_id);
        return false;
    }
    return true;
}

void WriteLocationManager::MarkFinishInProgress(const std::string &write_session_id) {
    std::lock_guard<std::mutex> lock(finished_mux_);
    ++finish_in_progress_[write_session_id];
}

void WriteLocationManager::UnmarkFinishInProgress(const std::string &write_session_id) {
    std::lock_guard<std::mutex> lock(finished_mux_);
    auto it = finish_in_progress_.find(write_session_id);
    if (it == finish_in_progress_.end()) {
        return;
    }
    if (--it->second == 0) {
        // 撤销后可能已无任何应用者：唤醒按“标记消失”等待的重试，让它们
        // 立刻重读结果而不是空等满超时。
        finish_in_progress_.erase(it);
        finish_cv_.notify_all();
    }
}

void WriteLocationManager::ResolveFinish(const std::string &write_session_id) {
    std::lock_guard<std::mutex> lock(finished_mux_);
    finish_in_progress_.erase(write_session_id);
    finish_cv_.notify_all();
}

bool WriteLocationManager::WaitFinishedMask(const std::string &write_session_id,
                                            std::chrono::milliseconds timeout,
                                            std::vector<bool> &applied_mask) {
    std::unique_lock<std::mutex> lock(finished_mux_);
    if (finish_in_progress_.find(write_session_id) != finish_in_progress_.end()) {
        finish_cv_.wait_for(
            lock, timeout, [&] { return finish_in_progress_.find(write_session_id) == finish_in_progress_.end(); });
    }
    auto it = finished_masks_.find(write_session_id);
    if (it == finished_masks_.end()) {
        return false;
    }
    applied_mask = it->second;
    return true;
}

void WriteLocationManager::PutFinishedMask(const std::string &write_session_id, std::vector<bool> applied_mask) {
    {
        std::lock_guard<std::mutex> lock(finished_mux_);
        if (finished_masks_.emplace(write_session_id, std::move(applied_mask)).second) {
            finished_order_.push_back(write_session_id);
            if (finished_order_.size() > kFinishedMaskLimit) {
                const size_t drop = finished_order_.size() - kFinishedMaskLimit / 2;
                for (size_t i = 0; i < drop; i++) {
                    finished_masks_.erase(finished_order_.front());
                    finished_order_.pop_front();
                }
                KVCM_LOG_WARN(
                    "Dropped %zu finished write_session records (limit %zu): their retries read \"not found\"",
                    drop,
                    kFinishedMaskLimit);
            }
        }
    }
    // 结果已登记：落定 in-progress，唤醒等待首请求结果的幂等重试。
    ResolveFinish(write_session_id);
}

bool WriteLocationManager::GetFinishedMask(const std::string &write_session_id, std::vector<bool> &applied_mask) const {
    std::lock_guard<std::mutex> lock(finished_mux_);
    auto it = finished_masks_.find(write_session_id);
    if (it == finished_masks_.end()) {
        return false;
    }
    applied_mask = it->second;
    return true;
}

bool WriteLocationManager::SessionIdMap::HasLocationId(const std::string &location_id) const {
    std::unique_lock lock(mux_);
    return location_id_index_.find(location_id) != location_id_index_.end();
}

bool WriteLocationManager::HasLocationId(const std::string &location_id) const {
    return session_id_map_.HasLocationId(location_id);
}

} // namespace kv_cache_manager