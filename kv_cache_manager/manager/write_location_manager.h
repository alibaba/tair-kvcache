#pragma once

#include <atomic>
#include <autil/LockFreeQueue.h>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include "kv_cache_manager/common/concurrent_hash_map.h"
#include "kv_cache_manager/manager/cache_location_view.h"

namespace kv_cache_manager {

class WriteLocationManager {
public:
    WriteLocationManager();
    ~WriteLocationManager();
    struct WriteLocationInfo {
        std::vector<int64_t> keys;
        std::vector<std::string> location_ids;
    };
    void Start();
    void Stop();
    void DoCleanup();
    void Put(const std::string &write_session_id,
             std::vector<int64_t> &&keys,
             std::vector<std::string> &&location_ids,
             int64_t write_timeout_seconds,
             std::function<void()> callback);
    bool GetAndDelete(const std::string &write_session_id, WriteLocationInfo &location_info);
    size_t ExpireSize() const { return expire_queue_.Size(); }

private:
    void ExpireLoop();
    void StoreMinNextSleepTime(int64_t next_sleep_time);
    struct ExpireUnit {
        int64_t expire_point;
        std::string write_session_id;
        std::function<void()> callback; // call CacheManager::FinishWriteCache -> WriteLocationManager::GetAndDelete
    };

    using ExpireUnitPtr = std::shared_ptr<ExpireUnit>;
    using SessionIdMap = ConcurrentHashMap<std::string, WriteLocationInfo>;

    SessionIdMap session_id_map_;
    autil::LockFreeQueue<ExpireUnitPtr> expire_queue_;
    std::thread expire_thread_;
    std::atomic_bool stop_ = false;
    std::atomic_int64_t next_sleep_time_;
    std::mutex do_expire_mutex_; // mutex for DoCleanup and ExpireLoop
    std::condition_variable cond_;
};

} // namespace kv_cache_manager