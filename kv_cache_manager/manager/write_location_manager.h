#pragma once

#include <atomic>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>

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
    size_t ExpireSize() const { return session_id_map_.Size(); }

private:
    void ExpireLoop();
    void StoreMinNextSleepTime(int64_t next_sleep_time);
    struct ExpireUnit {
        int64_t expire_point;
        std::string write_session_id;
        std::function<void()> callback; // call CacheManager::FinishWriteCache -> WriteLocationManager::GetAndDelete
        WriteLocationInfo write_location_info;
    };
    using ExpireUnitPtr = std::shared_ptr<ExpireUnit>;

    class SessionIdMap {
    public:
        size_t Size() const;
        bool Empty() const;
        int64_t DropByExpirePoint(int64_t cur_point); // drop by cur_point, return next expire_point
        void DropAll();
        void Put(ExpireUnitPtr unit);
        bool GetAndDelete(const std::string &write_session_id, WriteLocationInfo &location_info);

    private:
        mutable std::mutex mux_;
        std::unordered_map<std::string, int64_t> session_id_map_impl_;
        std::map<int64_t, ExpireUnitPtr> unit_map_;
    };

    SessionIdMap session_id_map_;
    std::thread expire_thread_;
    std::atomic_bool stop_ = false;
    std::atomic_int64_t next_sleep_time_;
};

} // namespace kv_cache_manager