#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <deque>
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
    using CallBack = std::function<void(std::unique_ptr<WriteLocationInfo>)>;
    void Start();
    void Stop();
    void DoCleanup();
    void Put(const std::string &write_session_id,
             std::vector<int64_t> &&keys,
             std::vector<std::string> &&location_ids,
             int64_t write_timeout_seconds,
             CallBack callback);
    bool GetAndDelete(const std::string &write_session_id, WriteLocationInfo &location_info);
    // FinishWriteCache 已应用结果的登记与查询。会话一旦删除（GetAndDelete 或
    // 超时/清理回调），重试的 finish 只能凭这份记录裁决“已应用”（幂等成功）
    // 还是“已被回滚/从未存在”（维持失败）：空的 keys 位图与掩码按位比对，
    // 见 CacheManager::FinishWriteCache 的幂等重试分支。
    void PutFinishedMask(const std::string &write_session_id, std::vector<bool> applied_mask);
    bool GetFinishedMask(const std::string &write_session_id, std::vector<bool> &applied_mask) const;
    bool HasLocationId(const std::string &location_id) const;
    size_t ExpireSize() const { return session_id_map_.Size(); }

private:
    void ExpireLoop();
    void StoreMinNextSleepTimeUs(int64_t next_sleep_time_us);
    struct ExpireUnit {
        int64_t expire_point;
        std::string write_session_id;
        CallBack callback;
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
        bool HasLocationId(const std::string &location_id) const;

    private:
        void AddToLocationIndexUnsafe(const std::vector<std::string> &location_ids);
        void RemoveFromLocationIndexUnsafe(const std::vector<std::string> &location_ids);

        mutable std::mutex mux_;
        std::unordered_map<std::string, int64_t> session_id_map_impl_;
        std::map<int64_t, ExpireUnitPtr> unit_map_;

        // inverted index to enable O(1) HasLocationId lookups:
        // location_id -> refcount (number of sessions referencing it)
        std::unordered_map<std::string, std::size_t> location_id_index_;
    };

    SessionIdMap session_id_map_;
    std::thread expire_thread_;
    std::atomic_bool stop_ = false;
    std::atomic_int64_t next_sleep_time_us_;
    std::mutex stop_mutex_;
    std::condition_variable stop_cond_;

    // 已完成会话的应用结果（PutFinishedMask）：服务线程的 finish 与超时回调
    // 线程的回滚都会写，读走服务线程的幂等重试，各自加 finished_mux_。
    mutable std::mutex finished_mux_;
    std::map<std::string, std::vector<bool>> finished_masks_;
    std::deque<std::string> finished_order_;
};

} // namespace kv_cache_manager