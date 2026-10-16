#pragma once

#include <atomic>
#include <chrono>
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
    // 首个 finish 正在应用中的登记与等待（见 finish_in_progress_）：GetAndDelete
    // 拿走会话到 PutFinishedMask 登记结果之间存在窗口，窗口内到达的幂等重试
    // 会读到“会话不存在且无记录”——三次重试都可能在首请求落定前耗尽，尽管
    // 首请求稍后提交成功。WaitFinishedMask 让这类重试等首请求落定再裁决；
    // 无 in-progress 时退化为一次 GetFinishedMask。
    bool WaitFinishedMask(const std::string &write_session_id,
                          std::chrono::milliseconds timeout,
                          std::vector<bool> &applied_mask);
    // 落定一次 in-progress（幂等）：PutFinishedMask 登记结果后、以及首个 finish
    // 未登记结果就退出的路径（掩码非法、instance 不存在）调用，唤醒等待者。
    void ResolveFinish(const std::string &write_session_id);
    bool HasLocationId(const std::string &location_id) const;
    size_t ExpireSize() const { return session_id_map_.Size(); }

private:
    void ExpireLoop();
    void StoreMinNextSleepTimeUs(int64_t next_sleep_time_us);
    // GetAndDelete 的 claim 记账：标记先于删除（同一调用者顺序执行、删除在
    // map 锁下原子可见），因此任何观察到会话已删的重试，必能观察到标记或
    // 已落定的结果。竞败方立即撤销自己的计数，存活的计数属于真正持有会话
    // 的应用者；ResolveFinish 清除标记并唤醒。
    void MarkFinishInProgress(const std::string &write_session_id);
    void UnmarkFinishInProgress(const std::string &write_session_id);
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
    // 正在应用中的 finish（引用计数，写方与读方同锁）：key 为 write_session_id。
    // 仅覆盖 GetAndDelete 拿走会话后的服务路径窗口；超时/清理回调的答案恒为
    // 失败（回滚），无需等待。
    std::map<std::string, size_t> finish_in_progress_;
    std::condition_variable finish_cv_;
};

} // namespace kv_cache_manager