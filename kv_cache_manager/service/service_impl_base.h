#pragma once

#include <atomic>
#include <condition_variable>
#include <memory>
#include <shared_mutex>

namespace kv_cache_manager {

class ServiceImplBase {
public:
    ServiceImplBase() = default;
    virtual ~ServiceImplBase() = default;

    // 禁止拷贝
    ServiceImplBase(const ServiceImplBase &) = delete;
    ServiceImplBase &operator=(const ServiceImplBase &) = delete;

    // 优雅降级方法：停止接受新请求并等待当前请求完成
    void DisableLeaderOnlyRequests();
    void EnableLeaderOnlyRequests();
    void WaitForAllLeaderOnlyRequestsToComplete();

protected:
    bool CheckAndIncrementRequestCount(bool is_leader_only);
    void DecrementRequestCount(bool is_leader_only);

private:
    // 请求控制机制
    std::atomic<bool> is_accepting_leader_only_requests_{true};
    std::atomic<int> active_request_count_{0};
    std::atomic<int> active_leader_only_request_count_{0};
    std::condition_variable_any leader_only_request_done_cv_;
    std::shared_mutex state_mutex_;
};

} // namespace kv_cache_manager