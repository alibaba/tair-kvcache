#include "kv_cache_manager/service/service_impl_base.h"

#include "kv_cache_manager/common/logger.h"

namespace kv_cache_manager {

void ServiceImplBase::DisableLeaderOnlyRequests() {
    is_accepting_leader_only_requests_.store(false, std::memory_order_release);
    KVCM_LOG_INFO("Service stopped accepting new leader-only requests");
}

void ServiceImplBase::EnableLeaderOnlyRequests() {
    is_accepting_leader_only_requests_.store(true, std::memory_order_release);
    KVCM_LOG_INFO("Service started accepting new leader-only requests");
}

void ServiceImplBase::WaitForAllLeaderOnlyRequestsToComplete() {
    std::unique_lock<std::shared_mutex> lock(state_mutex_);
    leader_only_request_done_cv_.wait(
        lock, [this]() { return active_leader_only_request_count_.load(std::memory_order_acquire) == 0; });
    KVCM_LOG_INFO("Service all leader-only requests completed");
}

bool ServiceImplBase::CheckAndIncrementRequestCount(bool is_leader_only) {
    std::shared_lock<std::shared_mutex> lock(state_mutex_);
    if (is_leader_only) {
        if (!is_accepting_leader_only_requests_.load(std::memory_order_acquire)) {
            return false;
        }
        active_leader_only_request_count_.fetch_add(1, std::memory_order_relaxed);
    }
    active_request_count_.fetch_add(1, std::memory_order_relaxed);
    return true;
}

void ServiceImplBase::DecrementRequestCount(bool is_leader_only) {
    active_request_count_.fetch_sub(1, std::memory_order_relaxed);
    if (is_leader_only) {
        int prev_count = active_leader_only_request_count_.fetch_sub(1, std::memory_order_relaxed);
        if (prev_count == 1) {
            leader_only_request_done_cv_.notify_all();
        }
    }
}

} // namespace kv_cache_manager