#include "kv_cache_manager/config/leader_elector.h"

#include <algorithm>

#include "kv_cache_manager/common/error_code.h"
#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/loop_thread.h"
#include "kv_cache_manager/common/timestamp_util.h"
#include "kv_cache_manager/config/coordination_backend.h"
#include "kv_cache_manager/config/node_endpoint_info.h"

namespace kv_cache_manager {

LeaderElector::LeaderElector(const std::shared_ptr<CoordinationBackend> &coordination_backend,
                             const std::string &lock_key,
                             const std::string &lock_value,
                             int64_t lease_ms,
                             int64_t loop_interval_ms)
    : coordination_backend_(coordination_backend)
    , lock_key_(lock_key)
    , lock_value_(lock_value)
    , lease_timeout_us_(lease_ms * 1000)
    , loop_interval_us_(loop_interval_ms * 1000) {
    if (lease_timeout_us_ <= loop_interval_us_ * 10) {
        KVCM_LOG_WARN("It is recommended that loop_interval_us_ < lease_timeout_us_ / 10");
    }
}

LeaderElector::~LeaderElector() { Stop(); }

bool LeaderElector::Start() {
    KVCM_LOG_INFO("start LeaderElector lock_key=[%s], lease_ms=[%ld], loop_interval_ms=[%ld]",
                  lock_key_.c_str(),
                  lease_timeout_us_ / 1000,
                  loop_interval_us_ / 1000);

    if (!become_leader_handler_ || !no_longer_leader_handler_) {
        KVCM_LOG_ERROR("you should set become_leader_handler and no_longer_leader_handler.");
        return false;
    }

    stop_flag_ = false;

    // 启动状态转换线程
    state_transition_thread_ = std::make_unique<std::thread>(&LeaderElector::StateTransitionWorker, this);

    // 启动选主和续约线程
    leader_lock_thread_ptr_ =
        LoopThread::CreateLoopThread([this] { return WorkLoop(); }, loop_interval_us_, "LeaderLock");

    // 启动租约检查线程
    lease_check_thread_ptr_ =
        LoopThread::CreateLoopThread([this] { CheckLeaseTimeout(); }, loop_interval_us_, "LeaseCheck");

    if (!leader_lock_thread_ptr_ || !lease_check_thread_ptr_) {
        KVCM_LOG_ERROR("create leader check thread fail");
        return false;
    }

    KVCM_LOG_INFO("LeaderElector started successfully");
    return true;
}

void LeaderElector::Stop() {
    if (!leader_lock_thread_ptr_ && !lease_check_thread_ptr_ && !state_transition_thread_) {
        return;
    }

    KVCM_LOG_INFO("stopping LeaderElector...");

    // Revoke readiness before waiting for an outstanding coordination request.
    {
        std::lock_guard guard(transition_mutex_);
        stop_flag_ = true;
        LoseLeaseLocked(TimestampUtil::GetCurrentTimeUs());
    }

    // 停止选主线程
    if (leader_lock_thread_ptr_) {
        leader_lock_thread_ptr_->Stop();
        leader_lock_thread_ptr_.reset();
    }

    if (lease_check_thread_ptr_) {
        lease_check_thread_ptr_->Stop();
        lease_check_thread_ptr_.reset();
    }

    // 释放锁
    ReleaseLock();

    // 停止状态转换线程
    // The worker exits only after any started recovery and its cleanup return.
    transition_cv_.notify_all();
    if (state_transition_thread_ && state_transition_thread_->joinable()) {
        state_transition_thread_->join();
    }
    state_transition_thread_.reset();

    KVCM_LOG_INFO("LeaderElector stopped");
}

// ==================== 选主逻辑 ====================

bool LeaderElector::WorkLoop() {
    int64_t current_time = TimestampUtil::GetCurrentTimeUs();
    DoWorkLoop(current_time);
    return lock_acquired_.load();
}

void LeaderElector::DoWorkLoop(int64_t current_time) {
    int64_t begin_time = TimestampUtil::GetCurrentTimeUs();
    int64_t last_loop_time = last_loop_time_.load();
    if (begin_time - last_loop_time < 0 || (begin_time - last_loop_time > loop_interval_us_ * 2)) {
        KVCM_LOG_WARN("LeaderElector workloop begin timeout, current[%ld] last[%ld]", begin_time, last_loop_time);
    }
    last_loop_time_ = begin_time;

    if (!lock_acquired_.load()) {
        CampaignLeader(current_time);
    }

    if (lock_acquired_.load()) {
        HoldLeader(current_time);
    }

    int64_t end_time = TimestampUtil::GetCurrentTimeUs();
    if (end_time - begin_time > lease_timeout_us_ / 2) {
        KVCM_LOG_WARN("workloop run timeout, begin[%ld], end[%ld]", begin_time, end_time);
    }
}

void LeaderElector::CampaignLeader(int64_t current_time) {
    uint64_t request_epoch;
    bool campaign_allowed;
    {
        std::lock_guard guard(transition_mutex_);
        if (stop_flag_ || lock_acquired_.load()) {
            return;
        }
        request_epoch = lock_epoch_;
        campaign_allowed = CanCampaignLocked(current_time);
    }

    // Keep discovery current while recovery, cleanup, release or cooldown
    // prevents this node from campaigning.
    std::string current_value;
    int64_t expire_time_ms = -1;
    ErrorCode ec = coordination_backend_->GetLockHolder(lock_key_, current_value, expire_time_ms);
    if (ec != EC_OK && ec != EC_NOENT) {
        KVCM_LOG_WARN("check lock status failed, error_code=%d", static_cast<int>(ec));
        return;
    }

    if (ec == EC_NOENT) {
        current_value.clear();
    }

    {
        std::lock_guard transition_guard(transition_mutex_);
        if (request_epoch != lock_epoch_ || stop_flag_ || lock_acquired_.load()) {
            return;
        }
        {
            std::unique_lock guard(current_lock_holder_mutex_);
            current_lock_holder_ = current_value;
        }
        // A discovery-only read cannot become an acquisition if cleanup or
        // release completed while the query was in flight.
        if (!campaign_allowed || !CanCampaignLocked(current_time)) {
            return;
        }
    }

    if (!current_value.empty()) {
        if (current_value != lock_value_) {
            KVCM_LOG_DEBUG("lock held by other instance[%s], expiration[%ld]", current_value.c_str(), expire_time_ms);
            return;
        } else {
            KVCM_LOG_INFO("lock already held by self, update status");
            AcceptLease(current_time, expire_time_ms * 1000, request_epoch);
            return;
        }
    }

    // 尝试获取锁
    ec = coordination_backend_->TryLock(lock_key_, lock_value_, lease_timeout_us_ / 1000);
    if (ec == EC_OK) {
        if (AcceptLease(current_time, current_time + lease_timeout_us_, request_epoch)) {
            KVCM_LOG_INFO("campaign leader success!");
        }
    } else if (ec == EC_EXIST) {
        KVCM_LOG_DEBUG("lock already held by other instance");
    } else {
        KVCM_LOG_WARN("try lock failed, error_code=%d", static_cast<int>(ec));
    }
}

void LeaderElector::HoldLeader(int64_t current_time) {
    uint64_t request_epoch;
    {
        std::lock_guard guard(transition_mutex_);
        if (!CheckLeaseLocked(current_time)) {
            return;
        }
        request_epoch = lock_epoch_;
    }

    ErrorCode ec = coordination_backend_->RenewLock(lock_key_, lock_value_, lease_timeout_us_ / 1000);
    std::unique_lock<std::mutex> lock(transition_mutex_);
    int64_t now = std::max(current_time, TimestampUtil::GetCurrentTimeUs());
    if (request_epoch != lock_epoch_ || !CheckLeaseLocked(now)) {
        // A successful late renewal must not resurrect a revoked lease.
        lock.unlock();
        if (ec == EC_OK) {
            ReleaseLock();
        }
        return;
    }

    if (ec == EC_OK) {
        lease_expiration_time_us_.store(current_time + lease_timeout_us_);
        if (!CheckLeaseLocked(now)) {
            lock.unlock();
            ReleaseLock();
        }
        return;
    }

    if (ec == EC_NOENT || ec == EC_MISMATCH) {
        KVCM_LOG_WARN("renew lock failed, lock not held by self, error_code=%d", static_cast<int>(ec));
        LoseLeaseLocked(now);
        return;
    }

    KVCM_LOG_WARN("renew lock failed, unknown error_code=%d", static_cast<int>(ec));
}

void LeaderElector::CheckLeaseTimeout() {
    int64_t current_time = TimestampUtil::GetCurrentTimeUs();
    DoCheckLeaseTimeout(current_time);
}

bool LeaderElector::DoCheckLeaseTimeout(int64_t current_time) {
    std::lock_guard guard(transition_mutex_);
    return CheckLeaseLocked(current_time);
}

bool LeaderElector::CheckLeaseLocked(int64_t current_time) {
    if (!lock_acquired_.load()) {
        return false;
    }

    int64_t lease_expiration_time_us = GetLeaseExpirationTime();
    if (current_time > lease_expiration_time_us - loop_interval_us_) {
        KVCM_LOG_WARN("lease timeout, current[%ld] expiration[%ld]", current_time, lease_expiration_time_us);
        LoseLeaseLocked(current_time);
        return false;
    }
    return true;
}

void LeaderElector::Demote() {
    int64_t current_time = TimestampUtil::GetCurrentTimeUs();
    DoDemote(current_time);
}

void LeaderElector::DoDemote(int64_t current_time) {
    {
        std::lock_guard guard(transition_mutex_);
        if (release_in_progress_) {
            return;
        }
        release_in_progress_ = true;
        // Invalidate even an acquisition which has not returned yet.
        LoseLeaseLocked(current_time);
    }

    // 释放锁
    ReleaseLock();

    {
        std::lock_guard guard(transition_mutex_);
        release_in_progress_ = false;
        transition_cv_.notify_all();
    }
}

void LeaderElector::ReleaseLock() {
    ErrorCode ec = coordination_backend_->Unlock(lock_key_, lock_value_);
    if (ec == EC_OK) {
        KVCM_LOG_INFO("release lock success");
    } else if (ec != EC_NOENT && ec != EC_MISMATCH) {
        KVCM_LOG_WARN("release lock failed, error_code=%d", static_cast<int>(ec));
    }

    // Serialize release completion with discovery publication. A query which
    // read the old self-held key must not publish it after this cleanup.
    std::lock_guard transition_guard(transition_mutex_);
    if (!lock_acquired_.load()) {
        ++lock_epoch_;
        std::unique_lock holder_guard(current_lock_holder_mutex_);
        if (current_lock_holder_ == lock_value_) {
            current_lock_holder_.clear();
        }
    }
}

// ==================== 状态机逻辑 ====================

bool LeaderElector::CanCampaignLocked(int64_t current_time) const {
    // A started recovery must be cleaned up before this node can campaign again.
    return !stop_flag_ && !release_in_progress_ && !lock_acquired_.load() &&
           role_state_.load() == RoleState::FOLLOWER && current_time >= next_can_campaign_time_us_.load();
}

bool LeaderElector::AcceptLease(int64_t current_time, int64_t expiration, uint64_t request_epoch) {
    {
        std::lock_guard guard(transition_mutex_);
        int64_t now = std::max(current_time, TimestampUtil::GetCurrentTimeUs());
        if (request_epoch == lock_epoch_ && CanCampaignLocked(now) && now <= expiration - loop_interval_us_) {
            ++lock_epoch_;
            lock_acquired_.store(true);
            lease_expiration_time_us_.store(expiration);
            {
                std::unique_lock holder_guard(current_lock_holder_mutex_);
                current_lock_holder_ = lock_value_;
            }
            transition_cv_.notify_all();
            return true;
        }
        KVCM_LOG_INFO("discard acquisition result, epoch=%lu, current_epoch=%lu", request_epoch, lock_epoch_);
    }
    // This lock-loop thread cannot start another acquisition before releasing
    // the rejected result; Unlock checks the holder before deleting the key.
    ReleaseLock();
    return false;
}

void LeaderElector::LoseLeaseLocked(int64_t current_time) {
    ++lock_epoch_;
    bool was_acquired = lock_acquired_.exchange(false);
    lease_expiration_time_us_.store(-1);
    if (was_acquired) {
        int64_t forbid_time = forbid_campaign_time_ms_.load();
        if (forbid_time > 0) {
            int64_t next_campaign_time = current_time + forbid_time * 1000;
            next_can_campaign_time_us_.store(next_campaign_time);
            KVCM_LOG_INFO("set next campaign time to %ld (delay %ld ms)", next_campaign_time, forbid_time);
        }
        {
            std::lock_guard holder_guard(current_lock_holder_mutex_);
            if (current_lock_holder_ == lock_value_) {
                current_lock_holder_.clear();
            }
        }
        // Only the request gate runs here. Draining requests and cleanup remain
        // on the worker, so lease checks never wait for a slow lifecycle callback.
        InvokeHandler(leadership_lost_handler_);
        KVCM_LOG_INFO("leadership revoked, role=%s, epoch=%lu", RoleStateToString(role_state_.load()), lock_epoch_);
    }
    transition_cv_.notify_all();
}

bool LeaderElector::RunIfCurrentPromotionValid(const HandlerFuncType &handler) {
    std::lock_guard guard(transition_mutex_);
    if (role_state_.load() != RoleState::PROMOTING || promotion_epoch_ != lock_epoch_ || stop_flag_ ||
        !CheckLeaseLocked(TimestampUtil::GetCurrentTimeUs())) {
        return false;
    }
    handler();
    return true;
}

bool LeaderElector::HasPendingTransitionLocked() const {
    return (role_state_.load() == RoleState::FOLLOWER && lock_acquired_.load()) ||
           (role_state_.load() == RoleState::LEADER && !lock_acquired_.load());
}

void LeaderElector::StateTransitionWorker() {
    KVCM_LOG_INFO("state transition worker thread started");

    while (true) {
        {
            std::unique_lock<std::mutex> lock(transition_mutex_);
            transition_cv_.wait(lock, [this] { return stop_flag_ || HasPendingTransitionLocked(); });

            if (stop_flag_ && !HasPendingTransitionLocked()) {
                break;
            }
        }

        ReconcileState();
    }

    KVCM_LOG_INFO("state transition worker thread exited");
}

void LeaderElector::ProcessStateTransitionsForTest() {
    // 收敛至当前目标状态，仅用于未启动后台线程的单元测试。
    while (ReconcileState()) {}
}

void LeaderElector::InvokeHandler(const HandlerFuncType &handler) {
    // Preserve the existing callback failure behavior: log exceptions and
    // continue the lifecycle; do not introduce retries or propagate errors.
    try {
        if (handler) {
            handler();
        }
    } catch (const std::exception &e) {
        KVCM_LOG_ERROR("exception during state transition: %s", e.what());
    } catch (...) {
        KVCM_LOG_ERROR("unknown exception during state transition");
    }
}

bool LeaderElector::ReconcileState() {
    uint64_t epoch = 0;
    bool recover = false;
    {
        std::lock_guard guard(transition_mutex_);
        if (!HasPendingTransitionLocked()) {
            return false;
        }
        recover = role_state_.load() == RoleState::FOLLOWER;
        if (recover) {
            if (stop_flag_ || !CheckLeaseLocked(TimestampUtil::GetCurrentTimeUs())) {
                return false;
            }
            promotion_epoch_ = epoch = lock_epoch_;
            TransitionToState(RoleState::PROMOTING);
        } else {
            TransitionToState(RoleState::DEMOTING);
        }
    }
    if (recover) {
        KVCM_LOG_INFO("begin recover, epoch=%lu", epoch);
        InvokeHandler(become_leader_handler_);
        std::lock_guard guard(transition_mutex_);
        if (epoch == lock_epoch_ && !stop_flag_ && CheckLeaseLocked(TimestampUtil::GetCurrentTimeUs())) {
            TransitionToState(RoleState::LEADER);
            KVCM_LOG_INFO("recovery callback completed, role=LEADER, epoch=%lu", epoch);
            transition_cv_.notify_all();
            return true;
        }
        // Recovery has started and may have side effects; cleanup is mandatory
        // even if the lock was lost before readiness could be published.
        TransitionToState(RoleState::DEMOTING);
        KVCM_LOG_INFO("discard recovery result, epoch=%lu, current_epoch=%lu", epoch, lock_epoch_);
    }
    KVCM_LOG_INFO("begin cleanup");
    InvokeHandler(no_longer_leader_handler_);
    {
        std::lock_guard guard(transition_mutex_);
        TransitionToState(RoleState::FOLLOWER);
        KVCM_LOG_INFO("cleanup callback completed, role=FOLLOWER, epoch=%lu", lock_epoch_);
        transition_cv_.notify_all();
    }
    return true;
}

bool LeaderElector::TransitionToState(RoleState target_state) {
    RoleState current = role_state_.load();

    // 验证状态转换的合法性
    bool valid_transition = false;
    switch (current) {
    case RoleState::FOLLOWER:
        valid_transition = (target_state == RoleState::PROMOTING);
        break;
    case RoleState::PROMOTING:
        valid_transition = (target_state == RoleState::LEADER || target_state == RoleState::FOLLOWER ||
                            target_state == RoleState::DEMOTING);
        break;
    case RoleState::LEADER:
        valid_transition = (target_state == RoleState::DEMOTING);
        break;
    case RoleState::DEMOTING:
        valid_transition = (target_state == RoleState::FOLLOWER);
        break;
    }

    if (!valid_transition) {
        KVCM_LOG_ERROR(
            "invalid state transition: %s -> %s", RoleStateToString(current), RoleStateToString(target_state));
        return false;
    }

    role_state_.store(target_state);
    KVCM_LOG_INFO("state transitioned: %s -> %s", RoleStateToString(current), RoleStateToString(target_state));
    return true;
}

// ==================== 查询接口 ====================

bool LeaderElector::IsLeader() const { return lock_acquired_.load(); }

RoleState LeaderElector::GetRoleState() const { return role_state_.load(); }

const char *LeaderElector::RoleStateToString(const RoleState state) {
    switch (state) {
    case RoleState::FOLLOWER:
        return "FOLLOWER";
    case RoleState::PROMOTING:
        return "PROMOTING";
    case RoleState::LEADER:
        return "LEADER";
    case RoleState::DEMOTING:
        return "DEMOTING";
    default:
        return "UNKNOWN";
    }
}

bool LeaderElector::IsStableState() const {
    std::lock_guard guard(transition_mutex_);
    return IsStableStateLocked();
}

bool LeaderElector::IsStableStateLocked() const {
    RoleState state = role_state_.load();
    return (state == RoleState::LEADER || state == RoleState::FOLLOWER) && !HasPendingTransitionLocked() &&
           !release_in_progress_;
}

bool LeaderElector::WaitForStableState(int64_t timeout_ms) {
    if (IsStableState()) {
        return true;
    }

    std::unique_lock<std::mutex> lock(transition_mutex_);

    if (timeout_ms < 0) {
        transition_cv_.wait(lock, [this] { return IsStableStateLocked(); });
        return true;
    } else {
        return transition_cv_.wait_for(
            lock, std::chrono::milliseconds(timeout_ms), [this] { return IsStableStateLocked(); });
    }
}

int64_t LeaderElector::GetLeaseExpirationTime() const { return lease_expiration_time_us_.load(); }

// ==================== 配置接口 ====================

void LeaderElector::SetNoLongerLeaderHandler(const HandlerFuncType &handler) { no_longer_leader_handler_ = handler; }

void LeaderElector::SetBecomeLeaderHandler(const HandlerFuncType &handler) { become_leader_handler_ = handler; }

void LeaderElector::SetLeadershipLostHandler(const HandlerFuncType &handler) { leadership_lost_handler_ = handler; }

void LeaderElector::SetForbidCampaignLeaderTimeMs(int64_t forbid_time) { forbid_campaign_time_ms_ = forbid_time; }

int64_t LeaderElector::GetForbidCampaignLeaderTimeMs() const { return forbid_campaign_time_ms_; }

int64_t LeaderElector::GetLastLoopTimeUs() const { return last_loop_time_; }

std::string LeaderElector::GetLeaderNodeID() const {
    std::unique_lock guard(current_lock_holder_mutex_);
    return current_lock_holder_;
}

std::string LeaderElector::GetSelfNodeID() const { return lock_value_; }

ErrorCode LeaderElector::SetNodeInfo(const std::string &node_id, const NodeEndpointInfo &node_info) {
    std::string key = "_TAIR_KVCM_NODE_INFO_" + node_id;
    return coordination_backend_->SetValue(key, node_info.ToJsonString());
}

ErrorCode LeaderElector::SetSelfNodeInfo(const NodeEndpointInfo &node_info) {
    ErrorCode ec = SetNodeInfo(GetSelfNodeID(), node_info);
    if (ec == EC_OK) {
        std::lock_guard<std::mutex> guard(self_node_info_mutex_);
        self_node_info_cache_ = std::make_unique<NodeEndpointInfo>(node_info);
    }
    return ec;
}

ErrorCode LeaderElector::GetSelfNodeInfo(NodeEndpointInfo &out_node_info) const {
    std::lock_guard<std::mutex> guard(self_node_info_mutex_);
    if (!self_node_info_cache_) {
        return EC_NOENT;
    }
    out_node_info = *self_node_info_cache_;
    return EC_OK;
}

ErrorCode LeaderElector::GetNodeInfo(const std::string &node_id, NodeEndpointInfo &out_node_info) {
    std::string key = "_TAIR_KVCM_NODE_INFO_" + node_id;
    std::string json_str;
    ErrorCode ec = coordination_backend_->GetValue(key, json_str);
    if (ec != EC_OK) {
        return ec;
    }
    if (!out_node_info.FromJsonString(json_str)) {
        KVCM_LOG_ERROR("Failed to parse node info JSON for node_id[%s]", node_id.c_str());
        return EC_ERROR;
    }
    return EC_OK;
}

ErrorCode LeaderElector::GetLeaderNodeInfo(NodeEndpointInfo &out_node_info) {
    std::string leader_node_id = GetLeaderNodeID();
    if (leader_node_id.empty()) {
        return EC_NOENT;
    }
    // 自身是 leader 时直接使用本地缓存，避免 coordination backend 的远程调用
    if (leader_node_id == GetSelfNodeID()) {
        return GetSelfNodeInfo(out_node_info);
    }
    return GetNodeInfo(leader_node_id, out_node_info);
}

} // namespace kv_cache_manager
