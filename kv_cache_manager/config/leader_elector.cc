#include "kv_cache_manager/config/leader_elector.h"

#include "kv_cache_manager/common/error_code.h"
#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/loop_thread.h"
#include "kv_cache_manager/common/timestamp_util.h"
#include "kv_cache_manager/config/distributed_lock_backend.h"

namespace kv_cache_manager {

LeaderElector::LeaderElector(const std::shared_ptr<DistributedLockBackend> &lock_backend,
                             const std::string &lock_key,
                             const std::string &lock_value,
                             int64_t lease_ms,
                             int64_t loop_interval_ms)
    : lock_backend_(lock_backend)
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

    stop_flag_ = true;

    // 停止选主线程
    if (leader_lock_thread_ptr_) {
        leader_lock_thread_ptr_->Stop();
        leader_lock_thread_ptr_.reset();
    }

    if (lease_check_thread_ptr_) {
        lease_check_thread_ptr_->Stop();
        lease_check_thread_ptr_.reset();
    }

    // 主动降级
    Demote();

    // 等待状态转换完成
    WaitForStableState(5000);

    // 停止状态转换线程
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
    if (begin_time - last_loop_time_ < 0 || (begin_time - last_loop_time_ > loop_interval_us_ * 2)) {
        KVCM_LOG_WARN("LeaderElector workloop begin timeout, current[%ld] last[%ld]", begin_time, last_loop_time_);
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
    if (current_time < next_can_campaign_time_us_.load()) {
        return;
    }

    // 检查当前锁是否被其他实例持有
    std::string current_value;
    int64_t expire_time_ms;
    ErrorCode ec = lock_backend_->GetLockHolder(lock_key_, current_value, expire_time_ms);
    if (ec != EC_OK && ec != EC_NOENT) {
        KVCM_LOG_WARN("check lock status failed, error_code=%d", static_cast<int>(ec));
        return;
    }

    {
        std::unique_lock guard(current_lock_holder_mutex_);
        current_lock_holder_ = current_value;
    }

    if (!current_value.empty()) {
        if (current_value != lock_value_) {
            KVCM_LOG_DEBUG("lock held by other instance[%s], expiration[%ld]", current_value.c_str(), expire_time_ms);
            return;
        } else {
            KVCM_LOG_INFO("lock already held by self, update status");
            UpdateLockStatus(current_time, true, expire_time_ms * 1000);
            return;
        }
    }

    // 尝试获取锁
    ec = lock_backend_->TryLock(lock_key_, lock_value_, lease_timeout_us_ / 1000);
    if (ec == EC_OK) {
        KVCM_LOG_INFO("campaign leader success!");
        UpdateLockStatus(current_time, true, current_time + lease_timeout_us_);
    } else if (ec == EC_EXIST) {
        KVCM_LOG_DEBUG("lock already held by other instance");
    } else {
        KVCM_LOG_WARN("try lock failed, error_code=%d", static_cast<int>(ec));
    }
}

void LeaderElector::HoldLeader(int64_t current_time) {
    if (!lock_acquired_.load()) {
        return;
    }

    ErrorCode ec = lock_backend_->RenewLock(lock_key_, lock_value_, lease_timeout_us_ / 1000);
    if (ec == EC_OK) {
        UpdateLockStatus(current_time, true, current_time + lease_timeout_us_);
        return;
    }

    if (ec == EC_NOENT || ec == EC_MISMATCH) {
        KVCM_LOG_WARN("renew lock failed, lock not held by self, error_code=%d", static_cast<int>(ec));
        UpdateLockStatus(current_time, false, -1);
        return;
    }

    KVCM_LOG_WARN("renew lock failed, unknown error_code=%d", static_cast<int>(ec));
    if (current_time > GetLeaseExpirationTime()) {
        UpdateLockStatus(current_time, false, -1);
    }
}

void LeaderElector::CheckLeaseTimeout() {
    int64_t current_time = TimestampUtil::GetCurrentTimeUs();
    DoCheckLeaseTimeout(current_time);
}
bool LeaderElector::DoCheckLeaseTimeout(int64_t current_time) {
    if (!lock_acquired_.load()) {
        return false;
    }

    int64_t lease_expiration_time_us = GetLeaseExpirationTime();
    if (current_time > lease_expiration_time_us - loop_interval_us_) {
        KVCM_LOG_WARN("lease timeout, current[%ld] expiration[%ld]", current_time, lease_expiration_time_us);
        UpdateLockStatus(current_time, false, -1);
        return false;
    }
    return true;
}

void LeaderElector::Demote() {
    int64_t current_time = TimestampUtil::GetCurrentTimeUs();
    DoDemote(current_time);
}

void LeaderElector::DoDemote(int64_t current_time) {
    if (!lock_acquired_.load()) {
        KVCM_LOG_INFO("not holding lock, no need to demote");
        return;
    }

    // 先设置延迟选主时间，保证抢主能按预期delay
    if (forbid_campaign_time_ms_ > 0) {
        int64_t next_campaign_time = current_time + forbid_campaign_time_ms_ * 1000;
        next_can_campaign_time_us_.store(next_campaign_time);
        KVCM_LOG_INFO("set next campaign time to %ld (delay %ld ms)", next_campaign_time, forbid_campaign_time_ms_);
    }

    // 释放锁
    ErrorCode ec = lock_backend_->Unlock(lock_key_, lock_value_);
    if (ec == EC_OK) {
        KVCM_LOG_INFO("release lock success");
    } else {
        KVCM_LOG_WARN("release lock failed, error_code=%d", static_cast<int>(ec));
    }

    // 更新状态
    UpdateLockStatus(current_time, false, -1);
}

// ==================== 状态机逻辑 ====================

void LeaderElector::UpdateLockStatus(int64_t current_time, bool acquired, int64_t lease_expiration_time) {
    bool old_acquired = lock_acquired_.load();

    // 先设置延迟选主时间，保证抢主能按预期delay
    if (old_acquired && !acquired && forbid_campaign_time_ms_ > 0) {
        int64_t next_campaign_time = current_time + forbid_campaign_time_ms_ * 1000;
        next_can_campaign_time_us_.store(next_campaign_time);
        KVCM_LOG_INFO("set next campaign time to %ld (delay %ld ms)", next_campaign_time, forbid_campaign_time_ms_);
    }

    // 更新锁状态
    lock_acquired_.store(acquired);
    lease_expiration_time_us_.store(lease_expiration_time);

    // 锁状态变化时，触发角色状态转换
    if (old_acquired != acquired) {
        KVCM_LOG_INFO("lock status changed: acquired=%d, lease_expiration=%ld", acquired, lease_expiration_time);

        if (acquired) {
            {
                std::unique_lock guard(current_lock_holder_mutex_);
                if (current_lock_holder_ != lock_value_) {
                    current_lock_holder_ = lock_value_;
                }
            }
            RequestPromoteToLeader();
        } else {
            {
                std::unique_lock guard(current_lock_holder_mutex_);
                if (current_lock_holder_ == lock_value_) {
                    current_lock_holder_.clear();
                }
            }
            RequestDemoteToFollower();
        }
    }
}

void LeaderElector::RequestPromoteToLeader() {
    std::unique_lock<std::mutex> lock(transition_mutex_);

    RoleState current = role_state_.load();

    // 如果已经是Leader或正在晋升，忽略
    if (current == RoleState::LEADER || current == RoleState::PROMOTING) {
        KVCM_LOG_DEBUG("already in LEADER or PROMOTING state, ignore promote request");
        return;
    }

    // 如果正在降级，等待降级完成后会自动处理
    if (current == RoleState::DEMOTING) {
        KVCM_LOG_INFO("currently demoting, promote will be queued");
    }

    uint64_t new_version = state_version_.fetch_add(1) + 1;
    RoleTransitionTask task;
    task.version = new_version;
    task.target_state = RoleState::LEADER;
    task.action = [this]() {
        if (become_leader_handler_) {
            std::lock_guard guard(callback_mutex_);
            become_leader_handler_();
        }
    };

    transition_queue_.push(std::move(task));
    transition_cv_.notify_one();

    KVCM_LOG_INFO("promote request queued, version=%lu", new_version);
}

void LeaderElector::RequestDemoteToFollower() {
    std::unique_lock<std::mutex> lock(transition_mutex_);

    RoleState current = role_state_.load();

    // 如果已经是Follower或正在降级，忽略
    if (current == RoleState::FOLLOWER || current == RoleState::DEMOTING) {
        KVCM_LOG_DEBUG("already in FOLLOWER or DEMOTING state, ignore demote request");
        return;
    }

    // 清空队列中的所有晋升任务
    std::queue<RoleTransitionTask> empty_queue;
    std::swap(transition_queue_, empty_queue);

    uint64_t new_version = state_version_.fetch_add(1) + 1;
    RoleTransitionTask task;
    task.version = new_version;
    task.target_state = RoleState::FOLLOWER;
    task.action = [this]() {
        KVCM_LOG_DEBUG("do demote action");
        if (no_longer_leader_handler_) {
            std::lock_guard guard(callback_mutex_);
            KVCM_LOG_DEBUG("no_longer_leader_handler_");
            no_longer_leader_handler_();
        }
    };

    transition_queue_.push(std::move(task));
    transition_cv_.notify_one();

    KVCM_LOG_INFO("demote request queued, version=%lu", new_version);
}

void LeaderElector::StateTransitionWorker() {
    KVCM_LOG_INFO("state transition worker thread started");

    while (!stop_flag_) {
        RoleTransitionTask task;

        {
            std::unique_lock<std::mutex> lock(transition_mutex_);
            transition_cv_.wait(lock, [this] { return stop_flag_ || !transition_queue_.empty(); });

            if (stop_flag_) {
                break;
            }

            if (transition_queue_.empty()) {
                continue;
            }

            task = std::move(transition_queue_.front());
            transition_queue_.pop();
        }

        ExecuteTransitionTask(task);
    }

    KVCM_LOG_INFO("state transition worker thread exited");
}

void LeaderElector::ProcessStateTransitionsForTest() {
    // 处理所有待处理的状态转换任务，仅用于未启动后台线程的单元测试
    while (true) {
        RoleTransitionTask task;

        {
            std::unique_lock<std::mutex> lock(transition_mutex_);

            if (transition_queue_.empty()) {
                break;
            }

            task = std::move(transition_queue_.front());
            transition_queue_.pop();
        }

        // 执行状态转换
        ExecuteTransitionTask(task);
    }
}

void LeaderElector::ExecuteTransitionTask(const RoleTransitionTask &task) {
    is_transitioning_.store(true);

    RoleState current = role_state_.load();
    KVCM_LOG_INFO("begin state transition: %s -> %s, version=%lu",
                  RoleStateToString(current),
                  RoleStateToString(task.target_state),
                  task.version);

    int64_t start_time = TimestampUtil::GetCurrentTimeUs();

    // 第一步：转换到中间状态
    RoleState intermediate_state;
    if (task.target_state == RoleState::LEADER) {
        intermediate_state = RoleState::PROMOTING;
    } else if (task.target_state == RoleState::FOLLOWER) {
        intermediate_state = RoleState::DEMOTING;
    } else {
        KVCM_LOG_ERROR("invalid target state: %s", RoleStateToString(task.target_state));
        is_transitioning_.store(false);
        transition_cv_.notify_all();
        return;
    }

    if (!TransitionToState(intermediate_state)) {
        KVCM_LOG_ERROR("failed to transition to intermediate state %s", RoleStateToString(intermediate_state));
        is_transitioning_.store(false);
        transition_cv_.notify_all();
        return;
    }

    // 第二步：执行状态转换操作（耗时操作）
    // TODO: 添加异常情况下回滚和重试
    try {
        if (task.action) {
            task.action();
        }
    } catch (const std::exception &e) {
        KVCM_LOG_ERROR("exception during state transition: %s", e.what());
    } catch (...) { KVCM_LOG_ERROR("unknown exception during state transition"); }

    // 第三步：转换到目标状态
    if (!TransitionToState(task.target_state)) {
        KVCM_LOG_ERROR("failed to transition to target state %s", RoleStateToString(task.target_state));
    }

    int64_t end_time = TimestampUtil::GetCurrentTimeUs();
    KVCM_LOG_INFO("state transition completed: %s -> %s, version=%lu, cost=%ld us",
                  RoleStateToString(current),
                  RoleStateToString(task.target_state),
                  task.version,
                  end_time - start_time);

    is_transitioning_.store(false);
    transition_cv_.notify_all();
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
        valid_transition = (target_state == RoleState::LEADER || target_state == RoleState::FOLLOWER);
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
    RoleState state = role_state_.load();
    return (state == RoleState::LEADER || state == RoleState::FOLLOWER) && !is_transitioning_.load();
}

bool LeaderElector::WaitForStableState(int64_t timeout_ms) {
    if (IsStableState()) {
        return true;
    }

    std::unique_lock<std::mutex> lock(transition_mutex_);

    if (timeout_ms < 0) {
        transition_cv_.wait(lock, [this] { return IsStableState(); });
        return true;
    } else {
        return transition_cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), [this] { return IsStableState(); });
    }
}

int64_t LeaderElector::GetLeaseExpirationTime() const { return lease_expiration_time_us_.load(); }

// ==================== 配置接口 ====================

void LeaderElector::SetNoLongerLeaderHandler(const HandlerFuncType &handler) { no_longer_leader_handler_ = handler; }

void LeaderElector::SetBecomeLeaderHandler(const HandlerFuncType &handler) { become_leader_handler_ = handler; }

void LeaderElector::SetForbidCampaignLeaderTimeMs(int64_t forbid_time) { forbid_campaign_time_ms_ = forbid_time; }

int64_t LeaderElector::GetForbidCampaignLeaderTimeMs() const { return forbid_campaign_time_ms_; }

int64_t LeaderElector::GetLastLoopTimeUs() const { return last_loop_time_; }

void LeaderElector::SetLeaderInfo(const std::string &leader_info) {
    std::unique_lock<std::mutex> lock(leader_info_mutex_);
    leader_info_ = leader_info;
}

std::string LeaderElector::GetLeaderInfo() const {
    std::unique_lock<std::mutex> lock(leader_info_mutex_);
    return leader_info_;
}

std::string LeaderElector::GetLeaderNodeID() const {
    std::unique_lock guard(current_lock_holder_mutex_);
    return current_lock_holder_;
}

std::string LeaderElector::GetSelfNodeID() const { return lock_value_; }

} // namespace kv_cache_manager
