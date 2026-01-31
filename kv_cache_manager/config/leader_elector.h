#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>

namespace kv_cache_manager {

class LoopThread;
class DistributedLockBackend;

// 角色状态枚举
enum class RoleState {
    FOLLOWER,  // 稳定的备节点状态
    PROMOTING, // 正在晋升为主的过渡状态
    LEADER,    // 稳定的主节点状态
    DEMOTING,  // 正在降级为备的过渡状态
};

// 状态转换任务
struct RoleTransitionTask {
    uint64_t version;
    RoleState target_state;
    std::function<void()> action;
};

class LeaderElector {
public:
    using HandlerFuncType = std::function<void()>;

    LeaderElector(const std::shared_ptr<DistributedLockBackend> &lock_backend,
                  const std::string &lock_key,
                  const std::string &lock_value,
                  int64_t lease_ms = 60,
                  int64_t loop_interval_ms = 6);
    ~LeaderElector();
    LeaderElector(const LeaderElector &) = delete;
    LeaderElector &operator=(const LeaderElector &) = delete;

    bool Start();
    void Stop();

    // 角色状态查询
    bool IsLeader() const;
    RoleState GetRoleState() const;
    bool IsStableState() const;

    // 等待状态稳定（主要用于测试和优雅关闭）
    bool WaitForStableState(int64_t timeout_ms = -1);

    // 租约信息
    int64_t GetLeaseExpirationTime() const;

    // 主动降级
    void Demote();

    // 设置回调（必须在 Start 之前调用）
    void SetNoLongerLeaderHandler(const HandlerFuncType &handler);
    void SetBecomeLeaderHandler(const HandlerFuncType &handler);

    // Leader 信息
    void SetLeaderInfo(const std::string &leader_info);
    std::string GetLeaderInfo() const;
    std::string GetLeaderNodeID() const;
    std::string GetSelfNodeID() const;

    // 选主控制
    void SetForbidCampaignLeaderTimeMs(int64_t forbid_time);
    int64_t GetForbidCampaignLeaderTimeMs() const;
    int64_t GetLastLoopTimeUs() const;
    static const char *RoleStateToString(const RoleState state);

private:
    // 选主相关
    bool WorkLoop();
    void DoWorkLoop(int64_t currentTime);
    void CampaignLeader(int64_t current_time);
    void HoldLeader(int64_t current_time);
    void DoDemote(int64_t current_time);
    bool DoCheckLeaseTimeout(int64_t current_time);
    void CheckLeaseTimeout();

    // 状态机相关
    void StateTransitionWorker();
    void ExecuteTransitionTask(const RoleTransitionTask &task);
    bool TransitionToState(RoleState target_state);
    void RequestPromoteToLeader();
    void RequestDemoteToFollower();
    void ProcessStateTransitionsForTest();

    // 锁状态更新
    void UpdateLockStatus(int64_t current_time, bool acquired, int64_t lease_expiration_time);

private:
    // === 分布式锁相关 ===
    std::shared_ptr<DistributedLockBackend> lock_backend_;
    std::string lock_key_;
    std::string lock_value_;
    int64_t lease_timeout_us_;
    int64_t loop_interval_us_;

    // 锁持有状态（由选主线程维护）
    std::atomic<bool> lock_acquired_{false}; // 是否持有锁
    std::atomic<int64_t> lease_expiration_time_us_{-1};

    std::string current_lock_holder_;
    mutable std::mutex current_lock_holder_mutex_;

    // === 角色状态机相关 ===
    std::atomic<RoleState> role_state_{RoleState::FOLLOWER};
    std::atomic<uint64_t> state_version_{0};
    std::atomic<bool> is_transitioning_{false};

    // 状态转换任务队列
    std::queue<RoleTransitionTask> transition_queue_;
    mutable std::mutex transition_mutex_;
    std::condition_variable transition_cv_;

    // === 线程管理 ===
    std::shared_ptr<LoopThread> leader_lock_thread_ptr_;   // 选主和续约线程
    std::shared_ptr<LoopThread> lease_check_thread_ptr_;   // 租约检查线程
    std::unique_ptr<std::thread> state_transition_thread_; // 状态转换线程

    std::atomic<bool> stop_flag_{false};
    int64_t last_loop_time_ = -1;

    // === 回调函数 ===
    HandlerFuncType become_leader_handler_;
    HandlerFuncType no_longer_leader_handler_;
    mutable std::mutex callback_mutex_;

    // === 其他配置 ===
    std::string leader_info_;
    mutable std::mutex leader_info_mutex_;

    std::atomic<int64_t> next_can_campaign_time_us_{0};
    int64_t forbid_campaign_time_ms_ = 0;
};

typedef std::shared_ptr<LeaderElector> LeaderElectorPtr;

} // namespace kv_cache_manager