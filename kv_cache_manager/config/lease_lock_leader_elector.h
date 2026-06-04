#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>

#include "kv_cache_manager/common/error_code.h"
#include "kv_cache_manager/config/leader_elector_interface.h"

namespace kv_cache_manager {

class LoopThread;
class CoordinationBackend;
class NodeEndpointInfo;

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

class LeaseLockLeaderElector : public ILeaderElector {
public:
    LeaseLockLeaderElector(const std::shared_ptr<CoordinationBackend> &coordination_backend,
                  const std::string &lock_key,
                  const std::string &lock_value,
                  int64_t lease_ms = 60,
                  int64_t loop_interval_ms = 6);
    ~LeaseLockLeaderElector() override;
    LeaseLockLeaderElector(const LeaseLockLeaderElector &) = delete;
    LeaseLockLeaderElector &operator=(const LeaseLockLeaderElector &) = delete;

    bool Start() override;
    void Stop() override;

    // 角色状态查询
    bool IsLeader() const override;
    RoleState GetRoleState() const;
    bool IsStableState() const;

    // 等待状态稳定（主要用于测试和优雅关闭）
    bool WaitForStableState(int64_t timeout_ms = -1);

    // 租约信息
    int64_t GetLeaseExpirationTime() const override;

    // 主动降级
    void Demote() override;

    // 设置回调（必须在 Start 之前调用）
    void SetNoLongerLeaderHandler(const HandlerFuncType &handler) override;
    void SetBecomeLeaderHandler(const HandlerFuncType &handler) override;

    // Leader 信息
    std::string GetLeaderNodeID() const override;
    std::string GetSelfNodeID() const override;

    // 节点连接信息存储（通过 CoordinationBackend KV）
    ErrorCode SetSelfNodeInfo(const NodeEndpointInfo &node_info) override;
    ErrorCode GetSelfNodeInfo(NodeEndpointInfo &out_node_info) const override;
    ErrorCode GetNodeInfo(const std::string &node_id, NodeEndpointInfo &out_node_info) override;
    ErrorCode GetLeaderNodeInfo(NodeEndpointInfo &out_node_info) override;

    // 选主控制
    void SetForbidCampaignLeaderTimeMs(int64_t forbid_time) override;
    int64_t GetForbidCampaignLeaderTimeMs() const override;
    int64_t GetLastLoopTimeUs() const override;
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

    // 节点信息存储（内部实现）
    ErrorCode SetNodeInfo(const std::string &node_id, const NodeEndpointInfo &node_info);

    // 锁状态更新
    void UpdateLockStatus(int64_t current_time, bool acquired, int64_t lease_expiration_time);

private:
    // === 分布式锁相关 ===
    std::shared_ptr<CoordinationBackend> coordination_backend_;
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
    std::atomic<int64_t> next_can_campaign_time_us_{0};
    int64_t forbid_campaign_time_ms_ = 0;

    // === 自身节点信息缓存 ===
    std::unique_ptr<NodeEndpointInfo> self_node_info_cache_;
    mutable std::mutex self_node_info_mutex_;
};

typedef std::shared_ptr<LeaseLockLeaderElector> LeaseLockLeaderElectorPtr;

} // namespace kv_cache_manager