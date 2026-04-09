#include <atomic>
#include <chrono>
#include <filesystem>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <memory>
#include <thread>

#include "kv_cache_manager/common/loop_thread.h"
#include "kv_cache_manager/common/standard_uri.h"
#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/config/coordination_file_backend.h"
#include "kv_cache_manager/config/leader_elector.h"
#include "kv_cache_manager/config/node_endpoint_info.h"

using namespace kv_cache_manager;

class LeaderElectorTest : public TESTBASE {
protected:
    void SetUp() override {
        // 创建临时目录用于锁文件
        test_dir_ = GetPrivateTestRuntimeDataPath();
        std::error_code std_ec;
        std::filesystem::remove_all(test_dir_, std_ec);
        std::filesystem::create_directories(test_dir_);

        // 初始化协调后端
        coordination_backend_ = std::make_shared<CoordinationFileBackend>();
        StandardUri uri("file://" + test_dir_);
        ErrorCode ec = coordination_backend_->Init(uri);
        ASSERT_EQ(ec, EC_OK);

        // 准备选举器参数
        lock_key_ = "test_lock_key";
        lock_value_ = "127.0.0.1-" + std::to_string(getpid());
        lease_ms_ = 1000; // 较短的租约时间以便测试
        loop_interval_ms_ = 10;
    }

    void TearDown() override {
        // 清理临时目录
        std::error_code ec;
        std::filesystem::remove_all(test_dir_, ec);
    }

    // 创建新的选举器实例
    std::shared_ptr<LeaderElector> CreateElector(const std::string &custom_value = "") {
        std::string value = custom_value.empty() ? lock_value_ : custom_value;
        return std::make_shared<LeaderElector>(coordination_backend_, lock_key_, value, lease_ms_, loop_interval_ms_);
    }

protected:
    std::string test_dir_;
    std::shared_ptr<CoordinationBackend> coordination_backend_;
    std::string lock_key_;
    std::string lock_value_;
    int64_t lease_ms_;
    int64_t loop_interval_ms_;
};

// 测试基本构造和析构
TEST_F(LeaderElectorTest, ConstructorAndDestructor) {
    auto elector = CreateElector();
    EXPECT_NE(elector, nullptr);

    // 初始状态应为非领导者
    EXPECT_FALSE(elector->IsLeader());
    EXPECT_EQ(elector->GetLeaseExpirationTime(), -1);
}

// 测试设置回调函数
TEST_F(LeaderElectorTest, SetHandlers) {
    auto elector = CreateElector();

    std::atomic<bool> become_leader_called{false};
    std::atomic<bool> no_longer_leader_called{false};

    elector->SetBecomeLeaderHandler([&become_leader_called]() { become_leader_called = true; });

    elector->SetNoLongerLeaderHandler([&no_longer_leader_called]() { no_longer_leader_called = true; });

    // 回调尚未被调用
    EXPECT_FALSE(become_leader_called);
    EXPECT_FALSE(no_longer_leader_called);
}

// 测试启动和停止（无回调）
TEST_F(LeaderElectorTest, StartStopWithoutHandlers) {
    auto elector = CreateElector();

    // 未设置回调，Start应该失败
    EXPECT_FALSE(elector->Start());

    // 设置回调后应该成功
    elector->SetBecomeLeaderHandler([]() {});
    elector->SetNoLongerLeaderHandler([]() {});
    EXPECT_TRUE(elector->Start());

    // 等待一小段时间让线程运行
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    // 停止选举器
    elector->Stop();

    // 确保停止后状态正确
    EXPECT_FALSE(elector->IsLeader());
}

// 测试单个选举器成为领导者
TEST_F(LeaderElectorTest, SingleElectorBecomesLeader) {
    auto elector = CreateElector();

    std::atomic<bool> become_leader_called{false};
    std::atomic<bool> no_longer_leader_called{false};

    elector->SetBecomeLeaderHandler([&become_leader_called]() { become_leader_called = true; });

    elector->SetNoLongerLeaderHandler([&no_longer_leader_called]() { no_longer_leader_called = true; });

    EXPECT_TRUE(elector->Start());

    for (int i = 0; i < 100; i++) {
        // 等待足够时间让选举发生
        if (elector->IsLeader() && become_leader_called) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // 应该成为领导者
    EXPECT_TRUE(become_leader_called);
    EXPECT_FALSE(no_longer_leader_called);
    EXPECT_TRUE(elector->IsLeader());

    // 租约时间应该被设置
    EXPECT_GT(elector->GetLeaseExpirationTime(), 0);

    elector->Stop();
}

// 测试手动降级
TEST_F(LeaderElectorTest, ManualDemote) {
    auto elector = CreateElector();

    std::atomic<bool> become_leader_called{false};
    std::atomic<bool> no_longer_leader_called{false};

    elector->SetBecomeLeaderHandler([&become_leader_called]() { become_leader_called = true; });

    elector->SetNoLongerLeaderHandler([&no_longer_leader_called]() { no_longer_leader_called = true; });

    EXPECT_TRUE(elector->Start());

    for (int i = 0; i < 100; i++) {
        // 等待足够时间让选举发生
        if (elector->IsLeader() && become_leader_called) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    ASSERT_TRUE(elector->IsLeader());
    ASSERT_TRUE(become_leader_called);
    elector->SetForbidCampaignLeaderTimeMs(1000000);
    // 手动降级
    elector->Demote();

    // 等待demote task执行
    for (int i = 0; i < 10; i++) {
        if (!elector->IsLeader() && no_longer_leader_called) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    // 应该不再是领导者，并且调用了 no_longer_leader 回调
    EXPECT_FALSE(elector->IsLeader());
    EXPECT_TRUE(no_longer_leader_called);

    elector->Stop();
}

// 测试两个选举器竞争，只有一个能成为领导者
TEST_F(LeaderElectorTest, TwoElectorsCompetition) {
    auto elector1 = CreateElector("instance1");
    auto elector2 = CreateElector("instance2");

    std::atomic<bool> elector1_leader{false};
    std::atomic<bool> elector2_leader{false};
    std::atomic<bool> elector1_lost{false};
    std::atomic<bool> elector2_lost{false};

    elector1->SetBecomeLeaderHandler([&elector1_leader]() { elector1_leader = true; });
    elector1->SetNoLongerLeaderHandler([&elector1_lost]() { elector1_lost = true; });

    elector2->SetBecomeLeaderHandler([&elector2_leader]() { elector2_leader = true; });
    elector2->SetNoLongerLeaderHandler([&elector2_lost]() { elector2_lost = true; });

    // 启动两个选举器
    EXPECT_TRUE(elector1->Start());
    EXPECT_TRUE(elector2->Start());

    for (int i = 0; i < 100; i++) {
        // 等待足够时间让选举发生
        if ((elector1->IsLeader() || elector2->IsLeader()) && (elector1_leader || elector2_leader)) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // 检查只有一个选举器成为领导者
    const bool is_elector1_leader = elector1->IsLeader();
    const bool is_elector2_leader = elector2->IsLeader();

    EXPECT_NE(is_elector1_leader, is_elector2_leader) << "只有一个选举器应该成为领导者";

    // 验证相应的回调被调用
    if (is_elector1_leader) {
        EXPECT_TRUE(elector1_leader);
        EXPECT_FALSE(elector2_leader);
        elector1_leader = false;

        elector1->Stop();
    } else {
        EXPECT_TRUE(elector2_leader);
        EXPECT_FALSE(elector1_leader);
        elector2_leader = false;
        elector2->Stop();
    }

    // 等待足够时间让选举发生
    std::this_thread::sleep_for(std::chrono::milliseconds(300));

    // 另外一个elector应该能够成为领导者
    if (is_elector1_leader) {
        EXPECT_TRUE(elector2->IsLeader());
        EXPECT_TRUE(elector2_leader);
        elector2->Stop();
    } else {
        EXPECT_TRUE(elector1->IsLeader());
        EXPECT_TRUE(elector1_leader);
        elector1->Stop();
    }
}

// 测试领导者失去锁后自动降级
TEST_F(LeaderElectorTest, LeaderLosesLock) {
    auto elector = CreateElector();

    std::atomic<bool> become_leader_called{false};
    std::atomic<bool> no_longer_leader_called{false};

    elector->SetBecomeLeaderHandler([&become_leader_called]() { become_leader_called = true; });

    elector->SetNoLongerLeaderHandler([&no_longer_leader_called]() { no_longer_leader_called = true; });

    // 不使用elector->Start以完全控制DoWorkLoop调用节奏。
    elector->DoWorkLoop(TimestampUtil::GetCurrentTimeUs());
    elector->ProcessStateTransitionsForTest();
    ASSERT_TRUE(elector->IsLeader());
    ASSERT_TRUE(become_leader_called);

    // 获取当前租约过期时间
    int64_t lease_expiration = elector->GetLeaseExpirationTime();
    EXPECT_GT(lease_expiration, 0);

    // 模拟锁被外部抢占
    ErrorCode unlock_ec = coordination_backend_->Unlock(lock_key_, lock_value_);
    EXPECT_EQ(unlock_ec, EC_OK) << "Unlock should succeed";
    ErrorCode lock_ec = coordination_backend_->TryLock(lock_key_, lock_value_ + "_other", 1000000);
    EXPECT_EQ(lock_ec, EC_OK) << "Lock by other should succeed";

    elector->DoWorkLoop(TimestampUtil::GetCurrentTimeUs());
    elector->ProcessStateTransitionsForTest();

    // 应该不再是领导者，并且调用了 no_longer_leader 回调
    EXPECT_FALSE(elector->IsLeader()) << "Leader elector failed to detect lock loss after lease expiration";
    EXPECT_TRUE(no_longer_leader_called) << "no_longer_leader callback should be called";

    elector->Stop();
}

// 测试禁止竞选时间
TEST_F(LeaderElectorTest, ForbidCampaignTime) {
    auto elector = CreateElector();

    // 设置禁止竞选时间为500ms
    elector->SetForbidCampaignLeaderTimeMs(500);

    // 启动选举器
    std::atomic<bool> become_leader_called{false};
    elector->SetBecomeLeaderHandler([&become_leader_called]() { become_leader_called = true; });
    elector->SetNoLongerLeaderHandler([]() {});

    int64_t base_time = TimestampUtil::GetCurrentTimeUs();

    // 成为领导者
    elector->DoWorkLoop(base_time);
    ASSERT_TRUE(elector->IsLeader());

    // 100ms时手动降级
    elector->DoDemote(base_time + 100 * 1000);
    EXPECT_FALSE(elector->IsLeader());

    // 200ms时：由于禁止竞选时间，应该不会立即重新成为领导者
    elector->DoWorkLoop(base_time + 200 * 1000);
    EXPECT_FALSE(elector->IsLeader());

    // 599ms时：由于禁止竞选时间，应该不会立即重新成为领导者
    elector->DoWorkLoop(base_time + 599 * 1000);
    EXPECT_FALSE(elector->IsLeader());

    // 等待禁止竞选时间过后，应该可以重新成为领导者
    elector->DoWorkLoop(base_time + 600 * 1000);
    EXPECT_TRUE(elector->IsLeader());
}

// 测试 WorkLoop 方法
TEST_F(LeaderElectorTest, WorkLoopMethod) {
    auto elector = CreateElector();

    // 不启动后台线程，直接调用 WorkLoop
    [[maybe_unused]] bool is_leader = elector->WorkLoop();

    // 第一次调用可能成功也可能失败，取决于锁的状态
    // 至少应该不会崩溃
    EXPECT_NO_THROW(elector->WorkLoop());

    // 多次调用
    for (int i = 0; i < 5; ++i) {
        EXPECT_NO_THROW(elector->WorkLoop());
    }
}

// 测试租约过期检测
TEST_F(LeaderElectorTest, LeaseExpiration) {
    // 指定租约时间
    auto elector = std::make_shared<LeaderElector>(coordination_backend_, lock_key_, lock_value_, 50, 10);

    std::atomic<bool> become_leader_called{false};
    std::atomic<bool> no_longer_leader_called{false};

    elector->SetBecomeLeaderHandler([&become_leader_called]() { become_leader_called = true; });

    elector->SetNoLongerLeaderHandler([&no_longer_leader_called]() { no_longer_leader_called = true; });

    elector->WorkLoop();
    elector->ProcessStateTransitionsForTest();
    ASSERT_TRUE(elector->IsLeader());
    ASSERT_TRUE(become_leader_called);

    // 模拟WorkLoop长时间阻塞,租约应该正常过期
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    elector->CheckLeaseTimeout();
    elector->ProcessStateTransitionsForTest();
    ASSERT_FALSE(elector->IsLeader());
    ASSERT_TRUE(no_longer_leader_called);

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    // WorkLoop应该能重新获得领导者身份
    become_leader_called = false;
    elector->WorkLoop();
    elector->ProcessStateTransitionsForTest();
    ASSERT_TRUE(elector->IsLeader());
    ASSERT_TRUE(become_leader_called);
}

// 测试重复启动和停止
TEST_F(LeaderElectorTest, RepeatedStartStop) {
    auto elector = CreateElector();

    elector->SetBecomeLeaderHandler([]() {});
    elector->SetNoLongerLeaderHandler([]() {});

    // 多次启动停止
    for (int i = 0; i < 3; ++i) {
        EXPECT_TRUE(elector->Start());
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        elector->Stop();
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
}

// 测试在停止状态下调用方法
TEST_F(LeaderElectorTest, MethodsWhenStopped) {
    auto elector = CreateElector();

    // 未启动状态下调用方法
    EXPECT_FALSE(elector->IsLeader());
    EXPECT_EQ(elector->GetLeaseExpirationTime(), -1);

    // 调用 Demote（应该没有效果）
    EXPECT_NO_THROW(elector->Demote());
}

// 测试多个选举器使用不同的锁键
TEST_F(LeaderElectorTest, MultipleElectorsDifferentKeys) {
    auto elector1 = std::make_shared<LeaderElector>(coordination_backend_, "key1", "instance1", lease_ms_, loop_interval_ms_);
    auto elector2 = std::make_shared<LeaderElector>(coordination_backend_, "key2", "instance2", lease_ms_, loop_interval_ms_);

    std::atomic<bool> elector1_leader{false};
    std::atomic<bool> elector2_leader{false};

    elector1->SetBecomeLeaderHandler([&elector1_leader]() { elector1_leader = true; });
    elector1->SetNoLongerLeaderHandler([]() {});

    elector2->SetBecomeLeaderHandler([&elector2_leader]() { elector2_leader = true; });
    elector2->SetNoLongerLeaderHandler([]() {});

    EXPECT_TRUE(elector1->Start());
    EXPECT_TRUE(elector2->Start());

    for (int i = 0; i < 100; i++) {
        // 等待足够时间让选举发生
        if (elector1_leader && elector2_leader) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // 两个选举器都应该成为领导者，因为它们使用不同的锁键
    EXPECT_TRUE(elector1_leader);
    EXPECT_TRUE(elector2_leader);
    EXPECT_TRUE(elector1->IsLeader());
    EXPECT_TRUE(elector2->IsLeader());

    elector1->Stop();
    elector2->Stop();
}

// 测试边界条件：极短的租约时间
TEST_F(LeaderElectorTest, VeryShortLease) {
    // 租约时间小于循环间隔的10倍，应该产生警告但正常工作
    auto elector = std::make_shared<LeaderElector>(coordination_backend_, lock_key_, lock_value_, 5, 2); // 5ms租约，2ms循环

    elector->SetBecomeLeaderHandler([]() {});
    elector->SetNoLongerLeaderHandler([]() {});

    EXPECT_TRUE(elector->Start());

    // 短暂运行
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    elector->Stop();
}

// 测试线程安全：从多个线程调用 IsLeader
TEST_F(LeaderElectorTest, ThreadSafeIsLeader) {
    auto elector = CreateElector();

    elector->SetBecomeLeaderHandler([]() {});
    elector->SetNoLongerLeaderHandler([]() {});

    EXPECT_TRUE(elector->Start());

    // 创建多个线程同时调用 IsLeader
    std::vector<std::thread> threads;
    std::atomic<int> call_count{0};

    for (int i = 0; i < 10; ++i) {
        threads.emplace_back([&elector, &call_count]() {
            for (int j = 0; j < 100; ++j) {
                elector->IsLeader();
                call_count++;
                std::this_thread::sleep_for(std::chrono::microseconds(10));
            }
        });
    }

    // 等待所有线程完成
    for (auto &t : threads) {
        t.join();
    }

    EXPECT_EQ(call_count, 1000);

    elector->Stop();
}

// 测试两个选举器竞争时，原领导者长期未续约，另一个选举器能正常获得租约
TEST_F(LeaderElectorTest, LeaseExpirationWithCompetition) {
    // 创建两个选举器，使用不同的锁值
    auto elector1 = CreateElector("instance1");
    auto elector2 = CreateElector("instance2");

    std::atomic<bool> elector1_leader_called{false};
    std::atomic<bool> elector1_no_longer_leader_called{false};
    std::atomic<bool> elector2_leader_called{false};
    std::atomic<bool> elector2_no_longer_leader_called{false};

    // 设置回调函数
    elector1->SetBecomeLeaderHandler([&elector1_leader_called]() { elector1_leader_called = true; });
    elector1->SetNoLongerLeaderHandler(
        [&elector1_no_longer_leader_called]() { elector1_no_longer_leader_called = true; });

    elector2->SetBecomeLeaderHandler([&elector2_leader_called]() { elector2_leader_called = true; });
    elector2->SetNoLongerLeaderHandler(
        [&elector2_no_longer_leader_called]() { elector2_no_longer_leader_called = true; });

    // 不使用 Start()，以便手动控制 WorkLoop 和 CheckLeaseTimeout 的调用
    // elector1 先成为领导者
    elector1->WorkLoop();
    elector1->ProcessStateTransitionsForTest();
    ASSERT_TRUE(elector1->IsLeader()) << "elector1 should become leader first";
    ASSERT_TRUE(elector1_leader_called) << "elector1's become leader callback should be called";

    // 记录 elector1 的租约过期时间
    int64_t initial_lease_expiration = elector1->GetLeaseExpirationTime();
    EXPECT_GT(initial_lease_expiration, 0) << "elector1 should have valid lease expiration time";

    // elector2 尝试成为领导者，但应该失败（因为 elector1 持有锁）
    elector2->WorkLoop();
    elector2->ProcessStateTransitionsForTest();
    EXPECT_FALSE(elector2->IsLeader()) << "elector2 should not become leader while elector1 holds the lock";
    EXPECT_FALSE(elector2_leader_called) << "elector2's become leader callback should not be called yet";

    // 模拟 elector1 长时间未续约，导致租约过期
    // 等待足够时间让租约过期
    std::this_thread::sleep_for(std::chrono::milliseconds(lease_ms_ * 2));

    // 现在 elector2 应该能够获得领导者身份
    elector2->WorkLoop();
    elector2->ProcessStateTransitionsForTest();
    EXPECT_TRUE(elector2->IsLeader()) << "elector2 should become leader after elector1's lease expired";
    EXPECT_TRUE(elector2_leader_called) << "elector2's become leader callback should be called";
    // 验证 elector2 有有效的租约时间
    EXPECT_GT(elector2->GetLeaseExpirationTime(), 0) << "elector2 should have valid lease expiration time";

    // 调用 elector1 应该失去领导者身份
    elector1->WorkLoop();
    elector1->ProcessStateTransitionsForTest();
    // 验证 elector1 不再是领导者
    EXPECT_FALSE(elector1->IsLeader()) << "elector1 should no longer be leader after lease expiration";
    EXPECT_TRUE(elector1_no_longer_leader_called) << "elector1's no longer leader callback should be called";
}

// 测试 SetSelfNodeInfo / GetNodeInfo 基本功能
TEST_F(LeaderElectorTest, SetSelfNodeInfoAndGetNodeInfo) {
    auto elector = CreateElector();

    std::string self_node_id = elector->GetSelfNodeID();

    // 写入节点信息
    NodeEndpointInfo write_info(self_node_id, "192.168.1.100", 8001, 8002, 9001, 9002, "region=us-east-1");
    ErrorCode ec = elector->SetSelfNodeInfo(write_info);
    EXPECT_EQ(EC_OK, ec);

    // 读取节点信息
    NodeEndpointInfo read_info;
    ec = elector->GetNodeInfo(self_node_id, read_info);
    EXPECT_EQ(EC_OK, ec);

    // 验证所有字段
    EXPECT_EQ(self_node_id, read_info.node_id());
    EXPECT_EQ("192.168.1.100", read_info.host());
    EXPECT_EQ(8001, read_info.meta_rpc_port());
    EXPECT_EQ(8002, read_info.meta_http_port());
    EXPECT_EQ(9001, read_info.admin_rpc_port());
    EXPECT_EQ(9002, read_info.admin_http_port());
    EXPECT_EQ("region=us-east-1", read_info.custom_info());
}

// 测试读取不存在的节点信息
TEST_F(LeaderElectorTest, ReadNonExistentNodeInfo) {
    auto elector = CreateElector();

    NodeEndpointInfo read_info;
    ErrorCode ec = elector->GetNodeInfo("nonexistent_node", read_info);
    EXPECT_NE(EC_OK, ec);
}

// 测试 Leader 写入节点信息后，通过 GetLeaderNodeInfo 可查到 Leader 连接方式
TEST_F(LeaderElectorTest, LeaderDiscoveryEndToEnd) {
    const std::string advertised_host = "10.0.0.42";
    const int32_t meta_rpc_port = 50051;
    const int32_t meta_http_port = 8080;
    const int32_t admin_rpc_port = 50052;
    const int32_t admin_http_port = 8081;
    const std::string custom_info = "az=cn-hangzhou-h";

    auto elector = CreateElector();

    std::atomic<bool> become_leader_called{false};

    elector->SetBecomeLeaderHandler([&become_leader_called]() { become_leader_called = true; });
    elector->SetNoLongerLeaderHandler([]() {});

    // 成为 leader
    elector->DoWorkLoop(TimestampUtil::GetCurrentTimeUs());
    elector->ProcessStateTransitionsForTest();
    ASSERT_TRUE(elector->IsLeader());
    ASSERT_TRUE(become_leader_called);

    // 模拟 server.cc 中 CreateLeaderElector 的行为：leader 写入自己的节点信息
    std::string self_node_id = elector->GetSelfNodeID();
    NodeEndpointInfo node_info(self_node_id, advertised_host,
                               meta_rpc_port, meta_http_port,
                               admin_rpc_port, admin_http_port, custom_info);
    ErrorCode ec = elector->SetSelfNodeInfo(node_info);
    ASSERT_EQ(EC_OK, ec);

    // 模拟客户端发现 leader 的流程：
    // 1. 通过 GetLeaderNodeID 获取当前 leader 的 node_id
    std::string leader_node_id = elector->GetLeaderNodeID();
    EXPECT_FALSE(leader_node_id.empty());
    EXPECT_EQ(self_node_id, leader_node_id);

    // 2. 通过 GetLeaderNodeInfo 读取 leader 的连接信息
    NodeEndpointInfo leader_info;
    ec = elector->GetLeaderNodeInfo(leader_info);
    ASSERT_EQ(EC_OK, ec);

    // 3. 验证返回的连接信息完全正确（使用 advertised_host 保证确定性）
    EXPECT_EQ(self_node_id, leader_info.node_id());
    EXPECT_EQ(advertised_host, leader_info.host());
    EXPECT_EQ(meta_rpc_port, leader_info.meta_rpc_port());
    EXPECT_EQ(meta_http_port, leader_info.meta_http_port());
    EXPECT_EQ(admin_rpc_port, leader_info.admin_rpc_port());
    EXPECT_EQ(admin_http_port, leader_info.admin_http_port());
    EXPECT_EQ(custom_info, leader_info.custom_info());

    elector->Stop();
}

// 测试两个选举器竞争时，Leader 的节点信息可被另一个节点读取
TEST_F(LeaderElectorTest, LeaderDiscoveryTwoNodes) {
    const std::string host1 = "10.0.0.1";
    const std::string host2 = "10.0.0.2";

    auto elector1 = CreateElector("instance1");
    auto elector2 = CreateElector("instance2");

    std::atomic<bool> elector1_leader{false};
    std::atomic<bool> elector2_leader{false};

    elector1->SetBecomeLeaderHandler([&elector1_leader]() { elector1_leader = true; });
    elector1->SetNoLongerLeaderHandler([]() {});
    elector2->SetBecomeLeaderHandler([&elector2_leader]() { elector2_leader = true; });
    elector2->SetNoLongerLeaderHandler([]() {});

    // 两个节点都写入自己的连接信息
    NodeEndpointInfo info1("instance1", host1, 8001, 8002, 9001, 9002, "rack=A");
    NodeEndpointInfo info2("instance2", host2, 8003, 8004, 9003, 9004, "rack=B");
    ASSERT_EQ(EC_OK, elector1->SetSelfNodeInfo(info1));
    ASSERT_EQ(EC_OK, elector2->SetSelfNodeInfo(info2));

    // 启动竞争
    EXPECT_TRUE(elector1->Start());
    EXPECT_TRUE(elector2->Start());

    for (int i = 0; i < 100; i++) {
        if ((elector1->IsLeader() || elector2->IsLeader()) && (elector1_leader || elector2_leader)) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    ASSERT_TRUE(elector1->IsLeader() || elector2->IsLeader());

    // 从任一 elector 获取 leader node id
    std::string leader_id = elector1->GetLeaderNodeID();
    EXPECT_FALSE(leader_id.empty());

    // 用非 leader 的 elector 读取 leader 的节点信息（模拟从 follower 查询 leader）
    LeaderElector *follower = elector1->IsLeader() ? elector2.get() : elector1.get();
    NodeEndpointInfo leader_info;
    ErrorCode ec = follower->GetNodeInfo(leader_id, leader_info);
    ASSERT_EQ(EC_OK, ec);

    // 验证读取到的信息和 leader 的注册信息一致
    EXPECT_EQ(leader_id, leader_info.node_id());
    if (elector1->IsLeader()) {
        EXPECT_EQ(host1, leader_info.host());
        EXPECT_EQ(8001, leader_info.meta_rpc_port());
        EXPECT_EQ(8002, leader_info.meta_http_port());
        EXPECT_EQ(9001, leader_info.admin_rpc_port());
        EXPECT_EQ(9002, leader_info.admin_http_port());
        EXPECT_EQ("rack=A", leader_info.custom_info());
    } else {
        EXPECT_EQ(host2, leader_info.host());
        EXPECT_EQ(8003, leader_info.meta_rpc_port());
        EXPECT_EQ(8004, leader_info.meta_http_port());
        EXPECT_EQ(9003, leader_info.admin_rpc_port());
        EXPECT_EQ(9004, leader_info.admin_http_port());
        EXPECT_EQ("rack=B", leader_info.custom_info());
    }

    elector1->Stop();
    elector2->Stop();
}

// 测试 NodeEndpointInfo 覆盖写入（模拟节点重启后更新端口）
TEST_F(LeaderElectorTest, NodeInfoOverwrite) {
    auto elector = CreateElector();

    std::string self_node_id = elector->GetSelfNodeID();

    // 第一次写入
    NodeEndpointInfo info1(self_node_id, "10.0.0.1", 8001, 8002, 9001, 9002, "v1");
    ASSERT_EQ(EC_OK, elector->SetSelfNodeInfo(info1));

    // 覆盖写入（模拟端口变更）
    NodeEndpointInfo info2(self_node_id, "10.0.0.1", 9999, 9998, 9997, 9996, "v2");
    ASSERT_EQ(EC_OK, elector->SetSelfNodeInfo(info2));

    // 读取应该是最新的值
    NodeEndpointInfo read_info;
    ASSERT_EQ(EC_OK, elector->GetNodeInfo(self_node_id, read_info));
    EXPECT_EQ("10.0.0.1", read_info.host());
    EXPECT_EQ(9999, read_info.meta_rpc_port());
    EXPECT_EQ(9998, read_info.meta_http_port());
    EXPECT_EQ(9997, read_info.admin_rpc_port());
    EXPECT_EQ(9996, read_info.admin_http_port());
    EXPECT_EQ("v2", read_info.custom_info());
}
