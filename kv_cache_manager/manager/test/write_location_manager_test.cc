#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/manager/write_location_manager.h"

namespace kv_cache_manager {

class WriteLocationManagerTest : public TESTBASE {
public:
    void SetUp() override {}
    void TearDown() override {}

private:
    WriteLocationManager manager_;
};

TEST_F(WriteLocationManagerTest, NoExpireLoopTest) {
    manager_.Put("session_1", {1, 2, 3}, {"id1", "id2", "id3"}, 1000, [this]() {
        WriteLocationManager::WriteLocationInfo info;
        this->manager_.GetAndDelete("session_1", info);
    });
    ASSERT_EQ(1, manager_.session_id_map_.Size());
    ASSERT_EQ(1, manager_.expire_queue_.Size());
    manager_.Put("session_2", {11, 22, 33}, {"id11", "id22", "id33"}, 1000, [this]() {
        WriteLocationManager::WriteLocationInfo info;
        this->manager_.GetAndDelete("session_2", info);
    });
    ASSERT_EQ(2, manager_.session_id_map_.Size());
    ASSERT_EQ(2, manager_.expire_queue_.Size());
    WriteLocationManager::WriteLocationInfo info;
    ASSERT_TRUE(manager_.GetAndDelete("session_1", info));
    ASSERT_EQ(std::vector<int64_t>({1, 2, 3}), info.keys);
    ASSERT_EQ(std::vector<std::string>({"id1", "id2", "id3"}), info.location_ids);
    ASSERT_EQ(1, manager_.session_id_map_.Size());
    ASSERT_EQ(2, manager_.expire_queue_.Size());
}

TEST_F(WriteLocationManagerTest, ExpireLoopTest) {
    manager_.Start();
    manager_.Put("session_1", {1, 2, 3}, {"id1", "id2", "id3"}, 1, [this]() {
        WriteLocationManager::WriteLocationInfo info;
        this->manager_.GetAndDelete("session_1", info);
    });
    ASSERT_EQ(1, manager_.session_id_map_.Size());
    ASSERT_EQ(1, manager_.expire_queue_.Size());
    std::this_thread::sleep_for(std::chrono::seconds(6));
    ASSERT_EQ(0, manager_.session_id_map_.Size());
    ASSERT_EQ(0, manager_.expire_queue_.Size());
}

TEST_F(WriteLocationManagerTest, MultiThreadTest) {
    manager_.Start();
    std::atomic_bool go = false;
    std::atomic_int count = 0;
    std::vector<std::thread> workers;
    for (int i = 0; i < 4; ++i) {
        workers.emplace_back(
            [&go, &count, this](int worker_id) {
                while (!go.load(std::memory_order_relaxed)) {
                    // busy wait
                }
                int base = worker_id * 10;
                std::vector<int64_t> keys;
                std::vector<std::string> ids;
                for (int j = 0; j < 3; ++j) {
                    keys.push_back(base + j);
                    ids.push_back("id_" + std::to_string(base + j));
                }
                std::string session_id = "session" + std::to_string(worker_id);
                this->manager_.Put(session_id, std::move(keys), std::move(ids), 2, [this, session_id]() {
                    WriteLocationManager::WriteLocationInfo info;
                    this->manager_.GetAndDelete(session_id, info);
                });
                count.fetch_add(1, std::memory_order_relaxed);
                ASSERT_LE(1, this->manager_.session_id_map_.Size());
                ASSERT_GE(4, this->manager_.session_id_map_.Size());
                ASSERT_LE(0, this->manager_.expire_queue_.Size());
                ASSERT_GE(4, this->manager_.expire_queue_.Size());
                if (worker_id == 0) {
                    WriteLocationManager::WriteLocationInfo info;
                    this->manager_.GetAndDelete(session_id, info);
                    ASSERT_EQ(std::vector<int64_t>({base, base + 1, base + 2}), info.keys);
                    ASSERT_GE(3, this->manager_.session_id_map_.Size());
                }
            },
            i);
    }
    go.store(true, std::memory_order_relaxed);
    while (count.load(std::memory_order_relaxed) < 4) {
        // busy wait
    }
    std::this_thread::sleep_for(std::chrono::seconds(10));
    ASSERT_EQ(0, manager_.session_id_map_.Size());
    ASSERT_EQ(0, manager_.expire_queue_.Size());
    for (int i = 0; i < 4; ++i) {
        workers[i].join();
    }
}

TEST_F(WriteLocationManagerTest, DoCleanupTest) {
    // 添加几个会话
    int cleanup_called_count = 0;

    manager_.Put("session_1", {1, 2, 3}, {"id1", "id2", "id3"}, 1000, [&cleanup_called_count, this]() {
        cleanup_called_count++;
        WriteLocationManager::WriteLocationInfo info;
        this->manager_.GetAndDelete("session_1", info);
    });

    manager_.Put("session_2", {4, 5, 6}, {"id4", "id5", "id6"}, 1000, [&cleanup_called_count, this]() {
        cleanup_called_count++;
        WriteLocationManager::WriteLocationInfo info;
        this->manager_.GetAndDelete("session_2", info);
    });

    manager_.Put("session_3", {7, 8, 9}, {"id7", "id8", "id9"}, 1000, [&cleanup_called_count, this]() {
        cleanup_called_count++;
        WriteLocationManager::WriteLocationInfo info;
        this->manager_.GetAndDelete("session_3", info);
    });

    // 验证初始状态
    ASSERT_EQ(3, manager_.session_id_map_.Size());
    ASSERT_EQ(3, manager_.expire_queue_.Size());

    // 删除一个会话（模拟正常消费）
    WriteLocationManager::WriteLocationInfo info;
    ASSERT_TRUE(manager_.GetAndDelete("session_2", info));
    ASSERT_EQ(std::vector<int64_t>({4, 5, 6}), info.keys);
    ASSERT_EQ(std::vector<std::string>({"id4", "id5", "id6"}), info.location_ids);

    // 验证删除后的状态
    ASSERT_EQ(2, manager_.session_id_map_.Size());
    ASSERT_EQ(3, manager_.expire_queue_.Size()); // 队列中还有3个，但session_2已从map中删除

    // 执行DoCleanup
    manager_.DoCleanup();

    // 验证DoCleanup后的状态
    // session_1和session_3应该被清理，session_2应该被跳过（因为已从map中删除）
    ASSERT_EQ(2, cleanup_called_count);            // 只有session_1和session_3的回调被调用
    ASSERT_EQ(0, manager_.session_id_map_.Size()); // 所有会话都被清理
    ASSERT_EQ(0, manager_.expire_queue_.Size());   // 队列应该为空
}

TEST_F(WriteLocationManagerTest, DoCleanupEmptyTest) {
    // 测试空状态下的DoCleanup
    ASSERT_EQ(0, manager_.session_id_map_.Size());
    ASSERT_EQ(0, manager_.expire_queue_.Size());

    // 执行DoCleanup，不应该崩溃
    manager_.DoCleanup();

    // 验证状态不变
    ASSERT_EQ(0, manager_.session_id_map_.Size());
    ASSERT_EQ(0, manager_.expire_queue_.Size());
}

TEST_F(WriteLocationManagerTest, DoCleanupWithExpiredSessionsTest) {
    // 测试DoCleanup处理已过期会话
    int cleanup_called_count = 0;

    // 添加一个很快过期的会话
    manager_.Put("session_short", {10, 11}, {"id10", "id11"}, 1, [&cleanup_called_count, this]() {
        cleanup_called_count++;
        WriteLocationManager::WriteLocationInfo info;
        this->manager_.GetAndDelete("session_short", info);
    });

    // 添加一个长时间过期的会话
    manager_.Put("session_long", {20, 21}, {"id20", "id21"}, 1000, [&cleanup_called_count, this]() {
        cleanup_called_count++;
        WriteLocationManager::WriteLocationInfo info;
        this->manager_.GetAndDelete("session_long", info);
    });

    ASSERT_EQ(2, manager_.session_id_map_.Size());
    ASSERT_EQ(2, manager_.expire_queue_.Size());

    // 等待短时间会话过期
    std::this_thread::sleep_for(std::chrono::seconds(2));

    // 执行DoCleanup
    manager_.DoCleanup();

    // 两个会话都应该被清理
    ASSERT_EQ(2, cleanup_called_count);
    ASSERT_EQ(0, manager_.session_id_map_.Size());
    ASSERT_EQ(0, manager_.expire_queue_.Size());
}

} // namespace kv_cache_manager