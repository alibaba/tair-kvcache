#include <atomic>
#include <chrono>
#include <exception>
#include <future>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <memory>
#include <system_error>
#include <thread>

#include "kv_cache_manager/common/loop_thread.h"
#include "kv_cache_manager/common/unittest.h"

using namespace kv_cache_manager;

class LoopThreadTest : public TESTBASE {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(LoopThreadTest, TestCreateWithNegativeInterval) {
    // 测试使用负数间隔创建LoopThread，应该返回nullptr
    auto loop_thread = LoopThread::CreateLoopThread([]() {}, -1);
    EXPECT_EQ(loop_thread, nullptr);
}

TEST_F(LoopThreadTest, TestCreateAndStop) {
    // 测试创建LoopThread并停止
    std::atomic<int> count(0);
    auto loop_thread = LoopThread::CreateLoopThread([&count]() { count++; }, 1000); // 1ms间隔
    EXPECT_NE(loop_thread, nullptr);

    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
    while (count.load() == 0 && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    int count_before_stop = count.load();
    EXPECT_GT(count_before_stop, 0);

    loop_thread->Stop();

    // A callback may already have been admitted between the observation above
    // and Stop() acquiring the loop mutex. Stop() joins that callback before
    // returning; only the value observed after Stop() is guaranteed to remain
    // stable.
    int count_after_stop = count.load();
    EXPECT_GE(count_after_stop, count_before_stop);
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    EXPECT_EQ(count_after_stop, count.load());
}

TEST_F(LoopThreadTest, TestRunOnce) {
    // 测试RunOnce功能
    std::atomic<int> count(0);
    auto loop_thread = LoopThread::CreateLoopThread([&count]() { count++; }, 10000000); // 10s间隔，确保不会自动执行
    EXPECT_NE(loop_thread, nullptr);

    // 等待LoopThread第一次执行（计数变成1），设置超时时间为500ms
    auto start_time = std::chrono::steady_clock::now();
    auto timeout = std::chrono::milliseconds(500);

    while (count.load() < 1) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        auto current_time = std::chrono::steady_clock::now();
        if (current_time - start_time > timeout) {
            FAIL() << "Timeout waiting for initial loop execution";
        }
    }

    // 此时计数应该为1
    EXPECT_EQ(count.load(), 1);

    // 调用RunOnce
    loop_thread->RunOnce();

    // 等待RunOnce执行（计数变成2），设置超时时间为500ms
    start_time = std::chrono::steady_clock::now();
    while (count.load() < 2) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        auto current_time = std::chrono::steady_clock::now();
        if (current_time - start_time > timeout) {
            FAIL() << "Timeout waiting for RunOnce execution";
        }
    }

    // 此时计数应该为2
    EXPECT_EQ(count.load(), 2);

    loop_thread->Stop();
}

TEST_F(LoopThreadTest, TestStrictMode) {
    // 测试严格模式
    std::atomic<int> count(0);
    auto loop_thread = LoopThread::CreateLoopThread(
        [&count]() {
            count++;
            // 模拟耗时操作
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        },
        1000,
        "test",
        true); // 严格模式
    EXPECT_NE(loop_thread, nullptr);

    auto start_time = std::chrono::steady_clock::now();
    const auto timeout = std::chrono::seconds(1);
    while (count.load() == 0) {
        if (std::chrono::steady_clock::now() - start_time > timeout) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    EXPECT_GT(count.load(), 0);

    loop_thread->Stop();
}

TEST_F(LoopThreadTest, TestRepeatedStop) {
    // 测试重复调用Stop
    auto loop_thread = LoopThread::CreateLoopThread([]() {}, 1000);
    EXPECT_NE(loop_thread, nullptr);

    // 多次调用Stop不应该出错
    loop_thread->Stop();
    loop_thread->Stop();
    loop_thread->Stop();
}

TEST_F(LoopThreadTest, TestStopFromOtherThreadDoesNotDetectSelfJoin) {
    std::atomic<int> count(0);
    auto loop_thread = LoopThread::CreateLoopThread([&count]() { count++; }, 1000, "normal");
    ASSERT_NE(loop_thread, nullptr);

    loop_thread->Stop();
    EXPECT_FALSE(loop_thread->SelfJoinDetected());
}

// 在自己的 loop 回调里调用 Stop() 属于自连接。检测标志置真、错误日志打出，
// 但 join 照常执行并抛出 std::system_error —— 不跳过 join、不 detach，问题被暴露而不是被掩盖。
TEST_F(LoopThreadTest, TestStopFromOwnLoopThreadDetectsSelfJoin) {
    std::promise<std::shared_ptr<LoopThread>> owner_promise;
    auto owner_future = owner_promise.get_future().share();
    std::promise<void> callback_done;
    auto callback_done_future = callback_done.get_future();
    std::exception_ptr self_join_error;

    auto loop_thread = LoopThread::CreateLoopThread(
        [&]() {
            // 回调可能早于 CreateLoopThread 返回就跑起来，先等实例发布完成。
            auto owner = owner_future.get();
            try {
                owner->Stop();
            } catch (const std::system_error &) {
                self_join_error = std::current_exception();
            }
            callback_done.set_value();
        },
        1000,
        "self-join");
    ASSERT_NE(loop_thread, nullptr);
    owner_promise.set_value(loop_thread);

    ASSERT_EQ(std::future_status::ready, callback_done_future.wait_for(std::chrono::seconds(5)));

    EXPECT_TRUE(loop_thread->SelfJoinDetected());
    ASSERT_NE(self_join_error, nullptr);
    EXPECT_THROW(std::rethrow_exception(self_join_error), std::system_error);

    // 回调返回后循环线程自行退出；由测试线程完成回收，避免 joinable 线程未被 join。
    if (loop_thread->thread_.joinable()) {
        loop_thread->thread_.join();
    }
    loop_thread->Stop();
    EXPECT_TRUE(loop_thread->SelfJoinDetected());
}
