#include <atomic>
#include <chrono>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <memory>
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

    // 等待一段时间让循环执行几次
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    int count_before_stop = count.load();
    EXPECT_GT(count_before_stop, 0);

    loop_thread->Stop();

    // 停止后计数应该不再增加
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    int count_after_stop = count.load();
    EXPECT_EQ(count_before_stop, count_after_stop);
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

    // 等待一段时间
    std::this_thread::sleep_for(std::chrono::milliseconds(20));

    int final_count = count.load();
    EXPECT_GT(final_count, 0);

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