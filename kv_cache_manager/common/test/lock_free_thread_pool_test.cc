#include <atomic>
#include <chrono>
#include <functional>
#include <future>
#include <memory>
#include <stdexcept>

#include "kv_cache_manager/common/lock_free_thread_pool.h"
#include "kv_cache_manager/common/unittest.h"

using namespace kv_cache_manager;
using namespace std::chrono_literals;

class LockFreeThreadPoolTest : public TESTBASE {};

TEST_F(LockFreeThreadPoolTest, AsyncSupportsGenericResultsAndExceptions) {
    LockFreeThreadPool pool(2, 8, "generic");
    ASSERT_TRUE(pool.start());
    std::function<int()> function = []() { return 42; };
    EXPECT_EQ(42, pool.async(function).get());
    auto move_only = pool.async([value = std::make_unique<int>(7)]() mutable { return std::move(value); });
    EXPECT_EQ(7, *move_only.get());
    std::atomic<bool> invoked{false};
    pool.async([&]() { invoked = true; }).get();
    EXPECT_TRUE(invoked.load());
    auto failed = pool.async([]() -> int { throw std::runtime_error("task failure"); });
    EXPECT_THROW(failed.get(), std::runtime_error);
    EXPECT_EQ(9, pool.async([]() { return 9; }).get());
}

TEST_F(LockFreeThreadPoolTest, TwoWorkersExecuteConcurrently) {
    LockFreeThreadPool pool(2, 2, "concurrent");
    ASSERT_TRUE(pool.start());
    std::promise<void> first_started, second_started, release;
    auto first_running = first_started.get_future();
    auto second_running = second_started.get_future();
    auto gate = release.get_future().share();
    auto first = pool.async([&first_started, gate]() {
        first_started.set_value();
        gate.wait();
    });
    auto second = pool.async([&second_started, gate]() {
        second_started.set_value();
        gate.wait();
    });
    // Always release both workers before collecting results, even on failure.
    EXPECT_EQ(std::future_status::ready, first_running.wait_for(1s));
    EXPECT_EQ(std::future_status::ready, second_running.wait_for(1s));
    release.set_value();
    first.get();
    second.get();
}
