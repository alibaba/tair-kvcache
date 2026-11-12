#include <atomic>
#include <chrono>
#include <condition_variable>
#include <future>
#include <mutex>
#include <stdexcept>
#include <thread>

#include "kv_cache_manager/common/bounded_parallel.h"
#include "kv_cache_manager/common/unittest.h"

namespace kv_cache_manager {

TEST(BoundedParallelTest, RunsEveryItemOnceAndBoundsConcurrentWork) {
    std::mutex mutex;
    std::condition_variable changed;
    bool release = false;
    size_t active = 0;
    size_t peak = 0;
    std::vector<int> visits(20, 0);
    bool result = false;
    std::thread runner([&] {
        result = RunBoundedParallel(visits.size(), 3, [&](size_t index) {
            std::unique_lock<std::mutex> lock(mutex);
            ++active;
            peak = std::max(peak, active);
            changed.notify_all();
            changed.wait(lock, [&] { return release; });
            ++visits[index];
            --active;
            return true;
        });
    });
    bool overlapped = false;
    {
        std::unique_lock<std::mutex> lock(mutex);
        overlapped = changed.wait_for(lock, std::chrono::seconds(5), [&] { return active == 3; });
        release = true;
        changed.notify_all();
    }
    runner.join();
    EXPECT_TRUE(overlapped);
    EXPECT_TRUE(result);
    EXPECT_EQ(3, peak);
    EXPECT_EQ(std::vector<int>(visits.size(), 1), visits);
}

TEST(BoundedParallelTest, FailureWaitsForOtherClaimedWork) {
    std::mutex mutex;
    std::condition_variable changed;
    bool sibling_started = false;
    bool failure_ready = false;
    bool release_sibling = false;
    std::promise<bool> completion;
    auto result = completion.get_future();
    std::thread runner([&] {
        const auto caller_id = std::this_thread::get_id();
        completion.set_value(RunBoundedParallel(10, 2, [&, caller_id](size_t) {
            std::unique_lock<std::mutex> lock(mutex);
            if (std::this_thread::get_id() == caller_id) {
                changed.wait(lock, [&] { return sibling_started || release_sibling; });
                failure_ready = true;
                changed.notify_all();
                return false;
            }
            sibling_started = true;
            changed.notify_all();
            changed.wait(lock, [&] { return release_sibling; });
            return true;
        }));
    });
    bool saw_failure = false;
    {
        std::unique_lock<std::mutex> lock(mutex);
        saw_failure = changed.wait_for(lock, std::chrono::seconds(5), [&] { return failure_ready; });
    }
    const auto status = result.wait_for(std::chrono::milliseconds(30));
    {
        std::lock_guard<std::mutex> lock(mutex);
        release_sibling = true;
        changed.notify_all();
    }
    runner.join();
    EXPECT_TRUE(saw_failure);
    EXPECT_EQ(std::future_status::timeout, status);
    EXPECT_FALSE(result.get());
}

TEST(BoundedParallelTest, ContainsTaskExceptionsAndSupportsNestedCalls) {
    size_t failed_calls = 0;
    EXPECT_FALSE(RunBoundedParallel(10, 1, [&](size_t) {
        ++failed_calls;
        return false;
    }));
    EXPECT_EQ(1, failed_calls);
    EXPECT_FALSE(RunBoundedParallel(3, 1, [](size_t) -> bool { throw std::runtime_error("task failed"); }));
    EXPECT_FALSE(RunBoundedParallel(1, 0, [](size_t) { return true; }));
    EXPECT_TRUE(RunBoundedParallel(0, 0, [](size_t) { return false; }));
    std::atomic<int> completed{0};
    EXPECT_TRUE(RunBoundedParallel(4, 2, [&](size_t) {
        return RunBoundedParallel(4, 2, [&](size_t) {
            ++completed;
            return true;
        });
    }));
    EXPECT_EQ(16, completed.load());
}

} // namespace kv_cache_manager
