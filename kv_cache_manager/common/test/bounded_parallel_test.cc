#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <dlfcn.h>
#include <future>
#include <mutex>
#include <pthread.h>
#include <thread>

#include "kv_cache_manager/common/bounded_parallel.h"
#include "kv_cache_manager/common/unittest.h"
#include "stub.h"

namespace kv_cache_manager {

TEST(BoundedParallelTest, RunsEveryItemOnceAndBoundsConcurrentWork) {
    std::mutex mutex;
    std::condition_variable changed;
    bool release = false;
    size_t active = 0;
    size_t peak = 0;
    std::vector<int> visits(20, 0);
    ErrorCode result = EC_UNKNOWN;
    std::thread runner([&] {
        result = RunBoundedParallel(visits.size(), 3, [&](size_t index) {
            std::unique_lock<std::mutex> lock(mutex);
            ++active;
            peak = std::max(peak, active);
            changed.notify_all();
            changed.wait(lock, [&] { return release; });
            ++visits[index];
            --active;
            return EC_OK;
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
    EXPECT_EQ(EC_OK, result);
    EXPECT_EQ(3, peak);
    EXPECT_EQ(std::vector<int>(visits.size(), 1), visits);
}

TEST(BoundedParallelTest, FailureWaitsForOtherClaimedWork) {
    std::mutex mutex;
    std::condition_variable changed;
    bool sibling_started = false;
    bool failure_ready = false;
    bool release_sibling = false;
    std::promise<ErrorCode> completion;
    auto result = completion.get_future();
    std::thread runner([&] {
        const auto caller_id = std::this_thread::get_id();
        completion.set_value(RunBoundedParallel(10, 2, [&, caller_id](size_t) {
            std::unique_lock<std::mutex> lock(mutex);
            if (std::this_thread::get_id() == caller_id) {
                changed.wait(lock, [&] { return sibling_started || release_sibling; });
                failure_ready = true;
                changed.notify_all();
                return EC_ERROR;
            }
            sibling_started = true;
            changed.notify_all();
            changed.wait(lock, [&] { return release_sibling; });
            return EC_OK;
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
    EXPECT_EQ(EC_ERROR, result.get());
}

TEST(BoundedParallelTest, ReportsErrorsAndSupportsNestedCalls) {
    size_t failed_calls = 0;
    EXPECT_EQ(EC_TIMEOUT, RunBoundedParallel(10, 1, [&](size_t) {
                  ++failed_calls;
                  return EC_TIMEOUT;
              }));
    EXPECT_EQ(1, failed_calls);
    EXPECT_EQ(EC_BADARGS, RunBoundedParallel(1, 0, [](size_t) { return EC_OK; }));
    EXPECT_EQ(EC_OK, RunBoundedParallel(0, 0, [](size_t) { return EC_ERROR; }));
    std::atomic<int> completed{0};
    EXPECT_EQ(EC_OK, RunBoundedParallel(4, 2, [&](size_t) {
                  return RunBoundedParallel(4, 2, [&](size_t) {
                      ++completed;
                      return EC_OK;
                  });
              }));
    EXPECT_EQ(16, completed.load());
}

TEST(BoundedParallelTest, ThreadCreationFailureReturnsErrorWithoutRunningNewWork) {
    auto create_thread = reinterpret_cast<decltype(&pthread_create)>(dlsym(RTLD_NEXT, "pthread_create"));
    ASSERT_NE(nullptr, create_thread);
    Stub stub;
    stub.set(
        create_thread, +[](pthread_t *, const pthread_attr_t *, void *(*)(void *), void *) { return EAGAIN; });
    size_t calls = 0;
    EXPECT_EQ(EC_ERROR, RunBoundedParallel(3, 2, [&](size_t) {
                  ++calls;
                  return EC_OK;
              }));
    EXPECT_EQ(0, calls);
}

} // namespace kv_cache_manager
