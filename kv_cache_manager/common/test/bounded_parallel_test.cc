#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <dlfcn.h>
#include <future>
#include <limits>
#include <mutex>
#include <new>
#include <pthread.h>
#include <stdexcept>
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

TEST(BoundedParallelTest, TaskExceptionsWaitForOtherClaimedWork) {
    // Exercise an exception on an autil worker and bad_alloc on the caller.
    for (bool fail_on_caller : {false, true}) {
        std::mutex mutex;
        std::condition_variable changed;
        bool sibling_started = false;
        bool failure_ready = false;
        bool release_sibling = false;
        std::promise<ErrorCode> completion;
        auto result = completion.get_future();
        std::thread runner([&] {
            const auto caller_id = std::this_thread::get_id();
            completion.set_value(RunBoundedParallel(10, 2, [&](size_t) -> ErrorCode {
                std::unique_lock<std::mutex> lock(mutex);
                if ((std::this_thread::get_id() == caller_id) == fail_on_caller) {
                    changed.wait(lock, [&] { return sibling_started || release_sibling; });
                    failure_ready = true;
                    changed.notify_all();
                    if (fail_on_caller) {
                        throw std::bad_alloc();
                    }
                    throw std::runtime_error("injected task failure");
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
}

TEST(BoundedParallelTest, ThreadFactoryExceptionWaitsForStartedWorker) {
    std::mutex mutex;
    std::condition_variable changed;
    bool worker_started = false;
    bool creation_failed = false;
    bool release_worker = false;
    size_t factory_calls = 0;
    size_t task_calls = 0;
    std::promise<ErrorCode> completion;
    auto result = completion.get_future();
    std::thread runner([&] {
        completion.set_value(detail::RunBoundedParallelWithFactory(
            10,
            3,
            [&](size_t) {
                std::unique_lock<std::mutex> lock(mutex);
                ++task_calls;
                worker_started = true;
                changed.notify_all();
                changed.wait(lock, [&] { return release_worker; });
                return EC_OK;
            },
            [&](const auto &work) -> autil::ThreadPtr {
                if (++factory_calls == 1) {
                    return autil::Thread::createThread(work, "kvcm_test");
                }
                std::unique_lock<std::mutex> lock(mutex);
                changed.wait(lock, [&] { return worker_started || release_worker; });
                creation_failed = true;
                changed.notify_all();
                throw std::bad_alloc();
            }));
    });
    bool saw_failure = false;
    {
        std::unique_lock<std::mutex> lock(mutex);
        saw_failure = changed.wait_for(lock, std::chrono::seconds(5), [&] { return creation_failed; });
    }
    const auto status = result.wait_for(std::chrono::milliseconds(30));
    {
        std::lock_guard<std::mutex> lock(mutex);
        release_worker = true;
        changed.notify_all();
    }
    runner.join();
    EXPECT_TRUE(saw_failure);
    EXPECT_EQ(std::future_status::timeout, status);
    EXPECT_EQ(EC_ERROR, result.get());
    EXPECT_EQ(2, factory_calls);
    EXPECT_EQ(1, task_calls);
}

TEST(BoundedParallelTest, WorkerStoragePreparationFailureReturnsError) {
    size_t calls = 0;
    EXPECT_EQ(EC_ERROR,
              RunBoundedParallel(std::numeric_limits<size_t>::max(), std::numeric_limits<size_t>::max(), [&](size_t) {
                  ++calls;
                  return EC_OK;
              }));
    EXPECT_EQ(0, calls);
}

TEST(BoundedParallelTest, ThreadFactoryPropagatesCallbackCopyException) {
    struct Callback {
        bool &fail_copy;
        int &calls;
        Callback(bool &fail_copy, int &calls) : fail_copy(fail_copy), calls(calls) {}
        Callback(const Callback &other) : fail_copy(other.fail_copy), calls(other.calls) {
            if (fail_copy) {
                throw std::runtime_error("injected callback copy failure");
            }
        }
        void operator()() const { ++calls; }
    };
    bool fail_copy = false;
    int calls = 0;
    std::function<void()> callback = Callback(fail_copy, calls);
    fail_copy = true;
    // With LeakSanitizer enabled, this also checks release of the wrapper argument.
    EXPECT_THROW(autil::Thread::createThread(callback, "kvcm_test"), std::runtime_error);
    callback();
    EXPECT_EQ(1, calls);
}

} // namespace kv_cache_manager
