#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <set>
#include <stdexcept>
#include <thread>
#include <vector>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/meta/query_executor.h"

namespace kv_cache_manager {
namespace {

using namespace std::chrono_literals;

TEST(QueryExecutorTest, EmptyAndBelowThresholdCallsStayOnCaller) {
    QueryExecutor executor(/*worker_count*/ 4,
                           /*parallel_threshold*/ 10,
                           /*chunk_size*/ 2,
                           /*queue_capacity*/ 8);
    std::size_t call_count = 0;
    std::thread::id callback_thread;
    const std::thread::id caller_thread = std::this_thread::get_id();

    EXPECT_TRUE(executor.ParallelFor(0, [&](std::size_t, std::size_t) { ++call_count; }));
    EXPECT_EQ(0u, call_count);
    EXPECT_TRUE(executor.ParallelFor(9, [&](std::size_t begin, std::size_t end) {
        ++call_count;
        callback_thread = std::this_thread::get_id();
        EXPECT_EQ(0u, begin);
        EXPECT_EQ(9u, end);
    }));
    EXPECT_EQ(1u, call_count);
    EXPECT_EQ(caller_thread, callback_thread);
}

TEST(QueryExecutorTest, ParallelRangesCoverEveryIndexExactlyOnce) {
    QueryExecutor executor(/*worker_count*/ 4,
                           /*parallel_threshold*/ 1,
                           /*chunk_size*/ 1,
                           /*queue_capacity*/ 16);
    constexpr std::size_t kCount = 512;
    std::vector<std::atomic<uint32_t>> visits(kCount);
    for (auto &visit : visits) {
        visit.store(0, std::memory_order_relaxed);
    }

    std::mutex threads_mutex;
    std::condition_variable threads_cv;
    std::set<std::thread::id> callback_threads;
    ASSERT_TRUE(executor.ParallelFor(kCount, [&](std::size_t begin, std::size_t end) {
        {
            std::unique_lock<std::mutex> lock(threads_mutex);
            callback_threads.insert(std::this_thread::get_id());
            threads_cv.notify_all();
            if (callback_threads.size() == 1) {
                threads_cv.wait_for(lock, 2s, [&] { return callback_threads.size() >= 2; });
            }
        }
        for (std::size_t i = begin; i < end; ++i) {
            visits[i].fetch_add(1, std::memory_order_relaxed);
        }
    }));

    EXPECT_GE(callback_threads.size(), 2u);
    EXPECT_LE(callback_threads.size(), 4u);
    for (std::size_t i = 0; i < kCount; ++i) {
        EXPECT_EQ(1u, visits[i].load(std::memory_order_relaxed)) << "index=" << i;
    }
}

TEST(QueryExecutorTest, PerCallChunkSizeOverridesConfiguredRangeSize) {
    QueryExecutor executor(/*worker_count*/ 4,
                           /*parallel_threshold*/ 1,
                           /*chunk_size*/ 1,
                           /*queue_capacity*/ 16);
    constexpr std::size_t kCount = 100;
    constexpr std::size_t kChunkSize = 17;
    std::vector<std::atomic<uint32_t>> visits(kCount);
    for (auto &visit : visits) {
        visit.store(0, std::memory_order_relaxed);
    }
    std::mutex ranges_mutex;
    std::vector<std::pair<size_t, size_t>> ranges;
    ASSERT_TRUE(executor.ParallelForWithChunkSize(kCount, kChunkSize, [&](size_t begin, size_t end) {
        {
            std::lock_guard<std::mutex> lock(ranges_mutex);
            ranges.emplace_back(begin, end);
        }
        for (size_t i = begin; i < end; ++i) {
            visits[i].fetch_add(1, std::memory_order_relaxed);
        }
    }));

    EXPECT_EQ((kCount + kChunkSize - 1) / kChunkSize, ranges.size());
    for (const auto &[begin, end] : ranges) {
        EXPECT_EQ(0u, begin % kChunkSize);
        EXPECT_GT(end, begin);
        EXPECT_LE(end - begin, kChunkSize);
    }
    for (size_t i = 0; i < kCount; ++i) {
        EXPECT_EQ(1u, visits[i].load(std::memory_order_relaxed)) << "index=" << i;
    }
}

TEST(QueryExecutorTest, CallbackExceptionIsReportedWithoutDroppingOtherRanges) {
    QueryExecutor executor(/*worker_count*/ 4,
                           /*parallel_threshold*/ 1,
                           /*chunk_size*/ 1,
                           /*queue_capacity*/ 16);
    constexpr std::size_t kCount = 128;
    std::vector<std::atomic<uint32_t>> visits(kCount);
    for (auto &visit : visits) {
        visit.store(0, std::memory_order_relaxed);
    }

    EXPECT_FALSE(executor.ParallelFor(kCount, [&](std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; ++i) {
            visits[i].fetch_add(1, std::memory_order_relaxed);
            if (i == 37) {
                throw std::runtime_error("expected test exception");
            }
        }
    }));
    for (std::size_t i = 0; i < kCount; ++i) {
        EXPECT_EQ(1u, visits[i].load(std::memory_order_relaxed)) << "index=" << i;
    }
}

TEST(QueryExecutorTest, NestedParallelForDoesNotDeadlockOrDuplicateWork) {
    QueryExecutor executor(/*worker_count*/ 4,
                           /*parallel_threshold*/ 1,
                           /*chunk_size*/ 1,
                           /*queue_capacity*/ 16);
    constexpr std::size_t kOuterCount = 32;
    constexpr std::size_t kInnerCount = 8;
    std::vector<std::atomic<uint32_t>> visits(kOuterCount * kInnerCount);
    for (auto &visit : visits) {
        visit.store(0, std::memory_order_relaxed);
    }

    ASSERT_TRUE(executor.ParallelFor(kOuterCount, [&](std::size_t outer_begin, std::size_t outer_end) {
        for (std::size_t outer = outer_begin; outer < outer_end; ++outer) {
            ASSERT_TRUE(executor.ParallelFor(kInnerCount, [&](std::size_t inner_begin, std::size_t inner_end) {
                for (std::size_t inner = inner_begin; inner < inner_end; ++inner) {
                    visits[outer * kInnerCount + inner].fetch_add(1, std::memory_order_relaxed);
                }
            }));
        }
    }));
    for (std::size_t i = 0; i < visits.size(); ++i) {
        EXPECT_EQ(1u, visits[i].load(std::memory_order_relaxed)) << "index=" << i;
    }
}

TEST(QueryExecutorTest, CompletedCallerDoesNotWaitForQueuedHelper) {
    QueryExecutor executor(/*worker_count*/ 2,
                           /*parallel_threshold*/ 1,
                           /*chunk_size*/ 1,
                           /*queue_capacity*/ 8);
    std::mutex blocker_mutex;
    std::condition_variable blocker_cv;
    std::size_t blocked_callbacks = 0;
    bool release_blocker = false;

    std::thread blocker([&] {
        EXPECT_TRUE(executor.ParallelFor(2, [&](std::size_t, std::size_t) {
            std::unique_lock<std::mutex> lock(blocker_mutex);
            ++blocked_callbacks;
            blocker_cv.notify_all();
            blocker_cv.wait(lock, [&] { return release_blocker; });
        }));
    });
    bool both_callbacks_blocked = false;
    {
        std::unique_lock<std::mutex> lock(blocker_mutex);
        both_callbacks_blocked = blocker_cv.wait_for(lock, 2s, [&] { return blocked_callbacks == 2; });
    }
    if (!both_callbacks_blocked) {
        {
            std::lock_guard<std::mutex> lock(blocker_mutex);
            release_blocker = true;
        }
        blocker_cv.notify_all();
        blocker.join();
        FAIL() << "query executor background worker did not enter the blocking callback";
    }

    std::mutex fast_mutex;
    std::condition_variable fast_cv;
    bool fast_done = false;
    std::thread fast([&] {
        EXPECT_TRUE(executor.ParallelFor(2, [](std::size_t, std::size_t) {}));
        {
            std::lock_guard<std::mutex> lock(fast_mutex);
            fast_done = true;
        }
        fast_cv.notify_one();
    });
    {
        std::unique_lock<std::mutex> lock(fast_mutex);
        EXPECT_TRUE(fast_cv.wait_for(lock, 1s, [&] { return fast_done; }));
    }
    {
        std::lock_guard<std::mutex> lock(blocker_mutex);
        release_blocker = true;
    }
    blocker_cv.notify_all();
    fast.join();
    blocker.join();
}

TEST(QueryExecutorTest, ConcurrentRequestsRemainExactWithSaturatedQueue) {
    QueryExecutor executor(/*worker_count*/ 4,
                           /*parallel_threshold*/ 1,
                           /*chunk_size*/ 3,
                           /*queue_capacity*/ 1);
    constexpr std::size_t kThreadCount = 16;
    constexpr std::size_t kRounds = 20;
    constexpr std::size_t kCount = 257;
    std::atomic<bool> start{false};
    std::vector<std::thread> threads;
    threads.reserve(kThreadCount);
    for (std::size_t thread_index = 0; thread_index < kThreadCount; ++thread_index) {
        threads.emplace_back([&, thread_index] {
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            for (std::size_t round = 0; round < kRounds; ++round) {
                std::vector<std::atomic<uint32_t>> visits(kCount);
                for (auto &visit : visits) {
                    visit.store(0, std::memory_order_relaxed);
                }
                EXPECT_TRUE(executor.ParallelFor(kCount,
                                                 [&](std::size_t begin, std::size_t end) {
                                                     for (std::size_t i = begin; i < end; ++i) {
                                                         visits[i].fetch_add(1, std::memory_order_relaxed);
                                                     }
                                                 }))
                    << "thread=" << thread_index << " round=" << round;
                for (std::size_t i = 0; i < kCount; ++i) {
                    EXPECT_EQ(1u, visits[i].load(std::memory_order_relaxed))
                        << "thread=" << thread_index << " round=" << round << " index=" << i;
                }
            }
        });
    }
    start.store(true, std::memory_order_release);
    for (auto &thread : threads) {
        thread.join();
    }
}

} // namespace
} // namespace kv_cache_manager
