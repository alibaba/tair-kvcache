#include <atomic>
#include <chrono>
#include <future>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

#include "kv_cache_manager/common/lock_free_thread_pool.h"
#include "kv_cache_manager/common/unittest.h"

using namespace kv_cache_manager;
using namespace std::chrono_literals;

namespace {

// Declare after the pool so an early assertion still releases its running task.
class TaskGate {
public:
    TaskGate() : future_(promise_.get_future().share()) {}
    ~TaskGate() { Open(); }
    void Wait() const { future_.wait(); }
    void Open() {
        std::call_once(opened_, [this]() { promise_.set_value(); });
    }

private:
    std::promise<void> promise_;
    std::shared_future<void> future_;
    std::once_flag opened_;
};

} // namespace

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

TEST_F(LockFreeThreadPoolTest, AsyncInvokesStoredCallableAsLvalue) {
    struct LvalueCallable {
        int operator()() & { return 21; }
    };
    LockFreeThreadPool pool(1, 1, "lvalue_callable");
    ASSERT_TRUE(pool.start());
    EXPECT_EQ(21, pool.async(LvalueCallable{}).get());
    LvalueCallable callable;
    EXPECT_EQ(21, pool.async(callable).get());
    pool.stop();
    EXPECT_EQ(21, pool.async(LvalueCallable{}).get());
}

TEST_F(LockFreeThreadPoolTest, RejectedAsyncPreservesCallerExecution) {
    LockFreeThreadPool pool(1, 1, "rejected");
    const auto caller = std::this_thread::get_id();
    std::atomic<size_t> rejected_calls{0};
    EXPECT_TRUE(pool.isFull());
    EXPECT_FALSE(pool.trySchedule([&]() { ++rejected_calls; }));
    EXPECT_FALSE(pool.enqueue([&]() { ++rejected_calls; }));
    EXPECT_EQ(caller, pool.async([]() { return std::this_thread::get_id(); }).get());
    ASSERT_TRUE(pool.start());
    EXPECT_FALSE(pool.start());
    pool.stop();
    pool.stop();
    EXPECT_FALSE(pool.start());
    EXPECT_FALSE(pool.trySchedule([&]() { ++rejected_calls; }));
    EXPECT_FALSE(pool.enqueue([&]() { ++rejected_calls; }));
    EXPECT_EQ(caller, pool.async([]() { return std::this_thread::get_id(); }).get());
    EXPECT_EQ(0u, rejected_calls.load());
}

TEST_F(LockFreeThreadPoolTest, TryScheduleRejectsFullQueueWithoutExecuting) {
    LockFreeThreadPool pool(1, 1, "bounded");
    TaskGate gate;
    ASSERT_TRUE(pool.start());
    std::promise<void> running;
    auto started = running.get_future();
    ASSERT_TRUE(pool.trySchedule([&]() {
        running.set_value();
        gate.Wait();
    }));
    EXPECT_EQ(std::future_status::ready, started.wait_for(1s));
    std::atomic<size_t> queued_calls{0}, rejected_calls{0};
    EXPECT_TRUE(pool.trySchedule([&]() { ++queued_calls; }));
    EXPECT_TRUE(pool.isFull());
    EXPECT_FALSE(pool.trySchedule([&]() { ++rejected_calls; }));
    EXPECT_FALSE(pool.trySchedule({}));
    gate.Open();
    pool.waitFinish();
    EXPECT_EQ(1u, queued_calls.load());
    EXPECT_EQ(0u, rejected_calls.load());
    EXPECT_FALSE(pool.isFull());
    EXPECT_EQ(3, pool.async([]() { return 3; }).get());
}

TEST_F(LockFreeThreadPoolTest, StopClosesAdmissionAndDrainsQueuedTasks) {
    LockFreeThreadPool pool(1, 8, "drain");
    TaskGate gate;
    ASSERT_TRUE(pool.start());
    std::promise<void> running;
    auto started = running.get_future();
    auto active = pool.async([&]() {
        running.set_value();
        gate.Wait();
        return 1;
    });
    EXPECT_EQ(std::future_status::ready, started.wait_for(1s));
    auto queued = pool.async([]() { return 2; });
    std::promise<void> stopped;
    auto finished = stopped.get_future();
    std::thread stopper([&]() {
        pool.stop();
        stopped.set_value();
    });
    const auto deadline = std::chrono::steady_clock::now() + 1s;
    while (!pool.isFull() && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::yield();
    }
    EXPECT_TRUE(pool.isFull());
    EXPECT_FALSE(pool.trySchedule([]() {}));
    EXPECT_EQ(std::future_status::timeout, finished.wait_for(20ms));
    gate.Open();
    stopper.join();
    EXPECT_EQ(1, active.get());
    ASSERT_EQ(std::future_status::ready, queued.wait_for(1s));
    EXPECT_EQ(2, queued.get());
}

TEST_F(LockFreeThreadPoolTest, DestructorDrainsAcceptedTasks) {
    auto pool = std::make_unique<LockFreeThreadPool>(1, 8, "destructor");
    TaskGate gate;
    ASSERT_TRUE(pool->start());
    std::promise<void> running;
    auto started = running.get_future();
    auto active = pool->async([&]() {
        running.set_value();
        gate.Wait();
    });
    EXPECT_EQ(std::future_status::ready, started.wait_for(1s));
    auto queued = pool->async([]() { return 5; });
    auto destroying = std::async(std::launch::async, [&]() { pool.reset(); });
    EXPECT_EQ(std::future_status::timeout, destroying.wait_for(20ms));
    gate.Open();
    destroying.get();
    active.get();
    ASSERT_EQ(std::future_status::ready, queued.wait_for(1s));
    EXPECT_EQ(5, queued.get());
}

TEST_F(LockFreeThreadPoolTest, BlockingSubmissionCanFinishWhileStopDrains) {
    LockFreeThreadPool pool(1, 1, "blocking");
    TaskGate gate;
    ASSERT_TRUE(pool.start());
    std::promise<void> running;
    auto started = running.get_future();
    auto active = pool.async([&]() {
        running.set_value();
        gate.Wait();
    });
    EXPECT_EQ(std::future_status::ready, started.wait_for(1s));
    auto producer = std::async(std::launch::async, [&]() {
        std::vector<std::future<int>> results;
        for (int value = 1; value <= 4; ++value) {
            results.push_back(pool.async([value]() { return value; }));
        }
        return results;
    });
    EXPECT_EQ(std::future_status::timeout, producer.wait_for(20ms));
    auto stopping = std::async(std::launch::async, [&]() { pool.stop(); });
    EXPECT_EQ(std::future_status::timeout, stopping.wait_for(20ms));
    gate.Open();
    auto results = producer.get();
    stopping.get();
    active.get();
    ASSERT_EQ(4u, results.size());
    for (size_t index = 0; index < results.size(); ++index) {
        EXPECT_EQ(index + 1, results[index].get());
    }
}

TEST_F(LockFreeThreadPoolTest, FireAndForgetExceptionsDoNotStrandAcceptedWork) {
    LockFreeThreadPool pool(1, 8, "exceptions", true);
    ASSERT_TRUE(pool.start());
    EXPECT_TRUE(pool.trySchedule([]() { throw std::runtime_error("expected"); }));
    EXPECT_TRUE(pool.trySchedule([]() { throw 1; }));
    std::atomic<bool> finished{false};
    EXPECT_TRUE(pool.trySchedule([&]() { finished = true; }));
    pool.waitFinish();
    EXPECT_TRUE(finished.load());
    EXPECT_EQ(6, pool.async([]() { return 6; }).get());
    pool.stop();
}

TEST_F(LockFreeThreadPoolTest, EnqueueWaitsForCapacityWithoutCallerExecution) {
    LockFreeThreadPool pool(1, 1, "enqueue");
    TaskGate gate;
    ASSERT_TRUE(pool.start());
    std::promise<void> running;
    auto started = running.get_future();
    ASSERT_TRUE(pool.enqueue([&]() {
        running.set_value();
        gate.Wait();
    }));
    EXPECT_EQ(std::future_status::ready, started.wait_for(1s));
    std::atomic<size_t> calls{0};
    EXPECT_TRUE(pool.enqueue([&]() { ++calls; }));
    EXPECT_TRUE(pool.isFull());
    std::thread::id producer_id, worker_id;
    auto producer = std::async(std::launch::async, [&]() {
        producer_id = std::this_thread::get_id();
        return pool.enqueue([&]() {
            worker_id = std::this_thread::get_id();
            ++calls;
        });
    });
    EXPECT_EQ(std::future_status::timeout, producer.wait_for(20ms));
    gate.Open();
    EXPECT_TRUE(producer.get());
    pool.waitFinish();
    EXPECT_EQ(2u, calls.load());
    EXPECT_NE(producer_id, worker_id);
    EXPECT_FALSE(pool.enqueue({}));
}

TEST_F(LockFreeThreadPoolTest, StopWakesBlockedProducerAndDrainsAcceptedTasks) {
    LockFreeThreadPool pool(1, 1, "stop_enqueue");
    TaskGate gate;
    ASSERT_TRUE(pool.start());
    std::promise<void> running;
    auto started = running.get_future();
    ASSERT_TRUE(pool.enqueue([&]() {
        running.set_value();
        gate.Wait();
    }));
    EXPECT_EQ(std::future_status::ready, started.wait_for(1s));
    std::atomic<size_t> accepted_calls{0}, rejected_calls{0};
    EXPECT_TRUE(pool.enqueue([&]() { ++accepted_calls; }));
    auto producer = std::async(std::launch::async, [&]() { return pool.enqueue([&]() { ++rejected_calls; }); });
    EXPECT_EQ(std::future_status::timeout, producer.wait_for(20ms));
    auto stopping = std::async(std::launch::async, [&]() { pool.stop(); });
    const auto producer_status = producer.wait_for(1s);
    EXPECT_EQ(std::future_status::ready, producer_status);
    EXPECT_EQ(std::future_status::timeout, stopping.wait_for(20ms));
    // Release the active task before collecting futures even if a check failed.
    gate.Open();
    EXPECT_FALSE(producer.get());
    stopping.get();
    EXPECT_EQ(1u, accepted_calls.load());
    EXPECT_EQ(0u, rejected_calls.load());
}

TEST_F(LockFreeThreadPoolTest, ConcurrentSubmittersExecuteEveryAcceptedTaskOnce) {
    LockFreeThreadPool pool(4, 2, "submitters");
    ASSERT_TRUE(pool.start());
    constexpr size_t kProducerCount = 12;
    constexpr size_t kTasksPerProducer = 80;
    std::vector<std::atomic<size_t>> calls(kProducerCount * kTasksPerProducer);
    for (auto &count : calls) {
        count = 0;
    }
    std::atomic<size_t> active{0}, maximum_active{0};
    std::vector<std::future<bool>> producers;
    for (size_t producer = 0; producer < kProducerCount; ++producer) {
        producers.push_back(std::async(std::launch::async, [&, producer]() {
            for (size_t i = 0; i < kTasksPerProducer; ++i) {
                const size_t index = producer * kTasksPerProducer + i;
                if (!pool.enqueue([&, index]() {
                        const size_t current = ++active;
                        size_t previous = maximum_active.load();
                        while (previous < current && !maximum_active.compare_exchange_weak(previous, current)) {}
                        ++calls[index];
                        std::this_thread::yield();
                        --active;
                    })) {
                    return false;
                }
            }
            return true;
        }));
    }
    for (auto &producer : producers) {
        EXPECT_TRUE(producer.get());
    }
    pool.stop();
    for (const auto &count : calls) {
        EXPECT_EQ(1u, count.load());
    }
    EXPECT_LE(maximum_active.load(), 4u);
    EXPECT_EQ(0u, active.load());
}

TEST_F(LockFreeThreadPoolTest, ConcurrentStopCallsJoinOnceAndWaitForAcceptedWork) {
    LockFreeThreadPool pool(1, 1, "stop_callers");
    TaskGate gate;
    ASSERT_TRUE(pool.start());
    std::promise<void> running;
    auto started = running.get_future();
    EXPECT_TRUE(pool.enqueue([&]() {
        running.set_value();
        gate.Wait();
    }));
    EXPECT_EQ(std::future_status::ready, started.wait_for(1s));
    std::atomic<size_t> completed{0};
    EXPECT_TRUE(pool.enqueue([&]() { ++completed; }));
    auto first = std::async(std::launch::async, [&]() { pool.stop(); });
    auto second = std::async(std::launch::async, [&]() { pool.stop(); });
    EXPECT_EQ(std::future_status::timeout, first.wait_for(20ms));
    EXPECT_EQ(std::future_status::timeout, second.wait_for(20ms));
    gate.Open();
    first.get();
    second.get();
    EXPECT_EQ(1u, completed.load());
}
