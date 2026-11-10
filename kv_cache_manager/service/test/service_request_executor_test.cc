#include <atomic>
#include <chrono>
#include <condition_variable>
#include <future>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <vector>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/service/service_request_executor.h"

namespace kv_cache_manager {
namespace {

using namespace std::chrono_literals;
using Executor = ServiceRequestExecutor;
using Token = Executor::CompletionToken;
using Result = Executor::SubmitResult;
using Mode = Executor::ShutdownMode;

class Gate {
public:
    void Enter() {
        std::unique_lock<std::mutex> lock(mutex_);
        ++arrivals_;
        changed_.notify_all();
        EXPECT_TRUE(changed_.wait_for(lock, 5s, [&] { return open_; }));
    }

    bool WaitForArrivals(std::size_t count) {
        std::unique_lock<std::mutex> lock(mutex_);
        return changed_.wait_for(lock, 2s, [&] { return arrivals_ >= count; });
    }

    void Open() {
        std::lock_guard<std::mutex> lock(mutex_);
        open_ = true;
        changed_.notify_all();
    }

private:
    std::mutex mutex_;
    std::condition_variable changed_;
    std::size_t arrivals_ = 0;
    bool open_ = false;
};

Executor::Request MakeTask(std::function<void(Token)> run) {
    return {std::move(run), [](Token) { ADD_FAILURE() << "unexpected task discard"; }};
}

TEST(ServiceRequestExecutorTest, EarlyRejectionCountsOnceAndDoesNotReserveQueueSlots) {
    Gate gate;
    Executor executor(1, 1);
    EXPECT_FALSE(executor.RejectIfUnavailable());
    EXPECT_EQ(Result::kAccepted, executor.TrySubmit(MakeTask([&](Token) { gate.Enter(); })));
    EXPECT_TRUE(gate.WaitForArrivals(1));
    EXPECT_FALSE(executor.RejectIfUnavailable());
    EXPECT_FALSE(executor.RejectIfUnavailable());
    EXPECT_EQ(Result::kAccepted, executor.TrySubmit(MakeTask([](Token) {})));
    EXPECT_EQ(Result::kFull, executor.RejectIfUnavailable());
    EXPECT_EQ(1u, executor.GetSnapshot().rejected);
    executor.CloseAdmission(Mode::kDrain);
    EXPECT_EQ(Result::kClosed, executor.RejectIfUnavailable());
    EXPECT_EQ(2u, executor.GetSnapshot().rejected);
    gate.Open();
    executor.Drain();
}

TEST(ServiceRequestExecutorTest, QueueWaitEndsWhenWorkerClaimsTask) {
    Gate gate;
    Executor executor(1);
    EXPECT_EQ(0u, executor.GetSnapshot().queue_wait.count);
    EXPECT_EQ(Result::kAccepted, executor.TrySubmit(MakeTask([&](Token) { gate.Enter(); })));
    EXPECT_TRUE(gate.WaitForArrivals(1));
    const auto first = executor.GetSnapshot().queue_wait;
    EXPECT_EQ(1u, first.count);
    for (int i = 0; i < 2; ++i) {
        EXPECT_EQ(Result::kAccepted, executor.TrySubmit(MakeTask([](Token) {})));
    }
    std::this_thread::sleep_for(20ms);
    const auto blocked = executor.GetSnapshot().queue_wait;
    // The running task's business time and tasks not yet claimed add no samples.
    EXPECT_EQ(first.count, blocked.count);
    EXPECT_EQ(first.sum_us, blocked.sum_us);
    EXPECT_EQ(first.buckets, blocked.buckets);
    executor.CloseAdmission(Mode::kDrain);
    gate.Open();
    executor.Drain();
    const auto done = executor.GetSnapshot().queue_wait;
    EXPECT_EQ(3u, done.count);
    EXPECT_GE(done.sum_us - first.sum_us, 40000u);
    EXPECT_EQ(done.count, std::accumulate(done.buckets.begin(), done.buckets.end(), uint64_t{0}));
    for (std::size_t i = 0; i < Executor::kQueueWaitBoundsUs.size(); ++i) {
        if (Executor::kQueueWaitBoundsUs[i] <= 10000) {
            EXPECT_EQ(first.buckets[i], done.buckets[i]);
        }
    }
}

TEST(ServiceRequestExecutorTest, ReportingWindowsMeasureWaitExecutionAndQueueDepth) {
    Gate gate;
    Executor executor(1, 1);
    EXPECT_EQ(Result::kAccepted, executor.TrySubmit(MakeTask([&](Token) { gate.Enter(); })));
    EXPECT_TRUE(gate.WaitForArrivals(1));
    executor.TakeMetricsSnapshot(); // A running task completes in the next window.
    EXPECT_EQ(Result::kAccepted, executor.TrySubmit(MakeTask([](Token) {})));
    EXPECT_EQ(Result::kFull, executor.TrySubmit(MakeTask([](Token) {})));
    std::this_thread::sleep_for(20ms);
    const auto busy = executor.TakeMetricsSnapshot();
    EXPECT_EQ(1u, busy.queue_length_max);
    EXPECT_GT(busy.queue_length_avg, 0.0);
    EXPECT_LE(busy.queue_length_avg, 1.0);
    EXPECT_EQ(0u, busy.wait_time.count);
    EXPECT_EQ(0u, busy.execution_time.count);
    gate.Open();
    executor.CloseAdmission(Mode::kDrain);
    executor.Drain();
    const auto done = executor.TakeMetricsSnapshot();
    EXPECT_EQ(1u, done.wait_time.count);
    EXPECT_GE(done.wait_time.max_us, 20000u);
    EXPECT_DOUBLE_EQ(done.wait_time.max_us, done.wait_time.AvgUs());
    EXPECT_EQ(2u, done.execution_time.count);
    EXPECT_GE(done.execution_time.max_us, 20000u);
    EXPECT_GE(done.execution_time.max_us, done.execution_time.AvgUs());
    EXPECT_DOUBLE_EQ(static_cast<double>(done.execution_time.sum_us) / 2, done.execution_time.AvgUs());
    EXPECT_EQ(1u, done.rejected);
    EXPECT_EQ(2u, done.queue_wait.count);
    const auto idle = executor.TakeMetricsSnapshot();
    EXPECT_EQ(0u, idle.wait_time.count);
    EXPECT_EQ(0u, idle.execution_time.count);
    EXPECT_EQ(0u, idle.wait_time.max_us);
    EXPECT_DOUBLE_EQ(0, idle.wait_time.AvgUs());
    EXPECT_EQ(0u, idle.queue_length_max);
    EXPECT_DOUBLE_EQ(0, idle.queue_length_avg);
    EXPECT_EQ(done.queue_wait.count, idle.queue_wait.count);
    EXPECT_EQ(done.queue_wait.sum_us, idle.queue_wait.sum_us);
}

TEST(ServiceRequestExecutorTest, ExplicitZeroCapacityAcceptsMoreThanDefault) {
    Gate gate;
    Executor executor(1, 0);
    EXPECT_EQ(Result::kAccepted, executor.TrySubmit(MakeTask([&](Token) { gate.Enter(); })));
    EXPECT_TRUE(gate.WaitForArrivals(1));
    for (size_t i = 0; i < 2049; ++i) {
        EXPECT_EQ(Result::kAccepted, executor.TrySubmit(MakeTask([](Token) {})));
    }
    EXPECT_EQ(2049u, executor.GetSnapshot().queued);
    executor.CloseAdmission(Mode::kDrain);
    gate.Open();
    executor.Drain();
}

TEST(ServiceRequestExecutorTest, InvalidRequestsAndWorkerCountsFailBeforeAdmission) {
    EXPECT_THROW(Executor(0), std::invalid_argument);
    Executor executor(1);
    EXPECT_THROW(executor.TrySubmit({}), std::invalid_argument);
    EXPECT_THROW(executor.TrySubmit({[](Token) {}, {}}), std::invalid_argument);
    EXPECT_EQ(0u, executor.GetSnapshot().queued);
    EXPECT_EQ(0u, executor.GetSnapshot().completion_tokens);
}

TEST(ServiceRequestExecutorTest, WorkersShareQueueAndBoundBusinessConcurrency) {
    Gate gate;
    Executor executor(2);
    const auto caller = std::this_thread::get_id();
    std::atomic<std::size_t> active{0};
    std::atomic<std::size_t> maximum_active{0};
    std::atomic<std::size_t> completed{0};
    auto run = [&](bool block) {
        EXPECT_NE(caller, std::this_thread::get_id());
        const auto current = ++active;
        auto maximum = maximum_active.load();
        while (maximum < current && !maximum_active.compare_exchange_weak(maximum, current)) {}
        if (block) {
            gate.Enter();
        }
        --active;
        ++completed;
    };

    for (std::size_t i = 0; i < 2; ++i) {
        EXPECT_EQ(Result::kAccepted, executor.TrySubmit(MakeTask([&](Token) { run(true); })));
    }
    EXPECT_TRUE(gate.WaitForArrivals(2));
    for (std::size_t i = 0; i < 32; ++i) {
        EXPECT_EQ(Result::kAccepted, executor.TrySubmit(MakeTask([&](Token) { run(false); })));
    }
    EXPECT_EQ(2u, executor.GetSnapshot().running);
    EXPECT_EQ(32u, executor.GetSnapshot().queued);
    executor.CloseAdmission(Mode::kDrain);
    gate.Open();
    executor.Drain();
    executor.Join();
    EXPECT_EQ(34u, completed.load());
    EXPECT_EQ(2u, maximum_active.load());
}

TEST(ServiceRequestExecutorTest, IdleWorkerDrainsAllWaitingTasksWhileAnotherIsBlocked) {
    Gate gate;
    Executor executor(2);
    EXPECT_EQ(Result::kAccepted, executor.TrySubmit(MakeTask([&](Token) { gate.Enter(); })));
    EXPECT_TRUE(gate.WaitForArrivals(1));
    constexpr std::size_t kPending = 32;
    std::atomic<std::size_t> completed{0};
    std::promise<void> pending_done;
    for (std::size_t i = 0; i < kPending; ++i) {
        EXPECT_EQ(Result::kAccepted, executor.TrySubmit(MakeTask([&](Token) {
            if (++completed == kPending) {
                pending_done.set_value();
            }
        })));
    }
    // A fixed per-worker assignment would leave work behind the blocked worker.
    EXPECT_EQ(std::future_status::ready, pending_done.get_future().wait_for(2s));
    executor.CloseAdmission(Mode::kDrain);
    gate.Open();
    executor.Drain();
    executor.Join();
    EXPECT_EQ(kPending, completed.load());
}

TEST(ServiceRequestExecutorTest, DefaultCapacityIs2048AndDrainPreservesFifo) {
    Gate gate;
    Executor executor(1);
    std::vector<std::size_t> order;
    EXPECT_EQ(Result::kAccepted, executor.TrySubmit(MakeTask([&](Token) { gate.Enter(); })));
    EXPECT_TRUE(gate.WaitForArrivals(1));

    constexpr std::size_t kPending = 2048;
    for (std::size_t i = 0; i < kPending; ++i) {
        EXPECT_EQ(Result::kAccepted, executor.TrySubmit(MakeTask([&, i](Token) { order.push_back(i); })));
    }
    EXPECT_EQ(kPending, executor.GetSnapshot().queued);
    EXPECT_EQ(0u, executor.GetSnapshot().rejected);
    EXPECT_EQ(Result::kFull, executor.TrySubmit(MakeTask([](Token) { ADD_FAILURE() << "overflow request ran"; })));
    executor.CloseAdmission(Mode::kDrain);
    // Repeating close must not change the already selected drain policy.
    executor.CloseAdmission();
    gate.Open();
    executor.Drain();
    executor.Join();
    ASSERT_EQ(kPending, order.size());
    for (std::size_t i = 0; i < kPending; ++i) {
        EXPECT_EQ(i, order[i]);
    }
    EXPECT_EQ(0u, executor.GetSnapshot().discarded);
}

TEST(ServiceRequestExecutorTest, PositiveCapacityCountsOnlyWaitingTasks) {
    Gate gate;
    Executor executor(1, 2);
    std::atomic<std::size_t> completed{0};
    EXPECT_EQ(Result::kAccepted, executor.TrySubmit(MakeTask([&](Token) { gate.Enter(); })));
    EXPECT_TRUE(gate.WaitForArrivals(1));
    for (std::size_t i = 0; i < 2; ++i) {
        EXPECT_EQ(Result::kAccepted, executor.TrySubmit(MakeTask([&](Token) { ++completed; })));
    }
    const Executor::Request rejected = {[](Token) { ADD_FAILURE() << "rejected task ran"; },
                                        [](Token) { ADD_FAILURE() << "rejected task was discarded"; }};
    EXPECT_EQ(Result::kFull, executor.TrySubmit(rejected));
    EXPECT_EQ(2u, executor.GetSnapshot().queued);
    EXPECT_EQ(1u, executor.GetSnapshot().running);
    executor.CloseAdmission(Mode::kDrain);
    EXPECT_EQ(Result::kClosed, executor.TrySubmit(rejected));
    gate.Open();
    executor.Drain();
    executor.Join();
    EXPECT_EQ(2u, completed.load());
    EXPECT_EQ(2u, executor.GetSnapshot().rejected);
    EXPECT_EQ(3u, executor.GetSnapshot().queue_wait.count);
}

TEST(ServiceRequestExecutorTest, DefaultCloseDiscardsPendingOutsideQueueMutex) {
    Gate gate;
    Executor executor(1);
    std::atomic<std::size_t> running_completed{0};
    std::size_t discarded = 0;
    const auto coordinator = std::this_thread::get_id();
    EXPECT_EQ(Result::kAccepted, executor.TrySubmit(MakeTask([&](Token) {
        gate.Enter();
        ++running_completed;
    })));
    EXPECT_TRUE(gate.WaitForArrivals(1));
    for (std::size_t i = 0; i < 8; ++i) {
        EXPECT_EQ(Result::kAccepted,
                  executor.TrySubmit({[](Token) { ADD_FAILURE() << "pending task ran after discard"; },
                                      [&](Token) {
                                          EXPECT_EQ(coordinator, std::this_thread::get_id());
                                          EXPECT_EQ(0u, executor.GetSnapshot().queued);
                                          // Reentrant read/submit/close must not hold the queue mutex.
                                          EXPECT_EQ(Result::kClosed, executor.TrySubmit(MakeTask([](Token) {})));
                                          executor.CloseAdmission(Mode::kDrain);
                                          ++discarded;
                                      }}));
    }
    executor.CloseAdmission();
    executor.CloseAdmission();
    EXPECT_EQ(8u, discarded);
    EXPECT_EQ(8u, executor.GetSnapshot().discarded);
    EXPECT_EQ(1u, executor.GetSnapshot().queue_wait.count);
    EXPECT_EQ(0u, executor.GetSnapshot().discarding);
    EXPECT_EQ(0u, running_completed.load());
    gate.Open();
    executor.Drain();
    executor.Join();
    executor.Join();
    EXPECT_EQ(1u, running_completed.load());
}

TEST(ServiceRequestExecutorTest, DrainWaitsForLastExternalCompletionToken) {
    Token retained;
    std::promise<void> callback_done;
    Executor executor(1);
    EXPECT_EQ(Result::kAccepted, executor.TrySubmit(MakeTask([&](Token token) {
        retained = std::move(token);
        callback_done.set_value();
    })));
    EXPECT_EQ(std::future_status::ready, callback_done.get_future().wait_for(2s));
    executor.CloseAdmission(Mode::kDrain);
    executor.Join();
    EXPECT_EQ(0u, executor.GetSnapshot().running);
    EXPECT_EQ(1u, executor.GetSnapshot().completion_tokens);

    auto copy = retained;
    auto draining = std::async(std::launch::async, [&] { executor.Drain(); });
    EXPECT_EQ(std::future_status::timeout, draining.wait_for(20ms));
    retained = {};
    EXPECT_EQ(std::future_status::timeout, draining.wait_for(20ms));
    copy = {};
    EXPECT_EQ(std::future_status::ready, draining.wait_for(2s));
    EXPECT_EQ(0u, executor.GetSnapshot().completion_tokens);
}

TEST(ServiceRequestExecutorTest, DiscardedHttpCompletionCanOutliveDiscardCallback) {
    Gate gate;
    Token retained;
    Executor executor(1);
    EXPECT_EQ(Result::kAccepted, executor.TrySubmit(MakeTask([&](Token) { gate.Enter(); })));
    EXPECT_TRUE(gate.WaitForArrivals(1));
    EXPECT_EQ(Result::kAccepted,
              executor.TrySubmit({[](Token) { ADD_FAILURE() << "discarded task ran"; },
                                  [&](Token token) { retained = std::move(token); }}));
    executor.CloseAdmission();
    gate.Open();
    executor.Join();
    EXPECT_EQ(0u, executor.GetSnapshot().discarding);
    EXPECT_EQ(1u, executor.GetSnapshot().completion_tokens);
    auto draining = std::async(std::launch::async, [&] { executor.Drain(); });
    EXPECT_EQ(std::future_status::timeout, draining.wait_for(20ms));
    retained = {};
    EXPECT_EQ(std::future_status::ready, draining.wait_for(2s));
}

TEST(ServiceRequestExecutorTest, ConcurrentCloseAndDequeueChooseExactlyOneCallback) {
    constexpr std::size_t kTasks = 64;
    for (std::size_t round = 0; round < 16; ++round) {
        Gate gate;
        std::vector<std::atomic<std::size_t>> visits(kTasks);
        for (auto &visit : visits) {
            visit.store(0);
        }
        Executor executor(2);
        for (std::size_t i = 0; i < 2; ++i) {
            EXPECT_EQ(Result::kAccepted, executor.TrySubmit(MakeTask([&](Token) { gate.Enter(); })));
        }
        EXPECT_TRUE(gate.WaitForArrivals(2));
        for (std::size_t i = 0; i < kTasks; ++i) {
            EXPECT_EQ(Result::kAccepted,
                      executor.TrySubmit({[&, i](Token) { ++visits[i]; }, [&, i](Token) { ++visits[i]; }}));
        }
        std::promise<void> start;
        auto ready = start.get_future().share();
        std::thread closer([&] {
            ready.wait();
            executor.CloseAdmission();
        });
        std::thread opener([&] {
            ready.wait();
            gate.Open();
        });
        start.set_value();
        closer.join();
        opener.join();
        executor.Drain();
        executor.Join();
        const auto snapshot = executor.GetSnapshot();
        EXPECT_EQ(kTasks + 2, snapshot.queue_wait.count + snapshot.discarded);
        EXPECT_EQ(snapshot.queue_wait.count,
                  std::accumulate(snapshot.queue_wait.buckets.begin(), snapshot.queue_wait.buckets.end(), uint64_t{0}));
        for (const auto &visit : visits) {
            EXPECT_EQ(1u, visit.load()) << "round=" << round;
        }
    }
}

TEST(ServiceRequestExecutorTest, DrainIncludesInProgressDiscardCallback) {
    Gate running_gate;
    Gate discard_gate;
    Executor executor(1);
    EXPECT_EQ(Result::kAccepted, executor.TrySubmit(MakeTask([&](Token) { running_gate.Enter(); })));
    EXPECT_TRUE(running_gate.WaitForArrivals(1));
    EXPECT_EQ(Result::kAccepted,
              executor.TrySubmit({[](Token) { ADD_FAILURE() << "discarded task ran"; },
                                  [&](Token token) {
                                      token = {};
                                      discard_gate.Enter();
                                  }}));
    std::thread closer([&] { executor.CloseAdmission(); });
    EXPECT_TRUE(discard_gate.WaitForArrivals(1));
    running_gate.Open();
    executor.Join();
    EXPECT_EQ(0u, executor.GetSnapshot().completion_tokens);
    EXPECT_EQ(1u, executor.GetSnapshot().discarding);
    auto draining = std::async(std::launch::async, [&] { executor.Drain(); });
    EXPECT_EQ(std::future_status::timeout, draining.wait_for(20ms));
    discard_gate.Open();
    closer.join();
    EXPECT_EQ(std::future_status::ready, draining.wait_for(2s));
}

TEST(ServiceRequestExecutorTest, EscapedCallbackExceptionDoesNotLoseOtherTasksOrDrainAccounting) {
    Gate gate;
    Executor executor(1);
    EXPECT_EQ(Result::kAccepted, executor.TrySubmit(MakeTask([&](Token) {
        gate.Enter();
        throw std::runtime_error("expected run exception");
    })));
    EXPECT_TRUE(gate.WaitForArrivals(1));
    std::size_t discarded = 0;
    EXPECT_EQ(Result::kAccepted,
              executor.TrySubmit({[](Token) { ADD_FAILURE(); },
                                  [](Token) { throw std::runtime_error("expected discard exception"); }}));
    EXPECT_EQ(Result::kAccepted, executor.TrySubmit({[](Token) { ADD_FAILURE(); }, [&](Token) { ++discarded; }}));
    executor.CloseAdmission();
    gate.Open();
    executor.Drain();
    executor.Join();
    EXPECT_EQ(1u, discarded);
    EXPECT_EQ(0u, executor.GetSnapshot().completion_tokens);
}

} // namespace
} // namespace kv_cache_manager
