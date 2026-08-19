// Runtime tests: the executor never lets a network wait occupy a worker, the
// timer is a pure clock, and the two admission lanes cannot starve each other.
#include <atomic>
#include <gtest/gtest.h>
#include <thread>
#include <vector>

#include "async_simple/coro/Collect.h"
#include "async_simple/coro/SyncAwait.h"
#include "tools/kvcm_swarm/evidence/observation.h"
#include "tools/kvcm_swarm/runtime/admission.h"
#include "tools/kvcm_swarm/runtime/executor.h"
#include "tools/kvcm_swarm/runtime/rng.h"
#include "tools/kvcm_swarm/runtime/sample_spec.h"
#include "tools/kvcm_swarm/runtime/stop_token.h"
#include "tools/kvcm_swarm/transport/transport.h"

namespace kvcm_swarm {
namespace {

template <typename T>
T RunSync(SwarmExecutor &executor, Task<T> task) {
    return async_simple::coro::syncAwait(std::move(task).via(&executor));
}

TEST(StopTokenTest, CallbackFiresOnceAndUnregisters) {
    StopSource source;
    int fired = 0;
    {
        StopCallbackGuard guard(source.Token(), [&fired]() { ++fired; });
        EXPECT_EQ(fired, 0);
    }
    source.RequestStop();
    EXPECT_EQ(fired, 0) << "an unregistered callback must not fire";

    StopSource second;
    second.RequestStop();
    int late = 0;
    StopCallbackGuard guard(second.Token(), [&late]() { ++late; });
    EXPECT_EQ(late, 1) << "registering after the stop runs the callback inline";
}

TEST(ExecutorTest, SleepUntilHonoursDeadlineAndCancellation) {
    SwarmExecutor executor(2);
    const TimePoint start = Now();
    EXPECT_TRUE(RunSync(executor, SleepFor(executor, std::chrono::milliseconds(30), StopToken())));
    EXPECT_GE(Now() - start, std::chrono::milliseconds(25));

    StopSource source;
    std::thread stopper([&source]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        source.RequestStop();
    });
    const TimePoint cancel_start = Now();
    EXPECT_FALSE(RunSync(executor, SleepFor(executor, std::chrono::seconds(30), source.Token())));
    EXPECT_LT(Now() - cancel_start, std::chrono::seconds(5));
    stopper.join();
    executor.Shutdown();
}

// A batch of slow "in-flight RPCs" must not stop unrelated operations from
// making progress, even with far fewer workers than pending operations.
TEST(ExecutorTest, ManySlowWaitsDoNotOccupyWorkers) {
    constexpr int kWorkers = 2;
    constexpr int kPending = 200;
    SwarmExecutor executor(kWorkers);
    std::vector<std::shared_ptr<AsyncSlot<bool>>> slots;
    std::atomic<int> finished{0};
    for (int i = 0; i < kPending; ++i) {
        auto slot = std::make_shared<AsyncSlot<bool>>(executor);
        slots.push_back(slot);
        auto waiter = [](std::shared_ptr<AsyncSlot<bool>> pending, std::atomic<int> *counter) -> Task<> {
            co_await *pending;
            counter->fetch_add(1);
            co_return;
        };
        waiter(slot, &finished).via(&executor).start([](auto &&) {});
    }

    // Unrelated timer-driven work must still advance while all of the above are
    // suspended on their "network" waits.
    std::atomic<int> ticks{0};
    for (int i = 0; i < 10; ++i) {
        executor.ScheduleAt(Now() + std::chrono::milliseconds(5 * (i + 1)), [&ticks]() { ticks.fetch_add(1); });
    }
    const TimePoint deadline = Now() + std::chrono::seconds(5);
    while (ticks.load() < 10 && Now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    EXPECT_EQ(ticks.load(), 10) << "pending network waits must not consume Executor workers";
    EXPECT_EQ(finished.load(), 0);

    for (auto &slot : slots) {
        slot->Complete(true);
    }
    const TimePoint drain_deadline = Now() + std::chrono::seconds(10);
    while (finished.load() < kPending && Now() < drain_deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    EXPECT_EQ(finished.load(), kPending);
    executor.Shutdown();
}

TEST(ExecutorTest, LateCompletionOfAnAlreadyCompletedSlotIsIgnored) {
    SwarmExecutor executor(1);
    AsyncSlot<bool> slot(executor);
    EXPECT_TRUE(slot.Complete(true));
    EXPECT_FALSE(slot.Complete(false)) << "only one result may complete an operation";
    EXPECT_TRUE(RunSync(executor, [](AsyncSlot<bool> &target) -> Task<bool> { co_return co_await target; }(slot)));
    executor.Shutdown();
}

TEST(AdmissionTest, BusinessSaturationDoesNotStarveControl) {
    SwarmExecutor executor(2);
    AdmissionLimits limits;
    limits.max_in_flight_business_rpcs = 2;
    limits.max_in_flight_control_rpcs = 2;
    AdmissionController admission(executor, limits);

    Permit first = admission.TryAcquire(TrafficLane::kBusiness);
    Permit second = admission.TryAcquire(TrafficLane::kBusiness);
    ASSERT_TRUE(first.valid());
    ASSERT_TRUE(second.valid());
    EXPECT_FALSE(admission.TryAcquire(TrafficLane::kBusiness).valid());

    // Control has its own reserved capacity.
    Permit control =
        RunSync(executor, admission.Acquire(TrafficLane::kControl, Now() + std::chrono::seconds(1), StopToken()));
    EXPECT_TRUE(control.valid());

    // A business waiter suspends asynchronously and is granted on release.
    std::atomic<bool> granted{false};
    auto waiter = [](AdmissionController *controller, std::atomic<bool> *flag) -> Task<> {
        Permit permit =
            co_await controller->Acquire(TrafficLane::kBusiness, Now() + std::chrono::seconds(5), StopToken());
        flag->store(permit.valid());
        co_return;
    };
    waiter(&admission, &granted).via(&executor).start([](auto &&) {});
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    EXPECT_FALSE(granted.load());
    first.Release();
    const TimePoint deadline = Now() + std::chrono::seconds(3);
    while (!granted.load() && Now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    EXPECT_TRUE(granted.load());
    const LaneStats business = admission.Snapshot(TrafficLane::kBusiness);
    EXPECT_EQ(business.waited, 1u);
    EXPECT_GT(business.wait_ns_total, 0u);
    executor.Shutdown();
}

TEST(AdmissionTest, PermitWaitTimeoutMarksGeneratorSaturated) {
    SwarmExecutor executor(2);
    AdmissionLimits limits;
    limits.max_in_flight_business_rpcs = 1;
    limits.max_in_flight_control_rpcs = 1;
    AdmissionController admission(executor, limits);
    Permit held = admission.TryAcquire(TrafficLane::kBusiness);
    ASSERT_TRUE(held.valid());
    Permit rejected = RunSync(
        executor, admission.Acquire(TrafficLane::kBusiness, Now() + std::chrono::milliseconds(30), StopToken()));
    EXPECT_FALSE(rejected.valid());
    EXPECT_TRUE(admission.saturated());
    EXPECT_EQ(admission.Snapshot(TrafficLane::kBusiness).rejected, 1u);
    executor.Shutdown();
}

TEST(AdmissionTest, ExplicitSaturationReasonsAreDeduplicated) {
    SwarmExecutor executor(1);
    AdmissionController admission(executor, AdmissionLimits{});
    admission.MarkSaturated("session_admission_rejected");
    admission.MarkSaturated("session_admission_rejected");
    admission.MarkSaturated("cache_backpressure");
    const auto reasons = admission.saturation_reasons();
    EXPECT_EQ(reasons.size(), 2u);
    EXPECT_EQ(admission.saturation_events(), 3u);
    executor.Shutdown();
}

TEST(AsyncCapacityBudgetTest, BoundsWeightedUsageAndReportsWaits) {
    SwarmExecutor executor(4);
    AsyncCapacityBudget budget(executor, 100);
    AsyncCapacityBudget::Guard sixty =
        RunSync(executor, budget.Acquire(60, Now() + std::chrono::seconds(1), StopToken()));
    AsyncCapacityBudget::Guard forty =
        RunSync(executor, budget.Acquire(40, Now() + std::chrono::seconds(1), StopToken()));
    ASSERT_TRUE(sixty.valid());
    ASSERT_TRUE(forty.valid());
    EXPECT_EQ(budget.in_use(), 100u);

    std::atomic<bool> acquired{false};
    auto waiter = [](AsyncCapacityBudget *target, std::atomic<bool> *done) -> Task<> {
        AsyncCapacityBudget::Guard guard = co_await target->Acquire(30, Now() + std::chrono::seconds(5), StopToken());
        if (!guard.valid()) {
            co_return;
        }
        done->store(true);
        co_return;
    };
    waiter(&budget, &acquired).via(&executor).start([](auto &&) {});
    const TimePoint queued_deadline = Now() + std::chrono::seconds(2);
    while (budget.waits() == 0 && Now() < queued_deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    EXPECT_FALSE(acquired.load());
    forty.Release();
    const TimePoint acquire_deadline = Now() + std::chrono::seconds(2);
    while (!acquired.load() && Now() < acquire_deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    EXPECT_TRUE(acquired.load());
    while (budget.in_use() != 60 && Now() < acquire_deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    EXPECT_EQ(budget.capacity(), 100u);
    EXPECT_EQ(budget.in_use(), 60u);
    EXPECT_EQ(budget.peak_in_use(), 100u);
    EXPECT_EQ(budget.timeouts(), 0u);
    EXPECT_GT(budget.waits(), 0u);
    sixty.Release();
    executor.Shutdown();
}

TEST(AsyncCapacityBudgetTest, DeadlineCancellationAndOversizeYieldAnInvalidGuard) {
    SwarmExecutor executor(2);
    AsyncCapacityBudget budget(executor, 10);
    StopSource stopped_before_acquire;
    stopped_before_acquire.RequestStop();
    AsyncCapacityBudget::Guard immediately_cancelled =
        RunSync(executor, budget.Acquire(1, Now() + std::chrono::seconds(1), stopped_before_acquire.Token()));
    EXPECT_FALSE(immediately_cancelled.valid());
    AsyncCapacityBudget::Guard expired =
        RunSync(executor, budget.Acquire(1, Now() - std::chrono::milliseconds(1), StopToken()));
    EXPECT_FALSE(expired.valid());

    AsyncCapacityBudget::Guard held =
        RunSync(executor, budget.Acquire(10, Now() + std::chrono::seconds(1), StopToken()));
    ASSERT_TRUE(held.valid());
    AsyncCapacityBudget::Guard timed_out =
        RunSync(executor, budget.Acquire(1, Now() + std::chrono::milliseconds(30), StopToken()));
    EXPECT_FALSE(timed_out.valid());
    EXPECT_EQ(budget.timeouts(), 3u);

    StopSource source;
    source.RequestStop();
    AsyncCapacityBudget::Guard cancelled =
        RunSync(executor, budget.Acquire(1, Now() + std::chrono::seconds(30), source.Token()));
    EXPECT_FALSE(cancelled.valid());
    AsyncCapacityBudget::Guard oversized =
        RunSync(executor, budget.Acquire(11, Now() + std::chrono::seconds(1), StopToken()));
    EXPECT_FALSE(oversized.valid());
    EXPECT_EQ(budget.timeouts(), 5u);
    executor.Shutdown();
}

TEST(RngTest, SeedDerivationIsStableAndIndependent) {
    SeedDeriver seeds(42);
    EXPECT_EQ(seeds.Derive("a"), seeds.Derive("a"));
    EXPECT_NE(seeds.Derive("a"), seeds.Derive("b"));
    EXPECT_NE(seeds.Derive("a", 1), seeds.Derive("a", 2));
    SeedDeriver other(43);
    EXPECT_NE(seeds.Derive("a"), other.Derive("a"));
}

TEST(RngTest, UniformRangeIncludesBothEndpoints) {
    Rng rng(7);
    bool saw_low = false;
    bool saw_high = false;
    for (int i = 0; i < 2000; ++i) {
        const uint64_t value = rng.NextInRange(3, 5);
        EXPECT_GE(value, 3u);
        EXPECT_LE(value, 5u);
        saw_low = saw_low || value == 3;
        saw_high = saw_high || value == 5;
    }
    EXPECT_TRUE(saw_low);
    EXPECT_TRUE(saw_high);
    EXPECT_EQ(rng.NextInRange(9, 9), 9u);
}

// The planned arrival timeline must be reproducible from the seed alone and
// must not depend on completion order. `even` uses a fixed gap, `poisson` an
// exponential gap, and both share the same long-run rate.
TEST(RngTest, ArrivalTimelinesAreReproducibleFromTheSeed) {
    const double rate = 25.0;
    auto even_timeline = [rate]() {
        std::vector<double> times;
        double now = 0.0;
        for (int i = 0; i < 10; ++i) {
            now += 1.0 / rate;
            times.push_back(now);
        }
        return times;
    };
    const std::vector<double> first_even = even_timeline();
    const std::vector<double> second_even = even_timeline();
    EXPECT_EQ(first_even, second_even);
    for (size_t i = 1; i < first_even.size(); ++i) {
        EXPECT_NEAR(first_even[i] - first_even[i - 1], 1.0 / rate, 1e-12);
    }

    auto poisson_timeline = [rate](uint64_t seed) {
        SeedDeriver seeds(seed);
        Rng rng = seeds.MakeRng("v6d/b/arrival");
        std::vector<double> times;
        double now = 0.0;
        for (int i = 0; i < 10; ++i) {
            now += rng.NextExponential(rate);
            times.push_back(now);
        }
        return times;
    };
    const std::vector<double> golden = poisson_timeline(20260819);
    EXPECT_EQ(golden, poisson_timeline(20260819)) << "same seed must give the same planned timeline";
    EXPECT_NE(golden, poisson_timeline(20260820));
    for (size_t i = 1; i < golden.size(); ++i) {
        EXPECT_GT(golden[i], golden[i - 1]) << "planned times advance monotonically";
    }
}

TEST(PhaseTest, SubmitTimeAttributionIsFixedByThePhaseSourceAtSubmit) {
    PhaseSource phase;
    EXPECT_EQ(phase.Current(), Phase::kValidate);
    phase.Set(Phase::kWarmup);
    const Phase captured = phase.Current();
    // A later transition must not rewrite an already captured attribution.
    phase.Set(Phase::kSteady);
    EXPECT_EQ(captured, Phase::kWarmup);
    EXPECT_EQ(phase.Current(), Phase::kSteady);
    EXPECT_STREQ(PhaseName(Phase::kDrain), "drain");
    EXPECT_STREQ(PhaseName(Phase::kPreflight), "preflight");
}

TEST(SampleSpecTest, ScalarNormalisesToEqualBounds) {
    IntSpec scalar(5);
    EXPECT_TRUE(scalar.IsScalar());
    Rng rng(1);
    EXPECT_EQ(Sample(scalar, rng), 5u);

    DurationSpec duration(std::chrono::milliseconds(10), std::chrono::milliseconds(20));
    for (int i = 0; i < 100; ++i) {
        const Duration value = Sample(duration, rng);
        EXPECT_GE(value, Duration(std::chrono::milliseconds(10)));
        EXPECT_LE(value, Duration(std::chrono::milliseconds(20)));
    }
}

TEST(RngTest, PoissonAndEvenArrivalsShareTheSameLongRunRate) {
    const double rate = 50.0;
    Rng rng(99);
    double total = 0.0;
    const int samples = 200000;
    for (int i = 0; i < samples; ++i) {
        total += rng.NextExponential(rate);
    }
    const double mean_gap = total / samples;
    EXPECT_NEAR(mean_gap, 1.0 / rate, 0.001);
}

} // namespace
} // namespace kvcm_swarm
