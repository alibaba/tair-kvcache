#include <atomic>
#include <chrono>
#include <future>
#include <memory>
#include <pthread.h>

#include "kv_cache_manager/common/client_pool.h"
#include "kv_cache_manager/common/unittest.h"

namespace {

struct UnlockGate {
    std::promise<void> reached;
    std::promise<void> resume;
    std::shared_future<void> resumed = resume.get_future().share();
    std::atomic<bool> timed_out{false};
};

// Only the acquiring thread arms this hook. The real lock is released before
// pausing, so a competing borrower can inspect and acquire the pool normally.
thread_local UnlockGate *pause_after_unlock = nullptr;
thread_local UnlockGate *arm_after_creation = nullptr;
thread_local std::promise<void> *observe_borrower_progress = nullptr;
thread_local int borrower_unlocks_remaining = 0;

} // namespace

extern "C" int __real_pthread_rwlock_unlock(pthread_rwlock_t *mutex);

extern "C" int __wrap_pthread_rwlock_unlock(pthread_rwlock_t *mutex) {
    const int result = __real_pthread_rwlock_unlock(mutex);
    if (observe_borrower_progress && --borrower_unlocks_remaining == 0) {
        observe_borrower_progress->set_value();
        observe_borrower_progress = nullptr;
    }
    if (auto *gate = pause_after_unlock) {
        pause_after_unlock = nullptr;
        gate->reached.set_value();
        if (gate->resumed.wait_for(std::chrono::seconds(10)) != std::future_status::ready) {
            gate->timed_out.store(true);
        }
    }
    return result;
}

namespace kv_cache_manager {

class ClientPoolConcurrencyTest : public TESTBASE {};

namespace {

void CheckAcquisitionWhilePeerKeepsItsClient(bool pause_new_client) {
    using Pool = DynamicClientPool<int>;
    using Handle = Pool::ClientHandle;
    UnlockGate gate;
    auto reached = gate.reached.get_future();
    std::atomic<int> created{0};
    Pool pool(
        [&] {
            auto client = std::make_shared<int>(++created);
            if (arm_after_creation) {
                // The next shared-mutex unlock publishes the pool's ownership
                // record. Older code also published this client as free here.
                pause_after_unlock = arm_after_creation;
                arm_after_creation = nullptr;
            }
            return client;
        },
        pause_new_client ? 0 : 1,
        2);
    ASSERT_TRUE(pool.Initialize());

    auto acquiring = std::async(std::launch::async, [&] {
        if (pause_new_client) {
            arm_after_creation = &gate;
        } else {
            // Old code pauses after merely observing a free client. Correct
            // code has already removed that client from the free queue.
            pause_after_unlock = &gate;
        }
        return std::make_unique<Handle>(pool.AcquireClient(5000));
    });
    EXPECT_EQ(reached.wait_for(std::chrono::seconds(2)), std::future_status::ready);

    // The future retains the borrowed handle until explicitly reset below.
    std::promise<void> borrower_progress;
    auto borrower_progress_future = borrower_progress.get_future();
    auto borrowing = std::async(std::launch::async, [&] {
        if (pause_new_client) {
            // In the old path these two unlocks finish FreeClientSize and
            // actually take the published client. In the fixed path they
            // observe an empty queue before waiting for A's creation mutex.
            observe_borrower_progress = &borrower_progress;
            borrower_unlocks_remaining = 2;
        }
        return std::make_unique<Handle>(pool.AcquireClient(5000));
    });
    if (pause_new_client) {
        EXPECT_EQ(borrower_progress_future.wait_for(std::chrono::seconds(2)), std::future_status::ready);
    } else {
        EXPECT_EQ(borrowing.wait_for(std::chrono::seconds(2)), std::future_status::ready);
    }
    gate.resume.set_value();
    EXPECT_EQ(acquiring.wait_for(std::chrono::milliseconds(300)), std::future_status::ready)
        << "acquisition waited for a borrowed client despite spare capacity";

    // Do not ASSERT after threads start: release B even on regression, then join
    // A. All injected waits are bounded and both acquisitions have a timeout.
    EXPECT_EQ(borrowing.wait_for(std::chrono::seconds(2)), std::future_status::ready);
    auto borrowed = borrowing.get();
    EXPECT_TRUE(*borrowed);
    const int borrowed_id = *borrowed ? **borrowed : -1;
    EXPECT_EQ(created.load(), 2);
    borrowed.reset();
    EXPECT_EQ(acquiring.wait_for(std::chrono::seconds(2)), std::future_status::ready);
    auto acquired = acquiring.get();
    EXPECT_TRUE(*acquired);
    if (*acquired) {
        EXPECT_NE(**acquired, borrowed_id);
    }
    EXPECT_FALSE(gate.timed_out.load());
}

} // namespace

TEST_F(ClientPoolConcurrencyTest, CompetingBorrowerCannotTurnFreeClientObservationIntoTimedWait) {
    CheckAcquisitionWhilePeerKeepsItsClient(false);
}

TEST_F(ClientPoolConcurrencyTest, NewlyCreatedClientRemainsOwnedByItsAcquiringCaller) {
    CheckAcquisitionWhilePeerKeepsItsClient(true);
}

} // namespace kv_cache_manager
