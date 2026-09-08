#include <chrono>
#include <functional>
#include <future>
#include <mutex>
#include <thread>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/service/server.h"

using namespace kv_cache_manager;
using namespace std::chrono_literals;

namespace {

bool WaitUntil(const std::function<bool()> &predicate, std::chrono::steady_clock::duration timeout) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
        if (predicate()) {
            return true;
        }
        std::this_thread::sleep_for(1ms);
    }
    return predicate();
}

} // namespace

class ServerLifecycleTest : public TESTBASE {};

TEST_F(ServerLifecycleTest, ConcurrentRecoveryCancellationWaitsForMovedWorker) {
    Server server;
    std::promise<void> worker_started_promise;
    auto worker_started = worker_started_promise.get_future();
    std::promise<void> release_worker_promise;
    auto release_worker = release_worker_promise.get_future().share();

    server.kv_meta_recovery_thread_ = std::thread([&]() {
        worker_started_promise.set_value();
        release_worker.wait();
    });

    const auto worker_status = worker_started.wait_for(2s);
    if (worker_status != std::future_status::ready) {
        release_worker_promise.set_value();
        server.kv_meta_recovery_thread_.join();
        FAIL() << "recovery worker did not start";
    }

    std::thread first_canceller([&]() { server.CancelAndJoinKvMetaRecovery(); });
    const bool worker_moved = WaitUntil(
        [&]() {
            std::lock_guard<std::mutex> lock(server.kv_meta_recovery_mutex_);
            return !server.kv_meta_recovery_thread_.joinable();
        },
        2s);
    if (!worker_moved) {
        release_worker_promise.set_value();
        first_canceller.join();
        FAIL() << "first cancellation did not take ownership of the recovery worker";
    }

    // Ownership has moved out of kv_meta_recovery_thread_, but the first
    // canceller must retain the lifecycle lock until its local thread joins.
    const bool lifecycle_lock_was_available = server.kv_meta_recovery_join_mutex_.try_lock();
    if (lifecycle_lock_was_available) {
        server.kv_meta_recovery_join_mutex_.unlock();
    }

    const auto epoch_before_second = server.kv_meta_recovery_epoch_.load(std::memory_order_acquire);
    std::promise<void> second_returned_promise;
    auto second_returned = second_returned_promise.get_future();
    std::thread second_canceller([&]() {
        server.CancelAndJoinKvMetaRecovery();
        second_returned_promise.set_value();
    });
    const bool second_entered = WaitUntil(
        [&]() { return server.kv_meta_recovery_epoch_.load(std::memory_order_acquire) > epoch_before_second; }, 2s);
    const auto second_status_before_release = second_returned.wait_for(50ms);

    release_worker_promise.set_value();
    first_canceller.join();
    second_canceller.join();

    EXPECT_FALSE(lifecycle_lock_was_available);
    EXPECT_TRUE(second_entered);
    EXPECT_EQ(std::future_status::timeout, second_status_before_release)
        << "a concurrent cancellation returned while the moved recovery worker was still running";
    EXPECT_EQ(std::future_status::ready, second_returned.wait_for(0s));
}
