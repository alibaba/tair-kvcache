// LocalCache tests: byte accounting, leases, victim selection, asynchronous
// capacity waiting and the resident/evicting lifecycle.
#include <atomic>
#include <gtest/gtest.h>
#include <thread>

#include "async_simple/coro/SyncAwait.h"
#include "tools/kvcm_swarm/clients/v6d/local_cache.h"

namespace kvcm_swarm {
namespace {

GroupObject MakeObject(const std::string &key, uint64_t size) {
    GroupObject object;
    object.object_key = key;
    object.block_key = static_cast<int64_t>(HashString(key));
    object.object_size = size;
    object.spec_name = "v6d_" + std::to_string(size);
    return object;
}

template <typename T>
T RunSync(SwarmExecutor &executor, Task<T> task) {
    return async_simple::coro::syncAwait(std::move(task).via(&executor));
}

TEST(LocalCacheTest, InsertAcquireAndByteAccounting) {
    SwarmExecutor executor(2);
    LocalCache cache(executor, 1000);
    {
        LocalLease lease = RunSync(
            executor, cache.ReserveAndInsert(MakeObject("a", 400), Now() + std::chrono::seconds(1), StopToken()));
        ASSERT_TRUE(lease.valid());
        EXPECT_EQ(cache.used_bytes(), 400u);
        EXPECT_TRUE(cache.IsResident("a"));
        // A leased object can never be selected as a victim.
        EXPECT_TRUE(cache.SelectVictims(400, 8).empty());
    }
    EXPECT_EQ(cache.Stats().local_hits, 0u);
    auto hit = cache.Acquire("a");
    ASSERT_TRUE(hit.has_value());
    EXPECT_EQ(cache.Stats().local_hits, 1u);
    hit.reset();
    EXPECT_FALSE(cache.Acquire("missing").has_value());
    EXPECT_EQ(cache.Stats().local_misses, 1u);
    executor.Shutdown();
}

TEST(LocalCacheTest, OversizeObjectIsRejected) {
    SwarmExecutor executor(1);
    LocalCache cache(executor, 100);
    LocalLease lease =
        RunSync(executor, cache.ReserveAndInsert(MakeObject("big", 200), Now() + std::chrono::seconds(1), StopToken()));
    EXPECT_FALSE(lease.valid());
    EXPECT_EQ(cache.Stats().insert_rejected_oversize, 1u);
    executor.Shutdown();
}

TEST(LocalCacheTest, VictimsComeFromTheUnleasedLruTail) {
    SwarmExecutor executor(2);
    LocalCache cache(executor, 1000);
    for (const char *key : {"a", "b", "c"}) {
        LocalLease lease = RunSync(
            executor, cache.ReserveAndInsert(MakeObject(key, 200), Now() + std::chrono::seconds(1), StopToken()));
        ASSERT_TRUE(lease.valid());
    }
    // Touch "a" so it becomes the most recently used entry.
    { auto lease = cache.Acquire("a"); }
    const std::vector<GroupObject> victims = cache.SelectVictims(200, 1);
    ASSERT_EQ(victims.size(), 1u);
    EXPECT_EQ(victims[0].object_key, "b") << "LRU order must decide the victim";
    // An entry marked evicting is no longer acquirable by a new turn.
    EXPECT_FALSE(cache.Acquire("b").has_value());
    EXPECT_TRUE(cache.Contains("b"));
    cache.RestoreResident("b");
    EXPECT_TRUE(cache.Acquire("b").has_value());
    executor.Shutdown();
}

TEST(LocalCacheTest, ReserveAndInsertWaitsForCapacityWithoutOvercommitting) {
    SwarmExecutor executor(2);
    LocalCache cache(executor, 400);
    std::atomic<int> trigger_calls{0};
    cache.SetEvictionTrigger([&trigger_calls]() { trigger_calls.fetch_add(1); });

    {
        LocalLease first = RunSync(
            executor, cache.ReserveAndInsert(MakeObject("a", 400), Now() + std::chrono::seconds(1), StopToken()));
        ASSERT_TRUE(first.valid());
    }
    EXPECT_EQ(cache.used_bytes(), 400u);

    std::atomic<bool> inserted{false};
    auto inserter = [](LocalCache *target, std::atomic<bool> *flag) -> Task<> {
        LocalLease lease =
            co_await target->ReserveAndInsert(MakeObject("b", 400), Now() + std::chrono::seconds(5), StopToken());
        flag->store(lease.valid());
        co_return;
    };
    inserter(&cache, &inserted).via(&executor).start([](auto &&) {});
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_FALSE(inserted.load()) << "capacity pressure must block the insert, not overcommit";
    EXPECT_GE(trigger_calls.load(), 1) << "the eviction pipeline must be woken";
    EXPECT_EQ(cache.pending_wait_bytes(), 400u);
    EXPECT_EQ(cache.used_bytes(), 400u);

    const std::vector<GroupObject> victims = cache.SelectVictims(cache.pending_wait_bytes(), 128);
    ASSERT_EQ(victims.size(), 1u);
    cache.MarkRemoved(victims[0].object_key);
    const TimePoint deadline = Now() + std::chrono::seconds(3);
    while (!inserted.load() && Now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    EXPECT_TRUE(inserted.load());
    EXPECT_EQ(cache.used_bytes(), 400u);
    EXPECT_GE(cache.Stats().backpressure_waits, 1u);
    executor.Shutdown();
}

TEST(LocalCacheTest, CapacityWaitRespectsTheDeadline) {
    SwarmExecutor executor(2);
    LocalCache cache(executor, 100);
    {
        LocalLease held = RunSync(
            executor, cache.ReserveAndInsert(MakeObject("a", 100), Now() + std::chrono::seconds(1), StopToken()));
        ASSERT_TRUE(held.valid());
        // The lease is still open, so nothing is evictable and the second insert
        // must time out instead of blocking forever.
        LocalLease blocked = RunSync(
            executor, cache.ReserveAndInsert(MakeObject("b", 100), Now() + std::chrono::milliseconds(40), StopToken()));
        EXPECT_FALSE(blocked.valid());
    }
    EXPECT_EQ(cache.Stats().backpressure_timeouts, 1u);
    executor.Shutdown();
}

TEST(LocalCacheTest, RemovedEntryFreesCapacityAndDisappears) {
    SwarmExecutor executor(1);
    LocalCache cache(executor, 500);
    {
        LocalLease lease = RunSync(
            executor, cache.ReserveAndInsert(MakeObject("a", 250), Now() + std::chrono::seconds(1), StopToken()));
        ASSERT_TRUE(lease.valid());
    }
    const std::vector<GroupObject> victims = cache.SelectVictims(250, 8);
    ASSERT_EQ(victims.size(), 1u);
    cache.MarkRemoved("a");
    EXPECT_EQ(cache.used_bytes(), 0u);
    EXPECT_FALSE(cache.Contains("a"));
    EXPECT_EQ(cache.Stats().removed, 1u);
    executor.Shutdown();
}

TEST(LocalCacheTest, ShutdownFlushSelectsEveryUnleasedResidentObject) {
    SwarmExecutor executor(1);
    LocalCache cache(executor, 1000);
    for (const char *key : {"a", "b", "c", "d"}) {
        LocalLease lease = RunSync(
            executor, cache.ReserveAndInsert(MakeObject(key, 100), Now() + std::chrono::seconds(1), StopToken()));
        ASSERT_TRUE(lease.valid());
    }
    auto held = cache.Acquire("c");
    ASSERT_TRUE(held.has_value());
    const std::vector<GroupObject> flush = cache.SelectAllEvictable(128);
    EXPECT_EQ(flush.size(), 3u) << "the leased object must be left alone";
    executor.Shutdown();
}

TEST(LocalCacheTest, InsertingAnAlreadyResidentObjectReturnsAnExtraLease) {
    SwarmExecutor executor(1);
    LocalCache cache(executor, 1000);
    LocalLease first =
        RunSync(executor, cache.ReserveAndInsert(MakeObject("a", 100), Now() + std::chrono::seconds(1), StopToken()));
    ASSERT_TRUE(first.valid());
    LocalLease second =
        RunSync(executor, cache.ReserveAndInsert(MakeObject("a", 100), Now() + std::chrono::seconds(1), StopToken()));
    EXPECT_TRUE(second.valid());
    EXPECT_EQ(cache.used_bytes(), 100u) << "the same key must not be double-counted";
    executor.Shutdown();
}

} // namespace
} // namespace kvcm_swarm
