#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <thread>
#include <vector>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/optimizer/service/event_subscriber/fanout_executor.h"

namespace kv_cache_manager {

class FanoutExecutorTest : public TESTBASE {};

TEST_F(FanoutExecutorTest, RunsOneBatchConcurrentlyAndWaitsForEveryTask) {
    constexpr std::size_t kTaskCount = 4;
    FanoutExecutor executor(kTaskCount);

    std::mutex mutex;
    std::condition_variable cv;
    std::size_t started = 0;
    bool release = false;
    std::atomic<std::size_t> finished{0};

    std::vector<FanoutExecutor::Task> tasks;
    for (std::size_t i = 0; i < kTaskCount; ++i) {
        tasks.emplace_back([&] {
            std::unique_lock<std::mutex> lock(mutex);
            ++started;
            cv.notify_all();
            cv.wait(lock, [&] { return release; });
            lock.unlock();
            ++finished;
        });
    }

    bool run_succeeded = false;
    std::thread caller([&] { run_succeeded = executor.Run(std::move(tasks)); });

    {
        std::unique_lock<std::mutex> lock(mutex);
        EXPECT_TRUE(cv.wait_for(lock, std::chrono::seconds(3), [&] { return started == kTaskCount; }));
        release = true;
    }
    cv.notify_all();
    caller.join();

    EXPECT_TRUE(run_succeeded);
    EXPECT_EQ(kTaskCount, finished.load());
}

} // namespace kv_cache_manager
