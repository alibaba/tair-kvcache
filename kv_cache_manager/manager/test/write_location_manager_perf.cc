#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdio.h>
#include <thread>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/manager/write_location_manager.h"

namespace kv_cache_manager {

class WriteLocationManagerPerf : public TESTBASE {
public:
    void SetUp() override {}
    void TearDown() override {}

private:
    WriteLocationManager manager_;
};

TEST_F(WriteLocationManagerPerf, RandomTest) {
    manager_.Start();

    constexpr size_t kCount = 1000;
    constexpr size_t kTimeout = 3;
    auto st = TimestampUtil::GetCurrentTimeMs();
    manager_.Put(std::to_string(0), {}, {}, kTimeout, []() {});
    std::thread provider([&]() {
        for (size_t i = 1; i < kCount; i++) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            manager_.Put(std::to_string(i), {}, {}, kTimeout + TimestampUtil::GetCurrentTimeUs() % 10, []() {});
        }
    });
    size_t expire_size = 0;
    while (expire_size = manager_.ExpireSize(), expire_size > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        KVCM_LOG_INFO("expre_size [%zu]", expire_size);
    }

    KVCM_LOG_INFO("[RandomTest] span time [%ld ms]", TimestampUtil::GetCurrentTimeMs() - st);

    provider.join();
    manager_.Stop();
}

TEST_F(WriteLocationManagerPerf, BurstTest) {
    manager_.Start();

    constexpr size_t kCount = 1000;
    constexpr size_t kTimeout = 3;
    auto st = TimestampUtil::GetCurrentTimeMs();
    manager_.Put(std::to_string(0), {}, {}, kTimeout, []() {});
    std::thread provider([&]() {
        for (size_t i = 1; i < kCount; i++) {
            std::this_thread::sleep_for(std::chrono::microseconds(1));
            manager_.Put(std::to_string(i), {}, {}, kTimeout, []() {});
        }
    });
    size_t expire_size = 0;
    while (expire_size = manager_.ExpireSize(), expire_size > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        KVCM_LOG_INFO("expre_size [%zu]", expire_size);
    }

    KVCM_LOG_INFO("[BurstTest] span time [%ld ms]", TimestampUtil::GetCurrentTimeMs() - st);

    provider.join();
    manager_.Stop();
}

TEST_F(WriteLocationManagerPerf, MultiProviderConsumerTest) {
    manager_.Start();

    constexpr size_t kCount = 1000;
    constexpr size_t kTimeout = 3;
    for (size_t i = 0; i < kCount; i++) {
        std::this_thread::sleep_for(std::chrono::microseconds(1));
        manager_.Put(std::to_string(i), {}, {}, kTimeout + TimestampUtil::GetCurrentTimeUs() % 3, []() {});
    }
    auto st = TimestampUtil::GetCurrentTimeMs();
    auto provider_fcn = [this](int id) {
        size_t base = id * 100000;
        for (size_t i = 1 + base; i < base + kCount; i++) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            manager_.Put(std::to_string(i), {}, {}, kTimeout + TimestampUtil::GetCurrentTimeUs() % 3, []() {});
        }
    };
    std::vector<std::thread> threads;
    for (int i = 1; i < 4; i++) {
        threads.push_back(std::thread(provider_fcn, i));
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    std::atomic_int consumer_count(0);
    auto consumer_fcn = [&consumer_count, this](int id) {
        size_t base = id * 100000;
        for (size_t i = 1 + base; i < base + kCount; i++) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10 + i % 10));
            WriteLocationManager::WriteLocationInfo location_info;
            if (manager_.GetAndDelete(std::to_string(i), location_info)) {
                consumer_count.fetch_add(1, std::memory_order_relaxed);
            }
        }
    };
    for (int i = 1; i < 4; i++) {
        threads.push_back(std::thread(consumer_fcn, i));
    }

    size_t expire_size = 0;
    while (expire_size = manager_.ExpireSize(), expire_size > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        KVCM_LOG_INFO("expre_size [%zu]", expire_size);
    }

    KVCM_LOG_INFO("[MultiProviderConsumerTest] span time [%ld ms], consumer_count [%d]",
                  TimestampUtil::GetCurrentTimeMs() - st,
                  consumer_count.load());

    for (auto &thread : threads) {
        thread.join();
    }
    manager_.Stop();
}

} // namespace kv_cache_manager