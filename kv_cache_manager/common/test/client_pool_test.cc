#include <atomic>
#include <memory>
#include <thread>

#include "kv_cache_manager/common/client_pool.h"
#include "kv_cache_manager/common/unittest.h"

namespace kv_cache_manager {

class ClientPoolTest : public TESTBASE {
public:
};

class FakeClient {
public:
    explicit FakeClient(std::shared_ptr<std::atomic_int> count) { count->fetch_add(1, std::memory_order_relaxed); }
};

TEST_F(ClientPoolTest, TestStaticClientPoolAcquireAndRelease) {
    auto count = std::make_shared<std::atomic_int>(0);
    StaticClientPool<FakeClient> client_pool([count]() { return std::make_shared<FakeClient>(count); }, 2);
    ASSERT_TRUE(client_pool.Initialize());
    ASSERT_EQ(2, count->load(std::memory_order_relaxed));
    {
        auto handle_1 = client_pool.AcquireClient(1);
        ASSERT_TRUE(handle_1);
        auto handle_2 = client_pool.AcquireClient(1);
        ASSERT_TRUE(handle_2);
        auto handle_3 = client_pool.AcquireClient(1);
        ASSERT_FALSE(handle_3);
    }
    ASSERT_EQ(2, count->load(std::memory_order_relaxed));
    {
        auto handle_1 = client_pool.AcquireClient(1);
        ASSERT_TRUE(handle_1);
        auto handle_2 = client_pool.AcquireClient(1);
        ASSERT_TRUE(handle_2);
        auto handle_3 = client_pool.AcquireClient(1);
        ASSERT_FALSE(handle_3);
    }
}

TEST_F(ClientPoolTest, TestStaticClientPoolAcquireAndReleaseMultithread) {
    size_t thread_num = 8;
    auto count = std::make_shared<std::atomic_int>(0);
    StaticClientPool<FakeClient> client_pool([count]() { return std::make_shared<FakeClient>(count); }, 2);
    ASSERT_TRUE(client_pool.Initialize());
    ASSERT_EQ(2, count->load(std::memory_order_relaxed));
    std::atomic<bool> go = false;
    auto thread_fcn = [this, &client_pool, &go]() {
        while (!go.load(std::memory_order_relaxed)) {}
        {
            auto handle = client_pool.AcquireClient();
            ASSERT_TRUE(handle);
        }
        {
            auto handle = client_pool.AcquireClient();
            ASSERT_TRUE(handle);
        }
    };
    for (int i = 0; i < 20; ++i) {
        std::vector<std::thread> threads;
        for (int j = 0; j < thread_num; ++j) {
            threads.push_back(std::thread(thread_fcn));
        }
        go.store(true, std::memory_order_relaxed);
        for (auto &thread : threads) {
            thread.join();
        }
        go.store(false, std::memory_order_relaxed);
    }
}

TEST_F(ClientPoolTest, TestDynamicClientPoolAcquireAndRelease) {
    auto count = std::make_shared<std::atomic_int>(0);
    DynamicClientPool<FakeClient> client_pool([count]() { return std::make_shared<FakeClient>(count); }, 1, 2);
    ASSERT_TRUE(client_pool.Initialize());
    ASSERT_EQ(1, count->load(std::memory_order_relaxed));
    {
        auto handle_1 = client_pool.AcquireClient(1);
        ASSERT_TRUE(handle_1);
        ASSERT_EQ(1, count->load(std::memory_order_relaxed));
        auto handle_2 = client_pool.AcquireClient(1);
        ASSERT_TRUE(handle_2);
        ASSERT_EQ(2, count->load(std::memory_order_relaxed));
        auto handle_3 = client_pool.AcquireClient(1);
        ASSERT_FALSE(handle_3);
        ASSERT_EQ(2, count->load(std::memory_order_relaxed));
    }
    {
        auto handle_1 = client_pool.AcquireClient(1);
        ASSERT_TRUE(handle_1);
        ASSERT_EQ(2, count->load(std::memory_order_relaxed));
        auto handle_2 = client_pool.AcquireClient(1);
        ASSERT_TRUE(handle_2);
        ASSERT_EQ(2, count->load(std::memory_order_relaxed));
        auto handle_3 = client_pool.AcquireClient(1);
        ASSERT_FALSE(handle_3);
        ASSERT_EQ(2, count->load(std::memory_order_relaxed));
    }
}

TEST_F(ClientPoolTest, TestDynamicClientPoolAcquireAndReleaseMultithread) {
    size_t thread_num = 8;
    auto count = std::make_shared<std::atomic_int>(0);
    DynamicClientPool<FakeClient> client_pool([count]() { return std::make_shared<FakeClient>(count); }, 0, 16);
    ASSERT_TRUE(client_pool.Initialize());
    ASSERT_EQ(0, count->load(std::memory_order_relaxed));
    std::atomic<bool> go = false;
    auto thread_fcn = [this, &client_pool, &go]() {
        while (!go.load(std::memory_order_relaxed)) {}
        {
            auto handle = client_pool.AcquireClient(5);
            ASSERT_TRUE(handle);
        }
        {
            auto handle = client_pool.AcquireClient(5);
            ASSERT_TRUE(handle);
        }
    };
    for (int i = 0; i < 20; ++i) {
        std::vector<std::thread> threads;
        for (int j = 0; j < thread_num; ++j) {
            threads.push_back(std::thread(thread_fcn));
        }
        go.store(true, std::memory_order_relaxed);
        for (auto &thread : threads) {
            thread.join();
        }
        go.store(false, std::memory_order_relaxed);
        ASSERT_LT(count->load(std::memory_order_relaxed), 9);
    }
}

} // namespace kv_cache_manager