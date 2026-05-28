#include <chrono>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <memory>
#include <thread>

#include "kv_cache_manager/common/service_discovery.h"
#include "kv_cache_manager/common/unittest.h"

using namespace kv_cache_manager;

/**
 * 实现一个 FakeCachedDiscovery 用于测试 CachedServiceDiscovery 基类行为。
 * 它允许我们注入期望的端点和控制 FetchEndpoints 的成功/失败。
 */
class FakeCachedDiscovery : public CachedServiceDiscovery {
public:
    explicit FakeCachedDiscovery(int cache_ttl_seconds = 30) : CachedServiceDiscovery(cache_ttl_seconds) {}

    void SetFetchEndpointsResult(bool success, const std::vector<ServiceEndpoint> &endpoints) {
        std::lock_guard<std::mutex> lock(mutex_);
        fetch_success_ = success;
        fetch_endpoints_ = endpoints;
    }

    void SetRefreshBehavior(bool always_fail) { always_fail_ = always_fail; }

protected:
    bool FetchEndpoints(std::vector<ServiceEndpoint> &endpoints) override {
        std::lock_guard<std::mutex> lock(mutex_);
        call_count_++;
        if (always_fail_) {
            return false;
        }
        endpoints = fetch_endpoints_;
        return fetch_success_;
    }

    bool Init(const std::string & /*service_address*/) override { return true; }

    std::string GetType() const override { return "FakeCached"; }

public:
    int call_count_ = 0;

private:
    std::mutex mutex_;
    bool fetch_success_ = true;
    std::vector<ServiceEndpoint> fetch_endpoints_;
    bool always_fail_ = false;
};

class FakeServiceDiscovery : public ServiceDiscovery {
public:
    bool Init(const std::string & /*service_address*/) override { return true; }

    void SetEndpoints(const std::vector<ServiceEndpoint> &endpoints) { endpoints_ = endpoints; }

    bool GetAllEndpoints(std::vector<ServiceEndpoint> &endpoints) override {
        endpoints = endpoints_;
        return !endpoints_.empty();
    }

    bool GetOneEndpoint(ServiceEndpoint &endpoint) override {
        if (endpoints_.empty())
            return false;
        endpoint = endpoints_[0];
        return true;
    }

    bool Refresh() override { return !endpoints_.empty(); }

    std::string GetType() const override { return "Fake"; }

private:
    std::vector<ServiceEndpoint> endpoints_;
};

class CachedServiceDiscoveryTest : public TESTBASE {};

class ServiceEndpointTest : public TESTBASE {};

class ServiceDiscoveryTest : public TESTBASE {};

// ========== ServiceEndpoint 测试 ==========

TEST_F(ServiceEndpointTest, TestDefaultConstructor) {
    ServiceEndpoint ep;
    EXPECT_EQ(ep.ip, "");
    EXPECT_EQ(ep.port, 0);
    EXPECT_EQ(ep.host, "");
    EXPECT_EQ(ep.weight, 100);
    EXPECT_TRUE(ep.healthy);
}

TEST_F(ServiceEndpointTest, TestParameterizedConstructor) {
    ServiceEndpoint ep("192.168.1.1", 8080, 200);
    EXPECT_EQ(ep.ip, "192.168.1.1");
    EXPECT_EQ(ep.port, 8080);
    EXPECT_EQ(ep.host, "192.168.1.1:8080");
    EXPECT_EQ(ep.weight, 200);
    EXPECT_TRUE(ep.healthy);
}

TEST_F(ServiceEndpointTest, TestParameterizedConstructorDefaultWeight) {
    ServiceEndpoint ep("10.0.0.1", 9090);
    EXPECT_EQ(ep.ip, "10.0.0.1");
    EXPECT_EQ(ep.port, 9090);
    EXPECT_EQ(ep.host, "10.0.0.1:9090");
    EXPECT_EQ(ep.weight, 100);
}

// ========== CachedServiceDiscovery 测试 ==========

TEST_F(CachedServiceDiscoveryTest, TestGetAllEndpointsSuccess) {
    auto discovery = std::make_unique<FakeCachedDiscovery>(30);

    std::vector<ServiceEndpoint> expected;
    expected.emplace_back("172.1.2.10", 8080, 100);
    expected.emplace_back("172.1.2.11", 8081, 200);
    discovery->SetFetchEndpointsResult(true, expected);

    std::vector<ServiceEndpoint> result;
    ASSERT_TRUE(discovery->GetAllEndpoints(result));
    ASSERT_EQ(result.size(), 2);
    EXPECT_EQ(result[0].ip, "172.1.2.10");
    EXPECT_EQ(result[1].ip, "172.1.2.11");
    EXPECT_EQ(discovery->call_count_, 1);

    // 第二次调用应命中缓存，call_count 不变
    result.clear();
    ASSERT_TRUE(discovery->GetAllEndpoints(result));
    ASSERT_EQ(result.size(), 2);
    EXPECT_EQ(discovery->call_count_, 1);
}

TEST_F(CachedServiceDiscoveryTest, TestGetAllEndpointsFetchFailure) {
    auto discovery = std::make_unique<FakeCachedDiscovery>(30);

    discovery->SetFetchEndpointsResult(false, {});

    std::vector<ServiceEndpoint> result;
    EXPECT_FALSE(discovery->GetAllEndpoints(result));
    EXPECT_TRUE(result.empty());
    EXPECT_EQ(discovery->call_count_, 1);
}

TEST_F(CachedServiceDiscoveryTest, TestGetAllEndpointsEmptyResult) {
    auto discovery = std::make_unique<FakeCachedDiscovery>(30);

    discovery->SetFetchEndpointsResult(true, {});

    std::vector<ServiceEndpoint> result;
    EXPECT_FALSE(discovery->GetAllEndpoints(result));
    EXPECT_TRUE(result.empty());
}

TEST_F(CachedServiceDiscoveryTest, TestCacheExpiration) {
    // 缓存 TTL 为 1 秒
    auto discovery = std::make_unique<FakeCachedDiscovery>(1);

    std::vector<ServiceEndpoint> endpoints;
    endpoints.emplace_back("172.1.2.10", 8080, 100);
    discovery->SetFetchEndpointsResult(true, endpoints);

    // 首次调用，FetchEndpoints 被调用一次
    std::vector<ServiceEndpoint> result;
    ASSERT_TRUE(discovery->GetAllEndpoints(result));
    EXPECT_EQ(discovery->call_count_, 1);

    // 等待缓存过期
    std::this_thread::sleep_for(std::chrono::milliseconds(1100));

    // 再次调用，缓存过期，应重新调用 FetchEndpoints
    result.clear();
    ASSERT_TRUE(discovery->GetAllEndpoints(result));
    EXPECT_EQ(discovery->call_count_, 2);
}

TEST_F(CachedServiceDiscoveryTest, TestGetOneEndpointRandomSelection) {
    auto discovery = std::make_unique<FakeCachedDiscovery>(30);

    std::vector<ServiceEndpoint> endpoints;
    endpoints.emplace_back("172.1.2.10", 8080, 100);
    endpoints.emplace_back("172.1.2.11", 8081, 200);
    endpoints.emplace_back("172.1.2.12", 8082, 150);
    discovery->SetFetchEndpointsResult(true, endpoints);

    ServiceEndpoint endpoint;
    ASSERT_TRUE(discovery->GetOneEndpoint(endpoint));
    EXPECT_FALSE(endpoint.ip.empty());
    EXPECT_GT(endpoint.port, 0);

    // 收集多次结果，验证负载均衡
    std::set<std::string> unique_endpoints;
    for (int i = 0; i < 100; i++) {
        endpoint = ServiceEndpoint();
        discovery->GetOneEndpoint(endpoint);
        unique_endpoints.insert(endpoint.host);
    }
    // 在 100 次随机中，应该能命中至少 2 个不同的端点
    EXPECT_GE(unique_endpoints.size(), 2);
}

TEST_F(CachedServiceDiscoveryTest, TestGetOneEndpointEmptyEndpoints) {
    auto discovery = std::make_unique<FakeCachedDiscovery>(30);

    discovery->SetFetchEndpointsResult(true, {});

    ServiceEndpoint endpoint;
    EXPECT_FALSE(discovery->GetOneEndpoint(endpoint));
}

TEST_F(CachedServiceDiscoveryTest, TestRefreshSuccess) {
    auto discovery = std::make_unique<FakeCachedDiscovery>(30);

    std::vector<ServiceEndpoint> endpoints;
    endpoints.emplace_back("172.1.2.10", 8080, 100);
    discovery->SetFetchEndpointsResult(true, endpoints);

    ASSERT_TRUE(discovery->Refresh());
    EXPECT_EQ(discovery->call_count_, 1);

    std::vector<ServiceEndpoint> result;
    ASSERT_TRUE(discovery->GetAllEndpoints(result));
    EXPECT_EQ(result.size(), 1);
    // GetAllEndpoints 命中缓存，不会再次调用 FetchEndpoints
    EXPECT_EQ(discovery->call_count_, 1);
}

TEST_F(CachedServiceDiscoveryTest, TestRefreshFailure) {
    auto discovery = std::make_unique<FakeCachedDiscovery>(30);

    discovery->SetFetchEndpointsResult(false, {});

    EXPECT_FALSE(discovery->Refresh());
    EXPECT_EQ(discovery->call_count_, 1);

    // 刷新失败后缓存无效，GetAllEndpoints 会再尝试一次
    std::vector<ServiceEndpoint> result;
    EXPECT_FALSE(discovery->GetAllEndpoints(result));
    EXPECT_EQ(discovery->call_count_, 2);
}

// ========== FakeServiceDiscovery 测试 ==========

TEST_F(ServiceDiscoveryTest, TestFakeServiceDiscovery) {
    FakeServiceDiscovery discovery;

    std::vector<ServiceEndpoint> endpoints;
    endpoints.emplace_back("10.0.0.1", 8080, 100);
    endpoints.emplace_back("10.0.0.2", 8081, 200);
    discovery.SetEndpoints(endpoints);

    ASSERT_TRUE(discovery.Init("fake_address"));
    EXPECT_EQ(discovery.GetType(), "Fake");

    std::vector<ServiceEndpoint> result;
    ASSERT_TRUE(discovery.GetAllEndpoints(result));
    EXPECT_EQ(result.size(), 2);

    ServiceEndpoint ep;
    ASSERT_TRUE(discovery.GetOneEndpoint(ep));
    EXPECT_EQ(ep.ip, "10.0.0.1");

    ASSERT_TRUE(discovery.Refresh());
}

TEST_F(ServiceDiscoveryTest, TestFakeServiceDiscoveryEmpty) {
    FakeServiceDiscovery discovery;

    EXPECT_TRUE(discovery.Init("fake_address"));

    std::vector<ServiceEndpoint> result;
    EXPECT_FALSE(discovery.GetAllEndpoints(result));

    ServiceEndpoint ep;
    EXPECT_FALSE(discovery.GetOneEndpoint(ep));

    EXPECT_FALSE(discovery.Refresh());
}
