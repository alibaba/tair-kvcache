#include <gtest/gtest.h>
#include <memory>

#include "kv_cache_manager/common/service_discovery.h"
#include "kv_cache_manager/common/service_discovery_factory.h"
#include "kv_cache_manager/common/static_service_discovery.h"
#include "kv_cache_manager/common/unittest.h"

using namespace kv_cache_manager;

class ServiceDiscoveryFactoryTest : public TESTBASE {};

TEST_F(ServiceDiscoveryFactoryTest, TestEmptyUrlReturnsNullptr) {
    EXPECT_EQ(ServiceDiscoveryFactory::CreateServiceDiscovery(""), nullptr);
}

TEST_F(ServiceDiscoveryFactoryTest, TestInvalidUrlReturnsNullptr) {
    EXPECT_EQ(ServiceDiscoveryFactory::CreateServiceDiscovery("not-a-url"), nullptr);
    EXPECT_EQ(ServiceDiscoveryFactory::CreateServiceDiscovery("nacos://my.service?cache_time=10"), nullptr);
    EXPECT_EQ(ServiceDiscoveryFactory::CreateServiceDiscovery("static://"), nullptr);
}

TEST_F(ServiceDiscoveryFactoryTest, TestStaticUrlSuccess) {
    auto discovery = ServiceDiscoveryFactory::CreateServiceDiscovery("static://10.0.0.1:8080,10.0.0.2:9090");
    ASSERT_NE(discovery, nullptr);
    EXPECT_EQ(discovery->GetType(), "Static");

    std::vector<ServiceEndpoint> eps;
    ASSERT_TRUE(discovery->GetAllEndpoints(eps));
    ASSERT_EQ(eps.size(), 2);
    EXPECT_EQ(eps[0].host, "10.0.0.1:8080");
    EXPECT_EQ(eps[1].host, "10.0.0.2:9090");
}

TEST_F(ServiceDiscoveryFactoryTest, TestStaticUrlMalformed) {
    EXPECT_EQ(ServiceDiscoveryFactory::CreateServiceDiscovery("static://10.0.0.1"), nullptr);
    EXPECT_EQ(ServiceDiscoveryFactory::CreateServiceDiscovery("static://10.0.0.1:abc"), nullptr);
}

TEST_F(ServiceDiscoveryFactoryTest, TestSpectrumEmptyVsidReturnsNullptr) {
    EXPECT_EQ(ServiceDiscoveryFactory::CreateServiceDiscovery("spectrum://"), nullptr);
}

TEST_F(ServiceDiscoveryFactoryTest, TestVipserverEmptyDomainReturnsNullptr) {
    EXPECT_EQ(ServiceDiscoveryFactory::CreateServiceDiscovery("vipserver://"), nullptr);
}
