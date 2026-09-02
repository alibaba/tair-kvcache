#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/config/registry_manager.h"
#include "kv_cache_manager/manager/cache_location_view.h"
#include "kv_cache_manager/manager/cache_manager.h"
#include "kv_cache_manager/metrics/metrics_registry.h"
#include "kv_cache_manager/service/meta_service_impl.h"
#include "stub.h"

namespace kv_cache_manager {
namespace {

std::pair<ErrorCode, CacheMetaDetailVec>
GetCacheMetaDetailAllFailedStub(void * /*obj*/,
                                RequestContext * /*request_context*/,
                                const std::string & /*instance_id*/,
                                const CacheManager::KeyVector & /*keys*/,
                                const CacheManager::TokenIdsVector & /*tokens*/,
                                const BlockMask & /*block_mask*/,
                                int32_t /*detail_level*/) {
    CacheKeyMetaDetail item;
    item.error_code = EC_TIMEOUT;
    item.request_index = 0;
    item.block_key = 22;
    return {EC_TIMEOUT, {std::move(item)}};
}

} // namespace

TEST(MetaServiceImplTest, PreservesPerKeyDetailsWhenEveryLookupFails) {
    auto metrics_registry = std::make_shared<MetricsRegistry>();
    auto registry_manager = std::make_shared<RegistryManager>("", metrics_registry);
    auto cache_manager = std::make_shared<CacheManager>(metrics_registry, registry_manager);
    MetaServiceImpl service(cache_manager, /*metrics_reporter*/ nullptr, /*leader_elector*/ nullptr);

    Stub stub;
    stub.set(ADDR(CacheManager, GetCacheMetaDetail), GetCacheMetaDetailAllFailedStub);

    RequestContext request_context("all_failed");
    proto::meta::GetCacheMetaDetailRequest request;
    request.set_trace_id("all_failed");
    request.set_instance_id("test_instance");
    request.add_block_keys(22);
    proto::meta::GetCacheMetaDetailResponse response;

    service.GetCacheMetaDetail(&request_context, &request, &response);

    EXPECT_EQ(proto::meta::INTERNAL_ERROR, response.header().status().code());
    ASSERT_EQ(1, response.items_size());
    EXPECT_EQ(0, response.items(0).request_index());
    EXPECT_EQ(22, response.items(0).block_key());
    EXPECT_EQ(proto::meta::INTERNAL_ERROR, response.items(0).status().code());
}

} // namespace kv_cache_manager
