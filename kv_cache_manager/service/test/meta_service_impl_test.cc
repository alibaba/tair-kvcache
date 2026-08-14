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
#include "kv_cache_manager/protocol/protobuf/meta_service.pb.h"
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

class MetaServiceImplTest : public TESTBASE {
protected:
    void SetUp() override {
        metrics_registry_ = std::make_shared<MetricsRegistry>();
        registry_manager_ = std::make_shared<RegistryManager>("", metrics_registry_);
        cache_manager_ = std::make_shared<CacheManager>(metrics_registry_, registry_manager_);
        service_ = std::make_unique<MetaServiceImpl>(cache_manager_, nullptr, nullptr);
        service_->DisableLeaderOnlyRequests();
    }

    std::shared_ptr<MetricsRegistry> metrics_registry_;
    std::shared_ptr<RegistryManager> registry_manager_;
    std::shared_ptr<CacheManager> cache_manager_;
    std::unique_ptr<MetaServiceImpl> service_;
};

TEST_F(MetaServiceImplTest, FollowerRejectsReportEventQueryAndWrite) {
    RequestContext query_context("follower-query");
    proto::meta::GetHostCacheStateRequest query_request;
    query_request.set_trace_id(query_context.trace_id());
    query_request.set_instance_id("follower-instance");
    query_request.set_query_type(proto::meta::QT_PREFIX_MATCH);
    query_request.add_block_cache_keys(1);
    proto::meta::GetHostCacheStateResponse query_response;

    service_->GetHostCacheState(&query_context, &query_request, &query_response);

    // Reject valid queries before accessing the follower's unrecovered instance state.
    EXPECT_EQ(proto::meta::SERVER_NOT_LEADER, query_response.header().status().code());

    RequestContext write_context("follower-write");
    proto::meta::ReportEventRequest write_request;
    write_request.set_trace_id(write_context.trace_id());
    proto::meta::ReportEventResponse write_response;

    service_->ReportEvent(&write_context, &write_request, &write_response);

    EXPECT_EQ(proto::meta::SERVER_NOT_LEADER, write_response.header().status().code());
}

TEST_F(MetaServiceImplTest, PreservesPerKeyDetailsWhenEveryLookupFails) {
    service_->EnableLeaderOnlyRequests();

    Stub stub;
    stub.set(ADDR(CacheManager, GetCacheMetaDetail), GetCacheMetaDetailAllFailedStub);

    RequestContext request_context("all_failed");
    proto::meta::GetCacheMetaDetailRequest request;
    request.set_trace_id("all_failed");
    request.set_instance_id("test_instance");
    request.add_block_keys(22);
    proto::meta::GetCacheMetaDetailResponse response;

    service_->GetCacheMetaDetail(&request_context, &request, &response);

    EXPECT_EQ(proto::meta::INTERNAL_ERROR, response.header().status().code());
    ASSERT_EQ(1, response.items_size());
    EXPECT_EQ(0, response.items(0).request_index());
    EXPECT_EQ(22, response.items(0).block_key());
    EXPECT_EQ(proto::meta::INTERNAL_ERROR, response.items(0).status().code());
}

} // namespace kv_cache_manager
