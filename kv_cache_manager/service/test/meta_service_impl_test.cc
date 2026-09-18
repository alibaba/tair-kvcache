#include <memory>

#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/config/registry_manager.h"
#include "kv_cache_manager/manager/cache_manager.h"
#include "kv_cache_manager/metrics/metrics_registry.h"
#include "kv_cache_manager/protocol/protobuf/meta_service.pb.h"
#include "kv_cache_manager/service/meta_service_impl.h"

namespace kv_cache_manager {

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

TEST_F(MetaServiceImplTest, FollowerAllowsReportEventQueryButRejectsWrite) {
    RequestContext query_context("follower-query");
    proto::meta::GetHostCacheStateRequest query_request;
    query_request.set_trace_id(query_context.trace_id());
    proto::meta::GetHostCacheStateResponse query_response;

    service_->GetHostCacheState(&query_context, &query_request, &query_response);

    // INVALID_ARGUMENT proves the read passed the follower gate and reached
    // normal request validation instead of being rejected as SERVER_NOT_LEADER.
    EXPECT_EQ(proto::meta::INVALID_ARGUMENT, query_response.header().status().code());

    RequestContext write_context("follower-write");
    proto::meta::ReportEventRequest write_request;
    write_request.set_trace_id(write_context.trace_id());
    proto::meta::ReportEventResponse write_response;

    service_->ReportEvent(&write_context, &write_request, &write_response);

    EXPECT_EQ(proto::meta::SERVER_NOT_LEADER, write_response.header().status().code());
}

} // namespace kv_cache_manager
