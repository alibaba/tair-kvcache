#include <memory>

#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/config/registry_manager.h"
#include "kv_cache_manager/manager/cache_manager.h"
#include "kv_cache_manager/metrics/local_metrics_reporter.h"
#include "kv_cache_manager/metrics/metrics_registry.h"
#include "kv_cache_manager/protocol/protobuf/meta_service.pb.h"
#include "kv_cache_manager/service/meta_service_impl.h"
#include "stub.h"

namespace kv_cache_manager {

namespace {
ErrorCode RequestResult(RequestContext *context) { return context->trace_id() == "failure" ? EC_ERROR : EC_OK; }

std::pair<ErrorCode, CacheLocationViewVecWrapper> GetCacheLocationForMetrics(CacheManager *,
                                                                             RequestContext *context,
                                                                             const std::string &,
                                                                             CacheManager::QueryType,
                                                                             const KeyVector &,
                                                                             const CacheManager::TokenIdsVector &,
                                                                             const BlockMask &,
                                                                             int32_t,
                                                                             const std::vector<std::string> &) {
    return {RequestResult(context), CacheLocationViewVecWrapper{}};
}

std::pair<ErrorCode, BatchLocationsView> GetCacheLocationsByBackendForMetrics(CacheManager *,
                                                                              RequestContext *context,
                                                                              const std::string &,
                                                                              CacheManager::QueryType,
                                                                              const KeyVector &,
                                                                              const CacheManager::TokenIdsVector &,
                                                                              const BlockMask &,
                                                                              int32_t,
                                                                              const std::vector<std::string> &,
                                                                              const std::vector<BackendSelector> &) {
    return {RequestResult(context), BatchLocationsView{}};
}

std::pair<ErrorCode, StartWriteCacheInfo> StartWriteCacheForMetrics(CacheManager *,
                                                                    RequestContext *context,
                                                                    const std::string &,
                                                                    const KeyVector &,
                                                                    const CacheManager::TokenIdsVector &,
                                                                    const std::vector<std::string> &,
                                                                    int64_t,
                                                                    int32_t) {
    return {RequestResult(context), StartWriteCacheInfo{}};
}

ErrorCode FinishWriteCacheForMetrics(CacheManager *,
                                     RequestContext *context,
                                     const std::string &,
                                     const std::string &,
                                     const BlockMask &,
                                     std::unique_ptr<WriteLocationManager::WriteLocationInfo>) {
    return RequestResult(context);
}

ErrorCode ReportEventForMetrics(CacheManager *,
                                RequestContext *context,
                                const proto::meta::ReportEventRequest *,
                                proto::meta::ReportEventResponse *response) {
    const auto ec = context->trace_id() == "partial" ? EC_PARTIAL_OK : RequestResult(context);
    response->mutable_header()->mutable_status()->set_code(ec == EC_OK ? proto::meta::OK : proto::meta::INTERNAL_ERROR);
    return ec;
}
} // namespace

class MetaServiceImplTest : public TESTBASE {
protected:
    void SetUp() override {
        metrics_registry_ = std::make_shared<MetricsRegistry>();
        registry_manager_ = std::make_shared<RegistryManager>("", metrics_registry_);
        cache_manager_ = std::make_shared<CacheManager>(metrics_registry_, registry_manager_);
        reporter_ = std::make_shared<LocalMetricsReporter>();
        ASSERT_TRUE(reporter_->Init(cache_manager_, metrics_registry_, ""));
        service_ = std::make_unique<MetaServiceImpl>(cache_manager_, reporter_, nullptr);
        service_->DisableLeaderOnlyRequests();
    }

    template <typename Request, typename Response>
    void CheckErrorMetrics(const std::string &api_name,
                           Request request,
                           void (MetaServiceImpl::*method)(RequestContext *, const Request *, Response *)) {
        service_->EnableLeaderOnlyRequests();
        auto collector =
            std::make_shared<ServiceMetricsCollector>(metrics_registry_, MetricsTags{{"api_name", api_name}});
        ASSERT_TRUE(collector->Init());
        std::uint64_t queries = 0;
        std::uint64_t errors = 0;
        for (const auto &trace : {"success", "failure", "success"}) {
            RequestContext context(trace, collector);
            Response response;
            (service_.get()->*method)(&context, &request, &response);
            const bool failed = context.trace_id() == "failure";
            EXPECT_EQ(failed ? proto::meta::INTERNAL_ERROR : proto::meta::OK, response.header().status().code());
            EXPECT_EQ(++queries, collector->get_service_query_counter_metrics());
            errors += failed;
            EXPECT_EQ(errors, collector->get_service_error_counter_metrics());
            EXPECT_DOUBLE_EQ(failed ? 1. : 0., collector->get_service_error_code_metrics());
        }

        request.Clear();
        request.set_instance_id("test-instance");
        RequestContext context("invalid-request", collector);
        Response response;
        (service_.get()->*method)(&context, &request, &response);
        EXPECT_EQ(proto::meta::INVALID_ARGUMENT, response.header().status().code());
        EXPECT_EQ(++queries, collector->get_service_query_counter_metrics());
        EXPECT_EQ(++errors, collector->get_service_error_counter_metrics());
    }

    std::shared_ptr<LocalMetricsReporter> reporter_;

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

TEST_F(MetaServiceImplTest, GetCacheLocationErrorMetrics) {
    Stub stub;
    stub.set(ADDR(CacheManager, GetCacheLocation), GetCacheLocationForMetrics);
    proto::meta::GetCacheLocationRequest request;
    request.set_instance_id("test-instance");
    request.add_block_keys(1);
    CheckErrorMetrics("GetCacheLocation", request, &MetaServiceImpl::GetCacheLocation);
}

TEST_F(MetaServiceImplTest, GetCacheLocationsByBackendErrorMetrics) {
    Stub stub;
    stub.set(ADDR(CacheManager, GetCacheLocationsByBackend), GetCacheLocationsByBackendForMetrics);
    proto::meta::GetCacheLocationsByBackendRequest request;
    request.set_instance_id("test-instance");
    request.add_block_keys(1);
    CheckErrorMetrics("GetCacheLocationsByBackend", request, &MetaServiceImpl::GetCacheLocationsByBackend);
}

TEST_F(MetaServiceImplTest, StartWriteCacheErrorMetrics) {
    Stub stub;
    stub.set(ADDR(CacheManager, StartWriteCache), StartWriteCacheForMetrics);
    proto::meta::StartWriteCacheRequest request;
    request.set_instance_id("test-instance");
    request.add_block_keys(1);
    CheckErrorMetrics("StartWriteCache", request, &MetaServiceImpl::StartWriteCache);
}

TEST_F(MetaServiceImplTest, FinishWriteCacheErrorMetrics) {
    Stub stub;
    stub.set(ADDR(CacheManager, FinishWriteCache), FinishWriteCacheForMetrics);
    proto::meta::FinishWriteCacheRequest request;
    request.set_instance_id("test-instance");
    request.set_write_session_id("session");
    request.mutable_success_blocks();
    CheckErrorMetrics("FinishWriteCache", request, &MetaServiceImpl::FinishWriteCache);
}

TEST_F(MetaServiceImplTest, ReportEventCountsFailuresAndPartialFailuresOnce) {
    Stub stub;
    stub.set(ADDR(CacheManager, ReportEvent), ReportEventForMetrics);
    service_->EnableLeaderOnlyRequests();
    auto collector = std::make_shared<ServiceMetricsCollector>(metrics_registry_);
    auto event_collector =
        std::make_shared<EventReportMetricsCollector>(metrics_registry_, MetricsTags{{"event_type", "block_add"}});
    ASSERT_TRUE(collector->Init());
    ASSERT_TRUE(event_collector->Init());

    std::uint64_t queries = 0;
    std::uint64_t errors = 0;
    for (const auto &trace : {"success", "failure", "partial", "success"}) {
        RequestContext context(trace, collector);
        context.GetMetricsCollectorsVehicle().AddMetricsCollector(event_collector);
        proto::meta::ReportEventRequest request;
        proto::meta::ReportEventResponse response;
        service_->ReportEvent(&context, &request, &response);
        const bool failed = context.trace_id() != "success";
        EXPECT_EQ(failed ? proto::meta::INTERNAL_ERROR : proto::meta::OK, response.header().status().code());
        ++queries;
        errors += failed;
        EXPECT_EQ(queries, collector->get_service_query_counter_metrics());
        EXPECT_EQ(errors, collector->get_service_error_counter_metrics());
        EXPECT_EQ(queries, event_collector->get_service_query_counter_metrics());
        EXPECT_EQ(errors, event_collector->get_service_error_counter_metrics());
    }
}

} // namespace kv_cache_manager
