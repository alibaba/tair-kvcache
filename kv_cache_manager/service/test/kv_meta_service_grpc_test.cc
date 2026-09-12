#include <memory>
#include <stdexcept>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/metrics/metrics_registry.h"
#include "kv_cache_manager/service/grpc_service/kv_meta_service_grpc.h"

namespace kv_cache_manager {

class KvMetaServiceGRpcTestPeer {
public:
    template <typename Response, typename Callback>
    static grpc::Status InvokeSafely(const char *operation, Response *response, Callback &&callback) {
        return KvMetaServiceGRpc::InvokeSafely(operation, response, std::forward<Callback>(callback));
    }
};

namespace {

TEST(KvMetaServiceGRpcTest, MetricsAreNamespacedAwayFromMainService) {
    auto registry = std::make_shared<MetricsRegistry>();
    auto main_metric =
        registry->GetCounter("service.query_counter", MetricsTags{{"api_name", "RegisterInstance"}});

    KvMetaServiceGRpc service(registry, nullptr);
    service.Init();

    auto kv_meta_metric =
        registry->GetCounter("service.query_counter", MetricsTags{{"api_name", "KvMeta.RegisterInstance"}});
    EXPECT_NE(main_metric.GetRaw(), kv_meta_metric.GetRaw());

    auto metrics_data = registry->GetMetricsData("service.query_counter");
    ASSERT_NE(nullptr, metrics_data);
    EXPECT_EQ(8U, metrics_data->GetSize());
}

template <typename Response>
void ExpectContainedFailure(const grpc::Status &status, const Response &response) {
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(grpc::StatusCode::INTERNAL, status.error_code());
    EXPECT_EQ("KVMeta request failed unexpectedly", status.error_message());
    EXPECT_EQ(proto::kv_meta::INTERNAL_ERROR, response.header().status().code());
    EXPECT_EQ("KVMeta request failed unexpectedly", response.header().status().message());
}

TEST(KvMetaServiceGRpcTest, EveryRpcContainsUnavailableImplementationException) {
    auto registry = std::make_shared<MetricsRegistry>();
    KvMetaServiceGRpc service(registry, nullptr);
    service.Init();
    grpc::ServerContext context;

    proto::kv_meta::RegisterInstanceRequest register_request;
    proto::kv_meta::RegisterInstanceResponse register_response;
    ExpectContainedFailure(service.RegisterInstance(&context, &register_request, &register_response),
                           register_response);

    proto::kv_meta::GetInstanceInfoRequest info_request;
    proto::kv_meta::GetInstanceInfoResponse info_response;
    ExpectContainedFailure(service.GetInstanceInfo(&context, &info_request, &info_response), info_response);

    proto::kv_meta::GetRequest get_request;
    proto::kv_meta::GetResponse get_response;
    ExpectContainedFailure(service.Get(&context, &get_request, &get_response), get_response);

    proto::kv_meta::PutStartRequest put_start_request;
    proto::kv_meta::PutStartResponse put_start_response;
    ExpectContainedFailure(service.PutStart(&context, &put_start_request, &put_start_response), put_start_response);

    proto::kv_meta::PutFinishRequest put_finish_request;
    proto::kv_meta::CommonResponse put_finish_response;
    ExpectContainedFailure(service.PutFinish(&context, &put_finish_request, &put_finish_response), put_finish_response);

    proto::kv_meta::RemoveRequest remove_request;
    proto::kv_meta::CommonResponse remove_response;
    ExpectContainedFailure(service.Remove(&context, &remove_request, &remove_response), remove_response);

    proto::kv_meta::TrimRequest trim_request;
    proto::kv_meta::CommonResponse trim_response;
    ExpectContainedFailure(service.Trim(&context, &trim_request, &trim_response), trim_response);
}

TEST(KvMetaServiceGRpcTest, ExceptionFirewallClearsPartialResponseAndContainsUnknownThrowables) {
    proto::kv_meta::CommonResponse response;
    response.mutable_header()->mutable_status()->set_code(proto::kv_meta::OK);
    response.mutable_header()->mutable_status()->set_message("provider-secret");

    auto status = KvMetaServiceGRpcTestPeer::InvokeSafely("test", &response, []() { throw 7; });

    ExpectContainedFailure(status, response);
    EXPECT_EQ(std::string::npos, response.header().status().message().find("provider-secret"));
}

TEST(KvMetaServiceGRpcTest, ExceptionFirewallPreservesSuccessfulResponse) {
    proto::kv_meta::CommonResponse response;
    auto status = KvMetaServiceGRpcTestPeer::InvokeSafely("test", &response, [&response]() {
        response.mutable_header()->mutable_status()->set_code(proto::kv_meta::OK);
    });

    EXPECT_TRUE(status.ok());
    EXPECT_EQ(proto::kv_meta::OK, response.header().status().code());
}

} // namespace
} // namespace kv_cache_manager
