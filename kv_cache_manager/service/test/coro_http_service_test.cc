#include <cstdint>
#include <memory>
#include <sstream>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/config/registry_manager.h"
#include "kv_cache_manager/manager/cache_manager.h"
#include "kv_cache_manager/metrics/metrics_registry.h"
#include "kv_cache_manager/service/http_service/coro_http_service.h"
#include "kv_cache_manager/service/http_service/meta_service_http.h"
#include "kv_cache_manager/service/meta_service_impl.h"
#include "kv_cache_manager/service/service_request_executor.h"
#include "kv_cache_manager/service/util/proto_message_json_util.h"
#include "kv_cache_manager/service/util/report_event_body_budget.h"
#include "ylt/coro_http/coro_http_client.hpp"

namespace kv_cache_manager {

namespace {

class TestHttpService final : public CoroHttpService {
public:
    void Init() override {}
    void RegisterHandler() override {}
};

struct BodyObservation {
    int status = 0;
    uint64_t connection_id = 0;
    size_t capacity = 0;
    size_t size = 0;
};

} // namespace

class CoroHttpServiceTest : public TESTBASE {
protected:
    void SetUp() override {
        CoroHttpService::HandlerType inspect =
            [](coro_http::coro_http_request &req,
               coro_http::coro_http_response &res) -> async_simple::coro::Lazy<void> {
            const std::string_view body = req.get_body();
            const std::string observation = std::to_string(req.get_conn()->conn_id_) + " " +
                                            std::to_string(req.get_conn()->body_.capacity()) + " " +
                                            std::to_string(body.size());
            const coro_http::status_type status =
                !body.empty() && body.front() == '!' ? coro_http::status_type::bad_request : coro_http::status_type::ok;
            res.set_status_and_content(status, observation);
            co_return;
        };
        server_.set_http_handler<coro_http::POST>("/inspect", service_.WrapWithLogger("/inspect", std::move(inspect)));
        server_task_ = std::make_unique<async_simple::Future<std::error_code>>(server_.async_start());
        ASSERT_GT(server_.port(), 0);
        url_ = "http://127.0.0.1:" + std::to_string(server_.port()) + "/inspect";
        client_ = std::make_shared<coro_http::coro_http_client>();
    }

    void TearDown() override {
        client_.reset();
        server_.stop();
        if (server_task_) {
            const std::error_code ec = std::move(*server_task_).get();
            EXPECT_TRUE(!ec || ec == asio::error::operation_aborted) << ec.message();
        }
    }

    BodyObservation Post(std::string body) {
        const coro_http::resp_data response = client_->post(url_, std::move(body), coro_http::req_content_type::none);
        EXPECT_FALSE(response.net_err) << response.net_err.message();
        BodyObservation observation;
        observation.status = response.status;
        std::istringstream fields{std::string(response.resp_body)};
        fields >> observation.connection_id >> observation.capacity >> observation.size;
        EXPECT_FALSE(fields.fail());
        return observation;
    }

private:
    TestHttpService service_;
    coro_http::coro_http_server server_{1, 0, std::string("127.0.0.1")};
    std::unique_ptr<async_simple::Future<std::error_code>> server_task_;
    std::shared_ptr<coro_http::coro_http_client> client_;
    std::string url_;
};

TEST_F(CoroHttpServiceTest, ShrinksLargeBodiesAndReusesSmallBuffersOnTheSameConnection) {
    constexpr size_t kThreshold = 8 * 1024 * 1024;
    const BodyObservation below_threshold = Post(std::string(kThreshold - 1, 'x'));
    ASSERT_EQ(200, below_threshold.status);
    ASSERT_EQ(kThreshold - 1, below_threshold.size);
    ASSERT_GE(below_threshold.capacity, below_threshold.size);

    const BodyObservation retained = Post("x");
    ASSERT_EQ(200, retained.status);
    EXPECT_EQ(below_threshold.connection_id, retained.connection_id);
    EXPECT_EQ(below_threshold.capacity, retained.capacity);

    const BodyObservation large = Post(std::string(kThreshold, 'x'));
    ASSERT_EQ(200, large.status);
    ASSERT_EQ(kThreshold, large.size);
    EXPECT_EQ(retained.connection_id, large.connection_id);
    EXPECT_GE(large.capacity, large.size);

    const BodyObservation shrunk = Post("x");
    ASSERT_EQ(200, shrunk.status);
    EXPECT_EQ(large.connection_id, shrunk.connection_id);
    EXPECT_LT(shrunk.capacity, kThreshold);

    const BodyObservation small = Post(std::string(4096, 'x'));
    ASSERT_EQ(200, small.status);
    const BodyObservation reused = Post("x");
    ASSERT_EQ(200, reused.status);
    EXPECT_EQ(large.connection_id, reused.connection_id);
    EXPECT_EQ(small.capacity, reused.capacity);
    EXPECT_GE(reused.capacity, small.size);
}

TEST_F(CoroHttpServiceTest, ShrinksLargeBodiesAfterHandlerErrors) {
    constexpr size_t kThreshold = 8 * 1024 * 1024;
    const BodyObservation rejected = Post(std::string(kThreshold + 1, '!'));
    ASSERT_EQ(400, rejected.status);
    ASSERT_EQ(kThreshold + 1, rejected.size);

    const BodyObservation next = Post("");
    ASSERT_EQ(200, next.status);
    EXPECT_EQ(rejected.connection_id, next.connection_id);
    EXPECT_EQ(0u, next.size);
    EXPECT_LT(next.capacity, kThreshold);
}

class AccessLogPayloadHttpTest : public TESTBASE {
protected:
    void SetUp() override {
        for (const bool include_access_log_payload : {false, true}) {
            const std::string prefix = include_access_log_payload ? "/full" : "/metadata";
            auto metrics_registry = std::make_shared<MetricsRegistry>();
            auto registry_manager = std::make_shared<RegistryManager>("", metrics_registry);
            auto cache_manager = std::make_shared<CacheManager>(metrics_registry, registry_manager);
            auto meta_impl =
                std::make_shared<MetaServiceImpl>(cache_manager, nullptr, nullptr, include_access_log_payload);
            auto executor = std::make_shared<ServiceRequestExecutor>();
            ASSERT_TRUE(executor->Init(1, 4));
            auto budget = std::make_shared<ReportEventBodyBudget>(128);
            auto service = std::make_unique<MetaServiceHttp>(
                metrics_registry, meta_impl, registry_manager, executor, nullptr, budget, nullptr);
            service->Init();
            service->RegisterHandler();
            for (const std::string api : {"/api/getInstanceInfo", "/api/reportEvent"}) {
                const std::string path = prefix + api;
                server_.set_http_handler<coro_http::POST>(
                    path, service->WrapWithLogger(path, service->post_handlers_.at(api)));
            }
            services_.push_back(std::move(service));
        }
        server_task_ = std::make_unique<async_simple::Future<std::error_code>>(server_.async_start());
        ASSERT_GT(server_.port(), 0);
    }

    void TearDown() override {
        server_.stop();
        if (server_task_) {
            const std::error_code ec = std::move(*server_task_).get();
            EXPECT_TRUE(!ec || ec == asio::error::operation_aborted) << ec.message();
        }
    }

    template <typename Response>
    void CheckResponses(const std::string &api) {
        std::string metadata_response;
        coro_http::coro_http_client client;
        for (const std::string prefix : {"/metadata", "/full"}) {
            SCOPED_TRACE(prefix + api);
            const std::string url = "http://127.0.0.1:" + std::to_string(server_.port()) + prefix + api;
            const coro_http::resp_data result =
                client.post(url, R"({"trace_id":"payload-mode"})", coro_http::req_content_type::none);
            ASSERT_FALSE(result.net_err) << result.net_err.message();
            ASSERT_EQ(200, result.status);

            Response response;
            ASSERT_TRUE(ProtoMessageJsonUtil::FromJson(result.resp_body, &response));
            ASSERT_TRUE(response.header().has_status());
            EXPECT_EQ(proto::meta::INVALID_ARGUMENT, response.header().status().code());
            EXPECT_FALSE(response.header().request_id().empty());
            response.mutable_header()->clear_request_id();

            if (prefix == "/metadata") {
                metadata_response = response.SerializeAsString();
            } else {
                EXPECT_EQ(metadata_response, response.SerializeAsString());
            }
        }
    }

private:
    std::vector<std::unique_ptr<MetaServiceHttp>> services_;
    coro_http::coro_http_server server_{1, 0, std::string("127.0.0.1")};
    std::unique_ptr<async_simple::Future<std::error_code>> server_task_;
};

TEST_F(AccessLogPayloadHttpTest, PreservesResponsesForFullJsonAndAccessLogSummaryHandlers) {
    ASSERT_NO_FATAL_FAILURE(CheckResponses<proto::meta::GetInstanceInfoResponse>("/api/getInstanceInfo"));
    ASSERT_NO_FATAL_FAILURE(CheckResponses<proto::meta::ReportEventResponse>("/api/reportEvent"));
}

} // namespace kv_cache_manager
