#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <future>
#include <grpcpp/generic/generic_stub.h>
#include <grpcpp/grpcpp.h>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/config/registry_manager.h"
#include "kv_cache_manager/manager/cache_manager.h"
#include "kv_cache_manager/metrics/local_metrics_reporter.h"
#include "kv_cache_manager/metrics/metrics_registry.h"
#include "kv_cache_manager/metrics/prometheus_exporter.h"
#include "kv_cache_manager/protocol/protobuf/admin_service.grpc.pb.h"
#include "kv_cache_manager/protocol/protobuf/meta_service.grpc.pb.h"
#include "kv_cache_manager/service/grpc_service/grpc_server_options.h"

namespace kv_cache_manager {
namespace {

constexpr const char *kClusterInfoMethod = "MetaService/GetClusterInfo";
constexpr const char *kCacheLocationMethod = "MetaService/GetCacheLocation";
constexpr const char *kMetaCacheMetaMethod = "MetaService/GetCacheMeta";
constexpr const char *kAdminCacheMetaMethod = "AdminService/GetCacheMeta";

class TestMetaService final : public proto::meta::MetaService::Service {
public:
    grpc::Status GetCacheMeta(grpc::ServerContext *,
                              const proto::meta::GetCacheMetaRequest *,
                              proto::meta::GetCacheMetaResponse *) override {
        return grpc::Status::OK;
    }

    grpc::Status GetClusterInfo(grpc::ServerContext *,
                                const proto::meta::GetClusterInfoRequest *request,
                                proto::meta::GetClusterInfoResponse *response) override {
        ++query_calls;
        if (request->trace_id() == "business_error") {
            response->mutable_header()->mutable_status()->set_code(proto::meta::INVALID_ARGUMENT);
            return grpc::Status::OK;
        }
        return grpc::Status(static_cast<grpc::StatusCode>(std::stoi(request->trace_id())), "");
    }

    grpc::Status GetCacheLocation(grpc::ServerContext *,
                                  const proto::meta::GetCacheLocationRequest *,
                                  proto::meta::GetCacheLocationResponse *) override {
        std::unique_lock lock(mutex_);
        entered_ = true;
        cv_.notify_all();
        cv_.wait(lock, [this] { return released_; });
        return grpc::Status::OK;
    }

    bool WaitUntilEntered() {
        std::unique_lock lock(mutex_);
        return cv_.wait_for(lock, std::chrono::seconds(5), [this] { return entered_; });
    }

    void Release() {
        std::lock_guard lock(mutex_);
        released_ = true;
        cv_.notify_all();
    }

    std::atomic<int> query_calls{0};

private:
    std::mutex mutex_;
    std::condition_variable cv_;
    bool entered_ = false;
    bool released_ = false;
};

class TestAdminService final : public proto::admin::AdminService::Service {
public:
    grpc::Status GetCacheMeta(grpc::ServerContext *,
                              const proto::admin::GetCacheMetaRequest *,
                              proto::admin::GetCacheMetaResponse *) override {
        return grpc::Status::OK;
    }
};

grpc::Status GenericQuery(const std::shared_ptr<grpc::Channel> &channel, const std::string &method) {
    grpc::GenericStub stub(channel);
    grpc::ClientContext context;
    context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(5));
    grpc::CompletionQueue queue;
    const grpc::Slice slice("", 0);
    const grpc::ByteBuffer request(&slice, 1);
    grpc::ByteBuffer response;
    auto call = stub.PrepareUnaryCall(&context, method, request, &queue);
    if (!call) {
        queue.Shutdown();
        return grpc::Status(grpc::StatusCode::INTERNAL, "Failed to prepare generic RPC");
    }
    grpc::Status status;
    call->StartCall();
    call->Finish(&response, &status, &status);
    void *tag = nullptr;
    bool ok = false;
    EXPECT_TRUE(queue.Next(&tag, &ok));
    EXPECT_EQ(&status, tag);
    EXPECT_TRUE(ok);
    queue.Shutdown();
    while (queue.Next(&tag, &ok)) {}
    return status;
}

grpc::Status Query(proto::meta::MetaService::Stub &stub, const std::string &status = "0") {
    grpc::ClientContext context;
    context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(5));
    proto::meta::GetClusterInfoRequest request;
    request.set_trace_id(status);
    proto::meta::GetClusterInfoResponse response;
    return stub.GetClusterInfo(&context, request, &response);
}

} // namespace

TEST(GrpcMaxThreadsTest, ResolvesConfiguredAndAutomaticLimits) {
    EXPECT_EQ(-1, ResolveGrpcMaxThreads(-1, 64));
    EXPECT_EQ(8, ResolveGrpcMaxThreads(8, 64));
    EXPECT_EQ(64, ResolveGrpcMaxThreads(0, 64));
    EXPECT_EQ(2, ResolveGrpcMaxThreads(0, 2));
    EXPECT_EQ(2, ResolveGrpcMaxThreads(0, 1));
    EXPECT_EQ(2, ResolveGrpcMaxThreads(0, 0));
    EXPECT_EQ(std::numeric_limits<int>::max(), ResolveGrpcMaxThreads(0, std::numeric_limits<unsigned int>::max()));
}

class GrpcServerOptionsTest : public TESTBASE, public testing::WithParamInterface<std::string> {
protected:
    void SetUp() override {
        registry_ = std::make_shared<MetricsRegistry>();
        auto registry_manager = std::make_shared<RegistryManager>("", registry_);
        auto cache_manager = std::make_shared<CacheManager>(registry_, registry_manager);
        reporter_ = std::make_shared<LocalMetricsReporter>();
        ASSERT_TRUE(reporter_->Init(cache_manager, registry_, ""));
    }

    void TearDown() override {
        // A fixture-owned future lets fatal assertions still release the blocked
        // handler before waiting for its thread or shutting down the server.
        service_.Release();
        if (blocked_.valid()) {
            blocked_.get();
        }
        if (server_) {
            server_->Shutdown();
        }
    }

    void StartServer(int max_threads, std::size_t memory_quota_bytes = 0, bool include_admin_service = false) {
        grpc::ServerBuilder builder;
        max_threads_ = ConfigureGrpcServer(builder, max_threads, memory_quota_bytes, GetParam(), registry_, reporter_);
        int port = 0;
        builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
        builder.RegisterService(&service_);
        if (include_admin_service) {
            builder.RegisterService(&admin_service_);
        }
        server_ = builder.BuildAndStart();
        ASSERT_NE(nullptr, server_);
        ASSERT_GT(port, 0);
        channel_ = grpc::CreateChannel("127.0.0.1:" + std::to_string(port), grpc::InsecureChannelCredentials());
        stub_ = proto::meta::MetaService::NewStub(channel_);
    }

    void StartBlockedRequest() {
        blocked_ = std::async(std::launch::async, [this] {
            grpc::ClientContext context;
            context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(10));
            proto::meta::GetCacheLocationRequest request;
            proto::meta::GetCacheLocationResponse response;
            return stub_->GetCacheLocation(&context, request, &response);
        });
    }

    uint64_t
    Count(const std::string &code, const std::string &server_name, const std::string &method = kClusterInfoMethod) {
        return registry_
            ->GetCounter("grpc.response_counter", {{"server", server_name}, {"method", method}, {"grpc_code", code}})
            .Get();
    }

    TestMetaService service_;
    TestAdminService admin_service_;
    std::shared_ptr<MetricsRegistry> registry_;
    std::shared_ptr<LocalMetricsReporter> reporter_;
    std::unique_ptr<grpc::Server> server_;
    std::shared_ptr<grpc::Channel> channel_;
    std::unique_ptr<proto::meta::MetaService::Stub> stub_;
    std::future<grpc::Status> blocked_;
    int max_threads_ = -1;
};

TEST_P(GrpcServerOptionsTest, CountsEveryGrpcStatusWithDefaultThreadQuota) {
    StartServer(-1);
    ASSERT_NE(nullptr, stub_);
    EXPECT_EQ(-1, max_threads_);
    for (int code = 0; code <= 16; ++code) {
        SCOPED_TRACE(code);
        EXPECT_EQ(code, Query(*stub_, std::to_string(code)).error_code());
    }
    // The new metric describes the transport status, not the response header.
    EXPECT_TRUE(Query(*stub_, "business_error").ok());
    EXPECT_EQ(2u, Count("OK", GetParam()));
    EXPECT_EQ(1u, Count("RESOURCE_EXHAUSTED", GetParam()));
    EXPECT_EQ(1u, Count("UNAUTHENTICATED", GetParam()));

    const std::string output = PrometheusExporter::Expose(*registry_);
    EXPECT_NE(output.find("kvcm_grpc_response_counter{grpc_code=\"RESOURCE_EXHAUSTED\",method=\"" +
                          std::string(kClusterInfoMethod) + "\",server=\"" + GetParam() + "\"} 1\n"),
              std::string::npos)
        << output;
    uint64_t responses = 0;
    std::vector<MetricsRegistry::metrics_tuple_t> all_metrics;
    registry_->GetAllMetrics(all_metrics);
    for (const auto &[name, tags, value] : all_metrics) {
        if (name == "grpc.response_counter") {
            EXPECT_EQ(GetParam(), tags.at("server"));
            responses += std::get<CounterValue>(value->value).load();
        }
    }
    EXPECT_EQ(18u, responses);
}

TEST_P(GrpcServerOptionsTest, CountsFrameworkRejectionsAndRecovers) {
    StartServer(2, 128 * 1024 * 1024); // one handler plus one polling/rejection thread, with a memory quota
    ASSERT_NE(nullptr, stub_);
    StartBlockedRequest();
    ASSERT_TRUE(service_.WaitUntilEntered());

    constexpr int kRejected = 8;
    for (int i = 0; i < kRejected; ++i) {
        EXPECT_EQ(grpc::StatusCode::RESOURCE_EXHAUSTED, Query(*stub_).error_code());
    }
    EXPECT_EQ(0, service_.query_calls.load()); // rejected before the business handler
    EXPECT_EQ(kRejected, Count("RESOURCE_EXHAUSTED", GetParam()));

    service_.Release();
    EXPECT_TRUE(blocked_.get().ok());
    // Receiving the first response can precede the worker returning to polling.
    // Account for transient rejections while checking that capacity recovers.
    uint64_t rejections = kRejected;
    bool recovered = false;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (std::chrono::steady_clock::now() < deadline) {
        const auto status = Query(*stub_);
        if (status.ok()) {
            recovered = true;
            break;
        }
        ASSERT_EQ(grpc::StatusCode::RESOURCE_EXHAUSTED, status.error_code());
        ++rejections;
    }
    EXPECT_TRUE(recovered);
    EXPECT_EQ(1, service_.query_calls.load());
    EXPECT_EQ(1u, Count("OK", GetParam()));
    EXPECT_EQ(1u, Count("OK", GetParam(), kCacheLocationMethod));
    EXPECT_EQ(rejections, Count("RESOURCE_EXHAUSTED", GetParam()));
}

TEST_P(GrpcServerOptionsTest, ServersHaveIndependentThreadQuotas) {
    StartServer(2);
    ASSERT_NE(nullptr, stub_);
    StartBlockedRequest();
    ASSERT_TRUE(service_.WaitUntilEntered());

    const std::string other_name = GetParam() == "meta" ? "admin" : "meta";
    TestMetaService other_service;
    grpc::ServerBuilder builder;
    ConfigureGrpcServer(builder, 2, 0, other_name, registry_, reporter_);
    int port = 0;
    builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
    builder.RegisterService(&other_service);
    auto other_server = builder.BuildAndStart();
    ASSERT_NE(nullptr, other_server);
    auto other_stub = proto::meta::MetaService::NewStub(
        grpc::CreateChannel("127.0.0.1:" + std::to_string(port), grpc::InsecureChannelCredentials()));

    EXPECT_EQ(grpc::StatusCode::RESOURCE_EXHAUSTED, Query(*stub_).error_code());
    EXPECT_TRUE(Query(*other_stub).ok());
    EXPECT_EQ(1u, Count("RESOURCE_EXHAUSTED", GetParam()));
    EXPECT_EQ(1u, Count("OK", other_name));
    EXPECT_EQ(0u, Count("RESOURCE_EXHAUSTED", other_name));
    other_server->Shutdown();
}

TEST_P(GrpcServerOptionsTest, AutomaticThreadQuotaAppliesAndServesRequests) {
    StartServer(0);
    ASSERT_NE(nullptr, stub_);
    EXPECT_EQ(ResolveGrpcMaxThreads(0, std::thread::hardware_concurrency()), max_threads_);
    EXPECT_GE(max_threads_, 2);
    EXPECT_TRUE(Query(*stub_).ok());
    EXPECT_EQ(1u, Count("OK", GetParam()));
}

TEST_P(GrpcServerOptionsTest, SameNamedMethodsInDifferentServicesHaveSeparateCounters) {
    StartServer(-1, 0, true);
    ASSERT_NE(nullptr, stub_);
    const std::string meta_path = "/kv_cache_manager.proto.meta.MetaService/GetCacheMeta";
    const std::string admin_path = "/kv_cache_manager.proto.admin.AdminService/GetCacheMeta";
    EXPECT_TRUE(GenericQuery(channel_, meta_path).ok());
    EXPECT_TRUE(GenericQuery(channel_, meta_path).ok());
    EXPECT_TRUE(GenericQuery(channel_, admin_path).ok());
    EXPECT_EQ(grpc::StatusCode::RESOURCE_EXHAUSTED, Query(*stub_, "8").error_code());

    EXPECT_EQ(2u, Count("OK", GetParam(), kMetaCacheMetaMethod));
    EXPECT_EQ(1u, Count("OK", GetParam(), kAdminCacheMetaMethod));
    EXPECT_EQ(0u, Count("OK", GetParam()));
    EXPECT_EQ(1u, Count("RESOURCE_EXHAUSTED", GetParam()));
    EXPECT_EQ(0u, Count("RESOURCE_EXHAUSTED", GetParam(), kMetaCacheMetaMethod));
    EXPECT_EQ(0u, Count("RESOURCE_EXHAUSTED", GetParam(), kAdminCacheMetaMethod));
}

TEST_P(GrpcServerOptionsTest, UnknownMethodsShareOneBoundedLabel) {
    StartServer(-1);
    ASSERT_NE(nullptr, stub_);
    const auto metric_count = registry_->GetSize();
    constexpr int kRequests = 8;
    for (int i = 0; i < kRequests; ++i) {
        EXPECT_EQ(grpc::StatusCode::UNIMPLEMENTED,
                  GenericQuery(channel_, "/unknown.Service/Method" + std::to_string(i)).error_code());
    }
    EXPECT_EQ(metric_count, registry_->GetSize());
    EXPECT_EQ(kRequests, Count("UNIMPLEMENTED", GetParam(), "unknown"));
}

TEST_P(GrpcServerOptionsTest, MemoryQuotaAppliesWithoutThreadLimit) {
    // This budget cannot accommodate even one RPC. Transport-level rejection
    // may close the connection or wait for the deadline, rather than send a status.
    StartServer(-1, 1);
    ASSERT_NE(nullptr, stub_);
    EXPECT_EQ(-1, max_threads_);
    EXPECT_FALSE(Query(*stub_).ok());
    EXPECT_EQ(0, service_.query_calls.load());
}

INSTANTIATE_TEST_SUITE_P(ServerRoles, GrpcServerOptionsTest, testing::Values("meta", "admin"));

} // namespace kv_cache_manager
