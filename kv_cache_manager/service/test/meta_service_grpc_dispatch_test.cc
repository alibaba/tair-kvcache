#include <atomic>
#include <chrono>
#include <future>
#include <grpcpp/generic/generic_stub.h>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/config/registry_manager.h"
#include "kv_cache_manager/manager/cache_manager.h"
#include "kv_cache_manager/manager/migration_manager.h"
#include "kv_cache_manager/metrics/local_metrics_reporter.h"
#include "kv_cache_manager/metrics/metrics_lifecycle.h"
#include "kv_cache_manager/metrics/metrics_registry.h"
#include "kv_cache_manager/protocol/protobuf/admin_service.grpc.pb.h"
#include "kv_cache_manager/service/grpc_service/grpc_server_options.h"
#include "kv_cache_manager/service/grpc_service/meta_service_grpc.h"
#include "kv_cache_manager/service/meta_service_impl.h"
#include "kv_cache_manager/service/service_request_executor.h"
#include "stub.h"

namespace kv_cache_manager {
namespace {

using namespace std::chrono_literals;

template <typename Predicate>
bool WaitUntil(Predicate predicate, std::chrono::milliseconds timeout = 5s) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (!predicate()) {
        if (std::chrono::steady_clock::now() >= deadline) {
            return false;
        }
        std::this_thread::sleep_for(1ms);
    }
    return true;
}

// The synchronous Admin interface must continue making progress when the
// explicit Meta CQ is the only frequently-polled CQ on their shared server.
class SyncAdmin final : public proto::admin::AdminService::Service {
public:
    grpc::Status CheckHealth(grpc::ServerContext *,
                             const proto::admin::CheckHealthRequest *,
                             proto::admin::CheckHealthResponse *response) override {
        response->set_is_health(true);
        return grpc::Status::OK;
    }
};

struct RpcResult {
    grpc::Status status;
    proto::meta::GetClusterInfoResponse response;
};

void ThrowingGetClusterInfo(MetaServiceImpl *,
                            RequestContext *,
                            const proto::meta::GetClusterInfoRequest *,
                            proto::meta::GetClusterInfoResponse *) {
    throw std::runtime_error("test business exception");
}

std::atomic<bool> migration_wait_entered{false};

std::vector<int64_t> KeepMigrationActive(const MigrationManager *, const std::string &instance_id) {
    if (instance_id == "removing-instance") {
        migration_wait_entered.store(true);
        return {42};
    }
    return {};
}

void ObserveQueuedRequestContext(MetaServiceImpl *,
                                 RequestContext *context,
                                 const proto::meta::GetClusterInfoRequest *,
                                 proto::meta::GetClusterInfoResponse *response) {
    EXPECT_NE(nullptr, context->metrics_collector());
    response->mutable_header()->set_request_id(context->request_id());
    response->set_self_node_id(std::to_string(context->request_begin_time_us()));
    response->set_leader_node_id(context->client_ip());
}

class MetaServiceGrpcDispatchTest : public TESTBASE {
public:
    void TearDown() override {
        ReleaseWorker();
        if (executor_) {
            executor_->CloseAdmission(ServiceRequestExecutor::ShutdownMode::kDrain);
        }
        if (service_) {
            service_->StopAccepting();
        }
        if (server_) {
            server_->Shutdown();
        }
        if (executor_) {
            executor_->Drain();
        }
        if (service_) {
            service_->WaitForCalls();
            service_->ShutdownCompletionQueueAndJoin();
        }
        if (executor_) {
            executor_->Join();
        }
    }

protected:
    void Start(std::size_t capacity = 0, bool mixed_admin = true, std::size_t worker_count = 1) {
        executor_ = std::make_shared<ServiceRequestExecutor>(worker_count, capacity);
        metrics_ = std::make_shared<MetricsRegistry>();
        metrics_lifecycle_ = std::make_shared<MetricsLifecycle>();
        registry_ = std::make_shared<RegistryManager>("", metrics_);
        cache_manager_ = std::make_shared<CacheManager>(metrics_, registry_, metrics_lifecycle_);
        impl_ = std::make_shared<MetaServiceImpl>(cache_manager_, nullptr, nullptr);
        service_ = std::make_shared<MetaServiceGRpc>(metrics_, impl_, registry_, executor_, metrics_lifecycle_);
        service_->Init();
        grpc_reporter_ = std::make_shared<LocalMetricsReporter>();
        ASSERT_TRUE(grpc_reporter_->Init(cache_manager_, metrics_, ""));

        grpc::ServerBuilder builder;
        EXPECT_EQ(2, ConfigureGrpcServer(builder, 2, 0, "meta", metrics_, grpc_reporter_));
        builder.AddChannelArgument("grpc.so_reuseport", 0);
        builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port_);
        builder.RegisterService(service_.get());
        if (mixed_admin) {
            builder.RegisterService(&admin_);
        }
        ASSERT_TRUE(service_->PrepareCompletionQueue(builder));
        server_ = builder.BuildAndStart();
        ASSERT_NE(nullptr, server_);
        ASSERT_GT(port_, 0);
        service_->StartAccepting();
        channel_ = grpc::CreateChannel("127.0.0.1:" + std::to_string(port_), grpc::InsecureChannelCredentials());
        ASSERT_TRUE(channel_->WaitForConnected(std::chrono::system_clock::now() + 5s));
    }

    void BlockWorker() {
        release_worker_ = std::make_shared<std::promise<void>>();
        auto release = release_worker_->get_future().share();
        auto entered = std::make_shared<std::promise<void>>();
        auto entered_future = entered->get_future();
        ServiceRequestExecutor::Task task;
        task.run = [release, entered](ServiceRequestExecutor::CompletionToken) {
            entered->set_value();
            release.wait();
        };
        task.on_discard = [](ServiceRequestExecutor::CompletionToken) {};
        ASSERT_EQ(ServiceRequestExecutor::SubmitResult::kAccepted, executor_->TrySubmit(std::move(task)));
        ASSERT_EQ(std::future_status::ready, entered_future.wait_for(5s));
    }

    void ReleaseWorker() {
        if (release_worker_) {
            release_worker_->set_value();
            release_worker_.reset();
        }
    }

    RpcResult GetClusterInfo(grpc::ClientContext *context = nullptr) {
        grpc::ClientContext default_context;
        default_context.set_deadline(std::chrono::system_clock::now() + 5s);
        auto stub = proto::meta::MetaService::NewStub(channel_);
        proto::meta::GetClusterInfoRequest request;
        request.set_trace_id("dispatch-test");
        RpcResult result;
        result.status = stub->GetClusterInfo(context ? context : &default_context, request, &result.response);
        return result;
    }

    std::future<RpcResult> BeginGetClusterInfo() {
        return std::async(std::launch::async, [this] { return GetClusterInfo(); });
    }

    uint64_t GrpcResponseCount(const std::string &code, const std::string &method = "MetaService/GetClusterInfo") {
        return metrics_
            ->GetCounter("grpc.response_counter", {{"server", "meta"}, {"method", method}, {"grpc_code", code}})
            .Get();
    }

    std::pair<grpc::Status, proto::meta::GetCacheMetaResponse>
    GetMissingInstance(std::chrono::milliseconds timeout = 2s) {
        auto stub = proto::meta::MetaService::NewStub(channel_);
        grpc::ClientContext context;
        context.set_deadline(std::chrono::system_clock::now() + timeout);
        proto::meta::GetCacheMetaRequest request;
        request.set_trace_id("missing-dispatch-test");
        request.set_instance_id("uncached-instance");
        proto::meta::GetCacheMetaResponse response;
        auto status = stub->GetCacheMeta(&context, request, &response);
        return std::make_pair(status, response);
    }

    void CheckAdmin(const std::shared_ptr<grpc::Channel> &channel) {
        auto stub = proto::admin::AdminService::NewStub(channel);
        grpc::ClientContext context;
        context.set_deadline(std::chrono::system_clock::now() + 2s);
        proto::admin::CheckHealthRequest request;
        proto::admin::CheckHealthResponse response;
        EXPECT_TRUE(stub->CheckHealth(&context, request, &response).ok());
        EXPECT_TRUE(response.is_health());
    }

    grpc::Status RawUnary(const std::string &method, const std::string &payload) {
        grpc::GenericStub stub(channel_);
        grpc::ClientContext context;
        context.set_deadline(std::chrono::system_clock::now() + 2s);
        grpc::Slice slice(payload);
        grpc::ByteBuffer request(&slice, 1);
        grpc::ByteBuffer response;
        grpc::CompletionQueue cq;
        grpc::Status status;
        auto call = stub.PrepareUnaryCall(&context, method, request, &cq);
        call->StartCall();
        int finish_tag = 0;
        call->Finish(&response, &status, &finish_tag);
        void *tag = nullptr;
        bool ok = false;
        EXPECT_TRUE(cq.Next(&tag, &ok));
        EXPECT_EQ(&finish_tag, tag);
        EXPECT_TRUE(ok);
        cq.Shutdown();
        EXPECT_FALSE(cq.Next(&tag, &ok));
        return status;
    }

    std::shared_ptr<ServiceRequestExecutor> executor_;
    std::shared_ptr<MetricsRegistry> metrics_;
    std::shared_ptr<MetricsLifecycle> metrics_lifecycle_;
    std::shared_ptr<LocalMetricsReporter> grpc_reporter_;
    std::shared_ptr<RegistryManager> registry_;
    std::shared_ptr<CacheManager> cache_manager_;
    std::shared_ptr<MetaServiceImpl> impl_;
    std::shared_ptr<MetaServiceGRpc> service_;
    SyncAdmin admin_;
    std::unique_ptr<grpc::Server> server_;
    std::shared_ptr<grpc::Channel> channel_;
    std::shared_ptr<std::promise<void>> release_worker_;
    int port_ = 0;
};

TEST_F(MetaServiceGrpcDispatchTest, QueuedMetaDoesNotBlockSharedOrIndependentAdmin) {
    Start();
    BlockWorker();
    auto pending = BeginGetClusterInfo();
    ASSERT_TRUE(WaitUntil([this] { return executor_->GetSnapshot().queued == 1; }));
    CheckAdmin(channel_);
    EXPECT_EQ(1u, GrpcResponseCount("OK", "AdminService/CheckHealth"));

    SyncAdmin independent_admin;
    grpc::ServerBuilder builder;
    int port = 0;
    builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
    builder.RegisterService(&independent_admin);
    auto server = builder.BuildAndStart();
    ASSERT_NE(nullptr, server);
    CheckAdmin(grpc::CreateChannel("127.0.0.1:" + std::to_string(port), grpc::InsecureChannelCredentials()));
    server->Shutdown();

    ReleaseWorker();
    auto result = pending.get();
    EXPECT_TRUE(result.status.ok());
    EXPECT_EQ(proto::meta::SERVICE_NOT_READY, result.response.header().status().code());
    EXPECT_EQ("Leader elector not initialized", result.response.header().status().message());
    EXPECT_FALSE(result.response.header().request_id().empty());
    EXPECT_EQ(1u, GrpcResponseCount("OK"));
}

TEST_F(MetaServiceGrpcDispatchTest, DefaultQueueDoesNotRejectWhenWorkerIsBusy) {
    Start();
    BlockWorker();
    std::vector<std::future<RpcResult>> pending;
    for (int i = 0; i < 8; ++i) {
        pending.push_back(BeginGetClusterInfo());
    }
    ASSERT_TRUE(WaitUntil([this] { return executor_->GetSnapshot().queued == 8; }));
    EXPECT_EQ(0, executor_->GetSnapshot().rejected);
    ReleaseWorker();
    for (auto &rpc : pending) {
        auto result = rpc.get();
        EXPECT_TRUE(result.status.ok());
        EXPECT_EQ("Leader elector not initialized", result.response.header().status().message());
    }
}

TEST_F(MetaServiceGrpcDispatchTest, CollectorPreparationPreservesArrivalContextAcrossQueueWait) {
    Start();
    BlockWorker();
    Stub stub;
    stub.set(ADDR(MetaServiceImpl, GetClusterInfo), ObserveQueuedRequestContext);
    auto pending = BeginGetClusterInfo();
    ASSERT_TRUE(WaitUntil([this] { return executor_->GetSnapshot().queued == 1; }));
    const auto queued_at = TimestampUtil::GetCurrentTimeUs();
    ReleaseWorker();
    const auto result = pending.get();
    ASSERT_TRUE(result.status.ok());
    EXPECT_LE(std::stoll(result.response.self_node_id()), queued_at);
    EXPECT_EQ("127.0.0.1", result.response.leader_node_id());
    EXPECT_EQ(0, result.response.header().request_id().find("dispatch-test_"));
}

TEST_F(MetaServiceGrpcDispatchTest, RemoveInstanceMigrationWaitDoesNotBlockCqOrSharedAdmin) {
    Start(0, true, 2);
    ASSERT_TRUE(registry_->Init());
    ASSERT_TRUE(cache_manager_->Init());

    // Exercise the actual RemoveInstance migration-drain loop and its lifecycle
    // write lock. A synthetic active key keeps cancellation pending until the
    // production five-second timeout, without depending on an external backend.
    Stub stub;
    migration_wait_entered.store(false);
    stub.set(ADDR(MigrationManager, GetActiveBlockKeysForInstance), KeepMigrationActive);
    auto removal = std::async(std::launch::async, [this] {
        RequestContext context("remove-during-collector-lookup");
        return cache_manager_->RemoveInstance(&context, "", "removing-instance");
    });
    ASSERT_TRUE(WaitUntil([] { return migration_wait_entered.load(); }));

    auto pending = std::async(std::launch::async, [this] { return GetMissingInstance(10s); });
    EXPECT_TRUE(WaitUntil([this] { return executor_->GetSnapshot().running == 1; }, 1s));
    EXPECT_EQ(std::future_status::timeout, removal.wait_for(0ms));
    CheckAdmin(channel_);
    // GetClusterInfo's empty-instance collector is warm. Its request must
    // still be dispatched to the other worker while the cold lookup waits.
    grpc::ClientContext warm_context;
    warm_context.set_deadline(std::chrono::system_clock::now() + 2s);
    EXPECT_TRUE(GetClusterInfo(&warm_context).status.ok());
    EXPECT_EQ(std::future_status::timeout, pending.wait_for(0ms));

    EXPECT_EQ(EC_TIMEOUT, removal.get());
    const auto result = pending.get();
    EXPECT_TRUE(result.first.ok());
    EXPECT_EQ(proto::meta::INSTANCE_NOT_EXIST, result.second.header().status().code());
    EXPECT_EQ("get GetCacheMeta metrics collector failed", result.second.header().status().message());
    EXPECT_TRUE(result.second.header().request_id().empty());
    EXPECT_TRUE(GetClusterInfo().status.ok());
}

TEST_F(MetaServiceGrpcDispatchTest, BusinessExceptionPreservesSynchronousGrpcStatus) {
    Start();
    {
        Stub stub;
        stub.set(ADDR(MetaServiceImpl, GetClusterInfo), ThrowingGetClusterInfo);
        auto result = GetClusterInfo();
        EXPECT_EQ(grpc::StatusCode::UNKNOWN, result.status.error_code());
        EXPECT_EQ("Unexpected error in RPC handling", result.status.error_message());
        EXPECT_EQ(1u, GrpcResponseCount("UNKNOWN"));
    }
    EXPECT_TRUE(GetClusterInfo().status.ok());
    EXPECT_EQ(1u, GrpcResponseCount("OK"));
}

TEST_F(MetaServiceGrpcDispatchTest, ConfiguredQueueLimitReturnsBusinessError) {
    Start(1);
    BlockWorker();
    auto pending = BeginGetClusterInfo();
    ASSERT_TRUE(WaitUntil([this] { return executor_->GetSnapshot().queued == 1; }));
    auto rejected = GetClusterInfo();
    EXPECT_TRUE(rejected.status.ok());
    EXPECT_EQ(proto::meta::SERVICE_NOT_READY, rejected.response.header().status().code());
    EXPECT_EQ("service request queue is full", rejected.response.header().status().message());
    EXPECT_FALSE(rejected.response.header().request_id().empty());
    EXPECT_EQ(1u, GrpcResponseCount("OK"));
    EXPECT_EQ(0u, GrpcResponseCount("RESOURCE_EXHAUSTED"));
    {
        std::unique_lock<std::shared_mutex> hold_lifecycle(metrics_lifecycle_->mut_);
        const auto missing = GetMissingInstance();
        EXPECT_TRUE(missing.first.ok());
        EXPECT_EQ(proto::meta::SERVICE_NOT_READY, missing.second.header().status().code());
        EXPECT_EQ("service request queue is full", missing.second.header().status().message());
        EXPECT_FALSE(missing.second.header().request_id().empty());
        EXPECT_EQ(1u, GrpcResponseCount("OK", "MetaService/GetCacheMeta"));
    }
    ReleaseWorker();
    EXPECT_TRUE(pending.get().status.ok());
    EXPECT_EQ(2u, GrpcResponseCount("OK"));
}

TEST_F(MetaServiceGrpcDispatchTest, DiscardPendingCompletesCallsWithoutWaitingForBusyWorker) {
    Start();
    BlockWorker();
    auto pending = BeginGetClusterInfo();
    ASSERT_TRUE(WaitUntil([this] { return executor_->GetSnapshot().queued == 1; }));
    executor_->CloseAdmission(ServiceRequestExecutor::ShutdownMode::kDiscardPending);
    ASSERT_EQ(std::future_status::ready, pending.wait_for(2s));
    auto result = pending.get();
    EXPECT_TRUE(result.status.ok());
    EXPECT_EQ(proto::meta::SERVICE_NOT_READY, result.response.header().status().code());
    EXPECT_EQ("service is stopping", result.response.header().status().message());
    EXPECT_FALSE(result.response.header().request_id().empty());
    EXPECT_EQ(1, executor_->GetSnapshot().discarded);

    auto closed = GetClusterInfo();
    EXPECT_TRUE(closed.status.ok());
    EXPECT_EQ("service is stopping", closed.response.header().status().message());
    EXPECT_EQ(2u, GrpcResponseCount("OK"));
    EXPECT_EQ(0u, GrpcResponseCount("RESOURCE_EXHAUSTED"));
    {
        std::unique_lock<std::shared_mutex> hold_lifecycle(metrics_lifecycle_->mut_);
        const auto missing = GetMissingInstance();
        EXPECT_TRUE(missing.first.ok());
        EXPECT_EQ(proto::meta::SERVICE_NOT_READY, missing.second.header().status().code());
        EXPECT_EQ("service is stopping", missing.second.header().status().message());
        EXPECT_FALSE(missing.second.header().request_id().empty());
        EXPECT_EQ(1u, GrpcResponseCount("OK", "MetaService/GetCacheMeta"));
    }
}

TEST_F(MetaServiceGrpcDispatchTest, DrainKeepsQueuedCallsAliveThroughServerShutdown) {
    Start();
    BlockWorker();
    auto pending = BeginGetClusterInfo();
    ASSERT_TRUE(WaitUntil([this] { return executor_->GetSnapshot().queued == 1; }));
    executor_->CloseAdmission(ServiceRequestExecutor::ShutdownMode::kDrain);
    service_->StopAccepting();
    auto shutdown = std::async(std::launch::async, [this] { server_->Shutdown(); });
    EXPECT_EQ(std::future_status::timeout, shutdown.wait_for(50ms));
    ReleaseWorker();
    EXPECT_TRUE(pending.get().status.ok());
    ASSERT_EQ(std::future_status::ready, shutdown.wait_for(5s));
    shutdown.get();
    executor_->Drain();
    service_->WaitForCalls();
}

TEST_F(MetaServiceGrpcDispatchTest, DeadlineAndCancellationStillDrainFinishEvents) {
    Start();
    BlockWorker();
    grpc::ClientContext deadline_context;
    deadline_context.set_deadline(std::chrono::system_clock::now() + 200ms);
    auto deadline =
        std::async(std::launch::async, [this, &deadline_context] { return GetClusterInfo(&deadline_context); });
    ASSERT_TRUE(WaitUntil([this] { return executor_->GetSnapshot().queued == 1; }));
    EXPECT_EQ(grpc::StatusCode::DEADLINE_EXCEEDED, deadline.get().status.error_code());

    grpc::ClientContext cancel_context;
    cancel_context.set_deadline(std::chrono::system_clock::now() + 5s);
    auto cancelled =
        std::async(std::launch::async, [this, &cancel_context] { return GetClusterInfo(&cancel_context); });
    ASSERT_TRUE(WaitUntil([this] { return executor_->GetSnapshot().queued == 2; }));
    cancel_context.TryCancel();
    EXPECT_EQ(grpc::StatusCode::CANCELLED, cancelled.get().status.error_code());
    ReleaseWorker();
    executor_->Drain();
    EXPECT_EQ(0, executor_->GetSnapshot().discarded);
    // A cancelled Finish must not terminate the CQ poller.
    EXPECT_TRUE(GetClusterInfo().status.ok());
}

TEST_F(MetaServiceGrpcDispatchTest, PureAsyncServerHandlesUnknownMethodAndMalformedProtobuf) {
    Start(0, false);
    EXPECT_EQ(grpc::StatusCode::UNIMPLEMENTED,
              RawUnary("/kv_cache_manager.proto.meta.MetaService/UnknownMethod", "").error_code());
    // Truncated length-delimited trace_id. The pinned gRPC AsyncService rejects
    // it before the application accept tag, then re-arms the same request.
    const std::string method = "/kv_cache_manager.proto.meta.MetaService/GetClusterInfo";
    const std::string malformed_payload("\x0a\x02x", 3);
    const auto async_status = RawUnary(method, malformed_payload);
    EXPECT_EQ(grpc::StatusCode::INTERNAL, async_status.error_code());
    EXPECT_EQ("Unable to parse request", async_status.error_message());

    // Exercise the previous synchronous transport with exactly the same wire
    // payload. Malformed protobuf never enters either business handler.
    class SyncMeta final : public proto::meta::MetaService::Service {
    public:
        grpc::Status GetClusterInfo(grpc::ServerContext *,
                                    const proto::meta::GetClusterInfoRequest *,
                                    proto::meta::GetClusterInfoResponse *) override {
            return grpc::Status::OK;
        }
    } sync_service;
    grpc::ServerBuilder builder;
    int sync_port = 0;
    builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &sync_port);
    builder.RegisterService(&sync_service);
    auto sync_server = builder.BuildAndStart();
    ASSERT_NE(nullptr, sync_server);
    auto async_channel = channel_;
    channel_ = grpc::CreateChannel("127.0.0.1:" + std::to_string(sync_port), grpc::InsecureChannelCredentials());
    const auto sync_status = RawUnary(method, malformed_payload);
    channel_ = std::move(async_channel);
    sync_server->Shutdown();

    EXPECT_EQ(grpc::StatusCode::INTERNAL, sync_status.error_code());
    EXPECT_NE(sync_status.error_message(), async_status.error_message());
    std::cout << "Malformed protobuf sync transport: code=" << sync_status.error_code() << ", message=["
              << sync_status.error_message() << "]\n"
              << "Malformed protobuf async transport: code=" << async_status.error_code() << ", message=["
              << async_status.error_message() << "]" << std::endl;
    EXPECT_TRUE(GetClusterInfo().status.ok());
}

TEST_F(MetaServiceGrpcDispatchTest, FailedBindCanStopPreparedPollerWithoutPostingRequests) {
    Start();
    MetaServiceGRpc second(metrics_, impl_, registry_, executor_);
    second.Init();
    grpc::ServerBuilder builder;
    builder.AddChannelArgument("grpc.so_reuseport", 0);
    builder.AddListeningPort("127.0.0.1:" + std::to_string(port_), grpc::InsecureServerCredentials());
    builder.RegisterService(&second);
    ASSERT_TRUE(second.PrepareCompletionQueue(builder));
    auto failed = builder.BuildAndStart();
    EXPECT_EQ(nullptr, failed);
    second.StopAccepting();
    second.WaitForCalls();
    second.ShutdownCompletionQueueAndJoin();
    EXPECT_TRUE(GetClusterInfo().status.ok());
}

} // namespace
} // namespace kv_cache_manager
