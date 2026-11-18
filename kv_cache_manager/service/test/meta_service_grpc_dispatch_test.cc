#include <atomic>
#include <chrono>
#include <future>
#include <google/protobuf/wire_format_lite.h>
#include <grpcpp/generic/generic_stub.h>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/common/standard_uri.h"
#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/config/coordination_memory_backend.h"
#include "kv_cache_manager/config/leader_elector.h"
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
#include "kv_cache_manager/service/server.h"
#include "kv_cache_manager/service/service_request_executor.h"
#include "kv_cache_manager/service/util/common.h"
#include "src/core/ext/transport/chttp2/transport/internal.h"
#include "src/core/lib/gpr/mpscq.h"
#include "src/core/lib/transport/timeout_encoding.h"
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

std::string ThrowingExtractIp(const std::string &) { throw std::bad_alloc(); }

std::atomic<bool> migration_wait_entered{false};
std::atomic<int> executed_cluster_calls{0};
std::atomic<bool> block_next_decode{false};
std::atomic<bool> decode_entered{false};
std::shared_future<void> release_decode;
std::atomic<bool> receive_post_entered{false};
std::shared_future<void> release_receive_post;

std::atomic<bool> block_next_request_push{false};
std::atomic<bool> request_push_entered{false};
std::atomic<int> first_request_decoded{0};
std::shared_future<void> release_request_push;

std::atomic<bool> timeout_decode_entered{false};
std::shared_future<void> release_timeout_decode;

bool DecodeTimeoutWithStaleClock(grpc_slice value, grpc_millis *timeout) {
    timeout_decode_entered.store(true);
    release_timeout_decode.wait();
    // The test restores the decoder before releasing this callback.
    const bool decoded = grpc_http2_decode_timeout(value, timeout);
    auto *context = grpc_core::ExecCtx::Get();
    context->TestOnlySetNow(context->Now() - 60000);
    return decoded;
}

struct TransportTraffic {
    grpc_combiner *combiner = nullptr;
    grpc_closure next;
    std::atomic<bool> selected{false};
    std::atomic<bool> entered{false};
    std::atomic<bool> stop{false};
    std::atomic<bool> finished{false};
    bool server = false;
    std::shared_future<void> resume;
    std::chrono::steady_clock::time_point deadline;
};

std::atomic<TransportTraffic *> transport_traffic{nullptr};

void ContinueTransportTraffic(void *arg, grpc_error *) {
    auto *traffic = static_cast<TransportTraffic *>(arg);
    if (traffic->stop.load() || std::chrono::steady_clock::now() >= traffic->deadline) {
        GRPC_COMBINER_UNREF(traffic->combiner, "test transport traffic");
        traffic->finished.store(true);
        return;
    }
    GRPC_CLOSURE_SCHED(&traffic->next, GRPC_ERROR_NONE);
}

void PauseTransportWrite(grpc_chttp2_transport *transport, grpc_chttp2_initiate_write_reason reason) {
    auto *traffic = transport_traffic.load();
    const bool first = !traffic->selected.exchange(true);
    if (first) {
        traffic->server = !transport->is_client;
        traffic->entered.store(true);
    }
    traffic->resume.wait();
    if (first) {
        traffic->combiner = GRPC_COMBINER_REF(transport->combiner, "test transport traffic");
        GRPC_CLOSURE_INIT(
            &traffic->next, ContinueTransportTraffic, traffic, grpc_combiner_scheduler(transport->combiner));
        GRPC_CLOSURE_SCHED(&traffic->next, GRPC_ERROR_NONE);
    }
    // The test restores this function before releasing the intercepted write.
    grpc_chttp2_initiate_write(transport, reason);
}

bool PausePublishedReceive(gpr_locked_mpscq *queue, gpr_mpscq_node *node) {
    const bool first = gpr_mpscq_push(&queue->queue, node);
    if (block_next_request_push.exchange(false)) {
        request_push_entered.store(true);
        release_request_push.wait();
    }
    return first;
}

bool ObserveReceiveOrder(google::protobuf::io::CodedInputStream *input, std::string *value) {
    uint32_t size = 0;
    if (!input->ReadVarint32(&size) || !input->ReadString(value, size)) {
        return false;
    }
    const int order = *value == "pending-first" ? 1 : *value == "new-arrival" ? 2 : 0;
    if (order) {
        int empty = 0;
        first_request_decoded.compare_exchange_strong(empty, order);
    }
    return true;
}

bool BlockingReadBytes(google::protobuf::io::CodedInputStream *input, std::string *value) {
    if (block_next_decode.exchange(false)) {
        decode_entered.store(true);
        release_decode.wait();
    }
    uint32_t size = 0;
    return input->ReadVarint32(&size) && input->ReadString(value, size);
}

void CountClusterCall(MetaServiceImpl *,
                      RequestContext *,
                      const proto::meta::GetClusterInfoRequest *,
                      proto::meta::GetClusterInfoResponse *) {
    ++executed_cluster_calls;
}

struct CollectorLookupGate {
    std::promise<void> entered;
    std::promise<void> release;
};

constexpr char kColdInstance[] = "cold-dispatch-test";
std::atomic<CollectorLookupGate *> collector_lookup_gate{nullptr};
std::atomic<bool> done_tag_observed{false};

std::string PauseCollectorLookup(const RegistryManager *, const std::string &instance_id) {
    if (instance_id == kColdInstance) {
        auto *gate = collector_lookup_gate.load();
        gate->entered.set_value();
        EXPECT_EQ(std::future_status::ready, gate->release.get_future().wait_for(10s));
    }
    return {};
}

bool ObserveCancelledDoneTag(const grpc::ServerContext *) {
    done_tag_observed.store(true);
    return true;
}

std::atomic<int> executed_register_calls{0};
void CountRegisterCall(MetaServiceImpl *,
                       RequestContext *,
                       const proto::meta::RegisterInstanceRequest *,
                       proto::meta::RegisterInstanceResponse *) {
    ++executed_register_calls;
}

void BlockReceivePost(grpc::ServerInterface::RegisteredAsyncRequest *request,
                      void *registered_method,
                      grpc_byte_buffer **payload,
                      grpc::ServerCompletionQueue *notification_cq) {
    receive_post_entered.store(true);
    release_receive_post.wait();
    // The test restores IssueRequest before releasing this registration.
    request->IssueRequest(registered_method, payload, notification_cq);
}

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

class MetaServiceGrpcDispatchTest : public TESTBASE, public ::testing::WithParamInterface<std::size_t> {
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
    void Start(std::size_t capacity = 0,
               bool mixed_admin = true,
               std::size_t worker_count = 1,
               std::shared_ptr<LeaderElector> elector = nullptr,
               bool start_accepting = true) {
        executor_ = std::make_shared<ServiceRequestExecutor>();
        ASSERT_TRUE(executor_->Init(worker_count, capacity));
        metrics_ = std::make_shared<MetricsRegistry>();
        metrics_lifecycle_ = std::make_shared<MetricsLifecycle>();
        registry_ = std::make_shared<RegistryManager>("", metrics_);
        cache_manager_ = std::make_shared<CacheManager>(metrics_, registry_, metrics_lifecycle_);
        impl_ = std::make_shared<MetaServiceImpl>(cache_manager_, nullptr, std::move(elector));
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
        ASSERT_TRUE(service_->PrepareCompletionQueue(builder, GetParam()));
        server_ = builder.BuildAndStart();
        ASSERT_NE(nullptr, server_);
        ASSERT_GT(port_, 0);
        if (start_accepting) {
            service_->StartAccepting();
        }
        channel_ = grpc::CreateChannel("127.0.0.1:" + std::to_string(port_), grpc::InsecureChannelCredentials());
        ASSERT_TRUE(channel_->WaitForConnected(std::chrono::system_clock::now() + 5s));
    }

    void BlockWorker() {
        release_worker_ = std::make_shared<std::promise<void>>();
        auto release = release_worker_->get_future().share();
        auto entered = std::make_shared<std::promise<void>>();
        auto entered_future = entered->get_future();
        ServiceRequestExecutor::Request task;
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

    RpcResult GetClusterInfo(grpc::ClientContext *context = nullptr, const std::string &instance_id = "") {
        grpc::ClientContext default_context;
        default_context.set_deadline(std::chrono::system_clock::now() + 5s);
        auto stub = proto::meta::MetaService::NewStub(channel_);
        proto::meta::GetClusterInfoRequest request;
        request.set_trace_id("dispatch-test");
        request.set_instance_id(instance_id);
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

TEST_P(MetaServiceGrpcDispatchTest, QueuedMetaDoesNotBlockSharedOrIndependentAdmin) {
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

TEST_P(MetaServiceGrpcDispatchTest, DefaultQueueDoesNotRejectWhenWorkerIsBusy) {
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

TEST_P(MetaServiceGrpcDispatchTest, CollectorPreparationPreservesArrivalContextAcrossQueueWait) {
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

TEST_P(MetaServiceGrpcDispatchTest, RemoveInstanceMigrationWaitDoesNotBlockCqOrSharedAdmin) {
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

TEST_P(MetaServiceGrpcDispatchTest, ColdCollectorLookupsCanOccupyAllWorkersButSharedAdminStillProgresses) {
    Start(0, true, 2);
    std::unique_lock<std::shared_mutex> hold_lifecycle(metrics_lifecycle_->mut_);
    auto first = std::async(std::launch::async, [this] { return GetMissingInstance(10s); });
    auto second = std::async(std::launch::async, [this] { return GetMissingInstance(10s); });
    EXPECT_TRUE(WaitUntil([this] { return executor_->GetSnapshot().running == 2; }));
    auto warm = BeginGetClusterInfo();
    EXPECT_TRUE(WaitUntil([this] { return executor_->GetSnapshot().queued == 1; }));
    // A warm collector does not bypass the shared worker budget. The CQ and
    // synchronous Admin still progress even though all workers wait on the lock.
    CheckAdmin(channel_);
    EXPECT_EQ(std::future_status::timeout, warm.wait_for(20ms));
    hold_lifecycle.unlock();
    EXPECT_EQ(proto::meta::INSTANCE_NOT_EXIST, first.get().second.header().status().code());
    EXPECT_EQ(proto::meta::INSTANCE_NOT_EXIST, second.get().second.header().status().code());
    EXPECT_TRUE(warm.get().status.ok());
}

TEST_P(MetaServiceGrpcDispatchTest, StopKeepsLeaderErrorsAndDiscoveryDuringLeaderCleanup) {
    auto backend = std::make_shared<CoordinationMemoryBackend>();
    ASSERT_EQ(EC_OK, backend->Init(StandardUri("memory://")));
    auto elector = std::make_shared<LeaderElector>(backend, "stop-test", "leader-node", 10000, 20);
    Start(0, true, 1, elector);
    Server coordinator;
    coordinator.request_executor_ = executor_;
    coordinator.meta_service_ = service_;
    coordinator.rpc_server_ = std::move(server_);
    coordinator.leader_elector_ = elector;

    std::promise<void> promoted, cleanup_entered, release_cleanup;
    auto cleanup = release_cleanup.get_future().share();
    elector->SetBecomeLeaderHandler([&] {
        impl_->EnableLeaderOnlyRequests();
        promoted.set_value();
    });
    elector->SetLeadershipLostHandler([&] { impl_->DisableLeaderOnlyRequests(); });
    elector->SetNoLongerLeaderHandler([&] {
        impl_->WaitForAllLeaderOnlyRequestsToComplete();
        cleanup_entered.set_value();
        cleanup.wait();
    });
    ASSERT_TRUE(elector->Start());
    EXPECT_EQ(std::future_status::ready, promoted.get_future().wait_for(5s));
    // Use an API with a global collector so this test isolates the Leader gate,
    // rather than an unrelated missing-Instance early return.
    auto leader_request = [this] {
        auto stub = proto::meta::MetaService::NewStub(channel_);
        grpc::ClientContext context;
        context.set_deadline(std::chrono::system_clock::now() + 5s);
        proto::meta::GetInstanceInfoRequest request;
        proto::meta::GetInstanceInfoResponse response;
        auto status = stub->GetInstanceInfo(&context, request, &response);
        EXPECT_TRUE(status.ok());
        return response.header().status().code();
    };
    BlockWorker();
    auto queued = std::async(std::launch::async, leader_request);
    EXPECT_TRUE(WaitUntil([this] { return executor_->GetSnapshot().queued == 1; }));
    auto stopped = std::async(std::launch::async, [&] { coordinator.Stop(); });
    EXPECT_EQ(std::future_status::ready, cleanup_entered.get_future().wait_for(5s));
    ReleaseWorker();
    EXPECT_EQ(proto::meta::SERVER_NOT_LEADER, queued.get());
    EXPECT_EQ(proto::meta::SERVER_NOT_LEADER, leader_request());
    auto discovery = GetClusterInfo();
    EXPECT_TRUE(discovery.status.ok());
    EXPECT_EQ(proto::meta::OK, discovery.response.header().status().code());
    // The stopping elector may no longer know a Leader; discovery remains
    // callable, but it is not a guarantee of a new Leader endpoint.
    EXPECT_EQ("leader-node", discovery.response.self_node_id());
    EXPECT_EQ(0, executor_->GetSnapshot().rejected);
    EXPECT_EQ(0, executor_->GetSnapshot().discarded);
    release_cleanup.set_value();
    EXPECT_EQ(std::future_status::ready, stopped.wait_for(5s));
    stopped.get();
}

TEST_P(MetaServiceGrpcDispatchTest, BusinessExceptionPreservesSynchronousGrpcStatus) {
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

TEST_P(MetaServiceGrpcDispatchTest, IngressExceptionCompletesRpcWithoutSubmittingBusinessWork) {
    Start();
    {
        Stub stub;
        stub.set(ExtractIpFromPeer, ThrowingExtractIp);
        const auto result = GetClusterInfo();
        EXPECT_EQ(grpc::StatusCode::UNKNOWN, result.status.error_code());
        EXPECT_EQ("Unexpected error in RPC handling", result.status.error_message());
        EXPECT_EQ(0u, executor_->GetSnapshot().queue_wait.count);
        EXPECT_EQ(0u, executor_->GetSnapshot().rejected);
    }
    // The receive slot is still armed after an ingress failure.
    EXPECT_TRUE(GetClusterInfo().status.ok());
    CheckAdmin(channel_);
}

TEST_P(MetaServiceGrpcDispatchTest, PendingCallIsMatchedBeforeNewArrivals) {
    Start(0, false, 1, nullptr, false);
    auto invoke = [this](const std::string &trace) {
        auto stub = proto::meta::MetaService::NewStub(channel_);
        grpc::ClientContext context;
        context.set_deadline(std::chrono::system_clock::now() + 5s);
        proto::meta::RegisterInstanceRequest request;
        request.set_trace_id(trace);
        proto::meta::RegisterInstanceResponse response;
        return stub->RegisterInstance(&context, request, &response);
    };
    auto pending = std::async(std::launch::async, invoke, "pending-first");
    // Let the connected RPC reach C-core before posting the first receive.
    std::this_thread::sleep_for(100ms);
    std::promise<void> release;
    release_request_push = release.get_future().share();
    request_push_entered.store(false);
    block_next_request_push.store(true);
    first_request_decoded.store(0);
    Stub stub;
    stub.set(gpr_locked_mpscq_push, PausePublishedReceive);
    using ReadBytes = bool (*)(google::protobuf::io::CodedInputStream *, std::string *);
    stub.set(static_cast<ReadBytes>(&google::protobuf::internal::WireFormatLite::ReadBytes), ObserveReceiveOrder);
    auto accepting = std::async(std::launch::async, [this] { service_->StartAccepting(); });
    const bool ready = WaitUntil([] { return request_push_entered.load() || first_request_decoded.load(); });
    auto newer = std::async(std::launch::async, invoke, "new-arrival");
    const bool decoded = WaitUntil([] { return first_request_decoded.load() != 0; });
    release.set_value();
    accepting.get();
    EXPECT_TRUE(pending.get().ok());
    EXPECT_TRUE(newer.get().ok());
    EXPECT_TRUE(ready);
    EXPECT_TRUE(decoded);
    EXPECT_EQ(1, first_request_decoded.load()) << "New arrivals bypassed the pending RPC";
}

TEST_P(MetaServiceGrpcDispatchTest, CancelledPendingCallsDoNotConsumeReceiveSlots) {
    Start(0, false, 1, nullptr, false);
    for (int i = 0; i < 16; ++i) {
        grpc::ClientContext context;
        context.set_deadline(std::chrono::system_clock::now() + 25ms);
        EXPECT_EQ(grpc::StatusCode::DEADLINE_EXCEEDED, GetClusterInfo(&context).status.error_code());
    }
    service_->StartAccepting();
    EXPECT_TRUE(GetClusterInfo().status.ok());
}


TEST_P(MetaServiceGrpcDispatchTest, SlowDecodeDoesNotSerializeAllReceivesForOneMethod) {
    if (GetParam() == 1) {
        GTEST_SKIP() << "A single CQ poller cannot decode two requests concurrently";
    }
    Start();
    std::promise<void> release;
    release_decode = release.get_future().share();
    decode_entered.store(false);
    block_next_decode.store(true);
    Stub stub;
    using ReadBytes = bool (*)(google::protobuf::io::CodedInputStream *, std::string *);
    stub.set(static_cast<ReadBytes>(&google::protobuf::internal::WireFormatLite::ReadBytes), BlockingReadBytes);
    auto slow = BeginGetClusterInfo();
    const bool entered = WaitUntil([] { return decode_entered.load(); });
    grpc::ClientContext context;
    context.set_deadline(std::chrono::system_clock::now() + 2s);
    const auto other = GetClusterInfo(&context);
    // Release before assertions or future destruction, including failure paths.
    release.set_value();
    EXPECT_TRUE(entered);
    EXPECT_TRUE(other.status.ok()) << other.status.error_message();
    EXPECT_TRUE(slow.get().status.ok());
    EXPECT_TRUE(GetClusterInfo().status.ok());
}

TEST_P(MetaServiceGrpcDispatchTest, SlowReceiveRegistrationDoesNotSerializeOtherPollers) {
    if (GetParam() == 1) {
        GTEST_SKIP() << "A single poller cannot register two receives concurrently";
    }
    Start();
    std::promise<void> release;
    release_receive_post = release.get_future().share();
    receive_post_entered.store(false);
    Stub stub;
    using RegisteredRequest = grpc::ServerInterface::RegisteredAsyncRequest;
    stub.set(ADDR(RegisteredRequest, IssueRequest), BlockReceivePost);
    auto first = std::async(std::launch::async, [this] { return GetClusterInfo(); });
    const bool entered = WaitUntil([] { return receive_post_entered.load(); });
    grpc::ClientContext context;
    context.set_deadline(std::chrono::system_clock::now() + 200ms);
    const auto second = GetClusterInfo(&context);
    stub.reset(ADDR(RegisteredRequest, IssueRequest));
    release.set_value();
    EXPECT_TRUE(entered);
    EXPECT_TRUE(second.status.ok()) << second.status.error_message();
    EXPECT_TRUE(first.get().status.ok());
    EXPECT_TRUE(GetClusterInfo().status.ok());
}

TEST_P(MetaServiceGrpcDispatchTest, ShutdownDrainsReceiveRegistrationStartedBeforeAdmissionClosed) {
    Start();
    std::promise<void> release;
    release_receive_post = release.get_future().share();
    receive_post_entered.store(false);
    Stub stub;
    using RegisteredRequest = grpc::ServerInterface::RegisteredAsyncRequest;
    stub.set(ADDR(RegisteredRequest, IssueRequest), BlockReceivePost);
    auto first = std::async(std::launch::async, [this] { return GetClusterInfo(); });
    const bool entered = WaitUntil([] { return receive_post_entered.load(); });
    std::atomic<bool> admission_closed{false};
    auto stopped = std::async(std::launch::async, [this, &admission_closed] {
        service_->StopAccepting();
        admission_closed.store(true);
        server_->Shutdown();
        service_->WaitForCalls();
    });
    const bool closed = WaitUntil([&] { return admission_closed.load(); }, 200ms);
    const auto waiting = stopped.wait_for(10ms);
    stub.reset(ADDR(RegisteredRequest, IssueRequest));
    release.set_value();
    EXPECT_TRUE(entered);
    EXPECT_TRUE(closed);
    EXPECT_EQ(std::future_status::timeout, waiting);
    EXPECT_TRUE(first.get().status.ok());
    EXPECT_EQ(std::future_status::ready, stopped.wait_for(5s));
    stopped.get();
}


TEST_P(MetaServiceGrpcDispatchTest, FullQueueRejectsBeforeReceiveRepostAndContextPreparation) {
    Start(1);
    BlockWorker();
    auto pending = BeginGetClusterInfo();
    ASSERT_TRUE(WaitUntil([this] { return executor_->GetSnapshot().queued == 1; }));
    std::promise<void> release;
    release_receive_post = release.get_future().share();
    receive_post_entered.store(false);
    Stub stub;
    using RegisteredRequest = grpc::ServerInterface::RegisteredAsyncRequest;
    stub.set(ADDR(RegisteredRequest, IssueRequest), BlockReceivePost);
    stub.set(ExtractIpFromPeer, ThrowingExtractIp);
    auto rejected = BeginGetClusterInfo();
    const bool entered = WaitUntil([] { return receive_post_entered.load(); });
    const auto ready = rejected.wait_for(200ms);
    stub.reset(ADDR(RegisteredRequest, IssueRequest));
    release.set_value();
    const auto result = rejected.get();
    stub.reset(ExtractIpFromPeer);
    EXPECT_TRUE(entered);
    EXPECT_EQ(std::future_status::ready, ready);
    EXPECT_TRUE(result.status.ok());
    EXPECT_EQ(proto::meta::SERVICE_BUSY, result.response.header().status().code());
    EXPECT_FALSE(result.response.header().request_id().empty());
    EXPECT_EQ(1u, executor_->GetSnapshot().rejected);
    ReleaseWorker();
    EXPECT_TRUE(pending.get().status.ok());
    EXPECT_TRUE(GetClusterInfo().status.ok());
}

TEST_P(MetaServiceGrpcDispatchTest, IncomingTimeoutUsesFreshTimeAfterTransportWork) {
    Start();
    BlockWorker();
    grpc::ClientContext context;
    // Use a distinct timeout so the interned header has no cached decode.
    context.set_deadline(std::chrono::system_clock::now() + 23s + std::chrono::milliseconds(GetParam() * 111));
    std::promise<void> resume_decode;
    release_timeout_decode = resume_decode.get_future().share();
    timeout_decode_entered.store(false);
    Stub decoder;
    decoder.set(grpc_http2_decode_timeout, DecodeTimeoutWithStaleClock);
    auto pending = std::async(std::launch::async, [this, &context] { return GetClusterInfo(&context); });
    const bool entered = WaitUntil([] { return timeout_decode_entered.load(); });
    decoder.reset(grpc_http2_decode_timeout);
    resume_decode.set_value();
    WaitUntil(
        [&] { return executor_->GetSnapshot().queued == 1 || pending.wait_for(0ms) == std::future_status::ready; });
    const bool queued = executor_->GetSnapshot().queued == 1;
    ReleaseWorker();
    const auto result = pending.get();
    EXPECT_TRUE(entered);
    EXPECT_TRUE(queued);
    EXPECT_TRUE(result.status.ok()) << result.status.error_message();
}

TEST_P(MetaServiceGrpcDispatchTest, FullQueueReplyProgressesDuringTransportTraffic) {
    Start(1);
    BlockWorker();
    auto pending = BeginGetClusterInfo();
    ASSERT_TRUE(WaitUntil([this] { return executor_->GetSnapshot().queued == 1; }));

    // Hold the next request after its bytes reach the server, then intercept
    // only its response write. The client request has already left its combiner.
    std::promise<void> resume_decode;
    release_decode = resume_decode.get_future().share();
    decode_entered.store(false);
    block_next_decode.store(true);
    using ReadBytes = bool (*)(google::protobuf::io::CodedInputStream *, std::string *);
    const auto read_bytes = static_cast<ReadBytes>(&google::protobuf::internal::WireFormatLite::ReadBytes);
    Stub decode_stub;
    decode_stub.set(read_bytes, BlockingReadBytes);
    auto rejected = BeginGetClusterInfo();
    const bool decoded = WaitUntil([] { return decode_entered.load(); });

    TransportTraffic traffic;
    std::promise<void> resume_write;
    traffic.resume = resume_write.get_future().share();
    traffic.deadline = std::chrono::steady_clock::now() + 10s;
    transport_traffic = &traffic;
    Stub write_stub;
    write_stub.set(grpc_chttp2_initiate_write, PauseTransportWrite);
    decode_stub.reset(read_bytes);
    resume_decode.set_value();
    const bool entered = WaitUntil([&] { return traffic.entered.load(); });
    write_stub.reset(grpc_chttp2_initiate_write);
    resume_write.set_value();

    // Continually arriving transport work must not postpone an already queued
    // BUSY response until the incoming work stops. Cleanup precedes assertions.
    const auto ready = rejected.wait_for(1s);
    traffic.stop.store(true);
    const bool finished = !entered || WaitUntil([&] { return traffic.finished.load(); });
    const auto result = rejected.get();
    transport_traffic = nullptr;
    ReleaseWorker();
    EXPECT_TRUE(pending.get().status.ok());
    EXPECT_TRUE(decoded);
    EXPECT_TRUE(entered);
    EXPECT_TRUE(traffic.server);
    EXPECT_TRUE(finished);
    EXPECT_EQ(std::future_status::ready, ready);
    EXPECT_TRUE(result.status.ok()) << result.status.error_message();
    EXPECT_EQ(proto::meta::SERVICE_BUSY, result.response.header().status().code());
    EXPECT_TRUE(WaitUntil([this] { return executor_->GetSnapshot().running == 0; }));
}

TEST_P(MetaServiceGrpcDispatchTest, ConfiguredQueueLimitReturnsBusinessError) {
    Start(1);
    BlockWorker();
    auto pending = BeginGetClusterInfo();
    ASSERT_TRUE(WaitUntil([this] { return executor_->GetSnapshot().queued == 1; }));
    auto rejected = GetClusterInfo();
    EXPECT_TRUE(rejected.status.ok());
    EXPECT_EQ(proto::meta::SERVICE_BUSY, rejected.response.header().status().code());
    EXPECT_EQ("service request queue is full", rejected.response.header().status().message());
    EXPECT_FALSE(rejected.response.header().request_id().empty());
    EXPECT_EQ(1u, GrpcResponseCount("OK"));
    EXPECT_EQ(0u, GrpcResponseCount("RESOURCE_EXHAUSTED"));
    {
        std::unique_lock<std::shared_mutex> hold_lifecycle(metrics_lifecycle_->mut_);
        const auto missing = GetMissingInstance();
        EXPECT_TRUE(missing.first.ok());
        EXPECT_EQ(proto::meta::SERVICE_BUSY, missing.second.header().status().code());
        EXPECT_EQ("service request queue is full", missing.second.header().status().message());
        EXPECT_FALSE(missing.second.header().request_id().empty());
        EXPECT_EQ(1u, GrpcResponseCount("OK", "MetaService/GetCacheMeta"));
    }
    ReleaseWorker();
    EXPECT_TRUE(pending.get().status.ok());
    EXPECT_EQ(2u, GrpcResponseCount("OK"));
}

TEST_P(MetaServiceGrpcDispatchTest, DiscardPendingCompletesCallsWithoutWaitingForBusyWorker) {
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
    EXPECT_EQ(proto::meta::SERVICE_NOT_READY, closed.response.header().status().code());
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

TEST_P(MetaServiceGrpcDispatchTest, DrainKeepsQueuedCallsAliveThroughServerShutdown) {
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

TEST_P(MetaServiceGrpcDispatchTest, DeadlineAndCancellationStillDrainFinishEvents) {
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
    executed_register_calls = 0;
    Stub business;
    business.set(ADDR(MetaServiceImpl, RegisterInstance), CountRegisterCall);
    auto invoke = [this](grpc::ClientContext &context) {
        auto stub = proto::meta::MetaService::NewStub(channel_);
        proto::meta::RegisterInstanceRequest request;
        proto::meta::RegisterInstanceResponse response;
        return stub->RegisterInstance(&context, request, &response);
    };
    auto cancelled = std::async(std::launch::async, [&] { return invoke(cancel_context); });
    ASSERT_TRUE(WaitUntil([this] { return executor_->GetSnapshot().queued == 2; }));
    cancel_context.TryCancel();
    EXPECT_EQ(grpc::StatusCode::CANCELLED, cancelled.get().error_code());
    // Let the CQ observe the cancellation; the original deadline is still far away.
    std::this_thread::sleep_for(50ms);
    ReleaseWorker();
    executor_->Drain();
    EXPECT_EQ(0, executor_->GetSnapshot().discarded);
    EXPECT_EQ(0, executed_register_calls.load());
    // A cancelled Finish must not terminate the CQ poller.
    grpc::ClientContext healthy;
    healthy.set_deadline(std::chrono::system_clock::now() + 5s);
    EXPECT_TRUE(invoke(healthy).ok());
    EXPECT_EQ(1, executed_register_calls.load());
}

TEST_P(MetaServiceGrpcDispatchTest, ExpiredQueuedCallDoesNotExecuteBusiness) {
    Start();
    BlockWorker();
    executed_cluster_calls = 0;
    Stub stub;
    stub.set(ADDR(MetaServiceImpl, GetClusterInfo), CountClusterCall);
    grpc::ClientContext context;
    context.set_deadline(std::chrono::system_clock::now() + 200ms);
    auto expired = std::async(std::launch::async, [this, &context] { return GetClusterInfo(&context); });
    ASSERT_TRUE(WaitUntil([this] { return executor_->GetSnapshot().queued == 1; }));
    EXPECT_EQ(grpc::StatusCode::DEADLINE_EXCEEDED, expired.get().status.error_code());
    // The wire timeout is rounded; the server deadline can trail the client's.
    std::this_thread::sleep_for(20ms);
    ReleaseWorker();
    executor_->Drain();
    EXPECT_EQ(0, executed_cluster_calls.load());
    EXPECT_TRUE(GetClusterInfo().status.ok());
    executor_->Drain();
    EXPECT_EQ(1, executed_cluster_calls.load());
}

TEST_P(MetaServiceGrpcDispatchTest, DeadlineDuringCollectorLookupDoesNotExecuteBusiness) {
    Start();
    CollectorLookupGate gate;
    auto entered = gate.entered.get_future();
    collector_lookup_gate.store(&gate);
    executed_cluster_calls = 0;
    Stub stub;
    stub.set(ADDR(RegistryManager, GetInstanceGroupName), PauseCollectorLookup);
    stub.set(ADDR(MetaServiceImpl, GetClusterInfo), CountClusterCall);
    auto expired = std::async(std::launch::async, [this] {
        grpc::ClientContext context;
        context.set_deadline(std::chrono::system_clock::now() + 500ms);
        return GetClusterInfo(&context, kColdInstance);
    });
    EXPECT_EQ(std::future_status::ready, entered.wait_for(5s));
    EXPECT_EQ(grpc::StatusCode::DEADLINE_EXCEEDED, expired.get().status.error_code());
    // Account for rounding of the relative deadline on the wire.
    std::this_thread::sleep_for(30ms);
    gate.release.set_value();
    executor_->Drain();
    collector_lookup_gate.store(nullptr);
    EXPECT_EQ(0, executed_cluster_calls.load());
    EXPECT_TRUE(WaitUntil([this] { return GrpcResponseCount("DEADLINE_EXCEEDED") == 1; }));
    EXPECT_TRUE(GetClusterInfo().status.ok());
    EXPECT_EQ(1, executed_cluster_calls.load());
}

TEST_P(MetaServiceGrpcDispatchTest, CancellationDuringCollectorLookupDoesNotExecuteBusiness) {
    if (GetParam() != 1) {
        GTEST_SKIP() << "The CQ barrier requires a single poller.";
    }
    Start(1);
    CollectorLookupGate gate;
    auto entered = gate.entered.get_future();
    collector_lookup_gate.store(&gate);
    done_tag_observed.store(false);
    executed_cluster_calls = 0;
    Stub stub;
    stub.set(ADDR(RegistryManager, GetInstanceGroupName), PauseCollectorLookup);
    stub.set(ADDR(MetaServiceImpl, GetClusterInfo), CountClusterCall);
    stub.set(ADDR(grpc::ServerContext, IsCancelled), ObserveCancelledDoneTag);

    grpc::ClientContext context;
    context.set_deadline(std::chrono::system_clock::now() + 15s);
    auto cancelled =
        std::async(std::launch::async, [this, &context] { return GetClusterInfo(&context, kColdInstance); });
    const auto preparing = entered.wait_for(5s);
    context.TryCancel();
    EXPECT_EQ(std::future_status::ready, preparing);
    EXPECT_EQ(grpc::StatusCode::CANCELLED, cancelled.get().status.error_code());
    const bool observed = WaitUntil([] { return done_tag_observed.load(); });
    EXPECT_TRUE(observed);
    if (preparing == std::future_status::ready && observed) {
        // The full-queue reply runs on the same poller after the DoneTag has
        // stored the cancellation flag. The worker is still inside Prepare.
        EXPECT_EQ(ServiceRequestExecutor::SubmitResult::kAccepted,
                  executor_->TrySubmit({[](ServiceRequestExecutor::CompletionToken) {},
                                        [](ServiceRequestExecutor::CompletionToken) {}}));
        const auto barrier = GetClusterInfo();
        EXPECT_TRUE(barrier.status.ok());
        EXPECT_EQ(proto::meta::SERVICE_BUSY, barrier.response.header().status().code());
    }
    gate.release.set_value();
    executor_->Drain();
    collector_lookup_gate.store(nullptr);
    EXPECT_EQ(0, executed_cluster_calls.load());
    EXPECT_TRUE(GetClusterInfo().status.ok());
    EXPECT_EQ(1, executed_cluster_calls.load());
}

TEST_P(MetaServiceGrpcDispatchTest, ExpiredBeforeAdmissionDoesNotConsumeQueue) {
    Start();
    std::promise<void> release;
    release_decode = release.get_future().share();
    decode_entered.store(false);
    block_next_decode.store(true);
    Stub stub;
    using ReadBytes = bool (*)(google::protobuf::io::CodedInputStream *, std::string *);
    stub.set(static_cast<ReadBytes>(&google::protobuf::internal::WireFormatLite::ReadBytes), BlockingReadBytes);
    grpc::ClientContext context;
    context.set_deadline(std::chrono::system_clock::now() + 200ms);
    auto expired = std::async(std::launch::async, [this, &context] { return GetClusterInfo(&context); });
    const bool entered = WaitUntil([] { return decode_entered.load(); });
    const auto result = expired.get();
    std::this_thread::sleep_for(20ms);
    release.set_value();
    EXPECT_TRUE(entered);
    EXPECT_EQ(grpc::StatusCode::DEADLINE_EXCEEDED, result.status.error_code());
    ASSERT_TRUE(WaitUntil([this] { return GrpcResponseCount("DEADLINE_EXCEEDED") == 1; }));
    EXPECT_EQ(0u, executor_->GetSnapshot().queue_wait.count);
    EXPECT_EQ(0u, executor_->GetSnapshot().rejected);
    EXPECT_TRUE(GetClusterInfo().status.ok());
}

TEST_P(MetaServiceGrpcDispatchTest, PureAsyncServerHandlesUnknownMethodAndMalformedProtobuf) {
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

TEST_P(MetaServiceGrpcDispatchTest, FailedBindCanStopPreparedPollerWithoutPostingRequests) {
    Start();
    MetaServiceGRpc second(metrics_, impl_, registry_, executor_);
    second.Init();
    grpc::ServerBuilder builder;
    ConfigureGrpcServer(builder, 2, 0, "failed-bind", metrics_, nullptr);
    builder.AddChannelArgument("grpc.so_reuseport", 0);
    builder.AddListeningPort("127.0.0.1:" + std::to_string(port_), grpc::InsecureServerCredentials());
    builder.RegisterService(&second);
    ASSERT_TRUE(second.PrepareCompletionQueue(builder, GetParam()));
    auto failed = builder.BuildAndStart();
    EXPECT_EQ(nullptr, failed);
    second.StopAccepting();
    second.WaitForCalls();
    second.ShutdownCompletionQueueAndJoin();
    EXPECT_TRUE(GetClusterInfo().status.ok());
}

INSTANTIATE_TEST_SUITE_P(PollerCounts, MetaServiceGrpcDispatchTest, ::testing::Values(1u, 2u, 3u, 4u, 5u));

} // namespace
} // namespace kv_cache_manager
