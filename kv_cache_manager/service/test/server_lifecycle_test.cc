#include <chrono>
#include <functional>
#include <future>
#include <grpcpp/grpcpp.h>
#include <mutex>
#include <string>
#include <thread>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/manager/cache_manager.h"
#include "kv_cache_manager/protocol/protobuf/kv_meta_service.grpc.pb.h"
#include "kv_cache_manager/protocol/protobuf/meta_service.grpc.pb.h"
#include "kv_cache_manager/service/kv_meta_service_impl.h"
#include "kv_cache_manager/service/server.h"

using namespace kv_cache_manager;
using namespace std::chrono_literals;

namespace {

bool WaitUntil(const std::function<bool()> &predicate, std::chrono::steady_clock::duration timeout) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
        if (predicate()) {
            return true;
        }
        std::this_thread::sleep_for(1ms);
    }
    return predicate();
}

} // namespace

class ServerLifecycleTest : public TESTBASE {
protected:
    void TearDown() override {
        if (server_initialized_) {
            server_.Stop();
        }
    }

    bool StartRpcServer(bool enable_kv_meta) {
        ServerConfig config;
        if (!config.Parse("",
                          {{"kvcm.service.rpc_port", "0"},
                           {"kvcm.service.admin_rpc_port", "0"},
                           {"kvcm.kv_meta.enabled", enable_kv_meta ? "true" : "false"}}) ||
            !server_.Init(config)) {
            return false;
        }
        server_initialized_ = true;
        return server_.StartRpcServer() && server_.bound_rpc_port_ > 0;
    }

    std::shared_ptr<grpc::Channel> PrimaryChannel() const {
        return grpc::CreateChannel("127.0.0.1:" + std::to_string(server_.bound_rpc_port_),
                                   grpc::InsecureChannelCredentials());
    }

    Server server_;
    bool server_initialized_ = false;
};

TEST_F(ServerLifecycleTest, KvMetaAndFixedBlockMetaSharePrimaryRpcListener) {
    ASSERT_TRUE(StartRpcServer(true));

    EXPECT_NE(nullptr, server_.rpc_server_);
    EXPECT_NE(nullptr, server_.meta_service_);
    EXPECT_NE(nullptr, server_.kv_meta_service_);
    EXPECT_EQ(nullptr, server_.admin_rpc_server_);

    const auto channel = PrimaryChannel();
    auto meta_stub = proto::meta::MetaService::NewStub(channel);
    auto kv_meta_stub = proto::kv_meta::MetaService::NewStub(channel);

    grpc::ClientContext meta_context;
    meta_context.set_deadline(std::chrono::system_clock::now() + 2s);
    proto::meta::GetInstanceInfoRequest meta_request;
    proto::meta::GetInstanceInfoResponse meta_response;
    meta_request.set_instance_id("route-probe");
    const grpc::Status meta_status = meta_stub->GetInstanceInfo(&meta_context, meta_request, &meta_response);

    grpc::ClientContext kv_meta_context;
    kv_meta_context.set_deadline(std::chrono::system_clock::now() + 2s);
    proto::kv_meta::GetInstanceInfoRequest kv_meta_request;
    proto::kv_meta::GetInstanceInfoResponse kv_meta_response;
    kv_meta_request.set_instance_id("route-probe");
    const grpc::Status kv_meta_status =
        kv_meta_stub->GetInstanceInfo(&kv_meta_context, kv_meta_request, &kv_meta_response);

    EXPECT_TRUE(meta_status.ok()) << meta_status.error_message();
    EXPECT_EQ(proto::meta::SERVER_NOT_LEADER, meta_response.header().status().code());
    EXPECT_TRUE(kv_meta_status.ok()) << kv_meta_status.error_message();
    EXPECT_EQ(proto::kv_meta::SERVER_NOT_LEADER, kv_meta_response.header().status().code());
}

TEST_F(ServerLifecycleTest, DisabledKvMetaDoesNotAlterPrimaryMetaListener) {
    ASSERT_TRUE(StartRpcServer(false));

    EXPECT_NE(nullptr, server_.rpc_server_);
    EXPECT_NE(nullptr, server_.meta_service_);
    EXPECT_EQ(nullptr, server_.kv_meta_service_);
    EXPECT_EQ(nullptr, server_.kv_meta_manager_);

    const auto channel = PrimaryChannel();
    auto meta_stub = proto::meta::MetaService::NewStub(channel);
    auto kv_meta_stub = proto::kv_meta::MetaService::NewStub(channel);

    grpc::ClientContext meta_context;
    meta_context.set_deadline(std::chrono::system_clock::now() + 2s);
    proto::meta::GetInstanceInfoRequest meta_request;
    proto::meta::GetInstanceInfoResponse meta_response;
    meta_request.set_instance_id("route-probe");
    const grpc::Status meta_status = meta_stub->GetInstanceInfo(&meta_context, meta_request, &meta_response);

    grpc::ClientContext kv_meta_context;
    kv_meta_context.set_deadline(std::chrono::system_clock::now() + 2s);
    proto::kv_meta::GetInstanceInfoRequest kv_meta_request;
    proto::kv_meta::GetInstanceInfoResponse kv_meta_response;
    kv_meta_request.set_instance_id("route-probe");
    const grpc::Status kv_meta_status =
        kv_meta_stub->GetInstanceInfo(&kv_meta_context, kv_meta_request, &kv_meta_response);

    EXPECT_TRUE(meta_status.ok()) << meta_status.error_message();
    EXPECT_EQ(proto::meta::SERVER_NOT_LEADER, meta_response.header().status().code());
    EXPECT_EQ(grpc::StatusCode::UNIMPLEMENTED, kv_meta_status.error_code());
}

TEST_F(ServerLifecycleTest, ConcurrentRecoveryCancellationWaitsForMovedWorker) {
    std::promise<void> worker_started_promise;
    auto worker_started = worker_started_promise.get_future();
    std::promise<void> release_worker_promise;
    auto release_worker = release_worker_promise.get_future().share();

    server_.kv_meta_recovery_thread_ = std::thread([&]() {
        worker_started_promise.set_value();
        release_worker.wait();
    });

    const auto worker_status = worker_started.wait_for(2s);
    if (worker_status != std::future_status::ready) {
        release_worker_promise.set_value();
        server_.kv_meta_recovery_thread_.join();
        FAIL() << "recovery worker did not start";
    }

    std::thread first_canceller([&]() { server_.CancelAndJoinKvMetaRecovery(); });
    const bool worker_moved = WaitUntil(
        [&]() {
            std::lock_guard<std::mutex> lock(server_.kv_meta_recovery_mutex_);
            return !server_.kv_meta_recovery_thread_.joinable();
        },
        2s);
    if (!worker_moved) {
        release_worker_promise.set_value();
        first_canceller.join();
        FAIL() << "first cancellation did not take ownership of the recovery worker";
    }

    // Ownership has moved out of kv_meta_recovery_thread_, but the first
    // canceller must retain the lifecycle lock until its local thread joins.
    const bool lifecycle_lock_was_available = server_.kv_meta_recovery_join_mutex_.try_lock();
    if (lifecycle_lock_was_available) {
        server_.kv_meta_recovery_join_mutex_.unlock();
    }

    const auto epoch_before_second = server_.kv_meta_recovery_epoch_.load(std::memory_order_acquire);
    std::promise<void> second_returned_promise;
    auto second_returned = second_returned_promise.get_future();
    std::thread second_canceller([&]() {
        server_.CancelAndJoinKvMetaRecovery();
        second_returned_promise.set_value();
    });
    const bool second_entered = WaitUntil(
        [&]() { return server_.kv_meta_recovery_epoch_.load(std::memory_order_acquire) > epoch_before_second; }, 2s);
    const auto second_status_before_release = second_returned.wait_for(50ms);

    release_worker_promise.set_value();
    first_canceller.join();
    second_canceller.join();

    EXPECT_FALSE(lifecycle_lock_was_available);
    EXPECT_TRUE(second_entered);
    EXPECT_EQ(std::future_status::timeout, second_status_before_release)
        << "a concurrent cancellation returned while the moved recovery worker was still running";
    EXPECT_EQ(std::future_status::ready, second_returned.wait_for(0s));
}

TEST_F(ServerLifecycleTest, KvMetaRecoveryWaitsForDeferredCacheRecoveryCompletion) {
    ASSERT_TRUE(StartRpcServer(true));
    ASSERT_TRUE(server_.cache_manager_);
    ASSERT_TRUE(server_.kv_meta_impl_);
    server_.cache_manager_->recover_complete_.store(false, std::memory_order_release);

    server_.StartKvMetaRecovery();
    std::this_thread::sleep_for(100ms);
    EXPECT_FALSE(server_.kv_meta_impl_->is_accepting_leader_only_requests_.load(std::memory_order_acquire));

    server_.cache_manager_->recover_complete_.store(true, std::memory_order_release);
    ASSERT_TRUE(WaitUntil(
        [&]() { return server_.kv_meta_impl_->is_accepting_leader_only_requests_.load(std::memory_order_acquire); },
        2s));
    server_.CancelAndJoinKvMetaRecovery();
}

TEST_F(ServerLifecycleTest, WaitJoinsKvMetaRecoveryWorker) {
    ASSERT_TRUE(StartRpcServer(true));
    std::promise<void> worker_started_promise;
    auto worker_started = worker_started_promise.get_future();
    std::promise<void> release_worker_promise;
    auto release_worker = release_worker_promise.get_future().share();
    server_.kv_meta_recovery_thread_ = std::thread([&]() {
        worker_started_promise.set_value();
        release_worker.wait();
    });
    ASSERT_EQ(std::future_status::ready, worker_started.wait_for(2s));

    server_.rpc_server_->Shutdown();
    auto wait_result = std::async(std::launch::async, [&]() { return server_.Wait(); });
    EXPECT_EQ(std::future_status::timeout, wait_result.wait_for(50ms));
    release_worker_promise.set_value();
    ASSERT_EQ(std::future_status::ready, wait_result.wait_for(2s));
    EXPECT_TRUE(wait_result.get());
    EXPECT_FALSE(server_.kv_meta_recovery_thread_.joinable());
}
