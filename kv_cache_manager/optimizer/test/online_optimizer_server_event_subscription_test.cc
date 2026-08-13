#include <atomic>
#include <chrono>
#include <fstream>
#include <grpcpp/grpcpp.h>
#include <memory>
#include <netinet/in.h>
#include <string>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/optimizer/service/kvcm_event_subscriber.h"
#include "kv_cache_manager/optimizer/service/online_optimizer_server.h"
#include "kv_cache_manager/protocol/protobuf/meta_service.grpc.pb.h"
#include "kv_cache_manager/protocol/protobuf/optimizer_service.grpc.pb.h"

namespace kv_cache_manager {

namespace {

template <typename Predicate>
bool WaitUntil(Predicate predicate, int timeout_ms = 3000) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (std::chrono::steady_clock::now() < deadline) {
        if (predicate()) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    return predicate();
}

class BlockingEventService final : public proto::optimizer::OptimizerEventStreamService::Service {
public:
    grpc::Status GetConfiguration(grpc::ServerContext *,
                                  const proto::optimizer::KvcmConfigurationRequest *,
                                  proto::optimizer::KvcmConfigurationResponse *response) override {
        response->mutable_header()->mutable_status()->set_code(proto::optimizer::OK);
        return grpc::Status::OK;
    }

    grpc::Status SubscribeEvents(grpc::ServerContext *context,
                                 const proto::optimizer::OptimizerEventSubscriptionRequest *,
                                 grpc::ServerWriter<proto::optimizer::TraceQueryRequest> *) override {
        subscribe_count_.fetch_add(1);
        while (!context->IsCancelled()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        return grpc::Status::OK;
    }

    std::atomic<int> subscribe_count_{0};
};

class LeaderMetaService final : public proto::meta::MetaService::Service {
public:
    grpc::Status GetClusterInfo(grpc::ServerContext *,
                                const proto::meta::GetClusterInfoRequest *,
                                proto::meta::GetClusterInfoResponse *response) override {
        response->mutable_header()->mutable_status()->set_code(proto::meta::OK);
        auto *endpoint = response->mutable_leader_endpoint();
        endpoint->set_host("127.0.0.1");
        endpoint->set_meta_rpc_port(port_);
        return grpc::Status::OK;
    }

    int port_ = 0;
};

int StartEventServer(LeaderMetaService *meta_service,
                     BlockingEventService *event_service,
                     std::unique_ptr<grpc::Server> *server) {
    int port = 0;
    grpc::ServerBuilder builder;
    builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
    builder.RegisterService(meta_service);
    builder.RegisterService(event_service);
    *server = builder.BuildAndStart();
    meta_service->port_ = port;
    return port;
}

int AllocatePort() {
    const int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        return 0;
    }
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = 0;
    if (bind(fd, reinterpret_cast<sockaddr *>(&address), sizeof(address)) != 0) {
        close(fd);
        return 0;
    }
    socklen_t length = sizeof(address);
    if (getsockname(fd, reinterpret_cast<sockaddr *>(&address), &length) != 0) {
        close(fd);
        return 0;
    }
    const int port = ntohs(address.sin_port);
    close(fd);
    return port;
}

} // namespace

class OnlineOptimizerServerEventSubscriptionTest : public TESTBASE {};

TEST_F(OnlineOptimizerServerEventSubscriptionTest, StartsAndStopsSubscriberWithServer) {
    LeaderMetaService meta_service;
    BlockingEventService event_service;
    std::unique_ptr<grpc::Server> event_server;
    const int event_port = StartEventServer(&meta_service, &event_service, &event_server);
    ASSERT_NE(nullptr, event_server);
    ASSERT_GT(event_port, 0);
    const int optimizer_rpc_port = AllocatePort();
    const int optimizer_http_port = AllocatePort();
    ASSERT_GT(optimizer_rpc_port, 0);
    ASSERT_GT(optimizer_http_port, 0);
    ASSERT_NE(optimizer_rpc_port, optimizer_http_port);

    const std::string config_path = GetPrivateTestRuntimeDataPath() + "optimizer.json";
    std::ofstream output(config_path);
    ASSERT_TRUE(output.is_open());
    output << "{\n"
           << "  \"rpc_port\": " << optimizer_rpc_port << ",\n"
           << "  \"http_port\": " << optimizer_http_port << ",\n"
           << R"(
        "registry_storage_uri": "",
        "metrics_report_interval_ms": 0,
        "enable_prometheus": false,
        "kvcm_event_subscription": {
            "enable": true,
            "service_discovery_url": "static://127.0.0.1:)"
           << event_port << R"(",
            "consumer_id": "server-lifecycle-test",
            "discovery_refresh_interval_ms": 50
        }
    })";
    output.close();

    OnlineOptimizerServer server;
    ASSERT_TRUE(server.Init(config_path));
    ASSERT_TRUE(server.Start());
    ASSERT_TRUE(WaitUntil([&event_service] { return event_service.subscribe_count_.load() == 1; }));
    ASSERT_NE(nullptr, server.kvcm_event_subscriber_);

    server.Stop();
    EXPECT_FALSE(server.kvcm_event_subscriber_->running_);
    event_server->Shutdown();
}

} // namespace kv_cache_manager
