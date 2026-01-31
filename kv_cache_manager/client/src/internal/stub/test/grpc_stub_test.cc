#include <grpcpp/grpcpp.h>
#include <memory>
#include <netinet/in.h>
#include <string>
#include <sys/socket.h>
#include <thread>

#include "kv_cache_manager/client/src/internal/stub/grpc_stub.h"
#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/config/model_deployment.h"
#include "kv_cache_manager/config/registry_manager.h"
#include "kv_cache_manager/event/event_manager.h"
#include "kv_cache_manager/manager/cache_manager.h"
#include "kv_cache_manager/manager/cache_reclaimer.h"
#include "kv_cache_manager/manager/meta_searcher_manager.h"
#include "kv_cache_manager/manager/reclaimer_task_supervisor.h"
#include "kv_cache_manager/manager/startup_config_loader.h"
#include "kv_cache_manager/metrics/dummy_metrics_reporter.h"
#include "kv_cache_manager/metrics/metrics_registry.h"
#include "kv_cache_manager/service/grpc_service/meta_service_grpc.h"
#include "kv_cache_manager/service/meta_service_impl.h"
using namespace kv_cache_manager;

namespace {
static const std::string default_storage_configs(
    "[{\"type\":\"file\",\"is_available\":true,\"global_unique_name\":\"nfs_01\",\"storage_spec\":{"
    "\"root_path\":\"/tmp/nfs/\",\"key_count_per_file\":8}}]");

template <typename Func>
bool WaitUntil(Func condition, int timeout_ms = 5000, int interval_ms = 100) {
    auto start_time = std::chrono::steady_clock::now();
    while (std::chrono::steady_clock::now() - start_time < std::chrono::milliseconds(timeout_ms)) {
        if (condition()) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms));
    }
    return false;
}

} // namespace

class GrpcStubTest : public TESTBASE {
public:
    void SetUp() override {
        StartService();
        InitStub();
    }

    void TearDown() override { rpc_server_->Shutdown(); }
    static void ExpectLocationsEq(const Locations &a, const Locations &b) {
        EXPECT_EQ(a.size(), b.size());
        size_t location_size = std::min(a.size(), b.size());
        for (size_t i = 0; i < location_size; i++) {
            EXPECT_EQ(a[i].size(), b[i].size());
            size_t spec_size = std::min(a[i].size(), b[i].size());
            for (size_t j = 0; j < spec_size; j++) {
                EXPECT_EQ(a[i][j].spec_name, b[i][j].spec_name);
                EXPECT_EQ(a[i][j].uri, b[i][j].uri);
            }
        }
    }
    static void ExpectLocationsValid(const Locations &locations, size_t location_count) {
        EXPECT_EQ(location_count, locations.size());
        for (auto &location : locations) {
            EXPECT_EQ(2, location.size());
            EXPECT_EQ(location[0].spec_name, "tp0");
            EXPECT_GT(location[0].uri.size(), 0);
            EXPECT_EQ(location[1].spec_name, "tp1");
            EXPECT_GT(location[1].uri.size(), 0);
        }
    }

private:
    int GetFreePort();
    void StartService(int port = -1);
    void InitStub();
    ModelDeployment createModelDeployment(int32_t tp_size = 1, int32_t pp_size = 1);
    Stub::LocationSpecInfoMap createLocationSpecInfos(int32_t spec_size = 1);

    int port_;
    std::shared_ptr<MetricsRegistry> metrics_registry_;
    std::shared_ptr<RegistryManager> registry_manager_;
    std::shared_ptr<CacheManager> cache_manager_;
    std::shared_ptr<MetricsReporter> metrics_reporter_;
    std::shared_ptr<MetaServiceImpl> meta_service_impl_;
    std::shared_ptr<MetaServiceGRpc> meta_service_;
    std::unique_ptr<grpc::Server> rpc_server_;
    std::shared_ptr<GrpcStub> stub_;
};

int GrpcStubTest::GetFreePort() {
    int sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd == -1) {
        KVCM_LOG_ERROR("create socket fail");
        return -1;
    }

    struct sockaddr_in addr;
    std::memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = 0;

    if (bind(sockfd, (struct sockaddr *)&addr, sizeof(addr)) == -1) {
        KVCM_LOG_ERROR("bind socket fail");
        close(sockfd);
        return -1;
    }

    socklen_t len = sizeof(addr);
    if (getsockname(sockfd, (struct sockaddr *)&addr, &len) == -1) {
        KVCM_LOG_ERROR("getsockname fail");
        close(sockfd);
        return -1;
    }

    int port = ntohs(addr.sin_port);

    close(sockfd);
    return port;
}

void GrpcStubTest::StartService(int port) {
    std::string registry_storage_uri = ""; // TODO
    metrics_registry_ = std::make_shared<MetricsRegistry>();
    registry_manager_ = std::make_shared<RegistryManager>(registry_storage_uri, metrics_registry_);
    registry_manager_->Init();
    cache_manager_ = std::make_shared<CacheManager>(metrics_registry_, registry_manager_);
    cache_manager_->Init();
    StartupConfigLoader loader;
    loader.Init(registry_manager_);
    loader.Load("");
    metrics_reporter_ = std::make_shared<DummyMetricsReporter>();
    metrics_reporter_->Init(cache_manager_, metrics_registry_, "");
    meta_service_impl_ = std::make_shared<MetaServiceImpl>(cache_manager_, metrics_reporter_);
    meta_service_ = std::make_shared<MetaServiceGRpc>(metrics_registry_, meta_service_impl_, registry_manager_);
    grpc::ServerBuilder builder;
    port_ = port < 0 ? GetFreePort() : port;
    ASSERT_LT(0, port_);
    std::string server_spec = std::string("0.0.0.0:") + std::to_string(port_);
    builder.AddListeningPort(server_spec, grpc::InsecureServerCredentials());
    builder.RegisterService(meta_service_.get());
    rpc_server_ = builder.BuildAndStart();
    ASSERT_TRUE(rpc_server_ != nullptr);
    KVCM_LOG_INFO("start service");
}

void GrpcStubTest::InitStub() {
    stub_ = std::make_shared<GrpcStub>();
    ASSERT_EQ(ER_OK, stub_->AddConnection("0.0.0.0:" + std::to_string(port_), 1000));
    KVCM_LOG_INFO("stub connected");
}

ModelDeployment GrpcStubTest::createModelDeployment(int32_t tp_size, int32_t pp_size) {
    ModelDeployment model_deployment;
    model_deployment.set_model_name("test_model");
    model_deployment.set_tp_size(tp_size);
    model_deployment.set_pp_size(pp_size);
    model_deployment.set_dtype("FP8");
    return model_deployment;
}

Stub::LocationSpecInfoMap GrpcStubTest::createLocationSpecInfos(int32_t spec_size) {
    Stub::LocationSpecInfoMap results;
    for (int i = 0; i < spec_size; i++) {
        results.emplace(std::string("tp") + std::to_string(i), 1024);
    }
    return results;
}

TEST_F(GrpcStubTest, TestBadAddress) {
    stub_ = std::make_shared<GrpcStub>();
    ASSERT_EQ(ER_CONNECT_FAIL, stub_->AddConnection("0.0.0.0:" + std::to_string(port_ + 1), 1000));
}

TEST_F(GrpcStubTest, TestRetry) {
    std::thread t_stub([this]() {
        stub_ = std::make_shared<GrpcStub>(100, 100000);
        ASSERT_EQ(ER_OK, stub_->AddConnection("0.0.0.0:" + std::to_string(port_), 10000));
        KVCM_LOG_INFO("retry stub connected");
        auto expected = std::pair<ClientErrorCode, std::string>(ER_OK, default_storage_configs);
        ASSERT_EQ(
            expected,
            stub_->RegisterInstance(
                "trace1", "default", "instance1", 64, createLocationSpecInfos(2), createModelDeployment(2, 1), {}));
        std::string write_session_id;
        Locations target_locations;
        {
            auto [success, write_location] =
                stub_->StartWriteCache("trace2", "instance1", {1, 2, 3, 4}, {}, {}, 1000000);
            ASSERT_EQ(ER_OK, success);
            write_session_id = write_location.write_session_id;
            target_locations = write_location.locations;
        }
        std::this_thread::sleep_for(std::chrono::seconds(1));
        {
            auto [success, locations] = stub_->GetCacheLocation(
                "trace3", "instance1", QueryType::QT_PREFIX_MATCH, {1, 2, 3, 4}, {}, static_cast<size_t>(0), 0, {});
            ASSERT_EQ(ER_OK, success);
            ASSERT_EQ(Locations({}), locations);
        }
        std::this_thread::sleep_for(std::chrono::seconds(1));
        {
            BlockMask success_block = BlockMaskVector({true, true, false, false});
            ASSERT_EQ(ER_OK, stub_->FinishWriteCache("trace4", "instance1", write_session_id, success_block, {}));
            target_locations.resize(target_locations.size() - 2);
        }
        std::this_thread::sleep_for(std::chrono::seconds(1));
        {
            auto [success, locations] = stub_->GetCacheLocation(
                "trace5", "instance1", QueryType::QT_PREFIX_MATCH, {1, 2, 3, 4}, {}, static_cast<size_t>(0), 0, {});
            ASSERT_EQ(ER_OK, success);
            ExpectLocationsEq(target_locations, locations);
            ASSERT_FALSE(HasFailure());
        }
        std::this_thread::sleep_for(std::chrono::seconds(1));
        {
            auto [success, locations] = stub_->GetCacheLocation(
                "trace6", "instance1", QueryType::QT_PREFIX_MATCH, {1, 2, 3}, {}, static_cast<size_t>(1), 0, {});
            ASSERT_EQ(ER_OK, success);
            ExpectLocationsEq(Locations(target_locations.begin() + 1, target_locations.end()), locations);
            ASSERT_FALSE(HasFailure());
        }
    });
    for (int i = 0; i < 4; ++i) {
        rpc_server_->Shutdown();
        std::this_thread::sleep_for(std::chrono::seconds(1));
        StartService(port_);
    }
    t_stub.join();
}

TEST_F(GrpcStubTest, TestRegisterInstance) {
    auto expected = std::pair<ClientErrorCode, std::string>(ER_OK, default_storage_configs);
    ASSERT_EQ(expected,
              stub_->RegisterInstance(
                  "trace1", "default", "instance1", 64, createLocationSpecInfos(), createModelDeployment(), {}));
    ASSERT_EQ(expected,
              stub_->RegisterInstance(
                  "trace2", "default", "instance2", 64, createLocationSpecInfos(4), createModelDeployment(4), {}));
}

TEST_F(GrpcStubTest, TestStartWriteCache) {
    auto expected = std::pair<ClientErrorCode, std::string>(ER_OK, default_storage_configs);
    ASSERT_EQ(expected,
              stub_->RegisterInstance(
                  "trace1", "default", "instance1", 64, createLocationSpecInfos(2), createModelDeployment(2, 1), {}));
    {
        auto [success, write_location] = stub_->StartWriteCache("trace2", "instance1", {1, 2, 3, 4}, {}, {}, 1000000);
        ASSERT_EQ(ER_OK, success);
        ASSERT_EQ(0, std::get<BlockMaskOffset>(write_location.block_mask));
        ExpectLocationsValid(write_location.locations, 4);
        ASSERT_FALSE(HasFailure());
    }
    {
        auto [success, write_location] = stub_->StartWriteCache("trace3", "instance1", {1, 2, 3, 4}, {}, {}, 1000000);
        ASSERT_EQ(ER_OK, success);
        ASSERT_EQ(4, std::get<BlockMaskOffset>(write_location.block_mask));
        ASSERT_EQ(Locations({}), write_location.locations);
    }
    {
        auto [success, write_location] =
            stub_->StartWriteCache("trace4", "instance1", {1, 2, 3, 4, 5, 6}, {}, {}, 1000000);
        ASSERT_EQ(ER_OK, success);
        ASSERT_EQ(4, std::get<BlockMaskOffset>(write_location.block_mask));
        ExpectLocationsValid(write_location.locations, 2);
        ASSERT_FALSE(HasFailure());
    }
    {
        auto [success, write_location] = stub_->StartWriteCache("trace4", "instance1", {5, 6, 7, 8}, {}, {}, 1000000);
        ASSERT_EQ(ER_OK, success);
        ASSERT_EQ(2, std::get<BlockMaskOffset>(write_location.block_mask));
        ExpectLocationsValid(write_location.locations, 2);
        ASSERT_FALSE(HasFailure());
    }
    {
        auto [success, write_location] = stub_->StartWriteCache("trace5", "instance1", {5, 55, 66, 6}, {}, {}, 1000000);
        ASSERT_EQ(ER_OK, success);
        ASSERT_EQ(BlockMaskVector({true, false, false, true}), std::get<BlockMaskVector>(write_location.block_mask));
        ExpectLocationsValid(write_location.locations, 2);
        ASSERT_FALSE(HasFailure());
    }
}

TEST_F(GrpcStubTest, TestStartWriteCacheWithLocationSpecGroup) {
    kv_cache_manager::Stub::LocationSpecInfoMap location_spec_info_map = {
        {"tp0_F0", 1024},
        {"tp1_F0", 1024},
        {"tp0_L1", 1024},
        {"tp1_L1", 1024},
    };
    kv_cache_manager::Stub::LocationSpecGroups location_spec_groups = {
        {"F0L1", {"tp0_F0", "tp1_F0", "tp0_L1", "tp1_L1"}},
        {"F0", {"tp0_F0", "tp1_F0"}},
    };
    auto expected = std::pair<ClientErrorCode, std::string>(ER_OK, default_storage_configs);
    ASSERT_EQ(expected,
              stub_->RegisterInstance("trace1",
                                      "default",
                                      "instance1",
                                      64,
                                      location_spec_info_map,
                                      createModelDeployment(2, 1),
                                      location_spec_groups));
    {
        auto [success, write_location] = stub_->StartWriteCache("trace4", "instance1", {1, 2, 3, 4}, {}, {}, 1000000);
        ASSERT_EQ(ER_OK, success);
        ASSERT_EQ(0, std::get<BlockMaskOffset>(write_location.block_mask));
        ASSERT_EQ(4, write_location.locations.size());
        for (const auto &location : write_location.locations) {
            ASSERT_EQ(4, location.size());
            ASSERT_EQ(std::string("tp0_F0"), location[0].spec_name);
            ASSERT_EQ(std::string("tp0_L1"), location[1].spec_name);
            ASSERT_EQ(std::string("tp1_F0"), location[2].spec_name);
            ASSERT_EQ(std::string("tp1_L1"), location[3].spec_name);
        }
        ASSERT_FALSE(HasFailure());
    }
    {
        auto [success, write_location] =
            stub_->StartWriteCache("trace4", "instance1", {11, 12, 13, 14}, {}, {"F0", "F0L1", "F0", "F0L1"}, 1000000);
        ASSERT_EQ(ER_OK, success);
        ASSERT_EQ(0, std::get<BlockMaskOffset>(write_location.block_mask));
        ASSERT_EQ(4, write_location.locations.size());
        for (size_t i : std::vector<size_t>({1, 3})) {
            const auto &location = write_location.locations[i];
            ASSERT_EQ(4, location.size());
            ASSERT_EQ(std::string("tp0_F0"), location[0].spec_name);
            ASSERT_EQ(std::string("tp0_L1"), location[1].spec_name);
            ASSERT_EQ(std::string("tp1_F0"), location[2].spec_name);
            ASSERT_EQ(std::string("tp1_L1"), location[3].spec_name);
        }
        for (size_t i : std::vector<size_t>({0, 2})) {
            const auto &location = write_location.locations[i];
            ASSERT_EQ(2, location.size());
            ASSERT_EQ(std::string("tp0_F0"), location[0].spec_name);
            ASSERT_EQ(std::string("tp1_F0"), location[1].spec_name);
        }
        ASSERT_FALSE(HasFailure());
    }
    {
        auto [success, write_location] =
            stub_->StartWriteCache("trace4", "instance1", {11, 22, 23, 24}, {}, {"F0", "F0L1", "F0", "F0L1"}, 1000000);
        ASSERT_EQ(ER_OK, success);
        ASSERT_EQ(1, std::get<BlockMaskOffset>(write_location.block_mask));
        ASSERT_EQ(3, write_location.locations.size());
        for (size_t i : std::vector<size_t>({0, 2})) {
            const auto &location = write_location.locations[i];
            ASSERT_EQ(4, location.size());
            ASSERT_EQ(std::string("tp0_F0"), location[0].spec_name);
            ASSERT_EQ(std::string("tp0_L1"), location[1].spec_name);
            ASSERT_EQ(std::string("tp1_F0"), location[2].spec_name);
            ASSERT_EQ(std::string("tp1_L1"), location[3].spec_name);
        }
        {
            const auto &location = write_location.locations[1];
            ASSERT_EQ(2, location.size());
            ASSERT_EQ(std::string("tp0_F0"), location[0].spec_name);
            ASSERT_EQ(std::string("tp1_F0"), location[1].spec_name);
        }
        ASSERT_FALSE(HasFailure());
    }
    { // not exist group
        auto [success, write_location] = stub_->StartWriteCache(
            "trace4", "instance1", {31, 32, 33, 34}, {}, {"F0", "F0L1", "F0", "F0L1_notexist"}, 1000000);
        ASSERT_EQ(ER_SERVICE_INTERNAL_ERROR, success);
    }
}

TEST_F(GrpcStubTest, TestFinishWriteCacheSuccess) {
    auto expected = std::pair<ClientErrorCode, std::string>(ER_OK, default_storage_configs);
    ASSERT_EQ(expected,
              stub_->RegisterInstance(
                  "trace1", "default", "instance1", 64, createLocationSpecInfos(2), createModelDeployment(2, 1), {}));
    std::string write_session_id;
    {
        auto [success, write_location] = stub_->StartWriteCache("trace2", "instance1", {1, 2, 3, 4}, {}, {}, 1000000);
        ASSERT_EQ(ER_OK, success);
        write_session_id = write_location.write_session_id;
    }
    {
        BlockMask success_block = static_cast<size_t>(4);
        ASSERT_EQ(ER_OK, stub_->FinishWriteCache("trace3", "instance1", write_session_id, success_block, {}));
    }
}

TEST_F(GrpcStubTest, TestFinishWriteCacheFail) {
    auto expected = std::pair<ClientErrorCode, std::string>(ER_OK, default_storage_configs);
    ASSERT_EQ(expected,
              stub_->RegisterInstance(
                  "trace1", "default", "instance1", 64, createLocationSpecInfos(2), createModelDeployment(2, 1), {}));
    std::string write_session_id;
    {
        auto [success, write_location] = stub_->StartWriteCache("trace2", "instance1", {1, 2, 3, 4}, {}, {}, 1000000);
        ASSERT_EQ(ER_OK, success);
        write_session_id = write_location.write_session_id;
    }
    {
        BlockMask success_block = static_cast<size_t>(4);
        ASSERT_EQ(ER_SERVICE_INTERNAL_ERROR,
                  stub_->FinishWriteCache("trace3", "instance1", write_session_id + "_bug", success_block, {}));
    }
    {
        BlockMask success_block = BlockMaskVector(2, true);
        ASSERT_EQ(ER_SERVICE_INVALID_ARGUMENT,
                  stub_->FinishWriteCache("trace4", "instance1", write_session_id, success_block, {}));
    }
    {
        BlockMask success_block = static_cast<size_t>(6);
        ASSERT_EQ(ER_SERVICE_INTERNAL_ERROR,
                  stub_->FinishWriteCache("trace5", "instance1", write_session_id, success_block, {}));
    }
}

TEST_F(GrpcStubTest, TestFinishWriteCacheTimout) {
    auto expected = std::pair<ClientErrorCode, std::string>(ER_OK, default_storage_configs);
    ASSERT_EQ(expected,
              stub_->RegisterInstance(
                  "trace1", "default", "instance1", 64, createLocationSpecInfos(2), createModelDeployment(2, 1), {}));
    std::string write_session_id;
    {
        auto [success, write_location] = stub_->StartWriteCache("trace2", "instance1", {1, 2, 3, 4}, {}, {}, 1);
        ASSERT_EQ(ER_OK, success);
        write_session_id = write_location.write_session_id;
    }
    std::this_thread::sleep_for(std::chrono::seconds(7));
    {
        BlockMask success_block = static_cast<size_t>(4);
        ASSERT_NE(ER_OK, stub_->FinishWriteCache("trace3", "instance1", write_session_id, success_block, {}));
    }
}

TEST_F(GrpcStubTest, TestGetCacheLocationPrefixMatch) {
    auto expected = std::pair<ClientErrorCode, std::string>(ER_OK, default_storage_configs);
    ASSERT_EQ(expected,
              stub_->RegisterInstance(
                  "trace1", "default", "instance1", 64, createLocationSpecInfos(2), createModelDeployment(2, 1), {}));
    std::string write_session_id;
    Locations target_locations;
    {
        auto [success, write_location] = stub_->StartWriteCache("trace2", "instance1", {1, 2, 3, 4}, {}, {}, 1000000);
        ASSERT_EQ(ER_OK, success);
        write_session_id = write_location.write_session_id;
        target_locations = write_location.locations;
    }
    {
        auto [success, locations] = stub_->GetCacheLocation(
            "trace3", "instance1", QueryType::QT_PREFIX_MATCH, {1, 2, 3, 4}, {}, static_cast<size_t>(0), 0, {});
        ASSERT_EQ(ER_OK, success);
        ASSERT_EQ(Locations({}), locations);
    }
    {
        BlockMask success_block = BlockMaskVector({true, true, false, false});
        ASSERT_EQ(ER_OK, stub_->FinishWriteCache("trace4", "instance1", write_session_id, success_block, {}));
        target_locations.resize(target_locations.size() - 2); // Only the successful blocks
    }
    {
        auto [success, locations] = stub_->GetCacheLocation(
            "trace5", "instance1", QueryType::QT_PREFIX_MATCH, {1, 2, 3, 4}, {}, static_cast<size_t>(0), 0, {});
        ASSERT_EQ(ER_OK, success);
        ExpectLocationsEq(target_locations, locations);
        ASSERT_FALSE(HasFailure());
    }
    {
        auto [success, locations] = stub_->GetCacheLocation(
            "trace6", "instance1", QueryType::QT_PREFIX_MATCH, {1, 2, 3}, {}, static_cast<size_t>(1), 0, {});
        ASSERT_EQ(ER_OK, success);
        ASSERT_EQ(locations.size(), 1);
        ExpectLocationsEq(Locations({target_locations[1]}), locations);
        ASSERT_FALSE(HasFailure());
    }
    {
        auto [success, locations] = stub_->GetCacheLocation(
            "trace6", "instance1", QueryType::QT_PREFIX_MATCH, {1, 2, 3}, {}, static_cast<size_t>(6), 0, {});
        ASSERT_EQ(ER_OK, success);
        ASSERT_EQ(Locations({}), locations);
    }
    {
        BlockMask block_mask = BlockMaskVector({true, false, false, false});
        auto [success, locations] = stub_->GetCacheLocation(
            "trace6", "instance1", QueryType::QT_PREFIX_MATCH, {1, 2, 3}, {}, block_mask, 0, {});
        ASSERT_EQ(ER_OK, success);
        ExpectLocationsEq(Locations({target_locations[1]}), locations);
        ASSERT_FALSE(HasFailure());
    }
}

TEST_F(GrpcStubTest, TestGetCacheLocationBatchGet) {
    auto expected = std::pair<ClientErrorCode, std::string>(ER_OK, default_storage_configs);
    ASSERT_EQ(expected,
              stub_->RegisterInstance(
                  "trace1", "default", "instance1", 64, createLocationSpecInfos(2), createModelDeployment(2, 1), {}));
    std::string write_session_id;
    Locations target_locations;
    {
        auto [success, write_location] = stub_->StartWriteCache("trace2", "instance1", {1, 2, 3, 4}, {}, {}, 1000000);
        ASSERT_EQ(ER_OK, success);
        write_session_id = write_location.write_session_id;
        target_locations = write_location.locations;
    }
    {
        auto [success, locations] = stub_->GetCacheLocation(
            "trace3", "instance1", QueryType::QT_BATCH_GET, {1, 2, 3, 4}, {}, static_cast<size_t>(0), 0, {});
        ASSERT_EQ(ER_OK, success);
        ASSERT_EQ(Locations({{{"tp0", ""}, {"tp1", ""}},
                             {{"tp0", ""}, {"tp1", ""}},
                             {{"tp0", ""}, {"tp1", ""}},
                             {{"tp0", ""}, {"tp1", ""}}}),
                  locations);
    }
    {
        BlockMask success_block = BlockMaskVector({true, true, true, false});
        ASSERT_EQ(ER_OK, stub_->FinishWriteCache("trace4", "instance1", write_session_id, success_block, {}));
    }
    {
        auto [success, locations] = stub_->GetCacheLocation(
            "trace5", "instance1", QueryType::QT_BATCH_GET, {0, 1, 22, 3, 4}, {}, static_cast<size_t>(0), 0, {});
        ASSERT_EQ(ER_OK, success);
        Locations expected_batch = {{{"tp0", ""}, {"tp1", ""}},
                                    target_locations[0],
                                    {{"tp0", ""}, {"tp1", ""}},
                                    target_locations[2],
                                    {{"tp0", ""}, {"tp1", ""}}};
        ExpectLocationsEq(expected_batch, locations);
        ASSERT_FALSE(HasFailure());
    }
}

TEST_F(GrpcStubTest, TestGetCacheLocationReverseRollSlideWindowMatch) {
    auto expected = std::pair<ClientErrorCode, std::string>(ER_OK, default_storage_configs);
    ASSERT_EQ(expected,
              stub_->RegisterInstance(
                  "trace1", "default", "instance1", 64, createLocationSpecInfos(2), createModelDeployment(2, 1), {}));
    std::string write_session_id;
    Locations target_locations;
    {
        auto [success, write_location] =
            stub_->StartWriteCache("trace2", "instance1", {1, 2, 3, 4, 5, 6}, {}, {}, 1000000);
        ASSERT_EQ(ER_OK, success);
        write_session_id = write_location.write_session_id;
        target_locations = write_location.locations;
    }
    {
        auto [success, locations] = stub_->GetCacheLocation("trace3",
                                                            "instance1",
                                                            QueryType::QT_REVERSE_ROLL_SW_MATCH,
                                                            {1, 2, 3, 4, 5, 6},
                                                            {},
                                                            static_cast<size_t>(0),
                                                            3,
                                                            {});
        ASSERT_EQ(ER_OK, success);
        ASSERT_EQ(Locations({{{"tp0", ""}, {"tp1", ""}},
                             {{"tp0", ""}, {"tp1", ""}},
                             {{"tp0", ""}, {"tp1", ""}},
                             {{"tp0", ""}, {"tp1", ""}},
                             {{"tp0", ""}, {"tp1", ""}},
                             {{"tp0", ""}, {"tp1", ""}}}),
                  locations);
    }
    {
        BlockMask success_block = BlockMaskVector({true, true, true, true, true, false});
        ASSERT_EQ(ER_OK, stub_->FinishWriteCache("trace4", "instance1", write_session_id, success_block, {}));
        // Update target_locations to reflect successful writes
        // Elements 0-4 keep their original values (successful writes)
        // Element 5 becomes empty (failed write)
        target_locations[5] = {};
    }
    {
        auto [success, locations] = stub_->GetCacheLocation("trace5",
                                                            "instance1",
                                                            QueryType::QT_REVERSE_ROLL_SW_MATCH,
                                                            {1, 2, 3, 4, 5, 6},
                                                            {},
                                                            static_cast<size_t>(0),
                                                            3,
                                                            {});
        ASSERT_EQ(ER_OK, success);
        Locations expected_locations_rrsw = {{{"tp0", ""}, {"tp1", ""}},
                                             {{"tp0", ""}, {"tp1", ""}},
                                             target_locations[2],
                                             target_locations[3],
                                             target_locations[4],
                                             {{"tp0", ""}, {"tp1", ""}}};
        ExpectLocationsEq(expected_locations_rrsw, locations);
        ASSERT_FALSE(HasFailure());
    }
    {
        auto [success, locations] = stub_->GetCacheLocation("trace5",
                                                            "instance1",
                                                            QueryType::QT_REVERSE_ROLL_SW_MATCH,
                                                            {1, 2, 3, 4, 6, 7, 8},
                                                            {},
                                                            static_cast<size_t>(0),
                                                            3,
                                                            {});
        ASSERT_EQ(ER_OK, success);
        Locations expected_locations_rrsw2 = {{{"tp0", ""}, {"tp1", ""}},
                                              target_locations[1],
                                              target_locations[2],
                                              target_locations[3],
                                              {{"tp0", ""}, {"tp1", ""}},
                                              {{"tp0", ""}, {"tp1", ""}},
                                              {{"tp0", ""}, {"tp1", ""}}};
        ExpectLocationsEq(expected_locations_rrsw2, locations);
        ASSERT_FALSE(HasFailure());
    }
    {
        auto [success, locations] = stub_->GetCacheLocation("trace5",
                                                            "instance1",
                                                            QueryType::QT_REVERSE_ROLL_SW_MATCH,
                                                            {1, 2, 3, 10, 5, 7, 8},
                                                            {},
                                                            static_cast<size_t>(0),
                                                            3,
                                                            {});
        ASSERT_EQ(ER_OK, success);
        Locations expected_locations_rrsw3 = {target_locations[0],
                                              target_locations[1],
                                              target_locations[2],
                                              {{"tp0", ""}, {"tp1", ""}},
                                              {{"tp0", ""}, {"tp1", ""}},
                                              {{"tp0", ""}, {"tp1", ""}},
                                              {{"tp0", ""}, {"tp1", ""}}};
        ExpectLocationsEq(expected_locations_rrsw3, locations);
        ASSERT_FALSE(HasFailure());
    }
}

TEST_F(GrpcStubTest, TestGetCacheMeta) {
    auto expected = std::pair<ClientErrorCode, std::string>(ER_OK, default_storage_configs);
    ASSERT_EQ(expected,
              stub_->RegisterInstance(
                  "trace1", "default", "instance1", 64, createLocationSpecInfos(2), createModelDeployment(2, 1), {}));
    std::string write_session_id;
    Locations target_locations;
    auto meta_pred = [](const std::vector<std::string> &expected, const std::vector<std::string> &metas) -> bool {
        if (expected.size() != metas.size()) {
            KVCM_LOG_ERROR("meta_pred size error, expected [%ld], real [%ld]", expected.size(), metas.size());
            return false;
        }
        for (int i = 0; i < expected.size(); ++i) {
            std::map<std::string, std::string> meta;
            if (!Jsonizable::FromJsonString(metas[i], meta)) {
                KVCM_LOG_ERROR("meta_pred [%d] json error [%s]", i, metas[i].c_str());
                return false;
            }
            auto iter = meta.find("status");
            if (iter == meta.end()) {
                KVCM_LOG_ERROR("meta_pred not find status in meta");
                return false;
            }
            if (expected[i] != iter->second) {
                KVCM_LOG_ERROR(
                    "meta_pred [%d] status expected [%s], real [%s]", i, expected[i].c_str(), iter->second.c_str());
                return false;
            }
        }
        return true;
    };
    {
        auto [success, write_location] = stub_->StartWriteCache("trace2", "instance1", {1, 2, 3, 4}, {}, {}, 1000000);
        ASSERT_EQ(ER_OK, success);
        write_session_id = write_location.write_session_id;
        target_locations = write_location.locations;
    }
    {
        auto [success, meta] =
            stub_->GetCacheMeta("trace3", "instance1", {1, 2, 3, 4}, {}, static_cast<size_t>(0), 100);
        ASSERT_EQ(ER_OK, success);
        ExpectLocationsEq(target_locations, meta.locations);
        ASSERT_FALSE(HasFailure());
        ASSERT_PRED2(meta_pred, std::vector<std::string>(4, "CLS_WRITING"), meta.metas);
    }
    {
        auto [success, meta] =
            stub_->GetCacheMeta("trace4", "instance1", {1, 2, 3, 4, 5}, {}, static_cast<size_t>(0), 100);
        ASSERT_EQ(ER_OK, success);
        Locations extended_target_locations = target_locations;
        extended_target_locations.push_back({});
        ExpectLocationsEq(extended_target_locations, meta.locations);
        ASSERT_FALSE(HasFailure());
        std::vector<std::string> expected(4, "CLS_WRITING");
        expected.push_back("CLS_NOT_FOUND");
        ASSERT_PRED2(meta_pred, expected, meta.metas);
    }
    {
        BlockMask success_block = BlockMaskVector({true, true, false, false});
        ASSERT_EQ(ER_OK, stub_->FinishWriteCache("trace4", "instance1", write_session_id, success_block, {}));
        // Update target_locations to reflect successful writes
        target_locations[2] = {};
        target_locations[3] = {};
    }

    // only 0 1 is success
    Locations expected_locations_after_finished(target_locations);

    {
        bool result = WaitUntil(
            [this, &expected_locations_after_finished, &meta_pred]() {
                auto [success, meta] =
                    stub_->GetCacheMeta("trace3", "instance1", {1, 2, 3, 4}, {}, static_cast<size_t>(0), 100);
                if (success != ER_OK) {
                    return false;
                }
                if (expected_locations_after_finished != meta.locations) {
                    return false;
                }
                if (!meta_pred(
                        std::vector<std::string>({"CLS_SERVING", "CLS_SERVING", "CLS_NOT_FOUND", "CLS_NOT_FOUND"}),
                        meta.metas)) {
                    return false;
                }
                return true;
            },
            5000,
            100);
        ASSERT_TRUE(result);
    }
    expected_locations_after_finished.push_back({});
    {
        bool result = WaitUntil(
            [this, &expected_locations_after_finished, &meta_pred]() {
                auto [success, meta] =
                    stub_->GetCacheMeta("trace3", "instance1", {1, 2, 3, 4, 5}, {}, static_cast<size_t>(0), 100);
                if (success != ER_OK) {
                    return false;
                }
                if (expected_locations_after_finished != meta.locations) {
                    return false;
                }
                if (!meta_pred(std::vector<std::string>(
                                   {"CLS_SERVING", "CLS_SERVING", "CLS_NOT_FOUND", "CLS_NOT_FOUND", "CLS_NOT_FOUND"}),
                               meta.metas)) {
                    return false;
                }
                return true;
            },
            5000,
            100);
        ASSERT_TRUE(result);
    }
}

TEST_F(GrpcStubTest, TestSpanTracer) {
    auto expected = std::pair<ClientErrorCode, std::string>(ER_OK, default_storage_configs);
    ASSERT_EQ(expected,
              stub_->RegisterInstance("trace1__kvcm_need_span_tracer",
                                      "default",
                                      "instance1",
                                      64,
                                      createLocationSpecInfos(2),
                                      createModelDeployment(2, 1),
                                      {}));
    std::string write_session_id;
    Locations target_locations;
    {
        auto [success, write_location] =
            stub_->StartWriteCache("trace2__kvcm_need_span_tracer", "instance1", {1, 2, 3, 4}, {}, {}, 1000000);
        ASSERT_EQ(ER_OK, success);
        write_session_id = write_location.write_session_id;
        target_locations = write_location.locations;
    }
    {
        auto [success, locations] = stub_->GetCacheLocation("trace3__kvcm_need_span_tracer",
                                                            "instance1",
                                                            QueryType::QT_PREFIX_MATCH,
                                                            {1, 2, 3, 4},
                                                            {},
                                                            static_cast<size_t>(0),
                                                            0,
                                                            {});
        ASSERT_EQ(ER_OK, success);
        ASSERT_EQ(Locations({}), locations);
    }
    {
        BlockMask success_block = BlockMaskVector({true, true, false, false});
        ASSERT_EQ(
            ER_OK,
            stub_->FinishWriteCache("trace4__kvcm_need_span_tracer", "instance1", write_session_id, success_block, {}));
        target_locations.resize(target_locations.size() - 2);
    }
    {
        auto [success, locations] = stub_->GetCacheLocation("trace5__kvcm_need_span_tracer",
                                                            "instance1",
                                                            QueryType::QT_PREFIX_MATCH,
                                                            {1, 2, 3, 4},
                                                            {},
                                                            static_cast<size_t>(0),
                                                            0,
                                                            {});
        ASSERT_EQ(ER_OK, success);
        ExpectLocationsEq(target_locations, locations);
        ASSERT_FALSE(HasFailure());
    }
}