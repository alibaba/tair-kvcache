#include <atomic>
#include <grpcpp/grpcpp.h>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "kv_cache_manager/client/include/kv_meta_client.h"
#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/protocol/protobuf/kv_meta_service.grpc.pb.h"

namespace kv_cache_manager {
namespace {

class FakeKvMetaService final : public proto::kv_meta::MetaService::Service {
public:
    explicit FakeKvMetaService(bool standby = false) : standby_(standby) {}

    void set_wrong_start_size(bool value) { wrong_start_size_.store(value); }
    void set_wrong_uri_size(bool value) { wrong_uri_size_.store(value); }
    void set_wrong_uri_scheme(bool value) { wrong_uri_scheme_.store(value); }
    void set_omit_last_start_location(bool value) { omit_last_start_location_.store(value); }
    void set_extra_start_mask_value(bool value) { extra_start_mask_value_.store(value); }

    grpc::Status RegisterInstance(grpc::ServerContext *,
                                  const proto::kv_meta::RegisterInstanceRequest *request,
                                  proto::kv_meta::RegisterInstanceResponse *response) override {
        ++register_calls;
        if (!SetReadyStatus(response)) {
            return grpc::Status::OK;
        }
        last_instance_id = request->instance_id();
        response->set_storage_configs("{}");
        return grpc::Status::OK;
    }

    grpc::Status GetInstanceInfo(grpc::ServerContext *,
                                 const proto::kv_meta::GetInstanceInfoRequest *request,
                                 proto::kv_meta::GetInstanceInfoResponse *response) override {
        if (!SetReadyStatus(response)) {
            return grpc::Status::OK;
        }
        response->set_instance_group("objects");
        auto *info = response->mutable_instance_info();
        info->set_quota_group_name("objects-quota");
        info->set_instance_group_name("objects");
        info->set_instance_id(request->instance_id());
        return grpc::Status::OK;
    }

    grpc::Status Get(grpc::ServerContext *,
                     const proto::kv_meta::GetRequest *request,
                     proto::kv_meta::GetResponse *response) override {
        if (!SetReadyStatus(response)) {
            return grpc::Status::OK;
        }
        for (const auto &key : request->keys()) {
            const bool hit = key != "miss";
            response->mutable_hit_mask()->add_values(hit);
            auto *location = response->add_locations();
            if (hit) {
                FillLocation(key == "a" ? 17 : 33, location);
            }
        }
        return grpc::Status::OK;
    }

    grpc::Status PutStart(grpc::ServerContext *,
                          const proto::kv_meta::PutStartRequest *request,
                          proto::kv_meta::PutStartResponse *response) override {
        ++put_start_calls;
        if (!SetReadyStatus(response)) {
            return grpc::Status::OK;
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            last_start_sizes.assign(request->value_sizes().begin(), request->value_sizes().end());
        }
        std::size_t write_count = 0;
        for (int i = 0; i < request->keys_size(); ++i) {
            const bool masked = request->keys(i) == "exists";
            response->mutable_key_mask()->add_values(masked);
            if (masked) {
                continue;
            }
            std::uint64_t size = request->value_sizes(i);
            if (wrong_start_size_.load()) {
                ++size;
            }
            if (!omit_last_start_location_.load() || i + 1 != request->keys_size()) {
                auto *location = response->add_locations();
                FillLocation(size, location);
                if (wrong_uri_size_.load()) {
                    location->mutable_location_specs(0)->set_uri(
                        "file://nfs/value?offset=0&size=" + std::to_string(size + 1));
                }
                if (wrong_uri_scheme_.load()) {
                    location->mutable_location_specs(0)->set_uri(
                        "dummy://nfs/value?offset=0&size=" + std::to_string(size));
                }
            }
            ++write_count;
        }
        if (write_count != 0) {
            response->set_write_session_id("session-1");
        }
        if (extra_start_mask_value_.load()) {
            response->mutable_key_mask()->add_values(false);
        }
        return grpc::Status::OK;
    }

    grpc::Status PutFinish(grpc::ServerContext *,
                           const proto::kv_meta::PutFinishRequest *request,
                           proto::kv_meta::CommonResponse *response) override {
        ++put_finish_calls;
        if (!SetReadyStatus(response)) {
            return grpc::Status::OK;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        last_finish_successes.assign(request->success_keys().values().begin(),
                                     request->success_keys().values().end());
        return grpc::Status::OK;
    }

    grpc::Status Remove(grpc::ServerContext *,
                        const proto::kv_meta::RemoveRequest *,
                        proto::kv_meta::CommonResponse *response) override {
        SetReadyStatus(response);
        return grpc::Status::OK;
    }

    grpc::Status Trim(grpc::ServerContext *,
                      const proto::kv_meta::TrimRequest *,
                      proto::kv_meta::CommonResponse *response) override {
        SetReadyStatus(response);
        return grpc::Status::OK;
    }

    std::vector<std::uint64_t> StartSizes() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return last_start_sizes;
    }

    std::vector<bool> FinishSuccesses() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return last_finish_successes;
    }

    std::atomic<int> register_calls{0};
    std::atomic<int> put_start_calls{0};
    std::atomic<int> put_finish_calls{0};
    std::string last_instance_id;

private:
    template <typename Response>
    bool SetReadyStatus(Response *response) const {
        response->mutable_header()->mutable_status()->set_code(standby_ ? proto::kv_meta::SERVER_NOT_LEADER
                                                                        : proto::kv_meta::OK);
        return !standby_;
    }

    static void FillLocation(std::uint64_t size, proto::kv_meta::ValueLocation *location) {
        location->set_type(proto::kv_meta::ST_NFS);
        location->set_spec_size(1);
        location->set_value_size(size);
        auto *spec = location->add_location_specs();
        spec->set_name("value");
        spec->set_uri("file://nfs/value?offset=0&size=" + std::to_string(size));
    }

    const bool standby_;
    std::atomic<bool> wrong_start_size_{false};
    std::atomic<bool> wrong_uri_size_{false};
    std::atomic<bool> wrong_uri_scheme_{false};
    std::atomic<bool> omit_last_start_location_{false};
    std::atomic<bool> extra_start_mask_value_{false};
    mutable std::mutex mutex_;
    std::vector<std::uint64_t> last_start_sizes;
    std::vector<bool> last_finish_successes;
};

class RunningServer {
public:
    explicit RunningServer(grpc::Service *service) {
        grpc::ServerBuilder builder;
        builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port_);
        builder.RegisterService(service);
        server_ = builder.BuildAndStart();
    }

    ~RunningServer() {
        if (server_) {
            server_->Shutdown();
            server_->Wait();
        }
    }

    bool valid() const { return server_ != nullptr && port_ > 0; }
    std::string address() const { return "127.0.0.1:" + std::to_string(port_); }

private:
    int port_ = 0;
    std::unique_ptr<grpc::Server> server_;
};

TEST(KvMetaClientTest, PreservesPerValueSizesAndAlignedResults) {
    FakeKvMetaService service;
    RunningServer server(&service);
    ASSERT_TRUE(server.valid());

    auto client = KvMetaClient::Create({{server.address()}, "emb-instance", 1000});
    ASSERT_TRUE(client);
    auto [register_ec, storage_config] = client->RegisterInstance("trace-register", "objects", "emb");
    EXPECT_EQ(ER_OK, register_ec);
    EXPECT_EQ("{}", storage_config);
    EXPECT_EQ("emb-instance", service.last_instance_id);

    auto [info_ec, info] = client->GetInstanceInfo("trace-info");
    ASSERT_EQ(ER_OK, info_ec);
    EXPECT_EQ("objects", info.instance_group_name);
    EXPECT_EQ("emb-instance", info.instance_id);

    auto [start_ec, start] =
        client->StartWrite("trace-start", {"exists", "a", "b"}, {1, 17, 33}, 30);
    ASSERT_EQ(ER_OK, start_ec);
    EXPECT_EQ((std::vector<bool>{true, false, false}), start.key_mask);
    ASSERT_EQ(2, start.locations.size());
    EXPECT_EQ(17, start.locations[0].value_size);
    EXPECT_EQ(33, start.locations[1].value_size);
    EXPECT_EQ((std::vector<std::uint64_t>{1, 17, 33}), service.StartSizes());

    ASSERT_EQ(ER_OK, client->FinishWrite("trace-finish", start.write_session_id, {true, true}));
    EXPECT_EQ((std::vector<bool>{true, true}), service.FinishSuccesses());

    auto [get_ec, get] = client->Get("trace-get", {"a", "miss", "b"});
    ASSERT_EQ(ER_OK, get_ec);
    EXPECT_EQ((std::vector<bool>{true, false, true}), get.hit_mask);
    ASSERT_EQ(3, get.locations.size());
    EXPECT_EQ(17, get.locations[0].value_size);
    EXPECT_EQ(0, get.locations[1].value_size);
    EXPECT_EQ(33, get.locations[2].value_size);
    EXPECT_EQ(ER_OK, client->Remove("trace-remove", {"a"}));
    EXPECT_EQ(ER_OK, client->TrimAll("trace-trim"));
}

TEST(KvMetaClientTest, FailsOverAndAbortsAMismatchedAllocation) {
    FakeKvMetaService standby(true);
    FakeKvMetaService leader;
    leader.set_wrong_start_size(true);
    RunningServer standby_server(&standby);
    RunningServer leader_server(&leader);
    ASSERT_TRUE(standby_server.valid());
    ASSERT_TRUE(leader_server.valid());

    auto client = KvMetaClient::Create(
        {{standby_server.address(), leader_server.address()}, "emb-instance", 1000});
    ASSERT_TRUE(client);
    EXPECT_EQ(ER_OK, client->RegisterInstance("trace-register", "objects", "emb").first);
    EXPECT_EQ(1, standby.register_calls.load());
    EXPECT_EQ(1, leader.register_calls.load());

    auto [start_ec, start] = client->StartWrite("trace-start", {"a"}, {17}, 30);
    EXPECT_EQ(ER_SERVICE_SIZE_MISMATCH, start_ec);
    EXPECT_TRUE(start.locations.empty());
    EXPECT_EQ(0, standby.put_start_calls.load());
    EXPECT_EQ(1, leader.put_start_calls.load());
    EXPECT_EQ(1, leader.put_finish_calls.load());
    EXPECT_EQ((std::vector<bool>{false}), leader.FinishSuccesses());
}

TEST(KvMetaClientTest, MalformedCompactLocationsAbortUsingTheRequestAlignedMask) {
    FakeKvMetaService service;
    service.set_omit_last_start_location(true);
    RunningServer server(&service);
    ASSERT_TRUE(server.valid());

    auto client = KvMetaClient::Create({{server.address()}, "emb-instance", 1000});
    ASSERT_TRUE(client);
    auto [start_ec, start] = client->StartWrite("trace-start", {"a", "b"}, {17, 33}, 30);
    EXPECT_EQ(ER_SERVICE_INTERNAL_ERROR, start_ec);
    EXPECT_TRUE(start.locations.empty());
    EXPECT_EQ(1, service.put_finish_calls.load());
    EXPECT_EQ((std::vector<bool>{false, false}), service.FinishSuccesses());
}

TEST(KvMetaClientTest, MalformedMaskFallsBackToCompactLocationCountWhenAborting) {
    FakeKvMetaService service;
    service.set_extra_start_mask_value(true);
    RunningServer server(&service);
    ASSERT_TRUE(server.valid());

    auto client = KvMetaClient::Create({{server.address()}, "emb-instance", 1000});
    ASSERT_TRUE(client);
    auto [start_ec, start] = client->StartWrite("trace-start", {"a", "b"}, {17, 33}, 30);
    EXPECT_EQ(ER_SERVICE_INTERNAL_ERROR, start_ec);
    EXPECT_TRUE(start.locations.empty());
    EXPECT_EQ(1, service.put_finish_calls.load());
    EXPECT_EQ((std::vector<bool>{false, false}), service.FinishSuccesses());
}

TEST(KvMetaClientTest, MalformedAllocationUriIsRejectedAndAborted) {
    FakeKvMetaService service;
    service.set_wrong_uri_size(true);
    RunningServer server(&service);
    ASSERT_TRUE(server.valid());

    auto client = KvMetaClient::Create({{server.address()}, "emb-instance", 1000});
    ASSERT_TRUE(client);
    EXPECT_EQ(ER_SERVICE_INTERNAL_ERROR,
              client->StartWrite("trace-size", {"a"}, {17}, 30).first);
    EXPECT_EQ(1, service.put_finish_calls.load());
    EXPECT_EQ((std::vector<bool>{false}), service.FinishSuccesses());

    service.set_wrong_uri_size(false);
    service.set_wrong_uri_scheme(true);
    EXPECT_EQ(ER_SERVICE_INTERNAL_ERROR,
              client->StartWrite("trace-scheme", {"a"}, {17}, 30).first);
    EXPECT_EQ(2, service.put_finish_calls.load());
    EXPECT_EQ((std::vector<bool>{false}), service.FinishSuccesses());
}

} // namespace
} // namespace kv_cache_manager
