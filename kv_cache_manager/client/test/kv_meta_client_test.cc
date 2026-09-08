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
    void set_write_in_progress(bool value) { write_in_progress_.store(value); }
    void set_wrong_uri_size(bool value) { wrong_uri_size_.store(value); }
    void set_wrong_uri_scheme(bool value) { wrong_uri_scheme_.store(value); }
    void set_non_singleton_uri(bool value) { non_singleton_uri_.store(value); }
    void set_oversized_session_id(bool value) { oversized_session_id_.store(value); }
    void set_omit_last_start_location(bool value) { omit_last_start_location_.store(value); }
    void set_extra_start_mask_value(bool value) { extra_start_mask_value_.store(value); }
    void set_register_transport_error(bool value) { register_transport_error_.store(value); }
    void set_get_transport_error(bool value) { get_transport_error_.store(value); }
    void set_put_start_transport_error(bool value) { put_start_transport_error_.store(value); }
    void set_put_finish_transport_error(bool value) { put_finish_transport_error_.store(value); }
    void set_remove_transport_error(bool value) { remove_transport_error_.store(value); }
    void set_trim_transport_error(bool value) { trim_transport_error_.store(value); }

    grpc::Status RegisterInstance(grpc::ServerContext *,
                                  const proto::kv_meta::RegisterInstanceRequest *request,
                                  proto::kv_meta::RegisterInstanceResponse *response) override {
        ++register_calls;
        if (register_transport_error_.load()) {
            return grpc::Status(grpc::StatusCode::UNAVAILABLE, "injected ambiguous transport error");
        }
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
        ++get_calls;
        if (get_transport_error_.load()) {
            return grpc::Status(grpc::StatusCode::UNAVAILABLE, "injected ambiguous transport error");
        }
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
        if (put_start_transport_error_.load()) {
            return grpc::Status(grpc::StatusCode::UNAVAILABLE, "injected ambiguous transport error");
        }
        if (!SetReadyStatus(response)) {
            return grpc::Status::OK;
        }
        if (write_in_progress_.load()) {
            response->mutable_header()->mutable_status()->set_code(proto::kv_meta::WRITE_IN_PROGRESS);
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
                    location->mutable_location_specs(0)->set_uri("file://nfs/value?offset=0&size=" +
                                                                 std::to_string(size + 1));
                }
                if (wrong_uri_scheme_.load()) {
                    location->mutable_location_specs(0)->set_uri("dummy://nfs/value?offset=0&size=" +
                                                                 std::to_string(size));
                }
                if (non_singleton_uri_.load()) {
                    location->mutable_location_specs(0)->set_uri("file://nfs/value?blkid=1&offset=0&size=" +
                                                                 std::to_string(size));
                }
            }
            ++write_count;
        }
        if (write_count != 0) {
            response->set_write_session_id(oversized_session_id_.load() ? std::string(513, 's') : "session-1");
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
        if (put_finish_transport_error_.load()) {
            return grpc::Status(grpc::StatusCode::UNAVAILABLE, "injected ambiguous transport error");
        }
        if (!SetReadyStatus(response)) {
            return grpc::Status::OK;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        last_finish_successes.assign(request->success_keys().values().begin(), request->success_keys().values().end());
        return grpc::Status::OK;
    }

    grpc::Status Remove(grpc::ServerContext *,
                        const proto::kv_meta::RemoveRequest *,
                        proto::kv_meta::CommonResponse *response) override {
        ++remove_calls;
        if (remove_transport_error_.load()) {
            return grpc::Status(grpc::StatusCode::UNAVAILABLE, "injected ambiguous transport error");
        }
        SetReadyStatus(response);
        return grpc::Status::OK;
    }

    grpc::Status Trim(grpc::ServerContext *,
                      const proto::kv_meta::TrimRequest *,
                      proto::kv_meta::CommonResponse *response) override {
        ++trim_calls;
        if (trim_transport_error_.load()) {
            return grpc::Status(grpc::StatusCode::UNAVAILABLE, "injected ambiguous transport error");
        }
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
    std::atomic<int> get_calls{0};
    std::atomic<int> put_start_calls{0};
    std::atomic<int> put_finish_calls{0};
    std::atomic<int> remove_calls{0};
    std::atomic<int> trim_calls{0};
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
    std::atomic<bool> write_in_progress_{false};
    std::atomic<bool> wrong_uri_size_{false};
    std::atomic<bool> wrong_uri_scheme_{false};
    std::atomic<bool> non_singleton_uri_{false};
    std::atomic<bool> oversized_session_id_{false};
    std::atomic<bool> omit_last_start_location_{false};
    std::atomic<bool> extra_start_mask_value_{false};
    std::atomic<bool> register_transport_error_{false};
    std::atomic<bool> get_transport_error_{false};
    std::atomic<bool> put_start_transport_error_{false};
    std::atomic<bool> put_finish_transport_error_{false};
    std::atomic<bool> remove_transport_error_{false};
    std::atomic<bool> trim_transport_error_{false};
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

    auto [start_ec, start] = client->StartWrite("trace-start", {"exists", "a", "b"}, {1, 17, 33}, 30);
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

    auto client = KvMetaClient::Create({{standby_server.address(), leader_server.address()}, "emb-instance", 1000});
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

TEST(KvMetaClientTest, MapsWriteInProgressWithoutRetryingAnotherServer) {
    FakeKvMetaService active_writer;
    active_writer.set_write_in_progress(true);
    FakeKvMetaService fallback;
    RunningServer active_writer_server(&active_writer);
    RunningServer fallback_server(&fallback);
    ASSERT_TRUE(active_writer_server.valid());
    ASSERT_TRUE(fallback_server.valid());

    auto client =
        KvMetaClient::Create({{active_writer_server.address(), fallback_server.address()}, "emb-instance", 1000});
    ASSERT_TRUE(client);
    const auto [start_ec, start] = client->StartWrite("trace-start", {"a"}, {17}, 30);

    EXPECT_EQ(ER_SERVICE_WRITE_IN_PROGRESS, start_ec);
    EXPECT_TRUE(start.write_session_id.empty());
    EXPECT_TRUE(start.key_mask.empty());
    EXPECT_TRUE(start.locations.empty());
    EXPECT_EQ(1, active_writer.put_start_calls.load());
    EXPECT_EQ(0, active_writer.put_finish_calls.load());
    EXPECT_EQ(0, fallback.put_start_calls.load());
}

TEST(KvMetaClientTest, ReadFailsOverAfterTransportError) {
    FakeKvMetaService unavailable;
    unavailable.set_get_transport_error(true);
    FakeKvMetaService leader;
    RunningServer unavailable_server(&unavailable);
    RunningServer leader_server(&leader);
    ASSERT_TRUE(unavailable_server.valid());
    ASSERT_TRUE(leader_server.valid());

    auto client = KvMetaClient::Create({{unavailable_server.address(), leader_server.address()}, "emb-instance", 1000});
    ASSERT_TRUE(client);
    auto [get_ec, get] = client->Get("trace-get", {"a"});
    ASSERT_EQ(ER_OK, get_ec);
    ASSERT_EQ(1, get.locations.size());
    EXPECT_EQ(17, get.locations[0].value_size);
    EXPECT_EQ(1, unavailable.get_calls.load());
    EXPECT_EQ(1, leader.get_calls.load());
}

TEST(KvMetaClientTest, IdempotentRegistrationFailsOverAfterTransportError) {
    FakeKvMetaService unavailable;
    unavailable.set_register_transport_error(true);
    FakeKvMetaService leader;
    RunningServer unavailable_server(&unavailable);
    RunningServer leader_server(&leader);
    ASSERT_TRUE(unavailable_server.valid());
    ASSERT_TRUE(leader_server.valid());

    auto client = KvMetaClient::Create({{unavailable_server.address(), leader_server.address()}, "emb-instance", 1000});
    ASSERT_TRUE(client);
    auto [register_ec, storage_config] = client->RegisterInstance("trace-register", "objects", "emb");
    EXPECT_EQ(ER_OK, register_ec);
    EXPECT_EQ("{}", storage_config);
    EXPECT_EQ(1, unavailable.register_calls.load());
    EXPECT_EQ(1, leader.register_calls.load());
}

TEST(KvMetaClientTest, AmbiguousWriteTransportErrorsAreNotRetried) {
    FakeKvMetaService ambiguous;
    ambiguous.set_put_start_transport_error(true);
    ambiguous.set_put_finish_transport_error(true);
    ambiguous.set_remove_transport_error(true);
    ambiguous.set_trim_transport_error(true);
    FakeKvMetaService fallback;
    RunningServer ambiguous_server(&ambiguous);
    RunningServer fallback_server(&fallback);
    ASSERT_TRUE(ambiguous_server.valid());
    ASSERT_TRUE(fallback_server.valid());

    auto client = KvMetaClient::Create({{ambiguous_server.address(), fallback_server.address()}, "emb-instance", 1000});
    ASSERT_TRUE(client);

    EXPECT_EQ(ER_INVALID_GRPCSTATUS, client->StartWrite("trace-start", {"a"}, {17}, 30).first);
    EXPECT_EQ(1, ambiguous.put_start_calls.load());
    EXPECT_EQ(0, fallback.put_start_calls.load());

    EXPECT_EQ(ER_INVALID_GRPCSTATUS, client->FinishWrite("trace-finish", "possibly-committed-session", {true}));
    EXPECT_EQ(1, ambiguous.put_finish_calls.load());
    EXPECT_EQ(0, fallback.put_finish_calls.load());

    EXPECT_EQ(ER_INVALID_GRPCSTATUS, client->Remove("trace-remove", {"a"}));
    EXPECT_EQ(1, ambiguous.remove_calls.load());
    EXPECT_EQ(0, fallback.remove_calls.load());

    EXPECT_EQ(ER_INVALID_GRPCSTATUS, client->TrimAll("trace-trim"));
    EXPECT_EQ(1, ambiguous.trim_calls.load());
    EXPECT_EQ(0, fallback.trim_calls.load());
}

TEST(KvMetaClientTest, ExplicitStandbyResponsesStillFailOverForWrites) {
    FakeKvMetaService standby(true);
    FakeKvMetaService leader;
    RunningServer standby_server(&standby);
    RunningServer leader_server(&leader);
    ASSERT_TRUE(standby_server.valid());
    ASSERT_TRUE(leader_server.valid());

    const KvMetaClientConfig config{{standby_server.address(), leader_server.address()}, "emb-instance", 1000};
    auto start_client = KvMetaClient::Create(config);
    ASSERT_TRUE(start_client);
    auto [start_ec, start] = start_client->StartWrite("trace-start", {"a"}, {17}, 30);
    ASSERT_EQ(ER_OK, start_ec);
    EXPECT_EQ(1, standby.put_start_calls.load());
    EXPECT_EQ(1, leader.put_start_calls.load());
    EXPECT_EQ(ER_OK, start_client->FinishWrite("trace-cleanup", start.write_session_id, {false}));

    auto finish_client = KvMetaClient::Create(config);
    ASSERT_TRUE(finish_client);
    EXPECT_EQ(ER_OK, finish_client->FinishWrite("trace-finish", "standby-redirect-session", {true}));
    EXPECT_EQ(1, standby.put_finish_calls.load());
    EXPECT_EQ(2, leader.put_finish_calls.load());

    auto remove_client = KvMetaClient::Create(config);
    ASSERT_TRUE(remove_client);
    EXPECT_EQ(ER_OK, remove_client->Remove("trace-remove", {"a"}));
    EXPECT_EQ(1, standby.remove_calls.load());
    EXPECT_EQ(1, leader.remove_calls.load());

    auto trim_client = KvMetaClient::Create(config);
    ASSERT_TRUE(trim_client);
    EXPECT_EQ(ER_OK, trim_client->TrimAll("trace-trim"));
    EXPECT_EQ(1, standby.trim_calls.load());
    EXPECT_EQ(1, leader.trim_calls.load());
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
    EXPECT_EQ(ER_SERVICE_INTERNAL_ERROR, client->StartWrite("trace-size", {"a"}, {17}, 30).first);
    EXPECT_EQ(1, service.put_finish_calls.load());
    EXPECT_EQ((std::vector<bool>{false}), service.FinishSuccesses());

    service.set_wrong_uri_size(false);
    service.set_wrong_uri_scheme(true);
    EXPECT_EQ(ER_SERVICE_INTERNAL_ERROR, client->StartWrite("trace-scheme", {"a"}, {17}, 30).first);
    EXPECT_EQ(2, service.put_finish_calls.load());
    EXPECT_EQ((std::vector<bool>{false}), service.FinishSuccesses());

    service.set_wrong_uri_scheme(false);
    service.set_non_singleton_uri(true);
    EXPECT_EQ(ER_SERVICE_INTERNAL_ERROR, client->StartWrite("trace-non-singleton", {"a"}, {17}, 30).first);
    EXPECT_EQ(3, service.put_finish_calls.load());
    EXPECT_EQ((std::vector<bool>{false}), service.FinishSuccesses());
}

TEST(KvMetaClientTest, OversizedSessionIdInStartResponseIsRejected) {
    FakeKvMetaService service;
    service.set_oversized_session_id(true);
    RunningServer server(&service);
    ASSERT_TRUE(server.valid());

    auto client = KvMetaClient::Create({{server.address()}, "emb-instance", 1000});
    ASSERT_TRUE(client);
    const auto [start_ec, start] = client->StartWrite("trace-start", {"a"}, {17}, 30);

    EXPECT_EQ(ER_SERVICE_INTERNAL_ERROR, start_ec);
    EXPECT_TRUE(start.write_session_id.empty());
    EXPECT_TRUE(start.key_mask.empty());
    EXPECT_TRUE(start.locations.empty());
    // PutFinish also rejects an oversized id, so the client must not issue an
    // invalid rollback RPC. A malformed server response is reclaimed by the
    // server-side write-session timeout.
    EXPECT_EQ(0, service.put_finish_calls.load());
}

TEST(KvMetaClientTest, RejectsServiceLimitViolationsBeforeRpc) {
    FakeKvMetaService service;
    RunningServer server(&service);
    ASSERT_TRUE(server.valid());

    EXPECT_EQ(nullptr, KvMetaClient::Create({{server.address()}, std::string(513, 'i'), 1000}));

    auto client = KvMetaClient::Create({{server.address()}, "emb-instance", 1000});
    ASSERT_TRUE(client);
    EXPECT_EQ(ER_INVALID_PARAMS, client->RegisterInstance("trace", std::string(513, 'g'), "").first);
    EXPECT_EQ(ER_INVALID_PARAMS, client->RegisterInstance("trace", "objects", std::string(64 * 1024 + 1, 'u')).first);
    EXPECT_EQ(0, service.register_calls.load());

    std::vector<std::string> too_many_keys;
    for (std::size_t i = 0; i < 65; ++i) {
        too_many_keys.push_back("key-" + std::to_string(i));
    }
    EXPECT_EQ(ER_INVALID_PARAMS, client->Get("trace", too_many_keys).first);
    EXPECT_EQ(ER_INVALID_PARAMS, client->Get("trace", {std::string(513, 'k')}).first);
    EXPECT_EQ(0, service.get_calls.load());

    EXPECT_EQ(ER_INVALID_PARAMS, client->Remove("trace", too_many_keys));
    EXPECT_EQ(ER_INVALID_PARAMS, client->Remove("trace", {std::string(513, 'k')}));
    EXPECT_EQ(ER_INVALID_PARAMS, client->Remove("trace", {"duplicate", "duplicate"}));
    EXPECT_EQ(0, service.remove_calls.load());

    EXPECT_EQ(ER_INVALID_PARAMS, client->StartWrite("trace", {"key"}, {1ULL * 1024 * 1024 * 1024 + 1}, 30).first);
    EXPECT_EQ(ER_INVALID_PARAMS, client->StartWrite("trace", {"key"}, {1}, 1801).first);
    EXPECT_EQ(ER_INVALID_PARAMS,
              client
                  ->StartWrite(
                      "trace", {"a", "b", "c", "d", "e"}, std::vector<std::uint64_t>(5, 1ULL * 1024 * 1024 * 1024), 30)
                  .first);
    EXPECT_EQ(0, service.put_start_calls.load());

    EXPECT_EQ(ER_INVALID_PARAMS, client->FinishWrite("trace", "session", std::vector<bool>(65, true)));
    EXPECT_EQ(ER_INVALID_PARAMS, client->FinishWrite("trace", std::string(513, 's'), {true}));
    EXPECT_EQ(0, service.put_finish_calls.load());
}

} // namespace
} // namespace kv_cache_manager
