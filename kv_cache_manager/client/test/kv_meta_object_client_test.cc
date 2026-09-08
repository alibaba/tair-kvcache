#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "kv_cache_manager/client/src/kv_meta_object_client_impl.h"
#include "kv_cache_manager/common/unittest.h"

namespace kv_cache_manager {
namespace {

KvMetaValueLocation MakeLocation(const std::string &uri, std::uint64_t size) {
    KvMetaValueLocation location;
    location.type = KvMetaStorageType::NFS;
    location.value_size = size;
    location.location_specs.push_back({"value", uri});
    return location;
}

BlockBuffer MakeBuffer(void *base, std::size_t size) {
    BlockBuffer buffer;
    buffer.iovs.push_back({MemoryType::CPU, base, size, false});
    return buffer;
}

class FakeKvMetaClient final : public KvMetaClient {
public:
    std::pair<ClientErrorCode, std::string>
    RegisterInstance(const std::string &, const std::string &, const std::string &) override {
        return {ER_OK, "[]"};
    }

    std::pair<ClientErrorCode, KvMetaInstanceInfo> GetInstanceInfo(const std::string &) override { return {ER_OK, {}}; }

    std::pair<ClientErrorCode, KvMetaGetResult> Get(const std::string &, const std::vector<std::string> &) override {
        ++get_calls;
        return {get_ec, get_result};
    }

    std::pair<ClientErrorCode, KvMetaStartWriteResult> StartWrite(const std::string &,
                                                                  const std::vector<std::string> &,
                                                                  const std::vector<std::uint64_t> &,
                                                                  std::int32_t) override {
        ++start_calls;
        return {start_ec, start_result};
    }

    ClientErrorCode FinishWrite(const std::string &,
                                const std::string &write_session_id,
                                const std::vector<bool> &success_keys) override {
        ++finish_calls;
        finished_session = write_session_id;
        finished_keys = success_keys;
        return finish_ec;
    }

    ClientErrorCode Remove(const std::string &, const std::vector<std::string> &keys) override {
        removed_keys = keys;
        return remove_ec;
    }

    ClientErrorCode TrimAll(const std::string &, bool) override { return ER_OK; }

    ClientErrorCode get_ec{ER_OK};
    ClientErrorCode start_ec{ER_OK};
    ClientErrorCode finish_ec{ER_OK};
    ClientErrorCode remove_ec{ER_OK};
    KvMetaGetResult get_result;
    KvMetaStartWriteResult start_result;
    int get_calls{0};
    int start_calls{0};
    int finish_calls{0};
    std::string finished_session;
    std::vector<bool> finished_keys;
    std::vector<std::string> removed_keys;
};

class FakeKvMetaTransferClient final : public KvMetaTransferClient {
public:
    ClientErrorCode LoadObjects(const UriStrVec &uris,
                                const std::vector<std::uint64_t> &value_sizes,
                                const BlockBuffers &buffers) override {
        ++load_calls;
        loaded_uris = uris;
        loaded_sizes = value_sizes;
        loaded_buffer_count = buffers.size();
        return load_ec;
    }

    std::pair<ClientErrorCode, UriStrVec> SaveObjects(const UriStrVec &uris,
                                                      const std::vector<std::uint64_t> &value_sizes,
                                                      const BlockBuffers &buffers) override {
        ++save_calls;
        saved_uris = uris;
        saved_sizes = value_sizes;
        saved_buffer_count = buffers.size();
        return {save_ec, actual_uris.empty() ? uris : actual_uris};
    }

    ClientErrorCode load_ec{ER_OK};
    ClientErrorCode save_ec{ER_OK};
    UriStrVec actual_uris;
    int load_calls{0};
    int save_calls{0};
    UriStrVec loaded_uris;
    UriStrVec saved_uris;
    std::vector<std::uint64_t> loaded_sizes;
    std::vector<std::uint64_t> saved_sizes;
    std::size_t loaded_buffer_count{0};
    std::size_t saved_buffer_count{0};
};

class KvMetaObjectClientTest : public TESTBASE {
protected:
    void SetUp() override {
        auto metadata = std::make_unique<FakeKvMetaClient>();
        metadata_ = metadata.get();
        auto transfer = std::make_unique<FakeKvMetaTransferClient>();
        transfer_ = transfer.get();
        client_ = std::make_unique<KvMetaObjectClientImpl>(std::move(metadata), std::move(transfer), 1024, 30);
        buffers_ = {MakeBuffer(first_, sizeof(first_)), MakeBuffer(second_, sizeof(second_))};
        sizes_ = {sizeof(first_), sizeof(second_)};
        keys_ = {"first", "second"};
    }

    char first_[5]{1, 2, 3, 4, 5};
    char second_[9]{9, 8, 7, 6, 5, 4, 3, 2, 1};
    std::vector<std::string> keys_;
    std::vector<std::uint64_t> sizes_;
    BlockBuffers buffers_;
    FakeKvMetaClient *metadata_{nullptr};
    FakeKvMetaTransferClient *transfer_{nullptr};
    std::unique_ptr<KvMetaObjectClientImpl> client_;
};

TEST_F(KvMetaObjectClientTest, SavesOnlyMissingObjectsAndCommits) {
    metadata_->start_result.write_session_id = "session";
    metadata_->start_result.key_mask = {true, false};
    metadata_->start_result.locations = {
        MakeLocation("file://nfs/object?size=9", sizeof(second_)),
    };
    metadata_->get_result.hit_mask = {true};
    metadata_->get_result.locations = {
        MakeLocation("file://nfs/existing?size=5", sizeof(first_)),
    };

    EXPECT_EQ(ER_OK, client_->SaveObjects("trace", keys_, sizes_, buffers_));
    EXPECT_EQ(1, metadata_->get_calls);
    EXPECT_EQ(1, transfer_->save_calls);
    EXPECT_EQ((std::vector<std::uint64_t>{sizeof(second_)}), transfer_->saved_sizes);
    EXPECT_EQ(1U, transfer_->saved_buffer_count);
    EXPECT_EQ(1, metadata_->finish_calls);
    EXPECT_EQ("session", metadata_->finished_session);
    EXPECT_EQ((std::vector<bool>{true}), metadata_->finished_keys);
}

TEST_F(KvMetaObjectClientTest, RejectsLegacyInflightMaskAndRollsBackOwnedMisses) {
    metadata_->start_result.write_session_id = "session";
    metadata_->start_result.key_mask = {true, false};
    metadata_->start_result.locations = {
        MakeLocation("file://nfs/object?size=9", sizeof(second_)),
    };
    // Old servers could mask an active reservation even though Get correctly
    // kept it invisible. The object client must not turn that state into a
    // successful save.
    metadata_->get_result.hit_mask = {false};
    metadata_->get_result.locations = {KvMetaValueLocation{}};

    EXPECT_EQ(ER_SERVICE_WRITE_IN_PROGRESS, client_->SaveObjects("trace", keys_, sizes_, buffers_));
    EXPECT_EQ(1, metadata_->get_calls);
    EXPECT_EQ(0, transfer_->save_calls);
    ASSERT_EQ(1, metadata_->finish_calls);
    EXPECT_EQ("session", metadata_->finished_session);
    EXPECT_EQ((std::vector<bool>{false}), metadata_->finished_keys);
}

TEST_F(KvMetaObjectClientTest, VerifiesAllCommittedHitsWithoutStartingDataIo) {
    metadata_->start_result.key_mask = {true, true};
    metadata_->get_result.hit_mask = {true, true};
    metadata_->get_result.locations = {
        MakeLocation("file://nfs/first?size=5", sizeof(first_)),
        MakeLocation("file://nfs/second?size=9", sizeof(second_)),
    };

    EXPECT_EQ(ER_OK, client_->SaveObjects("trace", keys_, sizes_, buffers_));
    EXPECT_EQ(1, metadata_->get_calls);
    EXPECT_EQ(0, transfer_->save_calls);
    EXPECT_EQ(0, metadata_->finish_calls);
}

TEST_F(KvMetaObjectClientTest, AbortsMalformedAllHitResponseWithAllocation) {
    metadata_->start_result.write_session_id = "unexpected-session";
    metadata_->start_result.key_mask = {true, true};
    metadata_->start_result.locations = {
        MakeLocation("file://nfs/unexpected?size=5", sizeof(first_)),
    };

    EXPECT_EQ(ER_SERVICE_INTERNAL_ERROR, client_->SaveObjects("trace", keys_, sizes_, buffers_));
    EXPECT_EQ(0, transfer_->save_calls);
    EXPECT_EQ(1, metadata_->finish_calls);
    EXPECT_EQ("unexpected-session", metadata_->finished_session);
    EXPECT_EQ((std::vector<bool>{false}), metadata_->finished_keys);
}

TEST_F(KvMetaObjectClientTest, UsesMissingMaskCountToAbortMalformedLocationResponse) {
    metadata_->start_result.write_session_id = "session";
    metadata_->start_result.key_mask = {true, false};
    // The mask says that the session owns one value, but a malformed service
    // response carries two locations. Abort must use the mask-derived session
    // size or PutFinish would reject the rollback and leave it active until
    // timeout.
    metadata_->start_result.locations = {
        MakeLocation("file://nfs/second?size=9", sizeof(second_)),
        MakeLocation("file://nfs/unexpected?size=5", sizeof(first_)),
    };

    EXPECT_EQ(ER_SERVICE_INTERNAL_ERROR, client_->SaveObjects("trace", keys_, sizes_, buffers_));
    EXPECT_EQ(0, transfer_->save_calls);
    EXPECT_EQ(1, metadata_->finish_calls);
    EXPECT_EQ("session", metadata_->finished_session);
    EXPECT_EQ((std::vector<bool>{false}), metadata_->finished_keys);
}

TEST_F(KvMetaObjectClientTest, AbortsWholeSessionWhenTransferFails) {
    metadata_->start_result.write_session_id = "session";
    metadata_->start_result.key_mask = {false, false};
    metadata_->start_result.locations = {
        MakeLocation("file://nfs/first?size=5", sizeof(first_)),
        MakeLocation("file://nfs/second?size=9", sizeof(second_)),
    };
    transfer_->save_ec = ER_SDKWRITE_ERROR;

    EXPECT_EQ(ER_SDKWRITE_ERROR, client_->SaveObjects("trace", keys_, sizes_, buffers_));
    EXPECT_EQ(1, metadata_->finish_calls);
    EXPECT_EQ((std::vector<bool>{false, false}), metadata_->finished_keys);
}

TEST_F(KvMetaObjectClientTest, ReturnsRollbackErrorWhenAbortOutcomeIsUnknown) {
    metadata_->start_result.write_session_id = "session";
    metadata_->start_result.key_mask = {false, false};
    metadata_->start_result.locations = {
        MakeLocation("file://nfs/first?size=5", sizeof(first_)),
        MakeLocation("file://nfs/second?size=9", sizeof(second_)),
    };
    metadata_->finish_ec = ER_INVALID_GRPCSTATUS;
    transfer_->save_ec = ER_SDKWRITE_ERROR;

    EXPECT_EQ(ER_INVALID_GRPCSTATUS, client_->SaveObjects("trace", keys_, sizes_, buffers_));
}

TEST_F(KvMetaObjectClientTest, RejectsBadBufferBeforeMetadataMutation) {
    buffers_[0].iovs[0].size -= 1;

    EXPECT_EQ(ER_INVALID_LOCAL_BUFFERS, client_->SaveObjects("trace", keys_, sizes_, buffers_));
    EXPECT_EQ(0, metadata_->start_calls);
    EXPECT_EQ(0, transfer_->save_calls);
}

TEST_F(KvMetaObjectClientTest, LoadsOnlyAfterEveryKeyAndSizeMatches) {
    metadata_->get_result.hit_mask = {true, true};
    metadata_->get_result.locations = {
        MakeLocation("file://nfs/first?size=5", sizeof(first_)),
        MakeLocation("file://nfs/second?size=9", sizeof(second_)),
    };

    EXPECT_EQ(ER_OK, client_->LoadObjects("trace", keys_, sizes_, buffers_));
    EXPECT_EQ(1, transfer_->load_calls);
    EXPECT_EQ(sizes_, transfer_->loaded_sizes);
    EXPECT_EQ(2U, transfer_->loaded_buffer_count);
}

TEST_F(KvMetaObjectClientTest, DoesNotReadDataForMetadataMiss) {
    metadata_->get_result.hit_mask = {true, false};
    metadata_->get_result.locations = {
        MakeLocation("file://nfs/first?size=5", sizeof(first_)),
        {},
    };

    EXPECT_EQ(ER_SERVICE_NOT_FOUND, client_->LoadObjects("trace", keys_, sizes_, buffers_));
    EXPECT_EQ(0, transfer_->load_calls);
}

TEST_F(KvMetaObjectClientTest, DoesNotReadDataForMetadataSizeMismatch) {
    metadata_->get_result.hit_mask = {true, true};
    metadata_->get_result.locations = {
        MakeLocation("file://nfs/first?size=5", sizeof(first_)),
        MakeLocation("file://nfs/second?size=8", sizeof(second_) - 1),
    };

    EXPECT_EQ(ER_SERVICE_SIZE_MISMATCH, client_->LoadObjects("trace", keys_, sizes_, buffers_));
    EXPECT_EQ(0, transfer_->load_calls);
}

TEST_F(KvMetaObjectClientTest, RejectsServiceLimitViolationsBeforeMetadataCalls) {
    char byte = 0;
    std::vector<std::string> keys;
    std::vector<std::uint64_t> sizes;
    BlockBuffers buffers;
    for (std::size_t i = 0; i < 65; ++i) {
        keys.push_back("key-" + std::to_string(i));
        sizes.push_back(1);
        buffers.push_back(MakeBuffer(&byte, 1));
    }
    EXPECT_EQ(ER_INVALID_PARAMS, client_->SaveObjects("trace", keys, sizes, buffers));
    EXPECT_EQ(0, metadata_->start_calls);

    keys = {std::string(513, 'k')};
    sizes = {1};
    buffers = {MakeBuffer(&byte, 1)};
    EXPECT_EQ(ER_INVALID_PARAMS, client_->LoadObjects("trace", keys, sizes, buffers));
    EXPECT_EQ(0, metadata_->get_calls);
}

TEST_F(KvMetaObjectClientTest, RejectsMalformedObjectVectorsAndIovsBeforeMetadataCalls) {
    const auto expect_invalid_params = [&](const std::vector<std::string> &keys,
                                           const std::vector<std::uint64_t> &sizes,
                                           const BlockBuffers &buffers) {
        EXPECT_EQ(ER_INVALID_PARAMS, client_->SaveObjects("trace", keys, sizes, buffers));
    };
    const auto expect_invalid_buffers = [&](const BlockBuffers &buffers) {
        EXPECT_EQ(ER_INVALID_LOCAL_BUFFERS, client_->SaveObjects("trace", keys_, sizes_, buffers));
    };

    expect_invalid_params({}, {}, {});
    expect_invalid_params(keys_, {sizeof(first_)}, buffers_);
    expect_invalid_params(keys_, sizes_, {buffers_[0]});
    expect_invalid_params({"duplicate", "duplicate"}, sizes_, buffers_);
    expect_invalid_params(keys_, {0, sizeof(second_)}, buffers_);
    expect_invalid_params(keys_, {1025, sizeof(second_)}, buffers_);

    BlockBuffers malformed = buffers_;
    malformed[0].iovs.clear();
    expect_invalid_params(keys_, sizes_, malformed);

    malformed = buffers_;
    malformed[0].iovs[0].ignore = true;
    expect_invalid_buffers(malformed);

    malformed = buffers_;
    malformed[0].iovs[0].size = 0;
    expect_invalid_buffers(malformed);

    malformed = buffers_;
    malformed[0].iovs[0].base = nullptr;
    expect_invalid_buffers(malformed);

    malformed = buffers_;
    malformed[0].iovs[0].type = static_cast<MemoryType>(999);
    expect_invalid_buffers(malformed);

    malformed = buffers_;
    malformed[0].iovs = {
        {MemoryType::CPU, first_, 3, false},
        {MemoryType::CPU, first_ + 3, 3, false},
    };
    expect_invalid_buffers(malformed);

    EXPECT_EQ(0, metadata_->start_calls);
    EXPECT_EQ(0, metadata_->get_calls);
    EXPECT_EQ(0, transfer_->save_calls);
    EXPECT_EQ(0, transfer_->load_calls);
}

TEST(KvMetaObjectClientLimitTest, AcceptsExactBatchByteLimitAndRejectsTheNextObjectBeforeMetadata) {
    constexpr std::uint64_t kOneGiB = 1ULL * 1024 * 1024 * 1024;
    char byte = 0;
    auto metadata = std::make_unique<FakeKvMetaClient>();
    auto *metadata_ptr = metadata.get();
    metadata_ptr->start_ec = ER_SERVICE_NOT_READY;
    auto transfer = std::make_unique<FakeKvMetaTransferClient>();
    auto *transfer_ptr = transfer.get();
    KvMetaObjectClientImpl client(std::move(metadata), std::move(transfer), kOneGiB, 30);

    std::vector<std::string> keys;
    std::vector<std::uint64_t> sizes;
    BlockBuffers buffers;
    for (std::size_t i = 0; i < 4; ++i) {
        keys.push_back("key-" + std::to_string(i));
        sizes.push_back(kOneGiB);
        buffers.push_back(MakeBuffer(&byte, kOneGiB));
    }

    EXPECT_EQ(ER_SERVICE_NOT_READY, client.SaveObjects("trace", keys, sizes, buffers));
    EXPECT_EQ(1, metadata_ptr->start_calls);

    keys.push_back("key-4");
    sizes.push_back(kOneGiB);
    buffers.push_back(MakeBuffer(&byte, kOneGiB));
    EXPECT_EQ(ER_INVALID_PARAMS, client.SaveObjects("trace", keys, sizes, buffers));
    EXPECT_EQ(1, metadata_ptr->start_calls);
    EXPECT_EQ(0, transfer_ptr->save_calls);
}

TEST(KvMetaObjectClientCreateTest, RejectsGroupIdentityMismatchBeforeMetadataRegistration) {
    KvMetaObjectClientConfig config;
    config.metadata.instance_id = "metadata-instance";
    // Deliberately leave addresses empty. If static transfer validation were
    // not performed first, Create would instead fail while initializing the
    // metadata client.
    config.instance_group = "metadata-group";
    config.transfer_client_config = R"({
        "instance_group": "different-group",
        "instance_id": "metadata-instance",
        "block_size": 1,
        "location_spec_infos": {"value": 1}
    })";
    config.transfer_init_params.role_type = RoleType::WORKER;
    config.transfer_init_params.self_location_spec_name = "value";

    auto [ec, client] = KvMetaObjectClient::Create("trace", config);
    EXPECT_EQ(ER_INVALID_CLIENT_CONFIG, ec);
    EXPECT_EQ(nullptr, client);
}

TEST(KvMetaObjectClientCreateTest, RejectsInstanceIdentityMismatchBeforeMetadataRegistration) {
    KvMetaObjectClientConfig config;
    config.metadata.instance_id = "metadata-instance";
    // Deliberately leave addresses empty for the same ordering assertion as
    // the group mismatch case above.
    config.instance_group = "metadata-group";
    config.transfer_client_config = R"({
        "instance_group": "metadata-group",
        "instance_id": "different-instance",
        "block_size": 1,
        "location_spec_infos": {"value": 1}
    })";
    config.transfer_init_params.role_type = RoleType::WORKER;
    config.transfer_init_params.self_location_spec_name = "value";

    auto [ec, client] = KvMetaObjectClient::Create("trace", config);
    EXPECT_EQ(ER_INVALID_CLIENT_CONFIG, ec);
    EXPECT_EQ(nullptr, client);
}

TEST(KvMetaObjectClientCreateTest, RejectsMissingSdkConfigBeforeMetadataRegistration) {
    KvMetaObjectClientConfig config;
    config.metadata.instance_id = "metadata-instance";
    config.instance_group = "metadata-group";
    config.transfer_client_config = R"({
        "instance_group": "metadata-group",
        "instance_id": "metadata-instance",
        "block_size": 1,
        "location_spec_infos": {"value": 1}
    })";
    config.transfer_init_params.role_type = RoleType::WORKER;
    config.transfer_init_params.self_location_spec_name = "value";

    auto [ec, client] = KvMetaObjectClient::Create("trace", config);
    EXPECT_EQ(ER_INVALID_SDKWRAPPER_CONFIG, ec);
    EXPECT_EQ(nullptr, client);
}

TEST(KvMetaObjectClientCreateTest, RejectsUndersizedSdkQueueBeforeMetadataRegistration) {
    KvMetaObjectClientConfig config;
    config.metadata.instance_id = "metadata-instance";
    // Addresses intentionally remain empty. Exact-object static validation
    // must reject a queue that cannot admit one full service batch before
    // metadata client setup or remote RegisterInstance can run.
    config.instance_group = "metadata-group";
    config.transfer_client_config = R"({
        "instance_group": "metadata-group",
        "instance_id": "metadata-instance",
        "block_size": 1,
        "sdk_config": {
            "thread_num": 2,
            "queue_size": 63,
            "sdk_backend_configs": [],
            "timeout_config": {
                "get_timeout_ms": 10000,
                "put_timeout_ms": 10000
            }
        },
        "location_spec_infos": {"value": 1}
    })";
    config.transfer_init_params.role_type = RoleType::WORKER;
    config.transfer_init_params.self_location_spec_name = "value";

    auto [ec, client] = KvMetaObjectClient::Create("trace", config);
    EXPECT_EQ(ER_INVALID_SDKWRAPPER_CONFIG, ec);
    EXPECT_EQ(nullptr, client);
}

TEST(KvMetaObjectClientCreateTest, RejectsWriteLeaseThatCannotCoverStartDataAndCommitBudgets) {
    KvMetaObjectClientConfig config;
    config.metadata.instance_id = "metadata-instance";
    config.metadata.call_timeout_ms = 3000;
    // Addresses intentionally remain empty. The timeout relationship must be
    // rejected by static data-plane validation before metadata client setup or
    // remote RegisterInstance can run.
    config.instance_group = "metadata-group";
    config.transfer_client_config = R"({
        "instance_group": "metadata-group",
        "instance_id": "metadata-instance",
        "block_size": 1,
        "sdk_config": {
            "thread_num": 2,
            "queue_size": 64,
            "sdk_backend_configs": [],
            "timeout_config": {
                "get_timeout_ms": 10000,
                "put_timeout_ms": 10000
            }
        },
        "location_spec_infos": {"value": 1}
    })";
    config.transfer_init_params.role_type = RoleType::WORKER;
    config.transfer_init_params.self_location_spec_name = "value";
    // 10s data I/O + three 3s metadata windows exactly consumes 19s. The
    // relationship is strict so the session cannot expire on the boundary.
    config.write_timeout_seconds = 19;

    auto [ec, client] = KvMetaObjectClient::Create("trace", config);
    EXPECT_EQ(ER_INVALID_CLIENT_CONFIG, ec);
    EXPECT_EQ(nullptr, client);

    // One second of lease headroom passes static transfer validation. Empty
    // metadata addresses then fail at the next stage, proving the boundary is
    // not over-rejected and no registration RPC was attempted.
    config.write_timeout_seconds = 20;
    auto [valid_ec, valid_client] = KvMetaObjectClient::Create("trace", config);
    EXPECT_EQ(ER_METACLIENT_INIT_ERROR, valid_ec);
    EXPECT_EQ(nullptr, valid_client);
}

} // namespace
} // namespace kv_cache_manager
