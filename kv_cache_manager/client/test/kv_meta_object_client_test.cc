#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>

#include "kv_cache_manager/client/src/kv_meta_object_client_impl.h"
#include "kv_cache_manager/common/unittest.h"

namespace kv_cache_manager {
namespace {

static_assert(kKvMetaObjectClientApiVersion == 1);

TEST(KvMetaObjectClientVersionTest, SharedLibraryExportsHeaderCapabilityVersion) {
    EXPECT_EQ(GetKvMetaObjectClientApiVersion(), kKvMetaObjectClientApiVersion);
}

enum class ThrowMode { NONE, STANDARD, UNKNOWN };

void ThrowIfRequested(ThrowMode mode) {
    if (mode == ThrowMode::STANDARD) {
        throw std::runtime_error("injected provider detail");
    }
    if (mode == ThrowMode::UNKNOWN) {
        throw 7;
    }
}

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

KvMetaObjectClientConfig MakeStaticallyValidCreateConfig() {
    KvMetaObjectClientConfig config;
    config.metadata.instance_id = "metadata-instance";
    config.metadata.call_timeout_ms = 1;
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
                "get_timeout_ms": 10,
                "put_timeout_ms": 10
            }
        },
        "location_spec_infos": {"value": 1}
    })";
    config.transfer_init_params.role_type = RoleType::WORKER;
    config.transfer_init_params.self_location_spec_name = "value";
    config.write_timeout_seconds = 1;
    return config;
}

class FakeKvMetaClient final : public KvMetaClient {
public:
    std::pair<ClientErrorCode, std::string>
    RegisterInstance(const std::string &, const std::string &, const std::string &) override {
        return {ER_OK, "[]"};
    }

    std::pair<ClientErrorCode, KvMetaInstanceInfo> GetInstanceInfo(const std::string &) override { return {ER_OK, {}}; }

    std::pair<ClientErrorCode, KvMetaGetResult> Get(const std::string &trace_id,
                                                    const std::vector<std::string> &keys) override {
        ++get_calls;
        ThrowIfRequested(get_throw);
        get_trace = trace_id;
        gotten_keys = keys;
        return {get_ec, get_result};
    }

    std::pair<ClientErrorCode, KvMetaStartWriteResult> StartWrite(const std::string &trace_id,
                                                                  const std::vector<std::string> &keys,
                                                                  const std::vector<std::uint64_t> &value_sizes,
                                                                  std::int32_t write_timeout_seconds) override {
        ++start_calls;
        ThrowIfRequested(start_throw);
        start_trace = trace_id;
        started_keys = keys;
        started_sizes = value_sizes;
        started_timeout_seconds = write_timeout_seconds;
        return {start_ec, start_result};
    }

    ClientErrorCode FinishWrite(const std::string &trace_id,
                                const std::string &write_session_id,
                                const std::vector<bool> &success_keys) override {
        ++finish_calls;
        ThrowIfRequested(finish_throw);
        finish_trace = trace_id;
        finished_session = write_session_id;
        finished_keys = success_keys;
        return finish_ec;
    }

    ClientErrorCode Remove(const std::string &trace_id, const std::vector<std::string> &keys) override {
        ++remove_calls;
        ThrowIfRequested(remove_throw);
        remove_trace = trace_id;
        removed_keys = keys;
        return remove_ec;
    }

    ClientErrorCode TrimAll(const std::string &, bool) override { return ER_OK; }

    ClientErrorCode get_ec{ER_OK};
    ClientErrorCode start_ec{ER_OK};
    ClientErrorCode finish_ec{ER_OK};
    ClientErrorCode remove_ec{ER_OK};
    ThrowMode get_throw{ThrowMode::NONE};
    ThrowMode start_throw{ThrowMode::NONE};
    ThrowMode finish_throw{ThrowMode::NONE};
    ThrowMode remove_throw{ThrowMode::NONE};
    KvMetaGetResult get_result;
    KvMetaStartWriteResult start_result;
    int get_calls{0};
    int start_calls{0};
    int finish_calls{0};
    int remove_calls{0};
    std::string get_trace;
    std::string start_trace;
    std::string finish_trace;
    std::string remove_trace;
    std::vector<std::string> gotten_keys;
    std::vector<std::string> started_keys;
    std::vector<std::uint64_t> started_sizes;
    std::int32_t started_timeout_seconds{0};
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
        ThrowIfRequested(load_throw);
        loaded_uris = uris;
        loaded_sizes = value_sizes;
        loaded_buffer_count = buffers.size();
        loaded_bases.clear();
        for (const auto &buffer : buffers) {
            loaded_bases.push_back(buffer.iovs.empty() ? nullptr : buffer.iovs.front().base);
        }
        return load_ec;
    }

    std::pair<ClientErrorCode, UriStrVec> SaveObjects(const UriStrVec &uris,
                                                      const std::vector<std::uint64_t> &value_sizes,
                                                      const BlockBuffers &buffers) override {
        ++save_calls;
        ThrowIfRequested(save_throw);
        saved_uris = uris;
        saved_sizes = value_sizes;
        saved_buffer_count = buffers.size();
        saved_bases.clear();
        for (const auto &buffer : buffers) {
            saved_bases.push_back(buffer.iovs.empty() ? nullptr : buffer.iovs.front().base);
        }
        return {save_ec, actual_uris.empty() ? uris : actual_uris};
    }

    ClientErrorCode load_ec{ER_OK};
    ClientErrorCode save_ec{ER_OK};
    ThrowMode load_throw{ThrowMode::NONE};
    ThrowMode save_throw{ThrowMode::NONE};
    UriStrVec actual_uris;
    int load_calls{0};
    int save_calls{0};
    UriStrVec loaded_uris;
    UriStrVec saved_uris;
    std::vector<std::uint64_t> loaded_sizes;
    std::vector<std::uint64_t> saved_sizes;
    std::vector<void *> loaded_bases;
    std::vector<void *> saved_bases;
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
    EXPECT_EQ("trace", metadata_->start_trace);
    EXPECT_EQ(keys_, metadata_->started_keys);
    EXPECT_EQ(sizes_, metadata_->started_sizes);
    EXPECT_EQ(30, metadata_->started_timeout_seconds);
    EXPECT_EQ(1, metadata_->get_calls);
    EXPECT_EQ("trace", metadata_->get_trace);
    EXPECT_EQ((std::vector<std::string>{"first"}), metadata_->gotten_keys);
    EXPECT_EQ(1, transfer_->save_calls);
    EXPECT_EQ((UriStrVec{"file://nfs/object?size=9"}), transfer_->saved_uris);
    EXPECT_EQ((std::vector<std::uint64_t>{sizeof(second_)}), transfer_->saved_sizes);
    EXPECT_EQ((std::vector<void *>{second_}), transfer_->saved_bases);
    EXPECT_EQ(1U, transfer_->saved_buffer_count);
    EXPECT_EQ(1, metadata_->finish_calls);
    EXPECT_EQ("trace", metadata_->finish_trace);
    EXPECT_EQ("session", metadata_->finished_session);
    EXPECT_EQ((std::vector<bool>{true}), metadata_->finished_keys);
}

TEST_F(KvMetaObjectClientTest, StartFailureDoesNotReadWriteOrFinish) {
    metadata_->start_ec = ER_SERVICE_NOT_READY;

    EXPECT_EQ(ER_SERVICE_NOT_READY, client_->SaveObjects("start-failed", keys_, sizes_, buffers_));

    EXPECT_EQ(1, metadata_->start_calls);
    EXPECT_EQ("start-failed", metadata_->start_trace);
    EXPECT_EQ(0, metadata_->get_calls);
    EXPECT_EQ(0, transfer_->save_calls);
    EXPECT_EQ(0, metadata_->finish_calls);
}

TEST_F(KvMetaObjectClientTest, StartExceptionsBecomeAmbiguousErrorsWithoutDataIo) {
    for (const auto mode : {ThrowMode::STANDARD, ThrowMode::UNKNOWN}) {
        SCOPED_TRACE(static_cast<int>(mode));
        metadata_->start_throw = mode;

        EXPECT_EQ(ER_INVALID_GRPCSTATUS, client_->SaveObjects("start-threw", keys_, sizes_, buffers_));
        EXPECT_EQ(0, metadata_->get_calls);
        EXPECT_EQ(0, transfer_->save_calls);
        EXPECT_EQ(0, metadata_->finish_calls);
    }
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

TEST_F(KvMetaObjectClientTest, ExistingSizeMismatchPreventsMissWriteAndRollsBack) {
    metadata_->start_result.write_session_id = "session";
    metadata_->start_result.key_mask = {true, false};
    metadata_->start_result.locations = {
        MakeLocation("file://nfs/second?size=9", sizeof(second_)),
    };
    metadata_->get_result.hit_mask = {true};
    metadata_->get_result.locations = {
        MakeLocation("file://nfs/first?size=4", sizeof(first_) - 1),
    };

    EXPECT_EQ(ER_SERVICE_SIZE_MISMATCH, client_->SaveObjects("trace", keys_, sizes_, buffers_));
    EXPECT_EQ(1, metadata_->get_calls);
    EXPECT_EQ(0, transfer_->save_calls);
    ASSERT_EQ(1, metadata_->finish_calls);
    EXPECT_EQ((std::vector<bool>{false}), metadata_->finished_keys);
}

TEST_F(KvMetaObjectClientTest, CompatibilityGetExceptionsAbortOwnedMisses) {
    metadata_->start_result.write_session_id = "session";
    metadata_->start_result.key_mask = {true, false};
    metadata_->start_result.locations = {
        MakeLocation("file://nfs/second?size=9", sizeof(second_)),
    };

    for (const auto mode : {ThrowMode::STANDARD, ThrowMode::UNKNOWN}) {
        SCOPED_TRACE(static_cast<int>(mode));
        metadata_->get_throw = mode;

        EXPECT_EQ(ER_SERVICE_INTERNAL_ERROR, client_->SaveObjects("trace", keys_, sizes_, buffers_));
        EXPECT_EQ(0, transfer_->save_calls);
        EXPECT_EQ((std::vector<bool>{false}), metadata_->finished_keys);
    }
    EXPECT_EQ(2, metadata_->get_calls);
    EXPECT_EQ(2, metadata_->finish_calls);
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

TEST_F(KvMetaObjectClientTest, DataPlaneSaveExceptionsAreContainedAndAbortTheSession) {
    metadata_->start_result.write_session_id = "session";
    metadata_->start_result.key_mask = {false, false};
    metadata_->start_result.locations = {
        MakeLocation("file://nfs/first?size=5", sizeof(first_)),
        MakeLocation("file://nfs/second?size=9", sizeof(second_)),
    };

    for (const auto mode : {ThrowMode::STANDARD, ThrowMode::UNKNOWN}) {
        SCOPED_TRACE(static_cast<int>(mode));
        transfer_->save_throw = mode;

        EXPECT_EQ(ER_SDKWRITE_ERROR, client_->SaveObjects("trace", keys_, sizes_, buffers_));
        EXPECT_EQ((std::vector<bool>{false, false}), metadata_->finished_keys);
    }
    EXPECT_EQ(2, transfer_->save_calls);
    EXPECT_EQ(2, metadata_->finish_calls);
}

TEST_F(KvMetaObjectClientTest, AbortsWholeSessionWhenBackendRewritesAnyUri) {
    metadata_->start_result.write_session_id = "session";
    metadata_->start_result.key_mask = {false, false};
    metadata_->start_result.locations = {
        MakeLocation("file://nfs/first?size=5", sizeof(first_)),
        MakeLocation("file://nfs/second?size=9", sizeof(second_)),
    };
    transfer_->actual_uris = {
        "file://nfs/first?size=5",
        "file://nfs/rewritten?size=9",
    };

    EXPECT_EQ(ER_SDKWRITE_ERROR, client_->SaveObjects("trace", keys_, sizes_, buffers_));
    EXPECT_EQ(1, transfer_->save_calls);
    EXPECT_EQ(1, metadata_->finish_calls);
    EXPECT_EQ((std::vector<bool>{false, false}), metadata_->finished_keys);
}

TEST_F(KvMetaObjectClientTest, PropagatesCommitFailureWithoutRepeatingDataWrite) {
    metadata_->start_result.write_session_id = "session";
    metadata_->start_result.key_mask = {false, false};
    metadata_->start_result.locations = {
        MakeLocation("file://nfs/first?size=5", sizeof(first_)),
        MakeLocation("file://nfs/second?size=9", sizeof(second_)),
    };
    metadata_->finish_ec = ER_INVALID_GRPCSTATUS;

    EXPECT_EQ(ER_INVALID_GRPCSTATUS, client_->SaveObjects("trace", keys_, sizes_, buffers_));
    EXPECT_EQ(1, metadata_->start_calls);
    EXPECT_EQ(1, transfer_->save_calls);
    EXPECT_EQ(1, metadata_->finish_calls);
    EXPECT_EQ((std::vector<bool>{true, true}), metadata_->finished_keys);
}

TEST_F(KvMetaObjectClientTest, CommitExceptionsBecomeAmbiguousErrorsWithoutRepeatingDataWrite) {
    metadata_->start_result.write_session_id = "session";
    metadata_->start_result.key_mask = {false, false};
    metadata_->start_result.locations = {
        MakeLocation("file://nfs/first?size=5", sizeof(first_)),
        MakeLocation("file://nfs/second?size=9", sizeof(second_)),
    };

    for (const auto mode : {ThrowMode::STANDARD, ThrowMode::UNKNOWN}) {
        SCOPED_TRACE(static_cast<int>(mode));
        metadata_->finish_throw = mode;

        EXPECT_EQ(ER_INVALID_GRPCSTATUS, client_->SaveObjects("trace", keys_, sizes_, buffers_));
    }
    EXPECT_EQ(2, transfer_->save_calls);
    EXPECT_EQ(2, metadata_->finish_calls);
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

TEST_F(KvMetaObjectClientTest, RollbackExceptionsBecomeAmbiguousErrors) {
    metadata_->start_result.write_session_id = "session";
    metadata_->start_result.key_mask = {false, false};
    metadata_->start_result.locations = {
        MakeLocation("file://nfs/first?size=5", sizeof(first_)),
        MakeLocation("file://nfs/second?size=9", sizeof(second_)),
    };
    transfer_->save_ec = ER_SDKWRITE_ERROR;

    for (const auto mode : {ThrowMode::STANDARD, ThrowMode::UNKNOWN}) {
        SCOPED_TRACE(static_cast<int>(mode));
        metadata_->finish_throw = mode;
        EXPECT_EQ(ER_INVALID_GRPCSTATUS, client_->SaveObjects("trace", keys_, sizes_, buffers_));
    }
    EXPECT_EQ(2, metadata_->finish_calls);
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
    EXPECT_EQ((UriStrVec{"file://nfs/first?size=5", "file://nfs/second?size=9"}), transfer_->loaded_uris);
    EXPECT_EQ(sizes_, transfer_->loaded_sizes);
    EXPECT_EQ((std::vector<void *>{first_, second_}), transfer_->loaded_bases);
    EXPECT_EQ(2U, transfer_->loaded_buffer_count);
}

TEST_F(KvMetaObjectClientTest, PropagatesLoadFailureAfterOneExactDataPlaneCall) {
    metadata_->get_result.hit_mask = {true, true};
    metadata_->get_result.locations = {
        MakeLocation("file://nfs/first?size=5", sizeof(first_)),
        MakeLocation("file://nfs/second?size=9", sizeof(second_)),
    };
    transfer_->load_ec = ER_SDKREAD_ERROR;

    EXPECT_EQ(ER_SDKREAD_ERROR, client_->LoadObjects("load", keys_, sizes_, buffers_));
    EXPECT_EQ(1, metadata_->get_calls);
    EXPECT_EQ("load", metadata_->get_trace);
    EXPECT_EQ(1, transfer_->load_calls);
    EXPECT_EQ(sizes_, transfer_->loaded_sizes);
}

TEST_F(KvMetaObjectClientTest, MetadataAndDataPlaneLoadExceptionsAreContained) {
    metadata_->get_result.hit_mask = {true, true};
    metadata_->get_result.locations = {
        MakeLocation("file://nfs/first?size=5", sizeof(first_)),
        MakeLocation("file://nfs/second?size=9", sizeof(second_)),
    };

    for (const auto mode : {ThrowMode::STANDARD, ThrowMode::UNKNOWN}) {
        SCOPED_TRACE("metadata");
        metadata_->get_throw = mode;
        EXPECT_EQ(ER_SERVICE_INTERNAL_ERROR, client_->LoadObjects("load", keys_, sizes_, buffers_));
    }
    EXPECT_EQ(0, transfer_->load_calls);

    metadata_->get_throw = ThrowMode::NONE;
    for (const auto mode : {ThrowMode::STANDARD, ThrowMode::UNKNOWN}) {
        SCOPED_TRACE("data plane");
        transfer_->load_throw = mode;
        EXPECT_EQ(ER_SDKREAD_ERROR, client_->LoadObjects("load", keys_, sizes_, buffers_));
    }
    EXPECT_EQ(2, transfer_->load_calls);
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

TEST_F(KvMetaObjectClientTest, MalformedLocationSchemaIsInternalErrorNotSizeMismatch) {
    metadata_->get_result.hit_mask = {true, true};
    const auto valid_second = MakeLocation("file://nfs/second?size=9", sizeof(second_));
    const auto expect_internal = [&](KvMetaValueLocation malformed) {
        metadata_->get_result.locations = {std::move(malformed), valid_second};
        EXPECT_EQ(ER_SERVICE_INTERNAL_ERROR, client_->LoadObjects("trace", keys_, sizes_, buffers_));
        EXPECT_EQ(0, transfer_->load_calls);
    };

    auto malformed = MakeLocation("file://nfs/first?size=5", sizeof(first_));
    malformed.type = KvMetaStorageType::UNSPECIFIED;
    expect_internal(std::move(malformed));

    malformed = MakeLocation("file://nfs/first?size=5", sizeof(first_));
    malformed.type = static_cast<KvMetaStorageType>(999);
    expect_internal(std::move(malformed));

    malformed = MakeLocation("file://nfs/first?size=5", sizeof(first_));
    malformed.location_specs.push_back({"unexpected", "file://nfs/other?size=5"});
    expect_internal(std::move(malformed));

    malformed = MakeLocation("file://nfs/first?size=5", sizeof(first_));
    malformed.location_specs[0].spec_name = "unexpected";
    expect_internal(std::move(malformed));

    malformed = MakeLocation("", sizeof(first_));
    expect_internal(std::move(malformed));
}

TEST_F(KvMetaObjectClientTest, DoesNotReadDataForMalformedMetadataAlignment) {
    metadata_->get_result.hit_mask = {true};
    metadata_->get_result.locations = {
        MakeLocation("file://nfs/first?size=5", sizeof(first_)),
        MakeLocation("file://nfs/second?size=9", sizeof(second_)),
    };

    EXPECT_EQ(ER_SERVICE_INTERNAL_ERROR, client_->LoadObjects("trace", keys_, sizes_, buffers_));
    EXPECT_EQ(0, transfer_->load_calls);

    metadata_->get_result.hit_mask = {true, true};
    metadata_->get_result.locations.pop_back();
    EXPECT_EQ(ER_SERVICE_INTERNAL_ERROR, client_->LoadObjects("trace", keys_, sizes_, buffers_));
    EXPECT_EQ(0, transfer_->load_calls);
}

TEST_F(KvMetaObjectClientTest, RemoveValidatesBeforeMetadataAndForwardsResult) {
    std::vector<std::string> too_many_keys;
    for (std::size_t i = 0; i < 65; ++i) {
        too_many_keys.push_back("key-" + std::to_string(i));
    }

    EXPECT_EQ(ER_INVALID_PARAMS, client_->Remove("trace", {}));
    EXPECT_EQ(ER_INVALID_PARAMS, client_->Remove("trace", too_many_keys));
    EXPECT_EQ(ER_INVALID_PARAMS, client_->Remove("trace", {""}));
    EXPECT_EQ(ER_INVALID_PARAMS, client_->Remove("trace", {std::string(513, 'k')}));
    EXPECT_EQ(ER_INVALID_PARAMS, client_->Remove("trace", {"duplicate", "duplicate"}));
    EXPECT_EQ(0, metadata_->remove_calls);

    metadata_->remove_ec = ER_SERVICE_NOT_READY;
    EXPECT_EQ(ER_SERVICE_NOT_READY, client_->Remove("remove-trace", keys_));
    EXPECT_EQ(1, metadata_->remove_calls);
    EXPECT_EQ("remove-trace", metadata_->remove_trace);
    EXPECT_EQ(keys_, metadata_->removed_keys);
}

TEST_F(KvMetaObjectClientTest, RemoveExceptionsBecomeAmbiguousErrors) {
    for (const auto mode : {ThrowMode::STANDARD, ThrowMode::UNKNOWN}) {
        SCOPED_TRACE(static_cast<int>(mode));
        metadata_->remove_throw = mode;
        EXPECT_EQ(ER_INVALID_GRPCSTATUS, client_->Remove("remove", keys_));
    }
    EXPECT_EQ(2, metadata_->remove_calls);
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
    malformed[0].iovs[0].base =
        reinterpret_cast<void *>(std::numeric_limits<std::uintptr_t>::max() - sizeof(first_) + 1);
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

TEST(KvMetaObjectClientDependencyTest, MissingInternalDependenciesFailClosed) {
    char payload = 0;
    KvMetaObjectClientImpl client(nullptr, nullptr, 1024, 30);
    const std::vector<std::string> keys{"key"};
    const std::vector<std::uint64_t> sizes{1};
    const BlockBuffers buffers{MakeBuffer(&payload, 1)};

    EXPECT_EQ(ER_CLIENT_NOT_EXISTS, client.SaveObjects("trace", keys, sizes, buffers));
    EXPECT_EQ(ER_CLIENT_NOT_EXISTS, client.LoadObjects("trace", keys, sizes, buffers));
    EXPECT_EQ(ER_CLIENT_NOT_EXISTS, client.Remove("trace", keys));
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

TEST(KvMetaObjectClientCreateTest, RejectsMalformedLocalRegistrationBeforeMetadataInitialization) {
    auto config = MakeStaticallyValidCreateConfig();
    RegistSpan span;
    config.transfer_init_params.regist_span = &span;

    span.base = reinterpret_cast<void *>(0x1000);
    auto [partial_span_ec, partial_span_client] = KvMetaObjectClient::Create("trace", config);
    EXPECT_EQ(ER_INVALID_PARAMS, partial_span_ec);
    EXPECT_EQ(nullptr, partial_span_client);

    span.base = reinterpret_cast<void *>(std::numeric_limits<std::uintptr_t>::max() - 3);
    span.size = 4;
    auto [overflow_span_ec, overflow_span_client] = KvMetaObjectClient::Create("trace", config);
    EXPECT_EQ(ER_INVALID_PARAMS, overflow_span_ec);
    EXPECT_EQ(nullptr, overflow_span_client);

    config.transfer_init_params.regist_span = nullptr;
    SharedMemoryRegistration partial_registration;
    partial_registration.base = reinterpret_cast<void *>(0x1000);
    auto [partial_shm_ec, partial_shm_client] =
        KvMetaObjectClient::Create("trace", config, partial_registration);
    EXPECT_EQ(ER_INVALID_PARAMS, partial_shm_ec);
    EXPECT_EQ(nullptr, partial_shm_client);

    SharedMemoryRegistration invalid_fd_registration;
    invalid_fd_registration.base = reinterpret_cast<void *>(0x1000);
    invalid_fd_registration.size = 4096;
    invalid_fd_registration.fd = std::numeric_limits<int>::max();
    auto [invalid_fd_ec, invalid_fd_client] =
        KvMetaObjectClient::Create("trace", config, invalid_fd_registration);
    EXPECT_EQ(ER_INVALID_PARAMS, invalid_fd_ec);
    EXPECT_EQ(nullptr, invalid_fd_client);

    std::unique_ptr<FILE, decltype(&std::fclose)> backing_file(std::tmpfile(), &std::fclose);
    ASSERT_NE(nullptr, backing_file);
    ASSERT_EQ(0, ftruncate(fileno(backing_file.get()), 8));

    SharedMemoryRegistration overflowing_registration;
    overflowing_registration.base =
        reinterpret_cast<void *>(std::numeric_limits<std::uintptr_t>::max() - 3);
    overflowing_registration.size = 4;
    overflowing_registration.fd = fileno(backing_file.get());
    auto [overflowing_registration_ec, overflowing_registration_client] =
        KvMetaObjectClient::Create("trace", config, overflowing_registration);
    EXPECT_EQ(ER_INVALID_PARAMS, overflowing_registration_ec);
    EXPECT_EQ(nullptr, overflowing_registration_client);

    SharedMemoryRegistration undersized_registration;
    undersized_registration.base = reinterpret_cast<void *>(0x1000);
    undersized_registration.size = 9;
    undersized_registration.fd = fileno(backing_file.get());
    auto [undersized_registration_ec, undersized_registration_client] =
        KvMetaObjectClient::Create("trace", config, undersized_registration);
    EXPECT_EQ(ER_INVALID_PARAMS, undersized_registration_ec);
    EXPECT_EQ(nullptr, undersized_registration_client);

    // The explicitly disabled registration remains valid and reaches the next
    // initialization stage. Empty metadata addresses then fail locally.
    SharedMemoryRegistration disabled_registration;
    auto [disabled_ec, disabled_client] = KvMetaObjectClient::Create("trace", config, disabled_registration);
    EXPECT_EQ(ER_METACLIENT_INIT_ERROR, disabled_ec);
    EXPECT_EQ(nullptr, disabled_client);
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
