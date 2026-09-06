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

    std::pair<ClientErrorCode, KvMetaInstanceInfo> GetInstanceInfo(const std::string &) override {
        return {ER_OK, {}};
    }

    std::pair<ClientErrorCode, KvMetaGetResult>
    Get(const std::string &, const std::vector<std::string> &) override {
        ++get_calls;
        return {get_ec, get_result};
    }

    std::pair<ClientErrorCode, KvMetaStartWriteResult>
    StartWrite(const std::string &,
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

    std::pair<ClientErrorCode, UriStrVec>
    SaveObjects(const UriStrVec &uris,
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
        client_ = std::make_unique<KvMetaObjectClientImpl>(
            std::move(metadata), std::move(transfer), 1024, 30);
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

    EXPECT_EQ(ER_OK, client_->SaveObjects("trace", keys_, sizes_, buffers_));
    EXPECT_EQ(1, transfer_->save_calls);
    EXPECT_EQ((std::vector<std::uint64_t>{sizeof(second_)}), transfer_->saved_sizes);
    EXPECT_EQ(1U, transfer_->saved_buffer_count);
    EXPECT_EQ(1, metadata_->finish_calls);
    EXPECT_EQ("session", metadata_->finished_session);
    EXPECT_EQ((std::vector<bool>{true}), metadata_->finished_keys);
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

} // namespace
} // namespace kv_cache_manager
