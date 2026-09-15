#include <cstddef>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <type_traits>
#include <unistd.h>

#include "kv_cache_manager/client/include/manager_client.h"
#include "kv_cache_manager/client/include/meta_client.h"
#include "kv_cache_manager/client/src/transfer_client_impl.h"
#include "kv_cache_manager/common/unittest.h"

using namespace kv_cache_manager;

static_assert(offsetof(RegistSpan, base) == 0);
static_assert(offsetof(RegistSpan, size) == sizeof(void *));
static_assert(sizeof(RegistSpan) == sizeof(void *) + sizeof(size_t));

class RecordingTransferClient : public TransferClient {
public:
    using TransferClient::LoadKvCaches;
    using TransferClient::SaveKvCaches;

    ClientErrorCode LoadKvCaches(const UriStrVec &uri_str_vec,
                                 const BlockBuffers &block_buffers,
                                 std::shared_ptr<TransferTraceInfo> trace_info) override {
        return LoadKvCaches(uri_str_vec, block_buffers, LoadKvCachesOptions::WithTraceInfo(std::move(trace_info)));
    }

    std::pair<ClientErrorCode, UriStrVec> SaveKvCaches(const UriStrVec &uri_str_vec,
                                                       const BlockBuffers &block_buffers,
                                                       std::shared_ptr<TransferTraceInfo> trace_info) override {
        auto [ec, result] =
            SaveKvCaches(uri_str_vec, block_buffers, SaveKvCachesOptions::WithTraceInfo(std::move(trace_info)));
        return {ec, std::move(result.uri_str_vec)};
    }

    ClientErrorCode LoadKvCaches(const UriStrVec &uri_str_vec,
                                 const BlockBuffers &block_buffers,
                                 const LoadKvCachesOptions &options) override {
        last_load_uri_count = uri_str_vec.size();
        last_load_buffer_count = block_buffers.size();
        last_load_options = options;
        return ER_OK;
    }

    std::pair<ClientErrorCode, SaveKvCachesResult> SaveKvCaches(const UriStrVec &uri_str_vec,
                                                                const BlockBuffers &block_buffers,
                                                                const SaveKvCachesOptions &options) override {
        last_save_uri_count = uri_str_vec.size();
        last_save_buffer_count = block_buffers.size();
        last_save_options = options;
        SaveKvCachesResult result;
        result.uri_str_vec = uri_str_vec;
        return {ER_OK, std::move(result)};
    }

    size_t last_load_uri_count{0};
    size_t last_load_buffer_count{0};
    LoadKvCachesOptions last_load_options;
    size_t last_save_uri_count{0};
    size_t last_save_buffer_count{0};
    SaveKvCachesOptions last_save_options;

protected:
    ClientErrorCode Init(const std::string &, const InitParams &) override { return ER_OK; }
};

class RecordingManagerClient : public ManagerClient {
public:
    using ManagerClient::FinishWrite;
    using ManagerClient::LoadKvCaches;
    using ManagerClient::MatchLocation;
    using ManagerClient::MatchMeta;
    using ManagerClient::SaveKvCaches;

    std::pair<ClientErrorCode, Locations> MatchLocation(const std::string &trace_id,
                                                        QueryType query_type,
                                                        const std::vector<int64_t> &keys,
                                                        const std::vector<int64_t> &tokens,
                                                        const BlockMask &block_mask,
                                                        int32_t sw_size,
                                                        const std::vector<std::string> &location_spec_names) override {
        auto [ec, result] = MatchLocation(trace_id,
                                          query_type,
                                          keys,
                                          tokens,
                                          block_mask,
                                          location_spec_names,
                                          MatchLocationOptions::WithSlideWindowSize(sw_size));
        return {ec, std::move(result.locations)};
    }

    std::pair<ClientErrorCode, MatchLocationResult> MatchLocation(const std::string &,
                                                                  QueryType,
                                                                  const std::vector<int64_t> &,
                                                                  const std::vector<int64_t> &,
                                                                  const BlockMask &,
                                                                  const std::vector<std::string> &,
                                                                  const MatchLocationOptions &options) override {
        last_match_location_options = options;
        return {ER_OK, MatchLocationResult{}};
    }

    std::pair<ClientErrorCode, WriteLocation> StartWrite(const std::string &,
                                                         const std::vector<int64_t> &,
                                                         const std::vector<int64_t> &,
                                                         const std::vector<std::string> &,
                                                         int64_t) override {
        return {ER_OK, WriteLocation{}};
    }

    ClientErrorCode FinishWrite(const std::string &trace_id,
                                const std::string &write_session_id,
                                const BlockMask &success_block,
                                const Locations &locations) override {
        return FinishWrite(trace_id, write_session_id, success_block, locations, FinishWriteOptions{});
    }

    ClientErrorCode FinishWrite(const std::string &,
                                const std::string &,
                                const BlockMask &,
                                const Locations &,
                                const FinishWriteOptions &options) override {
        last_finish_write_options = options;
        return ER_OK;
    }

    std::pair<ClientErrorCode, MatchMetaResult> MatchMeta(const std::string &,
                                                          const std::vector<int64_t> &,
                                                          const std::vector<int64_t> &,
                                                          const BlockMask &,
                                                          const MatchMetaOptions &options) override {
        last_match_meta_options = options;
        return {ER_OK, MatchMetaResult{}};
    }

    std::pair<ClientErrorCode, Metas> MatchMeta(const std::string &trace_id,
                                                const std::vector<int64_t> &keys,
                                                const std::vector<int64_t> &tokens,
                                                const BlockMask &block_mask,
                                                int32_t detail_level) override {
        auto [ec, result] =
            MatchMeta(trace_id, keys, tokens, block_mask, MatchMetaOptions::WithDetailLevel(detail_level));
        return {ec, std::move(result.metas)};
    }

    ClientErrorCode RemoveCache(const std::string &,
                                const std::vector<int64_t> &,
                                const std::vector<int64_t> &,
                                const BlockMask &) override {
        return ER_OK;
    }

    ClientErrorCode LoadKvCaches(const UriStrVec &, const BlockBuffers &, const LoadKvCachesOptions &options) override {
        last_load_options = options;
        return ER_OK;
    }

    ClientErrorCode LoadKvCaches(const UriStrVec &uri_str_vec, const BlockBuffers &block_buffers) override {
        return LoadKvCaches(uri_str_vec, block_buffers, LoadKvCachesOptions{});
    }

    std::pair<ClientErrorCode, SaveKvCachesResult>
    SaveKvCaches(const UriStrVec &, const BlockBuffers &, const SaveKvCachesOptions &options) override {
        last_save_options = options;
        return {ER_OK, SaveKvCachesResult{}};
    }

    std::pair<ClientErrorCode, UriStrVec> SaveKvCaches(const UriStrVec &uri_str_vec,
                                                       const BlockBuffers &block_buffers) override {
        auto [ec, result] = SaveKvCaches(uri_str_vec, block_buffers, SaveKvCachesOptions{});
        return {ec, std::move(result.uri_str_vec)};
    }

    MatchLocationOptions last_match_location_options;
    FinishWriteOptions last_finish_write_options;
    MatchMetaOptions last_match_meta_options;
    LoadKvCachesOptions last_load_options;
    SaveKvCachesOptions last_save_options;

protected:
    ClientErrorCode Init(const std::string &, InitParams &) override { return ER_OK; }
    void Shutdown() override {}
};

// These fixtures intentionally implement only the released virtual interface.
// If a future change removes or reorders the compatibility surface, they stop
// compiling or the fallback assertions below fail.
class LegacyOnlyTransferClient : public TransferClient {
public:
    using TransferClient::LoadKvCaches;
    using TransferClient::SaveKvCaches;

    ClientErrorCode
    LoadKvCaches(const UriStrVec &, const BlockBuffers &, std::shared_ptr<TransferTraceInfo> trace_info) override {
        ++load_calls;
        last_trace_info = std::move(trace_info);
        return ER_OK;
    }

    std::pair<ClientErrorCode, UriStrVec> SaveKvCaches(const UriStrVec &uri_str_vec,
                                                       const BlockBuffers &,
                                                       std::shared_ptr<TransferTraceInfo> trace_info) override {
        ++save_calls;
        last_trace_info = std::move(trace_info);
        return {ER_OK, uri_str_vec};
    }

    int load_calls{0};
    int save_calls{0};
    std::shared_ptr<TransferTraceInfo> last_trace_info;

protected:
    ClientErrorCode Init(const std::string &, const InitParams &) override { return ER_OK; }
};

class LegacyOnlyMetaClient : public MetaClient {
public:
    using MetaClient::FinishWrite;
    using MetaClient::MatchLocation;
    using MetaClient::MatchLocationLen;
    using MetaClient::MatchMeta;

    std::pair<ClientErrorCode, Locations> MatchLocation(const std::string &,
                                                        QueryType,
                                                        const std::vector<int64_t> &,
                                                        const std::vector<int64_t> &,
                                                        const BlockMask &,
                                                        int32_t sw_size,
                                                        const std::vector<std::string> &) override {
        last_sw_size = sw_size;
        return {ER_OK, Locations{}};
    }

    std::pair<ClientErrorCode, WriteLocation> StartWrite(const std::string &,
                                                         const std::vector<int64_t> &,
                                                         const std::vector<int64_t> &,
                                                         const std::vector<std::string> &,
                                                         int64_t) override {
        return {ER_OK, WriteLocation{}};
    }

    ClientErrorCode
    FinishWrite(const std::string &, const std::string &, const BlockMask &, const Locations &) override {
        ++finish_calls;
        return ER_OK;
    }

    std::pair<ClientErrorCode, Metas> MatchMeta(const std::string &,
                                                const std::vector<int64_t> &,
                                                const std::vector<int64_t> &,
                                                const BlockMask &,
                                                int32_t detail_level) override {
        last_detail_level = detail_level;
        return {ER_OK, Metas{}};
    }

    std::pair<ClientErrorCode, int64_t> MatchLocationLen(const std::string &,
                                                         QueryType,
                                                         const std::vector<int64_t> &,
                                                         const std::vector<int64_t> &,
                                                         int32_t sw_size) override {
        last_sw_size = sw_size;
        return {ER_OK, 0};
    }

    ClientErrorCode RemoveCache(const std::string &,
                                const std::vector<int64_t> &,
                                const std::vector<int64_t> &,
                                const BlockMask &) override {
        return ER_OK;
    }

    const std::string &GetStorageConfig() const override { return storage_config; }

    int32_t last_sw_size{-1};
    int32_t last_detail_level{-1};
    int finish_calls{0};
    std::string storage_config;

protected:
    ClientErrorCode Init(const std::string &, const InitParams &) override { return ER_OK; }
    void Shutdown() override {}
};

class LegacyOnlyManagerClient : public ManagerClient {
public:
    using ManagerClient::FinishWrite;
    using ManagerClient::LoadKvCaches;
    using ManagerClient::MatchLocation;
    using ManagerClient::MatchMeta;
    using ManagerClient::SaveKvCaches;

    std::pair<ClientErrorCode, Locations> MatchLocation(const std::string &,
                                                        QueryType,
                                                        const std::vector<int64_t> &,
                                                        const std::vector<int64_t> &,
                                                        const BlockMask &,
                                                        int32_t sw_size,
                                                        const std::vector<std::string> &) override {
        last_sw_size = sw_size;
        return {ER_OK, Locations{}};
    }

    std::pair<ClientErrorCode, WriteLocation> StartWrite(const std::string &,
                                                         const std::vector<int64_t> &,
                                                         const std::vector<int64_t> &,
                                                         const std::vector<std::string> &,
                                                         int64_t) override {
        return {ER_OK, WriteLocation{}};
    }

    ClientErrorCode
    FinishWrite(const std::string &, const std::string &, const BlockMask &, const Locations &) override {
        ++finish_calls;
        return ER_OK;
    }

    std::pair<ClientErrorCode, Metas> MatchMeta(const std::string &,
                                                const std::vector<int64_t> &,
                                                const std::vector<int64_t> &,
                                                const BlockMask &,
                                                int32_t detail_level) override {
        last_detail_level = detail_level;
        return {ER_OK, Metas{}};
    }

    ClientErrorCode RemoveCache(const std::string &,
                                const std::vector<int64_t> &,
                                const std::vector<int64_t> &,
                                const BlockMask &) override {
        return ER_OK;
    }

    ClientErrorCode LoadKvCaches(const UriStrVec &, const BlockBuffers &) override {
        ++load_calls;
        return ER_OK;
    }

    std::pair<ClientErrorCode, UriStrVec> SaveKvCaches(const UriStrVec &uri_str_vec, const BlockBuffers &) override {
        ++save_calls;
        return {ER_OK, uri_str_vec};
    }

    int32_t last_sw_size{-1};
    int32_t last_detail_level{-1};
    int finish_calls{0};
    int load_calls{0};
    int save_calls{0};

protected:
    ClientErrorCode Init(const std::string &, InitParams &) override { return ER_OK; }
    void Shutdown() override {}
};

static_assert(!std::is_abstract_v<LegacyOnlyTransferClient>);
static_assert(!std::is_abstract_v<LegacyOnlyMetaClient>);
static_assert(!std::is_abstract_v<LegacyOnlyManagerClient>);

TEST(ClientOptionsTest, LegacyEmptyBraceTraceArgumentRemainsUnambiguous) {
    LegacyOnlyTransferClient transfer;
    const UriStrVec uris = {"file://test"};
    const BlockBuffers buffers = {BlockBuffer{}};

    EXPECT_EQ(ER_OK, transfer.LoadKvCaches(uris, buffers, {}));
    auto [save_ec, saved_uris] = transfer.SaveKvCaches(uris, buffers, {});
    EXPECT_EQ(ER_OK, save_ec);
    EXPECT_EQ(uris, saved_uris);
    EXPECT_EQ(1, transfer.load_calls);
    EXPECT_EQ(1, transfer.save_calls);
    EXPECT_EQ(nullptr, transfer.last_trace_info);
}

TEST(ClientOptionsTest, LegacyOnlySubclassesUseSafeOptionFallbacks) {
    const std::string trace_id = "trace";
    const std::vector<int64_t> keys = {1};
    const std::vector<int64_t> tokens = {2};
    const BlockMask block_mask = static_cast<BlockMaskOffset>(0);
    const std::vector<std::string> spec_names = {"tp0"};
    const Locations locations = {{{"tp0", "file://test"}}};
    const UriStrVec uris = {"file://test"};
    const BlockBuffers buffers = {BlockBuffer{}};
    const std::vector<int64_t> checksums = {7};

    LegacyOnlyTransferClient transfer;
    auto trace_info = std::make_shared<TransferTraceInfo>();
    EXPECT_EQ(ER_OK, transfer.LoadKvCaches(uris, buffers, LoadKvCachesOptions::WithTraceInfo(trace_info)));
    EXPECT_EQ(1, transfer.load_calls);
    EXPECT_EQ(trace_info, transfer.last_trace_info);
    EXPECT_EQ(ER_CHECKSUM_UNAVAILABLE,
              transfer.LoadKvCaches(uris, buffers, LoadKvCachesOptions::VerifyWith(checksums)));
    EXPECT_EQ(1, transfer.load_calls);
    EXPECT_EQ(ER_OK,
              transfer.LoadKvCaches(
                  uris, buffers, LoadKvCachesOptions::VerifyWith({0}, std::vector<bool>{false})));
    EXPECT_EQ(2, transfer.load_calls);
    EXPECT_EQ(ER_CHECKSUM_MISMATCH,
              transfer.LoadKvCaches(
                  uris, buffers, LoadKvCachesOptions::VerifyWith(std::vector<int64_t>{})));
    EXPECT_EQ(2, transfer.load_calls);
    EXPECT_EQ(ER_OK, transfer.SaveKvCaches(uris, buffers, SaveKvCachesOptions::WithTraceInfo(trace_info)).first);
    EXPECT_EQ(1, transfer.save_calls);
    EXPECT_EQ(ER_CHECKSUM_UNAVAILABLE,
              transfer.SaveKvCaches(uris, buffers, SaveKvCachesOptions::WithChecksums()).first);
    EXPECT_EQ(1, transfer.save_calls);
    EXPECT_EQ(ER_CHECKSUM_MISMATCH,
              transfer.SaveKvCaches(
                          uris, buffers, SaveKvCachesOptions::VerifyCallerChecksums(std::vector<int64_t>{}))
                  .first);
    EXPECT_EQ(1, transfer.save_calls);

    LegacyOnlyMetaClient meta;
    EXPECT_EQ(ER_OK,
              meta.MatchLocation(trace_id,
                                 QueryType::QT_PREFIX_MATCH,
                                 keys,
                                 tokens,
                                 block_mask,
                                 spec_names,
                                 MatchLocationOptions::WithSlideWindowSize(3))
                  .first);
    EXPECT_EQ(3, meta.last_sw_size);
    EXPECT_EQ(ER_CHECKSUM_UNAVAILABLE,
              meta.MatchLocation(trace_id,
                                 QueryType::QT_PREFIX_MATCH,
                                 keys,
                                 tokens,
                                 block_mask,
                                 spec_names,
                                 MatchLocationOptions::WithChecksums())
                  .first);
    EXPECT_EQ(ER_OK, meta.FinishWrite(trace_id, "session", block_mask, locations, FinishWriteOptions{}));
    EXPECT_EQ(1, meta.finish_calls);
    EXPECT_EQ(ER_CHECKSUM_UNAVAILABLE,
              meta.FinishWrite(
                  trace_id, "session", block_mask, locations, FinishWriteOptions::WithChecksums("tp0", checksums)));
    EXPECT_EQ(1, meta.finish_calls);
    EXPECT_EQ(ER_OK, meta.MatchMeta(trace_id, keys, tokens, block_mask, MatchMetaOptions::WithDetailLevel(4)).first);
    EXPECT_EQ(4, meta.last_detail_level);
    EXPECT_EQ(ER_CHECKSUM_UNAVAILABLE,
              meta.MatchMeta(trace_id, keys, tokens, block_mask, MatchMetaOptions::WithChecksums()).first);
    EXPECT_EQ(
        ER_OK,
        meta.MatchLocationLen(
                trace_id, QueryType::QT_PREFIX_MATCH, keys, tokens, MatchLocationLenOptions::WithSlideWindowSize(5))
            .first);
    EXPECT_EQ(5, meta.last_sw_size);

    LegacyOnlyManagerClient manager;
    EXPECT_EQ(ER_OK,
              manager
                  .MatchLocation(trace_id,
                                 QueryType::QT_PREFIX_MATCH,
                                 keys,
                                 tokens,
                                 block_mask,
                                 spec_names,
                                 MatchLocationOptions::WithSlideWindowSize(6))
                  .first);
    EXPECT_EQ(6, manager.last_sw_size);
    EXPECT_EQ(ER_CHECKSUM_UNAVAILABLE,
              manager
                  .MatchLocation(trace_id,
                                 QueryType::QT_PREFIX_MATCH,
                                 keys,
                                 tokens,
                                 block_mask,
                                 spec_names,
                                 MatchLocationOptions::WithChecksums())
                  .first);
    EXPECT_EQ(ER_OK, manager.FinishWrite(trace_id, "session", block_mask, locations, FinishWriteOptions{}));
    EXPECT_EQ(1, manager.finish_calls);
    EXPECT_EQ(ER_CHECKSUM_UNAVAILABLE,
              manager.FinishWrite(
                  trace_id, "session", block_mask, locations, FinishWriteOptions::WithChecksums("tp0", checksums)));
    EXPECT_EQ(ER_OK, manager.MatchMeta(trace_id, keys, tokens, block_mask, MatchMetaOptions::WithDetailLevel(7)).first);
    EXPECT_EQ(7, manager.last_detail_level);
    EXPECT_EQ(ER_CHECKSUM_UNAVAILABLE,
              manager.MatchMeta(trace_id, keys, tokens, block_mask, MatchMetaOptions::WithChecksums()).first);
    EXPECT_EQ(ER_OK, manager.LoadKvCaches(uris, buffers, LoadKvCachesOptions{}));
    EXPECT_EQ(1, manager.load_calls);
    EXPECT_EQ(ER_CHECKSUM_UNAVAILABLE, manager.LoadKvCaches(uris, buffers, LoadKvCachesOptions::VerifyWith(checksums)));
    EXPECT_EQ(ER_OK,
              manager.LoadKvCaches(
                  uris, buffers, LoadKvCachesOptions::VerifyWith({0}, std::vector<bool>{false})));
    EXPECT_EQ(2, manager.load_calls);
    EXPECT_EQ(ER_CHECKSUM_MISMATCH,
              manager.LoadKvCaches(
                  uris, buffers, LoadKvCachesOptions::VerifyWith(std::vector<int64_t>{})));
    EXPECT_EQ(2, manager.load_calls);
    EXPECT_EQ(ER_OK, manager.SaveKvCaches(uris, buffers, SaveKvCachesOptions{}).first);
    EXPECT_EQ(1, manager.save_calls);
    EXPECT_EQ(ER_CHECKSUM_UNAVAILABLE, manager.SaveKvCaches(uris, buffers, SaveKvCachesOptions::WithChecksums()).first);
    EXPECT_EQ(ER_CHECKSUM_MISMATCH,
              manager.SaveKvCaches(
                         uris, buffers, SaveKvCachesOptions::VerifyCallerChecksums(std::vector<int64_t>{}))
                  .first);
    EXPECT_EQ(1, manager.save_calls);
}

TEST(ClientOptionsTest, TransferConvenienceOverloadsForwardOptions) {
    RecordingTransferClient client;
    TransferClient &client_api = client;
    UriStrVec uris = {"file://test_nfs/path?blkid=0&size=1"};
    BlockBuffers buffers = {BlockBuffer{}};

    EXPECT_EQ(ER_OK, client_api.LoadKvCaches(uris, buffers));
    EXPECT_EQ(uris.size(), client.last_load_uri_count);
    EXPECT_EQ(buffers.size(), client.last_load_buffer_count);
    EXPECT_EQ(nullptr, client.last_load_options.trace_info);
    EXPECT_TRUE(client.last_load_options.expected_checksums.empty());

    auto trace_info = std::make_shared<TransferTraceInfo>();
    trace_info->need_print = true;
    EXPECT_EQ(ER_OK, client_api.LoadKvCaches(uris, buffers, trace_info));
    EXPECT_EQ(trace_info, client.last_load_options.trace_info);
    EXPECT_TRUE(client.last_load_options.expected_checksums.empty());

    std::vector<int64_t> expected_checksums = {0x11};
    EXPECT_EQ(ER_OK,
              client_api.LoadKvCaches(
                  uris, buffers, LoadKvCachesOptions::VerifyWith(expected_checksums, trace_info, "load-trace")));
    EXPECT_EQ(trace_info, client.last_load_options.trace_info);
    EXPECT_EQ("load-trace", client.last_load_options.trace_id);
    EXPECT_EQ(expected_checksums, client.last_load_options.expected_checksums);
    EXPECT_TRUE(client.last_load_options.expected_checksum_present.empty());

    std::vector<bool> checksum_present = {true};
    EXPECT_EQ(ER_OK,
              client_api.LoadKvCaches(
                  uris, buffers, LoadKvCachesOptions::VerifyWith(expected_checksums, checksum_present, trace_info)));
    EXPECT_EQ(checksum_present, client.last_load_options.expected_checksum_present);

    auto save_result = client_api.SaveKvCaches(uris, buffers);
    EXPECT_EQ(ER_OK, save_result.first);
    EXPECT_EQ(uris.size(), client.last_save_uri_count);
    EXPECT_EQ(buffers.size(), client.last_save_buffer_count);
    EXPECT_EQ(nullptr, client.last_save_options.trace_info);
    EXPECT_FALSE(client.last_save_options.include_checksums);

    auto save_with_checksum_result =
        client_api.SaveKvCaches(uris, buffers, SaveKvCachesOptions::WithChecksums(trace_info, "save-trace"));
    EXPECT_EQ(ER_OK, save_with_checksum_result.first);
    EXPECT_EQ(trace_info, client.last_save_options.trace_info);
    EXPECT_EQ("save-trace", client.last_save_options.trace_id);
    EXPECT_TRUE(client.last_save_options.include_checksums);

    auto verify_options = SaveKvCachesOptions::VerifyCallerChecksums(expected_checksums, trace_info);
    auto verify_result = client_api.SaveKvCaches(uris, buffers, verify_options);
    EXPECT_EQ(ER_OK, verify_result.first);
    EXPECT_TRUE(client.last_save_options.include_checksums);
    EXPECT_TRUE(client.last_save_options.verify_caller_checksums);
    EXPECT_EQ(expected_checksums, client.last_save_options.expected_checksums);
}

TEST(ClientOptionsTest, ManagerConvenienceOverloadsForwardOptions) {
    RecordingManagerClient client;
    const std::string trace_id = "trace";
    const std::vector<int64_t> keys = {1, 2};
    const std::vector<int64_t> tokens = {3, 4};
    const BlockMask block_mask = static_cast<BlockMaskOffset>(0);
    const std::vector<std::string> spec_names = {"tp0"};
    const Locations locations = {{{"tp0", "file://test"}}};
    const UriStrVec uris = {"file://test"};
    const BlockBuffers buffers = {BlockBuffer{}};

    EXPECT_EQ(ER_OK,
              client.MatchLocation(trace_id, QueryType::QT_PREFIX_MATCH, keys, tokens, block_mask, spec_names).first);
    EXPECT_EQ(-1, client.last_match_location_options.sw_size);
    EXPECT_FALSE(client.last_match_location_options.include_checksums);

    EXPECT_EQ(
        ER_OK,
        client.MatchLocation(trace_id, QueryType::QT_REVERSE_ROLL_SW_MATCH, keys, tokens, block_mask, 7, spec_names)
            .first);
    EXPECT_EQ(7, client.last_match_location_options.sw_size);
    EXPECT_FALSE(client.last_match_location_options.include_checksums);

    EXPECT_EQ(ER_OK,
              client
                  .MatchLocation(trace_id,
                                 QueryType::QT_PREFIX_MATCH,
                                 keys,
                                 tokens,
                                 block_mask,
                                 spec_names,
                                 MatchLocationOptions::WithChecksums(8))
                  .first);
    EXPECT_EQ(8, client.last_match_location_options.sw_size);
    EXPECT_TRUE(client.last_match_location_options.include_checksums);

    EXPECT_EQ(ER_OK, client.FinishWrite(trace_id, "session", block_mask, locations));
    EXPECT_TRUE(client.last_finish_write_options.checksum_batches.empty());

    std::vector<int64_t> finish_checksums = {0x21, 0};
    EXPECT_EQ(
        ER_OK,
        client.FinishWrite(
            trace_id, "session", block_mask, locations, FinishWriteOptions::WithChecksums("tp0", finish_checksums)));
    ASSERT_EQ(1u, client.last_finish_write_options.checksum_batches.size());
    EXPECT_EQ("tp0", client.last_finish_write_options.checksum_batches[0].location_spec_name);
    EXPECT_EQ(finish_checksums, client.last_finish_write_options.checksum_batches[0].checksums);

    EXPECT_EQ(ER_OK, client.MatchMeta(trace_id, keys, tokens, block_mask, 1).first);
    EXPECT_EQ(1, client.last_match_meta_options.detail_level);
    EXPECT_FALSE(client.last_match_meta_options.include_checksums);

    EXPECT_EQ(ER_OK, client.MatchMeta(trace_id, keys, tokens, block_mask, MatchMetaOptions::WithChecksums(1)).first);
    EXPECT_EQ(1, client.last_match_meta_options.detail_level);
    EXPECT_TRUE(client.last_match_meta_options.include_checksums);

    EXPECT_EQ(ER_OK, client.LoadKvCaches(uris, buffers));
    EXPECT_TRUE(client.last_load_options.expected_checksums.empty());

    EXPECT_EQ(ER_OK, client.LoadKvCaches(uris, buffers, LoadKvCachesOptions::VerifyWith(finish_checksums)));
    EXPECT_EQ(finish_checksums, client.last_load_options.expected_checksums);

    MatchLocationResult match_result;
    match_result.checksum_results = {{"tp0", finish_checksums, {true, false}}};
    EXPECT_EQ(ER_OK, client.LoadKvCaches(uris, buffers, LoadKvCachesOptions::VerifyWith(match_result, "tp0")));
    EXPECT_EQ(match_result.checksum_results[0].checksum_present, client.last_load_options.expected_checksum_present);

    EXPECT_EQ(ER_OK, client.SaveKvCaches(uris, buffers).first);
    EXPECT_FALSE(client.last_save_options.include_checksums);

    EXPECT_EQ(ER_OK, client.SaveKvCaches(uris, buffers, SaveKvCachesOptions::WithChecksums()).first);
    EXPECT_TRUE(client.last_save_options.include_checksums);

    EXPECT_EQ(ER_OK,
              client.SaveKvCaches(uris, buffers, SaveKvCachesOptions::VerifyCallerChecksums(finish_checksums)).first);
    EXPECT_TRUE(client.last_save_options.verify_caller_checksums);
    EXPECT_EQ(finish_checksums, client.last_save_options.expected_checksums);
}

class TransferClientTest : public TESTBASE {
public:
    void SetUp() override {
        root_path_ = GetPrivateTestRuntimeDataPath();
        client_config_ = R"({
            "instance_group": "test_group",
            "instance_id": "test_instance",
            "block_size": 16,
            "sdk_config": {
                "thread_num": 4,
                "queue_size": 1000,
                "sdk_config": [],
                "timeout_config": {
                    "get_timeout_ms": 10000,
                    "put_timeout_ms": 30000
                }
            },
            "location_spec_infos": {
                "tp0": 1024
            }
        })";

        init_params_.role_type = RoleType::WORKER;
        init_params_.regist_span = new RegistSpan();
        auto buffer = malloc(1024 * 1024);
        init_params_.regist_span->base = buffer;
        init_params_.regist_span->size = 1024 * 1024;
        init_params_.self_location_spec_name = "tp0";
        init_params_.storage_configs = R"([
            {
                "type": "file",
                "global_unique_name": "test_nfs",
                "storage_spec": {
                    "root_path": "/tmp/test/",
                    "key_count_per_file": 5
                }
            }
        ])";
        ;
        InitFile();
    }

    void TearDown() override {
        client_config_.clear();
        free(init_params_.regist_span->base);
        delete init_params_.regist_span;
    }

private:
    void InitFile() {
        std::filesystem::create_directories(root_path_ + "tmp/test");
        std::string file_path = root_path_ + "tmp/test/key1";
        std::ofstream ofs(file_path);
        ASSERT_TRUE(ofs);
        ofs << test_data1_;
        ofs.close();

        file_path = root_path_ + "tmp/test/key2";
        ofs.open(file_path);
        ASSERT_TRUE(ofs);
        ofs << test_data2_;
        ofs.close();
        locations_ = {"file://test_nfs/" + root_path_ + "tmp/test/key1?blkid=0&size=1024",
                      "file://test_nfs/" + root_path_ + "tmp/test/key2?blkid=0&size=1024"};
    }

private:
    const char *test_data1_ = "test key1";
    const char *test_data2_ = "test key2";
    std::string root_path_;
    std::string client_config_;
    InitParams init_params_;
    UriStrVec locations_;
};

TEST_F(TransferClientTest, TestCreate) {
    {
        auto client = TransferClient::Create(client_config_, init_params_);
        EXPECT_NE(client, nullptr);
    }
    {
        std::string invalid_config = R"({})";
        auto client = TransferClient::Create(invalid_config, init_params_);
        EXPECT_EQ(client, nullptr);
    }
}

TEST_F(TransferClientTest, TestCreateWithSharedMemory) {
    FILE *file = tmpfile();
    ASSERT_NE(file, nullptr);
    ASSERT_EQ(ftruncate(fileno(file), static_cast<off_t>(init_params_.regist_span->size)), 0);

    SharedMemoryRegistration registration;
    registration.base = init_params_.regist_span->base;
    registration.size = init_params_.regist_span->size;
    registration.fd = fileno(file);

    auto client = TransferClient::Create(client_config_, init_params_, registration);
    ASSERT_NE(client, nullptr);

    ASSERT_EQ(fclose(file), 0);
    client.reset();
}

TEST_F(TransferClientTest, TestCreateWithDisabledSharedMemory) {
    SharedMemoryRegistration registration;
    auto client = TransferClient::Create(client_config_, init_params_, registration);
    EXPECT_NE(client, nullptr);
}

TEST_F(TransferClientTest, TestRejectsPartialSharedMemoryRegistration) {
    FILE *file = tmpfile();
    ASSERT_NE(file, nullptr);
    ASSERT_EQ(ftruncate(fileno(file), static_cast<off_t>(init_params_.regist_span->size)), 0);

    SharedMemoryRegistration registration;
    registration.base = init_params_.regist_span->base;
    registration.size = init_params_.regist_span->size;
    registration.fd = -1;
    EXPECT_EQ(TransferClient::Create(client_config_, init_params_, registration), nullptr);

    registration.fd = fileno(file);
    registration.base = nullptr;
    EXPECT_EQ(TransferClient::Create(client_config_, init_params_, registration), nullptr);

    registration.base = init_params_.regist_span->base;
    registration.size = 0;
    EXPECT_EQ(TransferClient::Create(client_config_, init_params_, registration), nullptr);

    registration.size = init_params_.regist_span->size;
    registration.base = reinterpret_cast<void *>(std::numeric_limits<uintptr_t>::max());
    EXPECT_EQ(TransferClient::Create(client_config_, init_params_, registration), nullptr);

    ASSERT_EQ(fclose(file), 0);
}

TEST_F(TransferClientTest, TestCreateWithEmptySelfLocationSpecName) {
    auto init_params = init_params_;
    init_params.self_location_spec_name = "";
    auto client = TransferClient::Create(client_config_, init_params);
    EXPECT_EQ(client, nullptr);
}

TEST_F(TransferClientTest, TestCreateWithEmptyAddress) {
    std::string client_config = R"({
        "instance_group": "group",
        "instance_id": "instance",
        "block_size": 128,
        "sdk_config": {},
        "model_deployment": {
            "model_name": "test_model",
            "dtype": "FP8",
            "use_mla": false,
            "tp_size": 1,
            "dp_size": 1,
            "pp_size": 1
        },
        "location_spec_infos": {
            "tp0": 1024
        }
    })";
    auto client = TransferClient::Create(client_config, init_params_);
    EXPECT_NE(client, nullptr);
}

TEST_F(TransferClientTest, TestLoadKvCaches) {
    auto client = TransferClient::Create(client_config_, init_params_);
    ASSERT_NE(client, nullptr);

    BlockBuffer buffer1, buffer2;
    BlockBuffers block_buffers = {buffer1, buffer2};

    EXPECT_EQ(ER_OK, client->LoadKvCaches(locations_, block_buffers));
}

TEST_F(TransferClientTest, TestSaveKvCaches) {
    auto client = TransferClient::Create(client_config_, init_params_);
    ASSERT_NE(client, nullptr);

    BlockBuffer buffer1, buffer2;
    BlockBuffers block_buffers = {buffer1, buffer2};

    auto result = client->SaveKvCaches(locations_, block_buffers);

    EXPECT_EQ(ER_OK, result.first);
    EXPECT_EQ(result.second.size(), locations_.size());
}

TEST_F(TransferClientTest, TestEmptyLocations) {
    auto client = TransferClient::Create(client_config_, init_params_);
    ASSERT_NE(client, nullptr);

    UriStrVec uri_str_vec = {};
    BlockBuffers block_buffers = {};

    EXPECT_EQ(ER_INVALID_PARAMS, client->LoadKvCaches(uri_str_vec, block_buffers));

    auto save_result = client->SaveKvCaches(uri_str_vec, block_buffers);
    EXPECT_EQ(ER_INVALID_PARAMS, save_result.first);
    EXPECT_TRUE(save_result.second.empty());
}

TEST_F(TransferClientTest, TestManyLocations) {
    auto client = TransferClient::Create(client_config_, init_params_);
    ASSERT_NE(client, nullptr);

    UriStrVec uri_str_vec;
    BlockBuffers block_buffers;

    for (int i = 0; i < 100; i++) {
        uri_str_vec.push_back("file://test_nfs/" + root_path_ + "tmp/test/key_" + std::to_string(i) +
                              "?blkid=0&size=1024");
        block_buffers.push_back(BlockBuffer());
    }

    auto save_result = client->SaveKvCaches(uri_str_vec, block_buffers);
    EXPECT_EQ(ER_OK, save_result.first);
    EXPECT_EQ(save_result.second.size(), uri_str_vec.size());
}

TEST_F(TransferClientTest, TestWrongRoleType) {
    auto init_params = init_params_;
    init_params.role_type = RoleType::SCHEDULER;
    auto client = TransferClient::Create(client_config_, init_params);
    EXPECT_EQ(client, nullptr);
}

TEST_F(TransferClientTest, TestMismatchedLocationsAndBuffers) {
    auto client = TransferClient::Create(client_config_, init_params_);
    ASSERT_NE(client, nullptr);

    BlockBuffer buffer1;
    BlockBuffers block_buffers = {buffer1};

    EXPECT_EQ(ER_INVALID_PARAMS, client->LoadKvCaches(locations_, block_buffers));
}

TEST_F(TransferClientTest, TestBlockBufferUsage) {
    auto client = TransferClient::Create(client_config_, init_params_);
    ASSERT_NE(client, nullptr);
    size_t len1 = strlen(test_data1_);
    size_t len2 = strlen(test_data2_);

    auto get_buffer = malloc(1024 * 1024);
    std::memcpy(get_buffer, test_data1_, len1);
    std::memcpy(static_cast<char *>(get_buffer) + len1, test_data2_, len2);

    BlockBuffer buffer1, buffer2;
    buffer1.iovs.resize(1);
    buffer2.iovs.resize(1);

    buffer1.iovs[0].type = MemoryType::CPU;
    buffer1.iovs[0].base = get_buffer;
    buffer1.iovs[0].size = len1;
    buffer1.iovs[0].ignore = false;

    buffer2.iovs[0].type = MemoryType::CPU;
    buffer2.iovs[0].base = static_cast<char *>(get_buffer) + len1;
    buffer2.iovs[0].size = len2;
    buffer2.iovs[0].ignore = false;

    BlockBuffers block_buffers = {buffer1, buffer2};
    ASSERT_EQ(ER_OK, client->LoadKvCaches(locations_, block_buffers));

    ASSERT_EQ(std::memcmp(buffer1.iovs[0].base, test_data1_, buffer1.iovs[0].size), 0);
    ASSERT_EQ(std::memcmp(buffer2.iovs[0].base, test_data2_, buffer2.iovs[0].size), 0);

    free(get_buffer);
}

// ============================================================================
// Multi-storage TransferClient tests
// ============================================================================
class TransferClientMultiStorageTest : public TESTBASE {
public:
    void SetUp() override {
        root_path_ = GetPrivateTestRuntimeDataPath();
        client_config_ = R"({
            "instance_group": "test_group",
            "instance_id": "test_instance",
            "block_size": 16,
            "sdk_config": {
                "thread_num": 4,
                "queue_size": 1000,
                "sdk_config": [],
                "timeout_config": {
                    "get_timeout_ms": 10000,
                    "put_timeout_ms": 30000
                }
            },
            "location_spec_infos": {
                "tp0": 1024
            }
        })";

        init_params_.role_type = RoleType::WORKER;
        init_params_.regist_span = new RegistSpan();
        auto buffer = malloc(1024 * 1024);
        init_params_.regist_span->base = buffer;
        init_params_.regist_span->size = 1024 * 1024;
        init_params_.self_location_spec_name = "tp0";
        init_params_.storage_configs = R"([
            {
                "type": "file",
                "global_unique_name": "nfs_a",
                "storage_spec": {
                    "root_path": "/tmp/nfs_a/",
                    "key_count_per_file": 5
                }
            },
            {
                "type": "file",
                "global_unique_name": "nfs_b",
                "storage_spec": {
                    "root_path": "/tmp/nfs_b/",
                    "key_count_per_file": 5
                }
            }
        ])";
    }

    void TearDown() override {
        free(init_params_.regist_span->base);
        delete init_params_.regist_span;
    }

protected:
    std::string root_path_;
    std::string client_config_;
    InitParams init_params_;
};

TEST_F(TransferClientMultiStorageTest, TestCreateWithMultiStorage) {
    auto client = TransferClient::Create(client_config_, init_params_);
    ASSERT_NE(client, nullptr);
}

TEST_F(TransferClientMultiStorageTest, TestSaveAndLoadMixedStorage) {
    auto client = TransferClient::Create(client_config_, init_params_);
    ASSERT_NE(client, nullptr);

    // 混合 URI: nfs_a, nfs_b, nfs_a（同 path 不同 blkid 保证 SDK 内部不重排）
    UriStrVec uri_str_vec = {
        "file://nfs_a/" + root_path_ + "tmp/nfs_a/file1?blkid=0&size=1024",
        "file://nfs_b/" + root_path_ + "tmp/nfs_b/file1?blkid=0&size=1024",
        "file://nfs_a/" + root_path_ + "tmp/nfs_a/file1?blkid=1&size=1024",
    };
    BlockBuffers block_buffers = {BlockBuffer(), BlockBuffer(), BlockBuffer()};

    // Save
    auto [save_ec, actual_uris] = client->SaveKvCaches(uri_str_vec, block_buffers);
    ASSERT_EQ(ER_OK, save_ec);
    ASSERT_EQ(3, actual_uris.size());
    // 验证返回 URI 顺序与输入一致
    EXPECT_EQ(uri_str_vec[0], actual_uris[0]);
    EXPECT_EQ(uri_str_vec[1], actual_uris[1]);
    EXPECT_EQ(uri_str_vec[2], actual_uris[2]);

    // Load
    EXPECT_EQ(ER_OK, client->LoadKvCaches(actual_uris, block_buffers));
}

// 回归测试：nfs_a/path1, nfs_b/pathX, nfs_a/path2 —— 同 backend（nfs_a）内多个
// 不同 path 被 nfs_b 交错隔开。修复前 LocalFileSdk::Put 按 path 分组后以
// unordered_map 迭代顺序回填 actual_uris，返回顺序与输入不一致，上层按契约回填
// 时会把 URI 写到错误位置，后续按返回 URI 读取将拿到错块数据。
TEST_F(TransferClientMultiStorageTest, TestSaveLoadInterleavedMultiPathOrdering) {
    auto client = TransferClient::Create(client_config_, init_params_);
    ASSERT_NE(client, nullptr);

    UriStrVec uri_str_vec = {
        "file://nfs_a/" + root_path_ + "tmp/nfs_a/path1?blkid=0&size=1024",
        "file://nfs_b/" + root_path_ + "tmp/nfs_b/pathX?blkid=0&size=1024",
        "file://nfs_a/" + root_path_ + "tmp/nfs_a/path2?blkid=0&size=1024",
    };

    const char *payload1 = "payload for nfs_a path1";
    const char *payloadX = "payload for nfs_b pathX";
    const char *payload2 = "payload for nfs_a path2";
    size_t len1 = strlen(payload1);
    size_t lenX = strlen(payloadX);
    size_t len2 = strlen(payload2);

    void *mem1 = malloc(1024);
    void *memX = malloc(1024);
    void *mem2 = malloc(1024);
    std::memcpy(mem1, payload1, len1);
    std::memcpy(memX, payloadX, lenX);
    std::memcpy(mem2, payload2, len2);

    auto make_buffer = [](void *base, size_t size) {
        BlockBuffer bb;
        Iov iov;
        iov.type = MemoryType::CPU;
        iov.base = base;
        iov.size = size;
        iov.ignore = false;
        bb.iovs.push_back(iov);
        return bb;
    };
    BlockBuffers block_buffers = {make_buffer(mem1, len1), make_buffer(memX, lenX), make_buffer(mem2, len2)};

    // Save：返回 URI 顺序必须与输入一致
    auto [save_ec, actual_uris] = client->SaveKvCaches(uri_str_vec, block_buffers);
    ASSERT_EQ(ER_OK, save_ec);
    ASSERT_EQ(uri_str_vec.size(), actual_uris.size());
    EXPECT_EQ(uri_str_vec[0], actual_uris[0]);
    EXPECT_EQ(uri_str_vec[1], actual_uris[1]);
    EXPECT_EQ(uri_str_vec[2], actual_uris[2]);

    // Load：用全新 buffer 按返回 URI 读回，验证数据确实落在各自的文件
    void *get1 = malloc(1024);
    void *getX = malloc(1024);
    void *get2 = malloc(1024);
    BlockBuffers get_buffers = {make_buffer(get1, len1), make_buffer(getX, lenX), make_buffer(get2, len2)};
    EXPECT_EQ(ER_OK, client->LoadKvCaches(actual_uris, get_buffers));
    EXPECT_EQ(std::memcmp(get1, payload1, len1), 0);
    EXPECT_EQ(std::memcmp(getX, payloadX, lenX), 0);
    EXPECT_EQ(std::memcmp(get2, payload2, len2), 0);

    free(mem1);
    free(memX);
    free(mem2);
    free(get1);
    free(getX);
    free(get2);
}

// Inline headers are not supported yet, so Init must reject that configuration.
TEST_F(TransferClientTest, TestCreateRejectsInlineHeader) {
    auto init_params = init_params_;
    init_params.regist_span = new RegistSpan();
    init_params.regist_span->base = malloc(1024 * 1024);
    init_params.regist_span->size = 1024 * 1024;
    init_params.storage_configs = R"([
        {
            "type": "file",
            "global_unique_name": "test_nfs",
            "storage_spec": {
                "root_path": "/tmp/test/",
                "key_count_per_file": 5
            },
            "integrity": {
                "enable_inline_header": true
            }
        }
    ])";
    auto client = TransferClient::Create(client_config_, init_params);
    EXPECT_EQ(client, nullptr);
    free(init_params.regist_span->base);
    delete init_params.regist_span;
}

// inline_header_version != 0 但开关没开 -> 同样被 Init 拒绝。
TEST_F(TransferClientTest, TestCreateRejectsOrphanInlineHeaderVersion) {
    auto init_params = init_params_;
    init_params.regist_span = new RegistSpan();
    init_params.regist_span->base = malloc(1024 * 1024);
    init_params.regist_span->size = 1024 * 1024;
    init_params.storage_configs = R"([
        {
            "type": "file",
            "global_unique_name": "test_nfs",
            "storage_spec": {
                "root_path": "/tmp/test/",
                "key_count_per_file": 5
            },
            "integrity": {
                "enable_inline_header": false,
                "inline_header_version": 1
            }
        }
    ])";
    auto client = TransferClient::Create(client_config_, init_params);
    EXPECT_EQ(client, nullptr);
    free(init_params.regist_span->base);
    delete init_params.regist_span;
}

// enable_meta_checksum=true 的 spec 能正常 Init (内部触发 hash pool 初始化或退化警告)。
TEST_F(TransferClientTest, TestCreateAcceptsMetaChecksumSpec) {
    auto init_params = init_params_;
    init_params.regist_span = new RegistSpan();
    init_params.regist_span->base = malloc(1024 * 1024);
    init_params.regist_span->size = 1024 * 1024;
    init_params.storage_configs = R"([
        {
            "type": "file",
            "global_unique_name": "test_nfs",
            "storage_spec": {
                "root_path": "/tmp/test/",
                "key_count_per_file": 5
            },
            "integrity": {
                "enable_meta_checksum": true,
                "algo": "crc32_xor_int64"
            }
        }
    ])";
    auto client = TransferClient::Create(client_config_, init_params);
    EXPECT_NE(client, nullptr);
    free(init_params.regist_span->base);
    delete init_params.regist_span;
}

TEST_F(TransferClientTest, TestCreateRejectsUnsupportedChecksumAlgo) {
    auto init_params = init_params_;
    init_params.regist_span = new RegistSpan();
    init_params.regist_span->base = malloc(1024 * 1024);
    init_params.regist_span->size = 1024 * 1024;
    init_params.storage_configs = R"([
        {
            "type": "file",
            "global_unique_name": "test_nfs",
            "storage_spec": {
                "root_path": "/tmp/test/",
                "key_count_per_file": 5
            },
            "integrity": {
                "enable_meta_checksum": true,
                "algo": "unknown_algo"
            }
        }
    ])";
    auto client = TransferClient::Create(client_config_, init_params);
    EXPECT_EQ(client, nullptr);
    free(init_params.regist_span->base);
    delete init_params.regist_span;
}

// 显式 presence=false 表示旧数据未携带 checksum；数值槽位即便为 0 也不会触发校验。
TEST_F(TransferClientTest, TestLoadKvCachesAbsentChecksumsSkipCheck) {
    auto client = TransferClient::Create(client_config_, init_params_);
    ASSERT_NE(client, nullptr);
    BlockBuffer buffer1, buffer2;
    BlockBuffers block_buffers = {buffer1, buffer2};
    std::vector<int64_t> expected_checksums = {0, 0};
    std::vector<bool> checksum_present = {false, false};
    EXPECT_EQ(ER_OK,
              client->LoadKvCaches(
                  locations_, block_buffers, LoadKvCachesOptions::VerifyWith(expected_checksums, checksum_present)));
}

// checksum=0 是合法值；显式请求校验时如果当前 client 没有可用计算能力，必须失败而非静默跳过。
TEST_F(TransferClientTest, TestLoadKvCachesPresentZeroDoesNotSkipCheck) {
    auto client = TransferClient::Create(client_config_, init_params_);
    ASSERT_NE(client, nullptr);
    BlockBuffers block_buffers = {BlockBuffer{}, BlockBuffer{}};
    EXPECT_EQ(ER_CHECKSUM_UNAVAILABLE,
              client->LoadKvCaches(locations_, block_buffers, LoadKvCachesOptions::VerifyWith({0, 0})));
}

// expected_checksums 长度与 block_buffers 不一致 -> ER_CHECKSUM_MISMATCH。
TEST_F(TransferClientTest, TestLoadKvCachesExpectedHashesSizeMismatchFails) {
    auto client = TransferClient::Create(client_config_, init_params_);
    ASSERT_NE(client, nullptr);
    BlockBuffer buffer1, buffer2;
    BlockBuffers block_buffers = {buffer1, buffer2};
    std::vector<int64_t> expected_checksums = {0}; // 长度 1，但 buffers 长度 2
    EXPECT_EQ(ER_CHECKSUM_MISMATCH,
              client->LoadKvCaches(locations_, block_buffers, LoadKvCachesOptions::VerifyWith(expected_checksums)));
}

// Options 字段公开以方便上层组合；即使调用方未显式置 verify_caller_checksums，
// 只要给了 expected_checksums 就不能静默忽略。在当前无 GPU checksum pool 的
// 测试配置下，应在 Put 之前明确返回 unavailable。
TEST_F(TransferClientTest, TestSaveKvCachesExpectedChecksumsImplicitlyRequestVerification) {
    auto client = TransferClient::Create(client_config_, init_params_);
    ASSERT_NE(client, nullptr);
    BlockBuffers block_buffers = {BlockBuffer{}, BlockBuffer{}};
    SaveKvCachesOptions options;
    options.expected_checksums = {11, 22};

    const auto result = client->SaveKvCaches(locations_, block_buffers, options);

    EXPECT_EQ(ER_CHECKSUM_UNAVAILABLE, result.first);
    EXPECT_TRUE(result.second.uri_str_vec.empty());
    EXPECT_TRUE(result.second.checksums.empty());
}

// SaveKvCaches 失败时 result.checksums 必须保持空 —— 老实现把计算完的 checksum 直接
// 返回给 caller，再让 caller 透传到 FinishWrite，就会给磁盘上不存在的数据落一条 hash。
// 此处触发 sdk_wrapper->Put 前置校验错误 (empty inputs)，覆盖两种 build 下的行为。
TEST_F(TransferClientTest, TestSaveKvCachesFailureLeavesChecksumsEmpty) {
    auto client = TransferClient::Create(client_config_, init_params_);
    ASSERT_NE(client, nullptr);
    UriStrVec empty_uris = {};
    BlockBuffers empty_buffers = {};
    auto result = client->SaveKvCaches(empty_uris, empty_buffers, SaveKvCachesOptions::WithChecksums());
    EXPECT_EQ(ER_INVALID_PARAMS, result.first);
    EXPECT_TRUE(result.second.checksums.empty()) << "checksums must stay empty on Save failure";
}
