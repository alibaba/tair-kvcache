#include <cstring>
#include <filesystem>
#include <fstream>

#include "kv_cache_manager/client/include/manager_client.h"
#include "kv_cache_manager/client/src/transfer_client_impl.h"
#include "kv_cache_manager/common/unittest.h"

using namespace kv_cache_manager;

class RecordingTransferClient : public TransferClient {
public:
    using TransferClient::LoadKvCaches;
    using TransferClient::SaveKvCaches;

    ClientErrorCode LoadKvCaches(const UriStrVec &uri_str_vec,
                                 const BlockBuffers &block_buffers,
                                 const LoadKvCachesOptions &options) override {
        last_load_uri_count = uri_str_vec.size();
        last_load_buffer_count = block_buffers.size();
        last_load_options = options;
        return ER_OK;
    }

    std::pair<ClientErrorCode, UriStrVec> SaveKvCaches(const UriStrVec &uri_str_vec,
                                                       const BlockBuffers &block_buffers,
                                                       const SaveKvCachesOptions &options) override {
        last_save_uri_count = uri_str_vec.size();
        last_save_buffer_count = block_buffers.size();
        last_save_options = options;
        return {ER_OK, uri_str_vec};
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

    std::pair<ClientErrorCode, Locations> MatchLocation(const std::string &,
                                                        QueryType,
                                                        const std::vector<int64_t> &,
                                                        const std::vector<int64_t> &,
                                                        const BlockMask &,
                                                        int32_t,
                                                        const std::vector<std::string> &,
                                                        const MatchLocationOptions &options) override {
        last_match_location_options = options;
        return {ER_OK, Locations{}};
    }

    std::pair<ClientErrorCode, WriteLocation> StartWrite(const std::string &,
                                                         const std::vector<int64_t> &,
                                                         const std::vector<int64_t> &,
                                                         const std::vector<std::string> &,
                                                         int64_t) override {
        return {ER_OK, WriteLocation{}};
    }

    ClientErrorCode FinishWrite(const std::string &,
                                const std::string &,
                                const BlockMask &,
                                const Locations &,
                                const FinishWriteOptions &options) override {
        last_finish_write_options = options;
        return ER_OK;
    }

    std::pair<ClientErrorCode, Metas> MatchMeta(const std::string &,
                                                const std::vector<int64_t> &,
                                                const std::vector<int64_t> &,
                                                const BlockMask &,
                                                int32_t,
                                                const MatchMetaOptions &options) override {
        last_match_meta_options = options;
        return {ER_OK, Metas{}};
    }

    ClientErrorCode RemoveCache(const std::string &,
                                const std::vector<int64_t> &,
                                const std::vector<int64_t> &,
                                const BlockMask &) override {
        return ER_OK;
    }

    ClientErrorCode LoadKvCaches(const UriStrVec &,
                                 const BlockBuffers &,
                                 const LoadKvCachesOptions &options) override {
        last_load_options = options;
        return ER_OK;
    }

    std::pair<ClientErrorCode, UriStrVec> SaveKvCaches(const UriStrVec &,
                                                       const BlockBuffers &,
                                                       const SaveKvCachesOptions &options) override {
        last_save_options = options;
        return {ER_OK, UriStrVec{}};
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

TEST(ClientOptionsTest, TransferConvenienceOverloadsForwardOptions) {
    RecordingTransferClient client;
    UriStrVec uris = {"file://test_nfs/path?blkid=0&size=1"};
    BlockBuffers buffers = {BlockBuffer{}};

    EXPECT_EQ(ER_OK, client.LoadKvCaches(uris, buffers));
    EXPECT_EQ(uris.size(), client.last_load_uri_count);
    EXPECT_EQ(buffers.size(), client.last_load_buffer_count);
    EXPECT_EQ(nullptr, client.last_load_options.trace_info);
    EXPECT_EQ(nullptr, client.last_load_options.expected_checksums);

    auto trace_info = std::make_shared<TransferTraceInfo>();
    trace_info->need_print = true;
    EXPECT_EQ(ER_OK, client.LoadKvCaches(uris, buffers, trace_info));
    EXPECT_EQ(trace_info, client.last_load_options.trace_info);
    EXPECT_EQ(nullptr, client.last_load_options.expected_checksums);

    std::vector<int64_t> expected_checksums = {0x11};
    EXPECT_EQ(ER_OK,
              client.LoadKvCaches(uris, buffers, LoadKvCachesOptions::VerifyWith(expected_checksums, trace_info)));
    EXPECT_EQ(trace_info, client.last_load_options.trace_info);
    EXPECT_EQ(&expected_checksums, client.last_load_options.expected_checksums);

    auto save_result = client.SaveKvCaches(uris, buffers);
    EXPECT_EQ(ER_OK, save_result.first);
    EXPECT_EQ(uris.size(), client.last_save_uri_count);
    EXPECT_EQ(buffers.size(), client.last_save_buffer_count);
    EXPECT_EQ(nullptr, client.last_save_options.trace_info);
    EXPECT_EQ(nullptr, client.last_save_options.out_checksums);

    std::vector<int64_t> out_checksums;
    save_result = client.SaveKvCaches(uris, buffers, SaveKvCachesOptions::CollectChecksums(out_checksums, trace_info));
    EXPECT_EQ(ER_OK, save_result.first);
    EXPECT_EQ(trace_info, client.last_save_options.trace_info);
    EXPECT_EQ(&out_checksums, client.last_save_options.out_checksums);
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
              client.MatchLocation(
                        trace_id, QueryType::QT_PREFIX_MATCH, keys, tokens, block_mask, 0, spec_names)
                  .first);
    EXPECT_EQ(nullptr, client.last_match_location_options.out_checksums);

    std::vector<int64_t> match_location_checksums;
    EXPECT_EQ(ER_OK,
              client.MatchLocation(trace_id,
                                   QueryType::QT_PREFIX_MATCH,
                                   keys,
                                   tokens,
                                   block_mask,
                                   0,
                                   spec_names,
                                   MatchLocationOptions::CollectChecksums(match_location_checksums))
                  .first);
    EXPECT_EQ(&match_location_checksums, client.last_match_location_options.out_checksums);

    EXPECT_EQ(ER_OK, client.FinishWrite(trace_id, "session", block_mask, locations));
    EXPECT_EQ(nullptr, client.last_finish_write_options.checksums);

    std::vector<int64_t> finish_checksums = {0x21, 0};
    EXPECT_EQ(ER_OK,
              client.FinishWrite(
                  trace_id, "session", block_mask, locations, FinishWriteOptions::WithChecksums(finish_checksums)));
    EXPECT_EQ(&finish_checksums, client.last_finish_write_options.checksums);

    EXPECT_EQ(ER_OK, client.MatchMeta(trace_id, keys, tokens, block_mask, 1).first);
    EXPECT_EQ(nullptr, client.last_match_meta_options.out_checksums);

    std::vector<int64_t> match_meta_checksums;
    EXPECT_EQ(ER_OK,
              client.MatchMeta(
                        trace_id, keys, tokens, block_mask, 1, MatchMetaOptions::CollectChecksums(match_meta_checksums))
                  .first);
    EXPECT_EQ(&match_meta_checksums, client.last_match_meta_options.out_checksums);

    EXPECT_EQ(ER_OK, client.LoadKvCaches(uris, buffers));
    EXPECT_EQ(nullptr, client.last_load_options.expected_checksums);

    EXPECT_EQ(ER_OK, client.LoadKvCaches(uris, buffers, LoadKvCachesOptions::VerifyWith(finish_checksums)));
    EXPECT_EQ(&finish_checksums, client.last_load_options.expected_checksums);

    EXPECT_EQ(ER_OK, client.SaveKvCaches(uris, buffers).first);
    EXPECT_EQ(nullptr, client.last_save_options.out_checksums);

    EXPECT_EQ(ER_OK,
              client.SaveKvCaches(uris, buffers, SaveKvCachesOptions::CollectChecksums(finish_checksums)).first);
    EXPECT_EQ(&finish_checksums, client.last_save_options.out_checksums);
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

// 任务 82620492：方案 B (inline_header) 在本期未实现，Init 期间必须拒绝。
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

// expected_checksums 全 0 -> sentinel 跳过校验，行为 == 老路径，不会因 checksum 不匹配返回错误。
TEST_F(TransferClientTest, TestLoadKvCachesExpectedHashesAllZeroSkipsCheck) {
    auto client = TransferClient::Create(client_config_, init_params_);
    ASSERT_NE(client, nullptr);
    BlockBuffer buffer1, buffer2;
    BlockBuffers block_buffers = {buffer1, buffer2};
    std::vector<int64_t> expected_checksums = {0, 0};
    EXPECT_EQ(ER_OK,
              client->LoadKvCaches(locations_, block_buffers, LoadKvCachesOptions::VerifyWith(expected_checksums)));
}

// expected_checksums 长度与 block_buffers 不一致 -> ER_CHECKSUM_MISMATCH。
TEST_F(TransferClientTest, TestLoadKvCachesExpectedHashesSizeMismatchFails) {
    auto client = TransferClient::Create(client_config_, init_params_);
    ASSERT_NE(client, nullptr);
    BlockBuffer buffer1, buffer2;
    BlockBuffers block_buffers = {buffer1, buffer2};
    std::vector<int64_t> expected_checksums = {0}; // 长度 1，但 buffers 长度 2
#if defined(USING_CUDA) || defined(USING_MUSA)
    EXPECT_EQ(ER_CHECKSUM_MISMATCH,
              client->LoadKvCaches(locations_, block_buffers, LoadKvCachesOptions::VerifyWith(expected_checksums)));
#else
    // 非 CUDA/MUSA build：校验路径整体退化为 no-op，长度不匹配也不报错。
    EXPECT_EQ(ER_OK,
              client->LoadKvCaches(locations_, block_buffers, LoadKvCachesOptions::VerifyWith(expected_checksums)));
#endif
}

// SaveKvCaches 失败时 out_checksums 必须保持空 —— 老实现把计算完的 checksum 直接
// 交给 caller，再让 caller 透传到 FinishWrite，就会给磁盘上不存在的数据落一条 hash。
// 此处触发 sdk_wrapper->Put 前置校验错误 (empty inputs)，覆盖两种 build 下的行为。
TEST_F(TransferClientTest, TestSaveKvCachesFailureLeavesOutChecksumsEmpty) {
    auto client = TransferClient::Create(client_config_, init_params_);
    ASSERT_NE(client, nullptr);
    UriStrVec empty_uris = {};
    BlockBuffers empty_buffers = {};
    // 预先塞一个「上次遗留」的 checksum，确保 SaveKvCaches 在失败路径上清理它。
    std::vector<int64_t> out_checksums = {0xDEADBEEFLL};
    auto result =
        client->SaveKvCaches(empty_uris, empty_buffers, SaveKvCachesOptions::CollectChecksums(out_checksums));
    EXPECT_EQ(ER_INVALID_PARAMS, result.first);
    EXPECT_TRUE(out_checksums.empty()) << "out_checksums must be cleared on Save failure";
}
