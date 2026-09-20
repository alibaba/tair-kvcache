#include <cstring>
#include <filesystem>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "kv_cache_manager/client/include/kv_meta_transfer_client.h"
#include "kv_cache_manager/client/include/transfer_client.h"
#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/data_storage/kv_meta_uri.h"

namespace kv_cache_manager {
namespace {

TEST(KvMetaUriTest, EnforcesBoundedUnambiguousCanonicalIdentity) {
    EXPECT_TRUE(IsCanonicalKvMetaBackendName("nfs_01.prod-a"));
    EXPECT_FALSE(IsCanonicalKvMetaBackendName(""));
    EXPECT_FALSE(IsCanonicalKvMetaBackendName("owner@nfs"));
    EXPECT_FALSE(IsCanonicalKvMetaBackendName("nfs:0"));
    EXPECT_FALSE(IsCanonicalKvMetaBackendName("nfs name"));
    EXPECT_TRUE(IsCanonicalKvMetaBackendName(std::string(kMaxKvMetaBackendNameBytes, 'a')));
    EXPECT_FALSE(IsCanonicalKvMetaBackendName(std::string(kMaxKvMetaBackendNameBytes + 1, 'a')));

    std::string maximum_parameter_uri = "file://nfs/object?size=5";
    for (std::size_t i = 1; i < kMaxKvMetaLocationUriQueryParams; ++i) {
        maximum_parameter_uri += "&p" + std::to_string(i) + "=x";
    }
    EXPECT_TRUE(HasUnambiguousKvMetaUriText(maximum_parameter_uri));
    EXPECT_FALSE(HasUnambiguousKvMetaUriText(maximum_parameter_uri + "&overflow=x"));

    EXPECT_TRUE(HasSameCanonicalKvMetaUri("file://nfs/object?size=5&blkid=0", "file://nfs/object?blkid=0&size=5"));
    EXPECT_FALSE(HasSameCanonicalKvMetaUri("file://nfs/object?size=5", "file://nfs/object?size=5&size=5"));
    EXPECT_FALSE(HasUnambiguousKvMetaUriText("file://nfs/object?size=5&flag"));
    EXPECT_TRUE(HasUnambiguousKvMetaUriText("file://nfs/object?size=5&flag="));
    EXPECT_FALSE(HasSameCanonicalKvMetaUri("file://nfs/object?size=5&flag", "file://nfs/object?flag=&size=5"));
    EXPECT_FALSE(HasSameCanonicalKvMetaUri("file:///object?size=5", "file:///object?size=5"));
    EXPECT_FALSE(HasUnambiguousKvMetaUriText("file://user@nfs/object?size=5"));
    EXPECT_FALSE(HasUnambiguousKvMetaUriText("file://nfs:0/object?size=5"));
    EXPECT_FALSE(HasUnambiguousKvMetaUriText("file://nfs:123/object?size=5"));
    EXPECT_FALSE(HasSameCanonicalKvMetaUri("file://nfs/" + std::string(kMaxKvMetaLocationUriBytes, 'x'),
                                           "file://nfs/" + std::string(kMaxKvMetaLocationUriBytes, 'x')));

    EXPECT_TRUE(HasOwnedKvMetaFilePath(DataStorageUri("file://nfs/object?size=5")));
    EXPECT_TRUE(HasOwnedKvMetaFilePath(DataStorageUri("file://nfs/dir/object?size=5")));
    EXPECT_FALSE(HasOwnedKvMetaFilePath(DataStorageUri("file://nfs?size=5")));
    EXPECT_FALSE(HasOwnedKvMetaFilePath(DataStorageUri("file://nfs/?size=5")));
    EXPECT_FALSE(HasOwnedKvMetaFilePath(DataStorageUri("file://nfs//object?size=5")));
    EXPECT_FALSE(HasOwnedKvMetaFilePath(DataStorageUri("file://nfs/dir//object?size=5")));
    EXPECT_FALSE(HasOwnedKvMetaFilePath(DataStorageUri("file://nfs/./object?size=5")));
    EXPECT_FALSE(HasOwnedKvMetaFilePath(DataStorageUri("file://nfs/dir/../object?size=5")));
    EXPECT_FALSE(HasOwnedKvMetaFilePath(DataStorageUri("file://nfs/dir/object/?size=5")));
    EXPECT_TRUE(HasOwnedKvMetaAllocationShape(DataStorageUri("file://nfs/object?blkid=0&size=5"),
                                              DataStorageType::DATA_STORAGE_TYPE_NFS));
    EXPECT_FALSE(HasOwnedKvMetaAllocationShape(DataStorageUri("file://nfs/object?blkid=1&size=5"),
                                               DataStorageType::DATA_STORAGE_TYPE_NFS));
    EXPECT_FALSE(HasOwnedKvMetaAllocationShape(DataStorageUri("event_report_l1p5://nfs/object?size=5"),
                                               DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L1P5));

    const std::string nonce = "0123456789abcdefghijklmnopqrstuv";
    EXPECT_TRUE(HasCanonicalKvMetaObjectKey("kvmeta/a/b/" + nonce));
    EXPECT_TRUE(HasCanonicalKvMetaObjectKey("kvmeta/0/0/" + nonce));
    EXPECT_FALSE(HasCanonicalKvMetaObjectKey("kvmeta/01/b/" + nonce));
    EXPECT_FALSE(HasCanonicalKvMetaObjectKey("kvmeta/a/0b/" + nonce));
    EXPECT_FALSE(HasCanonicalKvMetaObjectKey("kvmeta/A/b/" + nonce));
    EXPECT_FALSE(HasCanonicalKvMetaObjectKey("kvmeta/a/b/short"));
    EXPECT_FALSE(HasCanonicalKvMetaObjectKey("kvmeta/a/b/" + nonce + "/extra"));

    std::string_view object_key;
    const std::string canonical_path = "/cache/root/kvmeta/a/b/" + nonce;
    EXPECT_TRUE(TryGetCanonicalKvMetaObjectKeyFromPath(canonical_path, object_key));
    EXPECT_EQ("kvmeta/a/b/" + nonce, object_key);
    const std::string malformed_path = "/cache/root/kvmeta/a/not-hex/" + nonce;
    EXPECT_FALSE(TryGetCanonicalKvMetaObjectKeyFromPath(malformed_path, object_key));
    EXPECT_TRUE(object_key.empty());
}

TEST(KvMetaUriTest, ParsesTairMempoolOffsetWithoutAliasingMalformedPathsToZero) {
    std::uint64_t offset = 99;
    EXPECT_TRUE(TryGetExactTairMempoolOffset(DataStorageUri("pace://pace/0?size=1"), offset));
    EXPECT_EQ(0, offset);
    EXPECT_TRUE(TryGetExactTairMempoolOffset(DataStorageUri("pace://pace/18446744073709551615?size=1"), offset));
    EXPECT_EQ(std::numeric_limits<std::uint64_t>::max(), offset);
    EXPECT_TRUE(
        HasExactTairMempoolAddress(DataStorageUri("pace://pace/0?media_type=65535&node_id=0&range_id=1&size=1")));
    EXPECT_TRUE(HasExactTairMempoolAddress(DataStorageUri("pace://pace/0?size=1")));

    constexpr const char *kIncarnation = "01234567-89ab-4def-8abc-0123456789ab";
    const std::string allocation_token = "kvmeta/a/b/0123456789abcdefghijklmnopqrstuv";
    EXPECT_TRUE(HasCanonicalTairMempoolProviderIncarnation(
        DataStorageUri(std::string("pace://pace/1?provider_incarnation=") + kIncarnation + "&size=1")));
    EXPECT_TRUE(HasOwnedKvMetaAllocationShape(
        DataStorageUri(std::string("pace://pace/1?allocation_token=") + allocation_token +
                       "&provider_incarnation=" + kIncarnation + "&provider_uuid=stable-provider&size=1"),
        DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL));
    EXPECT_TRUE(
        HasSafeOptionalTairMempoolProviderUuid(DataStorageUri("pace://pace/1?provider_uuid=stable-provider&size=1")));
    // Missing is accepted only for rolling-upgrade compatibility with old
    // persisted exact URIs. New TairMempool allocations require the field in
    // the internal adapter.
    EXPECT_TRUE(HasSafeOptionalTairMempoolProviderUuid(DataStorageUri("pace://pace/1?size=1")));
    EXPECT_FALSE(HasSafeOptionalTairMempoolProviderUuid(DataStorageUri("pace://pace/1?provider_uuid=&size=1")));
    EXPECT_FALSE(
        HasSafeOptionalTairMempoolProviderUuid(DataStorageUri("pace://pace/1?provider_uuid=bad#route&size=1")));
    EXPECT_FALSE(
        HasSafeOptionalTairMempoolProviderUuid(DataStorageUri("pace://pace/1?provider_uuid=bad%25route&size=1")));
    EXPECT_FALSE(
        HasSafeOptionalTairMempoolProviderUuid(DataStorageUri("pace://pace/1?provider_uuid=bad%20route&size=1")));
    EXPECT_FALSE(
        HasSafeOptionalTairMempoolProviderUuid(DataStorageUri("pace://pace/1?provider_uuid=bad=route&size=1")));
    EXPECT_FALSE(HasSafeOptionalTairMempoolProviderUuid(
        DataStorageUri("pace://pace/1?provider_uuid=" + std::string(64, 'a') + "&size=1")));
    EXPECT_FALSE(HasOwnedKvMetaAllocationShape(
        DataStorageUri(std::string("pace://pace/1?provider_incarnation=") + kIncarnation + "&size=1"),
        DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL));
    EXPECT_FALSE(HasOwnedKvMetaAllocationShape(DataStorageUri("pace://pace/1?size=1"),
                                               DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL));

    for (const std::string &incarnation : {
             "01234567-89AB-4DEF-8ABC-0123456789AB",
             "0123456789ab-4def-8abc-0123456789ab",
             "01234567-89ab-4def-8abc-0123456789ag",
             "01234567-89ab-4def-8abc-0123456789ab-extra",
         }) {
        SCOPED_TRACE(incarnation);
        EXPECT_FALSE(HasCanonicalTairMempoolProviderIncarnation(
            DataStorageUri("pace://pace/1?provider_incarnation=" + incarnation + "&size=1")));
    }

    for (const std::string &uri : {
             "pace://pace/0?node_id=&size=1",
             "pace://pace/0?node_id=-1&size=1",
             "pace://pace/0?node_id=65536&size=1",
             "pace://pace/0?media_type=1x&size=1",
             "pace://pace/0?range_id=+1&size=1",
         }) {
        SCOPED_TRACE(uri);
        EXPECT_FALSE(HasExactTairMempoolAddress(DataStorageUri(uri)));
    }

    for (const std::string &uri : {
             "pace://pace?size=1",
             "pace://pace/?size=1",
             "pace://pace/-1?size=1",
             "pace://pace/+1?size=1",
             "pace://pace/not-a-number?size=1",
             "pace://pace/12trailing?size=1",
             "pace://pace/18446744073709551616?size=1",
         }) {
        SCOPED_TRACE(uri);
        offset = 99;
        EXPECT_FALSE(TryGetExactTairMempoolOffset(DataStorageUri(uri), offset));
        EXPECT_EQ(99, offset);
    }
}

BlockBuffer MakeBuffer(void *base, std::size_t size) {
    BlockBuffer buffer;
    buffer.iovs.push_back({MemoryType::CPU, base, size, false});
    return buffer;
}

class KvMetaTransferClientTest : public TESTBASE {
protected:
    void SetUp() override {
        root_path_ = GetPrivateTestRuntimeDataPath() + "kvmeta_objects/";
        std::filesystem::create_directories(root_path_);
        client_config_ = R"({
            "instance_group": "test_group",
            "instance_id": "test_instance",
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
            "location_spec_infos": {
                "value": 1
            }
        })";
        init_params_.role_type = RoleType::WORKER;
        init_params_.self_location_spec_name = "value";
        init_params_.storage_configs = R"([
            {
                "type": "file",
                "global_unique_name": "test_nfs",
                "storage_spec": {
                    "root_path": ")" + root_path_ +
                                       R"(",
                    "key_count_per_file": 1
                }
            }
        ])";
    }

    std::string ObjectPath(const std::string &key_hash, char nonce) const {
        return root_path_ + "kvmeta/123456789abcdef/" + key_hash + "/" + std::string(32, nonce);
    }

    std::string root_path_;
    std::string client_config_;
    InitParams init_params_;
};

TEST_F(KvMetaTransferClientTest, SavesAndLoadsDifferentObjectSizesInOneBatch) {
    auto client = KvMetaTransferClient::Create(client_config_, init_params_, 1024);
    ASSERT_NE(nullptr, client);

    std::vector<char> first{1, 2, 3, 4, 5};
    std::vector<char> second{9, 8, 7, 6, 5, 4, 3, 2, 1};
    const std::string first_path = ObjectPath("1", 'a');
    const std::string second_path = ObjectPath("2", 'b');
    const UriStrVec uris = {
        "file://test_nfs" + first_path + "?size=5&blkid=0",
        "file://test_nfs" + second_path + "?blkid=0&size=9",
    };
    const std::vector<std::uint64_t> sizes = {first.size(), second.size()};
    const BlockBuffers source = {
        MakeBuffer(first.data(), first.size()),
        MakeBuffer(second.data(), second.size()),
    };

    auto [save_ec, actual_uris] = client->SaveObjects(uris, sizes, source);
    ASSERT_EQ(ER_OK, save_ec);
    EXPECT_EQ((UriStrVec{
                  "file://test_nfs" + first_path + "?blkid=0&size=5",
                  "file://test_nfs" + second_path + "?blkid=0&size=9",
              }),
              actual_uris);
    ASSERT_TRUE(std::filesystem::exists(first_path));
    ASSERT_TRUE(std::filesystem::exists(second_path));
    EXPECT_EQ(first.size(), std::filesystem::file_size(first_path));
    EXPECT_EQ(second.size(), std::filesystem::file_size(second_path));

    std::vector<char> loaded_first(first.size());
    std::vector<char> loaded_second(second.size());
    const BlockBuffers destination = {
        MakeBuffer(loaded_first.data(), loaded_first.size()),
        MakeBuffer(loaded_second.data(), loaded_second.size()),
    };
    ASSERT_EQ(ER_OK, client->LoadObjects(uris, sizes, destination));
    EXPECT_EQ(first, loaded_first);
    EXPECT_EQ(second, loaded_second);
}

TEST_F(KvMetaTransferClientTest, RejectsUriAndBufferSizeMismatchBeforeIo) {
    auto client = KvMetaTransferClient::Create(client_config_, init_params_, 1024);
    ASSERT_NE(nullptr, client);
    std::vector<char> payload(5, 1);
    const std::string path = ObjectPath("3", 'c');
    const UriStrVec uris = {"file://test_nfs" + path + "?blkid=0&size=4"};

    auto [ec, actual_uris] = client->SaveObjects(uris, {payload.size()}, {MakeBuffer(payload.data(), payload.size())});
    EXPECT_EQ(ER_INVALID_PARAMS, ec);
    EXPECT_TRUE(actual_uris.empty());
    EXPECT_FALSE(std::filesystem::exists(path));
}

TEST_F(KvMetaTransferClientTest, RejectsStorageWithoutExactObjectOwnership) {
    init_params_.storage_configs = R"([
        {
            "type": "event_report_l1p5",
            "global_unique_name": "external_reporter",
            "storage_spec": {}
        }
    ])";
    EXPECT_EQ(nullptr, KvMetaTransferClient::Create(client_config_, init_params_, 1024));

    init_params_.storage_configs = R"([
        {
            "type": "file",
            "global_unique_name": "duplicate_nfs",
            "storage_spec": {"root_path": "/tmp/first/", "key_count_per_file": 1}
        },
        {
            "type": "file",
            "global_unique_name": "duplicate_nfs",
            "storage_spec": {"root_path": "/tmp/second/", "key_count_per_file": 1}
        }
    ])";
    EXPECT_EQ(nullptr, KvMetaTransferClient::Create(client_config_, init_params_, 1024));

    for (const std::string &unsafe_root : {"relative/", "/tmp/no-object-separator", "/tmp/cache/../unsafe/"}) {
        SCOPED_TRACE(unsafe_root);
        init_params_.storage_configs = R"([
            {
                "type": "file",
                "global_unique_name": "unsafe_nfs",
                "storage_spec": {"root_path": ")" +
                                       unsafe_root + R"(", "key_count_per_file": 1}
            }
        ])";
        EXPECT_EQ(nullptr, KvMetaTransferClient::Create(client_config_, init_params_, 1024));
    }
}

TEST_F(KvMetaTransferClientTest, RejectsSchemeMismatchAndNonSingletonBlockIdBeforeIo) {
    auto client = KvMetaTransferClient::Create(client_config_, init_params_, 1024);
    ASSERT_NE(nullptr, client);
    std::vector<char> payload(5, 1);
    auto buffer = MakeBuffer(payload.data(), payload.size());
    const std::string path = ObjectPath("4", 'd');

    const UriStrVec wrong_scheme = {"mooncake://test_nfs" + path + "?blkid=0&size=5"};
    const auto [scheme_save_ec, scheme_actual_uris] = client->SaveObjects(wrong_scheme, {payload.size()}, {buffer});
    EXPECT_EQ(ER_INVALID_PARAMS, scheme_save_ec);
    EXPECT_TRUE(scheme_actual_uris.empty());
    EXPECT_EQ(ER_INVALID_PARAMS, client->LoadObjects(wrong_scheme, {payload.size()}, {buffer}));

    for (const std::string &block_id : {"1", "-1", "not-a-number", "18446744073709551616"}) {
        SCOPED_TRACE(block_id);
        const UriStrVec non_singleton = {"file://test_nfs" + path + "?blkid=" + block_id + "&size=5"};
        const auto [save_ec, actual_uris] = client->SaveObjects(non_singleton, {payload.size()}, {buffer});
        EXPECT_EQ(ER_INVALID_PARAMS, save_ec);
        EXPECT_TRUE(actual_uris.empty());
        EXPECT_EQ(ER_INVALID_PARAMS, client->LoadObjects(non_singleton, {payload.size()}, {buffer}));
    }
    EXPECT_FALSE(std::filesystem::exists(path));
}

TEST_F(KvMetaTransferClientTest, RejectsAmbiguousRawUriSyntaxBeforeIo) {
    auto client = KvMetaTransferClient::Create(client_config_, init_params_, 1024);
    ASSERT_NE(nullptr, client);
    std::vector<char> payload(5, 1);
    const auto buffer = MakeBuffer(payload.data(), payload.size());
    const std::string path = ObjectPath("5", 'e');
    UriStrVec ambiguous_uris = {
        "file://test_nfs" + path + "?blkid=0&size=1&size=5",
        "file://test_nfs" + path + "?blkid=1&blkid=0&size=5",
        "file://test_nfs" + path + "?blkid=0&size=5&token=value#fragment",
        "file://test_nfs" + path + "?blkid=0&=empty-key&size=5",
        "file://test_nfs" + path + "?blkid=0&&size=5",
        "file://test_nfs" + path + "?blkid=0&size=5&",
        "file://test_nfs" + path + "?blkid=0&token=bad\nvalue&size=5",
        "file://test_nfs/" + std::string(kMaxKvMetaLocationUriBytes, 'x') + "?blkid=0&size=5",
        "file://test_nfs/" + std::string("bad\0path", 8) + "?blkid=0&size=5",
    };
    std::string too_many_parameters = "file://test_nfs" + path + "?size=5";
    for (std::size_t i = 0; i < kMaxKvMetaLocationUriQueryParams; ++i) {
        too_many_parameters += "&p" + std::to_string(i) + "=x";
    }
    ambiguous_uris.push_back(std::move(too_many_parameters));

    for (const auto &uri : ambiguous_uris) {
        SCOPED_TRACE(uri);
        const auto [save_ec, actual_uris] = client->SaveObjects({uri}, {payload.size()}, {buffer});
        EXPECT_EQ(ER_INVALID_PARAMS, save_ec);
        EXPECT_TRUE(actual_uris.empty());
        EXPECT_EQ(ER_INVALID_PARAMS, client->LoadObjects({uri}, {payload.size()}, {buffer}));
    }
    EXPECT_FALSE(std::filesystem::exists(path));
}

TEST_F(KvMetaTransferClientTest, RejectsIgnoredOrOversizedObjects) {
    auto client = KvMetaTransferClient::Create(client_config_, init_params_, 8);
    ASSERT_NE(nullptr, client);
    std::vector<char> payload(9, 1);
    const std::string oversized_path = ObjectPath("6", 'f');
    const UriStrVec uris = {"file://test_nfs" + oversized_path + "?blkid=0&size=9"};
    auto buffer = MakeBuffer(payload.data(), payload.size());
    EXPECT_EQ(ER_INVALID_PARAMS, client->LoadObjects(uris, {payload.size()}, {buffer}));

    payload.resize(5);
    buffer = MakeBuffer(payload.data(), payload.size());
    buffer.iovs[0].ignore = true;
    const std::string ignored_path = ObjectPath("7", 'g');
    const UriStrVec ignored_uri = {"file://test_nfs" + ignored_path + "?blkid=0&size=5"};
    EXPECT_EQ(ER_INVALID_LOCAL_BUFFERS, client->LoadObjects(ignored_uri, {payload.size()}, {buffer}));

    auto overflowing_buffer = MakeBuffer(
        reinterpret_cast<void *>(std::numeric_limits<std::uintptr_t>::max() - payload.size() + 1), payload.size());
    EXPECT_EQ(ER_INVALID_LOCAL_BUFFERS, client->LoadObjects(ignored_uri, {payload.size()}, {overflowing_buffer}));
}

TEST_F(KvMetaTransferClientTest, RejectsConfiguredNamespaceEscapeBeforeIo) {
    auto client = KvMetaTransferClient::Create(client_config_, init_params_, 1024);
    ASSERT_NE(nullptr, client);

    std::vector<char> payload(5, 1);
    const auto buffer = MakeBuffer(payload.data(), payload.size());
    const std::string foreign_path =
        GetPrivateTestRuntimeDataPath() + "foreign_objects/kvmeta/123456789abcdef/8/" + std::string(32, 'h');
    const UriStrVec foreign_uri = {"file://test_nfs" + foreign_path + "?blkid=0&size=5"};

    const auto [save_ec, actual_uris] = client->SaveObjects(foreign_uri, {payload.size()}, {buffer});
    EXPECT_EQ(ER_INVALID_PARAMS, save_ec);
    EXPECT_TRUE(actual_uris.empty());
    EXPECT_EQ(ER_INVALID_PARAMS, client->LoadObjects(foreign_uri, {payload.size()}, {buffer}));
    EXPECT_FALSE(std::filesystem::exists(foreign_path));
}

TEST_F(KvMetaTransferClientTest, RejectsMaxObjectSizeAboveTheServiceContract) {
    EXPECT_EQ(nullptr, KvMetaTransferClient::Create(client_config_, init_params_, 0));
    EXPECT_EQ(nullptr, KvMetaTransferClient::Create(client_config_, init_params_, 1ULL * 1024 * 1024 * 1024 + 1));
}

TEST_F(KvMetaTransferClientTest, RejectsMalformedConstructionInputsBeforeBackendInitialization) {
    EXPECT_EQ(nullptr, KvMetaTransferClient::Create("", init_params_, 1024));
    EXPECT_EQ(nullptr, KvMetaTransferClient::Create("{invalid-json", init_params_, 1024));

    auto invalid_init_params = init_params_;
    invalid_init_params.role_type = RoleType::SCHEDULER;
    EXPECT_EQ(nullptr, KvMetaTransferClient::Create(client_config_, invalid_init_params, 1024));

    invalid_init_params = init_params_;
    invalid_init_params.self_location_spec_name = "tp0";
    EXPECT_EQ(nullptr, KvMetaTransferClient::Create(client_config_, invalid_init_params, 1024));

    invalid_init_params = init_params_;
    invalid_init_params.storage_configs.clear();
    EXPECT_EQ(nullptr, KvMetaTransferClient::Create(client_config_, invalid_init_params, 1024));
}

TEST_F(KvMetaTransferClientTest, RejectsServiceBatchLimitsBeforeDataPlaneIo) {
    auto client = KvMetaTransferClient::Create(client_config_, init_params_, 1ULL * 1024 * 1024 * 1024);
    ASSERT_NE(nullptr, client);

    char payload = 0;
    UriStrVec too_many_uris;
    std::vector<std::uint64_t> too_many_sizes;
    BlockBuffers too_many_buffers;
    for (std::size_t i = 0; i < 65; ++i) {
        too_many_uris.push_back("file://test_nfs" + ObjectPath(std::to_string(i + 16), 'i') + "?blkid=0&size=1");
        too_many_sizes.push_back(1);
        too_many_buffers.push_back(MakeBuffer(&payload, 1));
    }
    EXPECT_EQ(ER_INVALID_PARAMS, client->LoadObjects(too_many_uris, too_many_sizes, too_many_buffers));

    constexpr std::size_t kOneGiB = 1ULL * 1024 * 1024 * 1024;
    UriStrVec oversized_batch_uris;
    std::vector<std::uint64_t> oversized_batch_sizes;
    BlockBuffers oversized_batch_buffers;
    for (std::size_t i = 0; i < 5; ++i) {
        oversized_batch_uris.push_back("file://test_nfs" + ObjectPath(std::to_string(i + 96), 'j') +
                                       "?blkid=0&size=" + std::to_string(kOneGiB));
        oversized_batch_sizes.push_back(kOneGiB);
        oversized_batch_buffers.push_back(MakeBuffer(&payload, kOneGiB));
    }
    const auto [save_ec, actual_uris] =
        client->SaveObjects(oversized_batch_uris, oversized_batch_sizes, oversized_batch_buffers);
    EXPECT_EQ(ER_INVALID_PARAMS, save_ec);
    EXPECT_TRUE(actual_uris.empty());
    for (std::size_t i = 0; i < 5; ++i) {
        EXPECT_FALSE(std::filesystem::exists(ObjectPath(std::to_string(i + 96), 'j')));
    }
}

TEST_F(KvMetaTransferClientTest, RejectsConfigsThatDoNotUseTheExactKvMetaMarker) {
    const auto replace_once = [](std::string value, const std::string &from, const std::string &to) {
        const auto position = value.find(from);
        EXPECT_NE(std::string::npos, position);
        if (position != std::string::npos) {
            value.replace(position, from.size(), to);
        }
        return value;
    };

    const auto wrong_marker_size = replace_once(client_config_, R"("value": 1)", R"("value": 2)");
    EXPECT_EQ(nullptr, KvMetaTransferClient::Create(wrong_marker_size, init_params_, 1024));

    const auto extra_location_spec = replace_once(client_config_, R"("value": 1)", R"("value": 1, "other": 1)");
    EXPECT_EQ(nullptr, KvMetaTransferClient::Create(extra_location_spec, init_params_, 1024));

    const auto wrong_block_size = replace_once(client_config_, R"("block_size": 1)", R"("block_size": 2)");
    EXPECT_EQ(nullptr, KvMetaTransferClient::Create(wrong_block_size, init_params_, 1024));

    const auto grouped_location_spec = replace_once(client_config_,
                                                    R"("value": 1)",
                                                    R"("value": 1
            },
            "location_spec_groups": {
                "value_group": ["value"])");
    EXPECT_EQ(nullptr, KvMetaTransferClient::Create(grouped_location_spec, init_params_, 1024));
}

TEST_F(KvMetaTransferClientTest, RegularTransferClientKeepsFixedSizePolicy) {
    auto client = TransferClient::Create(client_config_, init_params_);
    ASSERT_NE(nullptr, client);
    std::vector<char> payload(5, 1);
    const std::string path = root_path_ + "regular_reject";
    const UriStrVec uris = {"file://test_nfs" + path + "?blkid=0&size=5"};
    const auto result = client->SaveKvCaches(uris, {MakeBuffer(payload.data(), payload.size())});
    EXPECT_NE(ER_OK, result.first);
}

TEST_F(KvMetaTransferClientTest, RejectsQueueSmallerThanOneServiceBatchOnlyForKvMeta) {
    const auto position = client_config_.find(R"("queue_size": 64)");
    ASSERT_NE(std::string::npos, position);
    auto undersized_queue = client_config_;
    undersized_queue.replace(position, std::string(R"("queue_size": 64)").size(), R"("queue_size": 63)");

    EXPECT_EQ(nullptr, KvMetaTransferClient::Create(undersized_queue, init_params_, 1024));

    // The queue-capacity contract belongs to the new exact-object API. Keep
    // the established fixed-block TransferClient configuration path intact.
    EXPECT_NE(nullptr, TransferClient::Create(undersized_queue, init_params_));
}

} // namespace
} // namespace kv_cache_manager
