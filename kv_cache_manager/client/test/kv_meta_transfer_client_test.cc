#include <cstdint>
#include <limits>
#include <memory>
#include <string>

#include "kv_cache_manager/client/src/kv_meta_transfer_client_impl.h"
#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/data_storage/kv_meta_uri.h"

namespace kv_cache_manager {
namespace {

TEST(KvMetaUriTest, RejectsAmbiguousAuthoritiesAndQueries) {
    EXPECT_TRUE(IsCanonicalKvMetaBackendName("pace.prod-a"));
    EXPECT_FALSE(IsCanonicalKvMetaBackendName(""));
    EXPECT_FALSE(IsCanonicalKvMetaBackendName("owner@pace"));
    EXPECT_FALSE(IsCanonicalKvMetaBackendName("pace:0"));
    EXPECT_FALSE(IsCanonicalKvMetaBackendName("pace name"));

    EXPECT_TRUE(HasUnambiguousKvMetaUriText("pace://pace/1?media_type=0&node_id=1&range_id=0&size=5"));
    EXPECT_FALSE(HasUnambiguousKvMetaUriText("pace://pace/1?media_type=0&node_id=1&range_id=0&size=5&size=5"));
    EXPECT_FALSE(HasUnambiguousKvMetaUriText("pace://owner@pace/1?size=5"));
    EXPECT_FALSE(HasUnambiguousKvMetaUriText("pace://pace:123/1?size=5"));

    EXPECT_TRUE(HasSameCanonicalKvMetaUri("pace://pace/1?media_type=0&node_id=1&range_id=0&size=5",
                                          "pace://pace/1?size=5&range_id=0&node_id=1&media_type=0"));
    EXPECT_FALSE(HasSameCanonicalKvMetaUri("pace://pace/1?media_type=0&node_id=1&range_id=0&size=5",
                                           "pace://pace/2?media_type=0&node_id=1&range_id=0&size=5"));
}

TEST(KvMetaUriTest, ValidatesExactPaceAddresses) {
    std::uint64_t offset = 0;
    const DataStorageUri valid("pace://pace/18446744073709551615?media_type=5&node_id=1&range_id=0&size=1");
    EXPECT_TRUE(TryGetExactTairMempoolOffset(valid, offset));
    EXPECT_EQ(std::numeric_limits<std::uint64_t>::max(), offset);
    EXPECT_TRUE(HasOwnedKvMetaAllocationShape(valid, DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL));

    for (const std::string &uri : {
             "pace://pace/0?media_type=5&node_id=1&range_id=0&size=1",
             "pace://pace/1?media_type=5&node_id=0&range_id=0&size=1",
             "pace://pace/1?media_type=05&node_id=1&range_id=0&size=1",
             "pace://pace/1?media_type=5&node_id=1&size=1",
             "pace://pace/not-a-number?media_type=5&node_id=1&range_id=0&size=1",
         }) {
        SCOPED_TRACE(uri);
        EXPECT_FALSE(HasExactTairMempoolAddress(DataStorageUri(uri)));
    }
}

TEST(KvMetaUriTest, EnforcesConfiguredPaceMediaPool) {
    auto automatic_spec = std::make_shared<TairMemPoolStorageSpec>();
    const StorageConfig automatic(DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL, "pace", automatic_spec);
    EXPECT_TRUE(HasSafeConfiguredKvMetaNamespace(automatic));
    EXPECT_TRUE(UriMatchesConfiguredKvMetaNamespace(
        DataStorageUri("pace://pace/1?media_type=0&node_id=1&range_id=0&size=5"), automatic.type(), automatic));
    EXPECT_TRUE(UriMatchesConfiguredKvMetaNamespace(
        DataStorageUri("pace://pace/1?media_type=5&node_id=1&range_id=0&size=5"), automatic.type(), automatic));

    auto ssd_spec = std::make_shared<TairMemPoolStorageSpec>();
    ssd_spec->set_media_type(kTairMemPoolMediaTypeSsd);
    const StorageConfig ssd(DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL_SSD, "pace_ssd", ssd_spec);
    EXPECT_TRUE(UriMatchesConfiguredKvMetaNamespace(
        DataStorageUri("pace://pace_ssd/1?media_type=5&node_id=1&range_id=0&size=5"), ssd.type(), ssd));
    EXPECT_FALSE(UriMatchesConfiguredKvMetaNamespace(
        DataStorageUri("pace://pace_ssd/1?media_type=2&node_id=1&range_id=0&size=5"), ssd.type(), ssd));

    EXPECT_FALSE(SupportsKvMetaAdmission(DataStorageType::DATA_STORAGE_TYPE_NFS));
    EXPECT_TRUE(SupportsKvMetaAdmission(DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL));
}

class KvMetaTransferConfigTest : public testing::Test {
protected:
    KvMetaTransferConfigTest() {
        init_params_.self_location_spec_name = "value";
        config_ = R"({
            "instance_group": "group",
            "instance_id": "instance",
            "block_size": 1,
            "sdk_config": {
                "thread_num": 1,
                "queue_size": 1,
                "sdk_backend_configs": [{"type": "pace"}],
                "timeout_config": {"get_timeout_ms": 1000, "put_timeout_ms": 1000}
            },
            "location_spec_infos": {"value": 1}
        })";
    }

    InitParams init_params_;
    std::string config_;
};

TEST_F(KvMetaTransferConfigTest, AcceptsExistingSmallSdkQueue) {
    const std::string group = "group";
    const std::string instance = "instance";
    EXPECT_EQ(ER_OK, ValidateKvMetaTransferClientConfig(config_, init_params_, &group, &instance));
}

TEST_F(KvMetaTransferConfigTest, RequiresTheIsolatedValueMarkerAndIdentity) {
    const std::string wrong_group = "other";
    EXPECT_EQ(ER_INVALID_CLIENT_CONFIG,
              ValidateKvMetaTransferClientConfig(config_, init_params_, &wrong_group, nullptr));

    InitParams wrong_spec = init_params_;
    wrong_spec.self_location_spec_name = "tp0";
    EXPECT_EQ(ER_INVALID_PARAMS, ValidateKvMetaTransferClientConfig(config_, wrong_spec));

    auto wrong_block = config_;
    const auto position = wrong_block.find("\"block_size\": 1");
    ASSERT_NE(std::string::npos, position);
    wrong_block.replace(position, std::string("\"block_size\": 1").size(), "\"block_size\": 2");
    EXPECT_EQ(ER_INVALID_CLIENT_CONFIG, ValidateKvMetaTransferClientConfig(wrong_block, init_params_));
}

} // namespace
} // namespace kv_cache_manager
