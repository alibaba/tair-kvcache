#include <cstdint>
#include <map>
#include <memory>
#include <string>

#include "kv_cache_manager/client/src/internal/config/client_config.h"
#include "kv_cache_manager/client/src/internal/sdk/sdk_factory.h"
#include "kv_cache_manager/client/src/internal/sdk/sdk_interface.h"
#include "kv_cache_manager/client/src/internal/sdk/sdk_wrapper.h"
#include "kv_cache_manager/client/src/kv_meta_transfer_client_impl.h"
#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/data_storage/kv_meta_uri.h"

namespace kv_cache_manager {
namespace {

TEST(KvMetaUriTest, ChecksPaceAndExactSize) {
    const std::string uri = "pace://pace/1?media_type=0&node_id=1&range_id=0&size=5";
    std::uint64_t size = 0;
    EXPECT_TRUE(IsValidKvMetaLocation(uri, DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL, size));
    EXPECT_EQ(5, size);
    EXPECT_FALSE(IsValidKvMetaLocation(uri, DataStorageType::DATA_STORAGE_TYPE_NFS, size));
    EXPECT_FALSE(IsValidKvMetaLocation("file://pace/1?size=5", DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL, size));
    EXPECT_FALSE(IsValidKvMetaLocation("pace://pace/not-a-number?media_type=0&node_id=1&range_id=0&size=5",
                                       DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL,
                                       size));
    EXPECT_FALSE(IsValidKvMetaLocation("pace://pace/1?media_type=0&node_id=1&range_id=0&size=5&size=6",
                                       DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL,
                                       size));
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

class RecordingSdk final : public SdkInterface {
public:
    ClientErrorCode Init(const std::shared_ptr<SdkBackendConfig> &config,
                         const std::shared_ptr<StorageConfig> &) override {
        config_ = config;
        return ER_OK;
    }

    SdkType Type() override { return SdkType::TAIR_MEMPOOL; }

    ClientErrorCode Get(const std::vector<DataStorageUri> &uris, const BlockBuffers &) override {
        loaded = ToStrings(uris);
        seen_sizes = config_->spec_byte_sizes_per_block();
        return load_result;
    }

    ClientErrorCode Put(const std::vector<DataStorageUri> &uris,
                        const BlockBuffers &,
                        std::shared_ptr<std::vector<DataStorageUri>> actual_uris) override {
        saved = ToStrings(uris);
        seen_sizes = config_->spec_byte_sizes_per_block();
        if (save_result == ER_OK) {
            *actual_uris = uris;
        }
        return save_result;
    }

    ClientErrorCode load_result = ER_OK;
    ClientErrorCode save_result = ER_OK;
    UriStrVec loaded;
    UriStrVec saved;
    std::map<std::string, std::int64_t> seen_sizes;

protected:
    ClientErrorCode Alloc(const std::vector<DataStorageUri> &, std::vector<DataStorageUri> &) override {
        return ER_SDKALLOC_ERROR;
    }

private:
    static UriStrVec ToStrings(const std::vector<DataStorageUri> &uris) {
        UriStrVec result;
        for (const auto &uri : uris) {
            result.push_back(uri.ToUriString());
        }
        return result;
    }

    std::shared_ptr<SdkBackendConfig> config_;
};

class RecordingSdkFactory final : public SdkFactory {
public:
    std::shared_ptr<SdkInterface> CreateSdk(const DataStorageType &,
                                            const std::shared_ptr<SdkBackendConfig> &config,
                                            const std::shared_ptr<StorageConfig> &storage) override {
        sdk = std::make_shared<RecordingSdk>();
        return sdk->Init(config, storage) == ER_OK ? sdk : nullptr;
    }

    std::shared_ptr<RecordingSdk> sdk;
};

std::unique_ptr<KvMetaTransferClientImpl>
MakeTransferClient(RecordingSdkFactory &factory, RecordingSdk *&recording, std::uint64_t max_object_bytes = 32) {
    auto config = std::make_unique<ClientConfig>();
    EXPECT_TRUE(config->FromJsonString(R"({
        "instance_group": "group",
        "instance_id": "instance",
        "block_size": 1,
        "sdk_config": {
            "thread_num": 1,
            "queue_size": 8,
            "sdk_backend_configs": [{"type": "pace"}],
            "timeout_config": {"get_timeout_ms": 1000, "put_timeout_ms": 1000}
        },
        "location_spec_infos": {"value": 1}
    })"));
    InitParams params;
    params.self_location_spec_name = "value";
    params.storage_configs = R"([{
        "type": "pace",
        "global_unique_name": "pace",
        "storage_spec": {"domain": "pace", "timeout": 1000}
    }])";
    auto wrapper = std::make_unique<SdkWrapper>();
    wrapper->sdk_factory_ = &factory;
    EXPECT_EQ(ER_OK, wrapper->Init(config, params));
    recording = factory.sdk.get();

    auto client = std::make_unique<KvMetaTransferClientImpl>();
    client->client_config_ = std::move(config);
    client->sdk_wrapper_ = std::move(wrapper);
    client->max_object_bytes_ = max_object_bytes;
    return client;
}

TEST(KvMetaTransferClientTest, ValidatesAndAppliesOnlyTheCurrentBatchSizes) {
    RecordingSdkFactory factory;
    RecordingSdk *recording = nullptr;
    auto client = MakeTransferClient(factory, recording);
    ASSERT_NE(nullptr, recording);

    char first[5]{};
    char second[9]{};
    const UriStrVec uris{
        "pace://pace/1?media_type=0&node_id=1&range_id=0&size=5",
        "pace://pace/2?media_type=0&node_id=1&range_id=0&size=9",
    };
    BlockBuffer first_buffer;
    first_buffer.iovs.push_back({MemoryType::CPU, first, sizeof(first), false});
    BlockBuffer second_buffer;
    second_buffer.iovs.push_back({MemoryType::CPU, second, sizeof(second), false});
    const BlockBuffers buffers{first_buffer, second_buffer};
    const std::vector<std::uint64_t> sizes{sizeof(first), sizeof(second)};

    EXPECT_EQ(ER_OK, client->LoadObjects(uris, sizes, buffers));
    EXPECT_EQ(uris, recording->loaded);
    EXPECT_EQ((std::map<std::string, std::int64_t>{{"5", 5}, {"9", 9}}), recording->seen_sizes);
    const auto [save_ec, actual] = client->SaveObjects(uris, sizes, buffers);
    EXPECT_EQ(ER_OK, save_ec);
    EXPECT_EQ(uris, actual);
    EXPECT_EQ(uris, recording->saved);
    EXPECT_EQ((std::map<std::string, std::int64_t>{{"5", 5}, {"9", 9}}), recording->seen_sizes);

    EXPECT_EQ(ER_INVALID_PARAMS, client->LoadObjects(uris, {5, 8}, buffers));
    auto short_buffers = buffers;
    short_buffers[1].iovs[0].size = 8;
    EXPECT_EQ(ER_INVALID_LOCAL_BUFFERS, client->LoadObjects(uris, sizes, short_buffers));
    EXPECT_EQ(ER_INVALID_PARAMS, client->LoadObjects({"file://pace/1?size=5", uris[1]}, sizes, buffers));
}

TEST(KvMetaTransferClientTest, PropagatesSdkErrors) {
    RecordingSdkFactory factory;
    RecordingSdk *recording = nullptr;
    auto client = MakeTransferClient(factory, recording);
    ASSERT_NE(nullptr, recording);
    recording->load_result = ER_SDKREAD_ERROR;
    recording->save_result = ER_SDKWRITE_ERROR;

    char value[5]{};
    const UriStrVec uris{"pace://pace/1?media_type=0&node_id=1&range_id=0&size=5"};
    BlockBuffer buffer;
    buffer.iovs.push_back({MemoryType::CPU, value, sizeof(value), false});
    const BlockBuffers buffers{buffer};

    EXPECT_EQ(ER_SDKREAD_ERROR, client->LoadObjects(uris, {sizeof(value)}, buffers));
    EXPECT_EQ(ER_SDKWRITE_ERROR, client->SaveObjects(uris, {sizeof(value)}, buffers).first);
}

} // namespace
} // namespace kv_cache_manager
