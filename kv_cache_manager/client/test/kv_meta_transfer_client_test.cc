#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>

#include "kv_cache_manager/client/src/internal/config/client_config.h"
#include "kv_cache_manager/client/src/internal/sdk/sdk_factory.h"
#include "kv_cache_manager/client/src/internal/sdk/sdk_interface.h"
#include "kv_cache_manager/client/src/internal/sdk/sdk_wrapper.h"
#include "kv_cache_manager/client/src/kv_meta_object_client_impl.h"
#include "kv_cache_manager/client/src/kv_meta_transfer_client_impl.h"
#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/data_storage/kv_meta_uri.h"

namespace kv_cache_manager {
namespace {

TEST(KvMetaUriTest, ChecksPaceAndExactSize) {
    const std::string uri = "pace://pace/1?media_type=0&node_id=1&range_id=0&size=5";
    std::uint64_t size = 0;
    EXPECT_TRUE(IsValidKvMetaLocation(uri, size));
    EXPECT_EQ(5, size);
    EXPECT_TRUE(IsValidKvMetaLocation("pace://pace/0?media_type=0&node_id=0&range_id=0&size=5", size));
    EXPECT_FALSE(IsValidKvMetaLocation("file://pace/1?size=5", size));
    EXPECT_FALSE(IsValidKvMetaLocation("pace://pace/not-a-number?media_type=0&node_id=1&range_id=0&size=5", size));
    EXPECT_FALSE(IsValidKvMetaLocation("pace://pace/1?media_type=0&node_id=1&range_id=0&size=5&size=6", size));
}

TEST(KvMetaUriTest, AcceptsBackendOmittedZeroAddressComponents) {
    std::uint64_t size = 0;
    // The PACE backend omits media_type/range_id when their value is zero.
    EXPECT_TRUE(IsValidKvMetaLocation("pace://pace/1?media_type=2&node_id=1&size=65536", size));
    EXPECT_EQ(65536, size);
    EXPECT_TRUE(IsValidKvMetaLocation("pace://pace/1?node_id=1&size=65536", size));
    EXPECT_FALSE(IsValidKvMetaLocation("pace://pace/1?node_id=1&range_id=bad&size=65536", size));
    EXPECT_FALSE(IsValidKvMetaLocation("pace://pace/1?media_type=65536&node_id=1&size=65536", size));
    EXPECT_FALSE(IsValidKvMetaLocation("pace://pace/1?size=65536", size));
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

TEST_F(KvMetaTransferConfigTest, RejectsBadGpuRegistrationInStandaloneTransferClient) {
    init_params_.role_type = RoleType::WORKER;
    init_params_.storage_configs = "[]";
    ClientMemoryRegistrations registrations;
    registrations.gpu = {{nullptr, 4096, 0}};
    KvMetaTransferClientImpl client;
    EXPECT_EQ(ER_INVALID_PARAMS, client.Init(config_, init_params_, 4096, &registrations));
    registrations.gpu = {{reinterpret_cast<void *>(0x1000), 4096, 0}, {reinterpret_cast<void *>(0x1800), 4096, 0}};
    EXPECT_EQ(ER_INVALID_PARAMS, client.Init(config_, init_params_, 4096, &registrations));
}

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
        const auto pace = std::dynamic_pointer_cast<TairMempoolSdkConfig>(config);
        if (pace) {
            registered_base = pace->client_base();
            registered_size = pace->shm_size();
        }
        return ER_OK;
    }

    SdkType Type() override { return SdkType::TAIR_MEMPOOL; }

    ClientErrorCode Get(const std::vector<DataStorageUri> &uris, const BlockBuffers &) override {
        if (before_io) {
            before_io(uris);
        }
        std::this_thread::sleep_for(delay);
        std::lock_guard<std::mutex> lock(records_mutex_);
        loaded = ToStrings(uris);
        seen_sizes = config_->spec_byte_sizes_per_block();
        return result_for ? result_for(uris) : load_result;
    }

    ClientErrorCode Put(const std::vector<DataStorageUri> &uris,
                        const BlockBuffers &,
                        std::shared_ptr<std::vector<DataStorageUri>> actual_uris) override {
        if (before_io) {
            before_io(uris);
        }
        std::this_thread::sleep_for(delay);
        std::lock_guard<std::mutex> lock(records_mutex_);
        saved = ToStrings(uris);
        seen_sizes = config_->spec_byte_sizes_per_block();
        const auto result = result_for ? result_for(uris) : save_result;
        if (result == ER_OK) {
            *actual_uris = uris;
        }
        return result;
    }

    void *registered_base{nullptr};
    std::size_t registered_size{0};
    ClientErrorCode load_result = ER_OK;
    ClientErrorCode save_result = ER_OK;
    std::chrono::milliseconds delay{0};
    std::function<void(const std::vector<DataStorageUri> &)> before_io;
    std::function<ClientErrorCode(const std::vector<DataStorageUri> &)> result_for;
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
    std::mutex records_mutex_;
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

std::unique_ptr<KvMetaTransferClientImpl> MakeTransferClient(RecordingSdkFactory &factory,
                                                             RecordingSdk *&recording,
                                                             std::uint64_t max_object_bytes = 32,
                                                             int timeout_ms = 1000,
                                                             std::size_t thread_count = 1,
                                                             const ClientMemoryRegistrations *registrations = nullptr) {
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
    SdkTimeoutConfig timeout;
    timeout.set_get_timeout_ms(timeout_ms);
    timeout.set_put_timeout_ms(timeout_ms);
    config->sdk_wrapper_config()->set_timeout_config(timeout);
    config->sdk_wrapper_config()->set_thread_num(thread_count);
    InitParams params;
    params.self_location_spec_name = "value";
    params.storage_configs = R"([{
        "type": "pace",
        "global_unique_name": "pace",
        "storage_spec": {"domain": "pace", "timeout": 1000}
    }])";
    auto wrapper = std::make_unique<SdkWrapper>();
    wrapper->sdk_factory_ = &factory;
    EXPECT_EQ(ER_OK,
              wrapper->Init(config, params, registrations, /*wait_for_completion=*/true, /*exact_object_sizes=*/true));
    recording = factory.sdk.get();

    auto client = std::make_unique<KvMetaTransferClientImpl>();
    client->client_config_ = std::move(config);
    client->sdk_wrapper_ = std::move(wrapper);
    client->max_object_bytes_ = max_object_bytes;
    return client;
}

TEST(KvMetaTransferClientTest, ValidatesEachRequestWithoutMutatingTheFixedSpecTable) {
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
    EXPECT_EQ((std::map<std::string, std::int64_t>{{"value", 1}}), recording->seen_sizes);
    const auto [save_ec, actual] = client->SaveObjects(uris, sizes, buffers);
    EXPECT_EQ(ER_OK, save_ec);
    EXPECT_EQ(uris, actual);
    EXPECT_EQ(uris, recording->saved);
    EXPECT_EQ((std::map<std::string, std::int64_t>{{"value", 1}}), recording->seen_sizes);

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

TEST(KvMetaTransferClientTest, BackendErrorsCannotMasqueradeAsPreIoRejections) {
    RecordingSdkFactory factory;
    RecordingSdk *recording = nullptr;
    auto client = MakeTransferClient(factory, recording);
    ASSERT_NE(nullptr, recording);
    char value[5]{};
    const UriStrVec uris{"pace://pace/1?media_type=0&node_id=1&range_id=0&size=5"};
    BlockBuffer buffer;
    buffer.iovs.push_back({MemoryType::CPU, value, sizeof(value), false});
    const BlockBuffers buffers{buffer};
    for (const auto ec : {ER_INVALID_PARAMS, ER_INVALID_LOCAL_BUFFERS, ER_INVALID_SDKWRAPPER_CONFIG}) {
        recording->save_result = ec;
        EXPECT_EQ(ER_SDKWRITE_ERROR, client->SaveObjects(uris, {sizeof(value)}, buffers).first);
        EXPECT_EQ(uris, recording->saved);
    }
}

TEST(KvMetaTransferClientTest, CompletedBackendFailuresPreserveRegistrationForReadAndWrite) {
    for (const bool save : {false, true}) {
        for (const auto ec : {ER_SDK_TIMEOUT_BUFFER_SAFE, ER_SDK_ERROR_BUFFER_SAFE, ER_SDK_TIMEOUT}) {
            SCOPED_TRACE(save ? "save" : "load");
            SCOPED_TRACE(static_cast<int>(ec));
            RecordingSdkFactory factory;
            RecordingSdk *recording = nullptr;
            auto client = MakeTransferClient(factory, recording, 32, 100);
            ASSERT_NE(nullptr, recording);
            auto *wrapper = client->sdk_wrapper_.get();
            recording->save_result = recording->load_result = ec;
            std::promise<void> started;
            auto began = started.get_future();
            std::promise<void> release;
            auto released = release.get_future().share();
            recording->before_io = [&](const std::vector<DataStorageUri> &) {
                started.set_value();
                released.wait();
            };
            char value[5]{};
            const UriStrVec uris{"pace://pace/1?media_type=0&node_id=1&range_id=0&size=5"};
            BlockBuffer buffer;
            buffer.iovs.push_back({MemoryType::CPU, value, sizeof(value), false});
            const BlockBuffers buffers{buffer};
            auto operation = std::async(std::launch::async, [&] {
                return save ? client->SaveObjects(uris, {sizeof(value)}, buffers).first
                            : client->LoadObjects(uris, {sizeof(value)}, buffers);
            });
            const bool dispatched = began.wait_for(std::chrono::seconds(2)) == std::future_status::ready;
            EXPECT_TRUE(dispatched);
            if (dispatched) {
                EXPECT_EQ(std::future_status::timeout, operation.wait_for(std::chrono::milliseconds(150)));
            }
            release.set_value();
            EXPECT_EQ(ec, operation.get());
            // Unknown remote completion must still be reported to the caller;
            // preserving the MR is distinct from declaring its buffer safe.
            EXPECT_EQ(wrapper, client->sdk_wrapper_.get());
            recording->before_io = {};
            recording->save_result = recording->load_result = ER_OK;
            EXPECT_EQ(ER_OK, client->LoadObjects(uris, {sizeof(value)}, buffers));
            EXPECT_EQ(ER_OK, client->SaveObjects(uris, {sizeof(value)}, buffers).first);
        }
    }
}

TEST(KvMetaTransferClientTest, RejectsAdvisoryModeBeforeSubmittingAnyIo) {
    RecordingSdkFactory factory;
    RecordingSdk *recording = nullptr;
    auto client = MakeTransferClient(factory, recording);
    client->sdk_wrapper_->wait_for_completion_ = false;
    char value[5]{};
    const UriStrVec uris{"pace://pace/1?media_type=0&node_id=1&range_id=0&size=5"};
    BlockBuffer buffer;
    buffer.iovs.push_back({MemoryType::CPU, value, sizeof(value), false});
    EXPECT_EQ(ER_INVALID_SDKWRAPPER_CONFIG, client->LoadObjects(uris, {5}, {buffer}));
    EXPECT_EQ(ER_INVALID_SDKWRAPPER_CONFIG, client->SaveObjects(uris, {5}, {buffer}).first);
    EXPECT_TRUE(recording->loaded.empty());
    EXPECT_TRUE(recording->saved.empty());
}

TEST(KvMetaTransferClientTest, ExceptionDrainsItsOwnPeersAndPreservesTheSharedWrapper) {
    for (const bool save : {false, true}) {
        SCOPED_TRACE(save ? "save" : "load");
        RecordingSdkFactory factory;
        RecordingSdk *recording = nullptr;
        auto client = MakeTransferClient(factory, recording, 32, 5000, 2);
        client->sdk_wrapper_->sdk_map_["peer"] = factory.sdk;

        std::promise<void> peer_started;
        const auto started = peer_started.get_future().share();
        std::promise<void> release_peer;
        const auto release = release_peer.get_future().share();
        recording->before_io = [&](const std::vector<DataStorageUri> &uris) {
            if (uris.front().GetHostName() == "peer") {
                peer_started.set_value();
                release.wait();
            } else {
                if (started.wait_for(std::chrono::seconds(2)) != std::future_status::ready) {
                    throw std::runtime_error("peer did not start");
                }
                throw std::runtime_error("backend failed");
            }
        };
        char values[2]{};
        BlockBuffer first;
        first.iovs.push_back({MemoryType::CPU, &values[0], 1, false});
        BlockBuffer second;
        second.iovs.push_back({MemoryType::CPU, &values[1], 1, false});
        const BlockBuffers buffers{first, second};
        const UriStrVec uris{
            "pace://pace/1?media_type=0&node_id=1&range_id=0&size=1",
            "pace://peer/2?media_type=0&node_id=1&range_id=0&size=1",
        };
        auto operation = std::async(std::launch::async, [&] {
            return save ? client->SaveObjects(uris, {1, 1}, buffers).first : client->LoadObjects(uris, {1, 1}, buffers);
        });

        EXPECT_EQ(std::future_status::ready, started.wait_for(std::chrono::seconds(2)));
        EXPECT_EQ(std::future_status::timeout, operation.wait_for(std::chrono::milliseconds(50)));
        release_peer.set_value();
        EXPECT_THROW((void)operation.get(), std::runtime_error);
        recording->before_io = {};
        EXPECT_EQ(ER_OK, client->LoadObjects(uris, {1, 1}, buffers));
        EXPECT_EQ(ER_OK, client->SaveObjects(uris, {1, 1}, buffers).first);
    }
}

// The object API below uses the real transfer client, wrapper, and SDK task
// pool. Barriers are inside the backend Get/Put, not at Python/API entry.
class ConcurrentMetadata final : public KvMetaClient {
public:
    std::pair<ClientErrorCode, std::string>
    RegisterInstance(const std::string &, const std::string &, const std::string &) override {
        return {ER_OK, "[]"};
    }
    std::pair<ClientErrorCode, KvMetaInstanceInfo> GetInstanceInfo(const std::string &) override { return {ER_OK, {}}; }
    std::pair<ClientErrorCode, KvMetaGetResult> Get(const std::string &,
                                                    const std::vector<std::string> &keys) override {
        KvMetaGetResult result;
        for (const auto &key : keys) {
            result.hit_mask.push_back(true);
            result.locations.push_back(LocationFor(std::stoull(key)));
        }
        return {ER_OK, std::move(result)};
    }
    std::pair<ClientErrorCode, KvMetaStartWriteResult> StartWrite(const std::string &trace,
                                                                  const std::vector<std::string> &keys,
                                                                  const std::vector<std::uint64_t> &sizes,
                                                                  std::int32_t) override {
        std::lock_guard<std::mutex> lock(mutex);
        for (const auto &key : keys) {
            if (active_keys.count(key)) {
                return {ER_SERVICE_WRITE_IN_PROGRESS, {}};
            }
        }
        KvMetaStartWriteResult result;
        result.write_session_id = trace;
        result.key_mask.assign(keys.size(), false);
        for (std::size_t i = 0; i < sizes.size(); ++i) {
            active_keys.insert(keys[i]);
            result.locations.push_back(LocationFor(sizes[i]));
        }
        sessions[trace] = keys;
        return {ER_OK, std::move(result)};
    }
    ClientErrorCode FinishWrite(const std::string &, const std::string &session, const std::vector<bool> &) override {
        std::lock_guard<std::mutex> lock(mutex);
        ++finish_count;
        for (const auto &key : sessions[session]) {
            active_keys.erase(key);
        }
        sessions.erase(session);
        return ER_OK;
    }
    ClientErrorCode Remove(const std::string &, const std::vector<std::string> &) override { return ER_OK; }
    static KvMetaValueLocation LocationFor(std::uint64_t size) {
        KvMetaValueLocation location;
        location.type = KvMetaStorageType::TAIR_MEMPOOL;
        location.value_size = size;
        location.location_specs.push_back({"value",
                                           "pace://pace/" + std::to_string(size) +
                                               "?media_type=0&node_id=1&range_id=0&size=" + std::to_string(size)});
        return location;
    }
    std::atomic<int> finish_count{0};

private:
    std::mutex mutex;
    std::set<std::string> active_keys;
    std::map<std::string, std::vector<std::string>> sessions;
};

BlockBuffer RegionBuffer(char *base, std::size_t size) {
    BlockBuffer buffer;
    buffer.iovs.push_back({MemoryType::CPU, base, size, false});
    return buffer;
}

TEST(KvMetaTransferClientConcurrencyTest, SingleClientHasMultipleDifferentSizeReadsAndWritesInFlight) {
    using namespace std::chrono_literals;
    std::unique_ptr<std::FILE, decltype(&std::fclose)> file(std::tmpfile(), &std::fclose);
    ASSERT_NE(nullptr, file);
    ASSERT_EQ(0, ftruncate(fileno(file.get()), 17));
    void *mapping = mmap(nullptr, 17, PROT_READ | PROT_WRITE, MAP_SHARED, fileno(file.get()), 0);
    ASSERT_NE(MAP_FAILED, mapping);
    const auto unmap = [](char *base) { munmap(base, 17); };
    std::unique_ptr<char, decltype(unmap)> region(static_cast<char *>(mapping), unmap);
    ClientMemoryRegistrations registrations;
    registrations.host = {region.get(), 17, fileno(file.get())};
    RecordingSdkFactory factory;
    RecordingSdk *recording = nullptr;
    auto transfer = MakeTransferClient(factory, recording, 32, 5000, 4, &registrations);
    ASSERT_EQ(region.get(), recording->registered_base);
    ASSERT_EQ(17, recording->registered_size);
    auto metadata = std::make_unique<ConcurrentMetadata>();
    KvMetaObjectClientImpl client(std::move(metadata), std::move(transfer), 32, 30);
    std::mutex mutex;
    std::condition_variable condition;
    int active = 0;
    bool released = false;
    recording->before_io = [&](const auto &) {
        std::unique_lock<std::mutex> lock(mutex);
        ++active;
        condition.notify_all();
        condition.wait(lock, [&] { return released; });
    };
    std::vector<std::future<ClientErrorCode>> operations;
    std::size_t offset = 0;
    for (const std::size_t size : {2, 3, 5, 7}) {
        char *base = region.get() + offset;
        offset += size;
        const bool write = size > 3;
        operations.push_back(std::async(std::launch::async, [&, size, base, write] {
            const auto key = std::to_string(size);
            return write ? client.SaveObjects(key, {key}, {size}, {RegionBuffer(base, size)})
                         : client.LoadObjects(key, {key}, {size}, {RegionBuffer(base, size)});
        }));
    }
    {
        std::unique_lock<std::mutex> lock(mutex);
        EXPECT_TRUE(condition.wait_for(lock, 2s, [&] { return active == 4; }));
    }
    auto close = std::async(std::launch::async, [&] { client.Close(); });
    EXPECT_EQ(std::future_status::timeout, close.wait_for(50ms));
    {
        std::lock_guard<std::mutex> lock(mutex);
        released = true;
        condition.notify_all();
    }
    for (auto &operation : operations) {
        EXPECT_EQ(ER_OK, operation.get());
    }
    EXPECT_EQ(std::future_status::ready, close.wait_for(2s));
    close.get();
    EXPECT_EQ((std::map<std::string, std::int64_t>{{"value", 1}}), recording->seen_sizes);
    EXPECT_EQ(ER_CLIENT_NOT_EXISTS, client.LoadObjects("closed", {"2"}, {2}, {RegionBuffer(region.get(), 2)}));
}

TEST(KvMetaTransferClientConcurrencyTest, SameKeyWritesAreArbitratedBeforeASecondDataTransfer) {
    using namespace std::chrono_literals;
    RecordingSdkFactory factory;
    RecordingSdk *recording = nullptr;
    auto transfer = MakeTransferClient(factory, recording, 32, 5000, 2);
    KvMetaObjectClientImpl client(std::make_unique<ConcurrentMetadata>(), std::move(transfer), 32, 30);
    char region[10]{};
    std::promise<void> entered, release;
    auto began = entered.get_future();
    auto released = release.get_future().share();
    std::atomic<int> transfers{0};
    recording->before_io = [&](const auto &) {
        ++transfers;
        entered.set_value();
        released.wait();
    };
    auto first = std::async(std::launch::async,
                            [&] { return client.SaveObjects("first", {"5"}, {5}, {RegionBuffer(region, 5)}); });
    EXPECT_EQ(std::future_status::ready, began.wait_for(2s));
    auto conflict = std::async(std::launch::async,
                               [&] { return client.SaveObjects("second", {"5"}, {5}, {RegionBuffer(region + 5, 5)}); });
    EXPECT_EQ(std::future_status::ready, conflict.wait_for(2s));
    release.set_value();
    EXPECT_EQ(ER_SERVICE_WRITE_IN_PROGRESS, conflict.get());
    EXPECT_EQ(ER_OK, first.get());
    EXPECT_EQ(1, transfers);
}

TEST(KvMetaTransferClientConcurrencyTest, TimeoutOrExceptionDoesNotDrainAnUnrelatedOperation) {
    using namespace std::chrono_literals;
    for (const bool write : {false, true}) {
        for (int mode = 0; mode < 3; ++mode) {
            SCOPED_TRACE(write);
            SCOPED_TRACE(mode);
            RecordingSdkFactory factory;
            RecordingSdk *recording = nullptr;
            auto transfer = MakeTransferClient(factory, recording, 32, 5000, 3);
            auto metadata = std::make_unique<ConcurrentMetadata>();
            auto *meta = metadata.get();
            KvMetaObjectClientImpl client(std::move(metadata), std::move(transfer), 32, 30);
            std::promise<void> other_started, release;
            auto began = other_started.get_future().share();
            auto released = release.get_future().share();
            recording->before_io = [&](const auto &uris) {
                const auto size = uris.front().GetParam("size");
                if (size == "7") {
                    other_started.set_value();
                    released.wait();
                }
                if (size == "5" && mode == 2) {
                    throw std::runtime_error("injected task exception");
                }
            };
            recording->result_for = [&](const auto &uris) {
                return uris.front().GetParam("size") != "5" ? ER_OK
                       : mode == 0                          ? ER_SDK_TIMEOUT_BUFFER_SAFE
                                                            : ER_SDK_TIMEOUT;
            };
            char region[15]{};
            auto other = std::async(std::launch::async,
                                    [&] { return client.LoadObjects("other", {"7"}, {7}, {RegionBuffer(region, 7)}); });
            EXPECT_EQ(std::future_status::ready, began.wait_for(2s));
            auto failure = std::async(std::launch::async, [&] {
                return write
                           ? client.SaveObjectsWithTransferState("failure", {"5"}, {5}, {RegionBuffer(region + 7, 5)})
                           : client.LoadObjectsWithTransferState("failure", {"5"}, {5}, {RegionBuffer(region + 7, 5)});
            });
            const bool completed = failure.wait_for(2s) == std::future_status::ready;
            EXPECT_TRUE(completed);
            if (completed) {
                const auto result = failure.get();
                EXPECT_EQ(mode == 0   ? ER_SDK_TIMEOUT_BUFFER_SAFE
                          : mode == 1 ? ER_SDK_TIMEOUT
                          : write     ? ER_SDKWRITE_ERROR
                                      : ER_SDKREAD_ERROR,
                          result.ec);
                EXPECT_EQ(mode != 0, result.data_transfer_may_be_inflight);
                EXPECT_EQ(0, meta->finish_count);
                EXPECT_EQ(ER_OK, client.LoadObjects("next", {"3"}, {3}, {RegionBuffer(region + 12, 3)}));
            }
            EXPECT_EQ(std::future_status::timeout, other.wait_for(20ms));
            auto close = std::async(std::launch::async, [&] { client.Close(); });
            EXPECT_EQ(std::future_status::timeout, close.wait_for(20ms));
            release.set_value();
            if (!completed) {
                (void)failure.get();
            }
            EXPECT_EQ(ER_OK, other.get());
            close.get();
        }
    }
}

} // namespace
} // namespace kv_cache_manager
