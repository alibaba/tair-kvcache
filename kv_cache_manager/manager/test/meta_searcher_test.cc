#include <filesystem>
#include <future>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <thread>
#include <tuple>

#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/config/meta_indexer_config.h"
#include "kv_cache_manager/config/meta_storage_backend_config.h"
#include "kv_cache_manager/manager/meta_searcher.h"
#include "kv_cache_manager/meta/meta_indexer.h"
#include "kv_cache_manager/meta/meta_local_backend.h"

using namespace kv_cache_manager;

namespace {
// Helper class to create test data
class MetaSearcherTestHelper {
public:
    static LocationSpec CreateLocationSpec(const std::string &name = "", const std::string &uri = "") {
        LocationSpec spec(name, uri);
        return spec;
    }

    static CacheLocationConstPtr CreateCacheLocation(DataStorageType type = DataStorageType::DATA_STORAGE_TYPE_NFS,
                                                     size_t spec_size = 1,
                                                     const std::vector<LocationSpec> &specs = {}) {
        return std::make_shared<CacheLocation>(type, spec_size, specs);
    }

    static std::vector<LocationSpec> CreateDefaultLocationSpecs() {
        LocationSpec spec = CreateLocationSpec();
        return {spec};
    }
};

CheckLocDataExistFunc dummy_check_loc_data_exist = [](const CacheLocation &) -> bool { return true; };
SubmitDelReqFunc dummy_submit_del_req = [](const std::vector<std::int64_t> &,
                                           const std::vector<std::vector<std::string>> &,
                                           const std::vector<std::vector<std::string>> &,
                                           bool) -> void {};

class FaultyGetLocationIdsBackend : public MetaLocalBackend {
public:
    void SetFailedKey(KeyType key) { failed_key_ = key; }

    std::vector<ErrorCode> GetLocationIds(RequestContext *request_context,
                                          const KeyTypeVec &keys,
                                          LocationIdsPerKey &out_location_ids) noexcept override {
        auto results = MetaLocalBackend::GetLocationIds(request_context, keys, out_location_ids);
        if (!failed_key_.has_value()) {
            return results;
        }
        for (size_t i = 0; i < keys.size(); ++i) {
            if (keys[i] == failed_key_.value()) {
                results[i] = EC_ERROR;
                out_location_ids[i].clear();
            }
        }
        return results;
    }

private:
    std::optional<KeyType> failed_key_;
};
} // namespace

class MetaSearcherTest : public TESTBASE {
public:
    void SetUp() override {
        meta_indexer_ = CreateMetaIndexer();
        meta_searcher_ =
            std::make_shared<MetaSearcher>(meta_indexer_, dummy_check_loc_data_exist, dummy_submit_del_req);
        request_context_ = std::make_shared<RequestContext>("test_trace_id");
    }

    void TearDown() override {}

    std::shared_ptr<MetaStorageBackendConfig> ConstructMetaStorageBackendConfig() {
        auto meta_storage_backend_config = std::make_shared<MetaStorageBackendConfig>();
        std::string local_path = GetPrivateTestRuntimeDataPath() + "meta_local_backend_file_1";
        meta_storage_backend_config->SetStorageUri("file://" + local_path);
        std::error_code ec;
        bool exists = std::filesystem::exists(local_path, ec);
        EXPECT_FALSE(ec) << local_path; // false means correct
        if (exists) {
            std::remove(local_path.c_str());
        }
        return meta_storage_backend_config;
    }

    std::shared_ptr<MetaIndexer> CreateMetaIndexer() {
        auto meta_indexer_config = std::make_shared<MetaIndexerConfig>();
        auto backend_config = ConstructMetaStorageBackendConfig();
        meta_indexer_config->SetMetaStorageBackendConfig(backend_config);
        meta_indexer_config->SetMutexShardNum(32);
        meta_indexer_config->SetMaxKeyCount(10000);
        auto indexer = std::make_shared<MetaIndexer>();
        auto metaCachePolicyConfig = std::make_shared<MetaCachePolicyConfig>();
        metaCachePolicyConfig->SetCapacity(0);
        meta_indexer_config->SetMetaCachePolicyConfig(metaCachePolicyConfig);
        auto ec = indexer->Init(/*instance_id*/ "test", meta_indexer_config);
        if (ec != ErrorCode::EC_OK) {
            KVCM_LOG_ERROR("Init meta indexer failed");
            return nullptr;
        }
        return indexer;
    }

    FaultyGetLocationIdsBackend *ReplaceWithFaultyBackend() {
        auto backend_config = ConstructMetaStorageBackendConfig();
        auto faulty_backend = std::make_unique<FaultyGetLocationIdsBackend>();
        EXPECT_EQ(EC_OK, faulty_backend->Init("test", backend_config));
        EXPECT_EQ(EC_OK, faulty_backend->Open());
        auto backend_raw = faulty_backend.get();
        meta_indexer_->backend_manager_->persistent_backend_->Close();
        meta_indexer_->backend_manager_->persistent_backend_ = std::move(faulty_backend);
        meta_indexer_->backend_manager_->cache_backend_.reset();
        return backend_raw;
    }

    std::shared_ptr<MetaIndexer> meta_indexer_;
    std::shared_ptr<MetaSearcher> meta_searcher_;
    std::shared_ptr<RequestContext> request_context_;
    StaticWeightSLPolicy policy_;
};

TEST_F(MetaSearcherTest, TestBatchAddLocation) {
    // 准备测试数据
    MetaSearcher::KeyVector keys = {1, 2, 3};

    // 创建CacheLocation对象
    auto location_specs = MetaSearcherTestHelper::CreateDefaultLocationSpecs();
    CacheLocationConstPtr location1 =
        MetaSearcherTestHelper::CreateCacheLocation(DataStorageType::DATA_STORAGE_TYPE_NFS, 1, location_specs);
    CacheLocationConstPtr location2 =
        MetaSearcherTestHelper::CreateCacheLocation(DataStorageType::DATA_STORAGE_TYPE_HF3FS, 2, location_specs);
    CacheLocationConstPtr location3 =
        MetaSearcherTestHelper::CreateCacheLocation(DataStorageType::DATA_STORAGE_TYPE_MOONCAKE, 3, location_specs);

    CacheLocationVector locations = {location1, location2, location3};

    // 调用BatchAddLocation
    std::vector<std::string> out_location_ids;
    ErrorCode ec = meta_searcher_->BatchAddLocation(request_context_.get(), keys, locations, out_location_ids);

    // 验证结果
    EXPECT_EQ(ec, ErrorCode::EC_OK);
    EXPECT_EQ(out_location_ids.size(), 3);

    // 验证添加的位置信息可以被检索到
    std::vector<CacheLocationMap> out_location_maps;
    BlockMask mask; // 空mask，不跳过任何元素

    ec = meta_searcher_->BatchGetLocation(request_context_.get(), keys, mask, out_location_maps);
    EXPECT_EQ(ec, ErrorCode::EC_OK);
    EXPECT_EQ(out_location_maps.size(), 3);

    for (const auto &location_map : out_location_maps) {
        EXPECT_FALSE(location_map.empty());
        // 每个map应该只有一个location（我们刚添加的）
        EXPECT_EQ(location_map.size(), 1);
    }
}

TEST_F(MetaSearcherTest, TestBatchAddLocation2) {
    // 准备测试数据
    MetaSearcher::KeyVector keys = {1, 2, 3};

    // 创建CacheLocation对象
    std::vector<LocationSpec> specs1(
        1, MetaSearcherTestHelper::CreateLocationSpec("nfs", "file:///tmp/test1/test.txt?offset=1&length=2&size=3"));
    CacheLocationConstPtr location1 =
        MetaSearcherTestHelper::CreateCacheLocation(DataStorageType::DATA_STORAGE_TYPE_NFS, 1, specs1);

    std::vector<LocationSpec> specs2(
        1, MetaSearcherTestHelper::CreateLocationSpec("hf3fs", "hf3fs:///tmp/test1/test.txt?offset=1&length=2&size=4"));
    CacheLocationConstPtr location2 =
        MetaSearcherTestHelper::CreateCacheLocation(DataStorageType::DATA_STORAGE_TYPE_HF3FS, 2, specs2);

    std::vector<LocationSpec> specs3(2,
                                     MetaSearcherTestHelper::CreateLocationSpec(
                                         "mooncake", "mooncake:///tmp/test1/test.txt?offset=1&length=2&size=5"));
    CacheLocationConstPtr location3 =
        MetaSearcherTestHelper::CreateCacheLocation(DataStorageType::DATA_STORAGE_TYPE_MOONCAKE, 3, specs3);

    CacheLocationVector locations = {location1, location2, location3};

    // 调用BatchAddLocation
    std::vector<std::string> out_location_ids;
    ErrorCode ec = meta_searcher_->BatchAddLocation(request_context_.get(), keys, locations, out_location_ids);

    // 验证结果
    EXPECT_EQ(ec, ErrorCode::EC_OK);
    EXPECT_EQ(out_location_ids.size(), 3);

    // 验证添加的位置信息可以被检索到
    std::vector<CacheLocationMap> out_location_maps;
    BlockMask mask; // 空mask，不跳过任何元素

    ec = meta_searcher_->BatchGetLocation(request_context_.get(), keys, mask, out_location_maps);
    EXPECT_EQ(ec, ErrorCode::EC_OK);
    EXPECT_EQ(out_location_maps.size(), 3);

    for (const auto &location_map : out_location_maps) {
        EXPECT_FALSE(location_map.empty());
        // 每个map应该只有一个location（我们刚添加的）
        EXPECT_EQ(location_map.size(), 1);
    }

    EXPECT_EQ(3, meta_indexer_->GetStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_NFS));
    EXPECT_EQ(4, meta_indexer_->GetStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_HF3FS));
    EXPECT_EQ(10, meta_indexer_->GetStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_MOONCAKE));
}

TEST_F(MetaSearcherTest, TestBatchMergeLocationSpecsAppendsAndOverwrites) {
    MetaSearcher::KeyVector keys = {10001};
    const std::string location_id = "event_report#mem#127.0.0.1:8080";

    std::vector<ErrorCode> per_key_ec;
    std::vector<std::vector<MetaSearcher::MergeLocationSpecsTask>> tasks = {{
        {location_id,
         DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L1P5,
         CacheLocationStatus::CLS_SERVING,
         {LocationSpec("linear_0", "event_report://127.0.0.1:8080/mem")}},
    }};
    ASSERT_EQ(EC_OK, meta_searcher_->BatchMergeLocationSpecs(request_context_.get(), keys, tasks, per_key_ec));
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), per_key_ec);

    tasks = {{
        {location_id,
         DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L1P5,
         CacheLocationStatus::CLS_SERVING,
         {LocationSpec("linear_1", "event_report://127.0.0.1:8080/mem")}},
    }};
    ASSERT_EQ(EC_OK, meta_searcher_->BatchMergeLocationSpecs(request_context_.get(), keys, tasks, per_key_ec));
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), per_key_ec);

    tasks = {{
        {location_id,
         DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L1P5,
         CacheLocationStatus::CLS_SERVING,
         {LocationSpec("linear_0", "event_report://127.0.0.1:8080/mem"),
          LocationSpec("full_3", "event_report://127.0.0.1:8080/mem")}},
    }};
    ASSERT_EQ(EC_OK, meta_searcher_->BatchMergeLocationSpecs(request_context_.get(), keys, tasks, per_key_ec));
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), per_key_ec);

    std::vector<CacheLocationMap> location_maps;
    BlockMask mask;
    ASSERT_EQ(EC_OK, meta_searcher_->BatchGetLocation(request_context_.get(), keys, mask, location_maps));
    ASSERT_EQ(1u, location_maps.size());
    ASSERT_EQ(1u, location_maps[0].size());
    auto it = location_maps[0].find(location_id);
    ASSERT_NE(location_maps[0].end(), it);
    ASSERT_TRUE(it->second);
    EXPECT_EQ(CacheLocationStatus::CLS_SERVING, it->second->status());
    EXPECT_EQ(3u, it->second->spec_size());

    std::map<std::string, std::string> spec_uris;
    for (const auto &spec : it->second->location_specs()) {
        spec_uris[spec.name()] = spec.uri();
    }
    ASSERT_EQ(3u, spec_uris.size());
    EXPECT_EQ("event_report://127.0.0.1:8080/mem", spec_uris["linear_0"]);
    EXPECT_EQ("event_report://127.0.0.1:8080/mem", spec_uris["linear_1"]);
    EXPECT_EQ("event_report://127.0.0.1:8080/mem", spec_uris["full_3"]);
}

TEST_F(MetaSearcherTest, TestBatchMergeLocationSpecsPreservesUntouchedSpecsAcrossGenerationChange) {
    const MetaSearcher::KeyVector keys = {10007};
    const std::string location_id = "kvs#event_report_l2#mem#127.0.0.1:8080";
    const std::string version_a = "00112233445566778899aabbccddeeff";
    const std::string version_b = "ffeeddccbbaa99887766554433221100";
    auto uri = [](const std::string &source, const std::string &version) {
        return "event_report://127.0.0.1:8080/mem?source=" + source + "&s_version=" + version;
    };

    std::vector<ErrorCode> per_key_ec;
    std::vector<std::vector<MetaSearcher::ReplaceLocationSpecsTask>> seed_tasks = {{
        {location_id,
         DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2,
         CacheLocationStatus::CLS_SERVING,
         {LocationSpec("linear_0", uri("old_linear", version_a)),
          LocationSpec("mamba_0", uri("old_mamba", version_a)),
          LocationSpec("legacy", "event_report://127.0.0.1:8080/mem?source=legacy")}},
    }};
    ASSERT_EQ(EC_OK, meta_searcher_->BatchReplaceLocationSpecs(request_context_.get(), keys, seed_tasks, per_key_ec));

    std::vector<std::vector<MetaSearcher::MergeLocationSpecsTask>> tasks = {{
        {location_id,
         DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2,
         CacheLocationStatus::CLS_SERVING,
         {LocationSpec("linear_0", uri("new_linear", version_b))}},
    }};
    ASSERT_EQ(EC_OK, meta_searcher_->BatchMergeLocationSpecs(request_context_.get(), keys, tasks, per_key_ec));

    std::vector<CacheLocationMap> location_maps;
    BlockMask mask;
    ASSERT_EQ(EC_OK, meta_searcher_->BatchGetLocation(request_context_.get(), keys, mask, location_maps));
    ASSERT_EQ(1u, location_maps.size());
    ASSERT_EQ(1u, location_maps[0].size());
    const auto &after_version_change = location_maps[0].at(location_id)->location_specs();
    std::map<std::string, std::string> after_version_change_specs;
    for (const auto &spec : after_version_change) {
        after_version_change_specs[spec.name()] = spec.uri();
    }
    ASSERT_EQ(3u, after_version_change_specs.size());
    EXPECT_EQ(uri("new_linear", version_b), after_version_change_specs["linear_0"]);
    EXPECT_EQ(uri("old_mamba", version_a), after_version_change_specs["mamba_0"]);
    EXPECT_EQ("event_report://127.0.0.1:8080/mem?source=legacy", after_version_change_specs["legacy"]);

    tasks[0][0].specs = {
        LocationSpec("linear_0", uri("newer_linear", version_b)),
        LocationSpec("mamba_1", uri("new_mamba", version_b)),
    };
    ASSERT_EQ(EC_OK, meta_searcher_->BatchMergeLocationSpecs(request_context_.get(), keys, tasks, per_key_ec));
    ASSERT_EQ(EC_OK, meta_searcher_->BatchGetLocation(request_context_.get(), keys, mask, location_maps));

    std::map<std::string, std::string> specs;
    for (const auto &spec : location_maps[0].at(location_id)->location_specs()) {
        specs[spec.name()] = spec.uri();
    }
    ASSERT_EQ(4u, specs.size());
    EXPECT_EQ(uri("newer_linear", version_b), specs["linear_0"]);
    EXPECT_EQ(uri("new_mamba", version_b), specs["mamba_1"]);
    EXPECT_EQ(uri("old_mamba", version_a), specs["mamba_0"]);
    EXPECT_EQ("event_report://127.0.0.1:8080/mem?source=legacy", specs["legacy"]);
}

TEST_F(MetaSearcherTest, TestBatchMergeLocationSpecsRejectsMixedOrMalformedSnapshotVersions) {
    const MetaSearcher::KeyVector keys = {10012};
    const std::string location_id = "kvs#event_report_l2#mem#127.0.0.1:8080";
    const std::string version_a = "00112233445566778899aabbccddeeff";
    const std::string version_b = "ffeeddccbbaa99887766554433221100";
    auto uri = [](const std::string &source, const std::string &version) {
        return "event_report://127.0.0.1:8080/mem?source=" + source + "&s_version=" + version;
    };

    std::vector<ErrorCode> per_key_ec;
    std::vector<std::vector<MetaSearcher::MergeLocationSpecsTask>> tasks = {{
        {location_id,
         DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2,
         CacheLocationStatus::CLS_SERVING,
         {LocationSpec("baseline", uri("baseline", version_a))}},
    }};
    ASSERT_EQ(EC_OK, meta_searcher_->BatchMergeLocationSpecs(request_context_.get(), keys, tasks, per_key_ec));
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), per_key_ec);

    const std::vector<std::vector<LocationSpec>> invalid_specs = {
        {
            LocationSpec("valid", uri("valid", version_b)),
            LocationSpec("missing", "event_report://127.0.0.1:8080/mem?source=missing"),
        },
        {
            LocationSpec("valid", uri("valid", version_b)),
            LocationSpec("malformed", "event_report://127.0.0.1:8080/mem?source=malformed&s_version=not-a-token"),
        },
        {
            LocationSpec("version_a", uri("version_a", version_a)),
            LocationSpec("version_b", uri("version_b", version_b)),
        },
        {
            LocationSpec("duplicate",
                         "event_report://127.0.0.1:8080/mem?source=duplicate&s_version=" + version_b +
                             "&s_version=" + version_b),
        },
    };

    for (const auto &specs : invalid_specs) {
        tasks[0][0].specs = specs;
        per_key_ec.clear();
        meta_searcher_->BatchMergeLocationSpecs(request_context_.get(), keys, tasks, per_key_ec);
        ASSERT_EQ((std::vector<ErrorCode>{EC_BADARGS}), per_key_ec);

        std::vector<CacheLocationMap> location_maps;
        BlockMask mask;
        ASSERT_EQ(EC_OK, meta_searcher_->BatchGetLocation(request_context_.get(), keys, mask, location_maps));
        ASSERT_EQ(1u, location_maps.size());
        ASSERT_EQ(1u, location_maps[0].size());
        const auto &stored_specs = location_maps[0].at(location_id)->location_specs();
        ASSERT_EQ(1u, stored_specs.size());
        EXPECT_EQ("baseline", stored_specs[0].name());
        EXPECT_EQ(uri("baseline", version_a), stored_specs[0].uri());
    }

    const MetaSearcher::KeyVector new_keys = {10013};
    tasks[0][0].specs = invalid_specs.front();
    per_key_ec.clear();
    meta_searcher_->BatchMergeLocationSpecs(request_context_.get(), new_keys, tasks, per_key_ec);
    ASSERT_EQ((std::vector<ErrorCode>{EC_BADARGS}), per_key_ec);
    std::vector<CacheLocationMap> location_maps;
    BlockMask mask;
    ASSERT_EQ(EC_OK, meta_searcher_->BatchGetLocation(request_context_.get(), new_keys, mask, location_maps));
    ASSERT_EQ(1u, location_maps.size());
    EXPECT_TRUE(location_maps.front().empty());
}

TEST_F(MetaSearcherTest, TestConcurrentSnapshotReplaceIsAtomicAndSameTokenDeltasDoNotLoseUpdates) {
    const int64_t key = 10011;
    const std::string location_id = "kvs#event_report_l2#mem#127.0.0.1:8080";
    constexpr size_t kSnapshotContenders = 12;

    std::promise<void> replace_start;
    auto replace_signal = replace_start.get_future().share();
    std::vector<std::future<std::pair<ErrorCode, std::vector<ErrorCode>>>> replace_futures;
    replace_futures.reserve(kSnapshotContenders);
    for (size_t i = 0; i < kSnapshotContenders; ++i) {
        replace_futures.push_back(std::async(std::launch::async, [&, i] {
            replace_signal.wait();
            const std::string generation = std::to_string(i);
            const std::string token = std::string(31, '0') + "0123456789ab"[i];
            const std::string uri_prefix =
                "event_report://127.0.0.1:8080/mem?generation=" + generation + "&s_version=" + token;
            std::vector<std::vector<MetaSearcher::ReplaceLocationSpecsTask>> tasks = {{
                {location_id,
                 DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2,
                 CacheLocationStatus::CLS_SERVING,
                 {LocationSpec("tp0", uri_prefix), LocationSpec("tp1", uri_prefix)}},
            }};
            std::vector<ErrorCode> per_key_ec;
            RequestContext context("concurrent_replace_" + generation);
            const ErrorCode ec = meta_searcher_->BatchReplaceLocationSpecs(&context, {key}, tasks, per_key_ec);
            return std::make_pair(ec, per_key_ec);
        }));
    }
    replace_start.set_value();
    for (auto &future : replace_futures) {
        ASSERT_EQ(std::future_status::ready, future.wait_for(std::chrono::seconds(2)));
        const auto [ec, per_key_ec] = future.get();
        EXPECT_EQ(EC_OK, ec);
        EXPECT_EQ((std::vector<ErrorCode>{EC_OK}), per_key_ec);
    }

    std::vector<CacheLocationMap> location_maps;
    BlockMask mask;
    ASSERT_EQ(EC_OK, meta_searcher_->BatchGetLocation(request_context_.get(), {key}, mask, location_maps));
    ASSERT_EQ(1u, location_maps.size());
    ASSERT_EQ(1u, location_maps[0].size());
    const auto &replaced_specs = location_maps[0].at(location_id)->location_specs();
    ASSERT_EQ(2u, replaced_specs.size());
    SnapshotUriInfo first_info;
    SnapshotUriInfo second_info;
    ASSERT_TRUE(SnapshotUriUtils::ParseSnapshotUriInfo(replaced_specs[0].uri(), first_info));
    ASSERT_TRUE(SnapshotUriUtils::ParseSnapshotUriInfo(replaced_specs[1].uri(), second_info));
    EXPECT_EQ(first_info.version, second_info.version);
    const auto first_generation = DataStorageUri(replaced_specs[0].uri()).GetParam("generation");
    const auto second_generation = DataStorageUri(replaced_specs[1].uri()).GetParam("generation");
    EXPECT_EQ(first_generation, second_generation);

    constexpr size_t kDeltaWriters = 8;
    std::promise<void> merge_start;
    auto merge_signal = merge_start.get_future().share();
    std::vector<std::future<std::pair<ErrorCode, std::vector<ErrorCode>>>> merge_futures;
    merge_futures.reserve(kDeltaWriters);
    for (size_t i = 0; i < kDeltaWriters; ++i) {
        merge_futures.push_back(std::async(std::launch::async, [&, i] {
            merge_signal.wait();
            const std::string index = std::to_string(i);
            const std::string uri =
                "event_report://127.0.0.1:8080/mem?delta=" + index + "&s_version=" + first_info.version;
            std::vector<std::vector<MetaSearcher::MergeLocationSpecsTask>> tasks = {{
                {location_id,
                 DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2,
                 CacheLocationStatus::CLS_SERVING,
                 {LocationSpec("delta_" + index, uri)}},
            }};
            std::vector<ErrorCode> per_key_ec;
            RequestContext context("concurrent_merge_" + index);
            const ErrorCode ec = meta_searcher_->BatchMergeLocationSpecs(&context, {key}, tasks, per_key_ec);
            return std::make_pair(ec, per_key_ec);
        }));
    }
    merge_start.set_value();
    for (auto &future : merge_futures) {
        ASSERT_EQ(std::future_status::ready, future.wait_for(std::chrono::seconds(2)));
        const auto [ec, per_key_ec] = future.get();
        EXPECT_EQ(EC_OK, ec);
        EXPECT_EQ((std::vector<ErrorCode>{EC_OK}), per_key_ec);
    }

    ASSERT_EQ(EC_OK, meta_searcher_->BatchGetLocation(request_context_.get(), {key}, mask, location_maps));
    const auto &merged_specs = location_maps[0].at(location_id)->location_specs();
    ASSERT_EQ(2u + kDeltaWriters, merged_specs.size());
    std::set<std::string> names;
    for (const auto &spec : merged_specs) {
        SnapshotUriInfo info;
        ASSERT_TRUE(SnapshotUriUtils::ParseSnapshotUriInfo(spec.uri(), info));
        EXPECT_EQ(first_info.version, info.version);
        names.insert(spec.name());
    }
    EXPECT_EQ(2u + kDeltaWriters, names.size());
}

TEST_F(MetaSearcherTest, TestBatchMergeLocationSpecsCreatesLocationWithMultipleSpecs) {
    MetaSearcher::KeyVector keys = {10002};
    const std::string location_id = "event_report#mem#127.0.0.1:8080";

    std::vector<std::vector<MetaSearcher::MergeLocationSpecsTask>> tasks = {{
        {location_id,
         DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L1P5,
         CacheLocationStatus::CLS_SERVING,
         {LocationSpec("linear_0", "event_report://127.0.0.1:8080/mem"),
          LocationSpec("linear_1", "event_report://127.0.0.1:8080/mem")}},
    }};
    std::vector<ErrorCode> per_key_ec;
    ASSERT_EQ(EC_OK, meta_searcher_->BatchMergeLocationSpecs(request_context_.get(), keys, tasks, per_key_ec));

    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), per_key_ec);

    std::vector<CacheLocationMap> location_maps;
    BlockMask mask;
    ASSERT_EQ(EC_OK, meta_searcher_->BatchGetLocation(request_context_.get(), keys, mask, location_maps));
    ASSERT_EQ(1u, location_maps.size());
    ASSERT_EQ(1u, location_maps[0].size());
    const auto &loc = location_maps[0].at(location_id);
    ASSERT_TRUE(loc);
    EXPECT_EQ(2u, loc->spec_size());
    ASSERT_EQ(2u, loc->location_specs().size());
}

TEST_F(MetaSearcherTest, TestMergeAndReplaceLocationSpecsKeepStorageUsageExact) {
    const MetaSearcher::KeyVector keys = {10006};
    const std::string location_id = "event_report#mem#127.0.0.1:8080";
    std::vector<ErrorCode> per_key_ec;
    std::vector<std::vector<MetaSearcher::MergeLocationSpecsTask>> merge_tasks = {{
        {location_id,
         DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2,
         CacheLocationStatus::CLS_SERVING,
         {LocationSpec("linear_0", "event_report://127.0.0.1:8080/mem?size=10")}},
    }};

    ASSERT_EQ(EC_OK, meta_searcher_->BatchMergeLocationSpecs(request_context_.get(), keys, merge_tasks, per_key_ec));
    EXPECT_EQ(10u, meta_indexer_->GetStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2));

    // An at-least-once retry overwrites the same named spec and must not count
    // the bytes twice.
    ASSERT_EQ(EC_OK, meta_searcher_->BatchMergeLocationSpecs(request_context_.get(), keys, merge_tasks, per_key_ec));
    EXPECT_EQ(10u, meta_indexer_->GetStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2));

    merge_tasks[0][0].specs.emplace_back("full_3", "event_report://127.0.0.1:8080/mem?size=5");
    ASSERT_EQ(EC_OK, meta_searcher_->BatchMergeLocationSpecs(request_context_.get(), keys, merge_tasks, per_key_ec));
    EXPECT_EQ(15u, meta_indexer_->GetStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2));

    std::vector<std::vector<MetaSearcher::ReplaceLocationSpecsTask>> replace_tasks = {{
        {location_id,
         DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2,
         CacheLocationStatus::CLS_SERVING,
         {LocationSpec("linear_0", "event_report://127.0.0.1:8080/mem?size=7")}},
    }};
    ASSERT_EQ(EC_OK,
              meta_searcher_->BatchReplaceLocationSpecs(request_context_.get(), keys, replace_tasks, per_key_ec));
    EXPECT_EQ(7u, meta_indexer_->GetStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2));
    ASSERT_EQ(EC_OK,
              meta_searcher_->BatchReplaceLocationSpecs(request_context_.get(), keys, replace_tasks, per_key_ec));
    EXPECT_EQ(7u, meta_indexer_->GetStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2));

    // A location id has stable storage ownership.  Rejecting a type mutation
    // must leave both the stored location and per-type accounting unchanged.
    replace_tasks[0][0].type = DataStorageType::DATA_STORAGE_TYPE_NFS;
    EXPECT_EQ(EC_OK,
              meta_searcher_->BatchReplaceLocationSpecs(request_context_.get(), keys, replace_tasks, per_key_ec));
    ASSERT_EQ((std::vector<ErrorCode>{EC_BADARGS}), per_key_ec);
    EXPECT_EQ(7u, meta_indexer_->GetStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2));
    EXPECT_EQ(0u, meta_indexer_->GetStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_NFS));

    std::vector<CacheLocationMap> location_maps;
    BlockMask mask;
    ASSERT_EQ(EC_OK, meta_searcher_->BatchGetLocation(request_context_.get(), keys, mask, location_maps));
    ASSERT_EQ(1u, location_maps.size());
    ASSERT_EQ(1u, location_maps.front().size());
    EXPECT_EQ(DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2, location_maps.front().at(location_id)->type());
}

TEST_F(MetaSearcherTest, TestConditionalDeleteDoesNotRemoveRefreshedStableLocation) {
    const MetaSearcher::KeyVector keys = {10008};
    const std::string location_id = "kvs#event_report_l2#mem#127.0.0.8:8080";
    std::vector<ErrorCode> per_key_ec;
    std::vector<std::vector<MetaSearcher::ReplaceLocationSpecsTask>> replace_tasks = {{
        {location_id,
         DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2,
         CacheLocationStatus::CLS_SERVING,
         {LocationSpec("linear_0", "event_report://127.0.0.8:8080/mem?source=old&size=11")}},
    }};
    ASSERT_EQ(EC_OK,
              meta_searcher_->BatchReplaceLocationSpecs(request_context_.get(), keys, replace_tasks, per_key_ec));

    std::vector<CacheLocationMap> location_maps;
    BlockMask mask;
    ASSERT_EQ(EC_OK, meta_searcher_->BatchGetLocation(request_context_.get(), keys, mask, location_maps));
    const std::string stale_expected_value = location_maps[0].at(location_id)->ToJsonString();

    // Simulate a new lifecycle refreshing the stable location id after the
    // old host-cleanup scan captured its value.
    replace_tasks[0][0].specs = {
        LocationSpec("linear_0", "event_report://127.0.0.8:8080/mem?source=new&size=13"),
    };
    ASSERT_EQ(EC_OK,
              meta_searcher_->BatchReplaceLocationSpecs(request_context_.get(), keys, replace_tasks, per_key_ec));

    LocationIdsPerKey location_ids = {{location_id}};
    std::vector<std::vector<ErrorCode>> delete_results;
    ASSERT_EQ(EC_OK,
              meta_searcher_->BatchDeleteLocations(
                  request_context_.get(), keys, location_ids, delete_results, {{stale_expected_value}}));
    ASSERT_EQ((std::vector<std::vector<ErrorCode>>{{EC_MISMATCH}}), delete_results);

    ASSERT_EQ(EC_OK, meta_searcher_->BatchGetLocation(request_context_.get(), keys, mask, location_maps));
    ASSERT_EQ(1u, location_maps[0].count(location_id));
    EXPECT_NE(std::string::npos, location_maps[0].at(location_id)->location_specs()[0].uri().find("source=new"));
    EXPECT_EQ(13u, meta_indexer_->GetStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2));

    const std::string current_expected_value = location_maps[0].at(location_id)->ToJsonString();
    ASSERT_EQ(EC_BADARGS,
              meta_searcher_->BatchDeleteLocations(
                  request_context_.get(), keys, location_ids, delete_results, {{current_expected_value}, {}}));
    ASSERT_EQ(EC_OK,
              meta_searcher_->BatchDeleteLocations(
                  request_context_.get(), keys, location_ids, delete_results, {{current_expected_value}}));
    ASSERT_EQ((std::vector<std::vector<ErrorCode>>{{EC_OK}}), delete_results);

    ASSERT_EQ(EC_OK, meta_searcher_->BatchGetLocation(request_context_.get(), keys, mask, location_maps));
    EXPECT_TRUE(location_maps[0].empty());
    EXPECT_EQ(0u, meta_indexer_->GetStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2));
}

TEST_F(MetaSearcherTest, TestCleanupLocationsByHostSkipsKeysWithoutMatchingLocation) {
    const MetaSearcher::KeyVector keys = {10009, 10010, 10011};
    const std::string target_suffix = "#127.0.0.9:8080";
    const std::string target_location = "kvs#event_report_l2#mem" + target_suffix;
    const std::string other_location = "kvs#event_report_l2#mem#127.0.0.10:8080";
    std::vector<std::vector<MetaSearcher::ReplaceLocationSpecsTask>> replace_tasks = {
        {{
            target_location,
            DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2,
            CacheLocationStatus::CLS_SERVING,
            {LocationSpec("target_0", "event_report://127.0.0.9:8080/mem?size=3")},
        }},
        {{
            other_location,
            DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2,
            CacheLocationStatus::CLS_SERVING,
            {LocationSpec("other_0", "event_report://127.0.0.10:8080/mem?size=5")},
        }},
        {
            {
                target_location,
                DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2,
                CacheLocationStatus::CLS_SERVING,
                {LocationSpec("target_1", "event_report://127.0.0.9:8080/mem?size=7")},
            },
            {
                other_location,
                DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2,
                CacheLocationStatus::CLS_SERVING,
                {LocationSpec("other_1", "event_report://127.0.0.10:8080/mem?size=11")},
            },
        },
    };
    std::vector<ErrorCode> per_key_ec;
    ASSERT_EQ(EC_OK,
              meta_searcher_->BatchReplaceLocationSpecs(request_context_.get(), keys, replace_tasks, per_key_ec));
    ASSERT_EQ(EC_OK,
              meta_searcher_->CleanupLocationsByHost(request_context_.get(),
                                                     target_suffix,
                                                     DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2,
                                                     /*scan_batch_size=*/1000));

    std::vector<CacheLocationMap> location_maps;
    BlockMask mask;
    ASSERT_EQ(EC_OK, meta_searcher_->BatchGetLocation(request_context_.get(), keys, mask, location_maps));
    ASSERT_EQ(3u, location_maps.size());
    EXPECT_TRUE(location_maps[0].empty());
    ASSERT_EQ(1u, location_maps[1].size());
    EXPECT_EQ(1u, location_maps[1].count(other_location));
    ASSERT_EQ(1u, location_maps[2].size());
    EXPECT_EQ(1u, location_maps[2].count(other_location));
    EXPECT_EQ(16u, meta_indexer_->GetStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2));
}

TEST_F(MetaSearcherTest, TestBatchMergeLocationSpecsContinuesMergeAfterPartialBlockFailure) {
    auto faulty_backend = ReplaceWithFaultyBackend();
    ASSERT_NE(nullptr, faulty_backend);

    const int64_t merge_key = 10004;
    const int64_t failed_key = 10005;
    const std::string location_id = "event_report#mem#127.0.0.1:8080";
    const std::string failed_location_id = "event_report#mem#127.0.0.2:8080";

    std::vector<std::vector<MetaSearcher::MergeLocationSpecsTask>> seed_tasks = {{
        {location_id,
         DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L1P5,
         CacheLocationStatus::CLS_SERVING,
         {LocationSpec("linear_0", "event_report://127.0.0.1:8080/mem")}},
    }};
    std::vector<ErrorCode> per_key_ec;
    ASSERT_EQ(EC_OK,
              meta_searcher_->BatchMergeLocationSpecs(request_context_.get(), {merge_key}, seed_tasks, per_key_ec));
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), per_key_ec);

    faulty_backend->SetFailedKey(failed_key);
    std::vector<std::vector<MetaSearcher::MergeLocationSpecsTask>> merge_tasks = {
        {{location_id,
          DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L1P5,
          CacheLocationStatus::CLS_SERVING,
          {LocationSpec("full_3", "event_report://127.0.0.1:8080/mem")}}},
        {{failed_location_id,
          DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L1P5,
          CacheLocationStatus::CLS_SERVING,
          {LocationSpec("linear_0", "event_report://127.0.0.2:8080/mem")}}},
    };

    EXPECT_EQ(EC_PARTIAL_OK,
              meta_searcher_->BatchMergeLocationSpecs(
                  request_context_.get(), {merge_key, failed_key}, merge_tasks, per_key_ec));
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK, EC_ERROR}), per_key_ec);

    std::vector<CacheLocationMap> location_maps;
    BlockMask mask;
    ASSERT_EQ(EC_OK, meta_searcher_->BatchGetLocation(request_context_.get(), {merge_key}, mask, location_maps));
    ASSERT_EQ(1u, location_maps.size());
    ASSERT_EQ(1u, location_maps[0].size());

    std::set<std::string> spec_names;
    for (const auto &spec : location_maps[0].at(location_id)->location_specs()) {
        spec_names.insert(spec.name());
    }
    ASSERT_EQ((std::set<std::string>{"linear_0", "full_3"}), spec_names);
}

TEST_F(MetaSearcherTest, TestBatchDeleteLocationSpecsPartialDelete) {
    MetaSearcher::KeyVector keys = {10003};
    const std::string location_id = "event_report#mem#127.0.0.1:8080";

    std::vector<std::vector<MetaSearcher::MergeLocationSpecsTask>> merge_tasks = {{
        {location_id,
         DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L1P5,
         CacheLocationStatus::CLS_SERVING,
         {LocationSpec("linear_0", "event_report://127.0.0.1:8080/mem"),
          LocationSpec("linear_1", "event_report://127.0.0.1:8080/mem"),
          LocationSpec("full_3", "event_report://127.0.0.1:8080/mem")}},
    }};
    std::vector<ErrorCode> per_key_ec;
    ASSERT_EQ(EC_OK, meta_searcher_->BatchMergeLocationSpecs(request_context_.get(), keys, merge_tasks, per_key_ec));

    std::vector<std::vector<MetaSearcher::DeleteLocationSpecsTask>> delete_tasks = {{
        {location_id, {"linear_0"}},
    }};
    std::vector<std::vector<ErrorCode>> delete_results;
    ASSERT_EQ(EC_OK,
              meta_searcher_->BatchDeleteLocationSpecs(request_context_.get(), keys, delete_tasks, delete_results));
    ASSERT_EQ(1u, delete_results.size());
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), delete_results[0]);

    std::vector<CacheLocationMap> location_maps;
    BlockMask mask;
    ASSERT_EQ(EC_OK, meta_searcher_->BatchGetLocation(request_context_.get(), keys, mask, location_maps));
    ASSERT_EQ(1u, location_maps.size());
    ASSERT_EQ(1u, location_maps[0].size());
    auto spec_names = std::set<std::string>();
    for (const auto &spec : location_maps[0].at(location_id)->location_specs()) {
        spec_names.insert(spec.name());
    }
    ASSERT_EQ((std::set<std::string>{"linear_1", "full_3"}), spec_names);
    EXPECT_EQ(2u, location_maps[0].at(location_id)->spec_size());

    delete_tasks = {{
        {location_id, {"linear_1", "full_3"}},
    }};
    ASSERT_EQ(EC_OK,
              meta_searcher_->BatchDeleteLocationSpecs(request_context_.get(), keys, delete_tasks, delete_results));
    ASSERT_EQ(1u, delete_results.size());
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), delete_results[0]);
    ASSERT_EQ(EC_OK, meta_searcher_->BatchGetLocation(request_context_.get(), keys, mask, location_maps));
    ASSERT_EQ(1u, location_maps.size());
    EXPECT_TRUE(location_maps[0].empty());

    delete_tasks = {{
        {location_id, {}},
    }};
    ASSERT_EQ(EC_OK, meta_searcher_->BatchMergeLocationSpecs(request_context_.get(), keys, merge_tasks, per_key_ec));
    ASSERT_EQ(EC_OK,
              meta_searcher_->BatchDeleteLocationSpecs(request_context_.get(), keys, delete_tasks, delete_results));
    ASSERT_EQ(1u, delete_results.size());
    ASSERT_EQ((std::vector<ErrorCode>{EC_BADARGS}), delete_results[0]);
    ASSERT_EQ(EC_OK, meta_searcher_->BatchGetLocation(request_context_.get(), keys, mask, location_maps));
    ASSERT_EQ(1u, location_maps.size());
    EXPECT_FALSE(location_maps[0].empty());
}

TEST_F(MetaSearcherTest, TestBatchDeleteLocationSpecsValidatesShapeAndMissingLocationIsIdempotent) {
    MetaSearcher::KeyVector keys = {10004};
    const std::string existing_location_id = "event_report#mem#127.0.0.1:8080";

    std::vector<std::vector<MetaSearcher::MergeLocationSpecsTask>> merge_tasks = {{
        {existing_location_id,
         DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L1P5,
         CacheLocationStatus::CLS_SERVING,
         {LocationSpec("linear_0", "event_report://127.0.0.1:8080/mem")}},
    }};
    std::vector<ErrorCode> per_key_ec;
    ASSERT_EQ(EC_OK, meta_searcher_->BatchMergeLocationSpecs(request_context_.get(), keys, merge_tasks, per_key_ec));

    std::vector<std::vector<ErrorCode>> delete_results;
    std::vector<std::vector<MetaSearcher::DeleteLocationSpecsTask>> mismatched_tasks;
    EXPECT_EQ(EC_BADARGS,
              meta_searcher_->BatchDeleteLocationSpecs(request_context_.get(), keys, mismatched_tasks, delete_results));
    EXPECT_TRUE(delete_results.empty());

    std::vector<std::vector<MetaSearcher::DeleteLocationSpecsTask>> missing_location_tasks = {{
        {"event_report#mem#127.0.0.2:8080", {"linear_0"}},
    }};
    ASSERT_EQ(
        EC_OK,
        meta_searcher_->BatchDeleteLocationSpecs(request_context_.get(), keys, missing_location_tasks, delete_results));
    ASSERT_EQ(1u, delete_results.size());
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), delete_results[0]);

    std::vector<CacheLocationMap> location_maps;
    BlockMask mask;
    ASSERT_EQ(EC_OK, meta_searcher_->BatchGetLocation(request_context_.get(), keys, mask, location_maps));
    ASSERT_EQ(1u, location_maps.size());
    ASSERT_EQ(1u, location_maps[0].size());
    auto existing = location_maps[0].find(existing_location_id);
    ASSERT_NE(location_maps[0].end(), existing);
    ASSERT_TRUE(existing->second);
    ASSERT_EQ(1u, existing->second->location_specs().size());
    EXPECT_EQ("linear_0", existing->second->location_specs()[0].name());
}

TEST_F(MetaSearcherTest, TestBatchDeleteLocationSpecsIsIdempotentForMissingData) {
    const MetaSearcher::KeyVector keys = {10008};
    const std::string location_id = "kvs#event_report_l2#mem#127.0.0.1:8080";
    std::vector<ErrorCode> per_key_ec;
    std::vector<std::vector<MetaSearcher::MergeLocationSpecsTask>> merge_tasks = {{
        {location_id,
         DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2,
         CacheLocationStatus::CLS_SERVING,
         {LocationSpec("linear_0", "event_report://127.0.0.1:8080/mem")}},
    }};
    ASSERT_EQ(EC_OK, meta_searcher_->BatchMergeLocationSpecs(request_context_.get(), keys, merge_tasks, per_key_ec));

    std::vector<std::vector<ErrorCode>> delete_results;
    std::vector<std::vector<MetaSearcher::DeleteLocationSpecsTask>> delete_tasks = {{
        {location_id, {"missing_spec"}},
    }};
    std::vector<std::vector<bool>> missing_targets;
    ASSERT_EQ(EC_OK,
              meta_searcher_->BatchDeleteLocationSpecs(
                  request_context_.get(), keys, delete_tasks, delete_results, &missing_targets));
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), delete_results[0]);
    ASSERT_EQ(1u, missing_targets.size());
    ASSERT_EQ((std::vector<bool>{false}), missing_targets[0]);

    delete_tasks = {{{"kvs#event_report_l2#disk#127.0.0.1:8080", {"linear_0"}}}};
    ASSERT_EQ(EC_OK,
              meta_searcher_->BatchDeleteLocationSpecs(
                  request_context_.get(), keys, delete_tasks, delete_results, &missing_targets));
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), delete_results[0]);
    ASSERT_EQ((std::vector<bool>{true}), missing_targets[0]);

    const MetaSearcher::KeyVector absent_keys = {10080};
    delete_tasks = {{{location_id, {"linear_0"}}}};
    ASSERT_EQ(EC_OK,
              meta_searcher_->BatchDeleteLocationSpecs(
                  request_context_.get(), absent_keys, delete_tasks, delete_results, &missing_targets));
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), delete_results[0]);
    ASSERT_EQ((std::vector<bool>{true}), missing_targets[0]);

    std::vector<CacheLocationMap> location_maps;
    BlockMask mask;
    ASSERT_EQ(EC_OK, meta_searcher_->BatchGetLocation(request_context_.get(), keys, mask, location_maps));
    ASSERT_EQ(1u, location_maps.size());
    ASSERT_EQ(1u, location_maps[0].size());
    const auto &specs = location_maps[0].at(location_id)->location_specs();
    ASSERT_EQ(1u, specs.size());
    EXPECT_EQ("linear_0", specs[0].name());
}

TEST_F(MetaSearcherTest, TestCleanupLocationsByPredicateSubmitsExactObservedValue) {
    const MetaSearcher::KeyVector keys = {10009, 10010};
    const std::string stale_id = "kvs#event_report_l2#mem#127.0.0.1:8080";
    const std::string current_id = "kvs#event_report_l2#mem#127.0.0.2:8080";
    std::vector<std::vector<MetaSearcher::MergeLocationSpecsTask>> tasks = {
        {{stale_id,
          DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2,
          CacheLocationStatus::CLS_SERVING,
          {LocationSpec("linear_0", "event_report://127.0.0.1:8080/mem")}}},
        {{current_id,
          DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2,
          CacheLocationStatus::CLS_SERVING,
          {LocationSpec("linear_0", "event_report://127.0.0.2:8080/mem")}}},
    };
    std::vector<ErrorCode> per_key_ec;
    ASSERT_EQ(EC_OK, meta_searcher_->BatchMergeLocationSpecs(request_context_.get(), keys, tasks, per_key_ec));

    std::vector<CacheLocationMap> observed_locations;
    BlockMask mask;
    ASSERT_EQ(EC_OK, meta_searcher_->BatchGetLocation(request_context_.get(), keys, mask, observed_locations));
    ASSERT_EQ(2u, observed_locations.size());
    const std::string expected_stale_value = observed_locations[0].at(stale_id)->ToJsonString();

    std::vector<std::tuple<int64_t, std::string, std::string>> submitted;
    bool submitted_metadata_only = false;
    SubmitDelReqFunc capture = [&submitted,
                                &submitted_metadata_only](const std::vector<int64_t> &submitted_keys,
                                                          const std::vector<std::vector<std::string>> &location_ids,
                                                          const std::vector<std::vector<std::string>> &expected_values,
                                                          bool metadata_only) {
        submitted_metadata_only = metadata_only;
        for (size_t i = 0; i < submitted_keys.size(); ++i) {
            for (size_t j = 0; j < location_ids[i].size(); ++j) {
                submitted.emplace_back(submitted_keys[i], location_ids[i][j], expected_values[i][j]);
            }
        }
    };
    MetaSearcher cleanup_searcher(meta_indexer_, dummy_check_loc_data_exist, std::move(capture));
    ASSERT_EQ(EC_OK,
              cleanup_searcher.CleanupLocationsByPredicate(
                  request_context_.get(),
                  DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2,
                  1,
                  [&stale_id](int64_t, const std::string &location_id, const CacheLocation &) {
                      return location_id == stale_id;
                  }));

    ASSERT_EQ(1u, submitted.size());
    EXPECT_EQ(10009, std::get<0>(submitted[0]));
    EXPECT_EQ(stale_id, std::get<1>(submitted[0]));
    EXPECT_EQ(expected_stale_value, std::get<2>(submitted[0]));
    EXPECT_TRUE(submitted_metadata_only);

    bool predicate_called = false;
    ASSERT_EQ(EC_OK,
              cleanup_searcher.CleanupLocationsByPredicate(
                  request_context_.get(),
                  DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2,
                  1,
                  [&predicate_called](int64_t, const std::string &, const CacheLocation &) {
                      predicate_called = true;
                      return true;
                  },
                  [] { return true; }));
    EXPECT_FALSE(predicate_called);
    EXPECT_EQ(1u, submitted.size());

    bool epoch_changed = false;
    size_t visited_location_count = 0;
    const size_t submitted_before_early_abort = submitted.size();
    ASSERT_EQ(EC_OK,
              cleanup_searcher.CleanupLocationsByPredicate(
                  request_context_.get(),
                  DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2,
                  1,
                  [&epoch_changed, &visited_location_count](int64_t, const std::string &, const CacheLocation &) {
                      ++visited_location_count;
                      epoch_changed = true;
                      return true;
                  },
                  [&epoch_changed] { return epoch_changed; }));
    // The epoch changes while processing batch 1. Cancellation is observed at
    // the next batch boundary, while the per-location predicate remains the
    // final guard for every location already scanned in batch 1.
    EXPECT_EQ(1u, visited_location_count);
    EXPECT_EQ(submitted_before_early_abort + 1, submitted.size());
}

TEST_F(MetaSearcherTest, TestPrefixMatch) {
    // 首先添加一些测试数据
    MetaSearcher::KeyVector keys = {10, 20, 30};

    // 创建CacheLocation对象
    auto location_specs = MetaSearcherTestHelper::CreateDefaultLocationSpecs();
    CacheLocationConstPtr location1 =
        MetaSearcherTestHelper::CreateCacheLocation(DataStorageType::DATA_STORAGE_TYPE_NFS, 1, location_specs);
    CacheLocationConstPtr location2 =
        MetaSearcherTestHelper::CreateCacheLocation(DataStorageType::DATA_STORAGE_TYPE_HF3FS, 2, location_specs);
    CacheLocationConstPtr location3 =
        MetaSearcherTestHelper::CreateCacheLocation(DataStorageType::DATA_STORAGE_TYPE_MOONCAKE, 3, location_specs);

    CacheLocationVector locations = {location1, location2, location3};
    // 添加位置信息
    std::vector<std::string> out_location_ids;
    ErrorCode ec = meta_searcher_->BatchAddLocation(request_context_.get(), keys, locations, out_location_ids);
    EXPECT_EQ(ec, ErrorCode::EC_OK);
    EXPECT_EQ(out_location_ids.size(), 3);

    {
        // 改成serving之前match不到
        CacheLocationVector out_locations;
        BlockMask mask; // 空mask，不跳过任何元素

        ec = meta_searcher_->PrefixMatch(request_context_.get(), keys, mask, out_locations, &policy_);
        EXPECT_EQ(ec, ErrorCode::EC_OK);
        EXPECT_EQ(out_locations.size(), 0);
    }

    std::vector<CacheLocationStatus> new_status(out_location_ids.size(), CacheLocationStatus::CLS_SERVING);
    // 构建批量任务，每个key对应一个任务
    std::vector<std::vector<MetaSearcher::LocationUpdateTask>> batch_tasks;
    batch_tasks.reserve(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        std::vector<MetaSearcher::LocationUpdateTask> tasks;
        tasks.push_back({out_location_ids[i], new_status[i]});
        batch_tasks.push_back(tasks);
    }

    std::vector<std::vector<ErrorCode>> out_batch_results;
    ec = meta_searcher_->BatchUpdateLocationStatus(request_context_.get(), keys, batch_tasks, out_batch_results);
    EXPECT_EQ(ec, ErrorCode::EC_OK);

    struct PrefixMatchTestData {
        MetaSearcher::KeyVector keys;
        size_t result_length;
    };
    std::vector<PrefixMatchTestData> test_datas;
    test_datas.push_back({{10, 20, 30}, 3});
    test_datas.push_back({{10, 20, 999}, 2});
    test_datas.push_back({{10, 999, 999}, 1});
    test_datas.push_back({{999, 999, 999}, 0});
    test_datas.push_back({{}, 0});
    test_datas.push_back({{10}, 1});
    test_datas.push_back({{10, 20}, 2});
    test_datas.push_back({{10, 999, 30}, 1});

    for (auto &test_data : test_datas) {

        // 测试PrefixMatch
        CacheLocationVector out_locations;
        BlockMask mask; // 空mask，不跳过任何元素

        ec = meta_searcher_->PrefixMatch(request_context_.get(), test_data.keys, mask, out_locations, &policy_);

        // 验证结果
        EXPECT_EQ(ec, ErrorCode::EC_OK);
        EXPECT_EQ(out_locations.size(), test_data.result_length);

        // 验证返回的 locations（合并视图的 id 无单一元数据语义；type 为策略 winner 的后端类型）
        for (size_t i = 0; i < out_locations.size(); i++) {
            EXPECT_EQ(out_locations[i]->status(), CLS_SERVING);
        }
    }

    // 测试带mask的PrefixMatch
    std::vector<BlockMask> mask_vectors;
    mask_vectors.emplace_back(BlockMaskVector{true, false, false}); // 跳过第一个元素
    mask_vectors.emplace_back(BlockMaskOffset{1});                  // 跳过第一个元素

    for (auto &block_mask : mask_vectors) {
        CacheLocationVector out_locations;
        out_locations.clear();
        ec = meta_searcher_->PrefixMatch(request_context_.get(), keys, block_mask, out_locations, &policy_);

        // 验证结果 - 应该只返回后两个元素
        EXPECT_EQ(ec, ErrorCode::EC_OK);
        EXPECT_EQ(out_locations.size(), 2);

        // 验证返回的 locations（合并视图的 id 无单一元数据语义）
        for (size_t i = 0; i < out_locations.size(); i++) {
            EXPECT_EQ(out_locations[i]->status(), CLS_SERVING);
        }
    }
}

TEST_F(MetaSearcherTest, TestBatchGet) {
    // 首先添加一些测试数据
    MetaSearcher::KeyVector keys = {100, 200, 300};

    // 创建CacheLocation对象
    auto location_specs = MetaSearcherTestHelper::CreateDefaultLocationSpecs();
    CacheLocationConstPtr location1 =
        MetaSearcherTestHelper::CreateCacheLocation(DataStorageType::DATA_STORAGE_TYPE_NFS, 1, location_specs);
    CacheLocationConstPtr location2 =
        MetaSearcherTestHelper::CreateCacheLocation(DataStorageType::DATA_STORAGE_TYPE_HF3FS, 2, location_specs);
    CacheLocationConstPtr location3 =
        MetaSearcherTestHelper::CreateCacheLocation(DataStorageType::DATA_STORAGE_TYPE_MOONCAKE, 3, location_specs);

    CacheLocationVector locations = {location1, location2, location3};

    // 添加位置信息
    std::vector<std::string> out_location_ids;
    ErrorCode ec = meta_searcher_->BatchAddLocation(request_context_.get(), keys, locations, out_location_ids);
    EXPECT_EQ(ec, ErrorCode::EC_OK);
    EXPECT_EQ(out_location_ids.size(), 3);

    // 测试BatchGetLocation
    std::vector<CacheLocationMap> out_location_maps;
    BlockMask mask; // 空mask，不跳过任何元素

    ec = meta_searcher_->BatchGetLocation(request_context_.get(), keys, mask, out_location_maps);

    // 验证结果
    EXPECT_EQ(ec, ErrorCode::EC_OK);
    EXPECT_EQ(out_location_maps.size(), 3);

    // 验证返回的location maps
    for (size_t i = 0; i < out_location_maps.size(); i++) {
        const auto &location_map = out_location_maps[i];
        EXPECT_FALSE(location_map.empty());
        EXPECT_EQ(location_map.size(), 1);

        // 验证location信息
        auto it = location_map.find(out_location_ids[i]);
        EXPECT_NE(it, location_map.end());

        if (it != location_map.end()) {
            const auto &location = *it->second;
            EXPECT_EQ(location.id(), out_location_ids[i]);
            EXPECT_EQ(location.status(), CLS_WRITING);
            EXPECT_EQ(location.type(), locations[i]->type());
        }
    }

    // 测试带mask的PrefixMatch
    std::vector<BlockMask> mask_vectors;
    mask_vectors.emplace_back(BlockMaskVector{false, true, true}); // 跳过后二个元素
    mask_vectors.emplace_back(BlockMaskOffset{2});                 // 跳过前两个元素

    for (auto &block_mask : mask_vectors) {
        out_location_maps.clear();
        ec = meta_searcher_->BatchGetLocation(request_context_.get(), keys, block_mask, out_location_maps);

        // 验证结果 - 应该返回1个元素
        EXPECT_EQ(ec, ErrorCode::EC_OK);
        EXPECT_EQ(out_location_maps.size(), 1);

        // 应该有数据
        for (size_t idx = 0; idx < 1; idx++) {
            EXPECT_FALSE(out_location_maps[idx].empty());
            EXPECT_EQ(out_location_maps[idx].size(), 1);
        }
    }
}

TEST_F(MetaSearcherTest, TestBatchUpdateLocationStatus) {
    // 首先添加一些测试数据
    MetaSearcher::KeyVector keys = {1000, 2000, 3000};

    // 创建CacheLocation对象
    auto location_specs = MetaSearcherTestHelper::CreateDefaultLocationSpecs();
    CacheLocationConstPtr location1 =
        MetaSearcherTestHelper::CreateCacheLocation(DataStorageType::DATA_STORAGE_TYPE_NFS, 1, location_specs);
    CacheLocationConstPtr location2 =
        MetaSearcherTestHelper::CreateCacheLocation(DataStorageType::DATA_STORAGE_TYPE_HF3FS, 2, location_specs);
    CacheLocationConstPtr location3 =
        MetaSearcherTestHelper::CreateCacheLocation(DataStorageType::DATA_STORAGE_TYPE_MOONCAKE, 3, location_specs);

    CacheLocationVector locations = {location1, location2, location3};

    // 添加位置信息
    std::vector<std::string> out_location_ids;
    ErrorCode ec = meta_searcher_->BatchAddLocation(request_context_.get(), keys, locations, out_location_ids);
    EXPECT_EQ(ec, ErrorCode::EC_OK);
    EXPECT_EQ(out_location_ids.size(), 3);

    // 验证初始状态为CLS_WRITING
    std::vector<CacheLocationMap> out_location_maps;
    BlockMask mask; // 空mask，不跳过任何元素

    ec = meta_searcher_->BatchGetLocation(request_context_.get(), keys, mask, out_location_maps);
    EXPECT_EQ(ec, ErrorCode::EC_OK);
    EXPECT_EQ(out_location_maps.size(), 3);

    for (size_t i = 0; i < out_location_maps.size(); i++) {
        const auto &location_map = out_location_maps[i];
        EXPECT_FALSE(location_map.empty());
        EXPECT_EQ(location_map.size(), 1);

        // 验证location信息
        auto it = location_map.find(out_location_ids[i]);
        EXPECT_NE(it, location_map.end());

        if (it != location_map.end()) {
            const auto &location = *it->second;
            EXPECT_EQ(location.id(), out_location_ids[i]);
            EXPECT_EQ(location.status(), CLS_WRITING); // 初始状态应为CLS_WRITING
            EXPECT_EQ(location.type(), locations[i]->type());
        }
    }

    // 准备更新状态的数据
    std::vector<CacheLocationStatus> new_statuses = {CLS_SERVING, CLS_DELETING, CLS_NEW};

    // 构建批量任务，每个key对应一个任务
    std::vector<std::vector<MetaSearcher::LocationUpdateTask>> batch_tasks;
    batch_tasks.reserve(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        std::vector<MetaSearcher::LocationUpdateTask> tasks;
        tasks.push_back({out_location_ids[i], new_statuses[i]});
        batch_tasks.push_back(tasks);
    }

    std::vector<std::vector<ErrorCode>> out_batch_results;
    ec = meta_searcher_->BatchUpdateLocationStatus(request_context_.get(), keys, batch_tasks, out_batch_results);
    EXPECT_EQ(ec, ErrorCode::EC_OK);

    // 验证状态已更新
    out_location_maps.clear();
    ec = meta_searcher_->BatchGetLocation(request_context_.get(), keys, mask, out_location_maps);
    EXPECT_EQ(ec, ErrorCode::EC_OK);
    EXPECT_EQ(out_location_maps.size(), 3);

    for (size_t i = 0; i < out_location_maps.size(); i++) {
        const auto &location_map = out_location_maps[i];
        EXPECT_FALSE(location_map.empty());
        EXPECT_EQ(location_map.size(), 1);

        // 验证location信息
        auto it = location_map.find(out_location_ids[i]);
        EXPECT_NE(it, location_map.end());

        if (it != location_map.end()) {
            const auto &location = *it->second;
            EXPECT_EQ(location.id(), out_location_ids[i]);
            EXPECT_EQ(location.status(), new_statuses[i]); // 状态应已更新
            EXPECT_EQ(location.type(), locations[i]->type());
        }
    }

    // 测试错误情况：key、location_id和status数量不匹配
    std::vector<std::string> mismatched_location_ids = {out_location_ids[0], out_location_ids[1]}; // 只有两个
    std::vector<CacheLocationStatus> mismatched_statuses = {CLS_SERVING, CLS_DELETING, CLS_NEW, CLS_WRITING}; // 四个

    // 测试错误情况：keys和batch_tasks大小不匹配
    std::vector<std::vector<MetaSearcher::LocationUpdateTask>> mismatched_batch_tasks;
    // 只为前两个keys创建任务，但keys有三个，应该返回EC_BADARGS
    for (size_t i = 0; i < 2; ++i) { // 只为前两个key创建任务
        std::vector<MetaSearcher::LocationUpdateTask> tasks;
        tasks.push_back({out_location_ids[i], new_statuses[i]});
        mismatched_batch_tasks.push_back(tasks);
    }

    std::vector<std::vector<ErrorCode>> out_batch_results1;
    ec = meta_searcher_->BatchUpdateLocationStatus(
        request_context_.get(), keys, mismatched_batch_tasks, out_batch_results1);
    EXPECT_EQ(ec, ErrorCode::EC_BADARGS);

    // 为三个keys创建任务，但keys只有两个，应该返回EC_BADARGS
    std::vector<std::vector<MetaSearcher::LocationUpdateTask>> mismatched_batch_tasks2;
    for (size_t i = 0; i < 3; ++i) { // 为三个key创建任务
        std::vector<MetaSearcher::LocationUpdateTask> tasks;
        tasks.push_back({out_location_ids[i], new_statuses[i]});
        mismatched_batch_tasks2.push_back(tasks);
    }

    std::vector<MetaSearcher::KeyVector::value_type> mismatched_keys = {keys[0], keys[1]}; // 只有两个key
    std::vector<std::vector<ErrorCode>> out_batch_results2;
    ec = meta_searcher_->BatchUpdateLocationStatus(
        request_context_.get(), mismatched_keys, mismatched_batch_tasks2, out_batch_results2);
    EXPECT_EQ(ec, ErrorCode::EC_BADARGS);
}

TEST_F(MetaSearcherTest, TestBlockKeyWithMultipleLocations) {
    // 准备测试数据 - 使用相同的key添加多个location
    MetaSearcher::KeyVector keys = {12345}; // 只使用一个key

    // 创建多个不同的CacheLocation对象
    std::vector<LocationSpec> location_specs1 = {MetaSearcherTestHelper::CreateLocationSpec("tp0", "uri1")};
    std::vector<LocationSpec> location_specs2 = {MetaSearcherTestHelper::CreateLocationSpec("tp1", "uri2")};
    std::vector<LocationSpec> location_specs3 = {MetaSearcherTestHelper::CreateLocationSpec("tp2", "uri3")};

    CacheLocationConstPtr location1 =
        MetaSearcherTestHelper::CreateCacheLocation(DataStorageType::DATA_STORAGE_TYPE_NFS, 1, location_specs1);
    CacheLocationConstPtr location2 =
        MetaSearcherTestHelper::CreateCacheLocation(DataStorageType::DATA_STORAGE_TYPE_HF3FS, 2, location_specs2);
    CacheLocationConstPtr location3 =
        MetaSearcherTestHelper::CreateCacheLocation(DataStorageType::DATA_STORAGE_TYPE_MOONCAKE, 3, location_specs3);

    // 将三个location添加到同一个key
    CacheLocationVector locations1 = {location1};
    CacheLocationVector locations2 = {location2};
    CacheLocationVector locations3 = {location3};

    // 分别调用三次BatchAddLocation，为同一个key添加三个不同的location
    std::vector<std::string> out_location_ids1, out_location_ids2, out_location_ids3;

    ErrorCode ec1 = meta_searcher_->BatchAddLocation(request_context_.get(), keys, locations1, out_location_ids1);
    ErrorCode ec2 = meta_searcher_->BatchAddLocation(request_context_.get(), keys, locations2, out_location_ids2);
    ErrorCode ec3 = meta_searcher_->BatchAddLocation(request_context_.get(), keys, locations3, out_location_ids3);

    // 验证结果
    EXPECT_EQ(ec1, ErrorCode::EC_OK);
    EXPECT_EQ(ec2, ErrorCode::EC_OK);
    EXPECT_EQ(ec3, ErrorCode::EC_OK);
    EXPECT_EQ(out_location_ids1.size(), 1);
    EXPECT_EQ(out_location_ids2.size(), 1);
    EXPECT_EQ(out_location_ids3.size(), 1);

    // 验证添加的位置信息可以被检索到
    std::vector<CacheLocationMap> out_location_maps;
    BlockMask mask; // 空mask，不跳过任何元素

    ErrorCode ec = meta_searcher_->BatchGetLocation(request_context_.get(), keys, mask, out_location_maps);
    EXPECT_EQ(ec, ErrorCode::EC_OK);
    EXPECT_EQ(out_location_maps.size(), 1); // 只有一个key

    // 验证该key对应的location map包含三个location
    const auto &location_map = out_location_maps[0];
    EXPECT_EQ(location_map.size(), 3); // 应该有三个location

    // 验证三个location都存在且信息正确
    EXPECT_NE(location_map.find(out_location_ids1[0]), location_map.end());
    EXPECT_NE(location_map.find(out_location_ids2[0]), location_map.end());
    EXPECT_NE(location_map.find(out_location_ids3[0]), location_map.end());

    // 验证每个location的信息
    const auto &retrieved_location1 = *location_map.at(out_location_ids1[0]);
    EXPECT_EQ(retrieved_location1.type(), DataStorageType::DATA_STORAGE_TYPE_NFS);
    EXPECT_EQ(retrieved_location1.spec_size(), 1);
    EXPECT_EQ(retrieved_location1.location_specs().size(), 1);

    const auto &retrieved_location2 = *location_map.at(out_location_ids2[0]);
    EXPECT_EQ(retrieved_location2.type(), DataStorageType::DATA_STORAGE_TYPE_HF3FS);
    EXPECT_EQ(retrieved_location2.spec_size(), 2);
    EXPECT_EQ(retrieved_location2.location_specs().size(), 1);

    const auto &retrieved_location3 = *location_map.at(out_location_ids3[0]);
    EXPECT_EQ(retrieved_location3.type(), DataStorageType::DATA_STORAGE_TYPE_MOONCAKE);
    EXPECT_EQ(retrieved_location3.spec_size(), 3);
    EXPECT_EQ(retrieved_location3.location_specs().size(), 1);
}

TEST_F(MetaSearcherTest, TestBatchVsSequentialPerformance) {
    const size_t num_keys = 100;
    MetaSearcher::KeyVector keys;
    CacheLocationVector locations;

    for (size_t i = 0; i < num_keys; i++) {
        keys.push_back(10000 + i);

        auto location_specs = MetaSearcherTestHelper::CreateDefaultLocationSpecs();
        CacheLocationConstPtr location =
            MetaSearcherTestHelper::CreateCacheLocation(DataStorageType::DATA_STORAGE_TYPE_NFS, 1, location_specs);
        locations.push_back(location);
    }

    std::vector<std::string> out_location_ids;
    ErrorCode ec = meta_searcher_->BatchAddLocation(request_context_.get(), keys, locations, out_location_ids);
    EXPECT_EQ(ec, ErrorCode::EC_OK);

    std::vector<CacheLocationStatus> statuses(out_location_ids.size(), CacheLocationStatus::CLS_SERVING);
    // 构建批量任务，每个key对应一个任务
    std::vector<std::vector<MetaSearcher::LocationUpdateTask>> batch_tasks;
    batch_tasks.reserve(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        std::vector<MetaSearcher::LocationUpdateTask> tasks;
        tasks.push_back({out_location_ids[i], statuses[i]});
        batch_tasks.push_back(tasks);
    }

    std::vector<std::vector<ErrorCode>> out_batch_results;
    ec = meta_searcher_->BatchUpdateLocationStatus(request_context_.get(), keys, batch_tasks, out_batch_results);
    EXPECT_EQ(ec, ErrorCode::EC_OK);
    auto batch_start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i <= 100; ++i) {
        CacheLocationVector out_locations;
        BlockMask mask; // 空mask，不跳过任何元素
        ec = meta_searcher_->PrefixMatch(request_context_.get(), keys, mask, out_locations, &policy_);
        EXPECT_EQ(ec, ErrorCode::EC_OK);
        EXPECT_EQ(out_locations.size(), num_keys);
    }
    auto batch_end = std::chrono::high_resolution_clock::now();
    auto batch_duration = std::chrono::duration_cast<std::chrono::microseconds>(batch_end - batch_start);
    KVCM_LOG_INFO("Batch PrefixMatch duration for 100 runs: %ld ms", batch_duration.count() / 1000);
    KVCM_LOG_INFO("Average per run: %.2f ms", batch_duration.count() / 100000.0);

    auto sequential_start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i <= 100; ++i) {
        CacheLocationVector out_locations;
        BlockMask mask;
        ErrorCode result_ec = ErrorCode::EC_OK;
        for (size_t j = 0; j < keys.size(); ++j) {
            MetaSearcher::KeyVector single_key = {keys[j]};
            result_ec = meta_searcher_->PrefixMatch(request_context_.get(), single_key, mask, out_locations, &policy_);
            if (result_ec != ErrorCode::EC_OK) {
                break;
            }
        }
        EXPECT_EQ(result_ec, ErrorCode::EC_OK);
    }
    auto sequential_end = std::chrono::high_resolution_clock::now();
    auto sequential_duration = std::chrono::duration_cast<std::chrono::microseconds>(sequential_end - sequential_start);
    KVCM_LOG_INFO("Sequential PrefixMatch duration for 100 runs: %ld ms", sequential_duration.count() / 1000);
    KVCM_LOG_INFO("Average per run: %.2f ms", sequential_duration.count() / 100000.0);
}

TEST_F(MetaSearcherTest, TestBatchCASLocationStatus) {
    // 准备测试数据
    MetaSearcher::KeyVector keys = {100, 200, 300};

    // 创建CacheLocation对象
    auto location_specs = MetaSearcherTestHelper::CreateDefaultLocationSpecs();
    CacheLocationConstPtr location1 =
        MetaSearcherTestHelper::CreateCacheLocation(DataStorageType::DATA_STORAGE_TYPE_NFS, 1, location_specs);
    CacheLocationConstPtr location2 =
        MetaSearcherTestHelper::CreateCacheLocation(DataStorageType::DATA_STORAGE_TYPE_HF3FS, 2, location_specs);
    CacheLocationConstPtr location3 =
        MetaSearcherTestHelper::CreateCacheLocation(DataStorageType::DATA_STORAGE_TYPE_MOONCAKE, 3, location_specs);

    CacheLocationVector locations = {location1, location2, location3};

    // 添加位置信息
    std::vector<std::string> out_location_ids;
    ErrorCode ec = meta_searcher_->BatchAddLocation(request_context_.get(), keys, locations, out_location_ids);
    EXPECT_EQ(ec, ErrorCode::EC_OK);
    EXPECT_EQ(out_location_ids.size(), 3);

    // 验证初始状态为CLS_WRITING
    std::vector<CacheLocationMap> out_location_maps;
    BlockMask mask; // 空mask，不跳过任何元素

    ec = meta_searcher_->BatchGetLocation(request_context_.get(), keys, mask, out_location_maps);
    EXPECT_EQ(ec, ErrorCode::EC_OK);
    EXPECT_EQ(out_location_maps.size(), 3);

    for (size_t i = 0; i < out_location_maps.size(); i++) {
        const auto &location_map = out_location_maps[i];
        EXPECT_FALSE(location_map.empty());
        EXPECT_EQ(location_map.size(), 1);

        // 验证location信息
        auto it = location_map.find(out_location_ids[i]);
        EXPECT_NE(it, location_map.end());

        if (it != location_map.end()) {
            const auto &location = *it->second;
            EXPECT_EQ(location.id(), out_location_ids[i]);
            EXPECT_EQ(location.status(), CLS_WRITING); // 初始状态应为CLS_WRITING
        }
    }

    // 准备CAS任务：将状态从WRITING更新为SERVING
    std::vector<std::vector<MetaSearcher::LocationCASTask>> batch_tasks;
    for (size_t i = 0; i < keys.size(); i++) {
        std::vector<MetaSearcher::LocationCASTask> tasks;
        tasks.push_back({out_location_ids[i], CLS_WRITING, CLS_SERVING}); // 从WRITING状态更新为SERVING状态
        batch_tasks.push_back(tasks);
    }

    // 调用BatchCASLocationStatus
    std::vector<std::vector<ErrorCode>> out_batch_results;
    ec = meta_searcher_->BatchCASLocationStatus(request_context_.get(), keys, batch_tasks, out_batch_results);
    EXPECT_EQ(ec, ErrorCode::EC_OK);
    EXPECT_EQ(out_batch_results.size(), 3);

    // 验证每个任务的结果
    for (const auto &results : out_batch_results) {
        EXPECT_EQ(results.size(), 1);            // 每个key只有一个任务
        EXPECT_EQ(results[0], ErrorCode::EC_OK); // 应该成功
    }

    // 验证状态已更新
    out_location_maps.clear();
    ec = meta_searcher_->BatchGetLocation(request_context_.get(), keys, mask, out_location_maps);
    EXPECT_EQ(ec, ErrorCode::EC_OK);
    EXPECT_EQ(out_location_maps.size(), 3);

    for (size_t i = 0; i < out_location_maps.size(); i++) {
        const auto &location_map = out_location_maps[i];
        EXPECT_FALSE(location_map.empty());
        EXPECT_EQ(location_map.size(), 1);

        // 验证location信息
        auto it = location_map.find(out_location_ids[i]);
        EXPECT_NE(it, location_map.end());

        if (it != location_map.end()) {
            const auto &location = *it->second;
            EXPECT_EQ(location.id(), out_location_ids[i]);
            EXPECT_EQ(location.status(), CLS_SERVING); // 状态应已更新为SERVING
            EXPECT_EQ(location.type(), locations[i]->type());
        }
    }

    // 再次尝试CAS操作，这次应该失败，因为状态已经不是WRITING了
    std::vector<std::vector<MetaSearcher::LocationCASTask>> batch_tasks_fail;
    for (size_t i = 0; i < keys.size(); i++) {
        std::vector<MetaSearcher::LocationCASTask> tasks;
        tasks.push_back(
            {out_location_ids[i], CLS_WRITING, CLS_DELETING}); // 从WRITING状态更新为DELETING状态，但实际是SERVING
        batch_tasks_fail.push_back(tasks);
    }

    std::vector<std::vector<ErrorCode>> out_batch_results_fail;
    ec = meta_searcher_->BatchCASLocationStatus(request_context_.get(), keys, batch_tasks_fail, out_batch_results_fail);
    EXPECT_EQ(ec, ErrorCode::EC_OK); // 整体操作成功，但单个任务会失败

    // 验证CAS失败的结果
    for (const auto &results : out_batch_results_fail) {
        EXPECT_EQ(results.size(), 1);                  // 每个key只有一个任务
        EXPECT_EQ(results[0], ErrorCode::EC_MISMATCH); // 应该失败，因为状态不匹配
    }
}

TEST_F(MetaSearcherTest, TestBatchCASLocationStatusChecksExactLocationValue) {
    MetaSearcher::KeyVector keys = {150};
    auto location = MetaSearcherTestHelper::CreateCacheLocation(
        DataStorageType::DATA_STORAGE_TYPE_NFS, 1, MetaSearcherTestHelper::CreateDefaultLocationSpecs());
    CacheLocationVector locations = {location};
    std::vector<std::string> location_ids;
    ASSERT_EQ(EC_OK, meta_searcher_->BatchAddLocation(request_context_.get(), keys, locations, location_ids));
    ASSERT_EQ(1u, location_ids.size());

    std::vector<std::vector<MetaSearcher::LocationCASTask>> mismatch_tasks = {{
        MetaSearcher::LocationCASTask{location_ids[0], CLS_WRITING, CLS_DELETING, "stale serialized location"},
    }};
    std::vector<std::vector<ErrorCode>> results;
    ASSERT_EQ(EC_OK, meta_searcher_->BatchCASLocationStatus(request_context_.get(), keys, mismatch_tasks, results));
    ASSERT_EQ((std::vector<std::vector<ErrorCode>>{{EC_MISMATCH}}), results);

    std::vector<CacheLocationMap> location_maps;
    BlockMask empty_mask;
    ASSERT_EQ(EC_OK, meta_searcher_->BatchGetLocation(request_context_.get(), keys, empty_mask, location_maps));
    ASSERT_EQ(CLS_WRITING, location_maps[0].at(location_ids[0])->status());

    const std::string expected_value = location_maps[0].at(location_ids[0])->ToJsonString();
    std::vector<std::vector<MetaSearcher::LocationCASTask>> matching_tasks = {{
        MetaSearcher::LocationCASTask{location_ids[0], CLS_WRITING, CLS_DELETING, expected_value},
    }};
    ASSERT_EQ(EC_OK, meta_searcher_->BatchCASLocationStatus(request_context_.get(), keys, matching_tasks, results));
    ASSERT_EQ((std::vector<std::vector<ErrorCode>>{{EC_OK}}), results);
}

TEST_F(MetaSearcherTest, TestBatchCADLocationStatus) {
    // 准备测试数据
    MetaSearcher::KeyVector keys = {400, 500, 600};

    // 创建CacheLocation对象
    auto location_specs = MetaSearcherTestHelper::CreateDefaultLocationSpecs();
    CacheLocationConstPtr location1 =
        MetaSearcherTestHelper::CreateCacheLocation(DataStorageType::DATA_STORAGE_TYPE_NFS, 1, location_specs);
    CacheLocationConstPtr location2 =
        MetaSearcherTestHelper::CreateCacheLocation(DataStorageType::DATA_STORAGE_TYPE_HF3FS, 2, location_specs);
    CacheLocationConstPtr location3 =
        MetaSearcherTestHelper::CreateCacheLocation(DataStorageType::DATA_STORAGE_TYPE_MOONCAKE, 3, location_specs);

    CacheLocationVector locations = {location1, location2, location3};

    // 添加位置信息
    std::vector<std::string> out_location_ids;
    ErrorCode ec = meta_searcher_->BatchAddLocation(request_context_.get(), keys, locations, out_location_ids);
    EXPECT_EQ(ec, ErrorCode::EC_OK);
    EXPECT_EQ(out_location_ids.size(), 3);

    // 首先将状态更新为DELETING
    std::vector<CacheLocationStatus> new_statuses = {CLS_DELETING, CLS_DELETING, CLS_DELETING};
    // 构建批量任务，每个key对应一个任务
    std::vector<std::vector<MetaSearcher::LocationUpdateTask>> update_batch_tasks;
    update_batch_tasks.reserve(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        std::vector<MetaSearcher::LocationUpdateTask> tasks;
        tasks.push_back({out_location_ids[i], new_statuses[i]});
        update_batch_tasks.push_back(tasks);
    }

    std::vector<std::vector<ErrorCode>> update_out_batch_results;
    ec = meta_searcher_->BatchUpdateLocationStatus(
        request_context_.get(), keys, update_batch_tasks, update_out_batch_results);
    EXPECT_EQ(ec, ErrorCode::EC_OK);

    // 准备CAD任务：删除状态为DELETING的位置
    std::vector<std::vector<MetaSearcher::LocationCADTask>> batch_tasks;
    for (size_t i = 0; i < keys.size(); i++) {
        std::vector<MetaSearcher::LocationCADTask> tasks;
        tasks.push_back({out_location_ids[i], CLS_DELETING}); // 删除状态为DELETING的位置
        batch_tasks.push_back(tasks);
    }

    // 调用BatchCADLocationStatus
    std::vector<std::vector<ErrorCode>> out_batch_results;
    ec = meta_searcher_->BatchCADLocationStatus(request_context_.get(), keys, batch_tasks, out_batch_results);
    EXPECT_EQ(ec, ErrorCode::EC_OK);
    EXPECT_EQ(out_batch_results.size(), 3);

    // 验证每个任务的结果
    for (const auto &results : out_batch_results) {
        EXPECT_EQ(results.size(), 1);            // 每个key只有一个任务
        EXPECT_EQ(results[0], ErrorCode::EC_OK); // 应该成功
    }

    // 验证位置已被删除
    std::vector<CacheLocationMap> out_location_maps;
    BlockMask mask; // 空mask，不跳过任何元素

    ec = meta_searcher_->BatchGetLocation(request_context_.get(), keys, mask, out_location_maps);
    EXPECT_EQ(ec, ErrorCode::EC_OK);
    EXPECT_EQ(out_location_maps.size(), 3);

    for (const auto &location_map : out_location_maps) {
        // 由于位置已被删除，location_map应该是空的
        EXPECT_TRUE(location_map.empty());
    }

    // 再次尝试CAD操作，这次应该失败，因为位置已经不存在了
    std::vector<std::vector<MetaSearcher::LocationCADTask>> batch_tasks_fail;
    for (size_t i = 0; i < keys.size(); i++) {
        std::vector<MetaSearcher::LocationCADTask> tasks;
        tasks.push_back({out_location_ids[i], CLS_DELETING}); // 尝试删除已不存在的位置
        batch_tasks_fail.push_back(tasks);
    }

    std::vector<std::vector<ErrorCode>> out_batch_results_fail;
    ec = meta_searcher_->BatchCADLocationStatus(request_context_.get(), keys, batch_tasks_fail, out_batch_results_fail);
    EXPECT_EQ(ec, ErrorCode::EC_OK); // location已删除，不存在，会SKIP并返回OK。

    // 验证CAD失败的结果
    for (const auto &results : out_batch_results_fail) {
        EXPECT_EQ(results.size(), 1); // 每个key只有一个任务
        // 位置不存在时，可能返回EC_NOENT或EC_IO_ERROR
        EXPECT_TRUE(results[0] == ErrorCode::EC_NOENT); // 应该失败，因为位置不存在
    }
}

TEST_F(MetaSearcherTest, TestBatchCADLocationStatus2) {
    // 准备测试数据
    MetaSearcher::KeyVector keys = {400, 500, 600};

    // 创建CacheLocation对象
    std::vector<LocationSpec> specs1(
        1, MetaSearcherTestHelper::CreateLocationSpec("nfs", "file:///tmp/test1/test.txt?offset=1&length=2&size=3"));
    CacheLocationConstPtr location1 =
        MetaSearcherTestHelper::CreateCacheLocation(DataStorageType::DATA_STORAGE_TYPE_NFS, 1, specs1);

    std::vector<LocationSpec> specs2(
        1, MetaSearcherTestHelper::CreateLocationSpec("hf3fs", "hf3fs:///tmp/test1/test.txt?offset=1&length=2&size=4"));
    CacheLocationConstPtr location2 =
        MetaSearcherTestHelper::CreateCacheLocation(DataStorageType::DATA_STORAGE_TYPE_HF3FS, 2, specs2);

    std::vector<LocationSpec> specs3(2,
                                     MetaSearcherTestHelper::CreateLocationSpec(
                                         "mooncake", "mooncake:///tmp/test1/test.txt?offset=1&length=2&size=5"));
    CacheLocationConstPtr location3 =
        MetaSearcherTestHelper::CreateCacheLocation(DataStorageType::DATA_STORAGE_TYPE_MOONCAKE, 3, specs3);

    CacheLocationVector locations = {location1, location2, location3};

    // 添加位置信息
    std::vector<std::string> out_location_ids;
    ErrorCode ec = meta_searcher_->BatchAddLocation(request_context_.get(), keys, locations, out_location_ids);
    EXPECT_EQ(ec, ErrorCode::EC_OK);
    EXPECT_EQ(out_location_ids.size(), 3);

    // 首先将状态更新为DELETING
    std::vector<CacheLocationStatus> new_statuses = {CLS_DELETING, CLS_DELETING, CLS_DELETING};
    // 构建批量任务，每个key对应一个任务
    std::vector<std::vector<MetaSearcher::LocationUpdateTask>> update_batch_tasks;
    update_batch_tasks.reserve(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        std::vector<MetaSearcher::LocationUpdateTask> tasks;
        tasks.push_back({out_location_ids[i], new_statuses[i]});
        update_batch_tasks.push_back(tasks);
    }

    std::vector<std::vector<ErrorCode>> update_out_batch_results;
    ec = meta_searcher_->BatchUpdateLocationStatus(
        request_context_.get(), keys, update_batch_tasks, update_out_batch_results);
    EXPECT_EQ(ec, ErrorCode::EC_OK);

    // 准备CAD任务：删除状态为DELETING的位置
    std::vector<std::vector<MetaSearcher::LocationCADTask>> batch_tasks;
    for (size_t i = 0; i < keys.size(); i++) {
        std::vector<MetaSearcher::LocationCADTask> tasks;
        tasks.push_back({out_location_ids[i], CLS_DELETING}); // 删除状态为DELETING的位置
        batch_tasks.push_back(tasks);
    }

    // 调用BatchCADLocationStatus
    std::vector<std::vector<ErrorCode>> out_batch_results;
    ec = meta_searcher_->BatchCADLocationStatus(request_context_.get(), keys, batch_tasks, out_batch_results);
    EXPECT_EQ(ec, ErrorCode::EC_OK);
    EXPECT_EQ(out_batch_results.size(), 3);

    EXPECT_EQ(0, meta_indexer_->GetStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_NFS));
    EXPECT_EQ(0, meta_indexer_->GetStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_HF3FS));
    EXPECT_EQ(0, meta_indexer_->GetStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_MOONCAKE));
}

TEST_F(MetaSearcherTest, TestBatchCASLocationStatusMultipleTasksPerKey) {
    // 测试一个key对应多个CAS任务的情况
    MetaSearcher::KeyVector keys = {700};

    // 创建CacheLocation对象
    auto location_specs = MetaSearcherTestHelper::CreateDefaultLocationSpecs();
    CacheLocationConstPtr location1 =
        MetaSearcherTestHelper::CreateCacheLocation(DataStorageType::DATA_STORAGE_TYPE_NFS, 1, location_specs);

    CacheLocationVector locations = {location1};

    // 添加位置信息
    std::vector<std::string> out_location_ids;
    ErrorCode ec = meta_searcher_->BatchAddLocation(request_context_.get(), keys, locations, out_location_ids);
    EXPECT_EQ(ec, ErrorCode::EC_OK);
    EXPECT_EQ(out_location_ids.size(), 1);

    // 添加第二个位置到同一个key
    CacheLocationConstPtr location2 =
        MetaSearcherTestHelper::CreateCacheLocation(DataStorageType::DATA_STORAGE_TYPE_HF3FS, 1, location_specs);
    std::vector<std::string> out_location_ids2;
    ec = meta_searcher_->BatchAddLocation(request_context_.get(), keys, {location2}, out_location_ids2);
    EXPECT_EQ(ec, ErrorCode::EC_OK);
    EXPECT_EQ(out_location_ids2.size(), 1);

    // 验证现在这个key有两个位置
    std::vector<CacheLocationMap> out_location_maps;
    BlockMask mask;
    ec = meta_searcher_->BatchGetLocation(request_context_.get(), keys, mask, out_location_maps);
    EXPECT_EQ(ec, ErrorCode::EC_OK);
    EXPECT_EQ(out_location_maps.size(), 1);
    EXPECT_EQ(out_location_maps[0].size(), 2); // 应该有两个位置

    // 准备CAS任务：对同一个key的两个位置进行状态更新
    std::vector<std::vector<MetaSearcher::LocationCASTask>> batch_tasks;
    std::vector<MetaSearcher::LocationCASTask> tasks;
    tasks.push_back({out_location_ids[0], CLS_WRITING, CLS_SERVING});   // 更新第一个位置
    tasks.push_back({out_location_ids2[0], CLS_WRITING, CLS_DELETING}); // 更新第二个位置
    batch_tasks.push_back(tasks);

    // 调用BatchCASLocationStatus
    std::vector<std::vector<ErrorCode>> out_batch_results;
    ec = meta_searcher_->BatchCASLocationStatus(request_context_.get(), keys, batch_tasks, out_batch_results);
    EXPECT_EQ(ec, ErrorCode::EC_OK);
    EXPECT_EQ(out_batch_results.size(), 1);
    EXPECT_EQ(out_batch_results[0].size(), 2); // 应该有两个结果

    // 验证结果
    EXPECT_EQ(out_batch_results[0][0], ErrorCode::EC_OK); // 第一个位置更新成功
    EXPECT_EQ(out_batch_results[0][1], ErrorCode::EC_OK); // 第二个位置更新成功

    // 验证状态已更新
    ec = meta_searcher_->BatchGetLocation(request_context_.get(), keys, mask, out_location_maps);
    EXPECT_EQ(ec, ErrorCode::EC_OK);
    EXPECT_EQ(out_location_maps.size(), 1);
    EXPECT_EQ(out_location_maps[0].size(), 2); // 仍然有两个位置

    // 检查每个位置的状态
    auto it1 = out_location_maps[0].find(out_location_ids[0]);
    auto it2 = out_location_maps[0].find(out_location_ids2[0]);
    EXPECT_NE(it1, out_location_maps[0].end());
    EXPECT_NE(it2, out_location_maps[0].end());
    EXPECT_EQ(it1->second->status(), CLS_SERVING);
    EXPECT_EQ(it2->second->status(), CLS_DELETING);
}

TEST_F(MetaSearcherTest, TestBatchCADLocationStatusMultipleTasksPerKey) {
    // 测试一个key对应多个CAD任务的情况
    MetaSearcher::KeyVector keys = {800};

    // 创建CacheLocation对象
    auto location_specs = MetaSearcherTestHelper::CreateDefaultLocationSpecs();
    CacheLocationConstPtr location1 =
        MetaSearcherTestHelper::CreateCacheLocation(DataStorageType::DATA_STORAGE_TYPE_NFS, 1, location_specs);
    CacheLocationConstPtr location2 =
        MetaSearcherTestHelper::CreateCacheLocation(DataStorageType::DATA_STORAGE_TYPE_HF3FS, 1, location_specs);

    // 添加第一个位置信息
    CacheLocationVector locations1 = {location1};
    std::vector<std::string> out_location_ids1;
    ErrorCode ec = meta_searcher_->BatchAddLocation(request_context_.get(), keys, locations1, out_location_ids1);
    EXPECT_EQ(ec, ErrorCode::EC_OK);
    EXPECT_EQ(out_location_ids1.size(), 1);

    // 添加第二个位置到同一个key
    CacheLocationVector locations2 = {location2};
    std::vector<std::string> out_location_ids2;
    ec = meta_searcher_->BatchAddLocation(request_context_.get(), keys, locations2, out_location_ids2);
    EXPECT_EQ(ec, ErrorCode::EC_OK);
    EXPECT_EQ(out_location_ids2.size(), 1);

    // 合并位置ID
    std::vector<std::string> out_location_ids = {out_location_ids1[0], out_location_ids2[0]};

    // 验证位置已添加
    std::vector<CacheLocationMap> out_location_maps;
    BlockMask mask;
    ec = meta_searcher_->BatchGetLocation(request_context_.get(), keys, mask, out_location_maps);
    EXPECT_EQ(ec, ErrorCode::EC_OK);
    EXPECT_EQ(out_location_maps.size(), 1);
    EXPECT_EQ(out_location_maps[0].size(), 2); // 应该有两个位置
    for (const auto &loc_pair : out_location_maps[0]) {
        EXPECT_EQ(loc_pair.second->status(), CLS_WRITING);
    }

    // 将第一个位置的状态都更新为DELETING
    std::vector<CacheLocationStatus> new_statuses1 = {CLS_DELETING};
    // 构建批量任务，每个key对应一个任务
    std::vector<std::vector<MetaSearcher::LocationUpdateTask>> batch_tasks1;
    batch_tasks1.reserve(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        std::vector<MetaSearcher::LocationUpdateTask> tasks;
        if (i < out_location_ids1.size()) {
            tasks.push_back({out_location_ids1[i], new_statuses1[0]});
        }
        batch_tasks1.push_back(tasks);
    }

    std::vector<std::vector<ErrorCode>> out_batch_results1;
    ec = meta_searcher_->BatchUpdateLocationStatus(request_context_.get(), keys, batch_tasks1, out_batch_results1);
    EXPECT_EQ(ec, ErrorCode::EC_OK);

    // 验证状态已更新
    out_location_maps.clear();
    ec = meta_searcher_->BatchGetLocation(request_context_.get(), keys, mask, out_location_maps);
    EXPECT_EQ(ec, ErrorCode::EC_OK);
    EXPECT_EQ(out_location_maps.size(), 1);
    EXPECT_EQ(out_location_maps[0].size(), 2); // 仍然有两个位置
    auto it1 = out_location_maps[0].find(out_location_ids[0]);
    auto it2 = out_location_maps[0].find(out_location_ids[1]);
    EXPECT_NE(it1, out_location_maps[0].end());
    EXPECT_NE(it2, out_location_maps[0].end());
    EXPECT_EQ(it1->second->status(), CLS_DELETING);
    EXPECT_EQ(it2->second->status(), CLS_WRITING);

    // CAD任务：删除状态为DELETING的两个位置
    std::vector<std::vector<MetaSearcher::LocationCADTask>> batch_tasks;
    std::vector<MetaSearcher::LocationCADTask> tasks;
    tasks.push_back({out_location_ids[0], CLS_DELETING}); // 删除第一个位置
    tasks.push_back({out_location_ids[1], CLS_DELETING}); // 删除第二个位置
    batch_tasks.push_back(tasks);
    // 调用BatchCADLocationStatus
    std::vector<std::vector<ErrorCode>> out_batch_results;
    ec = meta_searcher_->BatchCADLocationStatus(request_context_.get(), keys, batch_tasks, out_batch_results);
    EXPECT_EQ(ec, ErrorCode::EC_OK);
    EXPECT_EQ(out_batch_results.size(), 1);
    EXPECT_EQ(out_batch_results[0].size(), 2); // 应该有两个结果
    // 验证结果
    EXPECT_EQ(out_batch_results[0][0], ErrorCode::EC_OK);       // 第一个位置删除成功
    EXPECT_EQ(out_batch_results[0][1], ErrorCode::EC_MISMATCH); // 第二个位置状态不匹配

    // CAD任务：删除状态为WRITING的两个位置
    batch_tasks.clear();
    tasks.clear();
    tasks.push_back({out_location_ids[0], CLS_WRITING}); // 删除第一个位置
    tasks.push_back({out_location_ids[1], CLS_WRITING}); // 删除第二个位置
    batch_tasks.push_back(tasks);
    // 调用BatchCADLocationStatus
    out_batch_results.clear();
    ec = meta_searcher_->BatchCADLocationStatus(request_context_.get(), keys, batch_tasks, out_batch_results);
    EXPECT_EQ(ec, ErrorCode::EC_OK);
    EXPECT_EQ(out_batch_results.size(), 1);
    EXPECT_EQ(out_batch_results[0].size(), 2); // 应该有两个结果
    // 验证结果
    EXPECT_EQ(out_batch_results[0][0], ErrorCode::EC_NOENT); // 第一个位置不存在
    EXPECT_EQ(out_batch_results[0][1], ErrorCode::EC_OK);    // 第二个位置删除成功

    // 验证两个位置均被删除
    out_location_maps.clear();
    ec = meta_searcher_->BatchGetLocation(request_context_.get(), keys, mask, out_location_maps);
    EXPECT_EQ(ec, ErrorCode::EC_OK);
    EXPECT_EQ(out_location_maps.size(), 1);
    EXPECT_TRUE(out_location_maps[0].empty()); // 应该为空
}

TEST_F(MetaSearcherTest, TestBatchCASLocationStatusErrorCases) {
    // 测试错误情况：keys和batch_tasks大小不匹配
    MetaSearcher::KeyVector keys = {900, 901};
    std::vector<std::vector<MetaSearcher::LocationCASTask>> batch_tasks;
    std::vector<MetaSearcher::LocationCASTask> tasks;
    tasks.push_back({"location_id_1", CLS_NEW, CLS_SERVING});
    batch_tasks.push_back(tasks);
    // 注意：这里batch_tasks只有一个元素，而keys有两个元素，应该返回EC_BADARGS

    std::vector<std::vector<ErrorCode>> out_batch_results;
    ErrorCode ec = meta_searcher_->BatchCASLocationStatus(request_context_.get(), keys, batch_tasks, out_batch_results);
    EXPECT_EQ(ec, ErrorCode::EC_BADARGS);
}

TEST_F(MetaSearcherTest, TestBatchCADLocationStatusErrorCases) {
    // 测试错误情况：keys和batch_tasks大小不匹配
    MetaSearcher::KeyVector keys = {1000, 1001};
    std::vector<std::vector<MetaSearcher::LocationCADTask>> batch_tasks;
    std::vector<MetaSearcher::LocationCADTask> tasks;
    tasks.push_back({"location_id_1", CLS_DELETING});
    batch_tasks.push_back(tasks);
    // 注意：这里batch_tasks只有一个元素，而keys有两个元素，应该返回EC_BADARGS

    std::vector<std::vector<ErrorCode>> out_batch_results;
    ErrorCode ec = meta_searcher_->BatchCADLocationStatus(request_context_.get(), keys, batch_tasks, out_batch_results);
    EXPECT_EQ(ec, ErrorCode::EC_BADARGS);
}

TEST_F(MetaSearcherTest, TestPrefixMatchMergesSpecsByStorageType) {
    MetaSearcher::KeyVector keys = {50000, 50001, 50002};

    // Location A: NFS with spec "tp0"
    std::vector<LocationSpec> specs_a = {MetaSearcherTestHelper::CreateLocationSpec("tp0", "nfs:///a/tp0")};
    CacheLocationConstPtr loc_a =
        MetaSearcherTestHelper::CreateCacheLocation(DataStorageType::DATA_STORAGE_TYPE_NFS, 1, specs_a);

    // Location B: NFS with spec "tp1" (same storage type, different spec)
    std::vector<LocationSpec> specs_b = {MetaSearcherTestHelper::CreateLocationSpec("tp1", "nfs:///b/tp1")};
    CacheLocationConstPtr loc_b =
        MetaSearcherTestHelper::CreateCacheLocation(DataStorageType::DATA_STORAGE_TYPE_NFS, 1, specs_b);

    // Add location A to all keys
    CacheLocationVector locations_a = {loc_a, loc_a, loc_a};
    std::vector<std::string> out_ids_a;
    ErrorCode ec = meta_searcher_->BatchAddLocation(request_context_.get(), keys, locations_a, out_ids_a);
    ASSERT_EQ(ec, ErrorCode::EC_OK);
    ASSERT_EQ(out_ids_a.size(), 3);

    // Add location B to all keys
    CacheLocationVector locations_b = {loc_b, loc_b, loc_b};
    std::vector<std::string> out_ids_b;
    ec = meta_searcher_->BatchAddLocation(request_context_.get(), keys, locations_b, out_ids_b);
    ASSERT_EQ(ec, ErrorCode::EC_OK);
    ASSERT_EQ(out_ids_b.size(), 3);

    // Mark all locations as SERVING
    std::vector<std::vector<MetaSearcher::LocationUpdateTask>> batch_tasks;
    for (size_t i = 0; i < keys.size(); ++i) {
        std::vector<MetaSearcher::LocationUpdateTask> tasks;
        tasks.push_back({out_ids_a[i], CLS_SERVING});
        tasks.push_back({out_ids_b[i], CLS_SERVING});
        batch_tasks.push_back(tasks);
    }
    std::vector<std::vector<ErrorCode>> update_results;
    ec = meta_searcher_->BatchUpdateLocationStatus(request_context_.get(), keys, batch_tasks, update_results);
    ASSERT_EQ(ec, ErrorCode::EC_OK);

    // PrefixMatch should merge specs from all same-type locations
    CacheLocationVector out_locations;
    BlockMask mask;
    ec = meta_searcher_->PrefixMatch(request_context_.get(), keys, mask, out_locations, &policy_);
    ASSERT_EQ(ec, ErrorCode::EC_OK);
    ASSERT_EQ(out_locations.size(), 3);

    for (size_t i = 0; i < out_locations.size(); ++i) {
        const auto &loc = out_locations[i];
        // Both locations are NFS, so specs from both should be merged
        ASSERT_EQ(loc->location_specs().size(), 2) << "key index " << i << " should have 2 specs merged from same type";
        EXPECT_EQ(loc->spec_size(), loc->location_specs().size())
            << "spec_size must equal location_specs count after merge";
        EXPECT_EQ(loc->type(), DataStorageType::DATA_STORAGE_TYPE_NFS);
        for (const auto &spec : loc->location_specs()) {
            EXPECT_FALSE(spec.uri().empty());
        }
    }
}

TEST_F(MetaSearcherTest, TestBatchGetMergesSpecsByStorageType) {
    MetaSearcher::KeyVector keys = {60000, 60001};

    // Key 60000: add 2 locations with different specs but SAME storage type (NFS)
    std::vector<LocationSpec> specs_a = {MetaSearcherTestHelper::CreateLocationSpec("tp0", "nfs:///a/tp0")};
    std::vector<LocationSpec> specs_b = {MetaSearcherTestHelper::CreateLocationSpec("tp1", "nfs:///b/tp1")};
    CacheLocationConstPtr loc_a =
        MetaSearcherTestHelper::CreateCacheLocation(DataStorageType::DATA_STORAGE_TYPE_NFS, 1, specs_a);
    CacheLocationConstPtr loc_b =
        MetaSearcherTestHelper::CreateCacheLocation(DataStorageType::DATA_STORAGE_TYPE_NFS, 1, specs_b);

    // Key 60001: add only 1 location
    std::vector<LocationSpec> specs_c = {MetaSearcherTestHelper::CreateLocationSpec("tp0", "mooncake:///c/tp0")};
    CacheLocationConstPtr loc_c =
        MetaSearcherTestHelper::CreateCacheLocation(DataStorageType::DATA_STORAGE_TYPE_MOONCAKE, 1, specs_c);

    // Add loc_a to key 60000, loc_c to key 60001
    {
        CacheLocationVector locs = {loc_a, loc_c};
        std::vector<std::string> out_ids;
        ErrorCode ec = meta_searcher_->BatchAddLocation(request_context_.get(), keys, locs, out_ids);
        ASSERT_EQ(ec, ErrorCode::EC_OK);

        std::vector<std::vector<MetaSearcher::LocationUpdateTask>> tasks;
        for (size_t i = 0; i < keys.size(); ++i) {
            tasks.push_back({{out_ids[i], CLS_SERVING}});
        }
        std::vector<std::vector<ErrorCode>> results;
        ec = meta_searcher_->BatchUpdateLocationStatus(request_context_.get(), keys, tasks, results);
        ASSERT_EQ(ec, ErrorCode::EC_OK);
    }

    // Add loc_b to key 60000 only
    {
        MetaSearcher::KeyVector key_60000 = {60000};
        CacheLocationVector locs = {loc_b};
        std::vector<std::string> out_ids;
        ErrorCode ec = meta_searcher_->BatchAddLocation(request_context_.get(), key_60000, locs, out_ids);
        ASSERT_EQ(ec, ErrorCode::EC_OK);

        std::vector<std::vector<MetaSearcher::LocationUpdateTask>> tasks;
        tasks.push_back({{out_ids[0], CLS_SERVING}});
        std::vector<std::vector<ErrorCode>> results;
        ec = meta_searcher_->BatchUpdateLocationStatus(request_context_.get(), key_60000, tasks, results);
        ASSERT_EQ(ec, ErrorCode::EC_OK);
    }

    // BatchGetBestLocation
    CacheLocationVector out_locations;
    ErrorCode ec = meta_searcher_->BatchGetBestLocation(request_context_.get(), keys, out_locations, &policy_);
    ASSERT_EQ(ec, ErrorCode::EC_OK);
    ASSERT_EQ(out_locations.size(), 2);

    // Key 60000: both locations are NFS, specs should be merged → 2 specs
    {
        const auto &loc = out_locations[0];
        ASSERT_EQ(loc->location_specs().size(), 2);
        EXPECT_EQ(loc->spec_size(), loc->location_specs().size())
            << "spec_size must equal location_specs count after merge";
        EXPECT_EQ(loc->type(), DataStorageType::DATA_STORAGE_TYPE_NFS);
    }

    // Key 60001: single location → 1 spec
    {
        const auto &loc = out_locations[1];
        ASSERT_EQ(loc->location_specs().size(), 1);
        EXPECT_EQ(loc->spec_size(), loc->location_specs().size())
            << "spec_size must equal location_specs count for single location";
        EXPECT_EQ(loc->location_specs()[0].name(), "tp0");
        EXPECT_EQ(loc->type(), DataStorageType::DATA_STORAGE_TYPE_MOONCAKE);
    }
}

// ============================================================
// BatchGetBestLocationByBackend tests
// ============================================================

// Setup 5 keys across 3 event report peers + 1 Tair peer:
//   key 80000: event report peer_a, peer_b    + tair host_t
//   key 80001: event report peer_a, peer_b    + tair host_t
//   key 80002: event report peer_b            + tair host_t
//   key 80003: event report peer_a, peer_b    + tair host_t
//   key 80004: (no event report)              + tair host_t
//
// PREFIX (from key[0]): peer_a covers [80000,80001] then misses 80002 → prefix len 2
//                       peer_b covers [80000,80001,80002,80003] → prefix len 4  → winner = peer_b
// COVERAGE: peer_b covers 4 keys (80000-80003), peer_a covers 3 → winner = peer_b

class BatchGetBestLocationByBackendTest : public MetaSearcherTest {
protected:
    void AddRequestedSpecMatrixEventReportPeer() {
        // The requested spec is deliberately the second spec in the first
        // and third locations. The middle key has the same reporter but only
        // a different spec, forming a requested-spec gap.
        std::vector<std::vector<MetaSearcher::MergeLocationSpecsTask>> upserts = {
            {
                {"kvs#event_report_l2#mem#matrix_peer:8080",
                 DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2,
                 CLS_SERVING,
                 {
                     LocationSpec("full_0", "event_report://matrix_peer:8080/mem"),
                     LocationSpec("linear_1", "event_report://matrix_peer:8080/mem"),
                 }},
            },
            {
                {"kvs#event_report_l2#mem#matrix_peer:8080",
                 DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2,
                 CLS_SERVING,
                 {LocationSpec("full_0", "event_report://matrix_peer:8080/mem")}},
            },
            {
                {"kvs#event_report_l2#mem#matrix_peer:8080",
                 DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2,
                 CLS_SERVING,
                 {
                     LocationSpec("full_0", "event_report://matrix_peer:8080/mem"),
                     LocationSpec("linear_1", "event_report://matrix_peer:8080/mem"),
                 }},
            },
        };
        std::vector<ErrorCode> per_key_ec;
        ASSERT_EQ(ErrorCode::EC_OK,
                  meta_searcher_->BatchMergeLocationSpecs(
                      request_context_.get(), {82000, 82001, 82002}, upserts, per_key_ec));
        ASSERT_EQ(3u, per_key_ec.size());
        EXPECT_TRUE(std::all_of(per_key_ec.begin(), per_key_ec.end(), [](ErrorCode ec) { return ec == EC_OK; }));
    }

    void AddSpecFilteredEventReportPeers() {
        // full_peer has better raw coverage, but none of its locations match
        // linear_1. linear_peer must therefore win after requested-spec
        // filtering for both cross-key selection strategies.
        std::vector<std::vector<MetaSearcher::MergeLocationSpecsTask>> upserts = {
            {
                {"kvs#event_report_l2#mem#full_peer:8080",
                 DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2,
                 CLS_SERVING,
                 {LocationSpec("full_0", "event_report://full_peer:8080/mem")}},
                {"kvs#event_report_l2#mem#linear_peer:8080",
                 DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2,
                 CLS_SERVING,
                 {LocationSpec("linear_1", "event_report://linear_peer:8080/mem")}},
            },
            {
                {"kvs#event_report_l2#mem#full_peer:8080",
                 DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2,
                 CLS_SERVING,
                 {LocationSpec("full_0", "event_report://full_peer:8080/mem")}},
            },
        };
        std::vector<ErrorCode> per_key_ec;
        ASSERT_EQ(ErrorCode::EC_OK,
                  meta_searcher_->BatchMergeLocationSpecs(request_context_.get(), {81000, 81001}, upserts, per_key_ec));
        ASSERT_EQ(2u, per_key_ec.size());
        EXPECT_EQ(ErrorCode::EC_OK, per_key_ec[0]);
        EXPECT_EQ(ErrorCode::EC_OK, per_key_ec[1]);
    }

    void SetUp() override {
        MetaSearcherTest::SetUp();

        // event report locations
        std::vector<std::vector<MetaSearcher::MergeLocationSpecsTask>> er_upserts = {
            // key 80000: peer_a + peer_b
            {
                {"kvs#event_report_l2#mem#peer_a:8080",
                 DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2,
                 CLS_SERVING,
                 {LocationSpec("tp0", "event_report://peer_a:8080/tp0")}},
                {"kvs#event_report_l2#mem#peer_b:8080",
                 DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2,
                 CLS_SERVING,
                 {LocationSpec("tp0", "event_report://peer_b:8080/tp0")}},
            },
            // key 80001: peer_a + peer_b
            {
                {"kvs#event_report_l2#mem#peer_a:8080",
                 DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2,
                 CLS_SERVING,
                 {LocationSpec("tp0", "event_report://peer_a:8080/tp0")}},
                {"kvs#event_report_l2#mem#peer_b:8080",
                 DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2,
                 CLS_SERVING,
                 {LocationSpec("tp0", "event_report://peer_b:8080/tp0")}},
            },
            // key 80002: peer_b only
            {
                {"kvs#event_report_l2#mem#peer_b:8080",
                 DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2,
                 CLS_SERVING,
                 {LocationSpec("tp0", "event_report://peer_b:8080/tp0")}},
            },
            // key 80003: peer_a + peer_b
            {
                {"kvs#event_report_l2#mem#peer_a:8080",
                 DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2,
                 CLS_SERVING,
                 {LocationSpec("tp0", "event_report://peer_a:8080/tp0")}},
                {"kvs#event_report_l2#mem#peer_b:8080",
                 DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2,
                 CLS_SERVING,
                 {LocationSpec("tp0", "event_report://peer_b:8080/tp0")}},
            },
        };
        std::vector<ErrorCode> per_key_ec;
        ErrorCode ec = meta_searcher_->BatchMergeLocationSpecs(
            request_context_.get(), {80000, 80001, 80002, 80003}, er_upserts, per_key_ec);
        ASSERT_EQ(ec, ErrorCode::EC_OK);

        // Tair locations for all 5 keys
        MetaSearcher::KeyVector tair_keys = {80000, 80001, 80002, 80003, 80004};
        for (int64_t key : tair_keys) {
            auto tair_loc = MetaSearcherTestHelper::CreateCacheLocation(
                DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL,
                1,
                {MetaSearcherTestHelper::CreateLocationSpec("tp0", "tair://host_t:6379/tp0")});
            std::vector<std::string> out_ids;
            ec = meta_searcher_->BatchAddLocation(request_context_.get(), {key}, {tair_loc}, out_ids);
            ASSERT_EQ(ec, ErrorCode::EC_OK);
            std::vector<std::vector<MetaSearcher::LocationUpdateTask>> tasks = {{{out_ids[0], CLS_SERVING}}};
            std::vector<std::vector<ErrorCode>> results;
            meta_searcher_->BatchUpdateLocationStatus(request_context_.get(), {key}, tasks, results);
        }
    }
};

TEST_F(BatchGetBestLocationByBackendTest, EventReportPrefixStrategy) {
    MetaSearcher::KeyVector keys = {80000, 80001, 80002, 80003, 80004};
    std::vector<BackendSelector> selectors = {
        {DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2, LocationSelectStrategy::LSS_V6D_PREFIX},
    };

    LocationsPerKey out;
    ErrorCode ec =
        meta_searcher_->BatchGetBestLocationByBackend(request_context_.get(), keys, out, &policy_, selectors);
    ASSERT_EQ(ec, ErrorCode::EC_OK);
    ASSERT_EQ(out.size(), 5);

    // peer_b covers the longest prefix: keys 80000-80003 (4 keys).
    // peer_a only covers 80000-80001 (2 keys, breaks at 80002).
    for (size_t i = 0; i < 4; ++i) {
        ASSERT_EQ(out[i].size(), 1) << "key index " << i << " should have 1 event report location";
        EXPECT_EQ(out[i][0]->type(), DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2);
        // URI should contain peer_b
        EXPECT_NE(out[i][0]->location_specs()[0].uri().find("peer_b"), std::string::npos)
            << "key index " << i << " should be served by peer_b";
    }
    // key 80004 has no event report location
    EXPECT_TRUE(out[4].empty());
}

TEST_F(BatchGetBestLocationByBackendTest, EventReportCoverageStrategy) {
    MetaSearcher::KeyVector keys = {80000, 80001, 80002, 80003, 80004};
    std::vector<BackendSelector> selectors = {
        {DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2, LocationSelectStrategy::LSS_V6D_COVERAGE},
    };

    LocationsPerKey out;
    ErrorCode ec =
        meta_searcher_->BatchGetBestLocationByBackend(request_context_.get(), keys, out, &policy_, selectors);
    ASSERT_EQ(ec, ErrorCode::EC_OK);
    ASSERT_EQ(out.size(), 5);

    // peer_b covers 4 keys (80000-80003), peer_a covers 3 → winner = peer_b
    for (size_t i = 0; i < 4; ++i) {
        ASSERT_EQ(out[i].size(), 1) << "key index " << i;
        EXPECT_NE(out[i][0]->location_specs()[0].uri().find("peer_b"), std::string::npos);
    }
    EXPECT_TRUE(out[4].empty());
}

TEST_F(BatchGetBestLocationByBackendTest, EventReportPrefixTieBreaksByPeerAddress) {
    const std::vector<BackendSelector> selectors = {
        {DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2, LocationSelectStrategy::LSS_V6D_PREFIX},
    };
    LocationsPerKey out;
    ASSERT_EQ(ErrorCode::EC_OK,
              meta_searcher_->BatchGetBestLocationByBackend(
                  request_context_.get(), {80000, 80001}, out, &policy_, selectors));

    ASSERT_EQ(2u, out.size());
    for (const auto &locations : out) {
        ASSERT_EQ(1u, locations.size());
        EXPECT_NE(locations[0]->location_specs()[0].uri().find("peer_a"), std::string::npos);
    }
}

TEST_F(BatchGetBestLocationByBackendTest, EventReportCoverageTieBreaksByPeerAddress) {
    const std::vector<BackendSelector> selectors = {
        {DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2, LocationSelectStrategy::LSS_V6D_COVERAGE},
    };
    LocationsPerKey out;
    ASSERT_EQ(ErrorCode::EC_OK,
              meta_searcher_->BatchGetBestLocationByBackend(
                  request_context_.get(), {80000, 80001}, out, &policy_, selectors));

    ASSERT_EQ(2u, out.size());
    for (const auto &locations : out) {
        ASSERT_EQ(1u, locations.size());
        EXPECT_NE(locations[0]->location_specs()[0].uri().find("peer_a"), std::string::npos);
    }
}

TEST_F(BatchGetBestLocationByBackendTest, EventReportPrefixFiltersRequestedSpecBeforePeerSelection) {
    AddSpecFilteredEventReportPeers();
    const std::vector<BackendSelector> selectors = {
        {DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2, LocationSelectStrategy::LSS_V6D_PREFIX},
    };
    LocationsPerKey out;
    ASSERT_EQ(ErrorCode::EC_OK,
              meta_searcher_->BatchGetBestLocationByBackend(
                  request_context_.get(), {81000, 81001}, out, &policy_, selectors, {"linear_1"}));

    ASSERT_EQ(2u, out.size());
    ASSERT_EQ(1u, out[0].size());
    ASSERT_EQ(1u, out[0][0]->location_specs().size());
    EXPECT_EQ("linear_1", out[0][0]->location_specs()[0].name());
    EXPECT_NE(std::string::npos, out[0][0]->location_specs()[0].uri().find("linear_peer"));
    EXPECT_TRUE(out[1].empty());
}

TEST_F(BatchGetBestLocationByBackendTest, EventReportCoverageFiltersRequestedSpecBeforePeerSelection) {
    AddSpecFilteredEventReportPeers();
    const std::vector<BackendSelector> selectors = {
        {DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2, LocationSelectStrategy::LSS_V6D_COVERAGE},
    };
    LocationsPerKey out;
    ASSERT_EQ(ErrorCode::EC_OK,
              meta_searcher_->BatchGetBestLocationByBackend(
                  request_context_.get(), {81000, 81001}, out, &policy_, selectors, {"linear_1"}));

    ASSERT_EQ(2u, out.size());
    ASSERT_EQ(1u, out[0].size());
    ASSERT_EQ(1u, out[0][0]->location_specs().size());
    EXPECT_EQ("linear_1", out[0][0]->location_specs()[0].name());
    EXPECT_NE(std::string::npos, out[0][0]->location_specs()[0].uri().find("linear_peer"));
    EXPECT_TRUE(out[1].empty());
}

TEST_F(BatchGetBestLocationByBackendTest, EventReportUnknownRequestedSpecReturnsNoCandidate) {
    AddSpecFilteredEventReportPeers();
    for (const auto strategy : {LocationSelectStrategy::LSS_V6D_PREFIX, LocationSelectStrategy::LSS_V6D_COVERAGE}) {
        const std::vector<BackendSelector> selectors = {
            {DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2, strategy},
        };
        LocationsPerKey out;
        ASSERT_EQ(ErrorCode::EC_OK,
                  meta_searcher_->BatchGetBestLocationByBackend(
                      request_context_.get(), {81000, 81001}, out, &policy_, selectors, {"missing_spec"}));
        ASSERT_EQ(2u, out.size());
        EXPECT_TRUE(out[0].empty());
        EXPECT_TRUE(out[1].empty());
    }
}

TEST_F(BatchGetBestLocationByBackendTest, EventReportRequestedSpecMatchesAnyNameIncludingNonFirstSpec) {
    AddRequestedSpecMatrixEventReportPeer();
    for (const auto strategy : {LocationSelectStrategy::LSS_V6D_PREFIX, LocationSelectStrategy::LSS_V6D_COVERAGE}) {
        const std::vector<BackendSelector> selectors = {
            {DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2, strategy},
        };
        LocationsPerKey out;
        ASSERT_EQ(ErrorCode::EC_OK,
                  meta_searcher_->BatchGetBestLocationByBackend(
                      request_context_.get(), {82000}, out, &policy_, selectors, {"missing", "linear_1", "linear_1"}));
        ASSERT_EQ(1u, out.size());
        ASSERT_EQ(1u, out[0].size());
        ASSERT_EQ(2u, out[0][0]->location_specs().size());
        EXPECT_EQ("full_0", out[0][0]->location_specs()[0].name());
        EXPECT_EQ("linear_1", out[0][0]->location_specs()[1].name());
        EXPECT_NE(std::string::npos, out[0][0]->location_specs()[0].uri().find("matrix_peer"));
    }
}

TEST_F(BatchGetBestLocationByBackendTest, EventReportRequestedSpecGapStopsPrefixButNotCoverage) {
    AddRequestedSpecMatrixEventReportPeer();
    const MetaSearcher::KeyVector keys = {82000, 82001, 82002};

    LocationsPerKey prefix_out;
    const std::vector<BackendSelector> prefix_selectors = {
        {DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2, LocationSelectStrategy::LSS_V6D_PREFIX},
    };
    ASSERT_EQ(ErrorCode::EC_OK,
              meta_searcher_->BatchGetBestLocationByBackend(
                  request_context_.get(), keys, prefix_out, &policy_, prefix_selectors, {"linear_1"}));
    ASSERT_EQ(3u, prefix_out.size());
    ASSERT_EQ(1u, prefix_out[0].size());
    EXPECT_TRUE(prefix_out[1].empty());
    EXPECT_TRUE(prefix_out[2].empty());

    LocationsPerKey coverage_out;
    const std::vector<BackendSelector> coverage_selectors = {
        {DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2, LocationSelectStrategy::LSS_V6D_COVERAGE},
    };
    ASSERT_EQ(ErrorCode::EC_OK,
              meta_searcher_->BatchGetBestLocationByBackend(
                  request_context_.get(), keys, coverage_out, &policy_, coverage_selectors, {"linear_1"}));
    ASSERT_EQ(3u, coverage_out.size());
    ASSERT_EQ(1u, coverage_out[0].size());
    EXPECT_TRUE(coverage_out[1].empty());
    ASSERT_EQ(1u, coverage_out[2].size());
}

TEST_F(BatchGetBestLocationByBackendTest, PrefixStopsAtGap) {
    // key 80002 has only peer_b, not peer_a.
    // If we query keys in order [80000, 80002, 80003]:
    //   peer_a: covers 80000, then misses 80002 → prefix = 1
    //   peer_b: covers 80000, 80002, 80003 → prefix = 3  → winner = peer_b
    MetaSearcher::KeyVector keys = {80000, 80002, 80003};
    std::vector<BackendSelector> selectors = {
        {DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2, LocationSelectStrategy::LSS_V6D_PREFIX},
    };

    LocationsPerKey out;
    ErrorCode ec =
        meta_searcher_->BatchGetBestLocationByBackend(request_context_.get(), keys, out, &policy_, selectors);
    ASSERT_EQ(ec, ErrorCode::EC_OK);
    ASSERT_EQ(out.size(), 3);
    for (size_t i = 0; i < 3; ++i) {
        ASSERT_EQ(out[i].size(), 1);
        EXPECT_NE(out[i][0]->location_specs()[0].uri().find("peer_b"), std::string::npos);
    }
}

TEST_F(BatchGetBestLocationByBackendTest, PrefixStopsWhenNoEventReport) {
    // key 80004 has no event report. If it's the first key, PREFIX should stop immediately.
    MetaSearcher::KeyVector keys = {80004, 80000, 80001};
    std::vector<BackendSelector> selectors = {
        {DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2, LocationSelectStrategy::LSS_V6D_PREFIX},
    };

    LocationsPerKey out;
    ErrorCode ec =
        meta_searcher_->BatchGetBestLocationByBackend(request_context_.get(), keys, out, &policy_, selectors);
    ASSERT_EQ(ec, ErrorCode::EC_OK);
    ASSERT_EQ(out.size(), 3);
    // All empty because first key has no event report, PREFIX stops
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_TRUE(out[i].empty()) << "key index " << i << " should be empty";
    }
}

TEST_F(BatchGetBestLocationByBackendTest, CoverageSkipsGap) {
    // COVERAGE skips keys with no event report and picks from the rest.
    MetaSearcher::KeyVector keys = {80004, 80000, 80001};
    std::vector<BackendSelector> selectors = {
        {DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2, LocationSelectStrategy::LSS_V6D_COVERAGE},
    };

    LocationsPerKey out;
    ErrorCode ec =
        meta_searcher_->BatchGetBestLocationByBackend(request_context_.get(), keys, out, &policy_, selectors);
    ASSERT_EQ(ec, ErrorCode::EC_OK);
    ASSERT_EQ(out.size(), 3);
    EXPECT_TRUE(out[0].empty()); // 80004 has no event report
    EXPECT_EQ(out[1].size(), 1); // 80000
    EXPECT_EQ(out[2].size(), 1); // 80001
}

TEST_F(BatchGetBestLocationByBackendTest, WeightedRandomTair) {
    MetaSearcher::KeyVector keys = {80000, 80001, 80002, 80003, 80004};
    std::vector<BackendSelector> selectors = {
        {DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL, LocationSelectStrategy::LSS_WEIGHTED_RANDOM},
    };

    LocationsPerKey out;
    ErrorCode ec =
        meta_searcher_->BatchGetBestLocationByBackend(request_context_.get(), keys, out, &policy_, selectors);
    ASSERT_EQ(ec, ErrorCode::EC_OK);
    ASSERT_EQ(out.size(), 5);
    // All 5 keys have Tair locations
    for (size_t i = 0; i < 5; ++i) {
        ASSERT_EQ(out[i].size(), 1) << "key index " << i;
        EXPECT_EQ(out[i][0]->type(), DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL);
    }
}

TEST_F(BatchGetBestLocationByBackendTest, MixedEventReportAndTair) {
    MetaSearcher::KeyVector keys = {80000, 80001, 80002, 80003, 80004};
    std::vector<BackendSelector> selectors = {
        {DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2, LocationSelectStrategy::LSS_V6D_PREFIX},
        {DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL, LocationSelectStrategy::LSS_WEIGHTED_RANDOM},
    };

    LocationsPerKey out;
    ErrorCode ec =
        meta_searcher_->BatchGetBestLocationByBackend(request_context_.get(), keys, out, &policy_, selectors);
    ASSERT_EQ(ec, ErrorCode::EC_OK);
    ASSERT_EQ(out.size(), 5);

    // Keys 80000-80003: 1 event report (peer_b) + 1 Tair = 2 locations
    for (size_t i = 0; i < 4; ++i) {
        ASSERT_EQ(out[i].size(), 2) << "key index " << i;
        int er_count = 0, tair_count = 0;
        for (const auto &loc : out[i]) {
            if (loc->type() == DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2)
                er_count++;
            if (loc->type() == DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL)
                tair_count++;
        }
        EXPECT_EQ(er_count, 1);
        EXPECT_EQ(tair_count, 1);
    }
    // Key 80004: only Tair (no event report)
    ASSERT_EQ(out[4].size(), 1);
    EXPECT_EQ(out[4][0]->type(), DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL);
}

TEST_F(BatchGetBestLocationByBackendTest, EmptySelectorsBackwardCompat) {
    MetaSearcher::KeyVector keys = {80000};
    std::vector<BackendSelector> selectors = {};

    LocationsPerKey out;
    ErrorCode ec =
        meta_searcher_->BatchGetBestLocationByBackend(request_context_.get(), keys, out, &policy_, selectors);
    ASSERT_EQ(ec, ErrorCode::EC_OK);
    ASSERT_EQ(out.size(), 1);
    // No selectors → no backend processing → empty result per key
    EXPECT_TRUE(out[0].empty());
}

TEST_F(BatchGetBestLocationByBackendTest, NoLocationsAtAll) {
    MetaSearcher::KeyVector keys = {99999}; // nonexistent key
    std::vector<BackendSelector> selectors = {
        {DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2, LocationSelectStrategy::LSS_V6D_PREFIX},
        {DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL, LocationSelectStrategy::LSS_WEIGHTED_RANDOM},
    };

    LocationsPerKey out;
    ErrorCode ec =
        meta_searcher_->BatchGetBestLocationByBackend(request_context_.get(), keys, out, &policy_, selectors);
    ASSERT_EQ(ec, ErrorCode::EC_OK);
    ASSERT_EQ(out.size(), 1);
    EXPECT_TRUE(out[0].empty());
}
