#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/config/meta_indexer_config.h"
#include "kv_cache_manager/meta/cache_location.h"
#include "kv_cache_manager/meta/meta_indexer.h"
#include "kv_cache_manager/meta/types.h"

using namespace kv_cache_manager;

namespace {

// Build a CacheLocationMap with one location whose spec carries a node_id.
CacheLocationMap MakeNodeLocation(const std::string &loc_id, const std::string &node_id) {
    auto loc = std::make_shared<CacheLocation>();
    loc->set_id(loc_id);
    loc->set_status(CacheLocationStatus::CLS_SERVING);
    loc->set_type(DataStorageType::DATA_STORAGE_TYPE_HF3FS);
    loc->set_spec_size(1);
    std::vector<LocationSpec> specs;
    specs.emplace_back("default", "uri://" + loc_id, node_id);
    loc->set_location_specs(std::move(specs));
    CacheLocationMap map;
    map.emplace(loc_id, std::move(loc));
    return map;
}

CacheLocationConstPtr MakeSizedNodeLocation(const std::string &id,
                                            const std::string &node,
                                            uint64_t size,
                                            DataStorageType type,
                                            CacheLocationStatus status = CLS_SERVING) {
    auto location = std::make_shared<CacheLocation>();
    location->set_id(id);
    location->set_status(status);
    location->set_type(type);
    location->set_spec_size(1);
    location->push_location_spec(LocationSpec("default", "file:///cache?size=" + std::to_string(size), node));
    return location;
}

} // namespace

class MetaIndexerNodeLruTest : public TESTBASE {
public:
    void SetUp() override {
        meta_indexer_ = std::make_shared<MetaIndexer>();
        request_context_ = std::make_shared<RequestContext>("test_trace_id");
    }

    void TearDown() override {}

    ErrorCode InitIndexer(bool pure_local = false) {
        // Use cached mode: local (cache) + dummy (persistent) + node_lru reclaim indexer.
        std::string config_str = R"({
            "max_key_count" : 1000,
            "mutex_shard_num" : 8,
            "meta_storage_backend_config" : { "storage_type" : "cached" }
        })";
        auto config = std::make_shared<MetaIndexerConfig>();
        config->FromJsonString(config_str);
        if (pure_local) {
            config->meta_storage_backend_config_->SetStorageType("local");
        }
        std::string local_path = GetPrivateTestRuntimeDataPath() + "meta_node_lru_test";
        config->meta_storage_backend_config_->SetStorageUri(
            "file://" + local_path + "?reclaim_indexer_type=node_lru&persistent_type=dummy&cache_type=local");
        return meta_indexer_->Init("test_instance", config);
    }

protected:
    std::shared_ptr<MetaIndexer> meta_indexer_;
    std::shared_ptr<RequestContext> request_context_;
};

// After Put with node-aware locations, SampleReclaimKeys("node_lru", {node})
// should return exactly those keys belonging to the specified node.
TEST_F(MetaIndexerNodeLruTest, PutThenSampleByNode) {
    ASSERT_EQ(EC_OK, InitIndexer());

    // Put 4 keys: key 0,1 on nodeA; key 2,3 on nodeB.
    KeyVector keys = {100, 101, 200, 201};
    CacheLocationMapVector locations = {
        MakeNodeLocation("loc_100", "nodeA"),
        MakeNodeLocation("loc_101", "nodeA"),
        MakeNodeLocation("loc_200", "nodeB"),
        MakeNodeLocation("loc_201", "nodeB"),
    };
    PropertyMapVector properties(4); // empty properties
    auto result = meta_indexer_->Put(request_context_.get(), keys, locations, properties);
    ASSERT_EQ(EC_OK, result.ec);
    ASSERT_EQ(4u, meta_indexer_->GetKeyCount());

    // Sample from nodeA — should get keys 100 and 101.
    KeyVector sampled_a;
    ASSERT_EQ(EC_OK, meta_indexer_->SampleReclaimKeys(request_context_.get(), "node_lru", {"nodeA"}, 10, sampled_a));
    std::sort(sampled_a.begin(), sampled_a.end());
    EXPECT_EQ((KeyVector{100, 101}), sampled_a);

    // Sample from nodeB — should get keys 200 and 201.
    KeyVector sampled_b;
    ASSERT_EQ(EC_OK, meta_indexer_->SampleReclaimKeys(request_context_.get(), "node_lru", {"nodeB"}, 10, sampled_b));
    std::sort(sampled_b.begin(), sampled_b.end());
    EXPECT_EQ((KeyVector{200, 201}), sampled_b);

    // Sample from both nodes at once.
    KeyVector sampled_all;
    ASSERT_EQ(
        EC_OK,
        meta_indexer_->SampleReclaimKeys(request_context_.get(), "node_lru", {"nodeA", "nodeB"}, 10, sampled_all));
    std::sort(sampled_all.begin(), sampled_all.end());
    EXPECT_EQ((KeyVector{100, 101, 200, 201}), sampled_all);
}

// Sample from an unknown node returns empty (not an error).
TEST_F(MetaIndexerNodeLruTest, SampleUnknownNodeReturnsEmpty) {
    ASSERT_EQ(EC_OK, InitIndexer());

    KeyVector keys = {1};
    CacheLocationMapVector locations = {MakeNodeLocation("loc_1", "nodeA")};
    PropertyMapVector properties(1);
    meta_indexer_->Put(request_context_.get(), keys, locations, properties);

    KeyVector sampled;
    ASSERT_EQ(EC_OK, meta_indexer_->SampleReclaimKeys(request_context_.get(), "node_lru", {"nodeX"}, 10, sampled));
    EXPECT_TRUE(sampled.empty());
}

// Delete removes keys from the node_lru indexer — sample should no longer return them.
TEST_F(MetaIndexerNodeLruTest, DeleteRemovesFromNodeLru) {
    ASSERT_EQ(EC_OK, InitIndexer());

    KeyVector keys = {10, 20, 30};
    CacheLocationMapVector locations = {
        MakeNodeLocation("loc_10", "nodeA"),
        MakeNodeLocation("loc_20", "nodeA"),
        MakeNodeLocation("loc_30", "nodeA"),
    };
    PropertyMapVector properties(3);
    meta_indexer_->Put(request_context_.get(), keys, locations, properties);

    // Delete key 20.
    meta_indexer_->Delete(request_context_.get(), KeyVector{20});

    KeyVector sampled;
    ASSERT_EQ(EC_OK, meta_indexer_->SampleReclaimKeys(request_context_.get(), "node_lru", {"nodeA"}, 10, sampled));
    std::sort(sampled.begin(), sampled.end());
    EXPECT_EQ((KeyVector{10, 30}), sampled);
}

// LRU ordering: after a Get (touch), the accessed key moves to hot end.
// Sample should return the non-touched keys first.
TEST_F(MetaIndexerNodeLruTest, GetTouchesLruOrder) {
    ASSERT_EQ(EC_OK, InitIndexer());

    // Insert keys in order: 1, 2, 3. Oldest = 1.
    KeyVector keys = {1, 2, 3};
    CacheLocationMapVector locations = {
        MakeNodeLocation("loc_1", "nodeA"),
        MakeNodeLocation("loc_2", "nodeA"),
        MakeNodeLocation("loc_3", "nodeA"),
    };
    PropertyMapVector properties(3);
    meta_indexer_->Put(request_context_.get(), keys, locations, properties);

    // Touch key 1 via Get — it should move to the hot end.
    CacheLocationMapVector out_locations;
    PropertyMapVector out_properties;
    meta_indexer_->Get(request_context_.get(), KeyVector{1}, out_locations, out_properties);

    // Sample 1 key — should be key 2 (now oldest).
    KeyVector sampled;
    ASSERT_EQ(EC_OK, meta_indexer_->SampleReclaimKeys(request_context_.get(), "node_lru", {"nodeA"}, 1, sampled));
    ASSERT_EQ(1u, sampled.size());
    EXPECT_EQ(2, sampled[0]);
}

// Main's local metadata RMW optimization must keep the node-index hooks.
TEST_F(MetaIndexerNodeLruTest, PureLocalTargetedUpsertAndDeleteKeepNodeIndex) {
    ASSERT_EQ(EC_OK, InitIndexer(true));
    EXPECT_FALSE(meta_indexer_->SupportsSingleLocationRmw());
    CacheLocationMapVector locations = {MakeNodeLocation("loc_a", "nodeA")};
    PropertyMapVector properties(1);
    ASSERT_EQ(EC_OK, meta_indexer_->Put(request_context_.get(), {42}, locations, properties).ec);
    const auto modifier = [](const std::vector<ErrorCode> &, const LocationIdVector &, size_t,
                             CacheLocationVector &out, PropertyMap &) -> LocationModifierResult {
        out = {MakeNodeLocation("loc_b", "nodeB").at("loc_b")};
        return {MA_OK, {EC_OK}};
    };
    ASSERT_EQ(EC_OK, meta_indexer_->ReadModifyWriteTargetLocations(
                         request_context_.get(), {42}, {{"loc_b"}}, modifier).ec);
    KeyVector keys;
    ASSERT_EQ(EC_OK, meta_indexer_->SampleReclaimKeys(request_context_.get(), "node_lru", {"nodeB"}, 10, keys));
    EXPECT_EQ((KeyVector{42}), keys);
    ASSERT_EQ(EC_OK, meta_indexer_->Delete(request_context_.get(), {42}).ec);
    ASSERT_EQ(EC_OK, meta_indexer_->SampleReclaimKeys(request_context_.get(), "node_lru", {"nodeB"}, 10, keys));
    EXPECT_TRUE(keys.empty());
}

TEST_F(MetaIndexerNodeLruTest, PersistentRestartRebuildsNodeMembershipAndReconcilesUsage) {
    for (const bool stale_metadata : {false, true}) {
        SCOPED_TRACE(stale_metadata ? "stale counters" : "missing counters");
        auto config = std::make_shared<MetaIndexerConfig>();
        ASSERT_TRUE(config->FromJsonString(R"({
            "max_key_count": 1000,
            "mutex_shard_num": 8,
            "persist_metadata_interval_time_ms": 0,
            "meta_storage_backend_config": {"storage_type": "dummy"}
        })"));
        const std::string instance = stale_metadata ? "stale" : "missing";
        const std::string path = GetPrivateTestRuntimeDataPath() + "restart_node_lru";
        std::filesystem::remove(path + "_" + instance);
        config->meta_storage_backend_config_->SetStorageUri("file://" + path + "?reclaim_indexer_type=node_lru");

        // Persist real locations independently of the periodic counters,
        // as happens when a process exits before the next metadata snapshot.
        {
            MetaStorageBackendManager writer;
            ASSERT_EQ(EC_OK, writer.Init(instance, config->GetMetaStorageBackendConfig()));
            ASSERT_EQ(EC_OK, writer.Open());
            BatchMetaData batch;
            batch.batch_keys = {1, 2, 3};
            batch.batch_locations = {
                {{"base", MakeSizedNodeLocation("base", "nodeA", 100, DataStorageType::DATA_STORAGE_TYPE_HF3FS)},
                 {"alias", MakeSizedNodeLocation("alias", "nodeA", 25, DataStorageType::DATA_STORAGE_TYPE_VCNS_HF3FS)}},
                {{"writing",
                  MakeSizedNodeLocation(
                      "writing", "nodeB", 300, DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL_SSD, CLS_WRITING)}},
                {{"deleting",
                  MakeSizedNodeLocation(
                      "deleting", "nodeA", 40, DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2, CLS_DELETING)}},
            };
            auto multi_spec = std::make_shared<CacheLocation>(*batch.batch_locations[0].at("base"));
            multi_spec->set_spec_size(2);
            multi_spec->push_location_spec(LocationSpec("second", "file:///cache-second?size=50", "nodeC"));
            batch.batch_locations[0]["base"] = std::move(multi_spec);
            batch.batch_properties.resize(3);
            ASSERT_EQ((std::vector<ErrorCode>{EC_OK, EC_OK, EC_OK}), writer.Put(nullptr, batch));
            if (stale_metadata) {
                StorageUsageData stale_usage;
                stale_usage.Reset();
                stale_usage.SetStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_HF3FS, 9999);
                ASSERT_EQ(EC_OK,
                          writer.PutMetaData({{METADATA_PROPERTY_KEY_COUNT, "99"},
                                              {METADATA_PROPERTY_STORAGE_USAGE_DATA, stale_usage.Serialize()}}));
            }
            ASSERT_EQ(EC_OK, writer.Close());
        }

        meta_indexer_ = std::make_shared<MetaIndexer>();
        ASSERT_EQ(EC_OK, meta_indexer_->Init(instance, config));
        EXPECT_EQ(3u, meta_indexer_->GetKeyCount());
        EXPECT_EQ(175u, meta_indexer_->GetStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_HF3FS));
        EXPECT_EQ(175u, meta_indexer_->GetStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_VCNS_HF3FS));
        EXPECT_EQ(300u, meta_indexer_->GetStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL_SSD));
        EXPECT_EQ(40u, meta_indexer_->GetStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2));
        EXPECT_EQ(475u, meta_indexer_->GetStorageUsage());
        KeyTypeVec sampled;
        ASSERT_EQ(EC_OK, meta_indexer_->SampleReclaimKeys(nullptr, "node_lru", {"nodeA"}, 10, sampled));
        std::sort(sampled.begin(), sampled.end());
        EXPECT_EQ((KeyTypeVec{1, 3}), sampled);
        ASSERT_EQ(EC_OK, meta_indexer_->SampleReclaimKeys(nullptr, "node_lru", {"nodeB"}, 10, sampled));
        EXPECT_EQ((KeyTypeVec{2}), sampled);
        ASSERT_EQ(EC_OK, meta_indexer_->SampleReclaimKeys(nullptr, "node_lru", {"nodeC"}, 10, sampled));
        EXPECT_EQ((KeyTypeVec{1}), sampled);

        // The direct indexer API does not adjust usage (MetaSearcher does).
        // A second restart must recompute, not double-count the prior snapshot.
        ASSERT_EQ(EC_OK, meta_indexer_->Delete(request_context_.get(), {2}).ec);
        meta_indexer_.reset();
        meta_indexer_ = std::make_shared<MetaIndexer>();
        ASSERT_EQ(EC_OK, meta_indexer_->Init(instance, config));
        EXPECT_EQ(2u, meta_indexer_->GetKeyCount());
        EXPECT_EQ(175u, meta_indexer_->GetStorageUsage());
        EXPECT_EQ(0u, meta_indexer_->GetStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL_SSD));
        ASSERT_EQ(EC_OK, meta_indexer_->SampleReclaimKeys(nullptr, "node_lru", {"nodeB"}, 10, sampled));
        EXPECT_TRUE(sampled.empty());
        meta_indexer_.reset();
        std::filesystem::remove(path + "_" + instance);
    }
}
