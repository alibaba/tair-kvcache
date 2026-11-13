#include <cstdlib>
#include <string>
#include <unordered_map>

#include "kv_cache_manager/common/cache/allocation_size.h"
#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/config/meta_storage_backend_config.h"
#include "kv_cache_manager/meta/simple/simple_kv_cache_backend.h"

namespace kv_cache_manager {

class SimpleKVCacheChargeTest : public TESTBASE {
protected:
    void SetUp() override {
        spec_name_ = std::make_shared<const std::string>("spec");
        backend_ = std::make_shared<SimpleKVCacheBackend>(spec_name_);
        auto config = std::make_shared<MetaStorageBackendConfig>();
        config->SetStorageUri("local://?capacity=1&num_shard_bits=0");
        ASSERT_EQ(EC_OK, backend_->Init("charge_test", config));
        ASSERT_EQ(EC_OK, backend_->Open());
    }

    void TearDown() override { EXPECT_EQ(EC_OK, backend_->Close()); }

    CacheLocationMap Locations(size_t count, size_t uri_size) {
        CacheLocationMap result;
        for (size_t i = 0; i < count; ++i) {
            const std::string id = "kvs#event_report_l1p5#hbm#10.0.0." + std::to_string(i + 1) + ":1234";
            InternedLocationId interned = std::make_shared<const std::string>(id);
            ids_[id] = interned;
            auto location = std::make_shared<CacheLocation>();
            location->set_id(interned);
            location->set_type(DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L1P5);
            location->set_status(CLS_SERVING);
            location->set_spec_size(1);
            location->push_location_spec(LocationSpec(spec_name_, std::string(uri_size, 'u')));
            location->set_validated_total_size(4096);
            result[id] = location;
        }
        return result;
    }

    void Create(const CacheLocationMap &locations, std::shared_ptr<SimpleKVCacheItem> &out) {
        Cache::ObjectPtr raw = nullptr;
        ASSERT_EQ(EC_OK, SimpleKVCacheItem::Create(locations, spec_name_, ids_, raw));
        out = std::shared_ptr<SimpleKVCacheItem>(static_cast<SimpleKVCacheItem *>(raw), [](SimpleKVCacheItem *item) {
            SimpleKVCacheItem::Deleter(item, nullptr);
        });
    }

    SharedLocationSpecName spec_name_;
    std::unordered_map<LocationId, InternedLocationId> ids_;
    std::shared_ptr<SimpleKVCacheBackend> backend_;
};

TEST_F(SimpleKVCacheChargeTest, PredictedGrowthMatchesPublishedItemAcrossSizeClasses) {
    std::shared_ptr<SimpleKVCacheItem> current;
    Create(Locations(1, 15), current);
    ASSERT_NE(nullptr, current);
    for (size_t count : {1u, 2u, 5u, 6u, 60u}) {
        for (size_t uri_size : {15u, 16u, 31u, 32u, 48u, 60u, 64u, 80u}) {
            const CacheLocationMap updates = Locations(count, uri_size);
            for (bool replace : {false, true}) {
                size_t growth = 0;
                ASSERT_EQ(EC_OK, SimpleKVCacheItem::EstimateWriteGrowth(current, updates, replace, growth));
                Cache::ObjectPtr raw = nullptr;
                ASSERT_EQ(EC_OK,
                          replace ? SimpleKVCacheItem::Create(updates, spec_name_, ids_, raw)
                                  : SimpleKVCacheItem::Merge(*current, updates, spec_name_, ids_, raw));
                const std::shared_ptr<SimpleKVCacheItem> next(
                    static_cast<SimpleKVCacheItem *>(raw),
                    [](SimpleKVCacheItem *item) { SimpleKVCacheItem::Deleter(item, nullptr); });
                EXPECT_EQ(next->Size() > current->Size() ? next->Size() - current->Size() : 0, growth);
            }
        }
    }
}

TEST_F(SimpleKVCacheChargeTest, LocationRemovalRestoresChargeAndCapacityFailurePreservesValue) {
    const size_t buckets = backend_->GetMemUsage();
    EXPECT_GT(buckets, 0u);
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), backend_->Put(nullptr, {1}, {Locations(1, 60)}, PropertyMapVector(1)));
    const size_t single_usage = backend_->GetMemUsage();
    const CacheLocationMap many = Locations(60, 60);
    ASSERT_EQ(EC_OK, backend_->CheckWriteCapacity(1, many, {}, false));
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), backend_->Upsert(nullptr, {1}, {many}, PropertyMapVector(1)));
    EXPECT_GT(backend_->GetMemUsage(), single_usage);
    LocationIdVector removed;
    for (const CacheLocationMap::value_type &entry : many) {
        if (entry.first != "kvs#event_report_l1p5#hbm#10.0.0.1:1234") {
            removed.push_back(entry.first);
        }
    }
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), backend_->DeleteLocations(nullptr, {1}, {removed}));
    EXPECT_EQ(single_usage, backend_->GetMemUsage());
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}),
              backend_->Upsert(nullptr, {1}, {Locations(1, 15)}, PropertyMapVector(1)));
    EXPECT_LT(backend_->GetMemUsage(), single_usage);
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}),
              backend_->Upsert(nullptr, {1}, {Locations(1, 60)}, PropertyMapVector(1)));
    EXPECT_EQ(single_usage, backend_->GetMemUsage());
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}),
              backend_->DeleteLocations(nullptr, {1}, {{"kvs#event_report_l1p5#hbm#10.0.0.1:1234"}}));
    EXPECT_LT(backend_->GetMemUsage(), single_usage);
    EXPECT_GT(backend_->GetMemUsage(), buckets);
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}),
              backend_->Upsert(nullptr, {1}, {Locations(1, 60)}, PropertyMapVector(1)));
    EXPECT_EQ(single_usage, backend_->GetMemUsage());
    const CacheLocationMap large = Locations(1, 2 * 1024 * 1024);
    EXPECT_EQ(EC_NOSPC, backend_->CheckWriteCapacity(1, large, {}, false));
    EXPECT_EQ((std::vector<ErrorCode>{EC_NOSPC}), backend_->Upsert(nullptr, {1}, {large}, PropertyMapVector(1)));
    EXPECT_EQ(single_usage, backend_->GetMemUsage());
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), backend_->Delete(nullptr, {1}));
    EXPECT_EQ(buckets, backend_->GetMemUsage());
}

TEST_F(SimpleKVCacheChargeTest, JemallocSizeClassesMatchOneAndSixtyLocations) {
    const char *required = std::getenv("KVCM_EXPECT_JEMALLOC");
    if (!required || std::string(required) != "1") {
        GTEST_SKIP() << "Run without ASAN and preload jemalloc to validate allocator-specific charges";
    }
    ASSERT_EQ(32u, EstimateAllocationSize(24));
    ASSERT_EQ(80u, EstimateAllocationSize(79));
    std::shared_ptr<SimpleKVCacheItem> one;
    std::shared_ptr<SimpleKVCacheItem> sixty;
    Create(Locations(1, 60), one);
    Create(Locations(60, 60), sixty);
    ASSERT_NE(nullptr, one);
    ASSERT_NE(nullptr, sixty);
    EXPECT_EQ(160u, one->Size());
    EXPECT_EQ(9216u, sixty->Size());
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), backend_->Put(nullptr, {1}, {Locations(1, 60)}, PropertyMapVector(1)));
    EXPECT_EQ(128u + 288u, backend_->GetMemUsage());
}


} // namespace kv_cache_manager
