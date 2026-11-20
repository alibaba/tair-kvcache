#include <future>
#include <string>
#include <unordered_map>

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

TEST_F(SimpleKVCacheChargeTest, ChargeIncludesBucketReserveWithoutAllocationMargins) {
    std::shared_ptr<SimpleKVCacheItem> item;
    Create(Locations(1, 15), item);
    ASSERT_NE(nullptr, item);
    EXPECT_EQ(127u, item->Size());

    const CacheLocationMap single = Locations(1, 60);
    Create(single, item);
    ASSERT_NE(nullptr, item);
    EXPECT_EQ(172u, item->Size());
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), backend_->Put(nullptr, {1}, {single}, PropertyMapVector(1)));
    EXPECT_EQ(299u, backend_->GetMemUsage());

    Create(Locations(60, 60), item);
    ASSERT_NE(nullptr, item);
    EXPECT_EQ(8904u, item->Size());
}

TEST_F(SimpleKVCacheChargeTest, ReadViewsPinOldVersionsAcrossMutationAndBackendDestruction) {
    CompactLocationsPerKey first(true);
    CompactLocationsPerKey second(true);
    std::weak_ptr<const std::string> name_owner;
    const KeyType keys[]{1, 2};
    {
        auto name = std::make_shared<const std::string>("spec");
        name_owner = name;
        auto backend = std::make_shared<SimpleKVCacheBackend>(name);
        auto config = std::make_shared<MetaStorageBackendConfig>();
        config->SetStorageUri("local://?capacity=1&num_shard_bits=0");
        ASSERT_EQ(EC_OK, backend->Init("view_lifetime", config));
        ASSERT_EQ(EC_OK, backend->Open());
        ASSERT_EQ((std::vector<ErrorCode>{EC_OK}),
                  backend->Put(nullptr, {1}, {Locations(1, 80)}, PropertyMapVector(1)));
        const size_t usage = backend->GetMemUsage();
        ASSERT_EQ((std::vector<ErrorCode>{EC_OK, EC_NOENT}),
                  backend->GetLocationValuesCompact(nullptr, keys, 2, first));
        ASSERT_EQ((std::vector<ErrorCode>{EC_OK, EC_NOENT}),
                  backend->GetLocationValuesCompact(nullptr, keys, 2, second));
        ASSERT_TRUE(first.IsValid(2));
        EXPECT_TRUE(first.values.empty());
        ASSERT_EQ(1u, first.ReadViews(0).size());
        EXPECT_TRUE(first.ReadViews(1).empty());
        const CacheLocationReadView old = *first.ReadViews(0).begin();
        EXPECT_EQ(old.SpecUri(0).data(), (*second.ReadViews(0).begin()).SpecUri(0).data());
        EXPECT_EQ(usage, backend->GetMemUsage());
        const CacheLocationMap replacement = Locations(1, 96);
        std::future<bool> mutation = std::async(std::launch::async, [&] {
            if (backend->Upsert(nullptr, {1}, {replacement}, PropertyMapVector(1)) != std::vector<ErrorCode>{EC_OK}) {
                return false;
            }
            if (backend->GetLocationValuesCompact(nullptr, keys, 2, second) !=
                std::vector<ErrorCode>{EC_OK, EC_NOENT}) {
                return false;
            }
            return backend->Delete(nullptr, {1}) == std::vector<ErrorCode>{EC_OK};
        });
        const std::string old_uri(80, 'u');
        for (size_t i = 0; i < 1000; ++i) {
            EXPECT_EQ(old_uri, old.SpecUri(0));
        }
        ASSERT_TRUE(mutation.get());
        EXPECT_EQ(old_uri, old.SpecUri(0));
        EXPECT_EQ(std::string(96, 'u'), (*second.ReadViews(0).begin()).SpecUri(0));
        ASSERT_EQ(EC_OK, backend->Close());
    }
    EXPECT_FALSE(name_owner.expired());
    const CacheLocationReadView old = *first.ReadViews(0).begin();
    const CacheLocationReadView updated = *second.ReadViews(0).begin();
    EXPECT_EQ("spec", old.SpecName(0));
    EXPECT_EQ("kvs#event_report_l1p5#hbm#10.0.0.1:1234", old.id());
    EXPECT_EQ(DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L1P5, old.type());
    EXPECT_EQ(CLS_SERVING, old.status());
    EXPECT_EQ(1u, old.SpecSize());
    EXPECT_EQ(1u, old.SpecCount());
    EXPECT_TRUE(old.SpecChecksum(0).empty());
    EXPECT_EQ(std::string(80, 'u'), old.SpecUri(0));
    EXPECT_EQ(std::string(96, 'u'), updated.SpecUri(0));
    first.Clear(0, 0);
    second.Clear(0, 0);
    EXPECT_TRUE(name_owner.expired());
}

TEST_F(SimpleKVCacheChargeTest, PutIfAbsentPreservesExistingValuesAndSkipErrors) {
    const CacheLocationMap original = Locations(1, 60);
    const CacheLocationMap replacement = Locations(1, 80);
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), backend_->PutIfAbsent(nullptr, {1}, {original}, PropertyMapVector(1)));
    const size_t original_usage = backend_->GetMemUsage();
    EXPECT_EQ((std::vector<ErrorCode>{EC_EXIST}),
              backend_->PutIfAbsent(nullptr, {1}, {replacement}, PropertyMapVector(1)));
    EXPECT_EQ(original_usage, backend_->GetMemUsage());
    EXPECT_EQ((std::vector<ErrorCode>{EC_EXIST, EC_OK, EC_NOSPC, EC_EXIST}),
              backend_->PutIfAbsent(nullptr,
                                    {1, 2, 3, 2},
                                    CacheLocationMapVector(4, replacement),
                                    PropertyMapVector(4),
                                    {EC_OK, EC_OK, EC_NOSPC, EC_OK}));

    CacheLocationMapVector locations;
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK, EC_OK, EC_NOENT}), backend_->GetLocations(nullptr, {1, 2, 3}, locations));
    ASSERT_EQ(3u, locations.size());
    ASSERT_EQ(1u, locations[0].size());
    ASSERT_EQ(1u, locations[1].size());
    ASSERT_NE(nullptr, locations[0].begin()->second);
    ASSERT_NE(nullptr, locations[1].begin()->second);
    ASSERT_EQ(1u, locations[0].begin()->second->location_specs().size());
    ASSERT_EQ(1u, locations[1].begin()->second->location_specs().size());
    EXPECT_EQ(std::string(60, 'u'), locations[0].begin()->second->location_specs()[0].uri());
    EXPECT_EQ(std::string(80, 'u'), locations[1].begin()->second->location_specs()[0].uri());
}

TEST_F(SimpleKVCacheChargeTest, PredictedGrowthMatchesPublishedItemAcrossSizes) {
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

TEST_F(SimpleKVCacheChargeTest, ChecksumOnlyUpdatesConsumeCapacityAndPreservePinnedReadViews) {
    const CacheLocationMap initial = Locations(1, 90);
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), backend_->Put(nullptr, {1}, {initial}, PropertyMapVector(1)));
    const size_t initial_charge = backend_->GetMemUsage();
    const KeyType key = 1;
    CompactLocationsPerKey before(true);
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), backend_->GetLocationValuesCompact(nullptr, &key, 1, before));

    auto location = std::make_shared<CacheLocation>(*initial.begin()->second);
    const std::string checksum("crc\0test", 8);
    location->mutable_location_specs().front().set_checksum(checksum);
    location->set_validated_total_size(4096);
    const CacheLocationMap update{{location->id(), location}};
    ASSERT_EQ(EC_OK, backend_->CheckWriteCapacity(1, update, {}, false));
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), backend_->Upsert(nullptr, {1}, {update}, PropertyMapVector(1)));
    EXPECT_EQ(initial_charge + checksum.size(), backend_->GetMemUsage());

    CompactLocationsPerKey after(true);
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), backend_->GetLocationValuesCompact(nullptr, &key, 1, after));
    const CacheLocationReadView old = *before.ReadViews(0).begin();
    const CacheLocationReadView current = *after.ReadViews(0).begin();
    EXPECT_TRUE(old.SpecChecksum(0).empty());
    EXPECT_EQ(old.SpecUri(0), current.SpecUri(0));
    EXPECT_EQ(checksum, current.SpecChecksum(0));

    auto oversized = std::make_shared<CacheLocation>(*location);
    oversized->mutable_location_specs().front().set_checksum(std::string(1024 * 1024, 'h'));
    oversized->set_validated_total_size(4096);
    const CacheLocationMap rejected{{oversized->id(), oversized}};
    EXPECT_EQ(EC_NOSPC, backend_->CheckWriteCapacity(1, rejected, {}, false));
    EXPECT_EQ((std::vector<ErrorCode>{EC_NOSPC}), backend_->Upsert(nullptr, {1}, {rejected}, PropertyMapVector(1)));
    EXPECT_EQ(initial_charge + checksum.size(), backend_->GetMemUsage());

    CacheLocationMapVector read;
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), backend_->GetLocations(nullptr, {1}, read));
    EXPECT_EQ(checksum, read.front().at(location->id())->location_specs().front().checksum());
}

TEST_F(SimpleKVCacheChargeTest, LocationRemovalRestoresChargeAndCapacityFailurePreservesValue) {
    const size_t initial_usage = backend_->GetMemUsage();
    EXPECT_EQ(0u, initial_usage);
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
    EXPECT_GT(backend_->GetMemUsage(), initial_usage);
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}),
              backend_->Upsert(nullptr, {1}, {Locations(1, 60)}, PropertyMapVector(1)));
    EXPECT_EQ(single_usage, backend_->GetMemUsage());
    const CacheLocationMap large = Locations(1, 2 * 1024 * 1024);
    EXPECT_EQ(EC_NOSPC, backend_->CheckWriteCapacity(1, large, {}, false));
    EXPECT_EQ((std::vector<ErrorCode>{EC_NOSPC}), backend_->Upsert(nullptr, {1}, {large}, PropertyMapVector(1)));
    EXPECT_EQ(single_usage, backend_->GetMemUsage());
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), backend_->Delete(nullptr, {1}));
    EXPECT_EQ(initial_usage, backend_->GetMemUsage());
}

} // namespace kv_cache_manager
