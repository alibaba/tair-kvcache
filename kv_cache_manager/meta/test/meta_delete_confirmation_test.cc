#include <memory>

#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/config/meta_storage_backend_config.h"
#include "kv_cache_manager/meta/meta_local_backend.h"
#include "kv_cache_manager/meta/meta_storage_backend_manager.h"

namespace kv_cache_manager {
namespace {

class FaultyPersistentBackend : public MetaLocalBackend {
public:
    bool Sync(const KeyTypeVec &) noexcept override { return sync_succeeds; }

    std::vector<ErrorCode>
    DeleteLocations(RequestContext *context, const KeyTypeVec &keys, const LocationIdsPerKey &ids) noexcept override {
        return drop_delete ? std::vector<ErrorCode>(keys.size(), EC_OK)
                           : MetaLocalBackend::DeleteLocations(context, keys, ids);
    }

    std::vector<ErrorCode> DeleteLocations(RequestContext *context,
                                           const KeyTypeVec &keys,
                                           const LocationIdsPerKey &ids,
                                           const std::vector<ErrorCode> &) noexcept override {
        return drop_delete ? std::vector<ErrorCode>(keys.size(), EC_TIMEOUT)
                           : MetaLocalBackend::DeleteLocations(context, keys, ids);
    }

    std::vector<std::vector<ErrorCode>> GetLocationsForMaintenance(RequestContext *context,
                                                                   const KeyTypeVec &keys,
                                                                   const LocationIdsPerKey &ids,
                                                                   LocationsPerKey &locations) noexcept override {
        ++read_count;
        auto results = MetaLocalBackend::GetLocationsForMaintenance(context, keys, ids, locations);
        if (read_error != EC_OK) {
            results[0][0] = read_error;
        }
        if (malformed) {
            locations.clear();
        }
        return results;
    }

    bool sync_succeeds = true;
    bool drop_delete = false;
    bool malformed = false;
    ErrorCode read_error = EC_OK;
    int read_count = 0;
};

} // namespace

class MetaDeleteConfirmationTest : public TESTBASE {
protected:
    void SetUp() override {
        auto backend = std::make_unique<FaultyPersistentBackend>();
        ASSERT_EQ(EC_OK, backend->Init("delete-confirmation", config_));
        ASSERT_EQ(EC_OK, backend->Open());
        persistent_ = backend.get();
        manager_.persistent_backend_ = std::move(backend);
        manager_.recover_state_ = MetaStorageBackendManager::RecoverState::kRunning;
        auto target = std::make_shared<CacheLocation>();
        target->set_id("target");
        auto sibling = std::make_shared<CacheLocation>();
        sibling->set_id("sibling");
        locations_ = {{{"target", target}, {"sibling", sibling}}};
        ASSERT_EQ(std::vector<ErrorCode>{EC_OK}, persistent_->Put(&context_, {1}, locations_, {{}}));
    }

    void DeleteTarget() {
        int32_t reclaimed = 0;
        EXPECT_EQ(std::vector<ErrorCode>{EC_OK}, manager_.Delete(&context_, {1}, {{"target"}}, reclaimed));
    }

    RequestContext context_{"delete-confirmation"};
    std::shared_ptr<MetaStorageBackendConfig> config_ = std::make_shared<MetaStorageBackendConfig>();
    MetaStorageBackendManager manager_;
    FaultyPersistentBackend *persistent_ = nullptr;
    CacheLocationMapVector locations_;
};

TEST_F(MetaDeleteConfirmationTest, ConfirmsOnlyDeletedLocationAndPreservesSibling) {
    DeleteTarget();
    EXPECT_TRUE(manager_.SyncDeletedLocations(&context_, {1}, {{"target"}}));
    EXPECT_FALSE(manager_.SyncDeletedLocations(&context_, {1}, {{"sibling"}}));
}

TEST_F(MetaDeleteConfirmationTest, DoesNotAcceptFailedBarrier) {
    DeleteTarget();
    persistent_->sync_succeeds = false;
    EXPECT_FALSE(manager_.SyncDeletedLocations(&context_, {1}, {{"target"}}));
    EXPECT_EQ(0, persistent_->read_count);
}

TEST_F(MetaDeleteConfirmationTest, SuccessfulBarrierDoesNotHideEarlierFailedFlush) {
    persistent_->drop_delete = true;
    DeleteTarget();
    EXPECT_TRUE(manager_.Sync({1}));
    EXPECT_FALSE(manager_.SyncDeletedLocations(&context_, {1}, {{"target"}}));
}

TEST_F(MetaDeleteConfirmationTest, MemoryPrimaryBackupDropDoesNotAuthorizePhysicalDeletion) {
    auto cache = std::make_unique<MetaLocalBackend>();
    ASSERT_EQ(EC_OK, cache->Init("delete-confirmation-cache", config_));
    ASSERT_EQ(EC_OK, cache->Open());
    ASSERT_EQ(std::vector<ErrorCode>{EC_OK}, cache->Put(&context_, {1}, locations_, {{}}));
    manager_.cache_backend_ = std::move(cache);
    manager_.memory_primary_ = true;
    persistent_->drop_delete = true;

    DeleteTarget();
    LocationsPerKey hot_locations;
    ASSERT_EQ(std::vector<std::vector<ErrorCode>>{{EC_NOENT}},
              manager_.cache_backend_->GetLocationsForMaintenance(&context_, {1}, {{"target"}}, hot_locations));
    EXPECT_FALSE(manager_.SyncDeletedLocations(&context_, {1}, {{"target"}}));
}

TEST_F(MetaDeleteConfirmationTest, DoesNotAcceptReadFailure) {
    DeleteTarget();
    persistent_->read_error = EC_TIMEOUT;
    EXPECT_FALSE(manager_.SyncDeletedLocations(&context_, {1}, {{"target"}}));
}

TEST_F(MetaDeleteConfirmationTest, DoesNotAcceptMalformedReadOrMismatchedRequest) {
    DeleteTarget();
    persistent_->malformed = true;
    EXPECT_FALSE(manager_.SyncDeletedLocations(&context_, {1}, {{"target"}}));
    EXPECT_FALSE(manager_.SyncDeletedLocations(&context_, {1}, {}));
}

} // namespace kv_cache_manager
