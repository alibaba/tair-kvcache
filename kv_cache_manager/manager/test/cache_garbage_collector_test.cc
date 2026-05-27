#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "kv_cache_manager/common/error_code.h"
#include "kv_cache_manager/common/timestamp_util.h"
#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/config/instance_group.h"
#include "kv_cache_manager/config/instance_info.h"
#include "kv_cache_manager/config/model_deployment.h"
#include "kv_cache_manager/config/registry_manager.h"
#include "kv_cache_manager/data_storage/data_storage_manager.h"
#include "kv_cache_manager/manager/cache_garbage_collector.h"
#include "kv_cache_manager/manager/schedule_plan_executor.h"
#include "kv_cache_manager/manager/write_location_manager.h"
#include "kv_cache_manager/meta/cache_location.h"
#include "kv_cache_manager/meta/common.h"
#include "kv_cache_manager/meta/meta_indexer.h"
#include "kv_cache_manager/meta/meta_indexer_manager.h"
#include "kv_cache_manager/metrics/metrics_registry.h"
#include "stub.h"

using namespace kv_cache_manager;

/* ---------------- RegistryManager stubs ---------------- */

using ins_group_ptr_vec = std::vector<std::shared_ptr<const InstanceGroup>>;
ins_group_ptr_vec gc_instance_groups;

std::pair<ErrorCode, ins_group_ptr_vec> GC_ListInstanceGroup_stub(void *obj, RequestContext *rc) {
    return {ErrorCode::EC_OK, gc_instance_groups};
}

using ins_info_ptr_vec = std::vector<std::shared_ptr<const InstanceInfo>>;
ins_info_ptr_vec gc_instance_infos;

std::pair<ErrorCode, ins_info_ptr_vec>
GC_ListInstanceInfo_stub(void *obj, RequestContext *rc, const std::string &ig) {
    return {ErrorCode::EC_OK, gc_instance_infos};
}

/* ---------------- MetaIndexerManager stub ---------------- */

std::shared_ptr<MetaIndexer> gc_dummy_meta_indexer;

std::shared_ptr<MetaIndexer> GC_GetMetaIndexer_stub(void *obj, const std::string &i) {
    return gc_dummy_meta_indexer;
}

/* ---------------- MetaIndexer::Scan stub ---------------- */

KeyVector gc_scan_keys;
int gc_scan_call_count;

ErrorCode GC_Scan_stub(void *obj,
                       RequestContext *rc,
                       const std::string &cursor,
                       const std::size_t limit,
                       std::string &out_next_cursor,
                       KeyVector &out_keys) noexcept {
    ++gc_scan_call_count;
    if (cursor == SCAN_BASE_CURSOR && !gc_scan_keys.empty()) {
        out_keys = gc_scan_keys;
        out_next_cursor = SCAN_BASE_CURSOR;
    } else {
        out_keys.clear();
        out_next_cursor = SCAN_BASE_CURSOR;
    }
    return ErrorCode::EC_OK;
}

/* ---------------- MetaIndexer::GetLocations stub ---------------- */

CacheLocationMapVector gc_location_maps;

MetaIndexer::Result GC_GetLocations_stub(void *obj,
                                         RequestContext *rc,
                                         const KeyVector &keys,
                                         CacheLocationMapVector &out_location_maps) noexcept {
    out_location_maps = gc_location_maps;
    return MetaIndexer::Result(ErrorCode::EC_OK);
}

/* ---------------- SchedulePlanExecutor::SubmitNonBlocking stub ---------------- */

using spe_submit_nb_loc = bool (SchedulePlanExecutor::*)(const CacheLocationDelRequest &);
std::vector<CacheLocationDelRequest> gc_submitted_del_requests;

bool GC_SubmitNonBlocking_stub(void *obj, const CacheLocationDelRequest &request) {
    gc_submitted_del_requests.push_back(request);
    return true;
}

/* ---------------- WriteLocationManager::HasLocationId stub ---------------- */

std::vector<std::string> gc_active_location_ids;

bool GC_HasLocationId_stub(void *obj, const std::string &location_id) {
    for (const auto &id : gc_active_location_ids) {
        if (id == location_id) return true;
    }
    return false;
}

/* ---------------- DataStorageManager::Exist stub ---------------- */

std::vector<bool> gc_exist_results;

std::vector<bool>
GC_DSM_Exist_stub(void *obj, const std::string &name, const std::vector<DataStorageUri> &uris, bool fastpath) {
    if (!gc_exist_results.empty()) {
        return gc_exist_results;
    }
    return std::vector<bool>(uris.size(), true);
}

/* ---------------- RegistryManager::data_storage_manager stub ---------------- */

std::shared_ptr<DataStorageManager> gc_dsm;

std::shared_ptr<DataStorageManager> GC_data_storage_manager_stub(void *obj) { return gc_dsm; }

/* ---------------- Test fixture ---------------- */

class CacheGarbageCollectorTest : public TESTBASE {
public:
    void SetUp() override {
        stub_.set(ADDR(RegistryManager, ListInstanceGroup), GC_ListInstanceGroup_stub);
        stub_.set(ADDR(RegistryManager, ListInstanceInfo), GC_ListInstanceInfo_stub);
        stub_.set(ADDR(MetaIndexerManager, GetMetaIndexer), GC_GetMetaIndexer_stub);
        stub_.set(ADDR(MetaIndexer, Scan), GC_Scan_stub);
        stub_.set(ADDR(WriteLocationManager, HasLocationId), GC_HasLocationId_stub);
        stub_.set(ADDR(DataStorageManager, Exist), GC_DSM_Exist_stub);
        stub_.set(ADDR(RegistryManager, data_storage_manager), GC_data_storage_manager_stub);

        // stub the overload: GetLocations(rc, keys, CacheLocationMapVector&)
        using get_loc_t = MetaIndexer::Result (MetaIndexer::*)(RequestContext *, const KeyVector &,
                                                               CacheLocationMapVector &) noexcept;
        stub_.set(static_cast<get_loc_t>(ADDR(MetaIndexer, GetLocations)), GC_GetLocations_stub);

        stub_.set(static_cast<spe_submit_nb_loc>(ADDR(SchedulePlanExecutor, SubmitNonBlocking)),
                  GC_SubmitNonBlocking_stub);

        auto ig = std::make_shared<InstanceGroup>();
        ig->set_name("test_group");
        gc_instance_groups = {ig};

        auto ii = std::make_shared<InstanceInfo>();
        ii->set_instance_id("test_instance");
        ii->set_instance_group_name("test_group");
        gc_instance_infos = {ii};

        gc_dummy_meta_indexer = std::make_shared<MetaIndexer>();
        gc_scan_keys.clear();
        gc_scan_call_count = 0;
        gc_location_maps.clear();
        gc_submitted_del_requests.clear();
        gc_active_location_ids.clear();
        gc_exist_results.clear();

        mr_ = std::make_shared<MetricsRegistry>();
        rm_ = std::make_shared<RegistryManager>("", mr_);
        mim_ = std::make_shared<MetaIndexerManager>();
        gc_dsm = std::make_shared<DataStorageManager>(mr_);
        spe_ = std::make_shared<SchedulePlanExecutor>(0, mim_, gc_dsm, mr_);
        wlm_ = std::make_shared<WriteLocationManager>();
    }

    void TearDown() override {
        gc_instance_groups.clear();
        gc_instance_infos.clear();
        gc_dummy_meta_indexer.reset();
        gc_scan_keys.clear();
        gc_location_maps.clear();
        gc_submitted_del_requests.clear();
        gc_active_location_ids.clear();
        gc_exist_results.clear();
        gc_dsm.reset();
    }

    std::unique_ptr<CacheGarbageCollector> MakeGC(CacheGarbageCollector::Config config = {}) {
        config.inter_round_sleep_ms = 10;
        config.inter_batch_sleep_ms = 0;
        return std::make_unique<CacheGarbageCollector>(config, rm_, mim_, spe_, mr_, nullptr, wlm_);
    }

    static CacheLocationConstPtr MakeLocation(const std::string &id,
                                              CacheLocationStatus status,
                                              std::int64_t create_time_us = 0) {
        auto loc = std::make_shared<CacheLocation>();
        loc->set_id(id);
        loc->set_status(status);
        loc->set_create_time(create_time_us);
        return loc;
    }

protected:
    Stub stub_;
    std::shared_ptr<MetricsRegistry> mr_;
    std::shared_ptr<RegistryManager> rm_;
    std::shared_ptr<MetaIndexerManager> mim_;
    std::shared_ptr<SchedulePlanExecutor> spe_;
    std::shared_ptr<WriteLocationManager> wlm_;
};

/* ---------------- Tests ---------------- */

TEST_F(CacheGarbageCollectorTest, StartStopBasic) {
    auto gc = MakeGC();
    ASSERT_EQ(gc->Start(), ErrorCode::EC_OK);
    ASSERT_TRUE(gc->IsRunning());
    gc->Stop();
    ASSERT_FALSE(gc->IsRunning());
}

TEST_F(CacheGarbageCollectorTest, DisabledDoesNotStart) {
    CacheGarbageCollector::Config config;
    config.enabled = false;
    auto gc = MakeGC(config);
    ASSERT_EQ(gc->Start(), ErrorCode::EC_OK);
    ASSERT_FALSE(gc->IsRunning());
}

TEST_F(CacheGarbageCollectorTest, DoubleStartReturnsExist) {
    auto gc = MakeGC();
    ASSERT_EQ(gc->Start(), ErrorCode::EC_OK);
    ASSERT_EQ(gc->Start(), ErrorCode::EC_EXIST);
    gc->Stop();
}

TEST_F(CacheGarbageCollectorTest, DetectsOrphanedWriting) {
    // orphaned: CLS_WRITING, not in WriteLocationManager, old enough
    std::int64_t old_time = TimestampUtil::GetCurrentTimeUs() - 700 * 1000000LL;
    gc_scan_keys = {100};
    CacheLocationMap loc_map;
    loc_map["loc_orphan"] = MakeLocation("loc_orphan", CacheLocationStatus::CLS_WRITING, old_time);
    gc_location_maps = {loc_map};

    auto gc = MakeGC();
    ASSERT_EQ(gc->Start(), ErrorCode::EC_OK);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    gc->Stop();

    ASSERT_FALSE(gc_submitted_del_requests.empty());
    ASSERT_EQ(gc_submitted_del_requests[0].block_keys[0], 100);
    ASSERT_EQ(gc_submitted_del_requests[0].location_ids[0][0], "loc_orphan");
}

TEST_F(CacheGarbageCollectorTest, GracePeriodProtectsYoungWriting) {
    // young CLS_WRITING should not be collected
    std::int64_t recent_time = TimestampUtil::GetCurrentTimeUs() - 10 * 1000000LL;
    gc_scan_keys = {200};
    CacheLocationMap loc_map;
    loc_map["loc_young"] = MakeLocation("loc_young", CacheLocationStatus::CLS_WRITING, recent_time);
    gc_location_maps = {loc_map};

    auto gc = MakeGC();
    ASSERT_EQ(gc->Start(), ErrorCode::EC_OK);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    gc->Stop();

    ASSERT_TRUE(gc_submitted_del_requests.empty());
}

TEST_F(CacheGarbageCollectorTest, ActiveWriteSessionProtected) {
    // CLS_WRITING but tracked in WriteLocationManager
    std::int64_t old_time = TimestampUtil::GetCurrentTimeUs() - 700 * 1000000LL;
    gc_scan_keys = {300};
    gc_active_location_ids = {"loc_active"};
    CacheLocationMap loc_map;
    loc_map["loc_active"] = MakeLocation("loc_active", CacheLocationStatus::CLS_WRITING, old_time);
    gc_location_maps = {loc_map};

    auto gc = MakeGC();
    ASSERT_EQ(gc->Start(), ErrorCode::EC_OK);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    gc->Stop();

    ASSERT_TRUE(gc_submitted_del_requests.empty());
}

TEST_F(CacheGarbageCollectorTest, DetectsStaleServing) {
    gc_scan_keys = {400};
    auto loc = std::make_shared<CacheLocation>();
    loc->set_id("loc_stale");
    loc->set_status(CacheLocationStatus::CLS_SERVING);
    loc->push_location_spec(LocationSpec("spec1", "file://storage_01/path/to/data"));
    CacheLocationMap loc_map;
    loc_map["loc_stale"] = loc;
    gc_location_maps = {loc_map};

    // Exist returns false — data is missing
    gc_exist_results = {false};

    auto gc = MakeGC();
    ASSERT_EQ(gc->Start(), ErrorCode::EC_OK);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    gc->Stop();

    ASSERT_FALSE(gc_submitted_del_requests.empty());
    ASSERT_EQ(gc_submitted_del_requests[0].location_ids[0][0], "loc_stale");
}

TEST_F(CacheGarbageCollectorTest, HealthyServingNotCollected) {
    gc_scan_keys = {500};
    auto loc = std::make_shared<CacheLocation>();
    loc->set_id("loc_healthy");
    loc->set_status(CacheLocationStatus::CLS_SERVING);
    loc->push_location_spec(LocationSpec("spec1", "file://storage_01/path/to/data"));
    CacheLocationMap loc_map;
    loc_map["loc_healthy"] = loc;
    gc_location_maps = {loc_map};

    // Exist returns true — data present
    gc_exist_results = {true};

    auto gc = MakeGC();
    ASSERT_EQ(gc->Start(), ErrorCode::EC_OK);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    gc->Stop();

    ASSERT_TRUE(gc_submitted_del_requests.empty());
}

TEST_F(CacheGarbageCollectorTest, PauseStopsScan) {
    gc_scan_keys = {600};
    std::int64_t old_time = TimestampUtil::GetCurrentTimeUs() - 700 * 1000000LL;
    CacheLocationMap loc_map;
    loc_map["loc_pause"] = MakeLocation("loc_pause", CacheLocationStatus::CLS_WRITING, old_time);
    gc_location_maps = {loc_map};

    auto gc = MakeGC();
    gc->Pause();
    ASSERT_EQ(gc->Start(), ErrorCode::EC_OK);
    std::this_thread::sleep_for(std::chrono::milliseconds(80));

    // nothing submitted while paused
    ASSERT_TRUE(gc_submitted_del_requests.empty());

    gc->Resume();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    gc->Stop();

    ASSERT_FALSE(gc_submitted_del_requests.empty());
}

TEST_F(CacheGarbageCollectorTest, MaxDeletionsPerBatchRespected) {
    CacheGarbageCollector::Config config;
    config.max_deletions_per_batch = 1;

    // provide two keys with dirty locations
    std::int64_t old_time = TimestampUtil::GetCurrentTimeUs() - 700 * 1000000LL;
    gc_scan_keys = {700, 701};
    CacheLocationMap loc_map0;
    loc_map0["loc_a"] = MakeLocation("loc_a", CacheLocationStatus::CLS_WRITING, old_time);
    CacheLocationMap loc_map1;
    loc_map1["loc_b"] = MakeLocation("loc_b", CacheLocationStatus::CLS_WRITING, old_time);
    gc_location_maps = {loc_map0, loc_map1};

    auto gc = MakeGC(config);
    ASSERT_EQ(gc->Start(), ErrorCode::EC_OK);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    gc->Stop();

    // at most 1 deletion per batch submission
    ASSERT_FALSE(gc_submitted_del_requests.empty());
    ASSERT_EQ(gc_submitted_del_requests[0].block_keys.size(), 1u);
}
