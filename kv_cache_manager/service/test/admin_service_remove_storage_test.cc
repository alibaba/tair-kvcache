#include <atomic>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/config/registry_manager.h"
#include "kv_cache_manager/data_storage/storage_config.h"
#include "kv_cache_manager/manager/cache_manager.h"
#include "kv_cache_manager/metrics/local_metrics_reporter.h"
#include "kv_cache_manager/metrics/metrics_registry.h"
#include "kv_cache_manager/service/admin_service_impl.h"

using namespace kv_cache_manager;

class AdminServiceRemoveStorageTest : public TESTBASE {
public:
    void SetUp() override {
        metrics_registry_ = std::make_shared<MetricsRegistry>();
        auto rm_registry = std::make_shared<MetricsRegistry>();
        registry_manager_ = std::make_shared<RegistryManager>("local://", rm_registry);
        registry_manager_->Init();

        cache_manager_ = std::make_shared<CacheManager>(metrics_registry_, registry_manager_);

        admin_service_ =
            std::make_unique<AdminServiceImpl>(cache_manager_, nullptr, metrics_registry_, registry_manager_, nullptr);
        admin_service_->EnableLeaderOnlyRequests();
    }

    void SeedStorage(const std::string &unique_name) {
        auto spec = std::make_shared<DummyStorageSpec>();
        spec->set_root_path("/tmp/test_" + unique_name);
        StorageConfig config(DataStorageType::DATA_STORAGE_TYPE_DUMMY, unique_name, spec);

        RequestContext rc("seed-" + unique_name);
        auto ec = registry_manager_->AddStorage(&rc, config);
        ASSERT_EQ(EC_OK, ec);
    }

    std::shared_ptr<MetricsRegistry> metrics_registry_;
    std::shared_ptr<RegistryManager> registry_manager_;
    std::shared_ptr<CacheManager> cache_manager_;
    std::unique_ptr<AdminServiceImpl> admin_service_;
};

TEST_F(AdminServiceRemoveStorageTest, PurgesUniqueNameTaggedMetrics) {
    SeedStorage("store1");

    // seed metrics: storage-tagged and unrelated
    metrics_registry_->GetGauge("data_storage.healthy_status", {{"unique_name", "store1"}, {"type", "DUMMY"}});
    metrics_registry_->GetGauge("data_storage.storage_usage_ratio", {{"unique_name", "store1"}, {"type", "DUMMY"}});
    metrics_registry_->GetCounter("m1", {{"instance_id", "other"}});
    ASSERT_EQ(3, metrics_registry_->GetSize());

    RequestContext rc("test-trace");
    proto::admin::RemoveStorageRequest request;
    request.set_storage_unique_name("store1");
    proto::admin::CommonResponse response;
    admin_service_->RemoveStorage(&rc, &request, &response);

    ASSERT_EQ(proto::admin::OK, response.header().status().code());

    // storage-tagged metrics should be purged; only "other" remains
    ASSERT_EQ(1, metrics_registry_->GetSize());
    std::vector<MetricsRegistry::metrics_tuple_t> all;
    metrics_registry_->GetAllMetrics(all);
    ASSERT_EQ(1, all.size());
    auto &[name, tags, _] = all[0];
    ASSERT_EQ("other", tags.at("instance_id"));
}

TEST_F(AdminServiceRemoveStorageTest, RemoveNonexistentStorageReturnsError) {
    // no storage seeded — removal should fail
    RequestContext rc("test-trace");
    proto::admin::RemoveStorageRequest request;
    request.set_storage_unique_name("nonexistent");
    proto::admin::CommonResponse response;
    admin_service_->RemoveStorage(&rc, &request, &response);

    ASSERT_EQ(proto::admin::INTERNAL_ERROR, response.header().status().code());
}

TEST_F(AdminServiceRemoveStorageTest, PurgesOnlyTargetedStorageMetrics) {
    SeedStorage("store1");
    SeedStorage("store2");

    // seed metrics tagged with different storage names
    metrics_registry_->GetGauge("data_storage.healthy_status", {{"unique_name", "store1"}, {"type", "DUMMY"}});
    metrics_registry_->GetGauge("data_storage.storage_usage_ratio", {{"unique_name", "store1"}, {"type", "DUMMY"}});
    metrics_registry_->GetGauge("data_storage.healthy_status", {{"unique_name", "store2"}, {"type", "DUMMY"}});
    ASSERT_EQ(3, metrics_registry_->GetSize());

    RequestContext rc("test-trace");
    proto::admin::RemoveStorageRequest request;
    request.set_storage_unique_name("store1");
    proto::admin::CommonResponse response;
    admin_service_->RemoveStorage(&rc, &request, &response);

    ASSERT_EQ(proto::admin::OK, response.header().status().code());

    // only store2 metrics remain
    ASSERT_EQ(1, metrics_registry_->GetSize());
    std::vector<MetricsRegistry::metrics_tuple_t> all;
    metrics_registry_->GetAllMetrics(all);
    ASSERT_EQ(1, all.size());
    auto &[name, tags, _] = all[0];
    ASSERT_EQ("store2", tags.at("unique_name"));
}

// race ReportInterval (reader: shared lifecycle lock) against
// RemoveStorage (writer: unique lifecycle lock) to validate the
// lifecycle lock under contention; mirrors the pattern used for
// instance/group removal concurrency tests
TEST_F(AdminServiceRemoveStorageTest, ConcurrentReportIntervalAndRemoveStorageIsSafe) {
    constexpr int kStorageCount = 20;
    std::vector<std::string> storage_names;
    storage_names.reserve(kStorageCount);
    for (int i = 0; i < kStorageCount; ++i) {
        const auto name = "store_race_" + std::to_string(i);
        SeedStorage(name);
        storage_names.emplace_back(name);
    }
    // a sentinel that must survive the race
    SeedStorage("store_keeper");

    auto reporter = std::make_unique<LocalMetricsReporter>();
    ASSERT_TRUE(reporter->Init(cache_manager_, metrics_registry_, ""));

    std::atomic_bool reporter_stop{false};

    // reader: continuously publish metrics through the reporter; this
    // exercises the shared lifecycle lock around the data storage
    // section that creates DataStorageIntervalMetricsCollector entries
    std::thread reader([&]() {
        while (!reporter_stop.load(std::memory_order_relaxed)) {
            reporter->ReportInterval();
        }
    });

    // writer: remove every targeted storage; each removal acquires the
    // unique lifecycle lock and runs RemoveByTagFilter
    std::thread writer([&]() {
        for (const auto &name : storage_names) {
            RequestContext rc("race-" + name);
            proto::admin::RemoveStorageRequest request;
            request.set_storage_unique_name(name);
            proto::admin::CommonResponse response;
            admin_service_->RemoveStorage(&rc, &request, &response);
            ASSERT_EQ(proto::admin::OK, response.header().status().code());
        }
    });

    writer.join();
    reporter_stop.store(true, std::memory_order_relaxed);
    reader.join();

    // one final report to flush any pending interval metrics
    reporter->ReportInterval();

    // none of the removed storages should have any tagged metrics left
    for (const auto &name : storage_names) {
        std::vector<MetricsRegistry::metrics_tuple_t> all;
        metrics_registry_->GetAllMetrics(all);
        for (const auto &[metric_name, tags, _] : all) {
            auto it = tags.find("unique_name");
            ASSERT_TRUE(it == tags.end() || it->second != name)
                << "leaked metric for removed storage: " << name << " in metric: " << metric_name;
        }
    }

    // the sentinel storage's interval metrics should still be present
    std::vector<MetricsRegistry::metrics_tuple_t> all;
    metrics_registry_->GetAllMetrics(all);
    bool keeper_seen = false;
    for (const auto &[metric_name, tags, _] : all) {
        auto it = tags.find("unique_name");
        if (it != tags.end() && it->second == "store_keeper") {
            keeper_seen = true;
            break;
        }
    }
    ASSERT_TRUE(keeper_seen);
}
