#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/config/cache_config.h"
#include "kv_cache_manager/config/cache_reclaim_strategy.h"
#include "kv_cache_manager/config/instance_group.h"
#include "kv_cache_manager/config/meta_indexer_config.h"
#include "kv_cache_manager/protocol/protobuf/admin_service.pb.h"
#include "kv_cache_manager/service/util/manager_message_proto_util.h"

namespace kv_cache_manager {

class InstanceGroupTest : public TESTBASE {
public:
    void SetUp() override {}
    void TearDown() override {}
};

// --- InstanceGroup set_revisit_interval_buckets ---

TEST_F(InstanceGroupTest, SetValidBuckets) {
    InstanceGroup group;
    group.set_name("test");
    group.set_revisit_interval_buckets("1,5,30,60");
    ASSERT_EQ(group.revisit_interval_buckets().size(), 4);
    EXPECT_DOUBLE_EQ(group.revisit_interval_buckets()[0], 1.0);
    EXPECT_DOUBLE_EQ(group.revisit_interval_buckets()[3], 60.0);
    EXPECT_EQ(group.revisit_interval_buckets_raw(), "1,5,30,60");
}

TEST_F(InstanceGroupTest, SetInvalidBucketsClearsParsed) {
    InstanceGroup group;
    group.set_name("test");
    group.set_revisit_interval_buckets("5,1,30"); // not ascending
    EXPECT_TRUE(group.revisit_interval_buckets().empty());
    EXPECT_EQ(group.revisit_interval_buckets_raw(), "5,1,30"); // raw preserved
}

TEST_F(InstanceGroupTest, SetEmptyBuckets) {
    InstanceGroup group;
    group.set_name("test");
    group.set_revisit_interval_buckets("");
    EXPECT_TRUE(group.revisit_interval_buckets().empty());
}

// --- ValidateRequiredFields revisit_interval_buckets ---

// Helper: validate only the revisit_interval_buckets check in isolation.
// Builds a fully valid InstanceGroup, then sets the buckets to test.
static std::pair<bool, std::string> ValidateBucketsOnly(const std::string &buckets_str) {
    InstanceGroup group;
    group.set_name("test");
    group.set_storage_candidates({"local"});
    group.set_global_quota_group_name("default");
    group.set_max_instance_count(10);
    auto quota = InstanceGroupQuota();
    quota.set_capacity(1024);
    group.set_quota(quota);

    auto reclaim = std::make_shared<CacheReclaimStrategy>();
    reclaim->set_storage_unique_name("local");
    auto meta_config = std::make_shared<MetaIndexerConfig>();
    auto cache_config = std::make_shared<CacheConfig>(CachePreferStrategy::CPS_PREFER_3FS, reclaim, meta_config);
    group.set_cache_config(cache_config);
    group.set_revisit_interval_buckets(buckets_str);

    std::string invalid_fields;
    bool result = group.ValidateRequiredFields(invalid_fields);
    return {result, invalid_fields};
}

TEST_F(InstanceGroupTest, ValidateRejectsInvalidBuckets) {
    auto [valid, invalid_fields] = ValidateBucketsOnly("5,1,30"); // invalid: not ascending
    EXPECT_FALSE(valid);
    EXPECT_NE(invalid_fields.find("revisit_interval_buckets"), std::string::npos);
}

TEST_F(InstanceGroupTest, ValidateAcceptsValidBuckets) {
    auto [valid, invalid_fields] = ValidateBucketsOnly("1,5,30,60");
    EXPECT_TRUE(valid);
}

TEST_F(InstanceGroupTest, ValidateAcceptsEmptyBuckets) {
    auto [valid, invalid_fields] = ValidateBucketsOnly("");
    EXPECT_TRUE(valid);
}

// --- JSON round-trip ---

TEST_F(InstanceGroupTest, JsonRoundTripWithBuckets) {
    std::string json = R"({
        "name": "test_group",
        "storage_candidates": ["local"],
        "global_quota_group_name": "default",
        "max_instance_count": 10,
        "quota": {"capacity": 1024},
        "cache_config": {"meta_indexer_config": {"max_key_count": 1000, "mutex_shard_num": 16, "batch_key_size": 32, "persist_meta_data_interval_time_ms": 1000, "meta_storage_backend_config": {"storage_type": "local"}}},
        "version": 1,
        "revisit_interval_buckets": "1,5,30,60"
    })";

    InstanceGroup parsed;
    ASSERT_TRUE(parsed.FromJsonString(json));
    EXPECT_EQ("test_group", parsed.name());
    EXPECT_EQ("1,5,30,60", parsed.revisit_interval_buckets_raw());
    ASSERT_EQ(parsed.revisit_interval_buckets().size(), 4);
    EXPECT_DOUBLE_EQ(parsed.revisit_interval_buckets()[0], 1.0);
    EXPECT_DOUBLE_EQ(parsed.revisit_interval_buckets()[3], 60.0);
}

TEST_F(InstanceGroupTest, JsonRoundTripWithoutBuckets) {
    std::string json = R"({
        "name": "test_group",
        "storage_candidates": ["local"],
        "global_quota_group_name": "default",
        "max_instance_count": 10,
        "quota": {"capacity": 1024},
        "cache_config": {"meta_indexer_config": {"max_key_count": 1000, "mutex_shard_num": 16, "batch_key_size": 32, "persist_meta_data_interval_time_ms": 1000, "meta_storage_backend_config": {"storage_type": "local"}}},
        "version": 1
    })";

    InstanceGroup parsed;
    ASSERT_TRUE(parsed.FromJsonString(json));
    EXPECT_EQ("test_group", parsed.name());
    EXPECT_TRUE(parsed.revisit_interval_buckets_raw().empty());
    EXPECT_TRUE(parsed.revisit_interval_buckets().empty());
}

TEST_F(InstanceGroupTest, JsonRoundTripInvalidBuckets) {
    std::string json = R"({
        "name": "test_group",
        "storage_candidates": ["local"],
        "global_quota_group_name": "default",
        "max_instance_count": 10,
        "quota": {"capacity": 1024},
        "cache_config": {"meta_indexer_config": {"max_key_count": 1000, "mutex_shard_num": 16, "batch_key_size": 32, "persist_meta_data_interval_time_ms": 1000, "meta_storage_backend_config": {"storage_type": "local"}}},
        "version": 1,
        "revisit_interval_buckets": "5,1,30"
    })";

    InstanceGroup parsed;
    // FromRapidValue is lenient — accepts and preserves raw, but parsed is empty
    ASSERT_TRUE(parsed.FromJsonString(json));
    EXPECT_EQ("5,1,30", parsed.revisit_interval_buckets_raw());
    EXPECT_TRUE(parsed.revisit_interval_buckets().empty());

    // But ValidateRequiredFields should reject it
    std::string invalid_fields;
    EXPECT_FALSE(parsed.ValidateRequiredFields(invalid_fields));
    EXPECT_NE(invalid_fields.find("revisit_interval_buckets"), std::string::npos);
}

// --- Proto round-trip ---

TEST_F(InstanceGroupTest, ProtoRoundTripWithBuckets) {
    InstanceGroup original;
    original.set_name("test_group");
    original.set_storage_candidates({"local"});
    original.set_global_quota_group_name("default");
    original.set_max_instance_count(10);
    original.set_version(1);
    original.set_revisit_interval_buckets("1,5,30,60");

    // ToProto
    proto::admin::InstanceGroup proto_msg;
    ProtoConvert::InstanceGroupToProto(original, &proto_msg);
    EXPECT_EQ("1,5,30,60", proto_msg.revisit_interval_buckets());

    // FromProto
    InstanceGroup restored;
    ProtoConvert::InstanceGroupFromProto(&proto_msg, restored);
    EXPECT_EQ("test_group", restored.name());
    EXPECT_EQ("1,5,30,60", restored.revisit_interval_buckets_raw());
    ASSERT_EQ(restored.revisit_interval_buckets().size(), 4);
    EXPECT_DOUBLE_EQ(restored.revisit_interval_buckets()[0], 1.0);
    EXPECT_DOUBLE_EQ(restored.revisit_interval_buckets()[3], 60.0);
}

TEST_F(InstanceGroupTest, ProtoRoundTripWithoutBuckets) {
    InstanceGroup original;
    original.set_name("test_group");
    original.set_storage_candidates({"local"});
    original.set_global_quota_group_name("default");
    original.set_max_instance_count(10);
    original.set_version(1);
    // revisit_interval_buckets not set

    proto::admin::InstanceGroup proto_msg;
    ProtoConvert::InstanceGroupToProto(original, &proto_msg);
    EXPECT_TRUE(proto_msg.revisit_interval_buckets().empty());

    InstanceGroup restored;
    ProtoConvert::InstanceGroupFromProto(&proto_msg, restored);
    EXPECT_TRUE(restored.revisit_interval_buckets_raw().empty());
    EXPECT_TRUE(restored.revisit_interval_buckets().empty());
}

TEST_F(InstanceGroupTest, CacheConfigProtoRoundTripPreservesMigrationAdmission) {
    CacheConfig original;
    // CacheConfigToProto expects a structurally valid CacheConfig.  The
    // admission assertions below only exercise migration_config, but the
    // surrounding reclaim config must still be present.
    original.set_reclaim_strategy(std::make_shared<CacheReclaimStrategy>());
    auto strategy = std::make_shared<MigrationStrategy>();
    strategy->set_source_storage_name("hot");
    strategy->set_target_storage_name("cold");
    strategy->set_trigger_threshold(0.7);
    strategy->mutable_methods().mutable_copy().set_enabled(true);
    strategy->set_retention(MigrationRetention::MIGRATION_RETENTION_KEEP_BOTH);
    MigrationAdmissionConfig admission;
    admission.set_mode(MigrationAdmissionMode::SHADOW);
    auto policy = std::make_shared<MigrationAdmissionPolicyConfig>();
    policy->set_recent_access(std::make_shared<RecentAccessAdmissionConfig>(3600));
    admission.set_policies({policy});
    strategy->set_admission(admission);
    original.set_migration_strategies({strategy});

    proto::admin::CacheConfig proto_config;
    ProtoConvert::CacheConfigToProto(original, &proto_config);
    ASSERT_EQ(1, proto_config.migration_config().strategies_size());
    const auto &proto_admission = proto_config.migration_config().strategies(0).admission();
    EXPECT_EQ(proto::admin::MIGRATION_ADMISSION_SHADOW, proto_admission.mode());
    ASSERT_EQ(1, proto_admission.policies_size());
    ASSERT_TRUE(proto_admission.policies(0).has_recent_access());
    EXPECT_EQ(3600, proto_admission.policies(0).recent_access().window_seconds());

    CacheConfig restored;
    ProtoConvert::CacheConfigFromProto(&proto_config, restored);
    ASSERT_EQ(1u, restored.migration_strategies().size());
    const auto &restored_admission = restored.migration_strategies()[0]->admission();
    EXPECT_EQ(MigrationAdmissionMode::SHADOW, restored_admission.mode());
    ASSERT_EQ(1u, restored_admission.policies().size());
    ASSERT_NE(nullptr, restored_admission.policies()[0]);
    ASSERT_NE(nullptr, restored_admission.policies()[0]->recent_access());
    EXPECT_EQ(3600, restored_admission.policies()[0]->recent_access()->window_seconds());
}

TEST_F(InstanceGroupTest, EventReportStorageSpecProtoRoundTripPreservesSnapshotSettings) {
    proto::admin::StorageConfig legacy_proto_config;
    legacy_proto_config.set_global_unique_name("legacy_event_report");
    legacy_proto_config.set_storage_type(proto::admin::ST_EVENT_REPORT_L2);
    legacy_proto_config.mutable_event_report();
    StorageConfig legacy_restored;
    ProtoConvert::StorageFromProto(&legacy_proto_config, legacy_restored);
    auto legacy_spec = std::dynamic_pointer_cast<EventReportStorageSpec>(legacy_restored.storage_spec());
    ASSERT_NE(nullptr, legacy_spec);
    EXPECT_EQ(EventReportStorageSpec::kDefaultSnapshotDeltaDrainTimeoutMs,
              legacy_spec->snapshot_delta_drain_timeout_ms());

    auto spec = std::make_shared<EventReportStorageSpec>();
    spec->set_heartbeat_timeout_ms(1234);
    spec->set_cleanup_grace_ms(5678);
    spec->set_liveness_check_interval_ms(90);
    spec->set_snapshot_min_interval_ms(4321);
    spec->set_snapshot_delta_drain_timeout_ms(8765);
    StorageConfig original(DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2, "event_report_test", spec);

    proto::admin::StorageConfig proto_config;
    ProtoConvert::StorageConfigToProto(original, &proto_config);
    ASSERT_TRUE(proto_config.has_event_report());
    EXPECT_EQ(4321, proto_config.event_report().snapshot_min_interval_ms());
    EXPECT_EQ(8765, proto_config.event_report().snapshot_delta_drain_timeout_ms());

    StorageConfig restored;
    ProtoConvert::StorageFromProto(&proto_config, restored);
    auto restored_spec = std::dynamic_pointer_cast<EventReportStorageSpec>(restored.storage_spec());
    ASSERT_NE(nullptr, restored_spec);
    EXPECT_EQ(1234, restored_spec->heartbeat_timeout_ms());
    EXPECT_EQ(5678, restored_spec->cleanup_grace_ms());
    EXPECT_EQ(90, restored_spec->liveness_check_interval_ms());
    EXPECT_EQ(4321, restored_spec->snapshot_min_interval_ms());
    EXPECT_EQ(8765, restored_spec->snapshot_delta_drain_timeout_ms());

    proto::admin::StorageConfig invalid_proto_config;
    invalid_proto_config.set_global_unique_name("invalid_event_report");
    invalid_proto_config.set_storage_type(proto::admin::ST_EVENT_REPORT_L2);
    invalid_proto_config.mutable_event_report()->set_snapshot_delta_drain_timeout_ms(-1);
    StorageConfig invalid_restored;
    ProtoConvert::StorageFromProto(&invalid_proto_config, invalid_restored);
    auto invalid_spec = std::dynamic_pointer_cast<EventReportStorageSpec>(invalid_restored.storage_spec());
    ASSERT_NE(nullptr, invalid_spec);
    EXPECT_EQ(-1, invalid_spec->snapshot_delta_drain_timeout_ms());
    std::string invalid_fields;
    EXPECT_FALSE(invalid_restored.ValidateRequiredFields(invalid_fields));
    EXPECT_NE(std::string::npos, invalid_fields.find("snapshot_delta_drain_timeout_ms"));
}

TEST_F(InstanceGroupTest, UnknownProtoStorageTypeFailsClosed) {
    DataStorageType storage_type = DataStorageType::DATA_STORAGE_TYPE_NFS;
    ProtoConvert::DataStorageTypeFromProto(static_cast<proto::admin::StorageType>(263), storage_type);
    EXPECT_EQ(DataStorageType::DATA_STORAGE_TYPE_UNKNOWN, storage_type);

    proto::admin::CacheLocation proto_location;
    proto_location.set_type(static_cast<proto::admin::StorageType>(263));
    CacheLocation location;
    ProtoConvert::CacheLocationFromProto(&proto_location, location);
    EXPECT_EQ(DataStorageType::DATA_STORAGE_TYPE_UNKNOWN, location.type());
}

TEST_F(InstanceGroupTest, TairMempoolSsdStorageProtoRoundTripPreservesTypeAndMedia) {
    auto spec = std::make_shared<TairMemPoolStorageSpec>();
    spec->set_domain("pace.meta");
    spec->set_timeout(5000);
    spec->set_service_discovery_url("spectrum://pace-meta");
    spec->set_media_type(kTairMemPoolMediaTypeSsd);
    StorageConfig original(DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL_SSD, "pace_ssd_1", spec);

    proto::admin::StorageConfig proto_config;
    ProtoConvert::StorageConfigToProto(original, &proto_config);
    ASSERT_TRUE(proto_config.has_tair_mem_pool());
    EXPECT_EQ(proto::admin::ST_TAIRMEMPOOL_SSD, proto_config.storage_type());
    EXPECT_EQ(kTairMemPoolMediaTypeSsd, proto_config.tair_mem_pool().media_type());

    StorageConfig restored;
    ProtoConvert::StorageFromProto(&proto_config, restored);
    EXPECT_EQ(DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL_SSD, restored.type());
    const auto restored_spec = std::dynamic_pointer_cast<TairMemPoolStorageSpec>(restored.storage_spec());
    ASSERT_NE(nullptr, restored_spec);
    EXPECT_EQ(kTairMemPoolMediaTypeSsd, restored_spec->media_type());
    EXPECT_EQ("spectrum://pace-meta", restored_spec->service_discovery_url());
}

TEST_F(InstanceGroupTest, LegacyTairMempoolProtoWithoutStorageTypeRemainsDramType) {
    proto::admin::StorageConfig legacy;
    legacy.set_global_unique_name("legacy_pace");
    auto *spec = legacy.mutable_tair_mem_pool();
    spec->set_domain("pace.meta");
    spec->set_timeout(5000);
    spec->set_media_type(kTairMemPoolMediaTypeSsd);

    StorageConfig restored;
    ProtoConvert::StorageFromProto(&legacy, restored);
    EXPECT_EQ(DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL, restored.type());
    const auto restored_spec = std::dynamic_pointer_cast<TairMemPoolStorageSpec>(restored.storage_spec());
    ASSERT_NE(nullptr, restored_spec);
    EXPECT_EQ(kTairMemPoolMediaTypeSsd, restored_spec->media_type());
}

TEST_F(InstanceGroupTest, CacheReclaimBudgetPolicyProtoRoundTripPreservesFixedPerInstance) {
    CacheConfig original;
    auto reclaim_strategy = std::make_shared<CacheReclaimStrategy>();
    reclaim_strategy->set_instance_reclaim_budget_policy(InstanceReclaimBudgetPolicy::FIXED_PER_INSTANCE);
    original.set_reclaim_strategy(reclaim_strategy);

    proto::admin::CacheConfig proto_config;
    ProtoConvert::CacheConfigToProto(original, &proto_config);
    EXPECT_EQ(proto::admin::FIXED_PER_INSTANCE, proto_config.reclaim_strategy().instance_reclaim_budget_policy());

    CacheConfig restored;
    ProtoConvert::CacheConfigFromProto(&proto_config, restored);
    ASSERT_NE(nullptr, restored.reclaim_strategy());
    EXPECT_EQ(InstanceReclaimBudgetPolicy::FIXED_PER_INSTANCE,
              restored.reclaim_strategy()->instance_reclaim_budget_policy());
}

TEST_F(InstanceGroupTest, CacheReclaimBudgetPolicyProtoDefaultsToUsageProportional) {
    proto::admin::CacheConfig proto_config;
    proto_config.mutable_reclaim_strategy();

    CacheConfig restored;
    ProtoConvert::CacheConfigFromProto(&proto_config, restored);
    ASSERT_NE(nullptr, restored.reclaim_strategy());
    EXPECT_EQ(InstanceReclaimBudgetPolicy::USAGE_PROPORTIONAL,
              restored.reclaim_strategy()->instance_reclaim_budget_policy());
}

TEST_F(InstanceGroupTest, CacheReclaimBudgetPolicyJsonDefaultsToUsageProportionalAndAcceptsFixedPerInstance) {
    CacheReclaimStrategy reclaim_strategy;
    ASSERT_TRUE(reclaim_strategy.FromJsonString("{}"));
    EXPECT_EQ(InstanceReclaimBudgetPolicy::USAGE_PROPORTIONAL, reclaim_strategy.instance_reclaim_budget_policy());

    ASSERT_TRUE(reclaim_strategy.FromJsonString(R"({"instance_reclaim_budget_policy": 1})"));
    EXPECT_EQ(InstanceReclaimBudgetPolicy::FIXED_PER_INSTANCE, reclaim_strategy.instance_reclaim_budget_policy());
}

TEST_F(InstanceGroupTest, CacheReclaimBudgetPolicyValidationRejectsUnknownValue) {
    CacheReclaimStrategy reclaim_strategy;
    reclaim_strategy.set_storage_unique_name("nfs_01");
    reclaim_strategy.set_instance_reclaim_budget_policy(static_cast<InstanceReclaimBudgetPolicy>(999));

    std::string invalid_fields;
    EXPECT_FALSE(reclaim_strategy.ValidateRequiredFields(invalid_fields));
    EXPECT_NE(std::string::npos, invalid_fields.find("instance_reclaim_budget_policy"));
}

} // namespace kv_cache_manager
