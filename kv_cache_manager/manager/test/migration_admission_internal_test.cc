#include <limits>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/manager/migration_admission_internal.h"

using namespace kv_cache_manager;

namespace {

MigrationCandidateFeatures LastAccess(ObservedFeatureStatus status, std::int64_t value = 0) {
    MigrationCandidateFeatures features;
    ObservedFeature observed;
    observed.status = status;
    if (status == ObservedFeatureStatus::kAvailable) {
        observed.value = value;
    }
    features.Set(MigrationAdmissionFeature::kLastAccessTime, std::move(observed));
    return features;
}

MigrationAdmissionConfig RecentAccessConfig(MigrationAdmissionMode mode, std::int64_t window_seconds) {
    MigrationAdmissionConfig config;
    config.set_mode(mode);
    auto policy = std::make_shared<MigrationAdmissionPolicyConfig>();
    policy->set_recent_access(std::make_shared<RecentAccessAdmissionConfig>(window_seconds));
    config.set_policies({policy});
    return config;
}

} // namespace

TEST(MigrationAdmissionInternalTest, RecentAccessBoundariesAndUnknownReasons) {
    RecentAccessAdmissionPolicy policy(10 * 1000 * 1000);
    const MigrationAdmissionContext context{100 * 1000 * 1000};
    const auto decisions = policy.EvaluateBatch(
        {LastAccess(ObservedFeatureStatus::kAvailable, 95 * 1000 * 1000),
         LastAccess(ObservedFeatureStatus::kAvailable, 90 * 1000 * 1000),
         LastAccess(ObservedFeatureStatus::kAvailable, 89 * 1000 * 1000),
         LastAccess(ObservedFeatureStatus::kAvailable, 101 * 1000 * 1000),
         LastAccess(ObservedFeatureStatus::kMissing),
         LastAccess(ObservedFeatureStatus::kUnsupported),
         LastAccess(ObservedFeatureStatus::kReadError)},
        context);

    ASSERT_EQ(7u, decisions.size());
    EXPECT_EQ(MigrationAdmissionVerdict::kAccept, decisions[0].verdict);
    EXPECT_EQ(MigrationAdmissionVerdict::kAccept, decisions[1].verdict);
    EXPECT_EQ(MigrationAdmissionVerdict::kReject, decisions[2].verdict);
    EXPECT_EQ(MigrationAdmissionReason::kNotRecent, decisions[2].reason);
    EXPECT_EQ(MigrationAdmissionReason::kFeatureInvalid, decisions[3].reason);
    EXPECT_EQ(MigrationAdmissionReason::kFeatureMissing, decisions[4].reason);
    EXPECT_EQ(MigrationAdmissionReason::kFeatureUnsupported, decisions[5].reason);
    EXPECT_EQ(MigrationAdmissionReason::kFeatureReadError, decisions[6].reason);
}

TEST(MigrationAdmissionInternalTest, AvailableFeatureWithWrongTypeIsInvalid) {
    MigrationCandidateFeatures features;
    features.Set(MigrationAdmissionFeature::kLastAccessTime,
                 ObservedFeature{ObservedFeatureStatus::kAvailable, std::uint64_t{10}});
    RecentAccessAdmissionPolicy policy(10);
    const auto decisions = policy.EvaluateBatch({features}, MigrationAdmissionContext{20});
    ASSERT_EQ(1u, decisions.size());
    EXPECT_EQ(MigrationAdmissionVerdict::kUnknown, decisions[0].verdict);
    EXPECT_EQ(MigrationAdmissionReason::kFeatureInvalid, decisions[0].reason);
}

TEST(MigrationAdmissionInternalTest, FactoryOnlyBuildsValidRecentAccess) {
    std::string error;
    auto disabled = MigrationAdmissionPolicyFactory::Build(MigrationAdmissionConfig{}, error);
    EXPECT_EQ(nullptr, disabled);
    EXPECT_TRUE(error.empty());

    auto policy = MigrationAdmissionPolicyFactory::Build(
        RecentAccessConfig(MigrationAdmissionMode::SHADOW, 1), error);
    ASSERT_NE(nullptr, policy);
    EXPECT_TRUE(policy->RequiredFeatures().test(
        static_cast<std::size_t>(MigrationAdmissionFeature::kLastAccessTime)));

    auto overflow = RecentAccessConfig(MigrationAdmissionMode::ENFORCE,
                                       std::numeric_limits<std::int64_t>::max());
    EXPECT_EQ(nullptr, MigrationAdmissionPolicyFactory::Build(overflow, error));
    EXPECT_FALSE(error.empty());

    auto invalid_mode = RecentAccessConfig(static_cast<MigrationAdmissionMode>(99), 1);
    EXPECT_EQ(nullptr, MigrationAdmissionPolicyFactory::Build(invalid_mode, error));
    EXPECT_FALSE(error.empty());
}
