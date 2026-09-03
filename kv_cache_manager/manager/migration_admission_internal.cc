#include "kv_cache_manager/manager/migration_admission_internal.h"

#include <limits>

namespace kv_cache_manager {

namespace {

constexpr std::int64_t kMicrosPerSecond = 1000 * 1000;

MigrationAdmissionReason StatusToReason(ObservedFeatureStatus status) noexcept {
    switch (status) {
    case ObservedFeatureStatus::kMissing:
        return MigrationAdmissionReason::kFeatureMissing;
    case ObservedFeatureStatus::kInvalid:
        return MigrationAdmissionReason::kFeatureInvalid;
    case ObservedFeatureStatus::kUnsupported:
        return MigrationAdmissionReason::kFeatureUnsupported;
    case ObservedFeatureStatus::kReadError:
        return MigrationAdmissionReason::kFeatureReadError;
    case ObservedFeatureStatus::kAvailable:
        break;
    }
    return MigrationAdmissionReason::kFeatureInvalid;
}

} // namespace

MigrationAdmissionFeatureSet FeatureSetOf(MigrationAdmissionFeature feature) noexcept {
    MigrationAdmissionFeatureSet features;
    const auto index = static_cast<std::size_t>(feature);
    if (index < features.size()) {
        features.set(index);
    }
    return features;
}

const ObservedFeature &MigrationCandidateFeatures::Get(MigrationAdmissionFeature feature) const noexcept {
    static const ObservedFeature invalid{ObservedFeatureStatus::kInvalid, std::monostate{}};
    const auto index = static_cast<std::size_t>(feature);
    return index < values_.size() ? values_[index] : invalid;
}

void MigrationCandidateFeatures::Set(MigrationAdmissionFeature feature, ObservedFeature observed) noexcept {
    const auto index = static_cast<std::size_t>(feature);
    if (index < values_.size()) {
        values_[index] = std::move(observed);
    }
}

MigrationAdmissionFeatureSet RecentAccessAdmissionPolicy::RequiredFeatures() const noexcept {
    return FeatureSetOf(MigrationAdmissionFeature::kLastAccessTime);
}

std::vector<MigrationAdmissionDecision>
RecentAccessAdmissionPolicy::EvaluateBatch(const std::vector<MigrationCandidateFeatures> &features,
                                           const MigrationAdmissionContext &context) const {
    std::vector<MigrationAdmissionDecision> decisions;
    decisions.reserve(features.size());
    for (const auto &candidate : features) {
        MigrationAdmissionDecision decision;
        const auto &observed = candidate.Get(MigrationAdmissionFeature::kLastAccessTime);
        if (observed.status != ObservedFeatureStatus::kAvailable) {
            decision.reason = StatusToReason(observed.status);
            decisions.push_back(decision);
            continue;
        }
        const auto last_access_time =
            candidate.GetAvailableValue<std::int64_t>(MigrationAdmissionFeature::kLastAccessTime);
        if (!last_access_time || *last_access_time <= 0 || context.now_us <= 0 ||
            *last_access_time > context.now_us || window_us_ <= 0) {
            decision.reason = MigrationAdmissionReason::kFeatureInvalid;
            decisions.push_back(decision);
            continue;
        }
        const std::int64_t age_us = context.now_us - *last_access_time;
        if (age_us <= window_us_) {
            decision.verdict = MigrationAdmissionVerdict::kAccept;
            decision.reason = MigrationAdmissionReason::kSatisfied;
        } else {
            decision.verdict = MigrationAdmissionVerdict::kReject;
            decision.reason = MigrationAdmissionReason::kNotRecent;
        }
        decisions.push_back(decision);
    }
    return decisions;
}

std::unique_ptr<MigrationAdmissionPolicy>
MigrationAdmissionPolicyFactory::Build(const MigrationAdmissionConfig &config,
                                       std::string &error_message) {
    error_message.clear();
    if (config.mode() == MigrationAdmissionMode::DISABLED) {
        return nullptr;
    }
    if (config.mode() != MigrationAdmissionMode::SHADOW &&
        config.mode() != MigrationAdmissionMode::ENFORCE) {
        error_message = "migration admission mode is invalid";
        return nullptr;
    }
    if (config.policies().size() != 1 || config.policies().front() == nullptr ||
        config.policies().front()->recent_access() == nullptr) {
        error_message = "V1 admission requires exactly one recent_access policy";
        return nullptr;
    }
    const std::int64_t window_seconds = config.policies().front()->recent_access()->window_seconds();
    if (window_seconds <= 0 || window_seconds > std::numeric_limits<std::int64_t>::max() / kMicrosPerSecond) {
        error_message = "recent_access.window_seconds is invalid or overflows microseconds";
        return nullptr;
    }
    return std::make_unique<RecentAccessAdmissionPolicy>(window_seconds * kMicrosPerSecond);
}

} // namespace kv_cache_manager
