#pragma once

#include <array>
#include <bitset>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "kv_cache_manager/config/migration_strategy.h"

namespace kv_cache_manager {

// MigrationManager-owned value-admission vocabulary. These types intentionally
// do not depend on MetaIndexer or any storage backend.
enum class MigrationAdmissionFeature : std::uint8_t {
    kLastAccessTime = 0,
    kBusinessAccessCount = 1,
    kFeatureCount,
};

using MigrationAdmissionFeatureSet =
    std::bitset<static_cast<std::size_t>(MigrationAdmissionFeature::kFeatureCount)>;

MigrationAdmissionFeatureSet FeatureSetOf(MigrationAdmissionFeature feature) noexcept;

enum class ObservedFeatureStatus {
    kAvailable,
    kMissing,
    kInvalid,
    kUnsupported,
    kReadError,
};

struct ObservedFeature {
    ObservedFeatureStatus status = ObservedFeatureStatus::kMissing;
    std::variant<std::monostate, std::int64_t, std::uint64_t, double> value;
};

class MigrationCandidateFeatures {
public:
    const ObservedFeature &Get(MigrationAdmissionFeature feature) const noexcept;
    void Set(MigrationAdmissionFeature feature, ObservedFeature observed) noexcept;

    template <typename T>
    std::optional<T> GetAvailableValue(MigrationAdmissionFeature feature) const noexcept {
        const auto &observed = Get(feature);
        if (observed.status != ObservedFeatureStatus::kAvailable) {
            return std::nullopt;
        }
        const auto *value = std::get_if<T>(&observed.value);
        if (value == nullptr) {
            return std::nullopt;
        }
        return *value;
    }

private:
    std::array<ObservedFeature,
               static_cast<std::size_t>(MigrationAdmissionFeature::kFeatureCount)>
        values_{};
};

struct MigrationAdmissionContext {
    std::int64_t now_us = 0;
};

enum class MigrationAdmissionVerdict {
    kAccept,
    kReject,
    kUnknown,
};

enum class MigrationAdmissionReason {
    kSatisfied,
    kNotRecent,
    kInsufficientBusinessAccessCount,
    kFeatureMissing,
    kFeatureInvalid,
    kFeatureUnsupported,
    kFeatureReadError,
};

struct MigrationAdmissionDecision {
    MigrationAdmissionVerdict verdict = MigrationAdmissionVerdict::kUnknown;
    MigrationAdmissionReason reason = MigrationAdmissionReason::kFeatureMissing;
};

class MigrationAdmissionPolicy {
public:
    virtual ~MigrationAdmissionPolicy() = default;
    virtual MigrationAdmissionFeatureSet RequiredFeatures() const noexcept = 0;
    virtual std::vector<MigrationAdmissionDecision>
    EvaluateBatch(const std::vector<MigrationCandidateFeatures> &features,
                  const MigrationAdmissionContext &context) const = 0;
};

class RecentAccessAdmissionPolicy final : public MigrationAdmissionPolicy {
public:
    explicit RecentAccessAdmissionPolicy(std::int64_t window_us)
        : window_us_(window_us) {}

    MigrationAdmissionFeatureSet RequiredFeatures() const noexcept override;
    std::vector<MigrationAdmissionDecision>
    EvaluateBatch(const std::vector<MigrationCandidateFeatures> &features,
                  const MigrationAdmissionContext &context) const override;

private:
    std::int64_t window_us_ = 0;
};

class MigrationAdmissionPolicyFactory {
public:
    static std::unique_ptr<MigrationAdmissionPolicy>
    Build(const MigrationAdmissionConfig &config, std::string &error_message);
};

} // namespace kv_cache_manager
