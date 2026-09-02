#pragma once

#include <cstdint>
#include <vector>

namespace kv_cache_manager {

// Stable manager-layer vocabulary for explaining one migration dispatch batch.
// Protocol conversion stays in service; manager code does not depend on protobuf.
enum class MigrationOutcomeStage : std::uint8_t {
    kSnapshot,
    kValue,
    kExecution,
    kCopy,
    kMark,
};

enum class MigrationOutcomeClass : std::uint8_t {
    kAccepted,
    kRejected,
    kNoopAlreadySatisfied,
    kFailed,
};

enum class MigrationOutcomeReason : std::uint8_t {
    kUnspecified,
    kNotRecent,
    kFeatureMissing,
    kFeatureInvalid,
    kFeatureUnsupported,
    kFeatureReadError,
    kRouteNotReady,
    kLocationReadError,
    kSnapshotShapeError,
    kSourceNotFound,
    kTargetAlreadyCovered,
    kAlreadyMigrating,
    kTargetRejected,
    kSourceRecheckFailed,
    kCopySubmitted,
    kCopySubmitFailed,
    kMarkInserted,
    kMarkAlreadySameTarget,
    kMarkConflictDifferentTarget,
    kMarkMalformed,
    kBlockNotFound,
    kMarkReadError,
    kMarkWriteError,
    kPolicyContractError,
    kBudgetExhausted,
    kValueAccepted,
    kNoExecutionMethod,
    kCopySlotExhausted,
    kDispatchNotAvailable,
};

struct MigrationOutcomeCount {
    MigrationOutcomeStage stage = MigrationOutcomeStage::kSnapshot;
    MigrationOutcomeClass outcome_class = MigrationOutcomeClass::kFailed;
    MigrationOutcomeReason reason = MigrationOutcomeReason::kUnspecified;
    std::int64_t count = 0;
    // Exactly one terminal outcome is emitted for every input candidate. Other
    // entries retain projected SHADOW decisions or a failed Copy before a
    // successful Mark fallback.
    bool terminal = false;
};

using MigrationOutcomeCounts = std::vector<MigrationOutcomeCount>;

} // namespace kv_cache_manager
