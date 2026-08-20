#include "kv_cache_manager/optimizer/liteHit/lite_hit.h"

#include <algorithm>
#include <stdexcept>

namespace kv_cache_manager {

LiteHit::LiteHit() : LiteHit(CacheObjectConfig{}) {}

LiteHit::LiteHit(const CacheObjectConfig &object_config)
    : object_config_(object_config)
    , pool_(object_config.full_charge_bytes, object_config.linear_charge_bytes)
    , linear_policy_(object_config.linear_charge_bytes, object_config.linear_step_blocks) {
    if (object_config_.full_charge_bytes == 0) {
        throw std::invalid_argument("LiteHit full_charge_bytes must be positive");
    }
}

RequestFact LiteHit::ProcessRequest(const std::vector<int64_t> &block_keys) { return ProcessRequest(block_keys, 0); }

RequestFact LiteHit::ProcessRequest(const std::vector<int64_t> &block_keys, std::size_t alive_from_position) {
    const RequestFact fact = EvaluateRecoveryCurve(block_keys, alive_from_position);
    CommitRequest(block_keys, alive_from_position);
    return fact;
}

FullRequestFact LiteHit::ProcessFullRequest(const std::vector<int64_t> &block_keys) {
    return ProcessFullRequest(block_keys, 0);
}

FullRequestFact LiteHit::ProcessFullRequest(const std::vector<int64_t> &block_keys, std::size_t alive_from_position) {
    if (uses_linear()) {
        throw std::logic_error("ProcessFullRequest requires a Full-only LiteHit");
    }
    return EncodeFullFact(ProcessRequest(block_keys, alive_from_position));
}

RequestFact LiteHit::EvaluateRecoveryCurve(const std::vector<int64_t> &block_keys,
                                           std::size_t alive_from_position) const {
    RequestFact fact;
    if (block_keys.empty()) {
        return fact;
    }

    // All ranks come from one immutable request-start snapshot. A cold Full
    // block ends prefix coverage for every capacity.
    std::vector<uint64_t> prefix_full_required(block_keys.size(), 0);
    std::size_t covered_blocks = 0;
    uint64_t running_full_required = 0;
    for (std::size_t i = 0; i < block_keys.size(); ++i) {
        uint64_t required = 0;
        if (!pool_.RequiredBytes({CacheObjectType::kFull, block_keys[i]}, required, alive_from_position)) {
            break;
        }
        running_full_required = std::max(running_full_required, required);
        prefix_full_required[i] = running_full_required;
        covered_blocks = i + 1;
    }

    uint64_t last_full_threshold = 0;
    for (std::size_t position = 0; position < covered_blocks; ++position) {
        uint64_t threshold = prefix_full_required[position];
        if (uses_linear()) {
            uint64_t linear_required = 0;
            if (!linear_policy_.RequiredLinearBytes(
                    pool_, block_keys[position], alive_from_position, linear_required)) {
                continue;
            }
            threshold = std::max(threshold, linear_required);

            // A later restore point with no larger threshold dominates the
            // earlier point. Removing it keeps a strictly monotone envelope.
            while (!fact.points.empty() && fact.points.back().min_total_capacity_bytes >= threshold) {
                fact.points.pop_back();
            }
        } else {
            // Contract-valid prefix hashes make thresholds strictly increase.
            // For defensive duplicate-key input, force one Full charge of
            // progress per additional hit so projection is never optimistic.
            threshold = std::max(threshold, last_full_threshold + object_config_.full_charge_bytes);
            last_full_threshold = threshold;
        }
        fact.points.push_back(ByteStepPoint{threshold, static_cast<uint64_t>(position + 1)});
    }
    return fact;
}

FullRequestFact LiteHit::EncodeFullFact(const RequestFact &byte_fact) const {
    FullRequestFact fact;
    for (const ByteStepPoint &point : byte_fact.points) {
        const uint64_t required_blocks = point.min_total_capacity_bytes / object_config_.full_charge_bytes;
        if (!fact.hit_curve.empty() &&
            fact.hit_curve.back().start_required_blocks + fact.hit_curve.back().run_length == required_blocks) {
            ++fact.hit_curve.back().run_length;
        } else {
            fact.hit_curve.push_back(HitCurveSegment{required_blocks, 1});
        }
    }
    return fact;
}

WeightedLruPool::PositionRemap LiteHit::CommitRequest(const std::vector<int64_t> &block_keys,
                                                      std::size_t alive_from_position) {
    if (block_keys.empty()) {
        return {};
    }

    for (std::size_t i = block_keys.size(); i > 0; --i) {
        const std::size_t position = i - 1;
        linear_policy_.CommitLinearIfNeeded(pool_, block_keys[position], position, block_keys.size());
        pool_.Touch({CacheObjectType::kFull, block_keys[position]});
    }

    return pool_.MaybeCompactPositions(alive_from_position);
}

uint64_t LiteHit::current_unique_blocks() const { return pool_.resident_full_count(); }

uint64_t LiteHit::FullObjectsWithinTotalBytes(uint64_t total_capacity_bytes) const {
    return pool_.FullObjectsWithinBytes(total_capacity_bytes);
}

uint64_t LiteHit::resident_bytes() const { return pool_.resident_bytes(); }

uint64_t LiteHit::memory_usage_bytes() const { return pool_.memory_usage_bytes(); }

void LiteHit::Reset() { pool_.Reset(); }

} // namespace kv_cache_manager
