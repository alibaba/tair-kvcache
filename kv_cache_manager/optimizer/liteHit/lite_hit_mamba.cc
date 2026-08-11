#include "kv_cache_manager/optimizer/liteHit/lite_hit_mamba.h"

#include <algorithm>

namespace kv_cache_manager {

LiteHitMamba::LiteHitMamba(const Config &config)
    : config_(config), pool_(config.full_charge_bytes, config.mamba_charge_bytes) {
    if (config_.step_blocks == 0) {
        config_.step_blocks = 1;
    }
}

bool LiteHitMamba::IsCheckpoint(std::size_t position, std::size_t total_blocks) const {
    return (position + 1) % config_.step_blocks == 0 || position == total_blocks - 1;
}

MambaRequestFact LiteHitMamba::ProcessRequest(const std::vector<int64_t> &block_keys) {
    MambaRequestFact fact;
    const std::size_t n = block_keys.size();
    if (n == 0) {
        return fact;
    }

    // Phase 1: all residency ranks come from the request-start snapshot.
    // Walk the Full prefix from the head; the running max is the pool demand
    // of covering blocks [0..i]. The first non-resident block ends coverage
    // for every capacity.
    std::vector<uint64_t> prefix_full_required(n, 0);
    std::size_t covered_blocks = 0;
    uint64_t running_full_required = 0;
    for (std::size_t i = 0; i < n; ++i) {
        uint64_t required = 0;
        if (!pool_.RequiredBytes({CacheObjectType::kFull, block_keys[i]}, required)) {
            break;
        }
        running_full_required = std::max(running_full_required, required);
        prefix_full_required[i] = running_full_required;
        covered_blocks = i + 1;
    }

    // A checkpoint at position p recovers the request iff its MambaObject and
    // the Full prefix [0..p] both stay resident. Both live in the same pool, so
    // the demand is the larger of the two. Ascending positions with a Pareto
    // sweep yield the strictly monotone envelope.
    for (std::size_t p = 0; p < covered_blocks; ++p) {
        if (!IsCheckpoint(p, n)) {
            continue;
        }
        uint64_t mamba_required = 0;
        if (!pool_.RequiredBytes({CacheObjectType::kMamba, block_keys[p]}, mamba_required)) {
            continue;
        }
        const uint64_t threshold = std::max(prefix_full_required[p], mamba_required);
        while (!fact.points.empty() && fact.points.back().min_total_capacity_bytes >= threshold) {
            fact.points.pop_back();
        }
        fact.points.push_back(MambaCurvePoint{threshold, static_cast<uint64_t>(p + 1)});
    }

    // Phase 2: fixed commit set, tail-to-head. At a checkpoint position the
    // MambaObject is touched before the FullObject, so the FullObject ends up
    // closer to the MRU end.
    for (std::size_t i = n; i > 0; --i) {
        const std::size_t position = i - 1;
        if (IsCheckpoint(position, n)) {
            pool_.Touch({CacheObjectType::kMamba, block_keys[position]});
        }
        pool_.Touch({CacheObjectType::kFull, block_keys[position]});
    }
    return fact;
}

void LiteHitMamba::Reset() { pool_.Reset(); }

uint64_t LiteHitMamba::current_unique_blocks() const { return pool_.resident_full_count(); }

uint64_t LiteHitMamba::FullObjectsWithinTotalBytes(uint64_t total_capacity_bytes) const {
    return pool_.FullObjectsWithinBytes(total_capacity_bytes);
}

uint64_t LiteHitMamba::resident_bytes() const { return pool_.resident_bytes(); }

uint64_t LiteHitMamba::memory_usage_bytes() const { return pool_.memory_usage_bytes(); }

} // namespace kv_cache_manager
