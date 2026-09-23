#include "kv_cache_manager/optimizer/metrics/mrc_window.h"

#include <array>
#include <iterator>
#include <limits>

#include "kv_cache_manager/optimizer/liteHit/hit_curve.h"

namespace kv_cache_manager {

namespace {

constexpr uint32_t kBasisPointScale = 10000;
constexpr std::array<uint32_t, 6> kTargetBasisPoints = {6000, 8000, 9000, 9500, 9900, 9950};

uint64_t SaturatingAdd(uint64_t lhs, uint64_t rhs) {
    return rhs > std::numeric_limits<uint64_t>::max() - lhs ? std::numeric_limits<uint64_t>::max() : lhs + rhs;
}

uint64_t SaturatingMultiply(uint64_t lhs, uint64_t rhs) {
    if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs) {
        return std::numeric_limits<uint64_t>::max();
    }
    return lhs * rhs;
}

uint64_t ComputeTargetHits(uint64_t total_hits, uint32_t target_basis_points) {
    // ceil(total * target / 10000), split into quotient/remainder to avoid overflow.
    const uint64_t quotient = total_hits / kBasisPointScale;
    const uint64_t remainder = total_hits % kBasisPointScale;
    return quotient * target_basis_points + (remainder * target_basis_points + kBasisPointScale - 1) / kBasisPointScale;
}

} // namespace

void MrcWindow::Record(const FullRequestFact &fact) {
    uint64_t request_hits = 0;
    for (const auto &segment : fact.hit_curve) {
        if (segment.run_length == 0) {
            continue;
        }

        const uint64_t end_required_blocks = segment.start_required_blocks + segment.run_length - 1;
        ++hit_count_deltas_[segment.start_required_blocks];
        --hit_count_deltas_[end_required_blocks + 1];
        request_hits = SaturatingAdd(request_hits, segment.run_length);
    }

    total_hits_ = SaturatingAdd(total_hits_, request_hits);
}

std::vector<MrcWindowPoint> MrcWindow::Take() {
    std::vector<MrcWindowPoint> curve;
    curve.reserve(kTargetBasisPoints.size());
    for (uint32_t target_basis_points : kTargetBasisPoints) {
        curve.push_back({target_basis_points, ComputeRequiredBlocks(target_basis_points)});
    }
    Reset();
    return curve;
}

void MrcWindow::Reset() {
    hit_count_deltas_.clear();
    total_hits_ = 0;
}

uint64_t MrcWindow::ComputeRequiredBlocks(uint32_t target_basis_points) const {
    if (total_hits_ == 0) {
        return 0;
    }

    const uint64_t target_hits = ComputeTargetHits(total_hits_, target_basis_points);
    uint64_t accumulated_hits = 0;
    int64_t hits_at_required_blocks = 0;

    for (auto it = hit_count_deltas_.begin(); it != hit_count_deltas_.end(); ++it) {
        hits_at_required_blocks += it->second;
        const auto next = std::next(it);
        if (hits_at_required_blocks <= 0 || next == hit_count_deltas_.end()) {
            continue;
        }

        const uint64_t span = next->first - it->first;
        const uint64_t span_hits = SaturatingMultiply(static_cast<uint64_t>(hits_at_required_blocks), span);
        if (span_hits >= target_hits - accumulated_hits) {
            const uint64_t remaining_hits = target_hits - accumulated_hits;
            const uint64_t offset = (remaining_hits - 1) / static_cast<uint64_t>(hits_at_required_blocks);
            return it->first + offset;
        }
        accumulated_hits = SaturatingAdd(accumulated_hits, span_hits);
    }
    return 0;
}

void ByteMrcWindow::Record(const RequestFact &fact) {
    uint64_t request_hits = 0;
    for (const auto &point : fact.points) {
        if (point.hit_blocks <= request_hits) {
            continue;
        }
        const uint64_t additional_hits = point.hit_blocks - request_hits;
        auto &increment = hit_count_increments_[point.min_total_capacity_bytes];
        increment = SaturatingAdd(increment, additional_hits);
        request_hits = point.hit_blocks;
    }
    total_hits_ = SaturatingAdd(total_hits_, request_hits);
}

std::vector<ByteMrcWindowPoint> ByteMrcWindow::Take() {
    std::vector<ByteMrcWindowPoint> curve;
    curve.reserve(kTargetBasisPoints.size());
    for (uint32_t target_basis_points : kTargetBasisPoints) {
        curve.push_back({target_basis_points, ComputeRequiredBytes(target_basis_points)});
    }
    Reset();
    return curve;
}

void ByteMrcWindow::Reset() {
    hit_count_increments_.clear();
    total_hits_ = 0;
}

uint64_t ByteMrcWindow::ComputeRequiredBytes(uint32_t target_basis_points) const {
    if (total_hits_ == 0) {
        return 0;
    }

    const uint64_t target_hits = ComputeTargetHits(total_hits_, target_basis_points);
    uint64_t accumulated_hits = 0;
    for (const auto &[required_bytes, additional_hits] : hit_count_increments_) {
        accumulated_hits = SaturatingAdd(accumulated_hits, additional_hits);
        if (accumulated_hits >= target_hits) {
            return required_bytes;
        }
    }
    return 0;
}

} // namespace kv_cache_manager
