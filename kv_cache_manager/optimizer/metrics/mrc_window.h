#pragma once

#include <cstdint>
#include <map>
#include <vector>

namespace kv_cache_manager {

struct RequestFact;

struct MrcWindowPoint {
    uint32_t target_basis_points = 0;
    uint64_t required_blocks = 0;
};

// Accumulates one reporting window of full-attention theoretical hits.
// Synchronization is provided by the owning InstanceState mutex.
class MrcWindow {
public:
    void Record(const RequestFact &fact);
    std::vector<MrcWindowPoint> Take();
    void Reset();

private:
    uint64_t ComputeRequiredBlocks(uint32_t target_basis_points) const;

    // Sparse difference points of required capacity -> theoretical hit count.
    std::map<uint64_t, int64_t> hit_count_deltas_;
    uint64_t total_hits_ = 0;
};

} // namespace kv_cache_manager
