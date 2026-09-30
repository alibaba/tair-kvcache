#pragma once

#include <cstdint>
#include <map>
#include <vector>

namespace kv_cache_manager {

struct FullRequestFact;
struct RequestFact;

struct MrcWindowPoint {
    // Relative percentage of this window's theoretical maximum hit count,
    // expressed in basis points. It is not an absolute request hit rate.
    uint32_t target_basis_points = 0;
    uint64_t required_blocks = 0;
};

// Accumulates one reporting window of full-attention theoretical hits. Each
// output capacity retains a configured percentage of the theoretical maximum
// hit count in that window, rather than reaching that percentage as an
// absolute request hit rate.
// Synchronization is provided by the owning InstanceState mutex.
class MrcWindow {
public:
    void Record(const FullRequestFact &fact);
    std::vector<MrcWindowPoint> Take();
    void Reset();

private:
    uint64_t ComputeRequiredBlocks(uint32_t target_basis_points) const;

    // Sparse difference points of required capacity -> theoretical hit count.
    std::map<uint64_t, int64_t> hit_count_deltas_;
    uint64_t total_hits_ = 0;
};

struct ByteMrcWindowPoint {
    // Relative percentage of this window's theoretical maximum hit count,
    // expressed in basis points. It is not an absolute request hit rate.
    uint32_t target_basis_points = 0;
    uint64_t required_bytes = 0;
};

// Accumulates one reporting window of linear/Mamba theoretical hits. A
// RequestFact is already a byte-axis step curve: each point says that reaching
// required_bytes raises the recoverable prefix to hit_blocks. Recording only
// the increase at each threshold keeps the window sparse while allowing exact
// aggregation across requests.
// Synchronization is provided by the owning InstanceState mutex.
class ByteMrcWindow {
public:
    void Record(const RequestFact &fact);
    std::vector<ByteMrcWindowPoint> Take();
    void Reset();

private:
    uint64_t ComputeRequiredBytes(uint32_t target_basis_points) const;

    // Required byte capacity -> newly reachable theoretical hit blocks.
    std::map<uint64_t, uint64_t> hit_count_increments_;
    uint64_t total_hits_ = 0;
};

} // namespace kv_cache_manager
