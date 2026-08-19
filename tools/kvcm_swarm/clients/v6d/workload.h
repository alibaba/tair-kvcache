// Token-level workload model.
//
// Upstream traffic is described in logical tokens, not blocks, because cache
// groups can use different block sizes. Full Attention groups produce a key
// for every complete block; Mamba key existence is decided by stable content
// plus group hash, so it never depends on asynchronous completion order.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "tools/kvcm_swarm/clients/v6d/config.h"
#include "tools/kvcm_swarm/runtime/rng.h"

namespace kvcm_swarm {

struct GroupObject {
    const CacheGroupSpec *group = nullptr;
    uint32_t block_index = 0;
    uint64_t boundary_tokens = 0; // exclusive end token offset of this block
    std::string block_hash;
    std::string object_key;
    int64_t block_key = 0;
    uint64_t object_size = 0;
    std::string spec_name;
};

// Shared prefix roots: a fixed pool of token sequences that many independent
// sessions can attach to. Roots are chosen uniformly at session creation.
class SharedPrefixPoolState {
public:
    SharedPrefixPoolState(const SharedPrefixPool &config, const SeedDeriver &seeds, const std::string &behavior_id);

    uint32_t root_count() const { return static_cast<uint32_t>(roots_.size()); }
    const std::vector<uint64_t> &Root(uint32_t index) const { return roots_[index % roots_.size()]; }
    uint64_t max_root_tokens() const { return max_root_tokens_; }

private:
    std::vector<std::vector<uint64_t>> roots_;
    uint64_t max_root_tokens_ = 0;
};

// Per-group derived keyspace of one session: chained block hashes plus, for
// Mamba, whether each complete block actually has a key.
class GroupKeyspace {
public:
    void Reset(const CacheGroupSpec *group);
    // Recomputes every complete block from `first_changed_token` onwards.
    void Recompute(const std::vector<uint64_t> &tokens, uint64_t first_changed_token);

    const CacheGroupSpec *group() const { return group_; }
    size_t complete_blocks() const { return chain_.size(); }
    // Objects that actually exist (Full: all complete blocks; Mamba: only
    // blocks whose content hash selects them).
    const std::vector<GroupObject> &objects() const { return objects_; }

private:
    const CacheGroupSpec *group_ = nullptr;
    uint64_t salt_ = 0;
    std::vector<uint64_t> chain_;
    std::vector<GroupObject> objects_;
};

// Logical token history of one session. Sessions own only this logical state:
// never a process-local cache entry, never a cross-turn lease.
class SessionWorkload {
public:
    void Init(const SessionClass &session_class,
              const std::vector<CacheGroupSpec> &groups,
              const SharedPrefixPoolState &pool,
              bool use_shared_prefix,
              Rng &content_rng,
              Rng &shape_rng);

    // Applies one turn: rewrite the tail, then append new tokens, then
    // recompute only the affected group blocks.
    void ApplyTurn(const SessionClass &session_class, Rng &content_rng, Rng &shape_rng);

    size_t token_count() const { return tokens_.size(); }
    const std::vector<GroupKeyspace> &groups() const { return groups_; }
    // Sum of every object represented by the current turn. This intentionally
    // does not deduplicate equal object keys across groups or sessions.
    uint64_t WorkingSetBytes() const;
    uint64_t last_rewrite_tokens() const { return last_rewrite_tokens_; }
    uint64_t last_new_tokens() const { return last_new_tokens_; }
    bool used_shared_prefix() const { return used_shared_prefix_; }
    uint32_t shared_root_index() const { return shared_root_index_; }

private:
    void RecomputeFrom(uint64_t first_changed_token);

    std::vector<uint64_t> tokens_;
    std::vector<GroupKeyspace> groups_;
    bool used_shared_prefix_ = false;
    uint32_t shared_root_index_ = 0;
    uint64_t last_rewrite_tokens_ = 0;
    uint64_t last_new_tokens_ = 0;
};

} // namespace kvcm_swarm
