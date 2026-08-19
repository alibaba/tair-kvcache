#include "tools/kvcm_swarm/clients/v6d/workload.h"

#include <algorithm>
#include <limits>

#include "tools/kvcm_swarm/clients/v6d/key_mapper.h"

namespace kvcm_swarm {
namespace {

// Chain hash: block i depends on every preceding token of the same group, so a
// shared prefix produces identical keys across independent sessions.
inline uint64_t MixChain(uint64_t previous, uint64_t token) {
    uint64_t z = previous ^ (token * 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

inline uint64_t FinishChain(uint64_t accumulator, uint64_t salt) {
    uint64_t z = accumulator ^ salt;
    z = (z ^ (z >> 33)) * 0xff51afd7ed558ccdULL;
    z = (z ^ (z >> 33)) * 0xc4ceb9fe1a85ec53ULL;
    return z ^ (z >> 33);
}

} // namespace

SharedPrefixPoolState::SharedPrefixPoolState(const SharedPrefixPool &config,
                                             const SeedDeriver &seeds,
                                             const std::string &behavior_id) {
    if (config.root_count == 0) {
        return;
    }
    roots_.reserve(config.root_count);
    for (uint32_t i = 0; i < config.root_count; ++i) {
        Rng rng = seeds.MakeRng("v6d/" + behavior_id + "/shared_prefix_root", i);
        const uint64_t length = Sample(config.prefix_tokens, rng);
        std::vector<uint64_t> root;
        root.reserve(length);
        for (uint64_t t = 0; t < length; ++t) {
            root.push_back(rng.Next());
        }
        max_root_tokens_ = std::max(max_root_tokens_, length);
        roots_.push_back(std::move(root));
    }
}

void GroupKeyspace::Reset(const CacheGroupSpec *group) {
    group_ = group;
    salt_ = HashString("kvcm_swarm/group_salt/" + group->group_id);
    chain_.clear();
    objects_.clear();
}

void GroupKeyspace::Recompute(const std::vector<uint64_t> &tokens, uint64_t first_changed_token) {
    const uint32_t block_size = group_->block_size_tokens;
    const size_t complete_blocks = tokens.size() / block_size;
    const size_t first_block = static_cast<size_t>(first_changed_token / block_size);

    if (chain_.size() > first_block) {
        chain_.resize(first_block);
    }
    chain_.reserve(complete_blocks);
    for (size_t block = chain_.size(); block < complete_blocks; ++block) {
        uint64_t accumulator = block == 0 ? salt_ : chain_[block - 1];
        const size_t begin = block * block_size;
        for (size_t i = begin; i < begin + block_size; ++i) {
            accumulator = MixChain(accumulator, tokens[i]);
        }
        chain_.push_back(FinishChain(accumulator, salt_));
    }

    // Rebuild the object list for the affected suffix only.
    objects_.erase(
        std::remove_if(objects_.begin(),
                       objects_.end(),
                       [first_block](const GroupObject &object) { return object.block_index >= first_block; }),
        objects_.end());
    for (size_t block = first_block; block < complete_blocks; ++block) {
        if (group_->kind == CacheGroupKind::kMamba) {
            // Stable content + group hash decides key existence; not every
            // logical Mamba block has a key.
            const uint64_t roll = HashString("mamba_presence/" + group_->group_id + "/" + BlockHashHex(chain_[block]));
            const double normalized = static_cast<double>(roll >> 11) * (1.0 / 9007199254740992.0);
            if (normalized >= group_->key_presence_rate) {
                continue;
            }
        }
        GroupObject object;
        object.group = group_;
        object.block_index = static_cast<uint32_t>(block);
        object.boundary_tokens = static_cast<uint64_t>(block + 1) * block_size;
        object.block_hash = BlockHashHex(chain_[block]);
        object.object_key = MakeObjectKey(object.block_hash, group_->group_id);
        object.block_key = ObjectKeyToBlockKey(object.object_key);
        object.object_size = group_->object_size_bytes;
        object.spec_name = group_->spec_name;
        objects_.push_back(std::move(object));
    }
    std::sort(objects_.begin(), objects_.end(), [](const GroupObject &a, const GroupObject &b) {
        return a.block_index < b.block_index;
    });
}

void SessionWorkload::Init(const SessionClass &session_class,
                           const std::vector<CacheGroupSpec> &groups,
                           const SharedPrefixPoolState &pool,
                           bool use_shared_prefix,
                           Rng &content_rng,
                           Rng &shape_rng) {
    groups_.clear();
    groups_.reserve(groups.size());
    for (const auto &group : groups) {
        GroupKeyspace keyspace;
        keyspace.Reset(&group);
        groups_.push_back(std::move(keyspace));
    }
    const uint64_t initial_tokens = Sample(session_class.initial_tokens, shape_rng);
    tokens_.clear();
    tokens_.reserve(initial_tokens);
    used_shared_prefix_ = use_shared_prefix && pool.root_count() > 0;
    if (used_shared_prefix_) {
        shared_root_index_ = static_cast<uint32_t>(shape_rng.NextInRange(0, pool.root_count() - 1));
        const auto &root = pool.Root(shared_root_index_);
        for (uint64_t token : root) {
            if (tokens_.size() >= initial_tokens) {
                break;
            }
            tokens_.push_back(token);
        }
    }
    while (tokens_.size() < initial_tokens) {
        tokens_.push_back(content_rng.Next());
    }
    last_rewrite_tokens_ = 0;
    last_new_tokens_ = initial_tokens;
    RecomputeFrom(0);
}

void SessionWorkload::ApplyTurn(const SessionClass &session_class, Rng &content_rng, Rng &shape_rng) {
    const uint64_t requested_rewrite = Sample(session_class.rewrite_tail_tokens, shape_rng);
    const uint64_t rewrite = std::min<uint64_t>(requested_rewrite, tokens_.size());
    const uint64_t appended = Sample(session_class.new_tokens_per_turn, shape_rng);
    const uint64_t first_changed = tokens_.size() - rewrite;
    for (uint64_t i = first_changed; i < tokens_.size(); ++i) {
        tokens_[i] = content_rng.Next();
    }
    for (uint64_t i = 0; i < appended; ++i) {
        tokens_.push_back(content_rng.Next());
    }
    last_rewrite_tokens_ = rewrite;
    last_new_tokens_ = appended;
    RecomputeFrom(first_changed);
}

uint64_t SessionWorkload::WorkingSetBytes() const {
    uint64_t total = 0;
    for (const auto &keyspace : groups_) {
        for (const auto &object : keyspace.objects()) {
            if (object.object_size > std::numeric_limits<uint64_t>::max() - total) {
                return std::numeric_limits<uint64_t>::max();
            }
            total += object.object_size;
        }
    }
    return total;
}

void SessionWorkload::RecomputeFrom(uint64_t first_changed_token) {
    for (auto &keyspace : groups_) {
        keyspace.Recompute(tokens_, first_changed_token);
    }
}

} // namespace kvcm_swarm
