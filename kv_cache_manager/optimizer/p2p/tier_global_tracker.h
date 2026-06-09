#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "kv_cache_manager/optimizer/tier_flow/tier_flow_recorder.h"

namespace kv_cache_manager {

struct TierGlobalPeerSelection {
    std::string peer_infer_id;
    std::vector<size_t> hit_indices;
};

class TierGlobalTracker {
public:
    void Add(const std::string &cluster_id, const std::string &tier, int64_t key, const std::string &infer_id);
    void Remove(const std::string &cluster_id, const std::string &tier, int64_t key, const std::string &infer_id);
    void RemoveFromAllTiers(const std::string &cluster_id, int64_t key, const std::string &infer_id);
    void RemoveInfer(const std::string &cluster_id, const std::string &infer_id);
    void ApplyEvent(const std::string &cluster_id, const TierFlowKeyEvent &event);

    [[nodiscard]] bool
    Contains(const std::string &cluster_id, const std::string &tier, int64_t key, const std::string &infer_id) const;

    [[nodiscard]] TierGlobalPeerSelection SelectPeer(const std::string &engine_instance_id,
                                                     const std::string &cluster_id,
                                                     const std::string &tier,
                                                     const std::vector<std::string> &candidate_infer_ids,
                                                     const std::vector<int64_t> &block_ids,
                                                     const std::vector<bool> &satisfied_mask) const;

private:
    static std::string ScopeKey(const std::string &cluster_id, const std::string &tier);

    std::unordered_map<std::string, std::unordered_map<int64_t, std::unordered_set<std::string>>> holders_;
};

} // namespace kv_cache_manager
