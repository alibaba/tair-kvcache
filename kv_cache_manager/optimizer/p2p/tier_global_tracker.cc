#include "kv_cache_manager/optimizer/p2p/tier_global_tracker.h"

#include <utility>

namespace kv_cache_manager {

std::string TierGlobalTracker::ScopeKey(const std::string &cluster_id, const std::string &tier) {
    return cluster_id + "\x1f" + tier;
}

void TierGlobalTracker::Add(const std::string &cluster_id,
                            const std::string &tier,
                            int64_t key,
                            const std::string &infer_id) {
    holders_[ScopeKey(cluster_id, tier)][key].insert(infer_id);
}

void TierGlobalTracker::Remove(const std::string &cluster_id,
                               const std::string &tier,
                               int64_t key,
                               const std::string &infer_id) {
    auto scope_it = holders_.find(ScopeKey(cluster_id, tier));
    if (scope_it == holders_.end()) {
        return;
    }
    auto key_it = scope_it->second.find(key);
    if (key_it == scope_it->second.end()) {
        return;
    }
    key_it->second.erase(infer_id);
    if (key_it->second.empty()) {
        scope_it->second.erase(key_it);
    }
}

void TierGlobalTracker::RemoveFromAllTiers(const std::string &cluster_id, int64_t key, const std::string &infer_id) {
    const std::string prefix = cluster_id + "\x1f";
    for (auto &scope : holders_) {
        if (scope.first.rfind(prefix, 0) != 0) {
            continue;
        }
        auto key_it = scope.second.find(key);
        if (key_it == scope.second.end()) {
            continue;
        }
        key_it->second.erase(infer_id);
        if (key_it->second.empty()) {
            scope.second.erase(key_it);
        }
    }
}

void TierGlobalTracker::RemoveInfer(const std::string &cluster_id, const std::string &infer_id) {
    const std::string prefix = cluster_id + "\x1f";
    for (auto scope_it = holders_.begin(); scope_it != holders_.end();) {
        if (scope_it->first.rfind(prefix, 0) != 0) {
            ++scope_it;
            continue;
        }
        for (auto key_it = scope_it->second.begin(); key_it != scope_it->second.end();) {
            key_it->second.erase(infer_id);
            if (key_it->second.empty()) {
                key_it = scope_it->second.erase(key_it);
            } else {
                ++key_it;
            }
        }
        if (scope_it->second.empty()) {
            scope_it = holders_.erase(scope_it);
        } else {
            ++scope_it;
        }
    }
}

void TierGlobalTracker::ApplyEvent(const std::string &cluster_id, const TierFlowKeyEvent &event) {
    if (event.kind == TierFlowEventKind::ENTER_TIER && !event.to_tier.empty()) {
        Add(cluster_id, event.to_tier, event.block_key, event.instance_id);
    } else if (event.kind == TierFlowEventKind::LEAVE_TIER && !event.from_tier.empty()) {
        Remove(cluster_id, event.from_tier, event.block_key, event.instance_id);
    } else if (event.kind == TierFlowEventKind::FINAL_EVICT) {
        RemoveFromAllTiers(cluster_id, event.block_key, event.instance_id);
    }
}

bool TierGlobalTracker::Contains(const std::string &cluster_id,
                                 const std::string &tier,
                                 int64_t key,
                                 const std::string &infer_id) const {
    auto scope_it = holders_.find(ScopeKey(cluster_id, tier));
    if (scope_it == holders_.end()) {
        return false;
    }
    auto key_it = scope_it->second.find(key);
    return key_it != scope_it->second.end() && key_it->second.count(infer_id) > 0;
}

TierGlobalPeerSelection TierGlobalTracker::SelectPeer(const std::string &engine_instance_id,
                                                      const std::string &cluster_id,
                                                      const std::string &tier,
                                                      const std::vector<std::string> &candidate_infer_ids,
                                                      const std::vector<int64_t> &block_ids,
                                                      const std::vector<bool> &satisfied_mask) const {
    std::vector<size_t> missing_indices;
    for (size_t idx = 0; idx < block_ids.size(); ++idx) {
        if (idx >= satisfied_mask.size() || !satisfied_mask[idx]) {
            missing_indices.push_back(idx);
        }
    }
    if (missing_indices.empty()) {
        return {};
    }

    auto scope_it = holders_.find(ScopeKey(cluster_id, tier));
    if (scope_it == holders_.end()) {
        return {};
    }
    const size_t first_missing_idx = missing_indices.front();
    if (first_missing_idx >= block_ids.size()) {
        return {};
    }
    auto first_key_it = scope_it->second.find(block_ids[first_missing_idx]);
    if (first_key_it == scope_it->second.end()) {
        return {};
    }

    size_t best_len = 0;
    std::string best_peer;
    for (const auto &peer_id : candidate_infer_ids) {
        if (peer_id == engine_instance_id || first_key_it->second.count(peer_id) == 0) {
            continue;
        }
        size_t match_len = 0;
        while (match_len < missing_indices.size()) {
            const size_t block_idx = missing_indices[match_len];
            if (!Contains(cluster_id, tier, block_ids[block_idx], peer_id)) {
                break;
            }
            ++match_len;
        }
        if (match_len > best_len) {
            best_len = match_len;
            best_peer = peer_id;
        }
    }
    if (best_len == 0) {
        return {};
    }

    TierGlobalPeerSelection selection;
    selection.peer_infer_id = std::move(best_peer);
    selection.hit_indices.assign(missing_indices.begin(), missing_indices.begin() + best_len);
    return selection;
}

} // namespace kv_cache_manager
