#include "kv_cache_manager/manager/priority_then_random_policy.h"

#include <algorithm>
#include <random>
#include <vector>

namespace kv_cache_manager {

CacheLocation *PriorityThenRandomSLPolicy::SelectForMatch(CacheLocationMap &location_map,
                                                          CheckLocDataExistFunc check_loc_data_exist,
                                                          std::vector<std::string> &out_prune_loc_ids) const {
    thread_local std::mt19937 rng(std::random_device{}());

    std::vector<CacheLocation *> candidates;
    std::vector<uint32_t> weights;
    candidates.reserve(location_map.size());
    weights.reserve(location_map.size());
    out_prune_loc_ids.clear();

    // Pass 1: filter CLS_SERVING locations, collect prune candidates, gather
    // weights.
    for (auto &kv : location_map) {
        if (kv.second.status() != CacheLocationStatus::CLS_SERVING) {
            continue;
        }
        if (check_loc_data_exist) {
            auto result = check_loc_data_exist(kv.second);
            if (result == LocCheckResult::NOT_EXIST) {
                out_prune_loc_ids.emplace_back(kv.first);
                continue;
            }
            if (result == LocCheckResult::TEMPORARILY_UNREACHABLE) {
                continue;
            }
        }
        uint32_t w = GetWeight(kv);
        if (w == 0) {
            continue;
        }
        candidates.push_back(&kv.second);
        weights.push_back(w);
    }

    if (candidates.empty()) {
        return nullptr;
    }

    // Pass 2: find the maximum weight.
    uint32_t max_w = *std::max_element(weights.begin(), weights.end());

    // Pass 3: collect top-tier locations and choose uniformly at random.
    std::vector<CacheLocation *> top_tier;
    for (size_t i = 0; i < candidates.size(); ++i) {
        if (weights[i] == max_w) {
            top_tier.push_back(candidates[i]);
        }
    }

    std::uniform_int_distribution<size_t> dist(0, top_tier.size() - 1);
    return top_tier[dist(rng)];
}

} // namespace kv_cache_manager
