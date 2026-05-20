#pragma once

#include "kv_cache_manager/manager/select_location_policy.h"

namespace kv_cache_manager {

// PriorityThenRandomSLPolicy selects the CacheLocation with the highest weight
// (determined by StaticWeightSLPolicy::GetWeight) and, when multiple locations
// share that maximum weight, picks one uniformly at random.
//
// This gives deterministic cross-type ordering (e.g. VINEYARD=10 always beats
// TAIR_MEMPOOL=3) while distributing load evenly across nodes of the same type.
//
// ExistsForWrite is inherited from WeightSLPolicy and therefore behaves
// identically to StaticWeightSLPolicy.
class PriorityThenRandomSLPolicy : public StaticWeightSLPolicy {
public:
    CacheLocation *SelectForMatch(CacheLocationMap &location_map,
                                  CheckLocDataExistFunc check_loc_data_exist,
                                  std::vector<std::string> &out_prune_loc_ids) const override;
};

} // namespace kv_cache_manager
