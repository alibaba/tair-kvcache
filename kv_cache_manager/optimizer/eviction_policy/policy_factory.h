#pragma once
#include <memory>
#include <string>

#include "kv_cache_manager/optimizer/config/types.h"
#include "kv_cache_manager/optimizer/eviction_policy/base.h"
#include "kv_cache_manager/optimizer/eviction_policy/leaf_aware_lru.h"
#include "kv_cache_manager/optimizer/eviction_policy/lru.h"
#include "kv_cache_manager/optimizer/eviction_policy/random_lru.h"
namespace kv_cache_manager {

class EvictionPolicyFactory {
public:
    static std::shared_ptr<EvictionPolicy> CreatePolicy(EvictionPolicyType type,
                                                        const std::string &name,
                                                        const int32_t eviction_batch_size_per_instance,
                                                        EvictionPolicyParam param);
};

} // namespace kv_cache_manager