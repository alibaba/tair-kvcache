#include "kv_cache_manager/meta/reclaim_indexer/node_lru_reclaim_indexer.h"

#include <algorithm>

#include "kv_cache_manager/common/cache/lru_cache.h"
#include "kv_cache_manager/common/logger.h"

namespace kv_cache_manager {

namespace {
// Large capacity so the LRU never auto-evicts; we only use ordering.
constexpr size_t kNodeCacheCapacity = 1ULL << 40;
constexpr int kNumShardBits = 0; // single shard per node
} // namespace

// No-op deleter: entries carry no heap-allocated value.
const Cache::CacheItemHelper NodeLruReclaimIndexer::kItemHelper{};

std::vector<ErrorCode> NodeLruReclaimIndexer::Add(const KeyTypeVec &keys,
                                                  const CacheLocationMapVector &locations,
                                                  const std::vector<ErrorCode> &previous_error_codes) noexcept {
    std::vector<ErrorCode> results = previous_error_codes;
    if (keys.size() != locations.size() || keys.size() != previous_error_codes.size()) {
        results.assign(keys.size(), EC_BADARGS);
        return results;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    for (size_t i = 0; i < keys.size(); ++i) {
        if (results[i] != EC_OK) {
            continue;
        }
        const auto key = keys[i];
        auto key_sv = KeyToView(key);
        auto &loc_nodes = key_to_loc_nodes_[key];

        for (const auto &[loc_id, loc_ptr] : locations[i]) {
            if (!loc_ptr) {
                continue;
            }
            for (const auto &spec : loc_ptr->location_specs()) {
                const auto &node_id = spec.node_id();
                if (node_id.empty()) {
                    continue;
                }
                loc_nodes[{loc_id, spec.name()}] = node_id;
                auto node_cache = GetOrCreateNodeCache(node_id);
                node_cache->Insert(key_sv, nullptr, &kItemHelper, /*charge*/ 1);
            }
        }
    }
    return results;
}

std::vector<ErrorCode> NodeLruReclaimIndexer::Touch(const KeyTypeVec &keys,
                                                    const std::vector<ErrorCode> &previous_error_codes) noexcept {
    std::vector<ErrorCode> results = previous_error_codes;
    if (keys.size() != previous_error_codes.size()) {
        results.assign(keys.size(), EC_BADARGS);
        return results;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    for (size_t i = 0; i < keys.size(); ++i) {
        if (results[i] != EC_OK) {
            continue;
        }
        auto it = key_to_loc_nodes_.find(keys[i]);
        if (it == key_to_loc_nodes_.end()) {
            continue;
        }
        auto key_sv = KeyToView(keys[i]);
        std::unordered_set<std::string> touched_nodes;
        for (const auto &[loc_spec, node_id] : it->second) {
            touched_nodes.insert(node_id);
        }
        for (const auto &node_id : touched_nodes) {
            auto cache_it = node_caches_.find(node_id);
            if (cache_it == node_caches_.end()) {
                continue;
            }
            auto *handle = cache_it->second->Lookup(key_sv);
            if (handle) {
                cache_it->second->Release(handle);
            }
        }
    }
    return results;
}

std::vector<ErrorCode> NodeLruReclaimIndexer::Remove(const KeyTypeVec &keys,
                                                     const LocationIdsPerKey &location_ids,
                                                     const std::vector<ErrorCode> &previous_error_codes) noexcept {
    std::vector<ErrorCode> results = previous_error_codes;
    if (keys.size() != location_ids.size() || keys.size() != previous_error_codes.size()) {
        results.assign(keys.size(), EC_BADARGS);
        return results;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    for (size_t i = 0; i < keys.size(); ++i) {
        if (results[i] != EC_OK) {
            continue;
        }
        const auto key = keys[i];
        auto map_it = key_to_loc_nodes_.find(key);
        if (map_it == key_to_loc_nodes_.end()) {
            continue;
        }
        auto &loc_nodes = map_it->second;
        for (const auto &loc_id : location_ids[i]) {
            std::unordered_set<std::string> removed_nodes;
            for (auto it2 = loc_nodes.begin(); it2 != loc_nodes.end();) {
                if (it2->first.first == loc_id) {
                    removed_nodes.insert(it2->second);
                    it2 = loc_nodes.erase(it2);
                } else {
                    ++it2;
                }
            }
            for (const auto &node_id : removed_nodes) {
                bool still_on_node = false;
                for (const auto &[ls, nid] : loc_nodes) {
                    if (nid == node_id) {
                        still_on_node = true;
                        break;
                    }
                }
                if (!still_on_node) {
                    auto cache_it = node_caches_.find(node_id);
                    if (cache_it != node_caches_.end()) {
                        cache_it->second->Erase(KeyToView(key));
                    }
                }
            }
        }
        if (loc_nodes.empty()) {
            key_to_loc_nodes_.erase(map_it);
        }
    }
    return results;
}

std::vector<ErrorCode> NodeLruReclaimIndexer::Remove(const KeyTypeVec &keys,
                                                     const std::vector<ErrorCode> &previous_error_codes) noexcept {
    std::vector<ErrorCode> results = previous_error_codes;
    if (keys.size() != previous_error_codes.size()) {
        results.assign(keys.size(), EC_BADARGS);
        return results;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    for (size_t i = 0; i < keys.size(); ++i) {
        if (results[i] != EC_OK) {
            continue;
        }
        const auto key = keys[i];
        auto map_it = key_to_loc_nodes_.find(key);
        if (map_it == key_to_loc_nodes_.end()) {
            continue;
        }
        auto key_sv = KeyToView(key);
        std::unordered_set<std::string> nodes;
        for (const auto &[loc_spec, node_id] : map_it->second) {
            nodes.insert(node_id);
        }
        for (const auto &node_id : nodes) {
            auto cache_it = node_caches_.find(node_id);
            if (cache_it != node_caches_.end()) {
                cache_it->second->Erase(key_sv);
            }
        }
        key_to_loc_nodes_.erase(map_it);
    }
    return results;
}

ErrorCode NodeLruReclaimIndexer::Sample(size_t count,
                                        const std::unordered_set<std::string> &node_ids,
                                        KeyTypeVec &out_keys) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    out_keys.clear();
    out_keys.reserve(count);
    if (node_ids.empty() || count == 0) {
        return EC_OK;
    }
    size_t per_node = count / node_ids.size();
    size_t remainder = count % node_ids.size();
    size_t idx = 0;
    for (const auto &node_id : node_ids) {
        auto cache_it = node_caches_.find(node_id);
        if (cache_it == node_caches_.end()) {
            ++idx;
            continue;
        }
        size_t quota = per_node + (idx < remainder ? 1 : 0);
        std::vector<std::string> raw_keys;
        raw_keys.reserve(quota);
        cache_it->second->GetOldestKeysInShard(0, quota, raw_keys);
        for (const auto &raw : raw_keys) {
            if (raw.size() == sizeof(KeyType)) {
                out_keys.push_back(ViewToKey(raw));
            }
        }
        ++idx;
    }
    return EC_OK;
}

std::shared_ptr<Cache> NodeLruReclaimIndexer::GetOrCreateNodeCache(const std::string &node_id) {
    auto it = node_caches_.find(node_id);
    if (it != node_caches_.end()) {
        return it->second;
    }
    auto cache = NewLRUCache(kNodeCacheCapacity,
                             kNumShardBits,
                             /*strict_capacity_limit=*/false,
                             /*no_evict_on_insert=*/true);
    node_caches_[node_id] = cache;
    return cache;
}

} // namespace kv_cache_manager
