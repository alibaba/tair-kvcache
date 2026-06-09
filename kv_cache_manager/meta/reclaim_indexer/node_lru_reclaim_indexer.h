#pragma once

#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "kv_cache_manager/common/cache/advanced_cache.h"
#include "kv_cache_manager/meta/reclaim_indexer/reclaim_indexer.h"

namespace kv_cache_manager {

// LRU-based ReclaimIndexer. Extracts node_id from LocationSpec and maintains per-node LRU.
class NodeLruReclaimIndexer final : public ReclaimIndexer {
public:
    NodeLruReclaimIndexer() = default;
    ~NodeLruReclaimIndexer() override = default;

    std::vector<ErrorCode> Add(const KeyTypeVec &keys,
                               const CacheLocationMapVector &locations,
                               const std::vector<ErrorCode> &previous_error_codes) noexcept override;
    std::vector<ErrorCode> Touch(const KeyTypeVec &keys,
                                 const std::vector<ErrorCode> &previous_error_codes) noexcept override;
    std::vector<ErrorCode> Remove(const KeyTypeVec &keys,
                                  const LocationIdsPerKey &location_ids,
                                  const std::vector<ErrorCode> &previous_error_codes) noexcept override;
    std::vector<ErrorCode> Remove(const KeyTypeVec &keys,
                                  const std::vector<ErrorCode> &previous_error_codes) noexcept override;
    ErrorCode
    Sample(size_t count, const std::unordered_set<std::string> &node_ids, KeyTypeVec &out_keys) noexcept override;

private:
    static std::string_view KeyToView(const KeyType &key) {
        return {reinterpret_cast<const char *>(&key), sizeof(KeyType)};
    }
    static KeyType ViewToKey(std::string_view sv) {
        KeyType key = 0;
        std::memcpy(&key, sv.data(), sizeof(KeyType));
        return key;
    }

    std::shared_ptr<Cache> GetOrCreateNodeCache(const std::string &node_id);

    mutable std::mutex mutex_;
    // node_id -> per-node LRU cache
    std::unordered_map<std::string, std::shared_ptr<Cache>> node_caches_;
    // key -> { (location_id, spec_name) -> node_id }
    using LocSpecKey = std::pair<std::string, std::string>;
    using LocSpecNodeMap = std::map<LocSpecKey, std::string>;
    std::unordered_map<KeyType, LocSpecNodeMap> key_to_loc_nodes_;
    static const Cache::CacheItemHelper kItemHelper;
};

} // namespace kv_cache_manager
