#include "kv_cache_manager/optimizer/index/hash_table_index.h"

#include <utility>

namespace kv_cache_manager {
namespace {

void SetPoolLocation(BlockEntry *block, const std::string &tier_name, int64_t timestamp, size_t write_touch_count) {
    block->location_map[tier_name] = TierStat{0, timestamp, timestamp, write_touch_count};
}

} // namespace

HashTableIndex::HashTableIndex(std::string tier_name) : tier_name_(std::move(tier_name)) {}

BlockEntry *HashTableIndex::Find(int64_t key) const {
    auto it = blocks_.find(key);
    if (it == blocks_.end()) {
        return nullptr;
    }
    return it->second.get();
}

BlockEntry *HashTableIndex::Insert(int64_t key, int64_t timestamp, int64_t ttl_ns) {
    auto block = std::make_unique<BlockEntry>();
    block->key = key;
    block->writing_time = timestamp;
    block->last_access_time = timestamp;
    block->ttl_anchor_time = timestamp;
    block->ttl_ns = ttl_ns;
    SetPoolLocation(block.get(), tier_name_, timestamp, 1);

    BlockEntry *ptr = block.get();
    blocks_.emplace(key, std::move(block));
    return ptr;
}

void HashTableIndex::Touch(BlockEntry *block, int64_t timestamp, bool count_read, bool count_write_touch) {
    if (block == nullptr) {
        return;
    }
    auto loc_it = block->location_map.find(tier_name_);
    if (loc_it == block->location_map.end()) {
        return;
    }

    block->last_access_time = timestamp;
    loc_it->second.last_access_time = timestamp;
    if (count_read) {
        block->access_count += 1;
        loc_it->second.access_count += 1;
    }
    if (count_write_touch) {
        loc_it->second.write_touch_count += 1;
    }
}

void HashTableIndex::Remove(BlockEntry *block) {
    if (block == nullptr) {
        return;
    }
    blocks_.erase(block->key);
}

} // namespace kv_cache_manager
