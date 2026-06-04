#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>

#include "kv_cache_manager/optimizer/config/types.h"

namespace kv_cache_manager {

class HashTableIndex {
public:
    explicit HashTableIndex(std::string tier_name);
    ~HashTableIndex() = default;

    [[nodiscard]] const std::string &tier_name() const { return tier_name_; }
    [[nodiscard]] size_t Size() const { return blocks_.size(); }

    BlockEntry *Find(int64_t key) const;
    BlockEntry *Insert(int64_t key, int64_t timestamp, int64_t ttl_ns);
    void Touch(BlockEntry *block, int64_t timestamp, bool count_read, bool count_write_touch);
    void Remove(BlockEntry *block);

private:
    std::string tier_name_;
    std::unordered_map<int64_t, std::unique_ptr<BlockEntry>> blocks_;
};

} // namespace kv_cache_manager
