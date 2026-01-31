#pragma once
#include <unordered_map>

#include "kv_cache_manager/optimizer/config/eviction_config.h"
#include "kv_cache_manager/optimizer/config/types.h"
#include "kv_cache_manager/optimizer/eviction_policy/base.h"
#include "kv_cache_manager/optimizer/eviction_policy/common_structure.h"
namespace kv_cache_manager {

class LruEvictionPolicy : public EvictionPolicy {
private:
    std::string name_;
    LruParams params_;
    struct LRUListNode : public LinkedListNode {
        BlockEntry *payload_;
        int64_t priority() const { return -payload_->last_access_time; }
    };
    LinkedList lru_list_;
    std::unordered_map<BlockEntry *, LRUListNode *> node_map_;

public:
    explicit LruEvictionPolicy(const std::string &name, const LruParams &params);

    ~LruEvictionPolicy() override;

    std::string name() const override { return name_; }
    void set_name(const std::string &name) override { name_ = name; }
    // TODO 应该根据块的last_access_time来维护LRU顺序
    void OnBlockWritten(BlockEntry *block) override;
    void OnNodeWritten(std::vector<BlockEntry *> &blocks) override;
    void OnBlockAccessed(BlockEntry *block, int64_t timestamp) override;
    std::vector<BlockEntry *> EvictBlocks(size_t count) override;
    void Clear() override;
    size_t size() const override { return node_map_.size(); }
};
} // namespace kv_cache_manager