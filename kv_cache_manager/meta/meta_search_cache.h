#pragma once

#include <cstring>
#include <memory>
#include <shared_mutex>
#include <string_view>

#include "kv_cache_manager/common/cache/advanced_cache.h"

namespace kv_cache_manager {

class MetaCachePolicyConfig;

class MetaSearchCache {
public:
    using KeyType = int64_t;

public:
    MetaSearchCache();
    ~MetaSearchCache();

    ErrorCode Init(const std::shared_ptr<MetaCachePolicyConfig> &config);

    ErrorCode Put(const KeyType key, const std::string &value);

    ErrorCode Get(const KeyType key, std::string *out_value);

    void Delete(const KeyType key);

public:
    size_t GetCacheSize() const { return cache_size_; }
    size_t GetCacheUsage() const { return cache_->GetUsage(); }

private:
    std::shared_ptr<Cache> cache_;
    std::shared_ptr<Cache::CacheItemHelper> cache_item_helper_;
    size_t cache_size_;
};

struct MetaCacheItem {
    char *value_ = nullptr;
    uint32_t value_size_ = 0; // Maximum supported value size is 3.9GB

    size_t Size() const { return sizeof(MetaCacheItem) + value_size_; }

    const std::string_view GetValue() const { return std::string_view(value_, value_size_); }

    static MetaCacheItem *Create(const std::string &value) {
        MetaCacheItem *item = new MetaCacheItem();
        item->value_size_ = value.size();
        if (value.size() > 0) {
            item->value_ = new char[value.size()];
            memcpy(item->value_, value.data(), value.size());
        }
        return item;
    }
    static void Deleter(void *value, MemoryAllocator *allocator) {
        if (value) {
            MetaCacheItem *item = (MetaCacheItem *)value;
            if (item->value_size_ > 0) {
                delete[] item->value_;
                item->value_ = nullptr;
            }
            delete item;
        }
    }
};

} // namespace kv_cache_manager
