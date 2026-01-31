#include "kv_cache_manager/meta/meta_search_cache.h"

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/string_util.h"
#include "kv_cache_manager/config/meta_cache_policy_config.h"

namespace kv_cache_manager {

MetaSearchCache::MetaSearchCache() : cache_item_helper_(std::make_shared<Cache::CacheItemHelper>()), cache_size_(0) {}

MetaSearchCache::~MetaSearchCache() {
    cache_.reset();
    cache_item_helper_.reset();
}

ErrorCode MetaSearchCache::Init(const std::shared_ptr<MetaCachePolicyConfig> &config) {
    std::string cache_type = config->GetCachePolicyType();
    StringUtil::ToLower(cache_type);
    if (cache_type != "lru") {
        KVCM_LOG_ERROR("meta search cache init failed, only support lru, type [%s]",
                       cache_type.c_str());
        return EC_ERROR;
    }
    cache_size_ = config->GetCapacity();
    int32_t cache_shard_bits = config->GetCacheShardBits();
    double high_pri_pool_ratio = config->GetHighPriPoolRatio();
    cache_ =
        NewLRUCache(cache_size_ * 1024 * 1024, cache_shard_bits, /*strict_capacity_limit*/ true, high_pri_pool_ratio);
    if (!cache_) {
        KVCM_LOG_ERROR(
            "create meta search cache failed, cache size [%lu]MB, cache shard bits [%d], high pri pool ratio [%f]",
            cache_size_,
            cache_shard_bits,
            high_pri_pool_ratio);
        return EC_ERROR;
    }
    cache_item_helper_ = std::make_shared<Cache::CacheItemHelper>();
    cache_item_helper_->del_cb = MetaCacheItem::Deleter;
    KVCM_LOG_INFO(
        "create meta search cache success, cache size [%lu]MB, cache shard bits [%d], high pri pool ratio [%f]",
        cache_size_,
        cache_shard_bits,
        high_pri_pool_ratio);
    return EC_OK;
}

ErrorCode MetaSearchCache::Put(const KeyType key, const std::string &value) {
    MetaCacheItem *item = MetaCacheItem::Create(value);
    // if lru shard cache size is less than value size, lru will delete the item and return ok
    return cache_->Insert(std::to_string(key), item, cache_item_helper_.get(), item->Size());
}

ErrorCode MetaSearchCache::Get(const KeyType key, std::string *out_value) {
    Cache::Handle *handle = cache_->Lookup(std::to_string(key));
    if (handle) {
        MetaCacheItem *item = (MetaCacheItem *)cache_->Value(handle);
        out_value->assign(item->GetValue());
        cache_->Release(handle);
        handle = nullptr;
        return EC_OK;
    }
    return EC_NOENT;
}

void MetaSearchCache::Delete(const KeyType key) { cache_->Erase(std::to_string(key)); }

} // namespace kv_cache_manager
