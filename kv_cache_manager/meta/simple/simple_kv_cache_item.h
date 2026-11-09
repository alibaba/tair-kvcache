#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>

#include "kv_cache_manager/common/cache/advanced_cache.h"
#include "kv_cache_manager/meta/types.h"

namespace kv_cache_manager {

// 固定 header 后连续存放引用记录；ID/URI 在不可变数据对象中由新旧 item 共享。
// 块通过原 Cache::ObjectPtr/Deleter 交给 LRU；访问期间须持有 pin 或 item 的共享所有权。
class SimpleKVCacheItem {
public:
    // out_item 必须为空；成功后由调用方移交 LRU 或调用 Deleter，失败保持输出和旧块不变。
    // spec_name 由同一 instance 的 backend 提供，不在每个 item 中重复存储。
    static ErrorCode Create(const CacheLocationMap &locations,
                            const SharedLocationSpecName &spec_name,
                            const std::unordered_map<LocationId, InternedLocationId> &interned_ids,
                            Cache::ObjectPtr &out_item);
    static ErrorCode Merge(const SimpleKVCacheItem &current,
                           const CacheLocationMap &updates,
                           const SharedLocationSpecName &spec_name,
                           const std::unordered_map<LocationId, InternedLocationId> &interned_ids,
                           Cache::ObjectPtr &out_item);
    // 删除末条 location 后保留空 item，由原 manager 统一回收整 key 和记录恢复墓碑。
    static ErrorCode
    Remove(const SimpleKVCacheItem &current, const LocationIdVector &location_ids, Cache::ObjectPtr &out_item);
    static void Deleter(Cache::ObjectPtr value, MemoryAllocator *allocator);

    ErrorCode GetLocationValues(const SharedLocationSpecName &spec_name, CacheLocationVector &out_locations) const;
    ErrorCode GetLocations(const SharedLocationSpecName &spec_name, CacheLocationMap &out_locations) const;
    ErrorCode AppendLocationValues(const SharedLocationSpecName &spec_name, CacheLocationVector &out_locations) const;
    ErrorCode GetLocation(const SharedLocationSpecName &spec_name,
                          const LocationId &location_id,
                          CacheLocationConstPtr &out_location) const;
    ErrorCode GetLocationIds(LocationIdVector &out_location_ids) const;
    size_t LocationCount() const noexcept;
    // 返回稳定的 LRU charge 估算；共享数据按每条引用计费，不等同于实际 RSS。
    size_t Size() const noexcept;
    static ErrorCode EstimateWriteGrowth(const std::shared_ptr<SimpleKVCacheItem> &current,
                                         const CacheLocationMap &updates,
                                         bool replace,
                                         size_t &out_growth);

    SimpleKVCacheItem(const SimpleKVCacheItem &) = delete;
    SimpleKVCacheItem &operator=(const SimpleKVCacheItem &) = delete;

private:
    struct LocationData {
        InternedLocationId id;
        std::string uri;
    };
    struct LocationRecord {
        std::shared_ptr<const LocationData> data;
        int64_t create_time = 0;
    };
    static_assert(sizeof(LocationRecord) == 24, "Simple location reference must remain 24 bytes");

    static size_t EstimateDataCharge(size_t uri_capacity);
    static ErrorCode Allocate(size_t location_count, Cache::ObjectPtr &out_item);
    const LocationRecord &Record(size_t index) const noexcept;
    bool FindLocationIndex(const LocationId &location_id, size_t &out_index) const noexcept;
    ErrorCode AppendLocation(const std::shared_ptr<const LocationData> &data, int64_t create_time) noexcept;
    CacheLocationConstPtr ProjectLocation(const LocationRecord &record, const SharedLocationSpecName &spec_name) const;

    SimpleKVCacheItem() noexcept;
    ~SimpleKVCacheItem();

    uint32_t location_count_ = 0;
    uint32_t charge_bytes_ = 0;
};

static_assert(sizeof(SimpleKVCacheItem) == 8, "Simple item header must remain 8 bytes");

} // namespace kv_cache_manager
