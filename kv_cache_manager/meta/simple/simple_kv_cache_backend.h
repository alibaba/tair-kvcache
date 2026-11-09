#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#include "kv_cache_manager/common/cache/lru_cache.h"
#include "kv_cache_manager/meta/meta_cache_base_backend.h"
#include "kv_cache_manager/meta/simple/simple_kv_cache_item.h"

namespace kv_cache_manager {

// 紧凑 local 元数据存储，可独立使用或作为 cached 热层；LRU entry 发布不可变 item 的共享引用。
// Redis 队列、恢复状态机和 reporter 状态继续由原组件管理。
class SimpleKVCacheBackend final : public MetaCacheBaseBackend {
public:
    explicit SimpleKVCacheBackend(SharedLocationSpecName spec_name);
    ~SimpleKVCacheBackend() override;

    SimpleKVCacheBackend(const SimpleKVCacheBackend &) = delete;
    SimpleKVCacheBackend &operator=(const SimpleKVCacheBackend &) = delete;

    // 生命周期沿用原 local：初始化和配置注入完成后发布，Close 前由上层排空操作。
    std::string GetStorageType() noexcept override;
    ErrorCode Init(const std::string &instance_id,
                   const std::shared_ptr<MetaStorageBackendConfig> &config) noexcept override;
    ErrorCode Open() noexcept override;
    ErrorCode Close() noexcept override;
    std::vector<ErrorCode> Put(RequestContext *request_context,
                               const KeyTypeVec &keys,
                               const CacheLocationMapVector &locations,
                               const PropertyMapVector &properties) noexcept override;
    std::vector<ErrorCode> PutIfAbsent(RequestContext *request_context,
                                       const KeyTypeVec &keys,
                                       const CacheLocationMapVector &locations,
                                       const PropertyMapVector &properties) noexcept override;
    std::vector<ErrorCode> Upsert(RequestContext *request_context,
                                  const KeyTypeVec &keys,
                                  const CacheLocationMapVector &locations,
                                  const PropertyMapVector &properties) noexcept override;
    std::vector<ErrorCode> Delete(RequestContext *request_context, const KeyTypeVec &keys) noexcept override;
    std::vector<ErrorCode> DeleteLocations(RequestContext *request_context,
                                           const KeyTypeVec &keys,
                                           const LocationIdsPerKey &location_ids) noexcept override;
    std::vector<ErrorCode> Put(RequestContext *request_context,
                               const KeyTypeVec &keys,
                               const CacheLocationMapVector &locations,
                               const PropertyMapVector &properties,
                               const std::vector<ErrorCode> &previous_error_codes) noexcept override;
    std::vector<ErrorCode> PutIfAbsent(RequestContext *request_context,
                                       const KeyTypeVec &keys,
                                       const CacheLocationMapVector &locations,
                                       const PropertyMapVector &properties,
                                       const std::vector<ErrorCode> &previous_error_codes) noexcept override;
    std::vector<ErrorCode> Upsert(RequestContext *request_context,
                                  const KeyTypeVec &keys,
                                  const CacheLocationMapVector &locations,
                                  const PropertyMapVector &properties,
                                  const std::vector<ErrorCode> &previous_error_codes) noexcept override;
    std::vector<ErrorCode> Delete(RequestContext *request_context,
                                  const KeyTypeVec &keys,
                                  const std::vector<ErrorCode> &previous_error_codes) noexcept override;
    std::vector<ErrorCode> DeleteLocations(RequestContext *request_context,
                                           const KeyTypeVec &keys,
                                           const LocationIdsPerKey &location_ids,
                                           const std::vector<ErrorCode> &previous_error_codes) noexcept override;
    std::vector<ErrorCode> Get(RequestContext *request_context,
                               const KeyTypeVec &keys,
                               CacheLocationMapVector &out_locations,
                               PropertyMapVector &out_properties) noexcept override;
    std::vector<ErrorCode> GetLocations(RequestContext *request_context,
                                        const KeyTypeVec &keys,
                                        CacheLocationMapVector &out_locations) noexcept override;
    std::vector<ErrorCode> GetLocationValues(RequestContext *request_context,
                                             const KeyTypeVec &keys,
                                             LocationsPerKey &out_locations) noexcept override;
    std::vector<ErrorCode> GetLocationValuesCompact(RequestContext *request_context,
                                                    const KeyType *keys,
                                                    size_t key_count,
                                                    CompactLocationsPerKey &out_locations) noexcept override;
    std::vector<std::vector<ErrorCode>> GetLocations(RequestContext *request_context,
                                                     const KeyTypeVec &keys,
                                                     const LocationIdsPerKey &location_ids,
                                                     LocationsPerKey &out_locations) noexcept override;
    std::vector<std::vector<ErrorCode>>
    GetLocationsWithKeyStatus(RequestContext *request_context,
                              const KeyTypeVec &keys,
                              const LocationIdsPerKey &location_ids,
                              LocationsPerKey &out_locations,
                              std::vector<ErrorCode> &out_key_error_codes) noexcept override;
    std::vector<ErrorCode>
    GetSingleLocationsWithKeyStatus(RequestContext *request_context,
                                    const KeyTypeVec &keys,
                                    const LocationIdRefVector &location_ids,
                                    CacheLocationVector &out_locations,
                                    std::vector<ErrorCode> &out_key_error_codes) noexcept override;
    std::vector<ErrorCode> GetLocationsForMaintenance(RequestContext *request_context,
                                                      const KeyTypeVec &keys,
                                                      CacheLocationMapVector &out_locations) noexcept override;
    std::vector<std::vector<ErrorCode>> GetLocationsForMaintenance(RequestContext *request_context,
                                                                   const KeyTypeVec &keys,
                                                                   const LocationIdsPerKey &location_ids,
                                                                   LocationsPerKey &out_locations) noexcept override;
    std::vector<ErrorCode> GetLocationIds(RequestContext *request_context,
                                          const KeyTypeVec &keys,
                                          LocationIdsPerKey &out_location_ids) noexcept override;
    std::vector<ErrorCode> GetLocationIdsForMaintenance(RequestContext *request_context,
                                                        const KeyTypeVec &keys,
                                                        LocationIdsPerKey &out_location_ids) noexcept override;
    std::vector<ErrorCode> GetLocationMapsForMaintenance(RequestContext *request_context,
                                                         const KeyTypeVec &keys,
                                                         CacheLocationMapVector &out_locations) noexcept override;
    std::vector<ErrorCode> GetProperties(RequestContext *request_context,
                                         const KeyTypeVec &keys,
                                         const std::vector<std::string> &field_names,
                                         PropertyMapVector &out_properties) noexcept override;
    std::vector<ErrorCode> Exists(RequestContext *request_context,
                                  const KeyTypeVec &keys,
                                  std::vector<bool> &out_is_exist_vec) noexcept override;
    std::vector<ErrorCode> ExistsLocation(RequestContext *request_context,
                                          const KeyTypeVec &keys,
                                          std::vector<bool> &out_exists) noexcept override;
    ErrorCode ListKeys(RequestContext *request_context,
                       const std::string &cursor,
                       const int64_t limit,
                       std::string &out_next_cursor,
                       std::vector<KeyType> &out_keys) noexcept override;
    ErrorCode ScanLocationsForMaintenance(RequestContext *request_context,
                                          const std::string &cursor,
                                          int64_t limit,
                                          MaintenanceScanBatch &out) noexcept override;
    std::vector<ErrorCode> DeleteLocationsForMaintenance(RequestContext *request_context,
                                                         const KeyTypeVec &keys,
                                                         const LocationIdsPerKey &location_ids) noexcept override;
    std::vector<ErrorCode>
    DeleteLocationsForMaintenance(RequestContext *request_context,
                                  const KeyTypeVec &keys,
                                  const LocationIdsPerKey &location_ids,
                                  const std::vector<ErrorCode> &previous_error_codes) noexcept override;
    ErrorCode RandomSample(RequestContext *request_context,
                           const int64_t count,
                           std::vector<KeyType> &out_keys) noexcept override;
    ErrorCode SampleReclaimKeys(RequestContext *request_context,
                                const int64_t count,
                                std::vector<KeyType> &out_keys) noexcept override;
    ErrorCode SampleReclaimCandidates(RequestContext *request_context,
                                      int64_t count,
                                      ReclaimCandidateVector &out_candidates,
                                      bool require_read_success) noexcept override;
    std::vector<ErrorCode>
    GetLastAccessTimesForMaintenance(RequestContext *request_context,
                                     const KeyTypeVec &keys,
                                     std::vector<int64_t> &out_last_access_times) noexcept override;
    size_t TouchKeysForMaintenance(const KeyTypeVec &keys) noexcept override;
    ErrorCode PutMetaData(const FieldMap &field_maps) noexcept override;
    ErrorCode GetMetaData(FieldMap &field_maps) noexcept override;
    ErrorCode DeleteMetaData() noexcept override;
    size_t GetMemUsage() const noexcept override;
    ErrorCode CheckWriteCapacity(KeyType key,
                                 const CacheLocationMap &locations,
                                 const PropertyMap &properties,
                                 bool replace) noexcept override;
    size_t GetLocalKeyCount() const noexcept override;
    size_t GetLocalLocationCount() const noexcept override;
    int64_t GetOldestAccessTime() const noexcept override;

private:
    enum class WriteMode {
        Put,
        PutIfAbsent,
        Upsert
    };
    // 引用仅在 LRU shard 锁内读写，回调外使用已持有的共享快照。
    using CacheValue = std::shared_ptr<SimpleKVCacheItem>;

    std::vector<ErrorCode> Write(const KeyTypeVec &keys,
                                 const CacheLocationMapVector &locations,
                                 const PropertyMapVector &properties,
                                 const std::vector<ErrorCode> &previous_error_codes,
                                 WriteMode mode);
    std::vector<ErrorCode> DeleteKeys(const KeyTypeVec &keys, const std::vector<ErrorCode> &previous_error_codes);
    std::vector<ErrorCode> DeleteLocationBatch(const KeyTypeVec &keys,
                                               const LocationIdsPerKey &location_ids,
                                               const std::vector<ErrorCode> &previous_error_codes);
    std::shared_ptr<SimpleKVCacheItem> LoadItem(KeyType key);
    // 调用者持有对应 LRU shard 的 mutation 锁；引用替换和计费更新在该锁内完成。
    // LRU 回调不得获取 mutation 锁或再次调用 LRU。
    ErrorCode PublishItem(KeyType key, const std::shared_ptr<SimpleKVCacheItem> &item, bool replace);
    static void DeleteCacheValue(Cache::ObjectPtr value, MemoryAllocator *allocator);
    static size_t EstimateEntryOverhead() noexcept;
    uint32_t ShardIndex(KeyType key) const noexcept;
    std::vector<ErrorCode> ReadMaps(const KeyTypeVec &keys, CacheLocationMapVector &out_locations);
    std::vector<ErrorCode> ReadIds(const KeyTypeVec &keys, LocationIdsPerKey &out_location_ids);
    std::vector<std::vector<ErrorCode>> ReadLocations(const KeyTypeVec &keys,
                                                      const LocationIdsPerKey &location_ids,
                                                      LocationsPerKey &out_locations,
                                                      std::vector<ErrorCode> &out_key_error_codes);
    ErrorCode CollectScanBatch(const std::string &cursor,
                               int64_t limit,
                               bool with_values,
                               std::vector<std::pair<KeyType, std::shared_ptr<SimpleKVCacheItem>>> &out_entries,
                               std::string &out_next_cursor);

    SharedLocationSpecName spec_name_;
    LocationIdInterner location_id_interner_;
    std::shared_ptr<LRUCache> cache_;
    std::shared_ptr<Cache::CacheItemHelper> cache_item_helper_;
    std::vector<std::mutex> mutation_mutex_shards_;
    bool open_ = false;
};

} // namespace kv_cache_manager
