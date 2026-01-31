#pragma once

#include <memory>
#include <string>
#include <vector>

#include "cache_location.h"
#include "kv_cache_manager/common/error_code.h"
#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/manager/select_location_policy.h"

namespace kv_cache_manager {
class MetaIndexer;

class MetaSearcher {
public:
    using KeyType = int64_t;
    using KeyVector = std::vector<KeyType>;
    using UriType = std::string;
    using UriVector = std::vector<UriType>;
    static const std::string PROPERTY_PREV_BLOCK_KEY;

    explicit MetaSearcher(const std::shared_ptr<MetaIndexer> &meta_manager);
    ~MetaSearcher();

    static std::string BatchErrorCodeToStr(const std::vector<std::vector<ErrorCode>> &batch_results);

    ErrorCode PrefixMatch(RequestContext *request_context,
                          const KeyVector &keys,
                          const BlockMask &input_mask,
                          CacheLocationVector &out_locations,
                          SelectLocationPolicy *policy) const;
    ErrorCode BatchGetBestLocation(RequestContext *request_context,
                                   const KeyVector &keys,
                                   CacheLocationVector &out_locations,
                                   SelectLocationPolicy *policy) const;
    ErrorCode ReverseRollSlideWindowMatch(RequestContext *request_context,
                                          const KeyVector &keys,
                                          int32_t sw_size,
                                          CacheLocationVector &out_locations,
                                          SelectLocationPolicy *policy) const;
    ErrorCode BatchGetLocation(RequestContext *request_context,
                               const KeyVector &keys,
                               const BlockMask &input_mask,
                               std::vector<CacheLocationMap> &out_location_maps);
    ErrorCode BatchAddLocation(RequestContext *request_context,
                               const KeyVector &keys,
                               const CacheLocationVector &locations,
                               std::vector<std::string> &out_location_ids);
    struct LocationUpdateTask {
        std::string location_id;
        CacheLocationStatus new_status;
    };
    ErrorCode BatchUpdateLocationStatus(RequestContext *request_context,
                                        const KeyVector &keys,
                                        const std::vector<std::vector<LocationUpdateTask>> &batch_tasks,
                                        std::vector<std::vector<ErrorCode>> &out_batch_results);
    struct LocationCASTask {
        std::string location_id;
        CacheLocationStatus old_status;
        CacheLocationStatus new_status;
    };
    ErrorCode BatchCASLocationStatus(RequestContext *request_context,
                                     const KeyVector &keys,
                                     const std::vector<std::vector<LocationCASTask>> &batch_tasks,
                                     std::vector<std::vector<ErrorCode>> &out_batch_results);
    struct LocationCADTask {
        std::string location_id;
        CacheLocationStatus expect_status;
    };
    ErrorCode BatchCADLocationStatus(RequestContext *request_context,
                                     const KeyVector &keys,
                                     const std::vector<std::vector<LocationCADTask>> &batch_tasks,
                                     std::vector<std::vector<ErrorCode>> &out_batch_results);
    ErrorCode BatchDeleteLocation(RequestContext *request_context,
                                  const KeyVector &keys,
                                  const std::vector<std::string> &location_ids,
                                  std::vector<ErrorCode> &results);

private:
    struct StorageTypeWeights {
        static constexpr size_t NFS = 5;          // NFS存储权重较高
        static constexpr size_t MOONCAKE = 3;     // Mooncake存储权重中等
        static constexpr size_t THREEFS = 3;      // 3FS存储权重较低
        static constexpr size_t TAIR_MEMPOOL = 3; // Tair存储权重最低
        static constexpr size_t DEFAULT = 1;      // 默认权重
    };
    struct MetaSearcherMetrics {
        int64_t index_serialize_time_us = 0;
        int64_t index_deserialize_time_us = 0;
    };

    ErrorCode PrefixMatchBestLocationImpl(RequestContext *request_context,
                                          const KeyVector &keys,
                                          CacheLocationVector &out_locations,
                                          SelectLocationPolicy *policy) const;

    std::shared_ptr<MetaIndexer> meta_indexer_;
};

} // namespace kv_cache_manager
