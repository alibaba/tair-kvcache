#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "kv_cache_manager/common/error_code.h"
#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/manager/select_location_policy.h"
#include "kv_cache_manager/meta/cache_location.h"
#include "kv_cache_manager/meta/types.h"

namespace kv_cache_manager {

using SubmitDelReqFunc = std::function<void(const std::vector<std::int64_t> &blk_keys,
                                            const std::vector<std::vector<std::string>> &loc_ids)>;

class MetaIndexer;

enum class LocationSelectStrategy : int32_t {
    LSS_UNSPECIFIED = 0,
    LSS_V6D_PREFIX = 1,   // 对应v6d侧，best_effort = false
    LSS_V6D_COVERAGE = 2, // 对应v6d侧，best_effort = true
    LSS_WEIGHTED_RANDOM = 3,
};

struct BackendSelector {
    DataStorageType backend_type;
    LocationSelectStrategy strategy;
};

class MetaSearcher {
public:
    using KeyType = int64_t;
    using KeyVector = std::vector<KeyType>;
    using UriType = std::string;
    using UriVector = std::vector<UriType>;

    explicit MetaSearcher(const std::shared_ptr<MetaIndexer> &meta_manager);
    MetaSearcher(const std::shared_ptr<MetaIndexer> &meta_indexer,
                 CheckLocDataExistFunc check_loc_data_exist,
                 SubmitDelReqFunc submit_del_req);
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
    ErrorCode BatchGetBestLocationByBackend(RequestContext *request_context,
                                            const KeyVector &keys,
                                            LocationsPerKey &out_locations,
                                            SelectLocationPolicy *policy,
                                            const std::vector<BackendSelector> &selectors) const;
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
    struct UpsertLocation {
        std::string location_id;
        DataStorageType type;
        CacheLocationStatus status;
        std::vector<LocationSpec> specs;
    };
    ErrorCode BatchUpsertLocations(RequestContext *request_context,
                                   const KeyVector &keys,
                                   const std::vector<std::vector<UpsertLocation>> &new_locations_per_key,
                                   std::vector<ErrorCode> &out_per_key_ec);
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
    ErrorCode BatchDeleteLocations(RequestContext *request_context,
                                   const KeyVector &keys,
                                   const LocationIdsPerKey &location_ids_per_key,
                                   std::vector<std::vector<ErrorCode>> &out_per_location_ec);
    ErrorCode CleanupLocationsByHost(RequestContext *request_context,
                                     const std::string &host_suffix,
                                     DataStorageType storage_type,
                                     size_t scan_batch_size = 1000,
                                     std::function<bool()> should_abort = nullptr);

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
    CheckLocDataExistFunc check_loc_data_exist_func_;
    SubmitDelReqFunc submit_del_req_func_;
};

} // namespace kv_cache_manager
