#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "kv_cache_manager/common/error_code.h"
#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/data_storage/snapshot_uri_utils.h"
#include "kv_cache_manager/manager/select_location_policy.h"
#include "kv_cache_manager/meta/cache_location.h"
#include "kv_cache_manager/meta/types.h"

namespace kv_cache_manager {

using SubmitDelReqFunc = std::function<void(const std::vector<std::int64_t> &blk_keys,
                                            const std::vector<std::vector<std::string>> &loc_ids,
                                            const std::vector<std::vector<std::string>> &expected_location_values,
                                            // Skip physical URI deletion for
                                            // externally owned metadata.
                                            bool metadata_only)>;

class MetaIndexer;
class LocationSpecGroup;

enum class LocationSelectStrategy : int32_t {
    LSS_UNSPECIFIED = 0,
    LSS_V6D_PREFIX = 1,   // 对应 v6d 侧，best_effort = false
    LSS_V6D_COVERAGE = 2, // 对应 v6d 侧，best_effort = true
    LSS_WEIGHTED_RANDOM = 3,
};

struct BackendSelector {
    DataStorageType backend_type;
    LocationSelectStrategy strategy;
};

class MetaSearcher {
public:
    using MetadataWriteLease = std::shared_ptr<void>;
    using AcquireMetadataWriteLeaseFunc = std::function<std::pair<ErrorCode, MetadataWriteLease>()>;
    using KeyType = int64_t;
    using KeyVector = std::vector<KeyType>;
    using UriType = std::string;
    using UriVector = std::vector<UriType>;

    struct HostCacheMatch {
        std::string host_ip_port;
        int64_t local;
        int64_t p2p_1_fetch;
        int64_t p2p_1_total_match;
    };

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
                                            const std::vector<BackendSelector> &selectors,
                                            const std::vector<std::string> &requested_spec_names = {},
                                            const BlockMask &input_mask = BlockMask{}) const;
    ErrorCode ReverseRollSlideWindowMatch(RequestContext *request_context,
                                          const KeyVector &keys,
                                          int32_t sw_size,
                                          CacheLocationVector &out_locations,
                                          SelectLocationPolicy *policy) const;
    ErrorCode PrefixMatchByHost(RequestContext *request_context,
                                const KeyVector &keys,
                                bool use_eagle_pop,
                                const std::vector<std::string> &medium_filter,
                                std::vector<HostCacheMatch> &out_matches,
                                const CheckLocDataExistFunc *request_check_loc_data_exist = nullptr) const;
    ErrorCode PrefixMatchWithMambaByHost(RequestContext *request_context,
                                         const KeyVector &keys,
                                         bool use_eagle_pop,
                                         const std::vector<std::string> &medium_filter,
                                         const std::vector<LocationSpecGroup> &location_spec_groups,
                                         std::vector<HostCacheMatch> &out_matches,
                                         const CheckLocDataExistFunc *request_check_loc_data_exist = nullptr) const;
    ErrorCode BatchGetLocation(RequestContext *request_context,
                               const KeyVector &keys,
                               const BlockMask &input_mask,
                               std::vector<CacheLocationMap> &out_location_maps);
    struct AddLocationResult {
        ErrorCode ec = EC_UNKNOWN;
        // EC_OK 时是可供业务使用的 location id；失败时若非空，仅可作为回滚定位符。
        std::string location_id;
    };
    ErrorCode BatchAddLocation(RequestContext *request_context,
                               const KeyVector &keys,
                               const CacheLocationVector &locations,
                               std::vector<AddLocationResult> &out_results);
    struct ReplaceLocationSpecsTask {
        std::string location_id;
        DataStorageType type;
        CacheLocationStatus status;
        std::vector<LocationSpec> specs;
    };
    // Replaces existing specs or creates the stable location in one metadata
    // read-modify-write operation per batch. Keys and location ids within a
    // key must be unique. Every task requires non-empty, uniquely named specs
    // with valid URIs and cannot mix versioned/unversioned specs or multiple
    // snapshot versions. When supplied, the write-lease callback is invoked
    // once after the metadata read and before the first mutation; the returned
    // lease is retained until the operation returns.
    ErrorCode BatchReplaceLocationSpecs(RequestContext *request_context,
                                        const KeyVector &keys,
                                        const std::vector<std::vector<ReplaceLocationSpecsTask>> &tasks_per_key,
                                        std::vector<ErrorCode> &out_per_key_ec,
                                        AcquireMetadataWriteLeaseFunc acquire_write_lease = nullptr);
    struct MergeLocationSpecsTask {
        std::string location_id;
        DataStorageType type;
        CacheLocationStatus status;
        std::vector<LocationSpec> specs;
    };
    // Keys and location ids within a key must be unique. Every task requires
    // non-empty, uniquely named specs with valid URIs and cannot mix
    // versioned/unversioned specs or multiple snapshot versions. The block
    // existence state and every requested target location are read
    // together, then merged in one targeted RMW phase. The optional write
    // lease is acquired once after that read and before the first mutation;
    // it is retained through the upsert so a concurrent lifecycle change can
    // fence work that was admitted under an older reporter generation.
    ErrorCode BatchMergeLocationSpecs(RequestContext *request_context,
                                      const KeyVector &keys,
                                      const std::vector<std::vector<MergeLocationSpecsTask>> &tasks_per_key,
                                      std::vector<ErrorCode> &out_per_key_ec,
                                      AcquireMetadataWriteLeaseFunc acquire_write_lease = nullptr);
    struct DeleteLocationSpecsTask {
        std::string location_id;
        std::vector<std::string> spec_names;
    };
    // Missing block/location targets are idempotent EC_OK. Keys and location
    // ids within a key must be unique, and every task must name at least one
    // non-empty spec. When requested, out_missing_targets mirrors
    // tasks_per_key and marks missing block/location no-ops; an existing
    // location with only missing spec_names is not marked. The optional write
    // lease is acquired once after the metadata read and before the first
    // mutation.
    ErrorCode BatchDeleteLocationSpecs(RequestContext *request_context,
                                       const KeyVector &keys,
                                       const std::vector<std::vector<DeleteLocationSpecsTask>> &tasks_per_key,
                                       std::vector<std::vector<ErrorCode>> &out_batch_results,
                                       std::vector<std::vector<bool>> *out_missing_targets = nullptr,
                                       AcquireMetadataWriteLeaseFunc acquire_write_lease = nullptr);
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
        // Optional exact serialized value checked inside the metadata RMW.
        // This closes the gap between a cleanup scan and its status CAS when
        // a stable location id is refreshed by a newer snapshot.
        std::string expected_location_value;
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
    ErrorCode BatchDeleteLocations(RequestContext *request_context,
                                   const KeyVector &keys,
                                   const LocationIdsPerKey &location_ids_per_key,
                                   std::vector<std::vector<ErrorCode>> &out_per_location_ec,
                                   const std::vector<std::vector<std::string>> &expected_location_values = {},
                                   bool adjust_storage_usage = true,
                                   bool adjust_reclaimed_key_count = true);
    using LocationVisitor =
        std::function<void(KeyType block_key, const std::string &location_id, const CacheLocation &location)>;
    ErrorCode VisitAllLocations(RequestContext *request_context, size_t scan_batch_size, LocationVisitor visitor);
    using LocationCleanupPredicate =
        std::function<bool(KeyType block_key, const std::string &location_id, const CacheLocation &location)>;
    // Returning true from should_abort is a successful cancellation and therefore
    // returns EC_OK; it is not a storage or scan failure.
    ErrorCode CleanupLocationsByPredicate(RequestContext *request_context,
                                          DataStorageType storage_type,
                                          size_t scan_batch_size,
                                          LocationCleanupPredicate should_delete,
                                          std::function<bool()> should_abort = nullptr,
                                          AcquireMetadataWriteLeaseFunc acquire_cleanup_lease = nullptr);
    ErrorCode CleanupLocationsByHost(RequestContext *request_context,
                                     const std::string &host_suffix,
                                     DataStorageType storage_type,
                                     size_t scan_batch_size = 1000,
                                     std::function<bool()> should_abort = nullptr,
                                     AcquireMetadataWriteLeaseFunc acquire_cleanup_lease = nullptr);

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
