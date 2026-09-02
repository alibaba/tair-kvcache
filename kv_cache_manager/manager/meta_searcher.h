#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
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
class CacheManager;

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

    struct HostCacheLocationInfo {
        // When true, the checker has already parsed the EventReport location
        // id and validated every spec URI while applying query visibility.
        bool has_reporter_identity = false;
        // Views borrow the immutable CacheLocation id and are consumed while
        // that location is still held by the current projection; they are
        // never retained in output.
        std::string_view reporter_medium;
        std::string_view reporter_host;
    };
    using CheckHostCacheLocationFunc =
        std::function<bool(const CacheLocation &location, HostCacheLocationInfo &out_info)>;

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
                                const CheckHostCacheLocationFunc *request_check_location = nullptr,
                                size_t p2p_host_count = 0) const;
    ErrorCode PrefixMatchWithMambaByHost(RequestContext *request_context,
                                         const KeyVector &keys,
                                         bool use_eagle_pop,
                                         const std::vector<std::string> &medium_filter,
                                         const std::vector<LocationSpecGroup> &location_spec_groups,
                                         std::vector<HostCacheMatch> &out_matches,
                                         const CheckHostCacheLocationFunc *request_check_location = nullptr,
                                         size_t p2p_host_count = 0) const;
    ErrorCode BatchGetLocation(RequestContext *request_context,
                               const KeyVector &keys,
                               const BlockMask &input_mask,
                               std::vector<CacheLocationMap> &out_location_maps);
    struct AddLocationResult {
        ErrorCode ec = EC_UNKNOWN;
        // EC_OK 时是可供业务使用的 location id；失败时若非空，仅可作为回滚定位符。
        std::string location_id;
    };
    ErrorCode BatchGetRawMeta(RequestContext *request_context,
                              const KeyVector &keys,
                              CacheLocationMapVector &out_location_maps,
                              PropertyMapVector &out_properties,
                              std::vector<ErrorCode> &out_error_codes);
    ErrorCode BatchAddLocation(RequestContext *request_context,
                               const KeyVector &keys,
                               const CacheLocationVector &locations,
                               std::vector<AddLocationResult> &out_results);
    struct AddLocationRollbackPlan {
        // Confirmed-successful items (EC_OK + non-empty location id). The
        // caller submits these to the standard location delete pipeline.
        KeyVector pipeline_keys;
        std::vector<std::string> pipeline_location_ids;
        // Indices into the original batch whose metadata is confirmed to hold
        // no reference (or was never written). The caller may delete their
        // allocated URIs directly.
        std::vector<size_t> direct_delete_indices;
    };
    // Reconciles a failed BatchAddLocation batch into a rollback plan.
    // Confirmed-successful items are routed to the standard delete pipeline;
    // uncertain items (location id generated but write result failed/unknown)
    // are idempotently deleted from metadata and synced before their URIs may
    // be released. Whenever an item's metadata state cannot be confirmed, its
    // URI is retained (not added to direct_delete_indices).
    ErrorCode ReconcileAddLocationRollback(RequestContext *request_context,
                                           const KeyVector &keys,
                                           const std::vector<AddLocationResult> &add_results,
                                           AddLocationRollbackPlan &out_plan);
    // Metadata-free classification used when no MetaSearcher is available:
    // confirmed-successful items go to pipeline_* and items without a location
    // id go to direct_delete_indices, while uncertain items stay unclassified
    // so their URIs are retained. Returns the number of uncertain items.
    static size_t ClassifyAddLocationRollback(const KeyVector &keys,
                                              const std::vector<AddLocationResult> &add_results,
                                              AddLocationRollbackPlan &out_plan);
    struct ReplaceLocationSpecsTask {
        std::string location_id;
        DataStorageType type;
        CacheLocationStatus status;
        std::vector<LocationSpec> specs;
        InternedLocationId interned_location_id;
        const InternedLocationId *borrowed_interned_location_id = nullptr;

        [[nodiscard]] const std::string &ResolvedLocationId() const noexcept {
            const auto *interned = ResolvedInternedLocationId();
            return interned ? **interned : location_id;
        }
        [[nodiscard]] const InternedLocationId *ResolvedInternedLocationId() const noexcept {
            if (interned_location_id) {
                return &interned_location_id;
            }
            return borrowed_interned_location_id && *borrowed_interned_location_id ? borrowed_interned_location_id
                                                                                   : nullptr;
        }
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
    class PrevalidatedTotalSize {
    public:
        [[nodiscard]] std::uint64_t value() const noexcept { return value_; }

    private:
        friend class CacheManager;
        explicit PrevalidatedTotalSize(std::uint64_t value) noexcept : value_(value) {}

        std::uint64_t value_ = 0;
    };
    struct MergeLocationSpecsTask {
        std::string location_id;
        DataStorageType type;
        CacheLocationStatus status;
        std::vector<LocationSpec> specs;
        // ReportEvent fully validates and parses every input URI before it
        // acquires the reporter mutation fence. Supplying this value lets its
        // internal merge path reuse the already computed aggregate size
        // instead of parsing every versioned URI a second time. Other callers
        // must leave it empty and receive the normal strict validation.
        std::optional<PrevalidatedTotalSize> prevalidated_total_size;
        InternedLocationId interned_location_id;
        // ReportEvent owns one canonical id per medium for the duration of
        // the synchronous Batch* call. Borrow it here so a 20K-key request
        // does not perform 20K atomic shared_ptr increments/decrements merely
        // to route tasks. A persisted CacheLocation still takes ownership.
        const InternedLocationId *borrowed_interned_location_id = nullptr;
        // ReportEvent overwhelmingly carries one spec per block. Keep that
        // value inline so a 20K-block request does not allocate 20K one-item
        // vectors; generic/multi-spec callers continue using specs unchanged.
        std::optional<LocationSpec> inline_spec;

        [[nodiscard]] const std::string &ResolvedLocationId() const noexcept {
            const auto *interned = ResolvedInternedLocationId();
            return interned ? **interned : location_id;
        }
        [[nodiscard]] const InternedLocationId *ResolvedInternedLocationId() const noexcept {
            if (interned_location_id) {
                return &interned_location_id;
            }
            return borrowed_interned_location_id && *borrowed_interned_location_id ? borrowed_interned_location_id
                                                                                   : nullptr;
        }
        [[nodiscard]] size_t SpecCount() const noexcept { return specs.size() + (inline_spec ? 1 : 0); }
        [[nodiscard]] bool SpecsEmpty() const noexcept { return SpecCount() == 0; }
        [[nodiscard]] const LocationSpec &SpecAt(size_t index) const noexcept {
            return index < specs.size() ? specs[index] : *inline_spec;
        }
        [[nodiscard]] LocationSpec &MutableSpecAt(size_t index) noexcept {
            return index < specs.size() ? specs[index] : *inline_spec;
        }
        void PushReportEventSpec(LocationSpec &&spec, size_t max_spec_count) {
            if (specs.empty() && !inline_spec) {
                inline_spec.emplace(std::move(spec));
                return;
            }
            if (inline_spec) {
                specs.reserve(max_spec_count < 2 ? 2 : max_spec_count);
                specs.push_back(std::move(*inline_spec));
                inline_spec.reset();
            }
            specs.push_back(std::move(spec));
        }
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
    // Allocation-light equivalent used by ReportEvent. Tasks for key i are
    // stored in [task_offsets[i], task_offsets[i + 1]); offsets must start at
    // zero, be nondecreasing, and end at flat_tasks.size(). URI ownership in
    // CacheManager-prevalidated tasks can be consumed during the synchronous
    // call; their spec names remain available for per-item failure mapping.
    // Non-prevalidated flat tasks retain the generic copy semantics.
    ErrorCode BatchMergeLocationSpecsFlat(RequestContext *request_context,
                                          const KeyVector &keys,
                                          const std::vector<size_t> &task_offsets,
                                          std::vector<MergeLocationSpecsTask> &flat_tasks,
                                          std::vector<ErrorCode> &out_per_key_ec,
                                          AcquireMetadataWriteLeaseFunc acquire_write_lease = nullptr);
    struct DeleteLocationSpecsTask {
        std::string location_id;
        std::vector<std::string> spec_names;
        InternedLocationId interned_location_id;
        const InternedLocationId *borrowed_interned_location_id = nullptr;

        [[nodiscard]] const std::string &ResolvedLocationId() const noexcept {
            const auto *interned = ResolvedInternedLocationId();
            return interned ? **interned : location_id;
        }
        [[nodiscard]] const InternedLocationId *ResolvedInternedLocationId() const noexcept {
            if (interned_location_id) {
                return &interned_location_id;
            }
            return borrowed_interned_location_id && *borrowed_interned_location_id ? borrowed_interned_location_id
                                                                                   : nullptr;
        }
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
                                     std::vector<std::vector<ErrorCode>> &out_batch_results,
                                     bool refresh_cache_from_persistent = false);
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
    class MergeLocationSpecsTaskView;

    ErrorCode BatchMergeLocationSpecsImpl(RequestContext *request_context,
                                          const KeyVector &keys,
                                          MergeLocationSpecsTaskView tasks,
                                          std::vector<ErrorCode> &out_per_key_ec,
                                          AcquireMetadataWriteLeaseFunc acquire_write_lease);

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
