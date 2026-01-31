#include "kv_cache_manager/manager/data_storage_selector.h"

#include <array>
#include <cassert>
#include <cinttypes>
#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "kv_cache_manager/common/error_code.h"
#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/config/cache_config.h"
#include "kv_cache_manager/config/instance_group.h"
#include "kv_cache_manager/config/instance_info.h"
#include "kv_cache_manager/config/registry_manager.h"
#include "kv_cache_manager/data_storage/data_storage_backend.h"
#include "kv_cache_manager/data_storage/data_storage_manager.h"
#include "kv_cache_manager/data_storage/storage_config.h"
#include "kv_cache_manager/meta/meta_indexer.h"
#include "kv_cache_manager/meta/meta_indexer_manager.h"

namespace kv_cache_manager {

#define PREFIX_LOG(LEVEL, format, args...)                                                                             \
    do {                                                                                                               \
        KVCM_LOG_##LEVEL("trace_id [%s] | " format, trace_id.c_str(), ##args);                                         \
    } while (0)

DataStorageSelector::DataStorageSelector(std::shared_ptr<MetaIndexerManager> meta_indexer_manager,
                                         std::shared_ptr<RegistryManager> registry_manager)
    : meta_indexer_manager_(std::move(meta_indexer_manager)), registry_manager_(std::move(registry_manager)) {}

/**
 * @brief Calculate the intersection of available backends and
 * configured candidates
 *
 * This helper function filters the list of all available data storage
 * backends to only include those that are both available and in the
 * instance group's configured candidate list.  It also performs an
 * additional availability check to prevent false positives from the
 * data storage manager.
 *
 * @param request_context Shared pointer to the request context
 * @param avail_backends Vector of currently available data storage
 * backends
 * @param configured_candidates Vector of backend names configured as
 * candidates for this instance group
 * @param storage_quota_avail_array Store quota capacity availability
 * result, only true can be counted as final candidate
 * @param candidates_out Output vector to store the filtered list of
 * candidate backends
 */
void GetCandidates(RequestContext const *request_context,
                   const std::vector<std::shared_ptr<DataStorageBackend>> &avail_backends,
                   const std::vector<std::string> &configured_candidates,
                   const std::array<bool, 5> &storage_quota_avail_array,
                   std::vector<std::shared_ptr<DataStorageBackend>> &candidates_out) {
    const auto &trace_id = request_context->trace_id();
    for (const std::string &candidate : configured_candidates) {
        for (const std::shared_ptr<DataStorageBackend> &backend : avail_backends) {
            if (backend == nullptr) {
                PREFIX_LOG(WARN, "data storage backend is nullptr");
                continue;
            }
            if (backend->GetStorageConfig().global_unique_name() == candidate) {
                // NOTE: this availability check serves to prevent false
                // positive results that might be returned by
                // data_storage_manager->GetAvailableStorages() (which
                // is highly likely not to be, though), rather than to
                // enhance the accuracy of the storage availability
                // assessment
                if (!backend->Available()) {
                    PREFIX_LOG(WARN, "data storage backend is not available: %s", candidate.c_str());
                    continue;
                }
                auto type = backend->GetType();
                if (type == DataStorageType::DATA_STORAGE_TYPE_VCNS_HF3FS) {
                    // treat vcns_hf3fs as hf3fs
                    type = DataStorageType::DATA_STORAGE_TYPE_HF3FS;
                }
                try {
                    if (!storage_quota_avail_array.at(static_cast<std::uint8_t>(type))) {
                        PREFIX_LOG(WARN,
                                   "data storage type [%d] quota is reached or exceeded",
                                   static_cast<std::uint8_t>(type));
                        continue;
                    }
                } catch (const std::out_of_range &e) {
                    PREFIX_LOG(WARN, "data storage type out of range: %d", static_cast<std::uint8_t>(type));
                    continue;
                }
                candidates_out.emplace_back(backend);
            }
        }
    }
}

/**
 * @brief Select a backend of a specific type from the candidates list
 *
 * This helper function searches through the candidate backends to find
 * one that matches the target type. If no exact match is found and
 * fallback is enabled, it returns the first available candidate.
 *
 * @param request_context Shared pointer to the request context
 * @param candidates Vector of candidate backends to select from
 * @param target_type The desired data storage type to select
 * @param can_fallback Whether to allow fallback if no exact type match
 * is found
 * @return Shared pointer to the selected DataStorageBackend, or nullptr
 * if none found
 */
std::shared_ptr<DataStorageBackend> SelectByType(RequestContext const *request_context,
                                                 const std::vector<std::shared_ptr<DataStorageBackend>> &candidates,
                                                 const DataStorageType &target_type,
                                                 const bool can_fallback) noexcept {
    const auto &trace_id = request_context->trace_id();
    if (candidates.empty()) {
        PREFIX_LOG(WARN, "storage candidate list is empty");
        return nullptr;
    }

    for (const auto &backend : candidates) {
        assert(backend);
        if (backend->GetType() == target_type) {
            PREFIX_LOG(DEBUG, "selected a storage backend");
            return backend;
        }
    }

    if (can_fallback) {
        PREFIX_LOG(DEBUG, "fallback to a storage backend");
        return candidates.front(); // TODO: consider adapting a specific fallback policy
    }

    PREFIX_LOG(WARN, "no matching storage backend and fallback not allowed");
    return nullptr;
}

/**
 * @brief Select a backend based on the cache preference strategy
 *
 * This helper function implements the selection logic based on the
 * configured cache preference strategy.  Different strategies are
 * supported including enforcing/preferring specific backend types.
 *
 * @param request_context Shared pointer to the request context
 * @param candidates Vector of candidate backends to select from
 * @param preference The cache preference strategy to apply
 * @return Shared pointer to the selected DataStorageBackend, or nullptr
 * if none found
 */
std::shared_ptr<DataStorageBackend> Select(RequestContext const *request_context,
                                           const std::vector<std::shared_ptr<DataStorageBackend>> &candidates,
                                           const CachePreferStrategy &preference) {
    switch (preference) {
    case CachePreferStrategy::CPS_ALWAYS_3FS:
        return SelectByType(request_context, candidates, DataStorageType::DATA_STORAGE_TYPE_HF3FS, false);
    case CachePreferStrategy::CPS_PREFER_3FS:
        return SelectByType(request_context, candidates, DataStorageType::DATA_STORAGE_TYPE_HF3FS, true);
    case CachePreferStrategy::CPS_ALWAYS_VCNS_3FS:
        return SelectByType(request_context, candidates, DataStorageType::DATA_STORAGE_TYPE_VCNS_HF3FS, false);
    case CachePreferStrategy::CPS_PREFER_VCNS_3FS:
        return SelectByType(request_context, candidates, DataStorageType::DATA_STORAGE_TYPE_VCNS_HF3FS, true);
    case CachePreferStrategy::CPS_ALWAYS_MOONCAKE:
        return SelectByType(request_context, candidates, DataStorageType::DATA_STORAGE_TYPE_MOONCAKE, false);
    case CachePreferStrategy::CPS_PREFER_MOONCAKE:
        return SelectByType(request_context, candidates, DataStorageType::DATA_STORAGE_TYPE_MOONCAKE, true);
    case CachePreferStrategy::CPS_ALWAYS_TAIR_MEMPOOL:
        return SelectByType(request_context, candidates, DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL, false);
    case CachePreferStrategy::CPS_PREFER_TAIR_MEMPOOL:
        return SelectByType(request_context, candidates, DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL, true);
    case CachePreferStrategy::CPS_UNSPECIFIED:
        // break; skipped intentionally
    default:
        return SelectByType(request_context, candidates, DataStorageType::DATA_STORAGE_TYPE_UNKNOWN, true);
    }
}

void DataStorageSelector::DoCleanup() {}

DataStorageSelectResult
DataStorageSelector::SelectCacheWriteDataStorageBackend(RequestContext *request_context,
                                                        const std::string &instance_group) const noexcept {
    SPAN_TRACER(request_context);
    DataStorageSelectResult result{ErrorCode::EC_UNKNOWN, DataStorageType::DATA_STORAGE_TYPE_UNKNOWN, ""};
    if (!request_context) {
        KVCM_LOG_WARN("request context is nullptr");
        result.ec = ErrorCode::EC_BADARGS;
        return result;
    }
    const auto &trace_id = request_context->trace_id();

    if (meta_indexer_manager_ == nullptr) {
        PREFIX_LOG(WARN, "meta indexer manager is nullptr");
        result.ec = ErrorCode::EC_ERROR;
        return result;
    }

    if (registry_manager_ == nullptr) {
        PREFIX_LOG(WARN, "registry manager is nullptr");
        result.ec = ErrorCode::EC_ERROR;
        return result;
    }

    const auto data_storage_manager = registry_manager_->data_storage_manager();
    if (data_storage_manager == nullptr) {
        PREFIX_LOG(WARN, "data storage manager is nullptr");
        result.ec = ErrorCode::EC_INSTANCE_NOT_EXIST;
        return result;
    }

    // get currently available data storage backend list
    const std::vector<std::shared_ptr<DataStorageBackend>> avail_backends =
        data_storage_manager->GetAvailableStorages();
    if (avail_backends.empty()) {
        PREFIX_LOG(WARN, "no available data storage backend, instance group: %s", instance_group.c_str());
        result.ec = ErrorCode::EC_NOENT;
        return result;
    }

    // get the instance group
    const auto [ec0, ig] = registry_manager_->GetInstanceGroup(request_context, instance_group);
    if (ec0 != ErrorCode::EC_OK || ig == nullptr) {
        PREFIX_LOG(WARN, "get instance group failed, instance group: %s", instance_group.c_str());
        result.ec = ErrorCode::EC_INSTANCE_NOT_EXIST;
        return result;
    }

    // evaluate the total used byte size of this group
    const auto quota = ig->quota();
    const auto [ec1, instance_infos] = registry_manager_->ListInstanceInfo(request_context, instance_group);
    if (ec1 != ErrorCode::EC_OK) {
        PREFIX_LOG(WARN, "list instances info failed, error code: [%d]", static_cast<std::int32_t>(ec1));
        result.ec = ErrorCode::EC_ERROR;
        return result;
    }
    const std::size_t group_used_byte_size = CalcGroupUsedSize(request_context, instance_infos);
    if (quota.capacity() < 0 || group_used_byte_size >= static_cast<std::size_t>(quota.capacity())) {
        PREFIX_LOG(WARN,
                   "instance group: [%s], quota capacity: [%" PRId64 "], has been reached or exceeded: [%zu]",
                   instance_group.c_str(),
                   quota.capacity(),
                   group_used_byte_size);
        result.ec = ErrorCode::EC_ERROR;
        return result;
    }

    // construct the availability table of each storage type in this
    // instance group
    std::array<bool, 5> storage_quota_avail_array = {true, true, true, true, true};
    GenStorageQuotaAvailTable(request_context, quota, instance_infos, storage_quota_avail_array);

    // get the configured data storage candidate list of this instance group
    const std::vector<std::string> &configured_candidates = ig->storage_candidates();
    if (configured_candidates.empty()) {
        PREFIX_LOG(WARN, "data storage candidates is empty, instance group: %s", instance_group.c_str());
        result.ec = ErrorCode::EC_CONFIG_ERROR;
        return result;
    }

    // get the data storage backend preferring strategy configuration
    auto preference = CachePreferStrategy::CPS_UNSPECIFIED;
    const auto cache_config = ig->cache_config();
    if (cache_config == nullptr) {
        // preference remains CPS_UNSPECIFIED
        PREFIX_LOG(WARN,
                   "cache config is nullptr, use default cache prefer strategy, instance group: %s",
                   instance_group.c_str());
    } else {
        preference = cache_config->cache_prefer_strategy();
    }

    // start the selecting logic

    // 0. calculate the candidate list
    std::vector<std::shared_ptr<DataStorageBackend>> candidates;
    GetCandidates(request_context, avail_backends, configured_candidates, storage_quota_avail_array, candidates);

    // 1. select backend according to the specified preference
    const auto chosen_backend = Select(request_context, candidates, preference);

    if (chosen_backend == nullptr) {
        PREFIX_LOG(WARN, "unable to select a data storage backend, instance group: %s", instance_group.c_str());
        result.ec = ErrorCode::EC_NOENT;
        return result;
    }

    PREFIX_LOG(DEBUG,
               "the chosen data storage backend is: %s, type: %u, instance group: %s",
               chosen_backend->GetStorageConfig().global_unique_name().c_str(),
               static_cast<unsigned int>(chosen_backend->GetType()),
               instance_group.c_str());
    result.ec = ErrorCode::EC_OK;
    result.type = chosen_backend->GetType();
    result.name = chosen_backend->GetStorageConfig().global_unique_name();
    return result;
}

std::size_t DataStorageSelector::CalcGroupUsedSize(
    RequestContext const *request_context,
    const std::vector<std::shared_ptr<const InstanceInfo>> &instance_infos) const noexcept {
    const auto &trace_id = request_context->trace_id();

    std::size_t group_used_byte_size = 0;
    for (const auto &instance_info : instance_infos) {
        if (instance_info == nullptr) {
            PREFIX_LOG(WARN, "instance is nullptr");
            continue;
        }

        const std::string &ins_id = instance_info->instance_id();
        const auto meta_indexer = meta_indexer_manager_->GetMetaIndexer(ins_id);
        if (meta_indexer == nullptr) {
            PREFIX_LOG(WARN, "meta indexer is nullptr");
            continue;
        }

        meta_indexer->PersistMetaData();
        const std::size_t ins_used_key_cnt = meta_indexer->GetKeyCount();

        std::size_t byte_size_per_key = 0;
        for (auto &location_spec_info : instance_info->location_spec_infos()) {
            byte_size_per_key += location_spec_info.size();
        }
        const std::size_t ins_used_byte_size = byte_size_per_key * ins_used_key_cnt;

        group_used_byte_size += ins_used_byte_size;
    }

    return group_used_byte_size;
}

void DataStorageSelector::GenStorageQuotaAvailTable(
    RequestContext const *request_context,
    const InstanceGroupQuota &quota,
    const std::vector<std::shared_ptr<const InstanceInfo>> &instance_infos,
    std::array<bool, 5> &storage_quota_avail_array) const noexcept {
    const auto &trace_id = request_context->trace_id();

    for (const auto &storage_quota : quota.quota_config()) {
        auto type = storage_quota.storage_spec();
        if (type == DataStorageType::DATA_STORAGE_TYPE_VCNS_HF3FS) {
            // treat vcns_hf3fs as hf3fs
            type = DataStorageType::DATA_STORAGE_TYPE_HF3FS;
        }
        std::uint64_t total_sz = 0;

        for (const auto &ins : instance_infos) {
            if (ins == nullptr) {
                PREFIX_LOG(WARN, "instance is nullptr");
                continue;
            }
            const std::string &ins_id = ins->instance_id();
            const auto meta_indexer = meta_indexer_manager_->GetMetaIndexer(ins_id);
            if (meta_indexer == nullptr) {
                PREFIX_LOG(WARN, "meta indexer is nullptr");
                continue;
            }
            // TODO(rui): persist the storage stat data
            // meta_indexer->PersistMetaData();
            try {
                const std::uint64_t sz = meta_indexer->storage_usage_array_.at(static_cast<std::uint8_t>(type)).load();
                total_sz += sz;
            } catch (const std::out_of_range &e) {
                KVCM_LOG_WARN("data storage type out of range: %d", static_cast<std::uint8_t>(type));
            }
        }

        if (storage_quota.capacity() <= total_sz) {
            try {
                storage_quota_avail_array.at(static_cast<std::uint8_t>(type)) = false;
            } catch (const std::out_of_range &e) {
                KVCM_LOG_WARN("data storage type out of range: %d", static_cast<std::uint8_t>(type));
            }
        }
    }
}

} // namespace kv_cache_manager
