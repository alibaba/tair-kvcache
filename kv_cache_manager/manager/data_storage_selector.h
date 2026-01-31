#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "kv_cache_manager/common/error_code.h"
#include "kv_cache_manager/data_storage/storage_config.h"

namespace kv_cache_manager {

class DataStorageBackend;
class InstanceGroupQuota;
class InstanceInfo;
class MetaIndexerManager;
class RegistryManager;
class RequestContext;

/**
 * @brief Result structure containing the outcome of a data storage
 * selection operation
 *
 * This structure encapsulates the result of selecting a data storage
 * backend, including the error code of the operation, the type of the
 * selected storage, and its unique name.
 */
struct DataStorageSelectResult {
    ErrorCode ec;
    DataStorageType type;
    std::string name;
};

/**
 * @brief Manages the selection of appropriate data storage backends for
 * cache operations
 *
 * The DataStorageSelector is responsible for choosing the most
 * appropriate data storage backend based on instance group
 * configurations, available storage backends, and configured
 * preferences.  It implements a selection algorithm that considers:
 * 1. Available storage backends
 * 2. Configured candidate list for the instance group
 * 3. Storage preference strategies (e.g., prefer A, always B, etc.)
 * 4. Backend availability status
 *
 * The selector works by:
 * 1. Retrieving the list of currently available data storage backends
 * 2. Filtering the list based on the configured candidates
 * 3. Applying the instance group's storage preference strategy
 * 4. Returning the most appropriate backend based on these criteria
 *
 * @note This class is designed to be thread-safe as long as the
 * underlying RegistryManager is thread-safe.
 */
class DataStorageSelector {
public:
    /**
     * @brief Delete default constructor
     */
    DataStorageSelector() = delete;

    /**
     * @brief Construct a new DataStorageSelector object
     *
     * @param meta_indexer_manager Shared pointer to MetaIndexerManager
     * @param registry_manager Shared pointer to RegistryManager for
     * retrieving instance group configurations and available storage
     * backends
     */
    DataStorageSelector(std::shared_ptr<MetaIndexerManager> meta_indexer_manager,
                        std::shared_ptr<RegistryManager> registry_manager);

    /**
     * @brief Default copy constructor
     */
    DataStorageSelector(const DataStorageSelector &) = default;

    /**
     * @brief Default move constructor
     */
    DataStorageSelector(DataStorageSelector &&) = default;

    /**
     * @brief Default copy assignment operator
     */
    DataStorageSelector &operator=(const DataStorageSelector &) = default;

    /**
     * @brief Default move assignment operator
     */
    DataStorageSelector &operator=(DataStorageSelector &&) = default;

    /**
     * @brief Default destructor
     */
    ~DataStorageSelector() = default;

    /**
     * @brief Cleanup all inner data
     */
    void DoCleanup();

    /**
     * @brief Select an appropriate data storage backend for cache write
     * operations
     *
     * This method performs the core selection logic by:
     * 1. Checking for available data storage backends
     * 2. Retrieving the instance group configuration
     * 3. Filtering available backends by the candidate list
     * 4. Applying the configured preference strategy
     * 5. Returning the selected backend information
     *
     * @param request_context Shared pointer to the request context
     * @param instance_group Name of the instance group for which to
     * select a backend
     * @return DataStorageSelectResult containing the selection result:
     * - ec: ErrorCode indicating success or type of failure
     * - type: DataStorageType of the selected backend
     * - name: Unique name of the selected backend
     *
     * @retval EC_OK on successful selection
     * @retval EC_ERROR if registry manager is null
     * @retval EC_INSTANCE_NOT_EXIST if instance group does not exist
     * @retval EC_NOENT if no suitable backend found
     * @retval EC_CONFIG_ERROR if configuration is invalid
     */
    [[nodiscard]] DataStorageSelectResult
    SelectCacheWriteDataStorageBackend(RequestContext *request_context,
                                       const std::string &instance_group) const noexcept;

private:
    std::size_t
    CalcGroupUsedSize(RequestContext const *request_context,
                      const std::vector<std::shared_ptr<const InstanceInfo>> &instance_infos) const noexcept;
    void GenStorageQuotaAvailTable(RequestContext const *request_context,
                                   const InstanceGroupQuota &quota,
                                   const std::vector<std::shared_ptr<const InstanceInfo>> &instance_infos,
                                   std::array<bool, 5> &storage_quota_avail_array) const noexcept;
    std::shared_ptr<MetaIndexerManager> meta_indexer_manager_;
    std::shared_ptr<RegistryManager> registry_manager_;
};

} // namespace kv_cache_manager
