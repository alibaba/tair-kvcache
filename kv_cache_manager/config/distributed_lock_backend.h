#pragma once

#include <string>

#include "kv_cache_manager/common/error_code.h"

namespace kv_cache_manager {
class StandardUri;

/**
 * @brief Distributed lock backend interface
 *
 * Provides basic operations for distributed locks, supporting multiple backend implementations
 * (e.g., local filesystem, Redis, etc.). All interfaces are thread-safe, and implementations
 * should ensure atomicity of operations.
 */
class DistributedLockBackend {
public:
    virtual ~DistributedLockBackend() = default;

    /**
     * @brief Initialize the distributed lock backend
     *
     * @param standard_uri Standardized URI for configuring backend connection parameters
     *                     For local backend: file:///path/to/lock/dir
     *                     For Redis backend: redis://host:port/db
     * @return ErrorCode error code
     *         - EC_OK: Initialization successful
     *         - EC_BADARGS: Invalid arguments (e.g., unsupported URI protocol)
     *         - EC_IO_ERROR: IO error (e.g., cannot create directory or connection failed)
     *         - EC_ERROR: Other errors
     */
    virtual ErrorCode Init(const StandardUri &standard_uri) noexcept = 0;

    /**
     * @brief Attempt to acquire a distributed lock
     *
     * Attempts to acquire a lock for the specified key. If the lock is already held by
     * another client and not expired, acquisition fails. If the lock does not exist or
     * has expired, acquisition succeeds.
     *
     * @param lock_key The lock key used to uniquely identify a lock
     * @param lock_value The lock value used to identify the lock holder, typically a
     *                   client unique identifier
     * @param ttl_ms Time-to-live in milliseconds, the lock will automatically expire
     *               after the specified time
     * @return ErrorCode error code
     *         - EC_OK: Lock acquisition successful
     *         - EC_EXIST: Lock is already held by another client (and not expired)
     *         - EC_BADARGS: Invalid arguments (e.g., lock_key or lock_value is empty, or ttl_ms <= 0)
     *         - EC_ERROR: Other errors (e.g., backend connection failure)
     */
    virtual ErrorCode TryLock(const std::string &lock_key, const std::string &lock_value, int64_t ttl_ms) = 0;

    /**
     * @brief Renew a distributed lock
     *
     * Renews an already held lock, extending its expiration time. Only the current
     * holder of the lock can successfully renew it.
     *
     * @param lock_key The lock key
     * @param lock_value The lock value, must match the current lock's value
     * @param ttl_ms New time-to-live in milliseconds
     * @return ErrorCode error code
     *         - EC_OK: Renewal successful
     *         - EC_NOENT: Lock does not exist or has expired
     *         - EC_MISMATCH: Lock value mismatch (not the current holder)
     *         - EC_BADARGS: Invalid arguments
     *         - EC_ERROR: Other errors
     */
    virtual ErrorCode RenewLock(const std::string &lock_key, const std::string &lock_value, int64_t ttl_ms) = 0;

    /**
     * @brief Release a distributed lock
     *
     * Releases an already held lock. Only the current holder of the lock can
     * successfully release it.
     *
     * @param lock_key The lock key
     * @param lock_value The lock value, must match the current lock's value
     * @return ErrorCode error code
     *         - EC_OK: Release successful
     *         - EC_NOENT: Lock does not exist
     *         - EC_MISMATCH: Lock value mismatch (not the current holder)
     *         - EC_BADARGS: Invalid arguments
     *         - EC_ERROR: Other errors
     */
    virtual ErrorCode Unlock(const std::string &lock_key, const std::string &lock_value) = 0;

    /**
     * @brief Get current lock holder information
     *
     * Queries the current status of the specified lock, including holder identifier
     * and remaining expiration time. If the lock does not exist or has expired,
     * returns EC_NOENT.
     *
     * @param lock_key The lock key
     * @param out_current_value [out] Current lock value (holder identifier)
     * @param out_expire_time_ms [out] Lock expiration time (milliseconds timestamp)
     * @return ErrorCode error code
     *         - EC_OK: Query successful, lock exists and is not expired
     *         - EC_NOENT: Lock does not exist or has expired
     *         - EC_BADARGS: Invalid arguments
     *         - EC_ERROR: Other errors
     */
    virtual ErrorCode
    GetLockHolder(const std::string &lock_key, std::string &out_current_value, int64_t &out_expire_time_ms) = 0;
};

} // namespace kv_cache_manager