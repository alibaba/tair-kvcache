#include "kv_cache_manager/meta/meta_storage_backend_manager.h"

#include <algorithm>
#include <cassert>
#include <climits>
#include <exception>
#include <typeinfo>
#include <unordered_set>
#include <utility>

#include "kv_cache_manager/common/error_code.h"
#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/common/standard_uri.h"
#include "kv_cache_manager/common/timestamp_util.h"
#include "kv_cache_manager/config/meta_storage_backend_config.h"
#include "kv_cache_manager/meta/common.h"
#include "kv_cache_manager/meta/meta_local_backend.h"
#include "kv_cache_manager/meta/meta_storage_backend_factory.h"
#include "kv_cache_manager/metrics/metrics_collector.h"

namespace kv_cache_manager {

namespace {
constexpr int64_t kRecoverScanBatchSize = 1000;
constexpr int kRecoverMaxConsecutiveFailures = 3;

// Collects keys where results[i] == EC_NOENT. Returns {missing_keys, missing_indices}.
std::pair<KeyTypeVec, std::vector<size_t>> CollectMissingKeys(const KeyVector &keys,
                                                              const std::vector<ErrorCode> &results) {
    KeyTypeVec missing_keys;
    std::vector<size_t> missing_indices;
    // Callers validate the positional contract before recovery. Keep this
    // helper defensive as well so a newly added caller cannot index a short
    // backend response before its outer layer normalizes the failure.
    const size_t result_count = std::min(keys.size(), results.size());
    for (size_t i = 0; i < result_count; ++i) {
        if (results[i] == EC_NOENT) {
            missing_keys.push_back(keys[i]);
            missing_indices.push_back(i);
        }
    }
    return {std::move(missing_keys), std::move(missing_indices)};
}
} // namespace

MetaStorageBackendManager::~MetaStorageBackendManager() {
    // Close is idempotent and also owns recovery-thread shutdown, so an owner
    // that forgets the explicit lifecycle call cannot leak backend resources.
    (void)Close();
}

ErrorCode MetaStorageBackendManager::Init(const std::string &instance_id,
                                          const std::shared_ptr<MetaStorageBackendConfig> &config) noexcept {
    try {
        std::lock_guard<std::mutex> lifecycle_guard(lifecycle_mutex_);
        if (instance_id.empty()) {
            KVCM_LOG_ERROR("init meta storage backend manager failed, empty instance id");
            return EC_BADARGS;
        }
        if (!config) {
            KVCM_LOG_ERROR("init meta storage backend manager failed, null storage backend config");
            return EC_BADARGS;
        }
        if (persistent_backend_ || cache_backend_ || opened_) {
            KVCM_LOG_ERROR("meta storage backend manager is already initialized, instance[%s]", instance_id_.c_str());
            return EC_ERROR;
        }

        const std::string &storage_uri = config->GetStorageUri();
        if (config->GetStorageType() != META_CACHED_BACKEND_TYPE_STR) {
            // Single-backend mode: one backend serves every read/write directly.
            auto persistent_backend = MetaStorageBackendFactory::CreateAndInitStorageBackend(instance_id, config);
            if (!persistent_backend) {
                KVCM_LOG_ERROR("fail to create persistent backend uri[%s]", storage_uri.c_str());
                return EC_ERROR;
            }
            instance_id_ = instance_id;
            persistent_backend_ = std::move(persistent_backend);
            KVCM_LOG_INFO("meta storage backend manager init ok in single-backend mode, instance[%s] type[%s]",
                          instance_id_.c_str(),
                          config->GetStorageType().c_str());
            return EC_OK;
        }

        assert((config->GetStorageType() == META_CACHED_BACKEND_TYPE_STR));
        std::string persistent_type;
        std::string cache_type;
        if (!storage_uri.empty()) {
            StandardUri uri = StandardUri::FromUri(storage_uri);
            if (!uri.Valid()) {
                KVCM_LOG_ERROR("invalid storage uri[%s]", storage_uri.c_str());
                return EC_BADARGS;
            }
            persistent_type = uri.GetParam("persistent_type");
            cache_type = uri.GetParam("cache_type");
        }
        // default to redis / local
        persistent_type = persistent_type.empty() ? META_REDIS_BACKEND_TYPE_STR : persistent_type;
        cache_type = cache_type.empty() ? META_LOCAL_BACKEND_TYPE_STR : cache_type;

        auto persistent_config = std::make_shared<MetaStorageBackendConfig>(persistent_type);
        persistent_config->SetStorageUri(storage_uri);
        auto persistent_backend = MetaStorageBackendFactory::CreatePersistentBackend(instance_id, persistent_config);
        if (!persistent_backend) {
            KVCM_LOG_ERROR("fail to create persistent backend uri[%s]", storage_uri.c_str());
            return EC_ERROR;
        }
        auto cache_config = std::make_shared<MetaStorageBackendConfig>(cache_type);
        cache_config->SetStorageUri(storage_uri);
        auto cache_backend = MetaStorageBackendFactory::CreateCacheBackend(instance_id, cache_config);
        if (!cache_backend) {
            KVCM_LOG_ERROR("fail to create cache backend uri[%s]", storage_uri.c_str());
            return EC_ERROR;
        }

        // Publish the pair only after both factories have succeeded. A failed
        // dual-backend Init can therefore be retried on the same object and
        // never exposes a half-configured persistent side to Open().
        instance_id_ = instance_id;
        persistent_backend_ = std::move(persistent_backend);
        cache_backend_ = std::move(cache_backend);
        KVCM_LOG_INFO("meta storage backend manager init ok, instance[%s] cache[%s] persistent[%s]",
                      instance_id_.c_str(),
                      cache_type.c_str(),
                      persistent_type.c_str());
        return EC_OK;
    } catch (const std::exception &e) {
        KVCM_LOG_ERROR(
            "init meta storage backend manager raised, instance[%s] error[%s]", instance_id.c_str(), e.what());
    } catch (...) {
        KVCM_LOG_ERROR("init meta storage backend manager raised unknown exception, instance[%s]", instance_id.c_str());
    }
    return EC_ERROR;
}

ErrorCode MetaStorageBackendManager::Open() noexcept {
    std::lock_guard<std::mutex> lifecycle_guard(lifecycle_mutex_);
    if (opened_) {
        KVCM_LOG_ERROR("meta storage backend manager is already open, instance[%s]", instance_id_.c_str());
        return EC_ERROR;
    }
    if (!persistent_backend_) {
        KVCM_LOG_ERROR("persistent backend not inited! instance[%s]", instance_id_.c_str());
        return EC_ERROR;
    }

    ErrorCode ec = persistent_backend_->Open();
    if (ec != EC_OK) {
        // Open implementations may have allocated resources before reporting
        // failure. Roll them back even though the manager never publishes an
        // opened state.
        is_closed_.store(true, std::memory_order_release);
        const ErrorCode close_ec = persistent_backend_->Close();
        KVCM_LOG_ERROR(
            "open persistent failed, instance[%s] ec[%d] rollback_close_ec[%d]", instance_id_.c_str(), ec, close_ec);
        return ec;
    }
    is_closed_.store(false, std::memory_order_release);

    if (!cache_backend_) {
        recover_state_.store(RecoverState::kRunning, std::memory_order_release);
        opened_ = true;
        KVCM_LOG_INFO("meta storage backend manager opened in single-backend mode, instance[%s]", instance_id_.c_str());
        return EC_OK;
    }

    ec = cache_backend_->Open();
    if (ec != EC_OK) {
        KVCM_LOG_ERROR("open cache failed, instance[%s] ec[%d]", instance_id_.c_str(), ec);
        // Open is transactional from the manager's point of view. Do not
        // leave the persistent side live when the cache side cannot serve.
        // Also close the cache defensively in case its Open partially
        // initialized resources before returning an error.
        is_closed_.store(true, std::memory_order_release);
        ErrorCode cache_close_ec = cache_backend_->Close();
        ErrorCode persistent_close_ec = persistent_backend_->Close();
        if (cache_close_ec != EC_OK || persistent_close_ec != EC_OK) {
            KVCM_LOG_ERROR("rollback after cache open failure was incomplete, instance[%s] "
                           "cache_close_ec[%d] persistent_close_ec[%d]",
                           instance_id_.c_str(),
                           cache_close_ec,
                           persistent_close_ec);
        }
        return ec;
    }
    recover_state_.store(RecoverState::kRecover, std::memory_order_release);
    try {
        recover_thread_ = std::thread(&MetaStorageBackendManager::AsyncRecoverTask, this);
    } catch (const std::exception &e) {
        // Open is noexcept, so thread construction must not escape. Roll both
        // backends back to a closed state instead of publishing a manager that
        // can never finish recovery.
        is_closed_.store(true, std::memory_order_release);
        ErrorCode cache_close_ec = cache_backend_->Close();
        ErrorCode persistent_close_ec = persistent_backend_->Close();
        KVCM_LOG_ERROR("start async recover failed, instance[%s] error[%s] "
                       "cache_close_ec[%d] persistent_close_ec[%d]",
                       instance_id_.c_str(),
                       e.what(),
                       cache_close_ec,
                       persistent_close_ec);
        return EC_ERROR;
    } catch (...) {
        is_closed_.store(true, std::memory_order_release);
        ErrorCode cache_close_ec = cache_backend_->Close();
        ErrorCode persistent_close_ec = persistent_backend_->Close();
        KVCM_LOG_ERROR("start async recover failed with unknown exception, instance[%s] "
                       "cache_close_ec[%d] persistent_close_ec[%d]",
                       instance_id_.c_str(),
                       cache_close_ec,
                       persistent_close_ec);
        return EC_ERROR;
    }
    opened_ = true;
    KVCM_LOG_INFO("meta storage backend manager opened, instance[%s], async recover started", instance_id_.c_str());
    return EC_OK;
}

ErrorCode MetaStorageBackendManager::Close() noexcept {
    std::lock_guard<std::mutex> lifecycle_guard(lifecycle_mutex_);
    is_closed_.store(true, std::memory_order_release);
    if (recover_thread_.joinable()) {
        recover_thread_.join();
    }
    if (!opened_) {
        return EC_OK;
    }

    ErrorCode cache_ec = EC_OK;
    ErrorCode persistent_ec = EC_OK;
    if (cache_backend_) {
        cache_ec = cache_backend_->Close();
    }
    if (persistent_backend_) {
        persistent_ec = persistent_backend_->Close();
    }
    if (cache_ec != EC_OK) {
        KVCM_LOG_ERROR("close cache failed, instance[%s] ec[%d]", instance_id_.c_str(), cache_ec);
        return cache_ec;
    }
    if (persistent_ec != EC_OK) {
        KVCM_LOG_ERROR("close persistent failed, instance[%s] ec[%d]", instance_id_.c_str(), persistent_ec);
        return persistent_ec;
    }
    opened_ = false;
    KVCM_LOG_INFO("meta storage backend manager closed, instance[%s]", instance_id_.c_str());
    return EC_OK;
}

void MetaStorageBackendManager::AsyncRecoverTask() noexcept {
    KVCM_LOG_INFO("meta storage backend manager async recover started, instance[%s]", instance_id_.c_str());
    std::string cursor = SCAN_BASE_CURSOR;
    int64_t total_backfilled_keys = 0;
    int consecutive_failures = 0;
    std::string next_cursor;
    KeyTypeVec scanned_keys;
    FieldMapVec field_maps;
    bool has_pending_batch = false;
    bool recovery_complete = false;
    while (!recovery_complete) {
        if (is_closed_.load(std::memory_order_acquire)) {
            KVCM_LOG_INFO("async recover aborted due to close, instance[%s]", instance_id_.c_str());
            return;
        }

        if (!has_pending_batch) {
            scanned_keys.clear();
            field_maps.clear();
            ErrorCode scan_ec =
                persistent_backend_->ListKeys(nullptr, cursor, kRecoverScanBatchSize, next_cursor, scanned_keys);
            if (scan_ec != EC_OK) {
                ++consecutive_failures;
                KVCM_LOG_ERROR("async recover scan failed, instance[%s] cursor[%s] ec[%d] attempt[%d/%d]",
                               instance_id_.c_str(),
                               cursor.c_str(),
                               scan_ec,
                               consecutive_failures,
                               kRecoverMaxConsecutiveFailures);
                if (consecutive_failures >= kRecoverMaxConsecutiveFailures) {
                    KVCM_LOG_ERROR("async recover giving up after %d consecutive scan failures, "
                                   "leaving backend in Recover, instance[%s]",
                                   kRecoverMaxConsecutiveFailures,
                                   instance_id_.c_str());
                    break;
                }
                continue;
            }

            if (scanned_keys.empty()) {
                consecutive_failures = 0;
                cursor = next_cursor;
                recovery_complete = (cursor == SCAN_BASE_CURSOR);
                continue;
            }
            // Redis SCAN is not a stable snapshot.  Once a cursor yields a
            // batch, retain those exact keys across Get/backfill retries;
            // rescanning the cursor could return a different set and let the
            // failed keys disappear before recovery is published complete.
            has_pending_batch = true;
        }
        CacheLocationMapVector locations;
        PropertyMapVector properties;
        std::vector<ErrorCode> get_error_codes = persistent_backend_->Get(nullptr, scanned_keys, locations, properties);
        if (get_error_codes.size() != scanned_keys.size() || locations.size() != scanned_keys.size() ||
            properties.size() != scanned_keys.size()) {
            KVCM_LOG_ERROR("async recover Get results[%lu] locations[%lu] properties[%lu] mismatch keys[%lu], "
                           "skip batch",
                           get_error_codes.size(),
                           locations.size(),
                           properties.size(),
                           scanned_keys.size());
            ++consecutive_failures;
            if (consecutive_failures >= kRecoverMaxConsecutiveFailures) {
                KVCM_LOG_ERROR("async recover giving up after %d malformed Get responses, "
                               "leaving backend in Recover, instance[%s]",
                               kRecoverMaxConsecutiveFailures,
                               instance_id_.c_str());
                break;
            }
            continue;
        }
        bool get_complete = true;
        for (size_t i = 0; i < scanned_keys.size(); ++i) {
            if (get_error_codes[i] != EC_OK && get_error_codes[i] != EC_NOENT) {
                KVCM_LOG_WARN("async recover key[%ld] get failed ec[%d]", scanned_keys[i], get_error_codes[i]);
                get_complete = false;
            }
        }
        if (!get_complete) {
            ++consecutive_failures;
            if (consecutive_failures >= kRecoverMaxConsecutiveFailures) {
                KVCM_LOG_ERROR("async recover giving up after %d incomplete Get responses, "
                               "leaving backend in Recover, instance[%s]",
                               kRecoverMaxConsecutiveFailures,
                               instance_id_.c_str());
                break;
            }
            continue;
        }

        bool backfill_success = false;
        const int64_t backfilled_keys =
            BackfillKeysToCache(scanned_keys, locations, properties, get_error_codes, &backfill_success);
        if (!backfill_success) {
            ++consecutive_failures;
            KVCM_LOG_ERROR("async recover backfill failed, instance[%s] cursor[%s] attempt[%d/%d]",
                           instance_id_.c_str(),
                           cursor.c_str(),
                           consecutive_failures,
                           kRecoverMaxConsecutiveFailures);
            if (consecutive_failures >= kRecoverMaxConsecutiveFailures) {
                KVCM_LOG_ERROR("async recover giving up after %d consecutive backfill failures, "
                               "leaving backend in Recover, instance[%s]",
                               kRecoverMaxConsecutiveFailures,
                               instance_id_.c_str());
                break;
            }
            continue;
        }

        total_backfilled_keys += backfilled_keys;
        consecutive_failures = 0;
        cursor = next_cursor;
        recovery_complete = (cursor == SCAN_BASE_CURSOR);
        has_pending_batch = false;
    }

    if (!recovery_complete) {
        // Recover mode is deliberately retained. Reads can still fall back to
        // persistent storage, and delete tombstones must remain live so a
        // partial cache cannot resurrect keys through a late backfill.
        KVCM_LOG_ERROR("async recover incomplete, instance[%s] total_backfilled_keys[%ld], "
                       "backend remains in Recover",
                       instance_id_.c_str(),
                       total_backfilled_keys);
        return;
    }

    KVCM_LOG_INFO(
        "async recover completed instance[%s] total_backfilled_keys[%ld]", instance_id_.c_str(), total_backfilled_keys);
    recover_state_.store(RecoverState::kRunning, std::memory_order_release);
    {
        std::lock_guard<std::mutex> lock(deleted_keys_mutex_);
        deleted_keys_.clear();
    }
}

bool MetaStorageBackendManager::EnsureKeyInCache(RequestContext *request_context, const KeyTypeVec &keys) noexcept {
    if (keys.empty()) {
        return true;
    }
    std::vector<bool> exists_vec;
    std::vector<ErrorCode> exists_results = cache_backend_->Exists(request_context, keys, exists_vec);
    if (exists_results.size() != keys.size() || exists_vec.size() != keys.size()) {
        KVCM_LOG_ERROR("ensure cache Exists results[%lu] values[%lu] mismatch keys[%lu]",
                       exists_results.size(),
                       exists_vec.size(),
                       keys.size());
        return false;
    }
    KeyTypeVec missing_keys;
    for (size_t i = 0; i < keys.size(); ++i) {
        if (exists_results[i] != EC_OK || !exists_vec[i]) {
            missing_keys.emplace_back(keys[i]);
        }
    }
    if (missing_keys.empty()) {
        return true;
    }

    CacheLocationMapVector locations;
    PropertyMapVector properties;
    std::vector<ErrorCode> get_results = persistent_backend_->Get(request_context, missing_keys, locations, properties);
    if (get_results.size() != missing_keys.size() || locations.size() != missing_keys.size() ||
        properties.size() != missing_keys.size()) {
        KVCM_LOG_ERROR("ensure cache Get results[%lu] locations[%lu] properties[%lu] mismatch keys[%lu]",
                       get_results.size(),
                       locations.size(),
                       properties.size(),
                       missing_keys.size());
        return false;
    }

    // PutIfAbsent prevents a stale persistent read from overwriting a newer
    // dual-write that populated the cache after the Exists probe.
    std::vector<ErrorCode> put_results =
        cache_backend_->PutIfAbsent(request_context, missing_keys, locations, properties, get_results);
    if (put_results.size() != missing_keys.size()) {
        KVCM_LOG_ERROR(
            "ensure cache PutIfAbsent results[%lu] mismatch keys[%lu]", put_results.size(), missing_keys.size());
        return false;
    }
    bool hydrated = true;
    for (size_t i = 0; i < missing_keys.size(); ++i) {
        if (get_results[i] == EC_NOENT) {
            // The key is genuinely new. The following Upsert can safely
            // create it from the request's fields.
            continue;
        }
        if (get_results[i] != EC_OK || (put_results[i] != EC_OK && put_results[i] != EC_EXIST)) {
            KVCM_LOG_WARN("ensure key[%ld] in cache failed, get_ec[%d] put_ec[%d]",
                          missing_keys[i],
                          get_results[i],
                          put_results[i]);
            hydrated = false;
        }
    }
    return hydrated;
}

int64_t MetaStorageBackendManager::BackfillKeysToCache(const KeyTypeVec &keys,
                                                       const CacheLocationMapVector &locations,
                                                       const PropertyMapVector &properties,
                                                       const std::vector<ErrorCode> &get_error_codes,
                                                       bool *out_success) noexcept {
    if (out_success) {
        *out_success = false;
    }
    std::lock_guard<std::mutex> lock(deleted_keys_mutex_);

    // Merge get errors and deleted-key tombstones into a single error vector.
    if (get_error_codes.size() != keys.size() || locations.size() != keys.size() || properties.size() != keys.size()) {
        KVCM_LOG_ERROR("backfill results[%lu] locations[%lu] properties[%lu] mismatch keys[%lu], skip batch",
                       get_error_codes.size(),
                       locations.size(),
                       properties.size(),
                       keys.size());
        return 0;
    }
    int64_t valid_count = 0;
    std::vector<ErrorCode> merged_error_codes = get_error_codes;
    for (size_t i = 0; i < keys.size(); ++i) {
        if (merged_error_codes[i] != EC_OK && merged_error_codes[i] != EC_NOENT) {
            KVCM_LOG_ERROR("backfill source read failed key[%ld] ec[%d], skip batch", keys[i], merged_error_codes[i]);
            return 0;
        }
        if (merged_error_codes[i] == EC_OK && deleted_keys_.count(keys[i]) > 0) {
            merged_error_codes[i] = EC_NOENT;
        }
        valid_count += (merged_error_codes[i] == EC_OK);
    }
    if (valid_count == 0) {
        if (out_success) {
            *out_success = true;
        }
        return 0;
    }

    std::vector<ErrorCode> put_results =
        cache_backend_->PutIfAbsent(nullptr, keys, locations, properties, merged_error_codes);
    if (put_results.size() != keys.size()) {
        KVCM_LOG_ERROR(
            "backfill PutIfAbsent results[%lu] mismatch keys[%lu], skip accounting", put_results.size(), keys.size());
        return 0;
    }
    int64_t backfilled_count = 0;
    bool write_complete = true;
    for (size_t i = 0; i < keys.size(); ++i) {
        if (merged_error_codes[i] == EC_OK) {
            if (put_results[i] == EC_OK) {
                ++backfilled_count;
            } else if (put_results[i] != EC_EXIST) {
                KVCM_LOG_WARN("backfill PutIfAbsent failed key[%ld] ec[%d]", keys[i], put_results[i]);
                write_complete = false;
            }
        } else if (put_results[i] != merged_error_codes[i]) {
            // Conditional cache writes must preserve the source error for
            // skipped keys. EC_OK here could mean a tombstoned value was
            // inserted despite the guard, so never publish this cache.
            KVCM_LOG_WARN("backfill PutIfAbsent violated skip contract key[%ld] source_ec[%d] put_ec[%d]",
                          keys[i],
                          merged_error_codes[i],
                          put_results[i]);
            write_complete = false;
        }
    }
    if (out_success) {
        *out_success = write_complete;
    }
    return backfilled_count;
}

std::vector<ErrorCode> MetaStorageBackendManager::Put(RequestContext *request_context, BatchMetaData &batch) noexcept {
    const KeyVector &keys = batch.batch_keys;
    batch.EnsureLocationsAndPropertiesResized();
    CacheLocationMapVector &locations = batch.batch_locations;
    PropertyMapVector &properties = batch.batch_properties;
    std::vector<ErrorCode> persistent_results = persistent_backend_->Put(request_context, keys, locations, properties);
    if (persistent_results.size() != keys.size()) {
        KVCM_LOG_ERROR("persistent Put results[%lu] mismatch keys[%lu]", persistent_results.size(), keys.size());
        return std::vector<ErrorCode>(keys.size(), EC_ERROR);
    }
    if (!cache_backend_) {
        return persistent_results;
    }
    const int64_t cache_begin = TimestampUtil::GetCurrentTimeUs();
    auto results = cache_backend_->Put(request_context, keys, locations, properties, persistent_results);
    if (request_context) {
        auto *mc = dynamic_cast<ServiceMetricsCollector *>(request_context->metrics_collector());
        KVCM_METRICS_COLLECTOR_SET_METRICS(
            mc, meta_indexer, cache_backend_put_time_us, TimestampUtil::GetCurrentTimeUs() - cache_begin);
    }
    if (results.size() != keys.size()) {
        KVCM_LOG_ERROR("cache Put results[%lu] mismatch keys[%lu]", results.size(), keys.size());
        return std::vector<ErrorCode>(keys.size(), EC_ERROR);
    }
    return results;
}

std::vector<ErrorCode> MetaStorageBackendManager::Upsert(RequestContext *request_context,
                                                         BatchMetaData &batch,
                                                         MetaAccessIntent intent) noexcept {
    const KeyVector &keys = batch.batch_keys;
    batch.EnsureLocationsAndPropertiesResized();
    CacheLocationMapVector &locations = batch.batch_locations;
    PropertyMapVector &properties = batch.batch_properties;

    // Upsert may touch only a subset of fields, so Recover-time hydration
    // is needed to avoid overwriting unmentioned fields with empty values.
    if (cache_backend_ && recover_state_.load(std::memory_order_acquire) == RecoverState::kRecover) {
        if (!EnsureKeyInCache(request_context, keys)) {
            return std::vector<ErrorCode>(keys.size(), EC_ERROR);
        }
    }
    std::vector<ErrorCode> persistent_results =
        persistent_backend_->Upsert(request_context, keys, locations, properties, intent);
    if (persistent_results.size() != keys.size()) {
        KVCM_LOG_ERROR("persistent Upsert results[%lu] mismatch keys[%lu]", persistent_results.size(), keys.size());
        return std::vector<ErrorCode>(keys.size(), EC_ERROR);
    }
    if (!cache_backend_) {
        return persistent_results;
    }
    const int64_t cache_begin = TimestampUtil::GetCurrentTimeUs();
    auto results =
        cache_backend_->Upsert(request_context, keys, locations, properties, persistent_results, intent);
    if (request_context) {
        auto *mc = dynamic_cast<ServiceMetricsCollector *>(request_context->metrics_collector());
        KVCM_METRICS_COLLECTOR_SET_METRICS(
            mc, meta_indexer, cache_backend_upsert_time_us, TimestampUtil::GetCurrentTimeUs() - cache_begin);
    }
    if (results.size() != keys.size()) {
        KVCM_LOG_ERROR("cache Upsert results[%lu] mismatch keys[%lu]", results.size(), keys.size());
        return std::vector<ErrorCode>(keys.size(), EC_ERROR);
    }
    return results;
}

std::vector<ErrorCode> MetaStorageBackendManager::UpsertSingleLocations(RequestContext *request_context,
                                                                        const KeyVector &keys,
                                                                        const LocationIdRefVector &location_ids,
                                                                        const CacheLocationVector &locations) noexcept {
    if (!SupportsSingleLocationRmw()) {
        KVCM_LOG_ERROR("single-location upsert requires a pure local metadata backend");
        return std::vector<ErrorCode>(keys.size(), EC_UNIMPLEMENTED);
    }
    return persistent_backend_->UpsertSingleLocations(request_context, keys, location_ids, locations);
}

void MetaStorageBackendManager::PrepareSingleLocationRmwScratch(size_t max_count,
                                                                SingleLocationRmwScratch &scratch) noexcept {
    if (!SupportsSingleLocationRmw()) {
        return;
    }
    static_cast<MetaLocalBackend *>(persistent_backend_.get())->PrepareSingleLocationRmwScratch(max_count, scratch);
}

void MetaStorageBackendManager::UpsertSingleLocationsInto(RequestContext *request_context,
                                                          const KeyVector &keys,
                                                          const LocationIdRefVector &location_ids,
                                                          const CacheLocationVector &locations,
                                                          std::vector<ErrorCode> &out_results,
                                                          SingleLocationRmwScratch &scratch) noexcept {
    if (!SupportsSingleLocationRmw()) {
        KVCM_LOG_ERROR("single-location upsert requires a pure local metadata backend");
        out_results.assign(keys.size(), EC_UNIMPLEMENTED);
        return;
    }
    static_cast<MetaLocalBackend *>(persistent_backend_.get())
        ->UpsertSingleLocationsInto(request_context, keys, location_ids, locations, out_results, scratch);
}

void MetaStorageBackendManager::UpsertSingleLocationsUsingRetainedHandlesInto(
    RequestContext *request_context,
    const KeyVector &keys,
    const LocationIdRefVector &location_ids,
    CacheLocationVector &locations,
    const std::vector<size_t> &read_indices,
    std::vector<ErrorCode> &out_results,
    SingleLocationRmwScratch &scratch) noexcept {
    if (!SupportsSingleLocationRmw()) {
        KVCM_LOG_ERROR("retained-handle single-location upsert requires a pure local metadata backend");
        out_results.assign(keys.size(), EC_UNIMPLEMENTED);
        scratch.ReleaseRetainedHandles();
        return;
    }
    static_cast<MetaLocalBackend *>(persistent_backend_.get())
        ->UpsertSingleLocationsUsingRetainedHandlesInto(
            request_context, keys, location_ids, locations, read_indices, out_results, scratch);
}

std::vector<ErrorCode> MetaStorageBackendManager::Delete(RequestContext *request_context,
                                                         const KeyVector &keys) noexcept {
    std::vector<ErrorCode> persistent_results = persistent_backend_->Delete(request_context, keys);
    if (persistent_results.size() != keys.size()) {
        KVCM_LOG_ERROR("persistent Delete results[%lu] mismatch keys[%lu]", persistent_results.size(), keys.size());
        return std::vector<ErrorCode>(keys.size(), EC_ERROR);
    }
    if (!cache_backend_) {
        return persistent_results;
    }
    if (recover_state_.load(std::memory_order_acquire) == RecoverState::kRecover) {
        // Tombstone to prevent Recover backfill from resurrecting deleted keys.
        std::lock_guard<std::mutex> lock(deleted_keys_mutex_);
        for (size_t i = 0; i < keys.size(); ++i) {
            if (persistent_results[i] == EC_OK || persistent_results[i] == EC_NOENT) {
                deleted_keys_.insert(keys[i]);
            }
        }
    }
    const int64_t cache_begin = TimestampUtil::GetCurrentTimeUs();
    auto results = cache_backend_->Delete(request_context, keys, persistent_results);
    if (request_context) {
        auto *mc = dynamic_cast<ServiceMetricsCollector *>(request_context->metrics_collector());
        KVCM_METRICS_COLLECTOR_SET_METRICS(
            mc, meta_indexer, cache_backend_delete_time_us, TimestampUtil::GetCurrentTimeUs() - cache_begin);
    }
    if (results.size() != keys.size()) {
        KVCM_LOG_ERROR("cache Delete results[%lu] mismatch keys[%lu]", results.size(), keys.size());
        return std::vector<ErrorCode>(keys.size(), EC_ERROR);
    }
    return results;
}

std::vector<ErrorCode> MetaStorageBackendManager::Delete(RequestContext *request_context,
                                                         const KeyVector &keys,
                                                         const LocationIdsPerKey &location_ids,
                                                         int32_t &out_reclaimed_count) noexcept {
    out_reclaimed_count = 0;
    if (keys.empty()) {
        return {};
    }
    if (location_ids.size() != keys.size()) {
        return std::vector<ErrorCode>(keys.size(), EC_BADARGS);
    }

    // Partial-delete during Recover: hydrate cache from persistent first so
    // the conditional mirror write below has the full pre-restart field set
    // to delete against (and async backfill cannot later overwrite us).
    if (cache_backend_ && recover_state_.load(std::memory_order_acquire) == RecoverState::kRecover) {
        if (!EnsureKeyInCache(request_context, keys)) {
            return std::vector<ErrorCode>(keys.size(), EC_ERROR);
        }
    }

    std::vector<ErrorCode> persistent_results =
        persistent_backend_->DeleteLocations(request_context, keys, location_ids);
    if (persistent_results.size() != keys.size()) {
        KVCM_LOG_ERROR(
            "persistent DeleteLocations results[%lu] mismatch keys[%lu]", persistent_results.size(), keys.size());
        return std::vector<ErrorCode>(keys.size(), EC_ERROR);
    }
    std::vector<ErrorCode> results;
    if (!cache_backend_) {
        results = std::move(persistent_results);
    } else {
        const int64_t cache_begin = TimestampUtil::GetCurrentTimeUs();
        results = cache_backend_->DeleteLocations(request_context, keys, location_ids, persistent_results);
        if (request_context) {
            auto *mc = dynamic_cast<ServiceMetricsCollector *>(request_context->metrics_collector());
            KVCM_METRICS_COLLECTOR_SET_METRICS(
                mc, meta_indexer, cache_backend_delete_time_us, TimestampUtil::GetCurrentTimeUs() - cache_begin);
        }
    }
    if (results.size() != keys.size()) {
        KVCM_LOG_ERROR("cache DeleteLocations results[%lu] mismatch keys[%lu]", results.size(), keys.size());
        return std::vector<ErrorCode>(keys.size(), EC_ERROR);
    }

    out_reclaimed_count = MaybeReclaimEmptyKeys(request_context, keys, results);
    return results;
}

std::vector<ErrorCode>
MetaStorageBackendManager::DeleteLocationsForMaintenance(RequestContext *request_context,
                                                         const KeyVector &keys,
                                                         const LocationIdsPerKey &location_ids,
                                                         int32_t &out_reclaimed_count) noexcept {
    out_reclaimed_count = 0;
    if (keys.empty()) {
        return {};
    }
    if (keys.size() != location_ids.size()) {
        return std::vector<ErrorCode>(keys.size(), EC_BADARGS);
    }
    if (cache_backend_ && recover_state_.load(std::memory_order_acquire) == RecoverState::kRecover) {
        // A recover task can still publish an older persistent snapshot into
        // the hot cache. Retry this best-effort action after recovery rather
        // than backfilling or touching cache state from maintenance.
        return std::vector<ErrorCode>(keys.size(), EC_OUT_OF_LIMIT);
    }

    // Decide target-only versus whole-key deletion from both layers before
    // mutating either one. Looking only at the hot cache can erase a newer
    // persistent sibling; deleting targets first and reclaiming the key later
    // can also hide a failed whole-key delete from subsequent GC rounds.
    LocationIdsPerKey persistent_location_ids;
    const std::vector<ErrorCode> persistent_id_ecs =
        persistent_backend_->GetLocationIdsForMaintenance(request_context, keys, persistent_location_ids);
    LocationIdsPerKey hot_location_ids;
    std::vector<ErrorCode> hot_id_ecs;
    if (cache_backend_) {
        hot_id_ecs = cache_backend_->GetLocationIdsForMaintenance(request_context, keys, hot_location_ids);
    }

    std::vector<ErrorCode> results(keys.size(), EC_OK);
    std::vector<bool> id_reads_valid(keys.size(), true);
    const auto validate_id_reads =
        [&](const char *layer, const std::vector<ErrorCode> &id_ecs, const LocationIdsPerKey &ids) {
            if (id_ecs.size() != keys.size() || ids.size() != keys.size()) {
                KVCM_LOG_ERROR("%s maintenance location-id results[%lu] values[%lu] mismatch keys[%lu]",
                               layer,
                               id_ecs.size(),
                               ids.size(),
                               keys.size());
                std::fill(results.begin(), results.end(), EC_ERROR);
                std::fill(id_reads_valid.begin(), id_reads_valid.end(), false);
                return;
            }
            for (size_t i = 0; i < keys.size(); ++i) {
                if (id_ecs[i] != EC_OK && id_ecs[i] != EC_NOENT) {
                    results[i] = id_ecs[i];
                    id_reads_valid[i] = false;
                } else if (id_ecs[i] == EC_NOENT && !ids[i].empty()) {
                    KVCM_LOG_ERROR(
                        "%s maintenance location-id read returned NOENT with values for key[%ld]", layer, keys[i]);
                    results[i] = EC_ERROR;
                    id_reads_valid[i] = false;
                }
            }
        };
    validate_id_reads("persistent", persistent_id_ecs, persistent_location_ids);
    if (cache_backend_) {
        validate_id_reads("cache", hot_id_ecs, hot_location_ids);
    }

    std::vector<size_t> whole_key_indexes;
    std::vector<size_t> target_only_indexes;
    for (size_t i = 0; i < keys.size(); ++i) {
        if (!id_reads_valid[i] || location_ids[i].empty()) {
            continue;
        }
        const auto layer_has_only_targets = [&](ErrorCode id_ec, const LocationIdVector &existing_ids) {
            if (id_ec == EC_NOENT) {
                return true;
            }
            return std::all_of(existing_ids.begin(), existing_ids.end(), [&](const LocationId &existing_id) {
                return std::find(location_ids[i].begin(), location_ids[i].end(), existing_id) != location_ids[i].end();
            });
        };
        const bool persistent_safe = layer_has_only_targets(persistent_id_ecs[i], persistent_location_ids[i]);
        const bool hot_safe = !cache_backend_ || layer_has_only_targets(hot_id_ecs[i], hot_location_ids[i]);
        (persistent_safe && hot_safe ? whole_key_indexes : target_only_indexes).push_back(i);
    }

    if (!target_only_indexes.empty()) {
        KeyVector target_keys;
        LocationIdsPerKey target_location_ids;
        target_keys.reserve(target_only_indexes.size());
        target_location_ids.reserve(target_only_indexes.size());
        for (const size_t index : target_only_indexes) {
            target_keys.push_back(keys[index]);
            target_location_ids.push_back(location_ids[index]);
        }

        std::vector<ErrorCode> persistent_results =
            persistent_backend_->DeleteLocationsForMaintenance(request_context, target_keys, target_location_ids);
        if (persistent_results.size() != target_only_indexes.size()) {
            KVCM_LOG_ERROR("persistent maintenance target delete results[%lu] mismatch keys[%lu]",
                           persistent_results.size(),
                           target_only_indexes.size());
            persistent_results.assign(target_only_indexes.size(), EC_ERROR);
        }
        std::vector<ErrorCode> target_results = persistent_results;
        if (cache_backend_) {
            target_results = cache_backend_->DeleteLocationsForMaintenance(
                request_context, target_keys, target_location_ids, persistent_results);
            if (target_results.size() != target_only_indexes.size()) {
                KVCM_LOG_ERROR("cache maintenance target delete results[%lu] mismatch keys[%lu]",
                               target_results.size(),
                               target_only_indexes.size());
                target_results.assign(target_only_indexes.size(), EC_ERROR);
            }
        }
        for (size_t i = 0; i < target_only_indexes.size(); ++i) {
            const ErrorCode persistent_ec = persistent_results[i];
            const ErrorCode hot_ec = target_results[i];
            if (persistent_ec != EC_OK && persistent_ec != EC_NOENT) {
                results[target_only_indexes[i]] = persistent_ec;
            } else if (hot_ec != EC_OK && hot_ec != EC_NOENT) {
                results[target_only_indexes[i]] = hot_ec;
            } else if (persistent_ec == EC_OK || hot_ec == EC_OK) {
                // Either layer actually removed the logical target. Report
                // success so storage usage is accounted exactly once even if
                // the other layer had already converged to NOENT.
                results[target_only_indexes[i]] = EC_OK;
            } else {
                results[target_only_indexes[i]] = EC_NOENT;
            }
        }
    }

    if (!whole_key_indexes.empty()) {
        KeyVector whole_keys;
        whole_keys.reserve(whole_key_indexes.size());
        for (const size_t index : whole_key_indexes) {
            whole_keys.push_back(keys[index]);
        }

        std::vector<ErrorCode> persistent_results = persistent_backend_->Delete(request_context, whole_keys);
        if (persistent_results.size() != whole_key_indexes.size()) {
            KVCM_LOG_ERROR("persistent maintenance whole-key delete results[%lu] mismatch keys[%lu]",
                           persistent_results.size(),
                           whole_key_indexes.size());
            persistent_results.assign(whole_key_indexes.size(), EC_ERROR);
        }
        std::vector<ErrorCode> whole_results = persistent_results;
        if (cache_backend_) {
            // A key already absent from persistent still needs its hot copy
            // removed. Normalize only this idempotent maintenance gate.
            auto cache_gate = persistent_results;
            for (auto &ec : cache_gate) {
                if (ec == EC_NOENT) {
                    ec = EC_OK;
                }
            }
            whole_results = cache_backend_->Delete(request_context, whole_keys, cache_gate);
            if (whole_results.size() != whole_key_indexes.size()) {
                KVCM_LOG_ERROR("cache maintenance whole-key delete results[%lu] mismatch keys[%lu]",
                               whole_results.size(),
                               whole_key_indexes.size());
                whole_results.assign(whole_key_indexes.size(), EC_ERROR);
            }
        }

        std::unordered_set<KeyType> reclaimed_keys;
        for (size_t i = 0; i < whole_key_indexes.size(); ++i) {
            const size_t original_index = whole_key_indexes[i];
            if (whole_results[i] != EC_OK && whole_results[i] != EC_NOENT) {
                results[original_index] = whole_results[i];
                continue;
            }
            results[original_index] = EC_OK;
            // The preceding expected-value read admitted at least one target.
            // Count a converged whole-key action once even when one layer was
            // already missing due to an earlier partial attempt.
            reclaimed_keys.insert(keys[original_index]);
        }
        out_reclaimed_count = static_cast<int32_t>(reclaimed_keys.size());
    }
    return results;
}


int32_t MetaStorageBackendManager::MaybeReclaimEmptyKeys(RequestContext *request_context,
                                                         const KeyVector &keys,
                                                         const std::vector<ErrorCode> &delete_results) noexcept {
    if (delete_results.size() != keys.size()) {
        KVCM_LOG_ERROR(
            "delete results[%lu] mismatch keys[%lu], skip empty-key reclamation", delete_results.size(), keys.size());
        return 0;
    }

    KeyVector candidate_keys;
    std::unordered_set<KeyType> seen_candidates;
    candidate_keys.reserve(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        // A single block may contribute several location-deletion tasks to one
        // RMW batch.  Reclaim and account for that block at most once even when
        // several of its locations become empty in the same request.
        if (delete_results[i] == EC_OK && seen_candidates.insert(keys[i]).second) {
            candidate_keys.push_back(keys[i]);
        }
    }
    if (candidate_keys.empty()) {
        return 0;
    }

    std::vector<bool> has_locations;
    std::vector<ErrorCode> exists_ecs;
    if (cache_backend_) {
        exists_ecs = cache_backend_->ExistsLocation(request_context, candidate_keys, has_locations);
    } else {
        exists_ecs = persistent_backend_->ExistsLocation(request_context, candidate_keys, has_locations);
    }
    if (exists_ecs.size() != candidate_keys.size() || has_locations.size() != candidate_keys.size()) {
        KVCM_LOG_ERROR("ExistsLocation results[%lu] values[%lu] mismatch candidate keys[%lu], "
                       "skip empty-key reclamation",
                       exists_ecs.size(),
                       has_locations.size(),
                       candidate_keys.size());
        return 0;
    }

    KeyVector reclaimed_keys;
    for (size_t i = 0; i < candidate_keys.size(); ++i) {
        if (exists_ecs[i] == EC_OK && !has_locations[i]) {
            reclaimed_keys.push_back(candidate_keys[i]);
        }
    }
    if (reclaimed_keys.empty()) {
        return 0;
    }

    std::vector<ErrorCode> whole_ecs = Delete(request_context, reclaimed_keys);
    if (whole_ecs.size() != reclaimed_keys.size()) {
        KVCM_LOG_ERROR("whole-key delete results[%lu] mismatch reclaimed keys[%lu], "
                       "skip key-count adjustment",
                       whole_ecs.size(),
                       reclaimed_keys.size());
        return 0;
    }
    int32_t reclaimed = 0;
    for (const ErrorCode ec : whole_ecs) {
        if (ec == EC_OK || ec == EC_NOENT) {
            ++reclaimed;
        }
    }
    return reclaimed;
}

std::vector<ErrorCode> MetaStorageBackendManager::Get(RequestContext *request_context,
                                                      const KeyVector &keys,
                                                      CacheLocationMapVector &out_locations,
                                                      PropertyMapVector &out_properties) noexcept {
    if (!cache_backend_) {
        return persistent_backend_->Get(request_context, keys, out_locations, out_properties);
    }

    std::vector<ErrorCode> results = cache_backend_->Get(request_context, keys, out_locations, out_properties);
    if (results.size() != keys.size() || out_locations.size() != keys.size() || out_properties.size() != keys.size()) {
        KVCM_LOG_ERROR("cache Get results[%lu] locations[%lu] properties[%lu] mismatch keys[%lu]",
                       results.size(),
                       out_locations.size(),
                       out_properties.size(),
                       keys.size());
        results.assign(keys.size(), EC_ERROR);
        out_locations.assign(keys.size(), CacheLocationMap{});
        out_properties.assign(keys.size(), PropertyMap{});
    }
    if (recover_state_.load(std::memory_order_acquire) == RecoverState::kRunning) {
        return results;
    }

    auto [missing_keys, missing_indices] = CollectMissingKeys(keys, results);
    if (missing_keys.empty()) {
        return results;
    }

    CacheLocationMapVector persistent_locations;
    PropertyMapVector persistent_properties;
    std::vector<ErrorCode> persistent_results =
        persistent_backend_->Get(request_context, missing_keys, persistent_locations, persistent_properties);
    if (missing_keys.size() != persistent_results.size() || missing_keys.size() != persistent_locations.size() ||
        missing_keys.size() != persistent_properties.size()) {
        KVCM_LOG_ERROR("persistent Get size mismatch: results[%lu] locations[%lu] properties[%lu] vs keys[%lu]",
                       persistent_results.size(),
                       persistent_locations.size(),
                       persistent_properties.size(),
                       missing_keys.size());
        for (size_t i = 0; i < missing_keys.size(); ++i) {
            const size_t original_idx = missing_indices[i];
            results[original_idx] = EC_ERROR;
            out_locations[original_idx].clear();
            out_properties[original_idx].clear();
        }
        return results;
    }
    for (size_t i = 0; i < missing_keys.size(); ++i) {
        const size_t original_idx = missing_indices[i];
        results[original_idx] = persistent_results[i];
        if (persistent_results[i] == EC_OK) {
            out_locations[original_idx] = std::move(persistent_locations[i]);
            out_properties[original_idx] = std::move(persistent_properties[i]);
        }
    }
    return results;
}

std::vector<ErrorCode> MetaStorageBackendManager::GetLocations(RequestContext *request_context,
                                                               const KeyVector &keys,
                                                               CacheLocationMapVector &out_location_maps) noexcept {
    if (!cache_backend_) {
        return persistent_backend_->GetLocations(request_context, keys, out_location_maps);
    }

    std::vector<ErrorCode> results = cache_backend_->GetLocations(request_context, keys, out_location_maps);
    if (results.size() != keys.size() || out_location_maps.size() != keys.size()) {
        KVCM_LOG_ERROR("cache GetLocations results[%lu] locations[%lu] mismatch keys[%lu]",
                       results.size(),
                       out_location_maps.size(),
                       keys.size());
        results.assign(keys.size(), EC_ERROR);
        out_location_maps.assign(keys.size(), CacheLocationMap{});
    }
    if (recover_state_.load(std::memory_order_acquire) == RecoverState::kRunning) {
        return results;
    }

    auto [missing_keys, missing_indices] = CollectMissingKeys(keys, results);
    if (missing_keys.empty()) {
        return results;
    }

    CacheLocationMapVector persistent_locations;
    std::vector<ErrorCode> persistent_results =
        persistent_backend_->GetLocations(request_context, missing_keys, persistent_locations);
    if (missing_keys.size() != persistent_results.size() || missing_keys.size() != persistent_locations.size()) {
        KVCM_LOG_ERROR("persistent GetLocations results[%lu] locations[%lu] mismatch keys[%lu]",
                       persistent_results.size(),
                       persistent_locations.size(),
                       missing_keys.size());
        for (size_t i = 0; i < missing_keys.size(); ++i) {
            const size_t original_idx = missing_indices[i];
            results[original_idx] = EC_ERROR;
            out_location_maps[original_idx].clear();
        }
        return results;
    }
    for (size_t i = 0; i < missing_keys.size(); ++i) {
        const size_t original_idx = missing_indices[i];
        results[original_idx] = persistent_results[i];
        if (persistent_results[i] == EC_OK) {
            out_location_maps[original_idx] = std::move(persistent_locations[i]);
        }
    }
    return results;
}

std::vector<ErrorCode> MetaStorageBackendManager::GetLocationValues(RequestContext *request_context,
                                                                    const KeyVector &keys,
                                                                    LocationsPerKey &out_locations) noexcept {
    if (!cache_backend_) {
        return persistent_backend_->GetLocationValues(request_context, keys, out_locations);
    }

    std::vector<ErrorCode> results = cache_backend_->GetLocationValues(request_context, keys, out_locations);
    if (results.size() != keys.size() || out_locations.size() != keys.size()) {
        KVCM_LOG_ERROR("cache location values results[%lu] locations[%lu] mismatch keys[%lu]",
                       results.size(),
                       out_locations.size(),
                       keys.size());
        // The two arrays form one positional contract. Once either shape is
        // broken, even an in-range EC_OK cannot be trusted to describe the
        // value at the same index.
        results.assign(keys.size(), EC_ERROR);
        out_locations.assign(keys.size(), CacheLocationVector{});
    }
    if (recover_state_.load(std::memory_order_acquire) == RecoverState::kRunning) {
        return results;
    }

    auto [missing_keys, missing_indices] = CollectMissingKeys(keys, results);
    if (missing_keys.empty()) {
        return results;
    }

    LocationsPerKey persistent_locations;
    std::vector<ErrorCode> persistent_results =
        persistent_backend_->GetLocationValues(request_context, missing_keys, persistent_locations);
    if (missing_keys.size() != persistent_results.size() || missing_keys.size() != persistent_locations.size()) {
        KVCM_LOG_ERROR("persistent location values results[%lu] locations[%lu] mismatch keys[%lu]",
                       persistent_results.size(),
                       persistent_locations.size(),
                       missing_keys.size());
        for (size_t i = 0; i < missing_keys.size(); ++i) {
            results[missing_indices[i]] = EC_ERROR;
        }
        return results;
    }
    for (size_t i = 0; i < missing_keys.size(); ++i) {
        const size_t original_idx = missing_indices[i];
        results[original_idx] = persistent_results[i];
        if (persistent_results[i] == EC_OK) {
            out_locations[original_idx] = std::move(persistent_locations[i]);
        }
    }
    return results;
}

std::vector<ErrorCode> MetaStorageBackendManager::GetLocationsFromPersistent(
    RequestContext *request_context, const KeyVector &keys, CacheLocationMapVector &out_location_maps) noexcept {
    out_location_maps.clear();
    if (keys.empty()) {
        return {};
    }
    if (!persistent_backend_) {
        KVCM_LOG_ERROR("persistent GetLocations failed, backend is null, instance[%s]", instance_id_.c_str());
        out_location_maps.resize(keys.size());
        return std::vector<ErrorCode>(keys.size(), EC_ERROR);
    }
    std::vector<ErrorCode> results = persistent_backend_->GetLocations(request_context, keys, out_location_maps);
    if (results.size() != keys.size() || out_location_maps.size() != keys.size()) {
        KVCM_LOG_ERROR("persistent GetLocations shape mismatch, instance[%s] keys[%zu] results[%zu] locations[%zu]",
                       instance_id_.c_str(),
                       keys.size(),
                       results.size(),
                       out_location_maps.size());
        out_location_maps.resize(keys.size());
        return std::vector<ErrorCode>(keys.size(), EC_ERROR);
    }
    return results;
}

std::vector<ErrorCode> MetaStorageBackendManager::RefreshCacheFromPersistent(RequestContext *request_context,
                                                                             const KeyVector &keys) noexcept {
    if (keys.empty()) {
        return {};
    }
    if (!cache_backend_) {
        return std::vector<ErrorCode>(keys.size(), EC_OK);
    }

    CacheLocationMapVector locations;
    PropertyMapVector properties;
    std::vector<ErrorCode> persistent_results = persistent_backend_->Get(request_context, keys, locations, properties);
    if (persistent_results.size() != keys.size() || locations.size() != keys.size() ||
        properties.size() != keys.size()) {
        KVCM_LOG_ERROR("persistent refresh shape mismatch, instance[%s] keys[%zu] results[%zu] locations[%zu] "
                       "properties[%zu]",
                       instance_id_.c_str(),
                       keys.size(),
                       persistent_results.size(),
                       locations.size(),
                       properties.size());
        return std::vector<ErrorCode>(keys.size(), EC_ERROR);
    }

    std::vector<ErrorCode> results = persistent_results;
    const std::vector<ErrorCode> put_results =
        cache_backend_->Put(request_context, keys, locations, properties, persistent_results);
    if (put_results.size() != keys.size()) {
        KVCM_LOG_ERROR("cache refresh Put shape mismatch, instance[%s] keys[%zu] results[%zu]",
                       instance_id_.c_str(),
                       keys.size(),
                       put_results.size());
        return std::vector<ErrorCode>(keys.size(), EC_ERROR);
    }

    KeyVector missing_keys;
    for (size_t i = 0; i < keys.size(); ++i) {
        if (persistent_results[i] == EC_OK) {
            results[i] = put_results[i];
        } else if (persistent_results[i] == EC_NOENT) {
            // Do not let a stale hot-cache entry resurrect metadata that has
            // already disappeared from the source of truth.
            missing_keys.push_back(keys[i]);
        }
    }
    if (!missing_keys.empty()) {
        (void)cache_backend_->Delete(request_context, missing_keys);
    }
    return results;
}

std::vector<ErrorCode>
MetaStorageBackendManager::GetLocationValuesCompact(RequestContext *request_context,
                                                    const KeyType *keys,
                                                    size_t key_count,
                                                    CompactLocationsPerKey &out_locations) noexcept {
    if (!cache_backend_) {
        return persistent_backend_->GetLocationValuesCompact(request_context, keys, key_count, out_locations);
    }

    // Cached mode must retain its recovery/fallback behavior. The large-query
    // compact fast path is deliberately restricted to the single local backend,
    // so use the existing manager API when this method is reached in another
    // configuration.
    KeyVector key_vector;
    if (key_count != 0) {
        if (keys == nullptr) {
            out_locations.Clear(key_count);
            for (size_t i = 0; i < key_count; ++i) {
                out_locations.FinishKey();
            }
            return std::vector<ErrorCode>(key_count, EC_BADARGS);
        }
        key_vector.assign(keys, keys + key_count);
    }
    LocationsPerKey locations;
    auto results = GetLocationValues(request_context, key_vector, locations);
    out_locations.Clear(key_count);
    const size_t value_count = std::min(key_count, locations.size());
    for (size_t i = 0; i < key_count; ++i) {
        if (i < value_count) {
            out_locations.values.insert(out_locations.values.end(), locations[i].begin(), locations[i].end());
        }
        out_locations.FinishKey();
    }
    return results;
}

std::vector<std::vector<ErrorCode>> MetaStorageBackendManager::GetLocations(RequestContext *request_context,
                                                                            const KeyVector &keys,
                                                                            const LocationIdsPerKey &location_ids,
                                                                            LocationsPerKey &out_locations) noexcept {
    if (keys.size() != location_ids.size()) {
        out_locations.assign(keys.size(), CacheLocationVector{});
        return std::vector<std::vector<ErrorCode>>(keys.size(), std::vector<ErrorCode>{EC_BADARGS});
    }
    if (!cache_backend_) {
        return persistent_backend_->GetLocations(request_context, keys, location_ids, out_locations);
    }

    std::vector<std::vector<ErrorCode>> results =
        cache_backend_->GetLocations(request_context, keys, location_ids, out_locations);
    if (results.size() != keys.size() || out_locations.size() != keys.size()) {
        KVCM_LOG_ERROR("cache targeted GetLocations results[%lu] locations[%lu] mismatch keys[%lu]",
                       results.size(),
                       out_locations.size(),
                       keys.size());
        out_locations.assign(keys.size(), CacheLocationVector{});
        results.resize(keys.size());
        for (size_t i = 0; i < keys.size(); ++i) {
            results[i].assign(location_ids[i].size(), EC_ERROR);
            out_locations[i].assign(location_ids[i].size(), CacheLocationConstPtr{});
        }
        return results;
    }
    for (size_t i = 0; i < keys.size(); ++i) {
        if (results[i].size() != location_ids[i].size() || out_locations[i].size() != location_ids[i].size()) {
            KVCM_LOG_ERROR("cache targeted GetLocations key[%ld] results[%lu] locations[%lu] mismatch ids[%lu]",
                           keys[i],
                           results[i].size(),
                           out_locations[i].size(),
                           location_ids[i].size());
            results[i].assign(location_ids[i].size(), EC_ERROR);
            out_locations[i].assign(location_ids[i].size(), CacheLocationConstPtr{});
        }
    }
    if (recover_state_.load(std::memory_order_acquire) == RecoverState::kRunning) {
        return results;
    }

    // Per-location EC_NOENT does not say whether the cache missed the whole
    // key or found the key without that location. Mixed OK/NOENT is
    // unambiguously a cache hit and must never fall back to a potentially
    // older persistent value. When every requested location is absent, use a
    // cheap key-existence probe to distinguish the two cases.
    KeyTypeVec ambiguous_keys;
    std::vector<size_t> ambiguous_indices;
    for (size_t i = 0; i < keys.size(); ++i) {
        if (!results[i].empty() &&
            std::all_of(results[i].begin(), results[i].end(), [](ErrorCode ec) { return ec == EC_NOENT; })) {
            ambiguous_keys.push_back(keys[i]);
            ambiguous_indices.push_back(i);
        }
    }

    KeyTypeVec missing_keys;
    std::vector<size_t> missing_indices;
    LocationIdsPerKey missing_location_ids;
    if (!ambiguous_keys.empty()) {
        std::vector<bool> cache_key_exists;
        const std::vector<ErrorCode> exists_results =
            cache_backend_->Exists(request_context, ambiguous_keys, cache_key_exists);
        if (exists_results.size() != ambiguous_keys.size() || cache_key_exists.size() != ambiguous_keys.size()) {
            KVCM_LOG_ERROR("cache key-existence results[%lu] values[%lu] mismatch ambiguous keys[%lu]",
                           exists_results.size(),
                           cache_key_exists.size(),
                           ambiguous_keys.size());
            for (const size_t original_idx : ambiguous_indices) {
                results[original_idx].assign(location_ids[original_idx].size(), EC_ERROR);
                out_locations[original_idx].assign(location_ids[original_idx].size(), CacheLocationConstPtr{});
            }
            return results;
        }
        for (size_t i = 0; i < ambiguous_keys.size(); ++i) {
            const size_t original_idx = ambiguous_indices[i];
            if (exists_results[i] != EC_OK) {
                results[original_idx].assign(location_ids[original_idx].size(), exists_results[i]);
                out_locations[original_idx].assign(location_ids[original_idx].size(), CacheLocationConstPtr{});
                continue;
            }
            if (!cache_key_exists[i]) {
                missing_keys.push_back(keys[original_idx]);
                missing_indices.push_back(original_idx);
                missing_location_ids.push_back(location_ids[original_idx]);
            }
        }
    }
    if (missing_keys.empty()) {
        return results;
    }

    LocationsPerKey persistent_locations;
    std::vector<std::vector<ErrorCode>> persistent_results =
        persistent_backend_->GetLocations(request_context, missing_keys, missing_location_ids, persistent_locations);
    if (missing_keys.size() != persistent_results.size() || missing_keys.size() != persistent_locations.size()) {
        KVCM_LOG_ERROR("persistent targeted GetLocations results[%lu] locations[%lu] mismatch keys[%lu]",
                       persistent_results.size(),
                       persistent_locations.size(),
                       missing_keys.size());
        for (size_t i = 0; i < missing_keys.size(); ++i) {
            results[missing_indices[i]].assign(location_ids[missing_indices[i]].size(), EC_ERROR);
        }
        return results;
    }
    for (size_t i = 0; i < missing_keys.size(); ++i) {
        const size_t original_idx = missing_indices[i];
        if (persistent_results[i].size() != missing_location_ids[i].size() ||
            persistent_locations[i].size() != missing_location_ids[i].size()) {
            KVCM_LOG_ERROR("persistent targeted GetLocations key[%ld] results[%lu] locations[%lu] mismatch ids[%lu]",
                           missing_keys[i],
                           persistent_results[i].size(),
                           persistent_locations[i].size(),
                           missing_location_ids[i].size());
            results[original_idx].assign(location_ids[original_idx].size(), EC_ERROR);
            out_locations[original_idx].assign(location_ids[original_idx].size(), CacheLocationConstPtr{});
            continue;
        }
        results[original_idx] = std::move(persistent_results[i]);
        out_locations[original_idx] = std::move(persistent_locations[i]);
    }
    return results;
}

std::vector<std::vector<ErrorCode>>
MetaStorageBackendManager::GetLocationsForMaintenance(RequestContext *request_context,
                                                      const KeyVector &keys,
                                                      const LocationIdsPerKey &location_ids,
                                                      LocationsPerKey &out_locations) noexcept {
    if (keys.size() != location_ids.size()) {
        out_locations.assign(keys.size(), CacheLocationVector{});
        return std::vector<std::vector<ErrorCode>>(keys.size(), std::vector<ErrorCode>{EC_BADARGS});
    }
    if (!cache_backend_) {
        return persistent_backend_->GetLocationsForMaintenance(request_context, keys, location_ids, out_locations);
    }

    // A unified GC round scans the hot cache, but a stale hot value must not
    // authorize deletion of a newer persistent value left by a partial mirror
    // failure. Revalidate both layers without touching/backfilling the cache;
    // one missing copy is safe to converge, while two different present copies
    // fail closed as a compare mismatch.
    LocationsPerKey hot_locations;
    const auto hot_results =
        cache_backend_->GetLocationsForMaintenance(request_context, keys, location_ids, hot_locations);
    LocationsPerKey persistent_locations;
    const auto persistent_results =
        persistent_backend_->GetLocationsForMaintenance(request_context, keys, location_ids, persistent_locations);

    std::vector<std::vector<ErrorCode>> results(keys.size());
    out_locations.resize(keys.size());
    const bool outer_shape_valid = hot_results.size() == keys.size() && hot_locations.size() == keys.size() &&
                                   persistent_results.size() == keys.size() &&
                                   persistent_locations.size() == keys.size();
    if (!outer_shape_valid) {
        KVCM_LOG_ERROR("maintenance target read shape mismatch, keys[%lu] hot_ecs[%lu] hot_values[%lu] "
                       "persistent_ecs[%lu] persistent_values[%lu]",
                       keys.size(),
                       hot_results.size(),
                       hot_locations.size(),
                       persistent_results.size(),
                       persistent_locations.size());
        for (size_t i = 0; i < keys.size(); ++i) {
            results[i].assign(location_ids[i].size(), EC_ERROR);
            out_locations[i].assign(location_ids[i].size(), CacheLocationConstPtr{});
        }
        return results;
    }

    for (size_t i = 0; i < keys.size(); ++i) {
        results[i].assign(location_ids[i].size(), EC_ERROR);
        out_locations[i].assign(location_ids[i].size(), CacheLocationConstPtr{});
        if (hot_results[i].size() != location_ids[i].size() || hot_locations[i].size() != location_ids[i].size() ||
            persistent_results[i].size() != location_ids[i].size() ||
            persistent_locations[i].size() != location_ids[i].size()) {
            KVCM_LOG_ERROR("maintenance target read key[%ld] shape mismatch, ids[%lu] hot_ecs[%lu] hot_values[%lu] "
                           "persistent_ecs[%lu] persistent_values[%lu]",
                           keys[i],
                           location_ids[i].size(),
                           hot_results[i].size(),
                           hot_locations[i].size(),
                           persistent_results[i].size(),
                           persistent_locations[i].size());
            continue;
        }

        for (size_t j = 0; j < location_ids[i].size(); ++j) {
            const auto validate_layer = [&](ErrorCode ec, const CacheLocationConstPtr &location) {
                if (ec == EC_NOENT) {
                    return location ? EC_ERROR : EC_NOENT;
                }
                if (ec != EC_OK) {
                    return ec;
                }
                return location && location->id() == location_ids[i][j] ? EC_OK : EC_ERROR;
            };
            const ErrorCode hot_ec = validate_layer(hot_results[i][j], hot_locations[i][j]);
            const ErrorCode persistent_ec = validate_layer(persistent_results[i][j], persistent_locations[i][j]);
            if (hot_ec != EC_OK && hot_ec != EC_NOENT) {
                results[i][j] = hot_ec;
                continue;
            }
            if (persistent_ec != EC_OK && persistent_ec != EC_NOENT) {
                results[i][j] = persistent_ec;
                continue;
            }
            if (hot_ec == EC_NOENT && persistent_ec == EC_NOENT) {
                results[i][j] = EC_NOENT;
                continue;
            }
            if (hot_ec == EC_OK && persistent_ec == EC_OK &&
                hot_locations[i][j]->ToJsonString() != persistent_locations[i][j]->ToJsonString()) {
                results[i][j] = EC_MISMATCH;
                continue;
            }
            results[i][j] = EC_OK;
            out_locations[i][j] = hot_ec == EC_OK ? hot_locations[i][j] : persistent_locations[i][j];
        }
    }
    return results;
}

std::vector<std::vector<ErrorCode>>
MetaStorageBackendManager::GetLocationsWithKeyStatus(RequestContext *request_context,
                                                     const KeyVector &keys,
                                                     const LocationIdsPerKey &location_ids,
                                                     LocationsPerKey &out_locations,
                                                     std::vector<ErrorCode> &out_key_error_codes) noexcept {
    if (keys.size() != location_ids.size()) {
        out_locations.assign(keys.size(), CacheLocationVector{});
        out_key_error_codes.assign(keys.size(), EC_BADARGS);
        return std::vector<std::vector<ErrorCode>>(keys.size(), std::vector<ErrorCode>{EC_BADARGS});
    }
    if (!cache_backend_) {
        return persistent_backend_->GetLocationsWithKeyStatus(
            request_context, keys, location_ids, out_locations, out_key_error_codes);
    }

    std::vector<std::vector<ErrorCode>> results = cache_backend_->GetLocationsWithKeyStatus(
        request_context, keys, location_ids, out_locations, out_key_error_codes);
    auto response_shape_valid = [&keys, &location_ids](const std::vector<std::vector<ErrorCode>> &per_location_ecs,
                                                       const LocationsPerKey &locations,
                                                       const std::vector<ErrorCode> &per_key_ecs) {
        if (per_location_ecs.size() != keys.size() || locations.size() != keys.size() ||
            per_key_ecs.size() != keys.size()) {
            return false;
        }
        for (size_t i = 0; i < keys.size(); ++i) {
            if (per_location_ecs[i].size() != location_ids[i].size() || locations[i].size() != location_ids[i].size()) {
                return false;
            }
        }
        return true;
    };
    if (!response_shape_valid(results, out_locations, out_key_error_codes)) {
        KVCM_LOG_ERROR("cache targeted GetLocationsWithKeyStatus response shape mismatch keys[%lu]", keys.size());
        out_locations.resize(keys.size());
        results.resize(keys.size());
        out_key_error_codes.assign(keys.size(), EC_ERROR);
        for (size_t i = 0; i < keys.size(); ++i) {
            out_locations[i].assign(location_ids[i].size(), CacheLocationConstPtr{});
            results[i].assign(location_ids[i].size(), EC_ERROR);
        }
        return results;
    }
    if (recover_state_.load(std::memory_order_acquire) == RecoverState::kRunning) {
        return results;
    }

    KeyVector missing_keys;
    std::vector<size_t> missing_indices;
    LocationIdsPerKey missing_location_ids;
    for (size_t i = 0; i < keys.size(); ++i) {
        if (out_key_error_codes[i] == EC_NOENT) {
            missing_keys.push_back(keys[i]);
            missing_indices.push_back(i);
            missing_location_ids.push_back(location_ids[i]);
        }
    }
    if (missing_keys.empty()) {
        return results;
    }

    LocationsPerKey persistent_locations;
    std::vector<ErrorCode> persistent_key_error_codes;
    std::vector<std::vector<ErrorCode>> persistent_results = persistent_backend_->GetLocationsWithKeyStatus(
        request_context, missing_keys, missing_location_ids, persistent_locations, persistent_key_error_codes);
    const auto persistent_shape_valid =
        [&missing_keys, &missing_location_ids](const std::vector<std::vector<ErrorCode>> &per_location_ecs,
                                               const LocationsPerKey &locations,
                                               const std::vector<ErrorCode> &per_key_ecs) {
            if (per_location_ecs.size() != missing_keys.size() || locations.size() != missing_keys.size() ||
                per_key_ecs.size() != missing_keys.size()) {
                return false;
            }
            for (size_t i = 0; i < missing_keys.size(); ++i) {
                if (per_location_ecs[i].size() != missing_location_ids[i].size() ||
                    locations[i].size() != missing_location_ids[i].size()) {
                    return false;
                }
            }
            return true;
        };
    if (!persistent_shape_valid(persistent_results, persistent_locations, persistent_key_error_codes)) {
        KVCM_LOG_ERROR("persistent targeted GetLocationsWithKeyStatus response shape mismatch keys[%lu]",
                       missing_keys.size());
        for (const size_t original_index : missing_indices) {
            results[original_index].assign(location_ids[original_index].size(), EC_ERROR);
            out_locations[original_index].assign(location_ids[original_index].size(), CacheLocationConstPtr{});
            out_key_error_codes[original_index] = EC_ERROR;
        }
        return results;
    }
    for (size_t i = 0; i < missing_keys.size(); ++i) {
        const size_t original_index = missing_indices[i];
        results[original_index] = std::move(persistent_results[i]);
        out_locations[original_index] = std::move(persistent_locations[i]);
        out_key_error_codes[original_index] = persistent_key_error_codes[i];
    }
    return results;
}

std::vector<ErrorCode>
MetaStorageBackendManager::GetSingleLocationsWithKeyStatus(RequestContext *request_context,
                                                           const KeyVector &keys,
                                                           const LocationIdRefVector &location_ids,
                                                           CacheLocationVector &out_locations,
                                                           std::vector<ErrorCode> &out_key_error_codes) noexcept {
    if (!SupportsSingleLocationRmw()) {
        out_locations.assign(keys.size(), CacheLocationConstPtr{});
        out_key_error_codes.assign(keys.size(), EC_UNIMPLEMENTED);
        return std::vector<ErrorCode>(keys.size(), EC_UNIMPLEMENTED);
    }
    return persistent_backend_->GetSingleLocationsWithKeyStatus(
        request_context, keys, location_ids, out_locations, out_key_error_codes);
}

void MetaStorageBackendManager::GetSingleLocationsWithKeyStatusInto(RequestContext *request_context,
                                                                    const KeyVector &keys,
                                                                    const LocationIdRefVector &location_ids,
                                                                    CacheLocationVector &out_locations,
                                                                    std::vector<ErrorCode> &out_key_error_codes,
                                                                    std::vector<ErrorCode> &out_results,
                                                                    SingleLocationRmwScratch &scratch,
                                                                    bool retain_handles) noexcept {
    if (!SupportsSingleLocationRmw()) {
        out_locations.assign(keys.size(), CacheLocationConstPtr{});
        out_key_error_codes.assign(keys.size(), EC_UNIMPLEMENTED);
        out_results.assign(keys.size(), EC_UNIMPLEMENTED);
        scratch.ReleaseRetainedHandles();
        return;
    }
    static_cast<MetaLocalBackend *>(persistent_backend_.get())
        ->GetSingleLocationsWithKeyStatusInto(request_context,
                                              keys,
                                              location_ids,
                                              out_locations,
                                              out_key_error_codes,
                                              out_results,
                                              scratch,
                                              retain_handles);
}

void MetaStorageBackendManager::GetSingleLocationViewsWithKeyStatusInto(RequestContext *request_context,
                                                                        const KeyVector &keys,
                                                                        const LocationIdRefVector &location_ids,
                                                                        CacheLocationViewVector &out_locations,
                                                                        std::vector<ErrorCode> &out_key_error_codes,
                                                                        std::vector<ErrorCode> &out_results,
                                                                        SingleLocationRmwScratch &scratch) noexcept {
    if (!SupportsSingleLocationRmw()) {
        out_locations.assign(keys.size(), nullptr);
        out_key_error_codes.assign(keys.size(), EC_UNIMPLEMENTED);
        out_results.assign(keys.size(), EC_UNIMPLEMENTED);
        scratch.ReleaseRetainedHandles();
        return;
    }
    static_cast<MetaLocalBackend *>(persistent_backend_.get())
        ->GetSingleLocationViewsWithKeyStatusInto(
            request_context, keys, location_ids, out_locations, out_key_error_codes, out_results, scratch);
}

bool MetaStorageBackendManager::SupportsConcurrentLocationValueReads() const noexcept {
    return !cache_backend_ && persistent_backend_ &&
           persistent_backend_->GetStorageType() == META_LOCAL_BACKEND_TYPE_STR;
}

bool MetaStorageBackendManager::SupportsSingleLocationRmw() const noexcept {
    // The allocation-light operations bypass the older generic virtual
    // methods. Restrict the fast path to the concrete production backend so a
    // decorator/subclass that overrides those generic methods for additional
    // semantics (fault injection, auditing, admission, etc.) is not bypassed.
    // Such backends retain correctness through the generic targeted RMW.
    return SupportsConcurrentLocationValueReads() && typeid(*persistent_backend_) == typeid(MetaLocalBackend);
}

bool MetaStorageBackendManager::GetPureLocalCacheHashSeed(uint32_t &out_hash_seed) const noexcept {
    if (cache_backend_ || !persistent_backend_) {
        return false;
    }
    const auto *local_backend = dynamic_cast<const MetaLocalBackend *>(persistent_backend_.get());
    return local_backend != nullptr && local_backend->GetCacheHashSeed(out_hash_seed);
}

std::vector<ErrorCode> MetaStorageBackendManager::GetLocationIds(RequestContext *request_context,
                                                                 const KeyVector &keys,
                                                                 LocationIdsPerKey &out_location_ids) noexcept {
    if (!cache_backend_) {
        return persistent_backend_->GetLocationIds(request_context, keys, out_location_ids);
    }

    std::vector<ErrorCode> results = cache_backend_->GetLocationIds(request_context, keys, out_location_ids);
    if (results.size() != keys.size() || out_location_ids.size() != keys.size()) {
        KVCM_LOG_ERROR("cache location ids results[%lu] ids[%lu] mismatch keys[%lu]",
                       results.size(),
                       out_location_ids.size(),
                       keys.size());
        results.assign(keys.size(), EC_ERROR);
        out_location_ids.assign(keys.size(), LocationIdVector{});
    }
    if (recover_state_.load(std::memory_order_acquire) == RecoverState::kRunning) {
        return results;
    }

    auto [missing_keys, missing_indices] = CollectMissingKeys(keys, results);
    if (missing_keys.empty()) {
        return results;
    }

    LocationIdsPerKey persistent_location_ids;
    std::vector<ErrorCode> persistent_results =
        persistent_backend_->GetLocationIds(request_context, missing_keys, persistent_location_ids);
    if (missing_keys.size() != persistent_results.size() || missing_keys.size() != persistent_location_ids.size()) {
        KVCM_LOG_ERROR("persistent location ids results[%lu] ids[%lu] mismatch keys[%lu]",
                       persistent_results.size(),
                       persistent_location_ids.size(),
                       missing_keys.size());
        for (size_t i = 0; i < missing_keys.size(); ++i) {
            results[missing_indices[i]] = EC_ERROR;
        }
        return results;
    }
    for (size_t i = 0; i < missing_keys.size(); ++i) {
        const size_t original_idx = missing_indices[i];
        results[original_idx] = persistent_results[i];
        if (persistent_results[i] == EC_OK) {
            out_location_ids[original_idx] = std::move(persistent_location_ids[i]);
        }
    }
    return results;
}

std::vector<ErrorCode> MetaStorageBackendManager::GetProperties(RequestContext *request_context,
                                                                const KeyVector &keys,
                                                                const std::vector<std::string> &field_names,
                                                                PropertyMapVector &out_properties) noexcept {
    if (!cache_backend_) {
        return persistent_backend_->GetProperties(request_context, keys, field_names, out_properties);
    }

    std::vector<ErrorCode> results = cache_backend_->GetProperties(request_context, keys, field_names, out_properties);
    if (results.size() != keys.size() || out_properties.size() != keys.size()) {
        KVCM_LOG_ERROR("cache GetProperties results[%lu] properties[%lu] mismatch keys[%lu]",
                       results.size(),
                       out_properties.size(),
                       keys.size());
        results.assign(keys.size(), EC_ERROR);
        out_properties.assign(keys.size(), PropertyMap{});
    }
    if (recover_state_.load(std::memory_order_acquire) == RecoverState::kRunning) {
        return results;
    }

    auto [missing_keys, missing_indices] = CollectMissingKeys(keys, results);
    if (missing_keys.empty()) {
        return results;
    }

    PropertyMapVector persistent_properties;
    std::vector<ErrorCode> persistent_results =
        persistent_backend_->GetProperties(request_context, missing_keys, field_names, persistent_properties);
    if (missing_keys.size() != persistent_results.size() || missing_keys.size() != persistent_properties.size()) {
        KVCM_LOG_ERROR("persistent GetProperties results[%lu] properties[%lu] mismatch keys[%lu]",
                       persistent_results.size(),
                       persistent_properties.size(),
                       missing_keys.size());
        for (size_t i = 0; i < missing_keys.size(); ++i) {
            const size_t original_idx = missing_indices[i];
            results[original_idx] = EC_ERROR;
            out_properties[original_idx].clear();
        }
        return results;
    }
    for (size_t i = 0; i < missing_keys.size(); ++i) {
        const size_t original_idx = missing_indices[i];
        results[original_idx] = persistent_results[i];
        if (persistent_results[i] == EC_OK) {
            out_properties[original_idx] = std::move(persistent_properties[i]);
        }
    }
    return results;
}

std::vector<ErrorCode> MetaStorageBackendManager::GetPropertiesForMaintenance(
    RequestContext *request_context,
    const KeyVector &keys,
    const std::vector<std::string> &field_names,
    PropertyMapVector &out_properties) noexcept {
    if (!cache_backend_) {
        return persistent_backend_->GetPropertiesForMaintenance(request_context, keys, field_names, out_properties);
    }

    std::vector<ErrorCode> results =
        cache_backend_->GetPropertiesForMaintenance(request_context, keys, field_names, out_properties);
    if (results.size() != keys.size() || out_properties.size() != keys.size()) {
        KVCM_LOG_ERROR("cache maintenance properties results[%zu] properties[%zu] mismatch keys[%zu]",
                       results.size(),
                       out_properties.size(),
                       keys.size());
        results.assign(keys.size(), EC_MISMATCH);
        out_properties.assign(keys.size(), PropertyMap{});
    }
    const bool requires_process_local_lru =
        std::find(field_names.begin(), field_names.end(), PROPERTY_LRU_TIME) != field_names.end() &&
        GetMaintenancePropertyCapability(PROPERTY_LRU_TIME) ==
            MaintenancePropertyCapability::kProcessLocalVolatile;
    if (recover_state_.load(std::memory_order_acquire) == RecoverState::kRunning ||
        requires_process_local_lru) {
        // A cache miss means process-local recency is unknown. Persistent
        // metadata may still contain a field with the same name, but it does
        // not belong to the current recency epoch and must not authorize an
        // ENFORCE decision.
        return results;
    }

    // Recovery fallback is read-only: unlike online Get, maintenance reads
    // must never hydrate or promote the hot cache as a side effect.
    auto [missing_keys, missing_indices] = CollectMissingKeys(keys, results);
    if (missing_keys.empty()) {
        return results;
    }
    PropertyMapVector persistent_properties;
    std::vector<ErrorCode> persistent_results = persistent_backend_->GetPropertiesForMaintenance(
        request_context, missing_keys, field_names, persistent_properties);
    if (persistent_results.size() != missing_keys.size() || persistent_properties.size() != missing_keys.size()) {
        KVCM_LOG_ERROR("persistent maintenance properties results[%zu] properties[%zu] mismatch keys[%zu]",
                       persistent_results.size(),
                       persistent_properties.size(),
                       missing_keys.size());
        for (const size_t original_idx : missing_indices) {
            results[original_idx] = EC_MISMATCH;
            out_properties[original_idx].clear();
        }
        return results;
    }
    for (size_t i = 0; i < missing_indices.size(); ++i) {
        const size_t original_idx = missing_indices[i];
        results[original_idx] = persistent_results[i];
        if (persistent_results[i] == EC_OK) {
            out_properties[original_idx] = std::move(persistent_properties[i]);
        }
    }
    return results;
}

MaintenanceReadResult MetaStorageBackendManager::GetForMaintenance(
    RequestContext *request_context,
    const KeyVector &keys,
    const std::vector<std::string> &field_names,
    CacheLocationMapVector &out_locations,
    PropertyMapVector &out_properties) noexcept {
    if (!cache_backend_) {
        return persistent_backend_->GetForMaintenance(
            request_context, keys, field_names, out_locations, out_properties);
    }

    MaintenanceReadResult results =
        cache_backend_->GetForMaintenance(request_context, keys, field_names, out_locations, out_properties);
    const bool location_shape_ok =
        results.location_error_codes.size() == keys.size() && out_locations.size() == keys.size();
    const bool property_shape_ok =
        results.property_error_codes.size() == keys.size() && out_properties.size() == keys.size();
    if (!location_shape_ok) {
        KVCM_LOG_ERROR("cache maintenance locations results[%zu] values[%zu] mismatch keys[%zu]",
                       results.location_error_codes.size(),
                       out_locations.size(),
                       keys.size());
        results.location_error_codes.assign(keys.size(), EC_MISMATCH);
        out_locations.assign(keys.size(), CacheLocationMap{});
    }
    if (!property_shape_ok) {
        KVCM_LOG_ERROR("cache maintenance properties results[%zu] values[%zu] mismatch keys[%zu]",
                       results.property_error_codes.size(),
                       out_properties.size(),
                       keys.size());
        results.property_error_codes.assign(keys.size(), EC_MISMATCH);
        out_properties.assign(keys.size(), PropertyMap{});
    }
    if (recover_state_.load(std::memory_order_acquire) == RecoverState::kRunning) {
        return results;
    }

    const bool requires_process_local_lru =
        std::find(field_names.begin(), field_names.end(), PROPERTY_LRU_TIME) != field_names.end() &&
        GetMaintenancePropertyCapability(PROPERTY_LRU_TIME) ==
            MaintenancePropertyCapability::kProcessLocalVolatile;

    // Fetch the union once, then merge each component independently. A
    // feature failure must not erase a valid location result (SHADOW relies
    // on that distinction), and neither component is backfilled into cache.
    KeyTypeVec missing_keys;
    std::vector<size_t> missing_indices;
    for (size_t i = 0; i < keys.size(); ++i) {
        if (results.location_error_codes[i] == EC_NOENT ||
            (!requires_process_local_lru && results.property_error_codes[i] == EC_NOENT)) {
            missing_keys.push_back(keys[i]);
            missing_indices.push_back(i);
        }
    }
    if (missing_keys.empty()) {
        return results;
    }

    CacheLocationMapVector persistent_locations;
    PropertyMapVector persistent_properties;
    MaintenanceReadResult persistent_results = persistent_backend_->GetForMaintenance(
        request_context, missing_keys, field_names, persistent_locations, persistent_properties);
    const bool persistent_location_shape_ok = persistent_results.location_error_codes.size() == missing_keys.size() &&
                                              persistent_locations.size() == missing_keys.size();
    const bool persistent_property_shape_ok = persistent_results.property_error_codes.size() == missing_keys.size() &&
                                              persistent_properties.size() == missing_keys.size();
    for (size_t i = 0; i < missing_indices.size(); ++i) {
        const size_t original_idx = missing_indices[i];
        if (results.location_error_codes[original_idx] == EC_NOENT) {
            if (!persistent_location_shape_ok) {
                results.location_error_codes[original_idx] = EC_MISMATCH;
                out_locations[original_idx].clear();
            } else {
                results.location_error_codes[original_idx] = persistent_results.location_error_codes[i];
                if (persistent_results.location_error_codes[i] == EC_OK) {
                    out_locations[original_idx] = std::move(persistent_locations[i]);
                }
            }
        }
        if (!requires_process_local_lru && results.property_error_codes[original_idx] == EC_NOENT) {
            if (!persistent_property_shape_ok) {
                results.property_error_codes[original_idx] = EC_MISMATCH;
                out_properties[original_idx].clear();
            } else {
                results.property_error_codes[original_idx] = persistent_results.property_error_codes[i];
                if (persistent_results.property_error_codes[i] == EC_OK) {
                    out_properties[original_idx] = std::move(persistent_properties[i]);
                }
            }
        }
    }
    return results;
}

MaintenancePropertyCapability MetaStorageBackendManager::GetMaintenancePropertyCapability(
    const std::string &property_name) const noexcept {
    if (property_name != PROPERTY_LRU_TIME) {
        return MaintenancePropertyCapability::kUnsupported;
    }
    const MetaStorageBackend *authoritative_backend =
        cache_backend_ ? static_cast<const MetaStorageBackend *>(cache_backend_.get()) : persistent_backend_.get();
    if (dynamic_cast<const MetaLocalBackend *>(authoritative_backend) != nullptr) {
        return MaintenancePropertyCapability::kProcessLocalVolatile;
    }
    return MaintenancePropertyCapability::kUnsupported;
}

std::vector<ErrorCode> MetaStorageBackendManager::Exists(RequestContext *request_context,
                                                         const KeyVector &keys,
                                                         std::vector<bool> &out_is_exist_vec) noexcept {
    if (!cache_backend_) {
        return persistent_backend_->Exists(request_context, keys, out_is_exist_vec);
    }
    std::vector<ErrorCode> results = cache_backend_->Exists(request_context, keys, out_is_exist_vec);
    if (results.size() != keys.size() || out_is_exist_vec.size() != keys.size()) {
        KVCM_LOG_ERROR("cache Exists results[%lu] values[%lu] mismatch keys[%lu]",
                       results.size(),
                       out_is_exist_vec.size(),
                       keys.size());
        results.assign(keys.size(), EC_ERROR);
        out_is_exist_vec.assign(keys.size(), false);
    }
    if (recover_state_.load(std::memory_order_acquire) == RecoverState::kRunning) {
        return results;
    }

    KeyTypeVec missing_keys;
    std::vector<size_t> missing_indices;
    for (size_t i = 0; i < keys.size(); ++i) {
        if ((results[i] == EC_OK && !out_is_exist_vec[i]) || results[i] == EC_NOENT) {
            missing_keys.push_back(keys[i]);
            missing_indices.push_back(i);
        }
    }
    if (missing_keys.empty()) {
        return results;
    }

    std::vector<bool> persistent_exists;
    std::vector<ErrorCode> persistent_results =
        persistent_backend_->Exists(request_context, missing_keys, persistent_exists);
    if (missing_keys.size() != persistent_results.size() || missing_keys.size() != persistent_exists.size()) {
        KVCM_LOG_ERROR("persistent Exists results[%lu] values[%lu] mismatch keys[%lu]",
                       persistent_results.size(),
                       persistent_exists.size(),
                       missing_keys.size());
        for (size_t i = 0; i < missing_keys.size(); ++i) {
            const size_t original_idx = missing_indices[i];
            results[original_idx] = EC_ERROR;
            out_is_exist_vec[original_idx] = false;
        }
        return results;
    }
    for (size_t i = 0; i < missing_keys.size(); ++i) {
        const size_t original_idx = missing_indices[i];
        results[original_idx] = persistent_results[i];
        out_is_exist_vec[original_idx] = persistent_exists[i];
    }
    return results;
}

ErrorCode MetaStorageBackendManager::ListKeys(RequestContext *request_context,
                                              const std::string &cursor,
                                              const int64_t limit,
                                              std::string &out_next_cursor,
                                              KeyTypeVec &out_keys) noexcept {
    if (cache_backend_ && recover_state_.load(std::memory_order_acquire) == RecoverState::kRunning) {
        return cache_backend_->ListKeys(request_context, cursor, limit, out_next_cursor, out_keys);
    }
    return persistent_backend_->ListKeys(request_context, cursor, limit, out_next_cursor, out_keys);
}

ErrorCode MetaStorageBackendManager::ScanLocationsForMaintenance(RequestContext *request_context,
                                                                 const std::string &cursor,
                                                                 const int64_t limit,
                                                                 MaintenanceScanBatch &out) noexcept {
    out.Clear();
    // In dual-backend mode the in-memory backend is the GC discovery view.
    // It is expected to contain the useful metadata working set and avoids a
    // periodic full scan against Redis. Missing or evicted keys are an
    // accepted best-effort tradeoff; deletion admission still performs an
    // expected-value RMW under the ordinary metadata mutation fence.
    MetaStorageBackend *scan_backend = cache_backend_ ? cache_backend_.get() : persistent_backend_.get();
    if (!scan_backend) {
        KVCM_LOG_ERROR("maintenance scan failed, scan backend is null, instance[%s]", instance_id_.c_str());
        return EC_ERROR;
    }

    MaintenanceScanBatch batch;
    ErrorCode ec = scan_backend->ScanLocationsForMaintenance(request_context, cursor, limit, batch);
    if (ec != EC_OK) {
        return ec;
    }
    if (batch.next_cursor.empty() || batch.keys.size() != batch.locations.size() ||
        batch.keys.size() != batch.location_results.size()) {
        KVCM_LOG_ERROR(
            "maintenance scan result invalid, instance[%s] cursor_empty[%d] keys[%zu] locations[%zu] results[%zu]",
            instance_id_.c_str(),
            batch.next_cursor.empty(),
            batch.keys.size(),
            batch.locations.size(),
            batch.location_results.size());
        return EC_ERROR;
    }
    out = std::move(batch);
    return EC_OK;
}

ErrorCode MetaStorageBackendManager::RandomSample(RequestContext *request_context,
                                                  const int64_t count,
                                                  KeyTypeVec &out_keys) noexcept {
    if (cache_backend_ && recover_state_.load(std::memory_order_acquire) == RecoverState::kRunning) {
        return cache_backend_->RandomSample(request_context, count, out_keys);
    }
    return persistent_backend_->RandomSample(request_context, count, out_keys);
}

ErrorCode MetaStorageBackendManager::SampleReclaimKeys(RequestContext *request_context,
                                                       const int64_t count,
                                                       KeyTypeVec &out_keys) noexcept {
    if (count <= 0) {
        return EC_OK;
    }
    // Until recover finishes, the cache backend has not seen every key yet,
    // so we still go to persistent to avoid biased reclamation. In single-
    // backend mode (no cache) we always go to persistent.
    if (cache_backend_ && recover_state_.load(std::memory_order_acquire) == RecoverState::kRunning) {
        return cache_backend_->SampleReclaimKeys(request_context, count, out_keys);
    }
    return persistent_backend_->SampleReclaimKeys(request_context, count, out_keys);
}

ErrorCode MetaStorageBackendManager::PutMetaData(const FieldMap &field_maps) noexcept {
    return persistent_backend_->PutMetaData(field_maps);
}

ErrorCode MetaStorageBackendManager::GetMetaData(FieldMap &field_maps) noexcept {
    return persistent_backend_->GetMetaData(field_maps);
}

bool MetaStorageBackendManager::Sync(const KeyVector &keys) noexcept {
    if (!persistent_backend_) {
        return true;
    }
    return persistent_backend_->Sync(keys);
}

MetaStorageBackend::AsyncWriteStats MetaStorageBackendManager::GetAsyncWriteStats() noexcept {
    if (!persistent_backend_) {
        return {};
    }
    return persistent_backend_->GetAsyncWriteStats();
}

size_t MetaStorageBackendManager::GetMemUsage() const noexcept {
    if (cache_backend_) {
        return cache_backend_->GetMemUsage();
    }
    return 0;
}

int64_t MetaStorageBackendManager::GetOldestAccessTime() const noexcept {
    if (cache_backend_) {
        return cache_backend_->GetOldestAccessTime();
    }
    return INT64_MAX;
}

void MetaStorageBackendManager::SetRevisitHistogram(std::shared_ptr<RevisitIntervalHistogram> histogram) {
    // 无条件调用两个 backend 的 SetRevisitHistogram
    // 基类默认实现为空操作（no-op），只有支持重访间隔统计的 backend 才会重写
    if (persistent_backend_) {
        persistent_backend_->SetRevisitHistogram(histogram);
    }
    if (cache_backend_) {
        cache_backend_->SetRevisitHistogram(histogram);
    }
}

} // namespace kv_cache_manager
