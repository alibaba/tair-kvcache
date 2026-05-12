#include "kv_cache_manager/meta/meta_storage_backend_manager.h"

#include <cassert>
#include <utility>

#include "kv_cache_manager/common/error_code.h"
#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/common/standard_uri.h"
#include "kv_cache_manager/common/timestamp_util.h"
#include "kv_cache_manager/config/meta_storage_backend_config.h"
#include "kv_cache_manager/meta/common.h"
#include "kv_cache_manager/meta/meta_storage_backend_factory.h"
#include "kv_cache_manager/metrics/metrics_collector.h"

namespace kv_cache_manager {

namespace {
constexpr int64_t kRecoverScanBatchSize = 1000;
constexpr int kRecoverMaxConsecutiveFailures = 3;

void SetIndexSerializeTimeUs(RequestContext *request_context, int64_t delta_us) noexcept {
    if (request_context == nullptr || delta_us <= 0) {
        return;
    }
    auto *service_metrics_collector = dynamic_cast<ServiceMetricsCollector *>(request_context->metrics_collector());
    if (service_metrics_collector == nullptr) {
        return;
    }
    KVCM_METRICS_COLLECTOR_SET_METRICS(
        service_metrics_collector, meta_searcher, index_serialize_time_us, static_cast<double>(delta_us));
}

void SetIndexDeserializeTimeUs(RequestContext *request_context, int64_t delta_us) noexcept {
    if (request_context == nullptr || delta_us <= 0) {
        return;
    }
    auto *service_metrics_collector = dynamic_cast<ServiceMetricsCollector *>(request_context->metrics_collector());
    if (service_metrics_collector == nullptr) {
        return;
    }
    KVCM_METRICS_COLLECTOR_SET_METRICS(
        service_metrics_collector, meta_searcher, index_deserialize_time_us, static_cast<double>(delta_us));
}
} // namespace

MetaStorageBackendManager::~MetaStorageBackendManager() {
    // Defensive cleanup in case Close was never called explicitly.
    is_closed_.store(true, std::memory_order_release);
    if (recover_thread_.joinable()) {
        recover_thread_.join();
    }
}

ErrorCode MetaStorageBackendManager::Init(const std::string &instance_id,
                                          const std::shared_ptr<MetaStorageBackendConfig> &config) noexcept {
    if (instance_id.empty()) {
        KVCM_LOG_ERROR("init meta storage backend manager failed, empty instance id");
        return EC_BADARGS;
    }
    if (!config) {
        KVCM_LOG_ERROR("init meta storage backend manager failed, null storage backend config");
        return EC_BADARGS;
    }
    instance_id_ = instance_id;

    const std::string &storage_uri = config->GetStorageUri();
    if (config->GetStorageType() != META_CACHED_BACKEND_TYPE_STR) {
        // Single-backend mode: one backend serves every read/write directly.
        persistent_backend_ = MetaStorageBackendFactory::CreateAndInitStorageBackend(instance_id_, config);
        if (!persistent_backend_) {
            KVCM_LOG_ERROR("fail to create persistent backend uri[%s]", storage_uri.c_str());
            return EC_ERROR;
        }
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
    persistent_backend_ = MetaStorageBackendFactory::CreatePersistentBackend(instance_id_, persistent_config);
    if (!persistent_backend_) {
        KVCM_LOG_ERROR("fail to create persistent backend uri[%s]", storage_uri.c_str());
        return EC_ERROR;
    }
    auto cache_config = std::make_shared<MetaStorageBackendConfig>(cache_type);
    cache_config->SetStorageUri(storage_uri);
    cache_backend_ = MetaStorageBackendFactory::CreateCacheBackend(instance_id_, cache_config);
    if (!cache_backend_) {
        KVCM_LOG_ERROR("fail to create cache backend uri[%s]", storage_uri.c_str());
        return EC_ERROR;
    }
    KVCM_LOG_INFO("meta storage backend manager init ok, instance[%s] cache[%s] persistent[%s]",
                  instance_id_.c_str(),
                  cache_type.c_str(),
                  persistent_type.c_str());
    return EC_OK;
}

ErrorCode MetaStorageBackendManager::Open() noexcept {
    if (!persistent_backend_) {
        KVCM_LOG_ERROR("persistent backend not inited! instance[%s]", instance_id_.c_str());
        return EC_ERROR;
    }

    ErrorCode ec = persistent_backend_->Open();
    if (ec != EC_OK) {
        KVCM_LOG_ERROR("open persistent failed, instance[%s] ec[%d]", instance_id_.c_str(), ec);
        return ec;
    }
    is_closed_.store(false, std::memory_order_release);

    if (!cache_backend_) {
        recover_state_.store(RecoverState::kRunning, std::memory_order_release);
        KVCM_LOG_INFO("meta storage backend manager opened in single-backend mode, instance[%s]", instance_id_.c_str());
        return EC_OK;
    }

    ec = cache_backend_->Open();
    if (ec != EC_OK) {
        KVCM_LOG_ERROR("open cache failed, instance[%s] ec[%d]", instance_id_.c_str(), ec);
        return ec;
    }
    recover_state_.store(RecoverState::kRecover, std::memory_order_release);
    recover_thread_ = std::thread(&MetaStorageBackendManager::AsyncRecoverTask, this);
    KVCM_LOG_INFO("meta storage backend manager opened, instance[%s], async recover started", instance_id_.c_str());
    return EC_OK;
}

ErrorCode MetaStorageBackendManager::Close() noexcept {
    is_closed_.store(true, std::memory_order_release);
    if (recover_thread_.joinable()) {
        recover_thread_.join();
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
    do {
        if (is_closed_.load(std::memory_order_acquire)) {
            KVCM_LOG_INFO("async recover aborted due to close, instance[%s]", instance_id_.c_str());
            return;
        }

        scanned_keys.clear();
        field_maps.clear();
        ErrorCode scan_ec = persistent_backend_->ListKeys(cursor, kRecoverScanBatchSize, next_cursor, scanned_keys);
        if (scan_ec != EC_OK) {
            ++consecutive_failures;
            KVCM_LOG_ERROR("async recover scan failed, instance[%s] cursor[%s] ec[%d] attempt[%d/%d]",
                           instance_id_.c_str(),
                           cursor.c_str(),
                           scan_ec,
                           consecutive_failures,
                           kRecoverMaxConsecutiveFailures);
            if (consecutive_failures >= kRecoverMaxConsecutiveFailures) {
                KVCM_LOG_ERROR("async recover giving up after %d consecutive failures, "
                               "forcing transition to Running, instance[%s]",
                               kRecoverMaxConsecutiveFailures,
                               instance_id_.c_str());
                break;
            }
            continue;
        }
        consecutive_failures = 0;

        if (!scanned_keys.empty()) {
            std::vector<ErrorCode> get_error_codes = persistent_backend_->GetAllFields(scanned_keys, field_maps);
            for (size_t i = 0; i < scanned_keys.size(); ++i) {
                if (get_error_codes[i] != EC_OK && get_error_codes[i] != EC_NOENT) {
                    KVCM_LOG_WARN("async recover key[%ld] get failed ec[%d]", scanned_keys[i], get_error_codes[i]);
                }
            }
            total_backfilled_keys += BackfillKeysToCache(scanned_keys, field_maps, get_error_codes);
        }
        cursor = next_cursor;
    } while (cursor != SCAN_BASE_CURSOR);

    if (consecutive_failures == 0) {
        KVCM_LOG_INFO("async recover completed instance[%s] total_backfilled_keys[%ld]",
                      instance_id_.c_str(),
                      total_backfilled_keys);
    } else {
        KVCM_LOG_WARN("async recover partial, instance[%s] total_backfilled_keys[%ld], forcing transition to Running",
                      instance_id_.c_str(),
                      total_backfilled_keys);
    }
    recover_state_.store(RecoverState::kRunning, std::memory_order_release);
    {
        std::lock_guard<std::mutex> lock(deleted_keys_mutex_);
        deleted_keys_.clear();
    }
}

void MetaStorageBackendManager::EnsureKeyInCache(const KeyTypeVec &keys) noexcept {
    if (keys.empty()) {
        return;
    }
    std::vector<bool> exists_vec;
    std::vector<ErrorCode> exists_results = cache_backend_->Exists(keys, exists_vec);
    KeyTypeVec missing_keys;
    for (size_t i = 0; i < keys.size(); ++i) {
        if (exists_results[i] != EC_OK || !exists_vec[i]) {
            missing_keys.emplace_back(keys[i]);
        }
    }
    if (missing_keys.empty()) {
        return;
    }

    FieldMapVec field_maps;
    std::vector<ErrorCode> get_results = persistent_backend_->GetAllFields(missing_keys, field_maps);
    std::vector<ErrorCode> put_results = cache_backend_->Put(missing_keys, field_maps, get_results);
    for (size_t i = 0; i < missing_keys.size(); ++i) {
        if (put_results[i] != EC_OK && put_results[i] != EC_NOENT) {
            KVCM_LOG_WARN("ensure key[%ld] in cache failed, ec[%d]", missing_keys[i], put_results[i]);
        }
    }
}

int64_t MetaStorageBackendManager::BackfillKeysToCache(const KeyTypeVec &keys,
                                                       const FieldMapVec &field_maps,
                                                       const std::vector<ErrorCode> &get_error_codes) noexcept {
    std::lock_guard<std::mutex> lock(deleted_keys_mutex_);

    assert(get_error_codes.size() == keys.size());
    std::vector<ErrorCode> previous_error_codes(keys.size());
    int64_t eligible_count = 0;
    for (size_t i = 0; i < keys.size(); ++i) {
        if (deleted_keys_.count(keys[i]) > 0) {
            previous_error_codes[i] = EC_NOENT;
        } else {
            previous_error_codes[i] = get_error_codes[i];
            eligible_count += (previous_error_codes[i] == EC_OK);
        }
    }
    if (eligible_count == 0) {
        return 0;
    }

    std::vector<ErrorCode> put_results = cache_backend_->PutIfAbsent(keys, field_maps, previous_error_codes);
    int64_t backfilled_count = 0;
    for (size_t i = 0; i < keys.size(); ++i) {
        if (put_results[i] == EC_OK) {
            ++backfilled_count;
        } else if (put_results[i] != EC_NOENT) {
            KVCM_LOG_WARN("backfill PutIfAbsent failed key[%ld] ec[%d]", keys[i], put_results[i]);
        }
    }
    return backfilled_count;
}

std::string MetaStorageBackendManager::MakeLocationFieldName(const std::string &location_id) noexcept {
    return LOCATION_PREFIX + location_id;
}

PropertyMapVector &MetaStorageBackendManager::BuildEffectiveFieldMaps(RequestContext *request_context,
                                                                      BatchMetaData &batch) const noexcept {
    const size_t key_count = batch.batch_keys.size();

    // When the caller only supplied locations, grow it here to hold the upcoming location fields.
    if (batch.batch_properties.empty()) {
        batch.batch_properties.resize(key_count);
    }
    assert(batch.batch_properties.size() == key_count);

    if (batch.batch_locations.empty()) {
        return batch.batch_properties;
    }

    assert(batch.batch_locations.size() == key_count);
    const int64_t begin_us = TimestampUtil::GetCurrentTimeUs();
    for (size_t i = 0; i < key_count; ++i) {
        for (const auto &loc_kv : batch.batch_locations[i]) {
            const CacheLocation &location = loc_kv.second;
            batch.batch_properties[i][MakeLocationFieldName(location.id())] = location.ToJsonString();
        }
    }
    SetIndexSerializeTimeUs(request_context, TimestampUtil::GetCurrentTimeUs() - begin_us);
    return batch.batch_properties;
}

std::vector<ErrorCode> MetaStorageBackendManager::Put(RequestContext *request_context, BatchMetaData &batch) noexcept {
    const auto &keys = batch.batch_keys;
    PropertyMapVector &effective = BuildEffectiveFieldMaps(request_context, batch);
    std::vector<ErrorCode> persistent_results = persistent_backend_->Put(keys, effective);
    if (!cache_backend_) {
        return persistent_results;
    }
    return cache_backend_->Put(keys, effective, persistent_results);
}

std::vector<ErrorCode> MetaStorageBackendManager::UpdateFields(RequestContext *request_context,
                                                               BatchMetaData &batch) noexcept {
    const auto &keys = batch.batch_keys;
    PropertyMapVector &effective = BuildEffectiveFieldMaps(request_context, batch);

    // Partial-update during Recover: hydrate cache from persistent first so
    // the conditional mirror write below has the full pre-restart field set
    // to update against (and async backfill cannot later overwrite us).
    if (cache_backend_ && recover_state_.load(std::memory_order_acquire) == RecoverState::kRecover) {
        EnsureKeyInCache(keys);
    }
    std::vector<ErrorCode> persistent_results = persistent_backend_->UpdateFields(keys, effective);
    if (!cache_backend_) {
        return persistent_results;
    }
    return cache_backend_->UpdateFields(keys, effective, persistent_results);
}

std::vector<ErrorCode> MetaStorageBackendManager::Upsert(RequestContext *request_context,
                                                         BatchMetaData &batch) noexcept {
    const auto &keys = batch.batch_keys;
    PropertyMapVector &effective = BuildEffectiveFieldMaps(request_context, batch);

    // See UpdateFields(): Upsert may also touch only a subset of fields, so
    // the same Recover-time hydration is needed.
    if (cache_backend_ && recover_state_.load(std::memory_order_acquire) == RecoverState::kRecover) {
        EnsureKeyInCache(keys);
    }
    std::vector<ErrorCode> persistent_results = persistent_backend_->Upsert(keys, effective);
    if (!cache_backend_) {
        return persistent_results;
    }
    return cache_backend_->Upsert(keys, effective, persistent_results);
}

std::vector<ErrorCode> MetaStorageBackendManager::Delete(const KeyVector &keys) noexcept {
    std::vector<ErrorCode> persistent_results = persistent_backend_->Delete(keys);
    if (!cache_backend_) {
        return persistent_results;
    }
    if (recover_state_.load(std::memory_order_acquire) == RecoverState::kRecover) {
        // Tombstone to prevent Recover backfill from resurrecting deleted keys.
        std::lock_guard<std::mutex> lock(deleted_keys_mutex_);
        for (const auto &key : keys) {
            deleted_keys_.insert(key);
        }
    }
    return cache_backend_->Delete(keys, persistent_results);
}

std::vector<ErrorCode> MetaStorageBackendManager::Delete(const KeyVector &keys,
                                                         const LocationIdsPerKey &location_ids,
                                                         int32_t &out_reclaimed_count) noexcept {
    out_reclaimed_count = 0;
    if (keys.empty()) {
        return {};
    }
    assert(location_ids.size() == keys.size());

    // Partial-delete during Recover: hydrate cache from persistent first so
    // the conditional mirror write below has the full pre-restart field set
    // to delete against (and async backfill cannot later overwrite us).
    if (cache_backend_ && recover_state_.load(std::memory_order_acquire) == RecoverState::kRecover) {
        EnsureKeyInCache(keys);
    }

    std::vector<std::vector<std::string>> field_names_vec(keys.size());
    bool any_field_to_delete = false;
    for (size_t i = 0; i < keys.size(); ++i) {
        field_names_vec[i].reserve(location_ids[i].size());
        for (const auto &id : location_ids[i]) {
            field_names_vec[i].emplace_back(MakeLocationFieldName(id));
        }
        any_field_to_delete = any_field_to_delete || !field_names_vec[i].empty();
    }
    if (!any_field_to_delete) {
        return std::vector<ErrorCode>(keys.size(), EC_OK);
    }

    std::vector<ErrorCode> persistent_results = persistent_backend_->DeleteFields(keys, field_names_vec);
    std::vector<ErrorCode> results;
    if (!cache_backend_) {
        results = std::move(persistent_results);
    } else {
        results = cache_backend_->DeleteFields(keys, field_names_vec, persistent_results);
    }

    out_reclaimed_count = MaybeReclaimEmptyKeys(keys, results);
    return results;
}

int32_t MetaStorageBackendManager::MaybeReclaimEmptyKeys(const KeyVector &keys,
                                                         const std::vector<ErrorCode> &delete_results) noexcept {
    KeyVector candidate_keys;
    for (size_t i = 0; i < keys.size(); ++i) {
        if (delete_results[i] == EC_OK) {
            candidate_keys.push_back(keys[i]);
        }
    }
    if (candidate_keys.empty()) {
        return 0;
    }

    std::vector<bool> has_locations;
    std::vector<ErrorCode> exists_ecs;
    if (cache_backend_) {
        exists_ecs = cache_backend_->ExistsFieldWithPrefix(candidate_keys, LOCATION_PREFIX, has_locations);
    } else {
        exists_ecs = persistent_backend_->ExistsFieldWithPrefix(candidate_keys, LOCATION_PREFIX, has_locations);
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

    std::vector<ErrorCode> whole_ecs = Delete(reclaimed_keys);
    int32_t reclaimed = 0;
    for (const ErrorCode ec : whole_ecs) {
        if (ec == EC_OK || ec == EC_NOENT) {
            ++reclaimed;
        }
    }
    return reclaimed;
}

std::vector<ErrorCode> MetaStorageBackendManager::Get(const KeyVector &keys,
                                                      const std::vector<std::string> &field_names,
                                                      FieldMapVec &out_field_maps) noexcept {
    if (!cache_backend_) {
        return persistent_backend_->Get(keys, field_names, out_field_maps);
    }

    std::vector<ErrorCode> results = cache_backend_->Get(keys, field_names, out_field_maps);
    if (recover_state_.load(std::memory_order_acquire) == RecoverState::kRunning) {
        return results;
    }

    KeyTypeVec missing_keys;
    std::vector<size_t> missing_indices;
    for (size_t i = 0; i < keys.size(); ++i) {
        if (results[i] == EC_NOENT) {
            missing_keys.push_back(keys[i]);
            missing_indices.push_back(i);
        }
    }
    if (missing_keys.empty()) {
        return results;
    }

    FieldMapVec persistent_field_maps;
    std::vector<ErrorCode> persistent_results =
        persistent_backend_->Get(missing_keys, field_names, persistent_field_maps);
    if (missing_keys.size() != persistent_field_maps.size()) {
        KVCM_LOG_ERROR("persistent_field_maps size[%lu] mismatch missing_keys's[%lu]",
                       persistent_field_maps.size(),
                       missing_keys.size());
        for (size_t i = 0; i < missing_keys.size(); ++i) {
            results[missing_indices[i]] = EC_ERROR;
        }
        return results;
    }
    for (size_t i = 0; i < missing_keys.size(); ++i) {
        const size_t original_idx = missing_indices[i];
        results[original_idx] = persistent_results[i];
        out_field_maps[original_idx] = std::move(persistent_field_maps[i]);
    }
    return results;
}

std::vector<ErrorCode> MetaStorageBackendManager::Get(const KeyVector &keys,
                                                      const std::vector<std::vector<std::string>> &field_names_vec,
                                                      FieldMapVec &out_field_maps) noexcept {
    if (!cache_backend_) {
        return persistent_backend_->Get(keys, field_names_vec, out_field_maps);
    }

    std::vector<ErrorCode> results = cache_backend_->Get(keys, field_names_vec, out_field_maps);
    if (recover_state_.load(std::memory_order_acquire) == RecoverState::kRunning) {
        return results;
    }

    KeyTypeVec missing_keys;
    std::vector<std::vector<std::string>> missing_field_names;
    std::vector<size_t> missing_indices;
    for (size_t i = 0; i < keys.size(); ++i) {
        if (results[i] == EC_NOENT) {
            missing_keys.push_back(keys[i]);
            missing_field_names.push_back(field_names_vec[i]);
            missing_indices.push_back(i);
        }
    }
    if (missing_keys.empty()) {
        return results;
    }

    FieldMapVec persistent_field_maps;
    std::vector<ErrorCode> persistent_results =
        persistent_backend_->Get(missing_keys, missing_field_names, persistent_field_maps);
    if (missing_keys.size() != persistent_field_maps.size()) {
        KVCM_LOG_ERROR("persistent_field_maps size[%lu] mismatch missing_keys's[%lu]",
                       persistent_field_maps.size(),
                       missing_keys.size());
        for (size_t i = 0; i < missing_keys.size(); ++i) {
            results[missing_indices[i]] = EC_ERROR;
        }
        return results;
    }
    for (size_t i = 0; i < missing_keys.size(); ++i) {
        const size_t original_idx = missing_indices[i];
        results[original_idx] = persistent_results[i];
        out_field_maps[original_idx] = std::move(persistent_field_maps[i]);
    }
    return results;
}

std::vector<ErrorCode> MetaStorageBackendManager::GetAllFields(const KeyVector &keys,
                                                               FieldMapVec &out_field_maps) noexcept {
    if (!cache_backend_) {
        return persistent_backend_->GetAllFields(keys, out_field_maps);
    }

    std::vector<ErrorCode> results = cache_backend_->GetAllFields(keys, out_field_maps);
    if (recover_state_.load(std::memory_order_acquire) == RecoverState::kRunning) {
        return results;
    }

    KeyTypeVec missing_keys;
    std::vector<size_t> missing_indices;
    for (size_t i = 0; i < keys.size(); ++i) {
        if (results[i] == EC_NOENT) {
            missing_keys.push_back(keys[i]);
            missing_indices.push_back(i);
        }
    }
    if (missing_keys.empty()) {
        return results;
    }

    FieldMapVec persistent_field_maps;
    std::vector<ErrorCode> persistent_results = persistent_backend_->GetAllFields(missing_keys, persistent_field_maps);
    if (missing_keys.size() != persistent_field_maps.size()) {
        KVCM_LOG_ERROR("persistent_field_maps size[%lu] mismatch missing_keys's[%lu]",
                       persistent_field_maps.size(),
                       missing_keys.size());
        for (size_t i = 0; i < missing_keys.size(); ++i) {
            results[missing_indices[i]] = EC_ERROR;
        }
        return results;
    }
    for (size_t i = 0; i < missing_keys.size(); ++i) {
        const size_t original_idx = missing_indices[i];
        results[original_idx] = persistent_results[i];
        out_field_maps[original_idx] = std::move(persistent_field_maps[i]);
    }
    return results;
}

std::vector<ErrorCode> MetaStorageBackendManager::GetLocations(RequestContext *request_context,
                                                               const KeyVector &keys,
                                                               LocationMapVector &out_location_maps) noexcept {
    FieldMapVec field_maps;
    std::vector<ErrorCode> results = GetAllFields(keys, field_maps);

    out_location_maps.clear();
    out_location_maps.resize(keys.size());
    const int64_t begin_us = TimestampUtil::GetCurrentTimeUs();
    for (size_t i = 0; i < keys.size(); ++i) {
        if (results[i] != EC_OK) {
            continue;
        }
        for (const auto &field_kv : field_maps[i]) {
            const std::string &field_name = field_kv.first;
            const std::string &field_value = field_kv.second;
            if (field_name.compare(0, LOCATION_PREFIX.size(), LOCATION_PREFIX) != 0 || field_value.empty()) {
                continue;
            }
            CacheLocation location;
            if (!location.FromJsonString(field_value)) {
                KVCM_LOG_ERROR("deserialize CacheLocation failed, key[%ld] field[%s]", keys[i], field_name.c_str());
                results[i] = EC_CORRUPTION;
                out_location_maps[i].clear();
                break;
            }
            out_location_maps[i].emplace(location.id(), std::move(location));
        }
    }
    SetIndexDeserializeTimeUs(request_context, TimestampUtil::GetCurrentTimeUs() - begin_us);
    return results;
}

std::vector<std::vector<ErrorCode>> MetaStorageBackendManager::GetLocations(RequestContext *request_context,
                                                                            const KeyVector &keys,
                                                                            const LocationIdsPerKey &location_ids,
                                                                            LocationsPerKey &out_locations) noexcept {
    assert(keys.size() == location_ids.size());
    std::vector<std::vector<ErrorCode>> results(keys.size());
    out_locations.clear();
    out_locations.resize(keys.size());
    if (keys.empty()) {
        return results;
    }

    // Build per-key field name vectors from location ids.
    std::vector<std::vector<std::string>> field_names_vec(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        field_names_vec[i].reserve(location_ids[i].size());
        for (const auto &id : location_ids[i]) {
            field_names_vec[i].emplace_back(MakeLocationFieldName(id));
        }
    }

    // Delegate to the per-key field names Get which already handles
    // cache-first with persistent Recover fallback.
    FieldMapVec field_maps;
    std::vector<ErrorCode> get_results = Get(keys, field_names_vec, field_maps);

    // Deserialize the returned field values into CacheLocations.
    const int64_t begin_us = TimestampUtil::GetCurrentTimeUs();
    for (size_t i = 0; i < keys.size(); ++i) {
        if (get_results[i] != EC_OK) {
            results[i].assign(location_ids[i].size(), get_results[i]);
            continue;
        }
        results[i].assign(location_ids[i].size(), EC_OK);
        out_locations[i].resize(location_ids[i].size());
        const auto &fields = field_maps[i];
        for (size_t j = 0; j < location_ids[i].size(); ++j) {
            const std::string &field_name = field_names_vec[i][j];
            auto it = fields.find(field_name);
            // Empty value is a tombstone left by historical Delete(batch, location_ids)
            // writes that overwrote the field with "" instead of deleting it.
            if (it == fields.end() || it->second.empty()) {
                results[i][j] = EC_NOENT;
                continue;
            }
            CacheLocation location;
            if (!location.FromJsonString(it->second)) {
                KVCM_LOG_ERROR(
                    "deserialize CacheLocation failed, key[%ld] location_id[%s]", keys[i], location_ids[i][j].c_str());
                results[i][j] = EC_CORRUPTION;
                continue;
            }
            out_locations[i][j] = std::move(location);
        }
    }
    SetIndexDeserializeTimeUs(request_context, TimestampUtil::GetCurrentTimeUs() - begin_us);
    return results;
}

std::vector<ErrorCode> MetaStorageBackendManager::GetLocationIds(const KeyVector &keys,
                                                                 LocationIdsPerKey &out_location_ids) noexcept {
    out_location_ids.clear();
    if (keys.empty()) {
        return {};
    }

    std::vector<std::vector<std::string>> field_names_vec;
    std::vector<ErrorCode> results;
    if (cache_backend_) {
        results = cache_backend_->GetFieldNamesWithPrefix(keys, LOCATION_PREFIX, field_names_vec);
        if (recover_state_.load(std::memory_order_acquire) != RecoverState::kRunning) {
            KeyTypeVec missing_keys;
            std::vector<size_t> missing_indices;
            for (size_t i = 0; i < keys.size(); ++i) {
                if (results[i] == EC_NOENT) {
                    missing_keys.push_back(keys[i]);
                    missing_indices.push_back(i);
                }
            }
            if (!missing_keys.empty()) {
                std::vector<std::vector<std::string>> persistent_field_names;
                std::vector<ErrorCode> persistent_results =
                    persistent_backend_->GetFieldNamesWithPrefix(missing_keys, LOCATION_PREFIX, persistent_field_names);
                if (missing_keys.size() != persistent_field_names.size()) {
                    KVCM_LOG_ERROR("persistent_field_names size[%lu] mismatch missing_keys's[%lu]",
                                   persistent_field_names.size(),
                                   missing_keys.size());
                    for (size_t i = 0; i < missing_keys.size(); ++i) {
                        results[missing_indices[i]] = EC_ERROR;
                    }
                    return results;
                }
                for (size_t i = 0; i < missing_keys.size(); ++i) {
                    const size_t original_idx = missing_indices[i];
                    results[original_idx] = persistent_results[i];
                    field_names_vec[original_idx] = std::move(persistent_field_names[i]);
                }
            }
        }
    } else {
        results = persistent_backend_->GetFieldNamesWithPrefix(keys, LOCATION_PREFIX, field_names_vec);
    }

    // Strip LOCATION_PREFIX from field names to produce location ids.
    out_location_ids.resize(keys.size());
    const size_t prefix_len = LOCATION_PREFIX.size();
    for (size_t i = 0; i < keys.size(); ++i) {
        if (results[i] != EC_OK || field_names_vec[i].empty()) {
            continue;
        }
        out_location_ids[i].reserve(field_names_vec[i].size());
        for (auto &field_name : field_names_vec[i]) {
            if (field_name.size() > prefix_len) {
                out_location_ids[i].emplace_back(field_name.substr(prefix_len));
            }
        }
    }
    return results;
}

std::vector<ErrorCode> MetaStorageBackendManager::Exists(const KeyVector &keys,
                                                         std::vector<bool> &out_is_exist_vec) noexcept {
    if (!cache_backend_) {
        return persistent_backend_->Exists(keys, out_is_exist_vec);
    }
    std::vector<ErrorCode> results = cache_backend_->Exists(keys, out_is_exist_vec);
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
    std::vector<ErrorCode> persistent_results = persistent_backend_->Exists(missing_keys, persistent_exists);
    if (missing_keys.size() != persistent_exists.size()) {
        KVCM_LOG_ERROR(
            "persistent_exists size[%lu] mismatch missing_keys's[%lu]", persistent_exists.size(), missing_keys.size());
        for (size_t i = 0; i < missing_keys.size(); ++i) {
            results[missing_indices[i]] = EC_ERROR;
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

ErrorCode MetaStorageBackendManager::ListKeys(const std::string &cursor,
                                              const int64_t limit,
                                              std::string &out_next_cursor,
                                              KeyTypeVec &out_keys) noexcept {
    if (cache_backend_ && recover_state_.load(std::memory_order_acquire) == RecoverState::kRunning) {
        return cache_backend_->ListKeys(cursor, limit, out_next_cursor, out_keys);
    }
    return persistent_backend_->ListKeys(cursor, limit, out_next_cursor, out_keys);
}

ErrorCode MetaStorageBackendManager::RandomSample(const int64_t count, KeyTypeVec &out_keys) noexcept {
    if (cache_backend_ && recover_state_.load(std::memory_order_acquire) == RecoverState::kRunning) {
        return cache_backend_->RandomSample(count, out_keys);
    }
    return persistent_backend_->RandomSample(count, out_keys);
}

ErrorCode MetaStorageBackendManager::SampleReclaimKeys(const int64_t count, KeyTypeVec &out_keys) noexcept {
    if (count <= 0) {
        return EC_OK;
    }
    // Until recover finishes, the cache backend has not seen every key yet,
    // so we still go to persistent to avoid biased reclamation. In single-
    // backend mode (no cache) we always go to persistent.
    if (cache_backend_ && recover_state_.load(std::memory_order_acquire) == RecoverState::kRunning) {
        return cache_backend_->SampleReclaimKeys(count, out_keys);
    }
    return persistent_backend_->SampleReclaimKeys(count, out_keys);
}

ErrorCode MetaStorageBackendManager::PutMetaData(const FieldMap &field_maps) noexcept {
    return persistent_backend_->PutMetaData(field_maps);
}

ErrorCode MetaStorageBackendManager::GetMetaData(FieldMap &field_maps) noexcept {
    return persistent_backend_->GetMetaData(field_maps);
}

size_t MetaStorageBackendManager::GetMemUsage() const noexcept {
    if (cache_backend_) {
        return cache_backend_->GetMemUsage();
    }
    return 0;
}

} // namespace kv_cache_manager
