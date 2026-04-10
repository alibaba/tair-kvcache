#include "kv_cache_manager/meta/meta_cached_backend.h"

#include "kv_cache_manager/common/error_code.h"
#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/config/meta_storage_backend_config.h"
#include "kv_cache_manager/meta/common.h"
#include "kv_cache_manager/meta/meta_local_backend.h"
#include "kv_cache_manager/meta/meta_redis_backend.h"

namespace kv_cache_manager {

MetaCachedBackend::~MetaCachedBackend() {
    is_closed_.store(true, std::memory_order_release);
    if (recover_thread_.joinable()) {
        recover_thread_.join();
    }
}

std::string MetaCachedBackend::GetStorageType() noexcept { return META_CACHED_BACKEND_TYPE_STR; }

std::unique_ptr<MetaStorageBackend>
MetaCachedBackend::CreatePersistentBackend(const std::string &instance_id,
                                           const std::shared_ptr<MetaStorageBackendConfig> &config) const {
    const std::string &storage_type = config->GetStorageType();
    std::unique_ptr<MetaStorageBackend> backend;
    if (storage_type == META_REDIS_BACKEND_TYPE_STR) {
        backend = std::make_unique<MetaRedisBackend>();
    } else {
        KVCM_LOG_ERROR("unsupported persistent type[%s], only redis allowed", storage_type.c_str());
        return nullptr;
    }
    if (backend->Init(instance_id, config) != EC_OK) {
        KVCM_LOG_ERROR("init persistent backend failed, instance[%s]", instance_id.c_str());
        return nullptr;
    }
    return backend;
}

std::unique_ptr<MetaLocalBaseBackend>
MetaCachedBackend::CreateLocalBackend(const std::string &instance_id,
                                      const std::shared_ptr<MetaStorageBackendConfig> &config) const {
    const std::string &storage_type = config->GetStorageType();
    std::unique_ptr<MetaLocalBaseBackend> backend;
    if (storage_type == META_LOCAL_BACKEND_TYPE_STR) {
        backend = std::make_unique<MetaLocalBackend>();
    } else {
        KVCM_LOG_ERROR("unsupported cache backend type[%s], must be a local-base type", storage_type.c_str());
        return nullptr;
    }
    if (backend->Init(instance_id, config) != EC_OK) {
        KVCM_LOG_ERROR(
            "failed to init cache backend type[%s], instance[%s]", storage_type.c_str(), instance_id.c_str());
        return nullptr;
    }
    return backend;
}

ErrorCode MetaCachedBackend::Init(const std::string &instance_id,
                                  const std::shared_ptr<MetaStorageBackendConfig> &config) noexcept {
    if (instance_id.empty()) {
        KVCM_LOG_ERROR("fail to init meta cached backend, invalid empty instance id");
        return EC_BADARGS;
    }
    if (!config) {
        KVCM_LOG_ERROR("fail to init meta cached backend, invalid nullptr config");
        return EC_BADARGS;
    }
    instance_id_ = instance_id;
    config_ = config;

    // Parse persistent_type and cache_type from storage_uri, falling back to defaults.
    std::string persistent_type = META_REDIS_BACKEND_TYPE_STR;
    std::string cache_type = META_LOCAL_BACKEND_TYPE_STR;
    const std::string &storage_uri = config->GetStorageUri();
    if (!storage_uri.empty()) {
        StandardUri uri = StandardUri::FromUri(storage_uri);
        if (uri.Valid()) {
            std::string persistent_type_param = uri.GetParam("persistent_type");
            if (!persistent_type_param.empty()) {
                persistent_type = persistent_type_param;
            }
            std::string cache_type_param = uri.GetParam("cache_type");
            if (!cache_type_param.empty()) {
                cache_type = cache_type_param;
            }
        } else {
            KVCM_LOG_ERROR("invalid storage uri[%s]", storage_uri.c_str());
            return EC_BADARGS;
        }
    }

    // create persistent backend
    auto persistent_config = std::make_shared<MetaStorageBackendConfig>(persistent_type);
    persistent_config->SetStorageUri(storage_uri);
    persistent_backend_ = CreatePersistentBackend(instance_id, persistent_config);
    if (!persistent_backend_) {
        KVCM_LOG_ERROR("init persistent backend failed, instance[%s]", instance_id.c_str());
        return EC_ERROR;
    }

    // create local backend
    auto cache_config = std::make_shared<MetaStorageBackendConfig>(cache_type);
    cache_config->SetStorageUri(storage_uri);
    local_backend_ = CreateLocalBackend(instance_id, cache_config);
    if (!local_backend_) {
        KVCM_LOG_ERROR("init cache backend failed, instance[%s]", instance_id.c_str());
        return EC_ERROR;
    }

    KVCM_LOG_INFO("cached backend init ok, instance[%s] cache[%s] persistent[%s]",
                  instance_id_.c_str(),
                  cache_type.c_str(),
                  persistent_type.c_str());
    return EC_OK;
}

ErrorCode MetaCachedBackend::Open() noexcept {
    ErrorCode persistent_open_ec = persistent_backend_->Open();
    if (persistent_open_ec != EC_OK) {
        KVCM_LOG_ERROR("open persistent failed, instance[%s] ec[%d]", instance_id_.c_str(), persistent_open_ec);
        return persistent_open_ec;
    }
    ErrorCode cache_open_ec = local_backend_->Open();
    if (cache_open_ec != EC_OK) {
        KVCM_LOG_ERROR("open cache failed, instance[%s] ec[%d]", instance_id_.c_str(), cache_open_ec);
        return cache_open_ec;
    }

    // Start async recover: background thread scans persistent and backfills into local.
    recover_state_.store(RecoverState::kRecover, std::memory_order_release);
    recover_thread_ = std::thread(&MetaCachedBackend::AsyncRecoverTask, this);

    KVCM_LOG_INFO("meta cached backend open successfully, instance[%s], async recover started", instance_id_.c_str());
    return EC_OK;
}

ErrorCode MetaCachedBackend::Close() noexcept {
    is_closed_.store(true, std::memory_order_release);
    if (recover_thread_.joinable()) {
        recover_thread_.join();
    }

    ErrorCode cache_ec = EC_OK;
    ErrorCode persistent_ec = EC_OK;
    if (local_backend_) {
        cache_ec = local_backend_->Close();
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
    KVCM_LOG_INFO("cached backend closed, instance[%s]", instance_id_.c_str());
    return EC_OK;
}

void MetaCachedBackend::AsyncRecoverTask() noexcept {
    KVCM_LOG_INFO("meta cached backend async recover started, instance[%s]", instance_id_.c_str());
    constexpr int64_t SCAN_BATCH_SIZE = 1000;
    constexpr int MAX_CONSECUTIVE_FAILURES = 3;
    std::string cursor = SCAN_BASE_CURSOR;
    int64_t total_backfilled_keys = 0;
    int consecutive_failures = 0;
    std::string next_cursor;
    KeyTypeVec scanned_keys;
    FieldMapVec field_maps;
    do {
        if (is_closed_.load(std::memory_order_acquire)) {
            KVCM_LOG_INFO("meta cached backend async recover aborted due to close, instance[%s]", instance_id_.c_str());
            return;
        }

        scanned_keys.clear();
        field_maps.clear();
        ErrorCode scan_ec = persistent_backend_->ListKeys(cursor, SCAN_BATCH_SIZE, next_cursor, scanned_keys);
        if (scan_ec != EC_OK) {
            ++consecutive_failures;
            KVCM_LOG_ERROR("async recover scan failed, instance[%s] cursor[%s] ec[%d] attempt[%d/%d]",
                           instance_id_.c_str(),
                           cursor.c_str(),
                           scan_ec,
                           consecutive_failures,
                           MAX_CONSECUTIVE_FAILURES);
            if (consecutive_failures >= MAX_CONSECUTIVE_FAILURES) {
                KVCM_LOG_ERROR("async recover giving up after %d consecutive failures, "
                               "forcing transition to Running, instance[%s]",
                               MAX_CONSECUTIVE_FAILURES,
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
            int64_t backfilled = BackfillKeysToLocal(scanned_keys, field_maps, get_error_codes);
            total_backfilled_keys += backfilled;
        }
        cursor = next_cursor;
    } while (cursor != SCAN_BASE_CURSOR);

    if (consecutive_failures == 0) {
        KVCM_LOG_INFO("async recover completed instance[%s] total_backfilled_keys[%ld]",
                      instance_id_.c_str(),
                      total_backfilled_keys);
    } else {
        KVCM_LOG_WARN("async recover failed instance[%s] total_backfilled_keys[%ld], forcing finish recover",
                      instance_id_.c_str(),
                      total_backfilled_keys);
    }
    recover_state_.store(RecoverState::kRunning, std::memory_order_release);
    {
        std::lock_guard<std::mutex> lock(deleted_keys_mutex_);
        deleted_keys_.clear();
    }
    return;
}

int64_t MetaCachedBackend::BackfillKeysToLocal(const KeyTypeVec &keys,
                                               const FieldMapVec &field_maps,
                                               const std::vector<ErrorCode> &get_error_codes) noexcept {
    // Lock to atomically check deleted_keys and PutIfAbsent, preventing race with Delete.
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

    std::vector<ErrorCode> put_results = local_backend_->PutIfAbsent(keys, field_maps, previous_error_codes);
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

void MetaCachedBackend::EnsureKeyInLocal(const KeyTypeVec &keys) noexcept {
    std::vector<bool> exists_vec;
    std::vector<ErrorCode> exists_results = local_backend_->Exists(keys, exists_vec);
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
    std::vector<ErrorCode> put_results = local_backend_->Put(missing_keys, field_maps, get_results);
    for (size_t i = 0; i < missing_keys.size(); ++i) {
        if (put_results[i] != EC_OK && put_results[i] != EC_NOENT) {
            KVCM_LOG_WARN("put key[%ld] into local failed, ec[%d]", missing_keys[i], put_results[i]);
        }
    }
}

std::vector<ErrorCode> MetaCachedBackend::Put(const KeyTypeVec &keys, const FieldMapVec &field_maps) noexcept {
    std::vector<ErrorCode> persistent_results = persistent_backend_->Put(keys, field_maps);
    return local_backend_->Put(keys, field_maps, persistent_results);
}

std::vector<ErrorCode> MetaCachedBackend::UpdateFields(const KeyTypeVec &keys, const FieldMapVec &field_maps) noexcept {
    if (recover_state_.load(std::memory_order_acquire) == RecoverState::kRecover) {
        EnsureKeyInLocal(keys);
    }
    std::vector<ErrorCode> persistent_results = persistent_backend_->UpdateFields(keys, field_maps);
    return local_backend_->UpdateFields(keys, field_maps, persistent_results);
}

std::vector<ErrorCode> MetaCachedBackend::Upsert(const KeyTypeVec &keys, const FieldMapVec &field_maps) noexcept {
    if (recover_state_.load(std::memory_order_acquire) == RecoverState::kRecover) {
        EnsureKeyInLocal(keys);
    }
    std::vector<ErrorCode> persistent_results = persistent_backend_->Upsert(keys, field_maps);
    return local_backend_->Upsert(keys, field_maps, persistent_results);
}

std::vector<ErrorCode> MetaCachedBackend::Delete(const KeyTypeVec &keys) noexcept {
    std::vector<ErrorCode> persistent_results = persistent_backend_->Delete(keys);

    if (recover_state_.load(std::memory_order_acquire) == RecoverState::kRecover) {
        // Lock to prevent race with backfill: record deleted keys and delete from
        // local atomically, so backfill cannot insert a key between these two steps.
        std::lock_guard<std::mutex> lock(deleted_keys_mutex_);
        for (const auto &key : keys) {
            deleted_keys_.insert(key);
        }
        return local_backend_->Delete(keys, persistent_results);
    }

    return local_backend_->Delete(keys, persistent_results);
}

std::vector<ErrorCode> MetaCachedBackend::Get(const KeyTypeVec &keys,
                                              const std::vector<std::string> &field_names,
                                              FieldMapVec &out_field_maps) noexcept {
    std::vector<ErrorCode> results = local_backend_->Get(keys, field_names, out_field_maps);
    if (recover_state_.load(std::memory_order_acquire) == RecoverState::kRunning) {
        return results;
    }

    KeyTypeVec missing_keys;
    std::vector<size_t> missing_indices;
    for (size_t i = 0; i < keys.size(); ++i) {
        if ((results[i] == EC_OK && out_field_maps[i].empty()) || results[i] == EC_NOENT) {
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
    for (size_t i = 0; i < missing_keys.size(); ++i) {
        size_t original_idx = missing_indices[i];
        results[original_idx] = persistent_results[i];
        out_field_maps[original_idx] = std::move(persistent_field_maps[i]);
    }
    return results;
}

std::vector<ErrorCode> MetaCachedBackend::GetAllFields(const KeyTypeVec &keys, FieldMapVec &out_field_maps) noexcept {
    std::vector<ErrorCode> results = local_backend_->GetAllFields(keys, out_field_maps);
    if (recover_state_.load(std::memory_order_acquire) == RecoverState::kRunning) {
        return results;
    }

    KeyTypeVec missing_keys;
    std::vector<size_t> missing_indices;
    for (size_t i = 0; i < keys.size(); ++i) {
        if ((results[i] == EC_OK && out_field_maps[i].empty()) || results[i] == EC_NOENT) {
            missing_keys.push_back(keys[i]);
            missing_indices.push_back(i);
        }
    }
    if (missing_keys.empty()) {
        return results;
    }

    FieldMapVec persistent_field_maps;
    std::vector<ErrorCode> persistent_results = persistent_backend_->GetAllFields(missing_keys, persistent_field_maps);
    for (size_t i = 0; i < missing_keys.size(); ++i) {
        size_t original_idx = missing_indices[i];
        results[original_idx] = persistent_results[i];
        out_field_maps[original_idx] = std::move(persistent_field_maps[i]);
    }
    return results;
}

std::vector<ErrorCode> MetaCachedBackend::Exists(const KeyTypeVec &keys, std::vector<bool> &out_is_exist_vec) noexcept {
    std::vector<ErrorCode> results = local_backend_->Exists(keys, out_is_exist_vec);
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
    for (size_t i = 0; i < missing_keys.size(); ++i) {
        size_t original_idx = missing_indices[i];
        results[original_idx] = persistent_results[i];
        out_is_exist_vec[original_idx] = persistent_exists[i];
    }
    return results;
}

ErrorCode MetaCachedBackend::ListKeys(const std::string &cursor,
                                      const int64_t limit,
                                      std::string &out_next_cursor,
                                      std::vector<KeyType> &out_keys) noexcept {
    if (recover_state_.load(std::memory_order_acquire) == RecoverState::kRunning) {
        return local_backend_->ListKeys(cursor, limit, out_next_cursor, out_keys);
    }
    return persistent_backend_->ListKeys(cursor, limit, out_next_cursor, out_keys);
}

ErrorCode MetaCachedBackend::RandomSample(const int64_t count, std::vector<KeyType> &out_keys) noexcept {
    if (recover_state_.load(std::memory_order_acquire) == RecoverState::kRunning) {
        return local_backend_->RandomSample(count, out_keys);
    }
    return persistent_backend_->RandomSample(count, out_keys);
}

ErrorCode MetaCachedBackend::SampleReclaimKeys(const int64_t count, std::vector<KeyType> &out_keys) noexcept {
    // Validate input parameters
    if (count <= 0) {
        return EC_OK;
    }

    // Route to appropriate backend based on recover state:
    // - If recover is complete (kRunning), use local backend for better performance
    // - If still recovering, use persistent backend (Redis) to ensure data consistency
    if (recover_state_.load(std::memory_order_acquire) == RecoverState::kRunning) {
        return local_backend_->SampleReclaimKeys(count, out_keys);
    } else {
        return persistent_backend_->SampleReclaimKeys(count, out_keys);
    }
}

ErrorCode MetaCachedBackend::PutMetaData(const FieldMap &field_maps) noexcept {
    return persistent_backend_->PutMetaData(field_maps);
}

ErrorCode MetaCachedBackend::GetMetaData(FieldMap &field_maps) noexcept {
    return persistent_backend_->GetMetaData(field_maps);
}

} // namespace kv_cache_manager
