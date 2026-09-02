#include "kv_cache_manager/config/cache_config.h"

#include "kv_cache_manager/common/standard_uri.h"

namespace kv_cache_manager {

namespace {

bool HasEnforcedMigrationAdmission(const MigrationConfig &migration_config) {
    for (const auto &strategy : migration_config.strategies()) {
        if (strategy != nullptr && strategy->admission().mode() == MigrationAdmissionMode::ENFORCE) {
            return true;
        }
    }
    return false;
}

// Recent-access is currently backed by the process-local metadata cache. Keep
// this configuration check in config (rather than depending on meta) and let
// the runtime readiness gate handle recovery/warmup state.
bool SupportsRecentAccessAdmission(const std::shared_ptr<MetaIndexerConfig> &meta_indexer_config) {
    if (meta_indexer_config == nullptr || meta_indexer_config->GetMetaStorageBackendConfig() == nullptr) {
        return false;
    }
    const auto &backend_config = meta_indexer_config->GetMetaStorageBackendConfig();
    if (backend_config->GetStorageType() == "local") {
        return true;
    }
    if (backend_config->GetStorageType() != "cached") {
        return false;
    }

    const std::string &storage_uri = backend_config->GetStorageUri();
    if (storage_uri.empty()) {
        // MetaStorageBackendManager defaults cached mode to redis/local.
        return true;
    }
    const StandardUri uri = StandardUri::FromUri(storage_uri);
    if (!uri.Valid()) {
        return false;
    }
    const std::string cache_type = uri.GetParam("cache_type");
    return cache_type.empty() || cache_type == "local";
}

} // namespace

CacheConfig::~CacheConfig() = default;

bool CacheConfig::FromRapidValue(const rapidjson::Value &rapid_value) {
    KVCM_JSON_GET_MACRO(rapid_value, "reclaim_strategy", reclaim_strategy_);
    KVCM_JSON_GET_MACRO(rapid_value, "cache_prefer_strategy", cache_prefer_strategy_);
    KVCM_JSON_GET_MACRO(rapid_value, "meta_indexer_config", meta_indexer_config_);
    KVCM_JSON_GET_MACRO(rapid_value, "migration_config", migration_config_);
    return true;
}

void CacheConfig::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept {
    Put(writer, "reclaim_strategy", reclaim_strategy_);
    Put(writer, "cache_prefer_strategy", cache_prefer_strategy_);
    Put(writer, "meta_indexer_config", meta_indexer_config_);
    Put(writer, "migration_config", migration_config_);
}
bool CacheConfig::ValidateRequiredFields(std::string &invalid_fields) const {
    bool valid = true;
    std::string local_invalid_fields;
    if (cache_prefer_strategy_ == CachePreferStrategy::CPS_UNSPECIFIED) {
        valid = false;
        local_invalid_fields += "{cache_prefer_strategy}";
    }
    if (reclaim_strategy_ == nullptr) {
        valid = false;
        local_invalid_fields += "{reclaim_strategy}";
    } else if (!reclaim_strategy_->ValidateRequiredFields(local_invalid_fields)) {
        valid = false;
    }
    if (meta_indexer_config_ == nullptr) {
        valid = false;
        local_invalid_fields += "{meta_indexer_config}";
    } else if (!meta_indexer_config_->ValidateRequiredFields(local_invalid_fields)) {
        valid = false;
    }
    if (!migration_config_.ValidateRequiredFields(local_invalid_fields)) {
        valid = false;
    }
    if (HasEnforcedMigrationAdmission(migration_config_) && !SupportsRecentAccessAdmission(meta_indexer_config_)) {
        valid = false;
        local_invalid_fields += "{migration_admission:{unsupported_meta_backend}}";
    }
    if (!valid) {
        invalid_fields += "{CacheConfig: " + local_invalid_fields + "}";
    }
    return valid;
}
} // namespace kv_cache_manager
