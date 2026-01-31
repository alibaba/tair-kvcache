#include "kv_cache_manager/client/src/internal/config/sdk_config.h"

#include <algorithm>
#include <sstream>

#include "kv_cache_manager/common/logger.h"

namespace kv_cache_manager {

bool SdkTimeoutConfig::FromRapidValue(const rapidjson::Value &rapid_value) {
    KVCM_JSON_GET_MACRO(rapid_value, "put_timeout_ms", put_timeout_ms_);
    KVCM_JSON_GET_MACRO(rapid_value, "get_timeout_ms", get_timeout_ms_);
    return true;
}

void SdkTimeoutConfig::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept {
    Put(writer, "put_timeout_ms", put_timeout_ms_);
    Put(writer, "get_timeout_ms", get_timeout_ms_);
}

bool SdkTimeoutConfig::Validate() const { return put_timeout_ms_ > 0 && get_timeout_ms_ > 0; }

bool SdkBackendConfig::FromRapidValue(const rapidjson::Value &rapid_value) {
    auto type_itr = rapid_value.FindMember("type");
    if (type_itr != rapid_value.MemberEnd()) {
        if (type_itr->value.IsString()) {
            type_ = ToDataStorageType(type_itr->value.GetString());
        }
    }
    KVCM_JSON_GET_MACRO(rapid_value, "sdk_log_file_path", sdk_log_file_path_);
    KVCM_JSON_GET_MACRO(rapid_value, "sdk_log_level", sdk_log_level_);
    KVCM_JSON_GET_MACRO(rapid_value, "byte_size_per_block", byte_size_per_block_);
    return true;
}

void SdkBackendConfig::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept {
    Put(writer, "type", ToString(type_));
    Put(writer, "sdk_log_file_path", sdk_log_file_path_);
    Put(writer, "sdk_log_level", sdk_log_level_);
    Put(writer, "byte_size_per_block", byte_size_per_block_);
}

bool SdkBackendConfig::Validate() const { return DataStorageType::DATA_STORAGE_TYPE_UNKNOWN != type_; }

Hf3fsSdkConfig::Hf3fsSdkConfig() { set_type(DataStorageType::DATA_STORAGE_TYPE_HF3FS); }

bool Hf3fsSdkConfig::FromRapidValue(const rapidjson::Value &rapid_value) {
    KVCM_JSON_GET_MACRO(rapid_value, "mountpoint", mountpoint_);
    KVCM_JSON_GET_MACRO(rapid_value, "root_dir", root_dir_);
    KVCM_JSON_GET_MACRO(rapid_value, "read_iov_block_size", read_iov_block_size_);
    KVCM_JSON_GET_MACRO(rapid_value, "read_iov_size", read_iov_size_);
    KVCM_JSON_GET_MACRO(rapid_value, "write_iov_block_size", write_iov_block_size_);
    KVCM_JSON_GET_MACRO(rapid_value, "write_iov_size", write_iov_size_);
    return SdkBackendConfig::FromRapidValue(rapid_value);
}

void Hf3fsSdkConfig::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept {
    Put(writer, "mountpoint", mountpoint_);
    Put(writer, "root_dir", root_dir_);
    Put(writer, "read_iov_block_size", read_iov_block_size_);
    Put(writer, "read_iov_size", read_iov_size_);
    Put(writer, "write_iov_block_size", write_iov_block_size_);
    Put(writer, "write_iov_size", write_iov_size_);
    SdkBackendConfig::ToRapidWriter(writer);
}

bool Hf3fsSdkConfig::Validate() const {
    // TODO: add more validate
    return SdkBackendConfig::Validate();
}

std::string Hf3fsSdkConfig::ToString() const {
    std::ostringstream oss;
    oss << "mountpoint: " << mountpoint_ << ", root_dir: " << root_dir_
        << ", read_iov_block_size: " << read_iov_block_size_ << ", read_iov_size: " << read_iov_size_
        << ", write_iov_block_size: " << write_iov_block_size_ << ", write_iov_size: " << write_iov_size_;
    return oss.str();
}

MooncakeSdkConfig::MooncakeSdkConfig() { set_type(DataStorageType::DATA_STORAGE_TYPE_MOONCAKE); }

bool MooncakeSdkConfig::FromRapidValue(const rapidjson::Value &rapid_value) {
    KVCM_JSON_GET_MACRO(rapid_value, "location", location_);
    KVCM_JSON_GET_MACRO(rapid_value, "put_replica_num", put_replica_num_);
    return SdkBackendConfig::FromRapidValue(rapid_value);
}

void MooncakeSdkConfig::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept {
    Put(writer, "location", location_);
    Put(writer, "put_replica_num", put_replica_num_);
    SdkBackendConfig::ToRapidWriter(writer);
}

bool MooncakeSdkConfig::Validate() const {
    return !location_.empty() && put_replica_num_ > 0 && SdkBackendConfig::Validate();
}

std::string MooncakeSdkConfig::ToString() const {
    std::ostringstream oss;
    oss << "local_mem_ptr: " << local_mem_ptr_ << ", local_buffer_size:" << local_buffer_size_
        << ", location: " << location_ << ", put_replica_num:" << put_replica_num_
        << ", set_self_location_spec_name:" << self_location_spec_name_;
    return oss.str();
}

TairMempoolSdkConfig::TairMempoolSdkConfig() { set_type(DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL); }

bool TairMempoolSdkConfig::FromRapidValue(const rapidjson::Value &rapid_value) {
    return SdkBackendConfig::FromRapidValue(rapid_value);
};

void TairMempoolSdkConfig::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept {
    SdkBackendConfig::ToRapidWriter(writer);
}

NfsSdkConfig::NfsSdkConfig() { set_type(DataStorageType::DATA_STORAGE_TYPE_NFS); }

bool NfsSdkConfig::FromRapidValue(const rapidjson::Value &rapid_value) {
    return SdkBackendConfig::FromRapidValue(rapid_value);
};

void NfsSdkConfig::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept {
    SdkBackendConfig::ToRapidWriter(writer);
}

bool SdkWrapperConfig::FromRapidValue(const rapidjson::Value &rapid_value) {
    KVCM_JSON_GET_MACRO(rapid_value, "thread_num", thread_num_);
    KVCM_JSON_GET_MACRO(rapid_value, "queue_size", queue_size_);
    auto sdk_backend_configs_iter = rapid_value.FindMember("sdk_backend_configs");
    if (sdk_backend_configs_iter != rapid_value.MemberEnd() && sdk_backend_configs_iter->value.IsArray()) {
        for (const auto &sdk_val : sdk_backend_configs_iter->value.GetArray()) {
            auto type_it = sdk_val.FindMember("type");
            if (type_it == sdk_val.MemberEnd() || !type_it->value.IsString()) {
                continue;
            }
            DataStorageType type = ToDataStorageType(type_it->value.GetString());
            std::shared_ptr<SdkBackendConfig> sdk_backend_config;
            switch (type) {
            case DataStorageType::DATA_STORAGE_TYPE_HF3FS:
                sdk_backend_config = std::make_shared<Hf3fsSdkConfig>();
                break;
            case DataStorageType::DATA_STORAGE_TYPE_VCNS_HF3FS:
                sdk_backend_config = std::make_shared<Hf3fsSdkConfig>();
                break;
            case DataStorageType::DATA_STORAGE_TYPE_MOONCAKE:
                sdk_backend_config = std::make_shared<MooncakeSdkConfig>();
                break;
            case DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL:
                sdk_backend_config = std::make_shared<TairMempoolSdkConfig>();
                break;
            case DataStorageType::DATA_STORAGE_TYPE_NFS:
                sdk_backend_config = std::make_shared<NfsSdkConfig>();
                break;
            default:
                break;
            }
            if (!sdk_backend_config)
                continue;

            if (!sdk_backend_config->FromRapidValue(sdk_val))
                continue;
            sdk_backend_configs_map_.insert_or_assign(type, sdk_backend_config);
        }
    }
    KVCM_JSON_GET_MACRO(rapid_value, "timeout_config", timeout_config_);
    return true;
}

void SdkWrapperConfig::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept {
    Put(writer, "thread_num", thread_num_);
    Put(writer, "queue_size", queue_size_);
    std::vector<const SdkBackendConfig *> sdk_backend_configs;
    sdk_backend_configs.reserve(sdk_backend_configs_map_.size());
    for (const auto &pair : sdk_backend_configs_map_) {
        sdk_backend_configs.push_back(pair.second.get());
    }
    Put(writer, "sdk_backend_configs", sdk_backend_configs);
    Put(writer, "timeout_config", timeout_config_);
}

bool SdkWrapperConfig::Validate() const {
    return std::all_of(sdk_backend_configs_map_.begin(),
                       sdk_backend_configs_map_.end(),
                       [](const auto &pair) {
                           auto sdk_backend_config = pair.second;
                           return sdk_backend_config->Validate();
                       }) &&
           timeout_config_.Validate();
}

std::shared_ptr<SdkBackendConfig> SdkWrapperConfig::GetSdkBackendConfig(DataStorageType type) const {
    if (sdk_backend_configs_map_.find(type) != sdk_backend_configs_map_.end()) {
        return sdk_backend_configs_map_.at(type);
    }
    return nullptr;
}

bool SdkWrapperConfig::operator==(const SdkWrapperConfig &other) const {
    // Compare the maps by checking each entry
    if (sdk_backend_configs_map_.size() != other.sdk_backend_configs_map_.size()) {
        return false;
    }

    for (const auto &pair : sdk_backend_configs_map_) {
        auto it = other.sdk_backend_configs_map_.find(pair.first);
        if (it == other.sdk_backend_configs_map_.end()) {
            return false;
        }

        // We need to compare the actual config objects
        // Since they're stored as shared_ptr<SdkBackendConfig>, we need to downcast and compare
        // For simplicity, we'll just compare the JSON representations
        if (pair.second && it->second) {
            if (pair.second->ToJsonString() != it->second->ToJsonString()) {
                return false;
            }
        } else if (pair.second || it->second) {
            return false;
        }
    }

    return thread_num_ == other.thread_num_ && queue_size_ == other.queue_size_ &&
           timeout_config_ == other.timeout_config_;
}

bool SdkWrapperConfig::operator!=(const SdkWrapperConfig &other) const { return !(*this == other); }

} // namespace kv_cache_manager
