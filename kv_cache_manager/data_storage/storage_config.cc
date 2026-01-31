#include "kv_cache_manager/data_storage/storage_config.h"

#include <sstream>

namespace kv_cache_manager {

std::string ThreeFSStorageSpec::ToString() const {
    std::ostringstream oss;
    oss << "cluster_name: " << cluster_name_ << ", mountpoint: " << mountpoint_ << ", root_dir: " << root_dir_;
    return oss.str();
}

bool ThreeFSStorageSpec::ValidateRequiredFields(std::string &invalid_fields) const {
    bool valid = true;
    std::string local_invalid_fields;
    // cluster_name未使用
    // if (cluster_name_.empty()) {
    //     valid = false;
    //     invalid_fields += "{cluster_name}";
    // }
    if (mountpoint_.empty()) {
        valid = false;
        local_invalid_fields += "{mountpoint}";
    }
    if (root_dir_.empty()) {
        valid = false;
        local_invalid_fields += "{root_dir}";
    }
    if (!valid) {
        invalid_fields += "{ThreeFSStorageSpec: " + local_invalid_fields + "}";
    }
    return valid;
}

std::string VcnsThreeFSStorageSpec::ToString() const {
    std::ostringstream oss;
    oss << "cluster_name: " << cluster_name_ << ", mountpoint: " << mountpoint_ << ", root_dir: " << root_dir_
        << ", remote_host:" << remote_host_ << ", remote_port:" << remote_port_
        << ", meta_storage_uri:" << meta_storage_uri_;
    return oss.str();
}

bool VcnsThreeFSStorageSpec::ValidateRequiredFields(std::string &invalid_fields) const {
    bool valid = true;
    std::string local_invalid_fields;
    if (mountpoint_.empty()) {
        valid = false;
        local_invalid_fields += "{mountpoint}";
    }
    if (root_dir_.empty()) {
        valid = false;
        local_invalid_fields += "{root_dir}";
    }
    if (remote_host_.empty()) {
        valid = false;
        local_invalid_fields += "{remote_host}";
    }
    if (remote_port_ <= 0) {
        valid = false;
        local_invalid_fields += "{remote_port}";
    }
    if (meta_storage_uri_.empty()) {
        valid = false;
        local_invalid_fields += "{meta_storage_uri}";
    }
    if (!valid) {
        invalid_fields += "{VcnsThreeFSStorageSpec: " + local_invalid_fields + "}";
    }

    return valid;
}

std::string MooncakeStorageSpec::ToString() const {
    std::ostringstream oss;
    oss << "local_hostname: " << local_hostname_ << ", metadata_conn_string: " << metadata_connstring_
        << ", protocol: " << protocol_ << ", rdma_device: " << rdma_device_
        << ", master_server_entry: " << master_server_entry_;
    return oss.str();
}

bool MooncakeStorageSpec::ValidateRequiredFields(std::string &invalid_fields) const {
    bool valid = true;
    std::string local_invalid_fields;
    if (local_hostname_.empty()) {
        valid = false;
        local_invalid_fields += "{local_hostname}";
    }
    if (metadata_connstring_.empty()) {
        valid = false;
        local_invalid_fields += "{metadata_connstring}";
    }
    if (protocol_.empty()) {
        valid = false;
        local_invalid_fields += "{protocol}";
    } else if (protocol_ == "rdma" && rdma_device_.empty()) {
        valid = false;
        local_invalid_fields += "{rdma_device}";
    }
    if (master_server_entry_.empty()) {
        valid = false;
        local_invalid_fields += "{master_server_entry}";
    }
    if (!valid) {
        invalid_fields += "{MooncakeStorageSpec: " + local_invalid_fields + "}";
    }
    return valid;
}
std::string TairMemPoolStorageSpec::ToString() const {
    std::ostringstream oss;
    oss << "enable_vipserver: " << enable_vipserver_ << ", domain: " << domain_ << ", timeout: " << timeout_;
    return oss.str();
}
bool TairMemPoolStorageSpec::ValidateRequiredFields(std::string &invalid_fields) const {
    bool valid = true;
    std::string local_invalid_fields;
    if (domain_.empty()) {
        valid = false;
        local_invalid_fields += "{domain}";
    }

    if (!valid) {
        invalid_fields += "{TairMemPoolStorageSpec: " + local_invalid_fields + "}";
    }
    return valid;
}
std::string NfsStorageSpec::ToString() const {
    std::ostringstream oss;
    oss << "root_path: " << root_path_;
    oss << " , key_count_per_file: " << key_count_per_file_;
    return oss.str();
}
bool NfsStorageSpec::ValidateRequiredFields(std::string &invalid_fields) const {
    bool valid = true;
    std::string local_invalid_fields;
    if (root_path_.empty()) {
        valid = false;
        local_invalid_fields += "{root_path}";
    }
    if (!valid) {
        invalid_fields += "{NfsStorageSpec: " + local_invalid_fields + "}";
    }
    return valid;
}
std::string StorageConfig::ToString() const {
    std::ostringstream oss;
    oss << "type: " << kv_cache_manager::ToString(type_) << ", global_unique_name: " << global_unique_name_
        << ", storage_spec: " << (storage_spec_ ? storage_spec_->ToString() : "null");
    return oss.str();
}

std::string ToString(const DataStorageType &type) {
    switch (type) {
    case DataStorageType::DATA_STORAGE_TYPE_UNKNOWN:
        return "unknown";
    case DataStorageType::DATA_STORAGE_TYPE_HF3FS:
        return "hf3fs";
    case DataStorageType::DATA_STORAGE_TYPE_VCNS_HF3FS:
        return "vcns_hf3fs";
    case DataStorageType::DATA_STORAGE_TYPE_MOONCAKE:
        return "mooncake";
    case DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL:
        return "pace";
    case DataStorageType::DATA_STORAGE_TYPE_NFS:
        return "file";
    default:
        return "unrecognized";
    }
}

DataStorageType ToDataStorageType(const std::string &type) {
    if (type == "hf3fs") {
        return DataStorageType::DATA_STORAGE_TYPE_HF3FS;
    } else if (type == "vcns_hf3fs") {
        return DataStorageType::DATA_STORAGE_TYPE_VCNS_HF3FS;
    } else if (type == "mooncake") {
        return DataStorageType::DATA_STORAGE_TYPE_MOONCAKE;
    } else if (type == "pace") {
        return DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL;
    } else if (type == "file") {
        return DataStorageType::DATA_STORAGE_TYPE_NFS;
    } else {
        return DataStorageType::DATA_STORAGE_TYPE_UNKNOWN;
    }
}

// ThreeFSStorageSpec
bool ThreeFSStorageSpec::FromRapidValue(const rapidjson::Value &rapid_value) {
    KVCM_JSON_GET_MACRO(rapid_value, "cluster_name", cluster_name_);
    KVCM_JSON_GET_MACRO(rapid_value, "mountpoint", mountpoint_);
    KVCM_JSON_GET_MACRO(rapid_value, "root_dir", root_dir_);
    KVCM_JSON_GET_MACRO(rapid_value, "key_count_per_file", key_count_per_file_);
    return true;
}

void ThreeFSStorageSpec::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept {
    Put(writer, "cluster_name", cluster_name_);
    Put(writer, "mountpoint", mountpoint_);
    Put(writer, "root_dir", root_dir_);
    Put(writer, "key_count_per_file", key_count_per_file_);
}

// VcnsThreeFSStorageSpec
bool VcnsThreeFSStorageSpec::FromRapidValue(const rapidjson::Value &rapid_value) {
    KVCM_JSON_GET_MACRO(rapid_value, "cluster_name", cluster_name_);
    KVCM_JSON_GET_MACRO(rapid_value, "mountpoint", mountpoint_);
    KVCM_JSON_GET_MACRO(rapid_value, "root_dir", root_dir_);
    KVCM_JSON_GET_MACRO(rapid_value, "key_count_per_file", key_count_per_file_);
    KVCM_JSON_GET_MACRO(rapid_value, "remote_host", remote_host_);
    KVCM_JSON_GET_MACRO(rapid_value, "remote_port", remote_port_);
    KVCM_JSON_GET_MACRO(rapid_value, "meta_storage_uri", meta_storage_uri_);
    return true;
}

void VcnsThreeFSStorageSpec::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept {
    Put(writer, "cluster_name", cluster_name_);
    Put(writer, "mountpoint", mountpoint_);
    Put(writer, "root_dir", root_dir_);
    Put(writer, "key_count_per_file", key_count_per_file_);
    Put(writer, "remote_host", remote_host_);
    Put(writer, "remote_port", remote_port_);
    Put(writer, "meta_storage_uri", meta_storage_uri_);
}

// MooncakeStorageSpec
bool MooncakeStorageSpec::FromRapidValue(const rapidjson::Value &rapid_value) {
    KVCM_JSON_GET_MACRO(rapid_value, "local_hostname", local_hostname_);
    KVCM_JSON_GET_MACRO(rapid_value, "metadata_connstring", metadata_connstring_);
    KVCM_JSON_GET_MACRO(rapid_value, "protocol", protocol_);
    KVCM_JSON_GET_MACRO(rapid_value, "rdma_device", rdma_device_);
    KVCM_JSON_GET_MACRO(rapid_value, "master_server_entry", master_server_entry_);
    return true;
}

void MooncakeStorageSpec::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept {
    Put(writer, "local_hostname", local_hostname_);
    Put(writer, "metadata_connstring", metadata_connstring_);
    Put(writer, "protocol", protocol_);
    Put(writer, "rdma_device", rdma_device_);
    Put(writer, "master_server_entry", master_server_entry_);
}

// TairMemPoolStorageSpec
bool TairMemPoolStorageSpec::FromRapidValue(const rapidjson::Value &rapid_value) {
    KVCM_JSON_GET_MACRO(rapid_value, "enable_vipserver", enable_vipserver_);
    KVCM_JSON_GET_MACRO(rapid_value, "domain", domain_);
    KVCM_JSON_GET_MACRO(rapid_value, "timeout", timeout_);
    return true;
}

void TairMemPoolStorageSpec::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept {
    Put(writer, "enable_vipserver", enable_vipserver_);
    Put(writer, "domain", domain_);
    Put(writer, "timeout", timeout_);
}

bool NfsStorageSpec::FromRapidValue(const rapidjson::Value &rapid_value) {
    KVCM_JSON_GET_MACRO(rapid_value, "root_path", root_path_);
    KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "key_count_per_file", key_count_per_file_, 1);
    return true;
}

void NfsStorageSpec::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept {
    Put(writer, "root_path", root_path_);
    Put(writer, "key_count_per_file", key_count_per_file_);
}

bool StorageConfig::FromRapidValue(const rapidjson::Value &rapid_value) {
    std::string type_str;
    KVCM_JSON_GET_MACRO(rapid_value, "type", type_str);
    type_ = ToDataStorageType(type_str);
    KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "is_available", is_available_, true);
    KVCM_JSON_GET_MACRO(rapid_value, "global_unique_name", global_unique_name_);
    if (type_ == DataStorageType::DATA_STORAGE_TYPE_NFS) {
        auto tmp = std::make_shared<NfsStorageSpec>();
        KVCM_JSON_GET_MACRO(rapid_value, "storage_spec", tmp);
        storage_spec_ = tmp;
    } else if (type_ == DataStorageType::DATA_STORAGE_TYPE_HF3FS) {
        auto tmp = std::make_shared<ThreeFSStorageSpec>();
        KVCM_JSON_GET_MACRO(rapid_value, "storage_spec", tmp);
        storage_spec_ = tmp;
    } else if (type_ == DataStorageType::DATA_STORAGE_TYPE_VCNS_HF3FS) {
        auto tmp = std::make_shared<VcnsThreeFSStorageSpec>();
        KVCM_JSON_GET_MACRO(rapid_value, "storage_spec", tmp);
        storage_spec_ = tmp;
    } else if (type_ == DataStorageType::DATA_STORAGE_TYPE_MOONCAKE) {
        auto tmp = std::make_shared<MooncakeStorageSpec>();
        KVCM_JSON_GET_MACRO(rapid_value, "storage_spec", tmp);
        storage_spec_ = tmp;
    } else if (type_ == DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL) {
        auto tmp = std::make_shared<TairMemPoolStorageSpec>();
        KVCM_JSON_GET_MACRO(rapid_value, "storage_spec", tmp);
        storage_spec_ = tmp;
    } else {
        storage_spec_ = nullptr; // 对未知或未支持类型，设为空
    }
    return true;
}

void StorageConfig::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept {
    Put(writer, "type", kv_cache_manager::ToString(type_));
    Put(writer, "is_available", is_available_);
    Put(writer, "global_unique_name", global_unique_name_);
    Put(writer, "storage_spec", storage_spec_);
}

bool StorageConfig::ValidateRequiredFields(std::string &invalid_fields) const {
    bool valid = true;
    std::string local_invalid_fields;
    if (global_unique_name_.empty()) {
        valid = false;
        local_invalid_fields += "{global_unique_name}";
    }
    if (type_ == DataStorageType::DATA_STORAGE_TYPE_UNKNOWN || storage_spec_ == nullptr) {
        valid = false;
        local_invalid_fields += "{storage_spec}";
    }
    if (storage_spec_ && !storage_spec_->ValidateRequiredFields(local_invalid_fields)) {
        valid = false;
    }
    if (!valid) {
        invalid_fields += "{StorageConfig: " + local_invalid_fields + "}";
    }
    return valid;
}
} // namespace kv_cache_manager
