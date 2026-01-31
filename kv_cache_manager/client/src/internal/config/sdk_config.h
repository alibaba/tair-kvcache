#pragma once

#include <unordered_map>

#include "kv_cache_manager/common/jsonizable.h"
#include "kv_cache_manager/data_storage/storage_config.h"

namespace kv_cache_manager {
class SdkTimeoutConfig : public Jsonizable {
public:
    SdkTimeoutConfig() {}
    ~SdkTimeoutConfig() = default;

    bool Validate() const;

    bool FromRapidValue(const rapidjson::Value &rapid_value) override;
    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override;

    bool operator==(const SdkTimeoutConfig &other) const {
        return put_timeout_ms_ == other.put_timeout_ms_ && get_timeout_ms_ == other.get_timeout_ms_;
    }

    bool operator!=(const SdkTimeoutConfig &other) const { return !(*this == other); }

    int put_timeout_ms() const { return put_timeout_ms_; }
    int get_timeout_ms() const { return get_timeout_ms_; }

    void set_put_timeout_ms(int timeout_ms) { put_timeout_ms_ = timeout_ms; }
    void set_get_timeout_ms(int timeout_ms) { get_timeout_ms_ = timeout_ms; }

private:
    int put_timeout_ms_{2000};
    int get_timeout_ms_{2000};
};

class SdkBackendConfig : public Jsonizable {
public:
    bool FromRapidValue(const rapidjson::Value &rapid_value) override;
    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override;

    virtual bool Validate() const;

    bool operator==(const SdkBackendConfig &other) const {
        return type_ == other.type_ && sdk_log_file_path_ == other.sdk_log_file_path_ &&
               sdk_log_level_ == other.sdk_log_level_ && byte_size_per_block_ == other.byte_size_per_block_;
    }

    bool operator!=(const SdkBackendConfig &other) const { return !(*this == other); }

    DataStorageType type() const { return type_; }
    const std::string &sdk_log_file_path() const { return sdk_log_file_path_; }
    const std::string &sdk_log_level() const { return sdk_log_level_; }
    int64_t byte_size_per_block() const { return byte_size_per_block_; }

    void set_type(DataStorageType type) { type_ = type; }
    void set_sdk_log_file_path(const std::string &value) { sdk_log_file_path_ = value; }
    void set_sdk_log_level(const std::string &value) { sdk_log_level_ = value; }
    void set_byte_size_per_block(int64_t value) { byte_size_per_block_ = value; }

private:
    DataStorageType type_;
    std::string sdk_log_file_path_;
    std::string sdk_log_level_;
    int64_t byte_size_per_block_{0};
};

class Hf3fsSdkConfig : public SdkBackendConfig {
public:
    Hf3fsSdkConfig();
    bool FromRapidValue(const rapidjson::Value &rapid_value) override;
    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override;

    bool Validate() const override;
    std::string ToString() const;

    bool operator==(const Hf3fsSdkConfig &other) const {
        return SdkBackendConfig::operator==(other) && mountpoint_ == other.mountpoint_ &&
               root_dir_ == other.root_dir_ && read_iov_block_size_ == other.read_iov_block_size_ &&
               read_iov_size_ == other.read_iov_size_ && write_iov_block_size_ == other.write_iov_block_size_ &&
               write_iov_size_ == other.write_iov_size_;
    }

    bool operator!=(const Hf3fsSdkConfig &other) const { return !(*this == other); }

    std::string mountpoint() const { return mountpoint_; }
    std::string root_dir() const { return root_dir_; }
    size_t read_iov_block_size() const { return read_iov_block_size_; }
    size_t read_iov_size() const { return read_iov_size_; }
    size_t write_iov_block_size() const { return write_iov_block_size_; }
    size_t write_iov_size() const { return write_iov_size_; }

    void set_mountpoint(const std::string &mountpoint) { mountpoint_ = mountpoint; }
    void set_root_dir(const std::string &root_dir) { root_dir_ = root_dir; }
    void set_read_iov_block_size(size_t read_iov_block_size) { read_iov_block_size_ = read_iov_block_size; }
    void set_read_iov_size(size_t read_iov_size) { read_iov_size_ = read_iov_size; }
    void set_write_iov_block_size(size_t write_iov_block_size) { write_iov_block_size_ = write_iov_block_size; }
    void set_write_iov_size(size_t write_iov_size) { write_iov_size_ = write_iov_size; }

private:
    std::string mountpoint_{"/3fs/stage/3fs/"};
    std::string root_dir_{"kv_manager/"};

    size_t read_iov_block_size_{0};
    size_t read_iov_size_{1ULL << 32};        // 4GB
    size_t write_iov_block_size_{1ULL << 20}; // 1MB
    size_t write_iov_size_{1ULL << 32};       // 4GB
};

class MooncakeSdkConfig : public SdkBackendConfig {
public:
    MooncakeSdkConfig();
    bool FromRapidValue(const rapidjson::Value &rapid_value) override;
    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override;

    bool Validate() const override;
    std::string ToString() const;

    bool operator==(const MooncakeSdkConfig &other) const {
        // Note: We don't compare local_mem_ptr_ as it's a pointer value that may change
        return SdkBackendConfig::operator==(other) && local_buffer_size_ == other.local_buffer_size_ &&
               location_ == other.location_ && put_replica_num_ == other.put_replica_num_ &&
               self_location_spec_name_ == other.self_location_spec_name_;
    }

    bool operator!=(const MooncakeSdkConfig &other) const { return !(*this == other); }

    void *local_mem_ptr() const { return local_mem_ptr_; }
    size_t local_buffer_size() const { return local_buffer_size_; }
    const std::string &location() const { return location_; }
    size_t put_replica_num() const { return put_replica_num_; }
    const std::string &self_location_spec_name() const { return self_location_spec_name_; }

    void set_local_mem_ptr(void *local_mem_ptr) { local_mem_ptr_ = local_mem_ptr; }
    void set_local_buffer_size(size_t local_buffer_size) { local_buffer_size_ = local_buffer_size; }
    void set_location(const std::string &location) { location_ = location; }
    void set_put_replica_num(size_t put_replica_num) { put_replica_num_ = put_replica_num; }
    void set_self_location_spec_name(const std::string &self_location_spec_name) {
        self_location_spec_name_ = self_location_spec_name;
    }

private:
    void *local_mem_ptr_{nullptr};
    size_t local_buffer_size_{0};
    std::string location_{"*"};
    size_t put_replica_num_{1};
    std::string self_location_spec_name_{""};
};

class TairMempoolSdkConfig : public SdkBackendConfig {
public:
    TairMempoolSdkConfig();
    bool FromRapidValue(const rapidjson::Value &rapid_value) override;
    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override;

    bool operator==(const TairMempoolSdkConfig &other) const { return SdkBackendConfig::operator==(other); }

    bool operator!=(const TairMempoolSdkConfig &other) const { return !(*this == other); }
};

class NfsSdkConfig : public SdkBackendConfig {
public:
    NfsSdkConfig();
    bool FromRapidValue(const rapidjson::Value &rapid_value) override;
    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override;

    bool operator==(const NfsSdkConfig &other) const { return SdkBackendConfig::operator==(other); }

    bool operator!=(const NfsSdkConfig &other) const { return !(*this == other); }
};

class SdkWrapperConfig : public Jsonizable {
public:
    bool FromRapidValue(const rapidjson::Value &rapid_value) override;
    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override;

    bool Validate() const;

    std::shared_ptr<SdkBackendConfig> GetSdkBackendConfig(DataStorageType type) const;

    bool operator==(const SdkWrapperConfig &other) const;
    bool operator!=(const SdkWrapperConfig &other) const;

    size_t thread_num() const { return thread_num_; }
    size_t queue_size() const { return queue_size_; }
    const std::unordered_map<DataStorageType, std::shared_ptr<SdkBackendConfig>> &sdk_backend_configs_map() const {
        return sdk_backend_configs_map_;
    }
    const SdkTimeoutConfig &timeout_config() const { return timeout_config_; }

    void set_thread_num(size_t thread_num) { thread_num_ = thread_num; }
    void set_queue_size(size_t queue_size) { queue_size_ = queue_size; }
    void set_sdk_backend_configs_map(
        const std::unordered_map<DataStorageType, std::shared_ptr<SdkBackendConfig>> &sdk_backend_configs_map) {
        sdk_backend_configs_map_ = sdk_backend_configs_map;
    }
    void set_timeout_config(const SdkTimeoutConfig &timeout_config) { timeout_config_ = timeout_config; }

private:
    size_t thread_num_{8};
    size_t queue_size_{2000};
    // 为什么这里要默认加3个
    std::unordered_map<DataStorageType, std::shared_ptr<SdkBackendConfig>> sdk_backend_configs_map_ = {
        {DataStorageType::DATA_STORAGE_TYPE_HF3FS, std::make_shared<Hf3fsSdkConfig>()},
        {DataStorageType::DATA_STORAGE_TYPE_VCNS_HF3FS, std::make_shared<Hf3fsSdkConfig>()},
        {DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL, std::make_shared<TairMempoolSdkConfig>()},
        {DataStorageType::DATA_STORAGE_TYPE_NFS, std::make_shared<NfsSdkConfig>()}};
    SdkTimeoutConfig timeout_config_;
};

} // namespace kv_cache_manager