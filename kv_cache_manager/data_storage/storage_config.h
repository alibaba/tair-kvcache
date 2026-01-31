#pragma once

#include "kv_cache_manager/common/jsonizable.h"

namespace kv_cache_manager {

enum class DataStorageType : uint8_t {
    DATA_STORAGE_TYPE_UNKNOWN = 0,
    DATA_STORAGE_TYPE_HF3FS = 1,
    DATA_STORAGE_TYPE_MOONCAKE = 2,
    DATA_STORAGE_TYPE_TAIR_MEMPOOL = 3,
    DATA_STORAGE_TYPE_NFS = 4,
    DATA_STORAGE_TYPE_VCNS_HF3FS = 5,
};

std::string ToString(const DataStorageType &type);

DataStorageType ToDataStorageType(const std::string &type);

class StorageSpec : public Jsonizable {
public:
    bool FromRapidValue(const rapidjson::Value &rapid_value) override { return false; }
    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override {}
    virtual bool ValidateRequiredFields(std::string &invalid_fields) const = 0;

    virtual std::string ToString() const = 0;
};

// TODO 怎么跟backend的名字也对不上呢
class ThreeFSStorageSpec : public StorageSpec {
public:
    bool FromRapidValue(const rapidjson::Value &rapid_value) override;
    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override;
    bool ValidateRequiredFields(std::string &invalid_fields) const;
    std::string ToString() const override;
    const std::string &cluster_name() const { return cluster_name_; }
    const std::string &mountpoint() const { return mountpoint_; }
    const std::string &root_dir() const { return root_dir_; }
    int32_t key_count_per_file() const { return key_count_per_file_; }
    bool touch_file_when_create() const { return touch_file_when_create_; }

    void set_cluster_name(const std::string &cluster_name) { cluster_name_ = cluster_name; }
    void set_mountpoint(const std::string &mountpoint) { mountpoint_ = mountpoint; }
    void set_root_dir(const std::string &root_dir) { root_dir_ = root_dir; }
    void set_key_count_per_file(const int32_t value) { key_count_per_file_ = value; }
    void set_touch_file_when_create(bool value) { touch_file_when_create_ = value; }

private:
    std::string cluster_name_;
    std::string mountpoint_;
    std::string root_dir_;
    int32_t key_count_per_file_ = 0;
    bool touch_file_when_create_ = false;
};

class VcnsThreeFSStorageSpec : public StorageSpec {
public:
    bool FromRapidValue(const rapidjson::Value &rapid_value) override;
    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override;
    bool ValidateRequiredFields(std::string &invalid_fields) const;
    std::string ToString() const override;
    const std::string &cluster_name() const { return cluster_name_; }
    const std::string &mountpoint() const { return mountpoint_; }
    const std::string &root_dir() const { return root_dir_; }
    int32_t key_count_per_file() const { return key_count_per_file_; }
    bool touch_file_when_create() const { return touch_file_when_create_; }
    std::string remote_host() const { return remote_host_; }
    int32_t remote_port() const { return remote_port_; }
    std::string meta_storage_uri() const { return meta_storage_uri_; }

    void set_cluster_name(const std::string &cluster_name) { cluster_name_ = cluster_name; }
    void set_mountpoint(const std::string &mountpoint) { mountpoint_ = mountpoint; }
    void set_root_dir(const std::string &root_dir) { root_dir_ = root_dir; }
    void set_key_count_per_file(const int32_t value) { key_count_per_file_ = value; }
    void set_touch_file_when_create(bool value) { touch_file_when_create_ = value; }
    void set_remote_host(const std::string &remote_host) { remote_host_ = remote_host; }
    void set_remote_port(const int32_t &remote_port) { remote_port_ = remote_port; }
    void set_meta_storage_uri(const std::string &meta_storage_uri) { meta_storage_uri_ = meta_storage_uri; }

private:
    std::string cluster_name_;
    std::string mountpoint_;
    std::string root_dir_;
    int32_t key_count_per_file_ = 0;
    bool touch_file_when_create_ = false;
    std::string remote_host_;
    int32_t remote_port_;
    std::string meta_storage_uri_;
};

class MooncakeStorageSpec : public StorageSpec {
public:
    bool FromRapidValue(const rapidjson::Value &rapid_value) override;
    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override;
    bool ValidateRequiredFields(std::string &invalid_fields) const;

    std::string ToString() const override;
    const std::string &local_hostname() const { return local_hostname_; }
    const std::string &metadata_connstring() const { return metadata_connstring_; }
    const std::string &protocol() const { return protocol_; }
    const std::string &rdma_device() const { return rdma_device_; }
    const std::string &master_server_entry() const { return master_server_entry_; }

    void set_local_hostname(const std::string &local_hostname) { local_hostname_ = local_hostname; }

    void set_metadata_connstring(const std::string &metadata_connstring) { metadata_connstring_ = metadata_connstring; }

    void set_protocol(const std::string &protocol) { protocol_ = protocol; }

    void set_rdma_device(const std::string &rdma_device) { rdma_device_ = rdma_device; }

    void set_master_server_entry(const std::string &master_server_entry) { master_server_entry_ = master_server_entry; }

private:
    std::string local_hostname_{"localhost"}; // (IP:Port),如果没有port,则选择默认端口12001
    std::string metadata_connstring_{"http://localhost:8080/metadata"}; // Connection string for metadata service
    std::string protocol_{"tcp"};                                       // Transfer protocol ("rdma" or "tcp")
    std::string rdma_device_;
    std::string master_server_entry_{"localhost:50051"}; // master server
};

class TairMemPoolStorageSpec : public StorageSpec {
public:
    bool FromRapidValue(const rapidjson::Value &rapid_value) override;
    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override;
    bool ValidateRequiredFields(std::string &invalid_fields) const override;
    std::string ToString() const override;
    const std::string &cluster_name() const { return cluster_name_; }
    const std::string &domain() const { return domain_; }
    int64_t timeout() const { return timeout_; }
    const bool enable_vipserver() const { return enable_vipserver_; }
    void set_domain(const std::string &domain) { domain_ = domain; }
    void set_cluster_name(const std::string &cluster_name) { cluster_name_ = cluster_name; }
    void set_timeout(int64_t timeout) { timeout_ = timeout; }
    void set_enable_vipserver(bool enable_vipserver) { enable_vipserver_ = enable_vipserver; }

private:
    std::string domain_;            // metaservice域名
    int64_t timeout_{0};            // 目前连接超时、请求超时时间的值一样
    bool enable_vipserver_ = false; // 是否使用vipserver
    std::string cluster_name_;      // TODO proto中没有这个字段并且未使用，考虑删除
};

class NfsStorageSpec : public StorageSpec {
public:
    bool FromRapidValue(const rapidjson::Value &rapid_value) override;
    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override;
    bool ValidateRequiredFields(std::string &invalid_fields) const override;

    std::string ToString() const override;
    const std::string &root_path() const { return root_path_; }
    void set_root_path(const std::string &root_path) { root_path_ = root_path; }
    int32_t key_count_per_file() const { return key_count_per_file_; }
    void set_key_count_per_file(int32_t value) { key_count_per_file_ = value; }

private:
    std::string root_path_;
    int32_t key_count_per_file_ = 0;
};

class StorageConfig : public Jsonizable {
public:
    StorageConfig() = default;
    StorageConfig(DataStorageType type,
                  const std::string &global_unique_name,
                  const std::shared_ptr<StorageSpec> &storage_spec)
        : type_(type), global_unique_name_(global_unique_name), storage_spec_(storage_spec) {}

    bool FromRapidValue(const rapidjson::Value &rapid_value) override;
    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override;
    bool ValidateRequiredFields(std::string &invalid_fields) const;

    std::string ToString() const;
    DataStorageType type() const { return type_; }
    const std::string &global_unique_name() const { return global_unique_name_; }
    const std::shared_ptr<StorageSpec> &storage_spec() const { return storage_spec_; }
    const bool check_storage_available_when_open() const { return check_storage_available_when_open_; }
    bool is_available() const { return is_available_; }
    void set_global_unique_name(const std::string &global_unique_name) { global_unique_name_ = global_unique_name; }
    void set_type(DataStorageType type) { type_ = type; }
    void set_storage_spec(const std::shared_ptr<StorageSpec> &storage_spec) { storage_spec_ = storage_spec; }
    void set_check_storage_available_when_open(bool value) { check_storage_available_when_open_ = value; }
    void set_is_available(bool is_available) { is_available_ = is_available; }

private:
    DataStorageType type_{DataStorageType::DATA_STORAGE_TYPE_UNKNOWN};
    std::string global_unique_name_;
    std::shared_ptr<StorageSpec> storage_spec_;
    bool check_storage_available_when_open_ = false;
    bool is_available_ = true; // for recover
};

} // namespace kv_cache_manager
