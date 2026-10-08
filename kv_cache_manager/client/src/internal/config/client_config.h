#pragma once
#include <map>
#include <memory>
#include <string>

#include "kv_cache_manager/client/include/common.h"
#include "kv_cache_manager/common/jsonizable.h"
#include "kv_cache_manager/config/model_deployment.h"
#include "meta_channel_config.h"

namespace kv_cache_manager {
class SdkWrapperConfig;

class ClientConfig : public Jsonizable {
public:
    using LocationSpecInfoMap = std::map<std::string, int64_t>;
    using LocationSpecGroups = std::map<std::string, std::vector<std::string>>;
    bool FromRapidValue(const rapidjson::Value &rapid_value) override;
    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override;

    bool operator==(const ClientConfig &other) const;
    bool operator!=(const ClientConfig &other) const { return !(*this == other); }
    int32_t block_size() const { return block_size_; }
    const std::string &instance_group() const { return instance_group_; }
    const std::string &instance_id() const { return instance_id_; }
    const LocationSpecInfoMap &location_spec_infos() const { return location_spec_infos_; }
    const MetaChannelConfig &meta_channel_config() const { return meta_channel_config_; }
    const std::vector<std::string> &addresses() const { return addresses_; }
    std::shared_ptr<SdkWrapperConfig> sdk_wrapper_config() const { return sdk_wrapper_config_; }
    const ModelDeployment &model_deployment() const { return model_deployment_; }
    const LocationSpecGroups &location_spec_groups() const { return location_spec_groups_; }
    QueryType default_query_type() const { return static_cast<QueryType>(default_query_type_); }
    int32_t replication_workers() const { return replication_workers_; }
    uint64_t replication_max_buffer_bytes() const { return replication_max_buffer_bytes_; }
    uint64_t replication_max_pending_bytes() const { return replication_max_pending_bytes_; }
    uint64_t replication_node_bytes_per_second() const { return replication_node_bytes_per_second_; }
    uint32_t replication_max_age_ms() const { return replication_max_age_ms_; }
    bool auto_replicate() const { return auto_replicate_; }
    int32_t caller_node_refresh_seconds() const { return caller_node_refresh_seconds_; }

private:
    bool Check() const;

private:
    int32_t block_size_;
    std::string instance_group_;
    std::string instance_id_;
    LocationSpecInfoMap location_spec_infos_;
    std::vector<std::string> addresses_;
    MetaChannelConfig meta_channel_config_;
    std::shared_ptr<SdkWrapperConfig> sdk_wrapper_config_;
    ModelDeployment model_deployment_;
    LocationSpecGroups location_spec_groups_;
    int32_t default_query_type_{0};
    int32_t replication_workers_ = 2;
    uint64_t replication_max_buffer_bytes_ = 256 * 1024 * 1024;
    uint64_t replication_max_pending_bytes_ = 256 * 1024 * 1024;
    uint64_t replication_node_bytes_per_second_ = 0;
    uint32_t replication_max_age_ms_ = 30000;
    bool auto_replicate_ = false;
    // 0 refreshes the backend identity on every request. This is the safe
    // default for backends whose node id also identifies a data incarnation.
    int32_t caller_node_refresh_seconds_ = 0;
};

} // namespace kv_cache_manager
