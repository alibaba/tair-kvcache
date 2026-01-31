#pragma once
#include <map>
#include <memory>
#include <string>

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
};

} // namespace kv_cache_manager