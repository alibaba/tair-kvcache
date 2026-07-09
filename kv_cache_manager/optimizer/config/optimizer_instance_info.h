#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "kv_cache_manager/common/jsonizable.h"
#include "kv_cache_manager/config/instance_info.h"

namespace kv_cache_manager {

class OptimizerStateInfo : public Jsonizable {
public:
    OptimizerStateInfo() = default;
    OptimizerStateInfo(const std::string &full_location_spec_group_name,
                       const std::string &linear_location_spec_group_name)
        : full_location_spec_group_name_(full_location_spec_group_name)
        , linear_location_spec_group_name_(linear_location_spec_group_name) {}
    ~OptimizerStateInfo() override = default;

    bool FromRapidValue(const rapidjson::Value &rapid_value) override;
    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override;

    const std::string &full_location_spec_group_name() const { return full_location_spec_group_name_; }
    const std::string &linear_location_spec_group_name() const { return linear_location_spec_group_name_; }

    void set_full_location_spec_group_name(const std::string &v) { full_location_spec_group_name_ = v; }
    void set_linear_location_spec_group_name(const std::string &v) { linear_location_spec_group_name_ = v; }

private:
    std::string full_location_spec_group_name_;
    std::string linear_location_spec_group_name_;
};

class OptimizerInstanceInfo : public Jsonizable {
public:
    OptimizerInstanceInfo() = default;
    OptimizerInstanceInfo(const std::string &instance_group_name,
                          const std::string &instance_id,
                          int32_t block_size,
                          const std::vector<LocationSpecInfo> &location_spec_infos,
                          const std::vector<LocationSpecGroup> &location_spec_groups,
                          int32_t linear_step = 0,
                          const OptimizerStateInfo &optimizer_state_info = OptimizerStateInfo())
        : instance_group_name_(instance_group_name)
        , instance_id_(instance_id)
        , block_size_(block_size)
        , location_spec_infos_(location_spec_infos)
        , location_spec_groups_(location_spec_groups)
        , linear_step_(linear_step)
        , optimizer_state_info_(optimizer_state_info) {}
    ~OptimizerInstanceInfo() override = default;

    bool FromRapidValue(const rapidjson::Value &rapid_value) override;
    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override;

    const std::string &instance_group_name() const { return instance_group_name_; }
    const std::string &instance_id() const { return instance_id_; }
    int32_t block_size() const { return block_size_; }
    const std::vector<LocationSpecInfo> &location_spec_infos() const { return location_spec_infos_; }
    const std::vector<LocationSpecGroup> &location_spec_groups() const { return location_spec_groups_; }
    int32_t linear_step() const { return linear_step_; }
    const OptimizerStateInfo &optimizer_state_info() const { return optimizer_state_info_; }

    void set_instance_group_name(const std::string &v) { instance_group_name_ = v; }
    void set_instance_id(const std::string &v) { instance_id_ = v; }
    void set_block_size(int32_t v) { block_size_ = v; }
    void set_location_spec_infos(const std::vector<LocationSpecInfo> &v) { location_spec_infos_ = v; }
    void set_location_spec_groups(const std::vector<LocationSpecGroup> &v) { location_spec_groups_ = v; }
    void set_linear_step(int32_t v) { linear_step_ = v; }
    void set_optimizer_state_info(const OptimizerStateInfo &v) { optimizer_state_info_ = v; }

private:
    std::string instance_group_name_;
    std::string instance_id_;
    int32_t block_size_ = 0;
    std::vector<LocationSpecInfo> location_spec_infos_;
    std::vector<LocationSpecGroup> location_spec_groups_;
    int32_t linear_step_ = 0;
    OptimizerStateInfo optimizer_state_info_;
};

} // namespace kv_cache_manager
