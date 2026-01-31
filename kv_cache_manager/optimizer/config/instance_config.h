#pragma once
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include "kv_cache_manager/common/jsonizable.h"
#include "kv_cache_manager/optimizer/config/eviction_config.h"
#include "kv_cache_manager/optimizer/config/types.h"
namespace kv_cache_manager {

class OptInstanceConfig : public Jsonizable {
public:
    OptInstanceConfig() = default;
    ~OptInstanceConfig() override = default;
    bool FromRapidValue(const rapidjson::Value &rapid_value) override;
    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override;

public:
    [[nodiscard]] const std::string &instance_id() const { return instance_id_; }
    [[nodiscard]] int32_t block_size() const { return block_size_; }
    [[nodiscard]] const EvictionPolicyParam &eviction_policy_param() const { return eviction_policy_param_; }
    [[nodiscard]] const std::string &instance_group_name() const { return instance_group_name_; }
    [[nodiscard]] EvictionPolicyType eviction_policy_type() const { return eviction_policy_type_; }
    void set_instance_id(const std::string &id) { instance_id_ = id; }
    void set_block_size(int32_t size) { block_size_ = size; }
    void set_eviction_policy_param(const EvictionPolicyParam &params) { eviction_policy_param_ = params; }
    void set_instance_group_name(const std::string &name) { instance_group_name_ = name; }
    void set_eviction_policy_type(EvictionPolicyType type) { eviction_policy_type_ = type; }

private:
    std::string instance_group_name_;
    std::string instance_id_;
    int32_t block_size_ = 0;
    EvictionPolicyType eviction_policy_type_ = EvictionPolicyType::POLICY_UNSPECIFIED;
    EvictionPolicyParam eviction_policy_param_;
};
} // namespace kv_cache_manager