#pragma once

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "kv_cache_manager/common/jsonizable.h"
#include "kv_cache_manager/config/model_deployment.h"

namespace kv_cache_manager {

class LocationSpecInfo : public Jsonizable {
public:
    LocationSpecInfo() = default;
    LocationSpecInfo(const std::string &name, int64_t size) : name_(name), size_(size) {}
    ~LocationSpecInfo() override = default;

    friend bool operator==(const LocationSpecInfo &lhs, const LocationSpecInfo &rhs) {
        return lhs.name_ == rhs.name_ && lhs.size_ == rhs.size_;
    }

    friend bool operator!=(const LocationSpecInfo &lhs, const LocationSpecInfo &rhs) { return !(lhs == rhs); }

    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override {
        Put(writer, "name", name_);
        Put(writer, "size", size_);
    }

    bool FromRapidValue(const rapidjson::Value &rapid_value) override {
        KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "name", name_, std::string(""));
        KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "size", size_, int64_t(0));
        return true;
    }
    bool ValidateRequiredFields(std::string &invalid_fields) const {
        bool valid = true;
        std::string local_invalid_fields;
        if (name_.empty()) {
            valid = false;
            local_invalid_fields += "{name}";
        }
        if (!valid) {
            invalid_fields += "{LocationSpecInfo: " + local_invalid_fields + "}";
        }
        return valid;
    }
    const std::string &name() const { return name_; }
    int64_t size() const { return size_; }

    void set_name(const std::string &name) { name_ = name; }
    void set_size(int64_t size) { size_ = size; }

private:
    std::string name_;
    int64_t size_ = 0;
};

class LocationSpecGroup : public Jsonizable {
public:
    LocationSpecGroup() = default;
    LocationSpecGroup(const std::string &name, const std::vector<std::string> &spec_names)
        : name_(name), spec_names_(spec_names) {
        SortSpecNames();
    }
    ~LocationSpecGroup() override = default;

    friend bool operator==(const LocationSpecGroup &lhs, const LocationSpecGroup &rhs) {
        return lhs.name_ == rhs.name_ && lhs.spec_names_ == rhs.spec_names_;
    }

    friend bool operator!=(const LocationSpecGroup &lhs, const LocationSpecGroup &rhs) { return !(lhs == rhs); }

    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override {
        Put(writer, "name", name_);
        Put(writer, "spec_names", spec_names_);
    }

    bool FromRapidValue(const rapidjson::Value &rapid_value) override {
        KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "name", name_, std::string(""));
        KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "spec_names", spec_names_, std::vector<std::string>());
        SortSpecNames();
        return true;
    }
    bool ValidateRequiredFields(std::string &invalid_fields) const {
        bool valid = true;
        std::string local_invalid_fields;
        if (name_.empty()) {
            valid = false;
            local_invalid_fields += "{name}";
        }
        if (!valid) {
            invalid_fields += "{LocationSpecGroup: " + local_invalid_fields + "}";
        }
        return valid;
    }
    const std::string &name() const { return name_; }
    const std::vector<std::string> &spec_names() const { return spec_names_; }

    void set_name(const std::string &name) { name_ = name; }
    void set_spec_names(const std::vector<std::string> &spec_names) {
        spec_names_ = spec_names;
        SortSpecNames();
    }

private:
    void SortSpecNames() {
        // to use binary search
        std::sort(spec_names_.begin(), spec_names_.end());
    }

private:
    std::string name_;
    std::vector<std::string> spec_names_;
};

class InstanceInfo : public Jsonizable {
public:
    InstanceInfo() = default;
    InstanceInfo(const std::string &quota_group_name,
                 const std::string &instance_group_name,
                 const std::string &instance_id,
                 int32_t block_size,
                 const std::vector<LocationSpecInfo> &location_spec_infos,
                 const ModelDeployment &model_deployment,
                 const std::vector<LocationSpecGroup> &location_spec_groups = {})
        : quota_group_name_(quota_group_name)
        , instance_group_name_(instance_group_name)
        , instance_id_(instance_id)
        , block_size_(block_size)
        , location_spec_infos_(location_spec_infos)
        , model_deployment_(model_deployment)
        , location_spec_groups_(location_spec_groups) {
        SortLocationSpecGroups();
    };
    ~InstanceInfo() override;

    bool FromRapidValue(const rapidjson::Value &rapid_value) override;

    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override;
    std::string ToString() const;

    const std::string &quota_group_name() const { return quota_group_name_; }
    const std::string &instance_group_name() const { return instance_group_name_; }
    const std::string &instance_id() const { return instance_id_; }
    int32_t block_size() const { return block_size_; }
    const std::vector<LocationSpecInfo> &location_spec_infos() const { return location_spec_infos_; }
    const ModelDeployment &model_deployment() const { return model_deployment_; }
    const std::vector<LocationSpecGroup> &location_spec_groups() const { return location_spec_groups_; }
    void set_quota_group_name(const std::string &quota_group_name) { quota_group_name_ = quota_group_name; }
    void set_instance_group_name(const std::string &instance_group_name) { instance_group_name_ = instance_group_name; }
    void set_instance_id(const std::string &instance_id) { instance_id_ = instance_id; }
    void set_block_size(int32_t block_size) { block_size_ = block_size; }
    void set_location_spec_infos(const std::vector<LocationSpecInfo> &location_spec_infos) {
        location_spec_infos_ = location_spec_infos;
    }
    void set_model_deployment(const ModelDeployment &model_deployment) { model_deployment_ = model_deployment; }
    void set_location_spec_groups(const std::vector<LocationSpecGroup> &location_spec_groups) {
        location_spec_groups_ = location_spec_groups;
        SortLocationSpecGroups();
    }

private:
    void SortLocationSpecGroups();

private:
    std::string quota_group_name_;
    std::string instance_group_name_;
    std::string instance_id_;
    int32_t block_size_;
    std::vector<LocationSpecInfo> location_spec_infos_;
    ModelDeployment model_deployment_;
    std::vector<LocationSpecGroup> location_spec_groups_;
};

using InstanceInfoConstPtr = std::shared_ptr<const InstanceInfo>;

} // namespace kv_cache_manager