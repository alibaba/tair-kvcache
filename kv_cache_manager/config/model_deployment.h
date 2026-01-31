#pragma once
#include <string>
#include <vector>

#include "kv_cache_manager/common/jsonizable.h"
namespace kv_cache_manager {
class ModelDeployment : public Jsonizable {
public:
    ~ModelDeployment() override;
    bool FromRapidValue(const rapidjson::Value &rapid_value) override;
    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override;
    bool ValidateRequiredFields(std::string &invalid_fields) const;
    bool operator==(const ModelDeployment &other) const;
    bool operator!=(const ModelDeployment &other) const;
    // Getter methods
    const std::string &model_name() const noexcept { return model_name_; }
    const std::string &dtype() const noexcept { return dtype_; }
    bool use_mla() const noexcept { return use_mla_; }
    int32_t tp_size() const noexcept { return tp_size_; }
    int32_t dp_size() const noexcept { return dp_size_; }
    const std::string &lora_name() const noexcept { return lora_name_; }
    int32_t pp_size() const noexcept { return pp_size_; }
    const std::string &extra() const noexcept { return extra_; }
    const std::string &user_data() const noexcept { return user_data_; }

    // Setter methods
    void set_model_name(const std::string &v) { model_name_ = v; }
    void set_dtype(const std::string &v) { dtype_ = v; }
    void set_use_mla(bool v) noexcept { use_mla_ = v; }
    void set_tp_size(int32_t v) noexcept { tp_size_ = v; }
    void set_dp_size(int32_t v) noexcept { dp_size_ = v; }
    void set_lora_name(const std::string &v) { lora_name_ = v; }
    void set_pp_size(int32_t pp_size) { pp_size_ = pp_size; }
    void set_extra(const std::string &v) { extra_ = v; }
    void set_user_data(const std::string &v) { user_data_ = v; }

private:
    std::string model_name_;
    std::string dtype_;
    bool use_mla_ = false;
    int32_t tp_size_ = 0;
    int32_t dp_size_ = 0;
    std::string lora_name_;
    int32_t pp_size_ = 0;
    std::string extra_;
    std::string user_data_;
};
} // namespace kv_cache_manager