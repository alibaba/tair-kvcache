
#include "kv_cache_manager/config/model_deployment.h"

namespace kv_cache_manager {

ModelDeployment::~ModelDeployment() = default;

bool ModelDeployment::operator==(const ModelDeployment &other) const {
    if (this == &other) {
        return true;
    }
    return model_name_ == other.model_name_ && dtype_ == other.dtype_ && use_mla_ == other.use_mla_ &&
           tp_size_ == other.tp_size_ && dp_size_ == other.dp_size_ && lora_name_ == other.lora_name_ &&
           pp_size_ == other.pp_size_ && extra_ == other.extra_ /* && user_data_ == other.user_data_*/;
}
bool ModelDeployment::operator!=(const ModelDeployment &other) const { return !((*this) == other); }

bool ModelDeployment::FromRapidValue(const rapidjson::Value &rapid_value) {
    KVCM_JSON_GET_MACRO(rapid_value, "model_name", model_name_);
    KVCM_JSON_GET_MACRO(rapid_value, "dtype", dtype_);
    KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "use_mla", use_mla_, use_mla_);
    KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "tp_size", tp_size_, tp_size_);
    KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "dp_size", dp_size_, dp_size_);
    KVCM_JSON_GET_MACRO(rapid_value, "lora_name", lora_name_);
    KVCM_JSON_GET_MACRO(rapid_value, "pp_size", pp_size_);
    KVCM_JSON_GET_MACRO(rapid_value, "extra", extra_);
    KVCM_JSON_GET_MACRO(rapid_value, "user_data", user_data_);
    return true;
}

void ModelDeployment::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept {
    Put(writer, "model_name", model_name_);
    Put(writer, "dtype", dtype_);
    Put(writer, "use_mla", use_mla_);
    Put(writer, "tp_size", tp_size_);
    Put(writer, "dp_size", dp_size_);
    Put(writer, "lora_name", lora_name_);
    Put(writer, "pp_size", pp_size_);
    Put(writer, "extra", extra_);
    Put(writer, "user_data", user_data_);
}
bool ModelDeployment::ValidateRequiredFields(std::string &invalid_fields) const {
    bool valid = true;
    std::string local_invalid_fields;
    if (model_name_.empty()) {
        valid = false;
        local_invalid_fields += "{model_name}";
    }
    if (dtype_.empty()) {
        valid = false;
        local_invalid_fields += "{dtype}";
    }
    if (!valid) {
        invalid_fields += "{ModelDeployment: " + local_invalid_fields + "}";
    }
    return valid;
}
} // namespace kv_cache_manager