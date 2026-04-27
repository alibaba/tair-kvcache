#include "kv_cache_manager/data_storage/vineyard_storage_spec.h"

#include <sstream>

#include "kv_cache_manager/common/jsonizable.h"

namespace kv_cache_manager {

bool VineyardStorageSpec::FromRapidValue(const rapidjson::Value &rapid_value) {
    KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "cluster_name", cluster_name_, std::string(""));
    return true;
}

void VineyardStorageSpec::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept {
    Put(writer, "cluster_name", cluster_name_);
}

bool VineyardStorageSpec::ValidateRequiredFields(std::string &invalid_fields) const {
    if (cluster_name_.empty()) {
        invalid_fields += "{VineyardStorageSpec: {cluster_name}}";
        return false;
    }
    return true;
}

std::string VineyardStorageSpec::ToString() const {
    std::ostringstream oss;
    oss << "cluster_name: " << cluster_name_;
    return oss.str();
}

} // namespace kv_cache_manager
