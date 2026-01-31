#include "kv_cache_manager/common/jsonizable.h"

#include "kv_cache_manager/common/logger.h"

namespace kv_cache_manager {

Jsonizable::~Jsonizable() = default;

bool Jsonizable::Parse(rapidjson::Document &doc, const std::string &str) {
    doc.Parse(str.c_str());
    if (doc.HasParseError()) {
        KVCM_LOG_WARN("invalid json error code [%d], str [%s]", static_cast<int>(doc.GetParseError()), str.c_str());
        return false;
    }
    return true;
}

bool Jsonizable::FromJsonString(const std::string &str) {
    rapidjson::Document doc;
    if (!Parse(doc, str)) {
        return false;
    }
    return FromRapidValue(doc);
}

std::string Jsonizable::ToJsonString() const noexcept {
    rapidjson::StringBuffer s;
    rapidjson::Writer<rapidjson::StringBuffer> writer(s);
    writer.StartObject();
    ToRapidWriter(writer);
    writer.EndObject();
    return s.GetString();
}

} // namespace kv_cache_manager