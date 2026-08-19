#include "kv_cache_manager/common/jsonizable.h"

#include "kv_cache_manager/common/logger.h"
#include "rapidjson/error/en.h"

namespace kv_cache_manager {

Jsonizable::~Jsonizable() = default;

bool Jsonizable::Parse(rapidjson::Document &doc, const std::string &str) { return Parse(doc, str, nullptr); }

bool Jsonizable::Parse(rapidjson::Document &doc, const std::string &str, std::string *error) {
    doc.Parse(str.c_str(), str.size());
    if (doc.HasParseError()) {
        if (error != nullptr) {
            *error = std::string("JSON parse error at offset ") + std::to_string(doc.GetErrorOffset()) + ": " +
                     rapidjson::GetParseError_En(doc.GetParseError());
        }
        KVCM_LOG_WARN("invalid json error code [%d], str [%s]", static_cast<int>(doc.GetParseError()), str.c_str());
        return false;
    }
    return true;
}

bool Jsonizable::FromJsonString(const std::string &str) { return FromJsonString(str, nullptr); }

bool Jsonizable::FromJsonString(const std::string &str, std::string *error) {
    rapidjson::Document doc;
    if (!Parse(doc, str, error)) {
        return false;
    }
    return FromRapidValue(doc);
}

bool Jsonizable::CheckUnknownFields(const rapidjson::Value &rapid_value,
                                    std::initializer_list<std::string_view> known_fields,
                                    std::vector<std::string> *errors) {
    if (!rapid_value.IsObject()) {
        if (errors != nullptr) {
            errors->push_back("expected a JSON object");
        }
        return false;
    }
    bool ok = true;
    for (const auto &member : rapid_value.GetObject()) {
        const std::string_view name(member.name.GetString(), member.name.GetStringLength());
        bool known = false;
        for (const std::string_view field : known_fields) {
            if (field == name) {
                known = true;
                break;
            }
        }
        if (!known) {
            ok = false;
            if (errors != nullptr) {
                errors->push_back("unknown field '" + std::string(name) + "'");
            }
        }
    }
    return ok;
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
