#include "tools/kvcm_swarm/scenario/json_config.h"

#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"

namespace kvcm_swarm {
namespace {

std::string JoinPath(std::string_view prefix, std::string_view suffix) {
    if (prefix.empty()) {
        return std::string(suffix);
    }
    if (suffix.empty()) {
        return std::string(prefix);
    }
    return std::string(prefix) + "." + std::string(suffix);
}

} // namespace

void JsonConfig::AppendJsonErrors(std::string_view path, std::vector<std::string> *errors) const {
    for (const std::string &error : json_errors()) {
        constexpr std::string_view kUnknownPrefix = "unknown field '";
        if (error.size() > kUnknownPrefix.size() && error.compare(0, kUnknownPrefix.size(), kUnknownPrefix) == 0 &&
            error.back() == '\'') {
            const std::string_view field(error.data() + kUnknownPrefix.size(),
                                         error.size() - kUnknownPrefix.size() - 1);
            errors->push_back("unknown configuration field: " + JoinPath(path, field));
            continue;
        }
        const size_t separator = error.find(':');
        if (separator == std::string::npos) {
            errors->push_back(path.empty() ? error : std::string(path) + ": " + error);
            continue;
        }
        errors->push_back(JoinPath(path, std::string_view(error).substr(0, separator)) + error.substr(separator));
    }
}

void JsonConfig::MergeJsonErrors(std::string_view path, const JsonConfig &child) {
    constexpr std::string_view kUnknownPrefix = "unknown field '";
    for (const std::string &error : child.json_errors()) {
        if (error.size() > kUnknownPrefix.size() && error.compare(0, kUnknownPrefix.size(), kUnknownPrefix) == 0 &&
            error.back() == '\'') {
            const std::string_view field(error.data() + kUnknownPrefix.size(),
                                         error.size() - kUnknownPrefix.size() - 1);
            AddJsonError("unknown field '" + JoinPath(path, field) + "'");
            continue;
        }
        const size_t separator = error.find(':');
        if (separator == std::string::npos) {
            AddJsonError(std::string(path) + ": " + error);
            continue;
        }
        AddJsonError(JoinPath(path, std::string_view(error).substr(0, separator)) + error.substr(separator));
    }
}

bool RawJsonObject::FromRapidValue(const rapidjson::Value &value) {
    if (!value.IsObject()) {
        return false;
    }
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    value.Accept(writer);
    json_.assign(buffer.GetString(), buffer.GetSize());
    return true;
}

} // namespace kvcm_swarm
