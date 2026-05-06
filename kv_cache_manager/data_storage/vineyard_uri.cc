#include "kv_cache_manager/data_storage/vineyard_uri.h"

#include <sstream>

#include "kv_cache_manager/common/standard_uri.h"
#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"

namespace kv_cache_manager {

namespace {

constexpr const char *kScheme = "vineyard";
constexpr const char *kJsonAddrKey = "addr";
constexpr const char *kJsonTypeKey = "type";
constexpr const char *kDefaultMedium = "mem";

void AppendQuery(std::ostringstream &oss, const std::map<std::string, std::string> &params) {
    if (params.empty()) {
        return;
    }
    oss << '?';
    bool first = true;
    for (const auto &[k, v] : params) {
        if (!first) {
            oss << '&';
        }
        oss << k << '=' << v;
        first = false;
    }
}

} // anonymous namespace

std::string VineyardUri::Build(const std::string &host_ip_port,
                               const std::string &medium,
                               const std::map<std::string, std::string> &params) {
    std::ostringstream oss;
    oss << kScheme << "://" << host_ip_port;
    if (!medium.empty()) {
        oss << '/' << medium;
    }
    AppendQuery(oss, params);
    return oss.str();
}

bool VineyardUri::Parse(const std::string &uri,
                        std::string &out_host_ip_port,
                        std::string &out_medium,
                        std::map<std::string, std::string> &out_params) {
    StandardUri u;
    if (!u.Parse(uri)) {
        return false;
    }
    if (u.GetProtocol() != kScheme) {
        return false;
    }
    if (u.GetHostName().empty()) {
        return false;
    }
    out_host_ip_port = u.GetHostName();
    if (u.GetPort() > 0) {
        out_host_ip_port += ":" + std::to_string(u.GetPort());
    }
    out_medium.clear();
    const std::string &path = u.GetPath();
    if (path.size() > 1 && path[0] == '/') {
        out_medium = path.substr(1);
    } else if (!path.empty() && path[0] != '/') {
        out_medium = path;
    }
    // StandardUri does not expose its params map, but we can use GetParam per
    // known key. Since we don't know all keys in advance, fall back to
    // re-parsing the query segment ourselves to keep VineyardUri self-contained.
    out_params.clear();
    auto qpos = uri.find('?');
    if (qpos != std::string::npos && qpos + 1 < uri.size()) {
        const std::string query = uri.substr(qpos + 1);
        size_t start = 0;
        while (start < query.size()) {
            size_t end = query.find('&', start);
            if (end == std::string::npos) {
                end = query.size();
            }
            const std::string kv = query.substr(start, end - start);
            const auto eq = kv.find('=');
            if (eq != std::string::npos) {
                out_params[kv.substr(0, eq)] = kv.substr(eq + 1);
            } else if (!kv.empty()) {
                out_params[kv] = "";
            }
            start = end + 1;
        }
    }
    return true;
}

std::string VineyardUri::FromJson(const std::string &json) {
    rapidjson::Document doc;
    if (doc.Parse(json.c_str()).HasParseError() || !doc.IsObject()) {
        return {};
    }
    auto addr_it = doc.FindMember(kJsonAddrKey);
    if (addr_it == doc.MemberEnd() || !addr_it->value.IsString()) {
        return {};
    }
    const std::string host_ip_port = addr_it->value.GetString();
    std::string medium = kDefaultMedium;
    if (auto type_it = doc.FindMember(kJsonTypeKey); type_it != doc.MemberEnd() && type_it->value.IsString()) {
        medium = type_it->value.GetString();
    }
    std::map<std::string, std::string> params;
    for (auto m = doc.MemberBegin(); m != doc.MemberEnd(); ++m) {
        const std::string key = m->name.GetString();
        if (key == kJsonAddrKey || key == kJsonTypeKey) {
            continue;
        }
        if (m->value.IsString()) {
            params[key] = m->value.GetString();
        } else if (m->value.IsInt64()) {
            params[key] = std::to_string(m->value.GetInt64());
        } else if (m->value.IsUint64()) {
            params[key] = std::to_string(m->value.GetUint64());
        } else if (m->value.IsBool()) {
            params[key] = m->value.GetBool() ? "true" : "false";
        }
    }
    return Build(host_ip_port, medium, params);
}

std::string VineyardUri::ToJson(const std::string &uri) {
    std::string host_ip_port;
    std::string medium;
    std::map<std::string, std::string> params;
    if (!Parse(uri, host_ip_port, medium, params)) {
        return {};
    }
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    writer.StartObject();
    writer.Key(kJsonAddrKey);
    writer.String(host_ip_port.c_str());
    writer.Key(kJsonTypeKey);
    writer.String(medium.c_str());
    for (const auto &[k, v] : params) {
        writer.Key(k.c_str());
        writer.String(v.c_str());
    }
    writer.EndObject();
    return buffer.GetString();
}

} // namespace kv_cache_manager
