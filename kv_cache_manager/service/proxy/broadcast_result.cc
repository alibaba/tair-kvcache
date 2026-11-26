#include "kv_cache_manager/service/proxy/broadcast_result.h"

#include <map>
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

namespace kv_cache_manager {
bool MergeReplies(std::vector<proto::meta::CommonResponse> &replies, proto::meta::CommonResponse &out) {
    out.Swap(&replies.front());
    return true;
}
// Each partition can advertise different storage names. A name must still identify
// one configuration for clients that use a single storage SDK registry.
bool MergeReplies(std::vector<proto::meta::RegisterInstanceResponse> &replies,
                  proto::meta::RegisterInstanceResponse &out) {
    rapidjson::Document merged(rapidjson::kArrayType);
    std::map<std::string, size_t> names;
    for (const auto &reply : replies) {
        if (reply.extra_info() != replies.front().extra_info()) {
            return false;
        }
        rapidjson::Document configs;
        if (configs.Parse(reply.storage_configs().c_str()).HasParseError() || !configs.IsArray()) {
            return false;
        }
        for (const auto &config : configs.GetArray()) {
            if (!config.IsObject() || !config.HasMember("global_unique_name") ||
                !config["global_unique_name"].IsString()) {
                return false;
            }
            auto [it, fresh] = names.emplace(config["global_unique_name"].GetString(), merged.Size());
            if (fresh) {
                rapidjson::Value copy(config, merged.GetAllocator());
                merged.PushBack(copy, merged.GetAllocator());
            } else if (merged[static_cast<rapidjson::SizeType>(it->second)] != config) {
                return false;
            }
        }
    }
    rapidjson::StringBuffer buf;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buf);
    merged.Accept(writer);
    out.Swap(&replies.front());
    out.set_storage_configs(buf.GetString(), buf.GetSize());
    return true;
}

bool MergeReplies(std::vector<proto::meta::GetHostCacheStateResponse> &replies,
                  proto::meta::GetHostCacheStateResponse &out) {
    std::map<std::string, proto::meta::HostCacheMatch> hosts;
    for (auto &reply : replies) {
        for (const auto &host : reply.hosts()) {
            auto [it, fresh] = hosts.emplace(host.host_ip_port(), host);
            // The global registry may know the same host with no local blocks.
            if (!fresh && it->second.local() < host.local()) {
                it->second = host;
            }
        }
    }
    *out.mutable_header() = replies.front().header();
    for (auto &entry : hosts) {
        out.add_hosts()->Swap(&entry.second);
    }
    return true;
}

} // namespace kv_cache_manager
