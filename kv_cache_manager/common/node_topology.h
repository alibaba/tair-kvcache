#pragma once

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iterator>
#include <map>
#include <mutex>
#include <string>
#include "rapidjson/document.h"

namespace kv_cache_manager {

// A deployment-owned, atomically replaced JSON file maps stable node UUIDs to
// supernodes. Both SDK and manager refresh the same schema; no network RPC is
// added to the read path. Malformed/unavailable input expires after 30 seconds.
class NodeTopology {
public:
    using ClockFn = std::function<int64_t()>;
    explicit NodeTopology(std::string path = {}, ClockFn clock = {}) : path_(std::move(path)), clock_(std::move(clock)) {
        if (path_.empty()) {
            const auto *env = std::getenv("KVCM_NODE_TOPOLOGY_FILE");
            if (env) path_ = env;
        }
    }
    std::string Resolve(const std::string &node) const {
        if (node.empty() || path_.empty()) return {};
        std::lock_guard<std::mutex> lock(mu_);
        const int64_t now = clock_ ? clock_() : std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        if (!loaded_ || now - last_attempt_ms_ >= 5000) {
            loaded_ = true;
            last_attempt_ms_ = now;
            std::map<std::string, std::string> fresh;
            if (Read(fresh)) {
                nodes_ = std::move(fresh);
                last_success_ms_ = now;
            }
        }
        if (now - last_success_ms_ >= 30000) return {};
        const auto it = nodes_.find(node);
        return it == nodes_.end() ? std::string{} : it->second;
    }
private:
    bool Read(std::map<std::string, std::string> &nodes) const {
        std::ifstream file(path_, std::ios::binary | std::ios::ate);
        if (!file || file.tellg() < 0 || file.tellg() > 8 * 1024 * 1024) return false;
        file.seekg(0);
        const std::string json((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        rapidjson::Document doc;
        doc.Parse(json.data(), json.size());
        if (doc.HasParseError() || !doc.IsObject() || !doc.HasMember("nodes") || !doc["nodes"].IsObject()) return false;
        for (const auto &entry : doc["nodes"].GetObject()) {
            if (!entry.value.IsString() || entry.name.GetStringLength() == 0 || entry.value.GetStringLength() == 0 ||
                !nodes.emplace(entry.name.GetString(), entry.value.GetString()).second) return false;
        }
        return true;
    }
    std::string path_;
    ClockFn clock_;
    mutable std::mutex mu_;
    mutable bool loaded_{false};
    mutable int64_t last_attempt_ms_{0}, last_success_ms_{0};
    mutable std::map<std::string, std::string> nodes_;
};

} // namespace kv_cache_manager
