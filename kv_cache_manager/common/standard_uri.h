#pragma once

#include <charconv>
#include <map>
#include <string>
#include <string_view>

namespace kv_cache_manager {

class StandardUri {
public:
    StandardUri() = default;
    explicit StandardUri(const std::string &Uri);

public:
    bool Parse(const std::string &Uri);
    std::string ToUriString() const;
    // Serialize the URI as if one new query parameter had been inserted,
    // without cloning/mutating the parameter map. The output keeps the same
    // sorted canonical form as SetParam() followed by ToUriString().
    std::string ToUriStringWithExtraParam(const std::string &key, const std::string &value) const;

    bool Valid() const { return !protocol_.empty(); }
    const std::string &GetProtocol() const { return protocol_; }
    const std::string &GetUserInfo() const { return user_info_; }
    const std::string &GetHostName() const { return hostname_; }
    int64_t GetPort() const { return port_; }
    // Returns "hostname:port" if port > 0, otherwise just "hostname".
    std::string GetHostPort() const {
        if (port_ > 0) {
            return hostname_ + ":" + std::to_string(port_);
        }
        return hostname_;
    }
    bool HasParam(const std::string &key) const { return params_.find(key) != params_.end(); }
    bool HasParamWithPrefix(const std::string &prefix) const {
        if (prefix.empty()) {
            return false;
        }
        const auto it = params_.lower_bound(prefix);
        return it != params_.end() && it->first.compare(0, prefix.size(), prefix) == 0;
    }
    const std::string &GetPath() const { return path_; }
    std::string GetParam(const std::string &key) const;
    template <typename T>
    void GetParamAs(const std::string &key, T &t) const {
        const auto it = params_.find(key);
        if (it == params_.end() || it->second.empty()) {
            return;
        }
        const std::string &val = it->second;
        T result;
        auto [ptr, ec] = std::from_chars(val.data(), val.data() + val.size(), result);
        if (ec == std::errc{} && ptr == val.data() + val.size()) {
            t = result;
        }
    }

    void SetProtocol(const std::string &protocol) { protocol_ = protocol; }
    void SetUserInfo(const std::string &user_info) { user_info_ = user_info; }
    void SetHostName(const std::string &hostname) { hostname_ = hostname; }
    void SetPort(int64_t port) { port_ = port; }
    void SetPath(const std::string &path) { path_ = path; }
    void SetParam(const std::string &key, const std::string &value);

public:
    static StandardUri FromUri(const std::string &source);
    static std::string ToUri(const StandardUri &source);

private:
    bool ParseParams(std::string_view Uri_params);

private:
    std::string protocol_;
    std::string user_info_;
    std::string hostname_;
    int64_t port_ = 0;
    std::string path_;
    std::map<std::string, std::string> params_;
};

} // namespace kv_cache_manager
