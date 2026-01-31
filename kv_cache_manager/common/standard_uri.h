#pragma once

#include <charconv>
#include <map>
#include <string>

namespace kv_cache_manager {

class StandardUri {
public:
    StandardUri() = default;
    explicit StandardUri(const std::string &Uri);

public:
    bool Parse(const std::string &Uri);
    std::string ToUriString() const;

    bool Valid() const { return !protocol_.empty(); }
    const std::string &GetProtocol() const { return protocol_; }
    const std::string &GetUserInfo() const { return user_info_; }
    const std::string &GetHostName() const { return hostname_; }
    int64_t GetPort() const { return port_; }
    const std::string &GetPath() const { return path_; }
    std::string GetParam(const std::string &key) const;
    template <typename T>
    void GetParamAs(const std::string &key, T &t) const {
        std::string val = GetParam(key);
        if (val.empty()) {
            return;
        }
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
    bool ParseParams(const std::string &Uri_params);

private:
    std::string protocol_;
    std::string user_info_;
    std::string hostname_;
    int64_t port_ = 0;
    std::string path_;
    std::map<std::string, std::string> params_;
};

} // namespace kv_cache_manager
