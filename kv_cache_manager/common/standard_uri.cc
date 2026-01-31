#include "kv_cache_manager/common/standard_uri.h"

#include "kv_cache_manager/common/string_util.h"

namespace kv_cache_manager {

StandardUri::StandardUri(const std::string &uri) { Parse(uri); }
std::string StandardUri::GetParam(const std::string &key) const {
    auto it = params_.find(key);
    return it == params_.end() ? "" : it->second;
}

void StandardUri::SetParam(const std::string &key, const std::string &value) { params_[key] = value; }

bool StandardUri::Parse(const std::string &uri) {
    protocol_.clear();
    user_info_.clear();
    hostname_.clear();
    port_ = 0;
    path_.clear();
    params_.clear();

    // 找协议分隔符 "://"
    auto pos_protocol_end = uri.find("://");
    if (pos_protocol_end == std::string::npos) {
        return false;
    }
    protocol_ = uri.substr(0, pos_protocol_end);

    size_t authority_start = pos_protocol_end + 3; // skip ://
    size_t host_start = authority_start;
    size_t pos_at = uri.find('@', authority_start);
    if (pos_at != std::string::npos) {
        user_info_ = uri.substr(authority_start, pos_at - authority_start);
        host_start = pos_at + 1; // hostname 开始位置
    }

    // 找 hostname 结束的位置（可能有 port）
    size_t pos_path_start = uri.find('/', authority_start);
    size_t pos_query_start = uri.find('?', authority_start);
    size_t host_end = std::min((pos_path_start != std::string::npos ? pos_path_start : uri.size()),
                               (pos_query_start != std::string::npos ? pos_query_start : uri.size())

    );
    // 分离 hostname 和 port
    std::string host_port = uri.substr(host_start, host_end - host_start);
    size_t colon_pos = host_port.find(':');
    if (colon_pos == std::string::npos) {
        hostname_ = host_port;
    } else {
        hostname_ = host_port.substr(0, colon_pos);
        std::string port_str = host_port.substr(colon_pos + 1);
        int64_t tmp_port = 0;
        if (!StringUtil::StrToInt64(port_str.c_str(), tmp_port)) {
            return false;
        } else {
            port_ = tmp_port;
        }
    }

    // 提取 path 和 query
    if (pos_path_start != std::string::npos && pos_path_start < uri.size()) {
        if (pos_query_start != std::string::npos && pos_path_start < pos_query_start) {
            path_ = uri.substr(pos_path_start, pos_query_start - pos_path_start);
            std::string query_str = uri.substr(pos_query_start + 1);
            ParseParams(query_str);
        } else {
            path_ = uri.substr(pos_path_start);
        }
    } else if (pos_query_start != std::string::npos && pos_query_start < uri.size()) {
        std::string query_str = uri.substr(pos_query_start + 1);
        ParseParams(query_str);
    }
    return true;
}

bool StandardUri::ParseParams(const std::string &uri_params) {
    auto start = 0;
    while (start < uri_params.size()) {
        auto end = uri_params.find('&', start);
        if (end == std::string::npos) {
            end = uri_params.size();
        }
        auto eq_pos = uri_params.find('=', start);
        if (eq_pos != std::string::npos && eq_pos < end) {
            std::string key = uri_params.substr(start, eq_pos - start);
            std::string value = uri_params.substr(eq_pos + 1, end - eq_pos - 1);
            params_[key] = value;
        } else {
            // key但无value，value空字符串
            std::string key = uri_params.substr(start, end - start);
            params_[key] = "";
        }
        start = end + 1;
    }
    return true;
}

std::string StandardUri::ToUriString() const {
    if (!Valid()) {
        return "";
    }
    std::ostringstream ss;
    ss << protocol_ << "://";
    if (!user_info_.empty()) {
        ss << user_info_ << "@";
    }
    ss << hostname_;
    if (port_ > 0) {
        ss << ":" << port_;
    }
    if (!path_.empty()) {
        ss << path_;
    }
    if (!params_.empty()) {
        ss << '?';
        bool first = true;
        for (const auto &kv : params_) {
            if (!first)
                ss << '&';
            ss << kv.first << '=' << kv.second;
            first = false;
        }
    }
    return ss.str();
}

StandardUri StandardUri::FromUri(const std::string &source) {
    StandardUri result;
    if (!result.Parse(source)) {
        return {};
    }
    return result;
}

std::string StandardUri::ToUri(const StandardUri &source) { return source.ToUriString(); }

} // namespace kv_cache_manager
