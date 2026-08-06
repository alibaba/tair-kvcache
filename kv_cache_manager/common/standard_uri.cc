#include "kv_cache_manager/common/standard_uri.h"

#include <sstream>

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
    // Locate the end of authority before interpreting '@' or ':'. Delimiter
    // characters in the path/query belong to their values, not user-info or
    // host/port (for example callback URLs and email addresses).
    size_t pos_path_start = uri.find('/', authority_start);
    size_t pos_query_start = uri.find('?', authority_start);
    size_t host_end = std::min((pos_path_start != std::string::npos ? pos_path_start : uri.size()),
                               (pos_query_start != std::string::npos ? pos_query_start : uri.size()));
    size_t host_start = authority_start;
    size_t pos_at = uri.find('@', authority_start);
    if (pos_at != std::string::npos && pos_at < host_end) {
        user_info_ = uri.substr(authority_start, pos_at - authority_start);
        host_start = pos_at + 1; // hostname 开始位置
    }

    // 分离 hostname 和 port。直接在输入中定位，避免为每个 URI 先复制
    // 一份 host:port 临时字符串。
    size_t colon_pos = uri.find(':', host_start);
    if (colon_pos == std::string::npos || colon_pos >= host_end) {
        hostname_ = uri.substr(host_start, host_end - host_start);
    } else {
        hostname_ = uri.substr(host_start, colon_pos - host_start);
        int64_t tmp_port = 0;
        const char *port_begin = uri.data() + colon_pos + 1;
        const char *port_end = uri.data() + host_end;
        const auto [parsed_end, parse_ec] = std::from_chars(port_begin, port_end, tmp_port);
        if (port_begin == port_end || *port_begin == '-' || parse_ec != std::errc{} || parsed_end != port_end) {
            // Parse() is also used through the direct string constructor,
            // whose caller observes validity rather than the return value.
            // Do not leave a partially parsed object looking valid.
            protocol_.clear();
            user_info_.clear();
            hostname_.clear();
            port_ = 0;
            path_.clear();
            params_.clear();
            return false;
        } else {
            port_ = tmp_port;
        }
    }

    // 提取 path 和 query
    if (pos_path_start != std::string::npos &&
        (pos_query_start == std::string::npos || pos_path_start < pos_query_start)) {
        if (pos_query_start != std::string::npos && pos_path_start < pos_query_start) {
            path_ = uri.substr(pos_path_start, pos_query_start - pos_path_start);
            ParseParams(std::string_view(uri).substr(pos_query_start + 1));
        } else {
            path_ = uri.substr(pos_path_start);
        }
    } else if (pos_query_start != std::string::npos && pos_query_start < uri.size()) {
        ParseParams(std::string_view(uri).substr(pos_query_start + 1));
    }
    return true;
}

bool StandardUri::ParseParams(std::string_view uri_params) {
    size_t start = 0;
    while (start < uri_params.size()) {
        auto end = uri_params.find('&', start);
        if (end == std::string::npos) {
            end = uri_params.size();
        }
        auto eq_pos = uri_params.find('=', start);
        if (eq_pos != std::string::npos && eq_pos < end) {
            std::string key(uri_params.substr(start, eq_pos - start));
            std::string value(uri_params.substr(eq_pos + 1, end - eq_pos - 1));
            params_[key] = value;
        } else {
            // key但无value，value空字符串
            std::string key(uri_params.substr(start, end - start));
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

std::string StandardUri::ToUriStringWithExtraParam(const std::string &key, const std::string &value) const {
    if (!Valid() || key.empty() || HasParam(key)) {
        return "";
    }

    size_t estimated_size =
        protocol_.size() + user_info_.size() + hostname_.size() + path_.size() + key.size() + value.size() + 8;
    for (const auto &[param_key, param_value] : params_) {
        estimated_size += param_key.size() + param_value.size() + 2;
    }
    std::string result;
    result.reserve(estimated_size);
    result.append(protocol_).append("://");
    if (!user_info_.empty()) {
        result.append(user_info_).push_back('@');
    }
    result.append(hostname_);
    if (port_ > 0) {
        result.push_back(':');
        result.append(std::to_string(port_));
    }
    result.append(path_);
    result.push_back('?');

    bool first = true;
    bool extra_written = false;
    auto append_param = [&result, &first](const std::string &param_key, const std::string &param_value) {
        if (!first) {
            result.push_back('&');
        }
        result.append(param_key).push_back('=');
        result.append(param_value);
        first = false;
    };
    for (const auto &[param_key, param_value] : params_) {
        if (!extra_written && key < param_key) {
            append_param(key, value);
            extra_written = true;
        }
        append_param(param_key, param_value);
    }
    if (!extra_written) {
        append_param(key, value);
    }
    return result;
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
