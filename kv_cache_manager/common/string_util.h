#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace kv_cache_manager {

class StringUtil {
public:
    template <typename... Args>
    static std::string FormatString(const std::string &format, Args... args) {
        // This function came from a code snippet in stackoverflow under cc-by-1.0
        //   https://stackoverflow.com/questions/2342162/stdstring-formatting-like-sprintf

        // Disable format-security warning in this function.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-security"

        int size_s = std::snprintf(nullptr, 0, format.c_str(), args...) + 1; // Extra space for '\0'
        if (size_s <= 0) {
            throw std::runtime_error("Error during formatting.");
        }
        auto size = static_cast<size_t>(size_s);
        auto buf = std::make_unique<char[]>(size);
        std::snprintf(buf.get(), size, format.c_str(), args...);

#pragma GCC diagnostic pop
        return std::string(buf.get(), buf.get() + size - 1); // We don't want the '\0' inside
    }

    static bool StrToInt64(const char *str, int64_t &value) {
        if (NULL == str || *str == 0) {
            return false;
        }
        char *end_ptr = NULL;
        errno = 0;
        value = (int64_t)strtoll(str, &end_ptr, 10);
        if (errno == 0 && end_ptr && *end_ptr == 0) {
            return true;
        }
        return false;
    }
    static std::string GenerateRandomString(size_t length) {
        const std::string charset = "abcdefghijklmnopqrstuvwxyz0123456789";
        std::random_device rd;
        std::mt19937 generator(rd());
        std::uniform_int_distribution<> distribution(0, charset.size() - 1);
        std::string result;
        result.reserve(length);
        for (size_t i = 0; i < length; ++i) {
            result += charset[distribution(generator)];
        }
        return result;
    }

    inline static void Trim(std::string &s) {
        LefTrim(s);
        RightTrim(s);
    }
    inline static void LefTrim(std::string &s) {
        s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) { return !std::isspace(ch); }));
    }
    inline static void RightTrim(std::string &s) {
        s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(), s.end());
    }
    inline static void ToLower(std::string &str) {
        std::transform(str.begin(), str.end(), str.begin(), [](unsigned char c) { return std::tolower(c); });
    }

    inline static bool StartsWith(const std::string &str, const std::string &prefix) {
        return (str.size() >= prefix.size()) && (str.compare(0, prefix.size(), prefix) == 0);
    }

    inline static bool EndsWith(const std::string &str, const std::string &suffix) {
        size_t s1 = str.size();
        size_t s2 = suffix.size();
        return (s1 >= s2) && (str.compare(s1 - s2, s2, suffix) == 0);
    }

    static std::string Join(const std::vector<std::string> &parts, const std::string &sep) {
        if (parts.empty()) {
            return "";
        }
        std::string result = parts[0];
        result.reserve(parts.size() * (parts[0].size() + sep.size()));
        for (size_t i = 1; i < parts.size(); ++i) {
            result += sep;
            result += parts[i];
        }
        return result;
    }

    static std::vector<std::string> Split(const std::string &str, const std::string &sep) {
        std::vector<std::string> result;
        if (sep.empty()) {
            if (!str.empty()) {
                result.push_back(str);
            }
            return result;
        }

        size_t start = 0;
        while (true) {
            size_t pos = str.find(sep, start);
            if (pos == std::string::npos) {
                std::string token = str.substr(start);
                if (!token.empty()) {
                    result.push_back(token);
                }
                break;
            }
            std::string token = str.substr(start, pos - start);
            if (!token.empty()) {
                result.push_back(token);
            }
            start = pos + sep.size();
        }
        return result;
    }

    static std::string Uint64ToHex(uint64_t value) {
        std::stringstream ss;
        ss << std::hex << value;
        return ss.str();
    }
};

} // namespace kv_cache_manager
