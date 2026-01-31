#pragma once

#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"

// these macros is general but not efficient,
// please use rapidjson::Reader to more efficiently parse if needed
#define KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, key_str, value_ref, default_value)                                    \
    if (!Get(rapid_value, key_str, value_ref, default_value)) {                                                        \
        return false;                                                                                                  \
    }

#define KVCM_JSON_GET_MACRO(rapid_value, key_str, value_ref)                                                           \
    if (!Get(rapid_value, key_str, value_ref)) {                                                                       \
        return false;                                                                                                  \
    }

namespace kv_cache_manager {

class Jsonizable {
public:
    virtual bool FromJsonString(const std::string &str);
    virtual std::string ToJsonString() const noexcept;

    virtual bool FromRapidValue(const rapidjson::Value &rapid_value) { return false; };

    virtual void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept = 0;

    virtual ~Jsonizable();

    template <typename T>
    static bool FromJsonString(const std::string &str, T &value);

    template <typename T, typename A>
    static std::string ToJsonString(const std::vector<T, A> &values) noexcept;

    template <typename T, typename C, typename A>
    static std::string ToJsonString(const std::map<std::string, T, C, A> &values) noexcept;

protected:
    static bool Parse(rapidjson::Document &doc, const std::string &str);

    // read
    template <typename T>
    static bool Get(const rapidjson::Value &rapid_value, const std::string &key, T &value, const T &default_value);

    template <typename T>
    static bool Get(const rapidjson::Value &rapid_value, const std::string &key, T &value);

    template <typename T, typename A>
    static bool Get(const rapidjson::Value &rapid_value, const std::string &key, std::vector<T, A> &values);

    template <typename T, typename C, typename A>
    static bool
    Get(const rapidjson::Value &rapid_value, const std::string &key, std::map<std::string, T, C, A> &values);

    // write
    template <typename T>
    static void
    Put(rapidjson::Writer<rapidjson::StringBuffer> &writer, const std::string &key, const T &value) noexcept;

    template <typename T, typename A>
    static void Put(rapidjson::Writer<rapidjson::StringBuffer> &writer,
                    const std::string &key,
                    const std::vector<T, A> &values) noexcept;

    template <typename T, typename C, typename A>
    static void Put(rapidjson::Writer<rapidjson::StringBuffer> &writer,
                    const std::string &key,
                    const std::map<std::string, T, C, A> &values) noexcept;

private:
    // read
    template <typename T>
    static std::enable_if_t<std::is_base_of_v<Jsonizable, std::decay_t<T>>, bool>
    FromRapidValue(const rapidjson::Value &rapid_value, T &value);

    inline static bool FromRapidValue(const rapidjson::Value &rapid_value, bool &value);

    template <typename T>
    inline static std::enable_if_t<std::is_integral_v<std::decay_t<T>> && std::is_signed_v<std::decay_t<T>>, bool>
    FromRapidValue(const rapidjson::Value &rapid_value, T &value);

    template <typename T>
    inline static std::enable_if_t<std::is_integral_v<std::decay_t<T>> && std::is_unsigned_v<std::decay_t<T>>, bool>
    FromRapidValue(const rapidjson::Value &rapid_value, T &value);

    template <typename T>
    inline static std::enable_if_t<std::is_floating_point_v<std::decay_t<T>>, bool>
    FromRapidValue(const rapidjson::Value &rapid_value, T &value);

    inline static bool FromRapidValue(const rapidjson::Value &rapid_value, std::string &value);

    template <typename T, typename A>
    static bool FromRapidValue(const rapidjson::Value &rapid_value, std::vector<T, A> &values);

    template <typename T, typename C, typename A>
    static bool FromRapidValue(const rapidjson::Value &rapid_value, std::map<std::string, T, C, A> &values);

    template <typename T>
    inline static bool FromRapidValue(const rapidjson::Value &rapid_value, T *&value);

    template <typename T>
    inline static bool FromRapidValue(const rapidjson::Value &rapid_value, std::shared_ptr<T> &value);

    template <typename T>
    inline static std::enable_if_t<std::is_enum_v<std::decay_t<T>>, bool>
    FromRapidValue(const rapidjson::Value &rapid_value, T &value);

    // write
    template <typename T>
    static std::enable_if_t<std::is_base_of_v<Jsonizable, std::decay_t<T>>>
    ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer, const T &value) noexcept;

    inline static void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer, bool value) noexcept;

    template <typename T>
    inline static std::enable_if_t<std::is_integral_v<std::decay_t<T>> && std::is_signed_v<std::decay_t<T>>>
    ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer, T value) noexcept;

    template <typename T>
    inline static std::enable_if_t<std::is_integral_v<std::decay_t<T>> && std::is_unsigned_v<std::decay_t<T>>>
    ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer, T value) noexcept;

    template <typename T>
    inline static std::enable_if_t<std::is_floating_point_v<std::decay_t<T>>>
    ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer, T value) noexcept;

    inline static void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer,
                                     const std::string &value) noexcept;

    template <typename T>
    inline static void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer, const T *value);

    template <typename T>
    inline static void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer,
                                     const std::shared_ptr<T> &value);

    template <typename T>
    inline static std::enable_if_t<std::is_enum_v<std::decay_t<T>>>
    ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer, T value) noexcept;

    template <typename T, typename A>
    static void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer, const std::vector<T, A> &values);

    template <typename T, typename C, typename A>
    static void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer,
                              const std::map<std::string, T, C, A> &values);
};

template <typename T>
bool Jsonizable::FromJsonString(const std::string &str, T &value) {
    rapidjson::Document doc;
    if (!Parse(doc, str)) {
        return false;
    }
    return FromRapidValue(doc, value);
}

template <typename T, typename A>
std::string Jsonizable::ToJsonString(const std::vector<T, A> &values) noexcept {
    rapidjson::StringBuffer s;
    rapidjson::Writer<rapidjson::StringBuffer> writer(s);
    writer.StartArray();
    for (const auto &v : values) {
        ToRapidWriter(writer, v);
    }
    writer.EndArray();
    return s.GetString();
}

template <typename T, typename C, typename A>
std::string Jsonizable::ToJsonString(const std::map<std::string, T, C, A> &values) noexcept {
    rapidjson::StringBuffer s;
    rapidjson::Writer<rapidjson::StringBuffer> writer(s);
    writer.StartObject();
    for (const auto &item : values) {
        writer.Key(item.first.c_str(), item.first.size(), false);
        ToRapidWriter(writer, item.second);
    }
    writer.EndObject();
    return s.GetString();
}

template <typename T, typename A>
bool Jsonizable::FromRapidValue(const rapidjson::Value &rapid_value, std::vector<T, A> &values) {
    if (!rapid_value.IsArray()) {
        return false;
    }
    size_t size = rapid_value.Size();
    values.clear();
    values.reserve(size);
    for (auto &src_rapid_value : rapid_value.GetArray()) {
        T dst_value;
        if (!FromRapidValue(src_rapid_value, dst_value)) {
            return false;
        }
        if constexpr (std::is_move_assignable_v<T>) {
            values.push_back(std::move(dst_value));
        } else {
            values.push_back(dst_value);
        }
    }
    return true;
}

template <typename T, typename C, typename A>
bool Jsonizable::FromRapidValue(const rapidjson::Value &rapid_value, std::map<std::string, T, C, A> &values) {
    if (!rapid_value.IsObject()) {
        return false;
    }
    values.clear();
    for (auto &member : rapid_value.GetObject()) {
        T dst_value;
        if (!FromRapidValue(member.value, dst_value)) {
            return false;
        }
        if constexpr (std::is_move_assignable_v<T>) {
            values[member.name.GetString()] = std::move(dst_value);
        } else {
            values[member.name.GetString()] = dst_value;
        }
    }
    return true;
}

template <typename T>
std::enable_if_t<std::is_base_of_v<Jsonizable, std::decay_t<T>>, bool>
Jsonizable::FromRapidValue(const rapidjson::Value &rapid_value, T &value) {
    if (!rapid_value.IsObject()) {
        return false;
    }
    return value.FromRapidValue(rapid_value);
}

bool Jsonizable::FromRapidValue(const rapidjson::Value &rapid_value, bool &value) {
    if (!rapid_value.IsBool()) {
        return false;
    }
    value = rapid_value.GetBool();
    return true;
}

template <typename T>
std::enable_if_t<std::is_integral_v<std::decay_t<T>> && std::is_signed_v<std::decay_t<T>>, bool>
Jsonizable::FromRapidValue(const rapidjson::Value &rapid_value, T &value) {
    using DecayType = std::decay_t<T>;
    if (rapid_value.IsInt()) {
        value = static_cast<DecayType>(rapid_value.GetInt());
    } else if (rapid_value.IsInt64()) {
        value = static_cast<DecayType>(rapid_value.GetInt64());
    } else {
        return false;
    }
    return true;
}

template <typename T>
std::enable_if_t<std::is_integral_v<std::decay_t<T>> && std::is_unsigned_v<std::decay_t<T>>, bool>
Jsonizable::FromRapidValue(const rapidjson::Value &rapid_value, T &value) {
    using DecayType = std::decay_t<T>;
    if (rapid_value.IsUint()) {
        value = static_cast<DecayType>(rapid_value.GetUint());
    } else if (rapid_value.IsUint64()) {
        value = static_cast<DecayType>(rapid_value.GetUint64());
    } else {
        return false;
    }
    return true;
}

template <typename T>
std::enable_if_t<std::is_floating_point_v<std::decay_t<T>>, bool>
Jsonizable::FromRapidValue(const rapidjson::Value &rapid_value, T &value) {
    using DecayType = std::decay_t<T>;
    if (!rapid_value.IsLosslessDouble()) {
        return false;
    }
    value = static_cast<DecayType>(rapid_value.GetDouble());
    return true;
}

bool Jsonizable::FromRapidValue(const rapidjson::Value &rapid_value, std::string &value) {
    if (!rapid_value.IsString()) {
        return false;
    }
    value.assign(rapid_value.GetString(), rapid_value.GetStringLength());
    return true;
}

template <typename T>
bool Jsonizable::FromRapidValue(const rapidjson::Value &rapid_value, T *&value) {
    if (rapid_value.IsNull()) {
        value = nullptr;
        return false;
    }
    std::unique_ptr<T> guard = std::make_unique<T>();
    if (FromRapidValue(rapid_value, *guard)) {
        value = guard.release();
        return true;
    }
    return false;
}

template <typename T>
bool Jsonizable::FromRapidValue(const rapidjson::Value &rapid_value, std::shared_ptr<T> &value) {
    T *temp_value = nullptr;
    if (FromRapidValue(rapid_value, temp_value)) {
        value.reset(temp_value);
        return true;
    }
    return false;
}

template <typename T>
std::enable_if_t<std::is_enum_v<std::decay_t<T>>, bool> Jsonizable::FromRapidValue(const rapidjson::Value &rapid_value,
                                                                                   T &value) {
    int temp_value = 0;
    if (FromRapidValue(rapid_value, temp_value)) {
        value = static_cast<std::decay_t<T>>(temp_value);
        return true;
    }
    return false;
}

template <typename T>
std::enable_if_t<std::is_base_of_v<Jsonizable, std::decay_t<T>>>
Jsonizable::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer, const T &value) noexcept {
    writer.StartObject();
    value.ToRapidWriter(writer);
    writer.EndObject();
}

void Jsonizable::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer, bool value) noexcept {
    writer.Bool(value);
}

template <typename T>
std::enable_if_t<std::is_integral_v<std::decay_t<T>> && std::is_signed_v<std::decay_t<T>>>
Jsonizable::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer, T value) noexcept {
    writer.Int64(static_cast<int64_t>(value));
}

template <typename T>
std::enable_if_t<std::is_integral_v<std::decay_t<T>> && std::is_unsigned_v<std::decay_t<T>>>
Jsonizable::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer, T value) noexcept {
    writer.Uint64(static_cast<uint64_t>(value));
}

template <typename T>
std::enable_if_t<std::is_floating_point_v<std::decay_t<T>>>
Jsonizable::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer, T value) noexcept {
    writer.Double(static_cast<double>(value));
}

void Jsonizable::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer, const std::string &value) noexcept {
    writer.String(value.c_str(), value.size(), false);
}

template <typename T>
void Jsonizable::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer, const T *value) {
    if (value) {
        ToRapidWriter(writer, *value);
    } else {
        writer.Null();
    }
}

template <typename T>
void Jsonizable::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer, const std::shared_ptr<T> &value) {
    ToRapidWriter(writer, value.get());
}

template <typename T>
std::enable_if_t<std::is_enum_v<std::decay_t<T>>>
Jsonizable::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer, T value) noexcept {
    using UnderlyingType = std::underlying_type_t<std::decay_t<T>>;
    const auto underlying_value = static_cast<UnderlyingType>(value);
    ToRapidWriter(writer, underlying_value);
}

template <typename T, typename A>
void Jsonizable::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer, const std::vector<T, A> &values) {
    writer.StartArray();
    for (const auto &v : values) {
        ToRapidWriter(writer, v);
    }
    writer.EndArray();
}

template <typename T, typename C, typename A>
void Jsonizable::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer,
                               const std::map<std::string, T, C, A> &values) {
    writer.StartObject();
    for (const auto &item : values) {
        writer.Key(item.first.c_str(), item.first.size(), false);
        ToRapidWriter(writer, item.second);
    }
    writer.EndObject();
}

template <typename T>
bool Jsonizable::Get(const rapidjson::Value &rapid_value, const std::string &key, T &value, const T &default_value) {
    if (!rapid_value.IsObject()) {
        return false;
    }
    auto iter = rapid_value.FindMember(key);
    if (iter == rapid_value.MemberEnd()) {
        value = default_value;
        return true;
    }
    return FromRapidValue(iter->value, value);
}

template <typename T>
bool Jsonizable::Get(const rapidjson::Value &rapid_value, const std::string &key, T &value) {
    if (!rapid_value.IsObject()) {
        return false;
    }
    auto iter = rapid_value.FindMember(key);
    if (iter == rapid_value.MemberEnd()) {
        return true;
    }
    return FromRapidValue(iter->value, value);
}

template <typename T, typename A>
bool Jsonizable::Get(const rapidjson::Value &rapid_value, const std::string &key, std::vector<T, A> &values) {
    if (!rapid_value.IsObject()) {
        return false;
    }
    auto iter = rapid_value.FindMember(key);
    if (iter == rapid_value.MemberEnd()) {
        values.clear();
        return true;
    }
    return FromRapidValue<T, A>(iter->value, values);
}

template <typename T, typename C, typename A>
bool Jsonizable::Get(const rapidjson::Value &rapid_value,
                     const std::string &key,
                     std::map<std::string, T, C, A> &values) {
    if (!rapid_value.IsObject()) {
        return false;
    }
    auto iter = rapid_value.FindMember(key);
    if (iter == rapid_value.MemberEnd()) {
        values.clear();
        return true;
    }
    return FromRapidValue<T, C, A>(iter->value, values);
}

template <typename T>
void Jsonizable::Put(rapidjson::Writer<rapidjson::StringBuffer> &writer,
                     const std::string &key,
                     const T &value) noexcept {
    writer.Key(key.c_str(), key.size(), false);
    ToRapidWriter(writer, value);
}

template <typename T, typename A>
void Jsonizable::Put(rapidjson::Writer<rapidjson::StringBuffer> &writer,
                     const std::string &key,
                     const std::vector<T, A> &values) noexcept {
    writer.Key(key.c_str(), key.size(), false);
    ToRapidWriter(writer, values);
}

template <typename T, typename C, typename A>
void Jsonizable::Put(rapidjson::Writer<rapidjson::StringBuffer> &writer,
                     const std::string &key,
                     const std::map<std::string, T, C, A> &values) noexcept {
    writer.Key(key.c_str(), key.size(), false);
    ToRapidWriter(writer, values);
}

} // namespace kv_cache_manager
