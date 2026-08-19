// Jsonizable-based helpers for strict configuration objects.
#pragma once

#include <initializer_list>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "kv_cache_manager/common/jsonizable.h"

namespace kvcm_swarm {

// Base class for strict scenario and behavior configuration objects. It keeps
// JSON shape/type errors separate from semantic validation.
class JsonConfig : public kv_cache_manager::Jsonizable {
public:
    const std::vector<std::string> &json_errors() const { return json_errors_; }
    void AppendJsonErrors(std::string_view path, std::vector<std::string> *errors) const;

    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> & /*writer*/) const noexcept override {}

protected:
    void AddJsonError(std::string error) { json_errors_.push_back(std::move(error)); }
    void MergeJsonErrors(std::string_view path, const JsonConfig &child);

    bool BeginObject(const rapidjson::Value &value, std::initializer_list<std::string_view> known_fields) {
        json_errors_.clear();
        if (!value.IsObject()) {
            json_errors_.push_back("expected a JSON object");
            return false;
        }
        CheckUnknownFields(value, known_fields, &json_errors_);
        return true;
    }

    template <typename T>
    void Required(const rapidjson::Value &value, std::string_view key, T &destination) {
        GetRequired(value, key, destination, &json_errors_);
    }

    template <typename T>
    void Optional(const rapidjson::Value &value, std::string_view key, std::optional<T> &destination) {
        GetOptional(value, key, destination, &json_errors_);
    }

    template <typename T>
    void Optional(const rapidjson::Value &value, std::string_view key, T &destination, const T &default_value) {
        GetOptional(value, key, destination, default_value, &json_errors_);
    }

    template <typename T, typename Decoder>
    void RequiredCustom(const rapidjson::Value &value, std::string_view key, T &destination, Decoder &&decoder) {
        if (!value.IsObject()) {
            json_errors_.push_back("expected a JSON object");
            return;
        }
        const auto member = value.FindMember(rapidjson::StringRef(key.data(), key.size()));
        if (member == value.MemberEnd()) {
            json_errors_.push_back(std::string(key) + ": required field is missing");
            return;
        }
        std::string error;
        if (!std::forward<Decoder>(decoder)(member->value, &destination, &error)) {
            json_errors_.push_back(std::string(key) + ": " + error);
        }
    }

    template <typename T, typename Decoder>
    void OptionalCustom(const rapidjson::Value &value,
                        std::string_view key,
                        T &destination,
                        const T &default_value,
                        Decoder &&decoder) {
        if (!value.IsObject()) {
            json_errors_.push_back("expected a JSON object");
            return;
        }
        const auto member = value.FindMember(rapidjson::StringRef(key.data(), key.size()));
        if (member == value.MemberEnd()) {
            destination = default_value;
            return;
        }
        std::string error;
        if (!std::forward<Decoder>(decoder)(member->value, &destination, &error)) {
            json_errors_.push_back(std::string(key) + ": " + error);
        }
    }

private:
    std::vector<std::string> json_errors_;
};

// Captures an arbitrary JSON object so the common scenario loader can defer a
// behavior-specific subtree to the registered behavior factory.
class RawJsonObject final : public JsonConfig {
public:
    bool FromRapidValue(const rapidjson::Value &value) override;
    const std::string &json() const { return json_; }

private:
    std::string json_;
};

} // namespace kvcm_swarm
