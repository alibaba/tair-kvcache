// Immutable JSON configuration tree with consumed-key tracking.
//
// JSON is the only run-configuration entry point. The top-level loader parses
// runtime/target/behavior envelope/evidence; each behavior factory then parses
// its own `config` subtree. Any key nobody read is reported as an unknown
// field and fails local validation.
#pragma once

#include <cstdint>
#include <memory>
#include <set>
#include <string>
#include <string_view>
#include <vector>

namespace kvcm_swarm {

class ConfigValue;
using ConfigValuePtr = std::shared_ptr<const ConfigValue>;

class ConfigValue {
public:
    enum class Type {
        kNull,
        kBool,
        kInt,
        kDouble,
        kString,
        kArray,
        kObject
    };

    Type type = Type::kNull;
    bool bool_value = false;
    int64_t int_value = 0;
    double double_value = 0.0;
    std::string string_value;
    std::vector<ConfigValuePtr> array_value;
    std::vector<std::pair<std::string, ConfigValuePtr>> object_value;
};

class ConfigNode {
public:
    ConfigNode() = default;

    static ConfigNode Parse(const std::string &json, std::string *error);
    static ConfigNode
    FromValue(ConfigValuePtr value, std::string path, std::shared_ptr<std::set<std::string>> consumed);

    bool valid() const { return value_ != nullptr; }
    bool IsNull() const { return value_ == nullptr || value_->type == ConfigValue::Type::kNull; }
    bool IsObject() const { return value_ != nullptr && value_->type == ConfigValue::Type::kObject; }
    bool IsArray() const { return value_ != nullptr && value_->type == ConfigValue::Type::kArray; }
    bool IsString() const { return value_ != nullptr && value_->type == ConfigValue::Type::kString; }
    bool IsNumber() const {
        return value_ != nullptr &&
               (value_->type == ConfigValue::Type::kInt || value_->type == ConfigValue::Type::kDouble);
    }
    bool IsBool() const { return value_ != nullptr && value_->type == ConfigValue::Type::kBool; }

    const std::string &path() const { return path_; }

    bool Has(std::string_view key) const;
    // Returns the child node and records `key` as consumed.
    ConfigNode Get(std::string_view key) const;
    std::vector<std::string> Keys() const;
    std::vector<ConfigNode> Items() const;

    bool AsBool(bool *out) const;
    bool AsInt(int64_t *out) const;
    bool AsDouble(double *out) const;
    bool AsString(std::string *out) const;

    // Marks this node and its whole subtree as consumed.
    void MarkSubtreeConsumed() const;
    // Appends the dotted paths of every key that was never read.
    void CollectUnknown(std::vector<std::string> *unknown) const;

    std::string Serialize() const;

private:
    ConfigValuePtr value_;
    std::string path_;
    std::shared_ptr<std::set<std::string>> consumed_;
};

} // namespace kvcm_swarm
