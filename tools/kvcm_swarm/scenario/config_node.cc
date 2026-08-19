#include "tools/kvcm_swarm/scenario/config_node.h"

#include <algorithm>

#include "rapidjson/document.h"
#include "rapidjson/error/en.h"
#include "tools/kvcm_swarm/evidence/json_writer.h"

namespace kvcm_swarm {
namespace {

ConfigValuePtr Convert(const rapidjson::Value &value) {
    auto node = std::make_shared<ConfigValue>();
    switch (value.GetType()) {
    case rapidjson::kNullType:
        node->type = ConfigValue::Type::kNull;
        break;
    case rapidjson::kFalseType:
    case rapidjson::kTrueType:
        node->type = ConfigValue::Type::kBool;
        node->bool_value = value.GetBool();
        break;
    case rapidjson::kStringType:
        node->type = ConfigValue::Type::kString;
        node->string_value.assign(value.GetString(), value.GetStringLength());
        break;
    case rapidjson::kNumberType:
        if (value.IsInt64()) {
            node->type = ConfigValue::Type::kInt;
            node->int_value = value.GetInt64();
            node->double_value = static_cast<double>(value.GetInt64());
        } else if (value.IsUint64()) {
            node->type = ConfigValue::Type::kInt;
            node->int_value = static_cast<int64_t>(value.GetUint64());
            node->double_value = static_cast<double>(value.GetUint64());
        } else {
            node->type = ConfigValue::Type::kDouble;
            node->double_value = value.GetDouble();
            node->int_value = static_cast<int64_t>(value.GetDouble());
        }
        break;
    case rapidjson::kArrayType:
        node->type = ConfigValue::Type::kArray;
        for (const auto &item : value.GetArray()) {
            node->array_value.push_back(Convert(item));
        }
        break;
    case rapidjson::kObjectType:
        node->type = ConfigValue::Type::kObject;
        for (const auto &member : value.GetObject()) {
            node->object_value.emplace_back(std::string(member.name.GetString(), member.name.GetStringLength()),
                                            Convert(member.value));
        }
        break;
    }
    return node;
}

void SerializeInto(const ConfigValue &value, JsonWriter &writer) {
    switch (value.type) {
    case ConfigValue::Type::kNull:
        writer.Null();
        break;
    case ConfigValue::Type::kBool:
        writer.Bool(value.bool_value);
        break;
    case ConfigValue::Type::kInt:
        writer.Int(value.int_value);
        break;
    case ConfigValue::Type::kDouble:
        writer.Double(value.double_value);
        break;
    case ConfigValue::Type::kString:
        writer.String(value.string_value);
        break;
    case ConfigValue::Type::kArray:
        writer.BeginArray();
        for (const auto &item : value.array_value) {
            SerializeInto(*item, writer);
        }
        writer.EndArray();
        break;
    case ConfigValue::Type::kObject:
        writer.BeginObject();
        for (const auto &member : value.object_value) {
            writer.Key(member.first);
            SerializeInto(*member.second, writer);
        }
        writer.EndObject();
        break;
    }
}

void CollectUnknownInto(const ConfigValue &value,
                        const std::string &path,
                        const std::set<std::string> &consumed,
                        std::vector<std::string> *unknown) {
    if (value.type == ConfigValue::Type::kObject) {
        for (const auto &member : value.object_value) {
            const std::string child_path = path.empty() ? member.first : path + "." + member.first;
            if (consumed.find(child_path) == consumed.end()) {
                unknown->push_back(child_path);
                continue;
            }
            CollectUnknownInto(*member.second, child_path, consumed, unknown);
        }
        return;
    }
    if (value.type == ConfigValue::Type::kArray) {
        for (size_t i = 0; i < value.array_value.size(); ++i) {
            CollectUnknownInto(*value.array_value[i], path + "[" + std::to_string(i) + "]", consumed, unknown);
        }
    }
}

void MarkConsumed(const ConfigValue &value, const std::string &path, std::set<std::string> *consumed) {
    if (value.type == ConfigValue::Type::kObject) {
        for (const auto &member : value.object_value) {
            const std::string child_path = path.empty() ? member.first : path + "." + member.first;
            consumed->insert(child_path);
            MarkConsumed(*member.second, child_path, consumed);
        }
        return;
    }
    if (value.type == ConfigValue::Type::kArray) {
        for (size_t i = 0; i < value.array_value.size(); ++i) {
            MarkConsumed(*value.array_value[i], path + "[" + std::to_string(i) + "]", consumed);
        }
    }
}

} // namespace

ConfigNode ConfigNode::Parse(const std::string &json, std::string *error) {
    rapidjson::Document document;
    document.Parse(json.c_str(), json.size());
    if (document.HasParseError()) {
        if (error != nullptr) {
            *error = std::string("JSON parse error at offset ") + std::to_string(document.GetErrorOffset()) + ": " +
                     rapidjson::GetParseError_En(document.GetParseError());
        }
        return ConfigNode();
    }
    ConfigNode node;
    node.value_ = Convert(document);
    node.path_ = "";
    node.consumed_ = std::make_shared<std::set<std::string>>();
    return node;
}

ConfigNode
ConfigNode::FromValue(ConfigValuePtr value, std::string path, std::shared_ptr<std::set<std::string>> consumed) {
    ConfigNode node;
    node.value_ = std::move(value);
    node.path_ = std::move(path);
    node.consumed_ = std::move(consumed);
    return node;
}

bool ConfigNode::Has(std::string_view key) const {
    if (!IsObject()) {
        return false;
    }
    for (const auto &member : value_->object_value) {
        if (member.first == key) {
            return true;
        }
    }
    return false;
}

ConfigNode ConfigNode::Get(std::string_view key) const {
    const std::string child_path = path_.empty() ? std::string(key) : path_ + "." + std::string(key);
    if (consumed_) {
        consumed_->insert(child_path);
    }
    if (!IsObject()) {
        return ConfigNode();
    }
    for (const auto &member : value_->object_value) {
        if (member.first == key) {
            return FromValue(member.second, child_path, consumed_);
        }
    }
    return ConfigNode();
}

std::vector<std::string> ConfigNode::Keys() const {
    std::vector<std::string> keys;
    if (!IsObject()) {
        return keys;
    }
    keys.reserve(value_->object_value.size());
    for (const auto &member : value_->object_value) {
        keys.push_back(member.first);
    }
    return keys;
}

std::vector<ConfigNode> ConfigNode::Items() const {
    std::vector<ConfigNode> items;
    if (!IsArray()) {
        return items;
    }
    items.reserve(value_->array_value.size());
    for (size_t i = 0; i < value_->array_value.size(); ++i) {
        items.push_back(FromValue(value_->array_value[i], path_ + "[" + std::to_string(i) + "]", consumed_));
    }
    return items;
}

bool ConfigNode::AsBool(bool *out) const {
    if (!IsBool()) {
        return false;
    }
    *out = value_->bool_value;
    return true;
}

bool ConfigNode::AsInt(int64_t *out) const {
    if (value_ == nullptr || value_->type != ConfigValue::Type::kInt) {
        return false;
    }
    *out = value_->int_value;
    return true;
}

bool ConfigNode::AsDouble(double *out) const {
    if (!IsNumber()) {
        return false;
    }
    *out = value_->double_value;
    return true;
}

bool ConfigNode::AsString(std::string *out) const {
    if (!IsString()) {
        return false;
    }
    *out = value_->string_value;
    return true;
}

void ConfigNode::MarkSubtreeConsumed() const {
    if (value_ == nullptr || !consumed_) {
        return;
    }
    MarkConsumed(*value_, path_, consumed_.get());
}

void ConfigNode::CollectUnknown(std::vector<std::string> *unknown) const {
    if (value_ == nullptr || !consumed_) {
        return;
    }
    CollectUnknownInto(*value_, path_, *consumed_, unknown);
}

std::string ConfigNode::Serialize() const {
    JsonWriter writer(false);
    if (value_ == nullptr) {
        writer.Null();
    } else {
        SerializeInto(*value_, writer);
    }
    return writer.Take();
}

} // namespace kvcm_swarm
