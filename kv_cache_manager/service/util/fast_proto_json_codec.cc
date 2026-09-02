#include "fast_proto_json_codec.h"

#include <charconv>
#include <cmath>
#include <cstdint>
#include <google/protobuf/stubs/common.h>
#include <google/protobuf/stubs/strutil.h>
#include <limits>
#include <memory>
#include <rapidjson/document.h>
#include <rapidjson/memorystream.h>
#include <rapidjson/reader.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#include <string>
#include <unordered_set>
#include <vector>

namespace kv_cache_manager {
namespace {

using ::google::protobuf::Descriptor;
using ::google::protobuf::FieldDescriptor;
using ::google::protobuf::Message;
using ::google::protobuf::Reflection;
using JsonWriter = rapidjson::Writer<rapidjson::StringBuffer>;

constexpr int kMaxNestingDepth = 100;

template <typename Handler>
class DepthLimitedJsonHandler {
public:
    explicit DepthLimitedJsonHandler(Handler &handler) : handler_(handler) {}

    bool Null() { return handler_.Null(); }
    bool Bool(bool value) { return handler_.Bool(value); }
    bool Int(int value) { return handler_.Int(value); }
    bool Uint(unsigned value) { return handler_.Uint(value); }
    bool Int64(int64_t value) { return handler_.Int64(value); }
    bool Uint64(uint64_t value) { return handler_.Uint64(value); }
    bool Double(double value) { return handler_.Double(value); }
    bool RawNumber(const char *value, rapidjson::SizeType length, bool copy) {
        return handler_.RawNumber(value, length, copy);
    }
    bool String(const char *value, rapidjson::SizeType length, bool copy) {
        return handler_.String(value, length, copy);
    }
    bool Key(const char *value, rapidjson::SizeType length, bool copy) { return handler_.Key(value, length, copy); }

    bool StartObject() { return StartContainer(&Handler::StartObject); }
    bool EndObject(rapidjson::SizeType member_count) {
        const bool success = handler_.EndObject(member_count);
        --depth_;
        return success;
    }
    bool StartArray() { return StartContainer(&Handler::StartArray); }
    bool EndArray(rapidjson::SizeType element_count) {
        const bool success = handler_.EndArray(element_count);
        --depth_;
        return success;
    }

private:
    bool StartContainer(bool (Handler::*start)()) {
        if (depth_ >= kMaxNestingDepth) {
            return false;
        }
        ++depth_;
        if ((handler_.*start)()) {
            return true;
        }
        --depth_;
        return false;
    }

    Handler &handler_;
    int depth_ = 0;
};

class DepthLimitedDomGenerator {
public:
    explicit DepthLimitedDomGenerator(std::string_view json) : json_(json) {}

    template <typename Handler>
    bool operator()(Handler &handler) {
        rapidjson::MemoryStream input(json_.data(), json_.size());
        rapidjson::Reader reader;
        DepthLimitedJsonHandler<Handler> depth_limited_handler(handler);
        // The handler stops before the reader can descend past the same
        // bounded depth, so neither the DOM nor the recursive reader stack is
        // allowed to grow with adversarial input.
        const auto result = reader.Parse<rapidjson::kParseValidateEncodingFlag>(input, depth_limited_handler);
        success_ = !result.IsError();
        return success_;
    }

    bool success() const { return success_; }

private:
    std::string_view json_;
    bool success_ = false;
};

enum class WrapperKind {
    kNone,
    kInt32,
    kInt64,
    kUInt64,
    kUnsupportedWellKnownType,
};

WrapperKind GetWrapperKind(const Descriptor *descriptor) {
    const std::string &name = descriptor->full_name();
    if (name == "google.protobuf.Int32Value") {
        return WrapperKind::kInt32;
    }
    if (name == "google.protobuf.Int64Value") {
        return WrapperKind::kInt64;
    }
    if (name == "google.protobuf.UInt64Value") {
        return WrapperKind::kUInt64;
    }
    if (name.compare(0, sizeof("google.protobuf.") - 1, "google.protobuf.") == 0) {
        return WrapperKind::kUnsupportedWellKnownType;
    }
    return WrapperKind::kNone;
}

bool SupportsDescriptor(const Descriptor *descriptor, std::unordered_set<const Descriptor *> &visited) {
    if (!descriptor || descriptor->file()->syntax() != ::google::protobuf::FileDescriptor::SYNTAX_PROTO3) {
        return false;
    }

    const WrapperKind wrapper_kind = GetWrapperKind(descriptor);
    if (wrapper_kind == WrapperKind::kInt32 || wrapper_kind == WrapperKind::kInt64 ||
        wrapper_kind == WrapperKind::kUInt64) {
        return true;
    }
    if (wrapper_kind == WrapperKind::kUnsupportedWellKnownType || descriptor->extension_range_count() != 0) {
        return false;
    }
    if (!visited.insert(descriptor).second) {
        return true;
    }

    for (int i = 0; i < descriptor->field_count(); ++i) {
        const FieldDescriptor *field = descriptor->field(i);
        if (field->type() == FieldDescriptor::TYPE_BYTES || field->type() == FieldDescriptor::TYPE_GROUP ||
            field->is_extension()) {
            return false;
        }
        if (field->cpp_type() == FieldDescriptor::CPPTYPE_ENUM &&
            field->enum_type()->full_name() == "google.protobuf.NullValue") {
            return false;
        }
        if (field->is_map()) {
            const Descriptor *entry = field->message_type();
            if (!entry || entry->field_count() != 2 || entry->field(0)->type() != FieldDescriptor::TYPE_STRING ||
                entry->field(1)->type() != FieldDescriptor::TYPE_STRING) {
                return false;
            }
            continue;
        }
        if (field->cpp_type() == FieldDescriptor::CPPTYPE_MESSAGE &&
            !SupportsDescriptor(field->message_type(), visited)) {
            return false;
        }
    }
    return true;
}

void WriteRawNumber(const std::string &number, JsonWriter &writer) {
    writer.RawValue(number.data(), static_cast<rapidjson::SizeType>(number.size()), rapidjson::kNumberType);
}

void WriteInt64(int64_t value, JsonWriter &writer) {
    const std::string number = std::to_string(value);
    writer.String(number.data(), static_cast<rapidjson::SizeType>(number.size()));
}

void WriteUInt64(uint64_t value, JsonWriter &writer) {
    const std::string number = std::to_string(value);
    writer.String(number.data(), static_cast<rapidjson::SizeType>(number.size()));
}

void WriteDouble(double value, JsonWriter &writer) {
    if (std::isfinite(value)) {
        WriteRawNumber(::google::protobuf::SimpleDtoa(value), writer);
    } else if (std::isnan(value)) {
        writer.String("NaN");
    } else if (value > 0) {
        writer.String("Infinity");
    } else {
        writer.String("-Infinity");
    }
}

void WriteFloat(float value, JsonWriter &writer) {
    if (std::isfinite(value)) {
        WriteRawNumber(::google::protobuf::SimpleFtoa(value), writer);
    } else if (std::isnan(value)) {
        writer.String("NaN");
    } else if (value > 0) {
        writer.String("Infinity");
    } else {
        writer.String("-Infinity");
    }
}

bool WriteMessage(const Message &message, JsonWriter &writer, int depth);

bool WriteWrapper(const Message &message, WrapperKind kind, JsonWriter &writer) {
    const Reflection *reflection = message.GetReflection();
    const FieldDescriptor *value_field = message.GetDescriptor()->FindFieldByName("value");
    if (!value_field) {
        return false;
    }
    if (kind == WrapperKind::kInt32) {
        writer.Int(reflection->GetInt32(message, value_field));
        return true;
    }
    if (kind == WrapperKind::kInt64) {
        WriteInt64(reflection->GetInt64(message, value_field), writer);
        return true;
    }
    if (kind == WrapperKind::kUInt64) {
        WriteUInt64(reflection->GetUInt64(message, value_field), writer);
        return true;
    }
    return false;
}

bool WriteFieldValue(
    const Message &message, const FieldDescriptor *field, int repeated_index, JsonWriter &writer, int depth) {
    const Reflection *reflection = message.GetReflection();
    const bool repeated = repeated_index >= 0;
    switch (field->cpp_type()) {
    case FieldDescriptor::CPPTYPE_INT32:
        writer.Int(repeated ? reflection->GetRepeatedInt32(message, field, repeated_index)
                            : reflection->GetInt32(message, field));
        return true;
    case FieldDescriptor::CPPTYPE_UINT32:
        writer.Uint(repeated ? reflection->GetRepeatedUInt32(message, field, repeated_index)
                             : reflection->GetUInt32(message, field));
        return true;
    case FieldDescriptor::CPPTYPE_INT64:
        WriteInt64(repeated ? reflection->GetRepeatedInt64(message, field, repeated_index)
                            : reflection->GetInt64(message, field),
                   writer);
        return true;
    case FieldDescriptor::CPPTYPE_UINT64:
        WriteUInt64(repeated ? reflection->GetRepeatedUInt64(message, field, repeated_index)
                             : reflection->GetUInt64(message, field),
                    writer);
        return true;
    case FieldDescriptor::CPPTYPE_DOUBLE:
        WriteDouble(repeated ? reflection->GetRepeatedDouble(message, field, repeated_index)
                             : reflection->GetDouble(message, field),
                    writer);
        return true;
    case FieldDescriptor::CPPTYPE_FLOAT:
        WriteFloat(repeated ? reflection->GetRepeatedFloat(message, field, repeated_index)
                            : reflection->GetFloat(message, field),
                   writer);
        return true;
    case FieldDescriptor::CPPTYPE_BOOL:
        writer.Bool(repeated ? reflection->GetRepeatedBool(message, field, repeated_index)
                             : reflection->GetBool(message, field));
        return true;
    case FieldDescriptor::CPPTYPE_ENUM: {
        const int enum_number = repeated ? reflection->GetRepeatedEnumValue(message, field, repeated_index)
                                         : reflection->GetEnumValue(message, field);
        const auto *enum_value = field->enum_type()->FindValueByNumber(enum_number);
        if (enum_value) {
            writer.String(enum_value->name().data(), static_cast<rapidjson::SizeType>(enum_value->name().size()));
        } else {
            writer.Int(enum_number);
        }
        return true;
    }
    case FieldDescriptor::CPPTYPE_STRING: {
        std::string scratch;
        const std::string &value =
            repeated ? reflection->GetRepeatedStringReference(message, field, repeated_index, &scratch)
                     : reflection->GetStringReference(message, field, &scratch);
        if (!::google::protobuf::internal::IsStructurallyValidUTF8(value)) {
            return false;
        }
        writer.String(value.data(), static_cast<rapidjson::SizeType>(value.size()));
        return true;
    }
    case FieldDescriptor::CPPTYPE_MESSAGE: {
        const Message &nested = repeated ? reflection->GetRepeatedMessage(message, field, repeated_index)
                                         : reflection->GetMessage(message, field);
        const WrapperKind kind = GetWrapperKind(nested.GetDescriptor());
        return kind == WrapperKind::kNone ? WriteMessage(nested, writer, depth + 1)
                                          : WriteWrapper(nested, kind, writer);
    }
    }
    return false;
}

bool WriteMap(const Message &message, const FieldDescriptor *field, JsonWriter &writer, int depth) {
    const Reflection *reflection = message.GetReflection();
    const Descriptor *entry_descriptor = field->message_type();
    const FieldDescriptor *key_field = entry_descriptor->field(0);
    const FieldDescriptor *value_field = entry_descriptor->field(1);
    writer.StartObject();
    const int size = reflection->FieldSize(message, field);
    for (int i = 0; i < size; ++i) {
        const Message &entry = reflection->GetRepeatedMessage(message, field, i);
        const Reflection *entry_reflection = entry.GetReflection();
        std::string key_scratch;
        std::string value_scratch;
        const std::string &key = entry_reflection->GetStringReference(entry, key_field, &key_scratch);
        const std::string &value = entry_reflection->GetStringReference(entry, value_field, &value_scratch);
        if (!::google::protobuf::internal::IsStructurallyValidUTF8(key) ||
            !::google::protobuf::internal::IsStructurallyValidUTF8(value)) {
            return false;
        }
        writer.Key(key.data(), static_cast<rapidjson::SizeType>(key.size()));
        writer.String(value.data(), static_cast<rapidjson::SizeType>(value.size()));
    }
    writer.EndObject();
    return depth <= kMaxNestingDepth;
}

bool WriteMessage(const Message &message, JsonWriter &writer, int depth) {
    if (depth > kMaxNestingDepth) {
        return false;
    }
    const Descriptor *descriptor = message.GetDescriptor();
    const Reflection *reflection = message.GetReflection();
    writer.StartObject();
    for (int i = 0; i < descriptor->field_count(); ++i) {
        const FieldDescriptor *field = descriptor->field(i);
        if (field->is_repeated()) {
            writer.Key(field->name().data(), static_cast<rapidjson::SizeType>(field->name().size()));
            if (field->is_map()) {
                if (!WriteMap(message, field, writer, depth + 1)) {
                    return false;
                }
                continue;
            }
            writer.StartArray();
            const int size = reflection->FieldSize(message, field);
            for (int j = 0; j < size; ++j) {
                if (!WriteFieldValue(message, field, j, writer, depth)) {
                    return false;
                }
            }
            writer.EndArray();
            continue;
        }

        if (field->cpp_type() == FieldDescriptor::CPPTYPE_MESSAGE || field->containing_oneof()) {
            if (!reflection->HasField(message, field)) {
                continue;
            }
        }
        writer.Key(field->name().data(), static_cast<rapidjson::SizeType>(field->name().size()));
        if (!WriteFieldValue(message, field, -1, writer, depth)) {
            return false;
        }
    }
    writer.EndObject();
    return true;
}

template <typename Integer>
bool ParseIntegerString(const rapidjson::Value &value, Integer *result) {
    if (!value.IsString()) {
        return false;
    }
    const char *begin = value.GetString();
    const char *end = begin + value.GetStringLength();
    const auto parsed = std::from_chars(begin, end, *result);
    return parsed.ec == std::errc() && parsed.ptr == end;
}

const FieldDescriptor *FindJsonField(const Descriptor *descriptor, const char *name, size_t length) {
    const std::string field_name(name, length);
    if (const FieldDescriptor *field = descriptor->FindFieldByName(field_name)) {
        return field;
    }
    for (int i = 0; i < descriptor->field_count(); ++i) {
        const FieldDescriptor *field = descriptor->field(i);
        if (field->json_name() == field_name) {
            return field;
        }
    }
    return nullptr;
}

bool ParseMessageValue(const rapidjson::Value &value, Message *message, int depth);

bool ParseWrapperValue(const rapidjson::Value &value, Message *message, WrapperKind kind) {
    const Reflection *reflection = message->GetReflection();
    const FieldDescriptor *field = message->GetDescriptor()->FindFieldByName("value");
    if (!field) {
        return false;
    }
    if (kind == WrapperKind::kInt32) {
        int32_t result = 0;
        if (value.IsInt()) {
            result = value.GetInt();
        } else if (!ParseIntegerString(value, &result)) {
            return false;
        }
        reflection->SetInt32(message, field, result);
        return true;
    }
    if (kind == WrapperKind::kInt64) {
        int64_t result = 0;
        if (value.IsInt64()) {
            result = value.GetInt64();
        } else if (value.IsUint64() &&
                   value.GetUint64() <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
            result = static_cast<int64_t>(value.GetUint64());
        } else if (!ParseIntegerString(value, &result)) {
            return false;
        }
        reflection->SetInt64(message, field, result);
        return true;
    }
    if (kind == WrapperKind::kUInt64) {
        uint64_t result = 0;
        if (value.IsUint64()) {
            result = value.GetUint64();
        } else if (!ParseIntegerString(value, &result)) {
            return false;
        }
        reflection->SetUInt64(message, field, result);
        return true;
    }
    return false;
}

bool ParseScalarValue(const rapidjson::Value &value, Message *message, const FieldDescriptor *field, bool repeated) {
    const Reflection *reflection = message->GetReflection();
    switch (field->cpp_type()) {
    case FieldDescriptor::CPPTYPE_INT32: {
        int32_t result = 0;
        if (value.IsInt()) {
            result = value.GetInt();
        } else if (!ParseIntegerString(value, &result)) {
            return false;
        }
        repeated ? reflection->AddInt32(message, field, result) : reflection->SetInt32(message, field, result);
        return true;
    }
    case FieldDescriptor::CPPTYPE_UINT32: {
        uint32_t result = 0;
        if (value.IsUint()) {
            result = value.GetUint();
        } else if (!ParseIntegerString(value, &result)) {
            return false;
        }
        repeated ? reflection->AddUInt32(message, field, result) : reflection->SetUInt32(message, field, result);
        return true;
    }
    case FieldDescriptor::CPPTYPE_INT64: {
        int64_t result = 0;
        if (value.IsInt64()) {
            result = value.GetInt64();
        } else if (value.IsUint64() &&
                   value.GetUint64() <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
            result = static_cast<int64_t>(value.GetUint64());
        } else if (!ParseIntegerString(value, &result)) {
            return false;
        }
        repeated ? reflection->AddInt64(message, field, result) : reflection->SetInt64(message, field, result);
        return true;
    }
    case FieldDescriptor::CPPTYPE_UINT64: {
        uint64_t result = 0;
        if (value.IsUint64()) {
            result = value.GetUint64();
        } else if (!ParseIntegerString(value, &result)) {
            return false;
        }
        repeated ? reflection->AddUInt64(message, field, result) : reflection->SetUInt64(message, field, result);
        return true;
    }
    case FieldDescriptor::CPPTYPE_DOUBLE:
    case FieldDescriptor::CPPTYPE_FLOAT: {
        double result = 0;
        if (value.IsNumber()) {
            result = value.GetDouble();
        } else if (value.IsString() && value.GetStringLength() == 3 &&
                   std::string(value.GetString(), value.GetStringLength()) == "NaN") {
            result = std::numeric_limits<double>::quiet_NaN();
        } else if (value.IsString() && std::string(value.GetString(), value.GetStringLength()) == "Infinity") {
            result = std::numeric_limits<double>::infinity();
        } else if (value.IsString() && std::string(value.GetString(), value.GetStringLength()) == "-Infinity") {
            result = -std::numeric_limits<double>::infinity();
        } else {
            return false;
        }
        if (field->cpp_type() == FieldDescriptor::CPPTYPE_DOUBLE) {
            repeated ? reflection->AddDouble(message, field, result) : reflection->SetDouble(message, field, result);
        } else {
            const float float_result = static_cast<float>(result);
            if (std::isfinite(result) && !std::isfinite(float_result)) {
                return false;
            }
            repeated ? reflection->AddFloat(message, field, float_result)
                     : reflection->SetFloat(message, field, float_result);
        }
        return true;
    }
    case FieldDescriptor::CPPTYPE_BOOL:
        if (!value.IsBool()) {
            return false;
        }
        repeated ? reflection->AddBool(message, field, value.GetBool())
                 : reflection->SetBool(message, field, value.GetBool());
        return true;
    case FieldDescriptor::CPPTYPE_ENUM: {
        int number = 0;
        if (value.IsString()) {
            const auto *enum_value =
                field->enum_type()->FindValueByName(std::string(value.GetString(), value.GetStringLength()));
            if (!enum_value) {
                int32_t enum_number = 0;
                if (!ParseIntegerString(value, &enum_number)) {
                    return true; // ignore_unknown_fields also ignores unknown enum names
                }
                enum_value = field->enum_type()->FindValueByNumber(enum_number);
                if (!enum_value) {
                    return true; // protobuf ignores unknown quoted enum numbers
                }
            }
            number = enum_value->number();
        } else if (value.IsInt()) {
            number = value.GetInt();
        } else {
            return false;
        }
        repeated ? reflection->AddEnumValue(message, field, number) : reflection->SetEnumValue(message, field, number);
        return true;
    }
    case FieldDescriptor::CPPTYPE_STRING:
        if (!value.IsString()) {
            return false;
        }
        if (repeated) {
            reflection->AddString(message, field, std::string(value.GetString(), value.GetStringLength()));
        } else {
            reflection->SetString(message, field, std::string(value.GetString(), value.GetStringLength()));
        }
        return true;
    case FieldDescriptor::CPPTYPE_MESSAGE:
        return false;
    }
    return false;
}

bool ParseMapValue(const rapidjson::Value &value, Message *message, const FieldDescriptor *field) {
    if (!value.IsObject()) {
        return false;
    }
    const Descriptor *entry_descriptor = field->message_type();
    const FieldDescriptor *key_field = entry_descriptor->field(0);
    const FieldDescriptor *value_field = entry_descriptor->field(1);
    const Reflection *reflection = message->GetReflection();
    std::unordered_set<std::string> keys;
    for (auto member = value.MemberBegin(); member != value.MemberEnd(); ++member) {
        if (!member->value.IsString()) {
            return false;
        }
        std::string key(member->name.GetString(), member->name.GetStringLength());
        if (!keys.insert(key).second) {
            return false;
        }
        Message *entry = reflection->AddMessage(message, field);
        const Reflection *entry_reflection = entry->GetReflection();
        entry_reflection->SetString(entry, key_field, key);
        entry_reflection->SetString(
            entry, value_field, std::string(member->value.GetString(), member->value.GetStringLength()));
    }
    return true;
}

bool ParseFieldValue(
    const rapidjson::Value &value, Message *message, const FieldDescriptor *field, bool repeated, int depth) {
    if (field->cpp_type() != FieldDescriptor::CPPTYPE_MESSAGE) {
        return ParseScalarValue(value, message, field, repeated);
    }

    const WrapperKind kind = GetWrapperKind(field->message_type());
    Message *nested = repeated ? message->GetReflection()->AddMessage(message, field)
                               : message->GetReflection()->MutableMessage(message, field);
    return kind == WrapperKind::kNone ? ParseMessageValue(value, nested, depth + 1)
                                      : ParseWrapperValue(value, nested, kind);
}

bool ParseMessageValue(const rapidjson::Value &value, Message *message, int depth) {
    if (!value.IsObject() || depth > kMaxNestingDepth) {
        return false;
    }
    const Descriptor *descriptor = message->GetDescriptor();
    const Reflection *reflection = message->GetReflection();
    std::vector<bool> seen_fields(descriptor->field_count(), false);
    std::vector<const FieldDescriptor *> seen_oneofs(descriptor->oneof_decl_count(), nullptr);

    for (auto member = value.MemberBegin(); member != value.MemberEnd(); ++member) {
        const FieldDescriptor *field =
            FindJsonField(descriptor, member->name.GetString(), member->name.GetStringLength());
        if (!field) {
            continue;
        }
        if (seen_fields[field->index()]) {
            return false;
        }
        seen_fields[field->index()] = true;
        if (member->value.IsNull()) {
            continue;
        }

        if (const auto *oneof = field->containing_oneof()) {
            const FieldDescriptor *&seen = seen_oneofs[oneof->index()];
            if (seen && seen != field) {
                return false;
            }
            seen = field;
        }
        if (field->is_map()) {
            if (!ParseMapValue(member->value, message, field)) {
                return false;
            }
            continue;
        }
        if (field->is_repeated()) {
            if (!member->value.IsArray()) {
                return false;
            }
            reflection->ClearField(message, field);
            for (auto item = member->value.Begin(); item != member->value.End(); ++item) {
                if (item->IsNull() || !ParseFieldValue(*item, message, field, true, depth)) {
                    return false;
                }
            }
            continue;
        }
        if (!ParseFieldValue(member->value, message, field, false, depth)) {
            return false;
        }
    }
    return true;
}

} // namespace

bool FastProtoJsonCodec::Supports(const Descriptor *descriptor) {
    std::unordered_set<const Descriptor *> visited;
    return SupportsDescriptor(descriptor, visited);
}

bool FastProtoJsonCodec::TryToJson(const Message &message, std::string &json) {
    if (!Supports(message.GetDescriptor())) {
        return false;
    }
    rapidjson::StringBuffer buffer;
    JsonWriter writer(buffer);
    const WrapperKind kind = GetWrapperKind(message.GetDescriptor());
    const bool success =
        kind == WrapperKind::kNone ? WriteMessage(message, writer, 0) : WriteWrapper(message, kind, writer);
    if (!success || !writer.IsComplete()) {
        return false;
    }
    // protobuf 3.8's MessageToJsonString appends through StringOutputStream;
    // preserve that observable behavior for callers reusing an output string.
    json.append(buffer.GetString(), buffer.GetSize());
    return true;
}

bool FastProtoJsonCodec::TryFromJson(std::string_view json, Message *message) {
    if (!message || !Supports(message->GetDescriptor())) {
        return false;
    }
    rapidjson::Document document;
    DepthLimitedDomGenerator generator(json);
    document.Populate(generator);
    if (!generator.success()) {
        return false;
    }

    // Parse transactionally so a rejected fast-path shape and a failing
    // protobuf fallback preserve the caller's original message. Allocating
    // the temporary on the same arena keeps the successful swap cheap for
    // HTTP request-scoped messages.
    auto *arena = message->GetArena();
    Message *parsed = message->New(arena);
    std::unique_ptr<Message> heap_parsed(arena ? nullptr : parsed);
    const WrapperKind kind = GetWrapperKind(parsed->GetDescriptor());
    const bool success =
        kind == WrapperKind::kNone ? ParseMessageValue(document, parsed, 0) : ParseWrapperValue(document, parsed, kind);
    if (!success) {
        return false;
    }
    message->GetReflection()->Swap(message, parsed);
    return true;
}

} // namespace kv_cache_manager
