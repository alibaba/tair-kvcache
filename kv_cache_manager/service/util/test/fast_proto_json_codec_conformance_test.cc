#include <cstdint>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/message.h>
#include <google/protobuf/util/json_util.h>
#include <google/protobuf/util/message_differencer.h>
#include <limits>
#include <memory>
#include <rapidjson/document.h>
#include <string>
#include <string_view>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/protocol/protobuf/admin_service.pb.h"
#include "kv_cache_manager/protocol/protobuf/debug_service.pb.h"
#include "kv_cache_manager/protocol/protobuf/kv_meta_service.pb.h"
#include "kv_cache_manager/protocol/protobuf/meta_service.pb.h"
#include "kv_cache_manager/protocol/protobuf/optimizer_service.pb.h"
#include "service/util/fast_proto_json_codec.h"

namespace kv_cache_manager {
namespace {

using ::google::protobuf::Descriptor;
using ::google::protobuf::FieldDescriptor;
using ::google::protobuf::FileDescriptor;
using ::google::protobuf::Message;
using ::google::protobuf::Reflection;

constexpr int kPopulatedCaseCount = 8;
constexpr int kGeneratedMessageDepth = 3;

// This list is the only manual coverage registry. Everything else is derived
// from descriptors so adding a message or field to a registered file becomes
// part of the differential test automatically.
const FileDescriptor *const kProtocolFiles[] = {
    proto::admin::Status::descriptor()->file(),
    proto::debug::Status::descriptor()->file(),
    proto::kv_meta::Status::descriptor()->file(),
    proto::meta::Status::descriptor()->file(),
    proto::optimizer::Status::descriptor()->file(),
};

uint32_t DeriveCase(uint32_t test_case, int field_number, int repeated_index = -1) {
    return test_case * 37U + static_cast<uint32_t>(field_number) * 17U +
           static_cast<uint32_t>(repeated_index + 1) * 13U;
}

std::string StringValue(uint32_t test_case, int repeated_index) {
    const std::string suffix = ":" + std::to_string(repeated_index);
    switch (test_case % kPopulatedCaseCount) {
    case 0:
        return "plain" + suffix;
    case 1:
        return "";
    case 2:
        return "quote:\" slash:\\ newline:\n" + suffix;
    case 3:
        return "unicode:雪" + suffix;
    case 4:
        return std::string("nul\0value", 9) + suffix;
    case 5:
        return "</script><value>" + suffix;
    case 6:
        return "18446744073709551615" + suffix;
    default:
        return " spaces and tabs\t" + suffix;
    }
}

void SetScalarValue(Message *message, const FieldDescriptor *field, uint32_t test_case, bool repeated) {
    const Reflection *reflection = message->GetReflection();
    switch (field->cpp_type()) {
    case FieldDescriptor::CPPTYPE_INT32: {
        constexpr int32_t values[] = {
            0, 1, -1, std::numeric_limits<int32_t>::min(), std::numeric_limits<int32_t>::max(), 42, -42, 1000000};
        const int32_t value = values[test_case % kPopulatedCaseCount];
        repeated ? reflection->AddInt32(message, field, value) : reflection->SetInt32(message, field, value);
        return;
    }
    case FieldDescriptor::CPPTYPE_UINT32: {
        constexpr uint32_t values[] = {0, 1, std::numeric_limits<uint32_t>::max(), 42, 1000000, 7, 255, 65535};
        const uint32_t value = values[test_case % kPopulatedCaseCount];
        repeated ? reflection->AddUInt32(message, field, value) : reflection->SetUInt32(message, field, value);
        return;
    }
    case FieldDescriptor::CPPTYPE_INT64: {
        constexpr int64_t values[] = {0,
                                      1,
                                      -1,
                                      std::numeric_limits<int64_t>::min(),
                                      std::numeric_limits<int64_t>::max(),
                                      9007199254740991LL,
                                      -9007199254740991LL,
                                      1000000000000LL};
        const int64_t value = values[test_case % kPopulatedCaseCount];
        repeated ? reflection->AddInt64(message, field, value) : reflection->SetInt64(message, field, value);
        return;
    }
    case FieldDescriptor::CPPTYPE_UINT64: {
        constexpr uint64_t values[] = {0,
                                       1,
                                       std::numeric_limits<uint64_t>::max(),
                                       42,
                                       9007199254740991ULL,
                                       9007199254740992ULL,
                                       1000000000000ULL,
                                       4294967296ULL};
        const uint64_t value = values[test_case % kPopulatedCaseCount];
        repeated ? reflection->AddUInt64(message, field, value) : reflection->SetUInt64(message, field, value);
        return;
    }
    case FieldDescriptor::CPPTYPE_DOUBLE: {
        constexpr double values[] = {0.0,
                                     1.0,
                                     -1.0,
                                     0.1,
                                     -12345.5,
                                     std::numeric_limits<double>::min(),
                                     std::numeric_limits<double>::max(),
                                     9007199254740992.0};
        const double value = values[test_case % kPopulatedCaseCount];
        repeated ? reflection->AddDouble(message, field, value) : reflection->SetDouble(message, field, value);
        return;
    }
    case FieldDescriptor::CPPTYPE_FLOAT: {
        constexpr float values[] = {0.0F,
                                    1.0F,
                                    -1.0F,
                                    0.1F,
                                    -12345.5F,
                                    std::numeric_limits<float>::min(),
                                    std::numeric_limits<float>::max(),
                                    16777216.0F};
        const float value = values[test_case % kPopulatedCaseCount];
        repeated ? reflection->AddFloat(message, field, value) : reflection->SetFloat(message, field, value);
        return;
    }
    case FieldDescriptor::CPPTYPE_BOOL: {
        const bool value = test_case % 2 != 0;
        repeated ? reflection->AddBool(message, field, value) : reflection->SetBool(message, field, value);
        return;
    }
    case FieldDescriptor::CPPTYPE_ENUM: {
        const auto *enum_type = field->enum_type();
        const int value = enum_type->value(test_case % enum_type->value_count())->number();
        repeated ? reflection->AddEnumValue(message, field, value) : reflection->SetEnumValue(message, field, value);
        return;
    }
    case FieldDescriptor::CPPTYPE_STRING: {
        const std::string value = StringValue(test_case, repeated ? 1 : -1);
        repeated ? reflection->AddString(message, field, value) : reflection->SetString(message, field, value);
        return;
    }
    case FieldDescriptor::CPPTYPE_MESSAGE:
        return;
    }
}

void PopulateMessage(Message *message, uint32_t test_case, int depth);

void PopulateMap(Message *message, const FieldDescriptor *field, uint32_t test_case) {
    const Reflection *reflection = message->GetReflection();
    const int entry_count = 1 + static_cast<int>(test_case % 2);
    for (int i = 0; i < entry_count; ++i) {
        Message *entry = reflection->AddMessage(message, field);
        const Descriptor *entry_descriptor = entry->GetDescriptor();
        const Reflection *entry_reflection = entry->GetReflection();
        const uint32_t entry_case = DeriveCase(test_case, field->number(), i);
        entry_reflection->SetString(entry, entry_descriptor->field(0), StringValue(entry_case, i));
        entry_reflection->SetString(entry, entry_descriptor->field(1), StringValue(entry_case + 1, i));
    }
}

void PopulateField(Message *message, const FieldDescriptor *field, uint32_t test_case, int depth) {
    if (field->is_map()) {
        PopulateMap(message, field, test_case);
        return;
    }

    const Reflection *reflection = message->GetReflection();
    if (field->is_repeated()) {
        const int value_count = 1 + static_cast<int>(test_case % 2);
        for (int i = 0; i < value_count; ++i) {
            const uint32_t value_case = DeriveCase(test_case, field->number(), i);
            if (field->cpp_type() == FieldDescriptor::CPPTYPE_MESSAGE) {
                if (depth < kGeneratedMessageDepth) {
                    PopulateMessage(reflection->AddMessage(message, field), value_case, depth + 1);
                }
            } else {
                SetScalarValue(message, field, value_case, true);
            }
        }
        return;
    }

    if (field->cpp_type() == FieldDescriptor::CPPTYPE_MESSAGE) {
        if (depth < kGeneratedMessageDepth) {
            PopulateMessage(reflection->MutableMessage(message, field), test_case, depth + 1);
        }
        return;
    }
    SetScalarValue(message, field, test_case, false);
}

void PopulateMessage(Message *message, uint32_t test_case, int depth) {
    const Descriptor *descriptor = message->GetDescriptor();
    for (int i = 0; i < descriptor->field_count(); ++i) {
        const FieldDescriptor *field = descriptor->field(i);
        if (field->containing_oneof()) {
            const auto *oneof = field->containing_oneof();
            if (field != oneof->field(test_case % oneof->field_count())) {
                continue;
            }
        }
        PopulateField(message, field, DeriveCase(test_case, field->number()), depth);
    }
}

bool ProtobufToJson(const Message &message, std::string *json) {
    google::protobuf::util::JsonPrintOptions options;
    options.always_print_primitive_fields = true;
    options.preserve_proto_field_names = true;
    return google::protobuf::util::MessageToJsonString(message, json, options).ok();
}

bool ProtobufFromJson(std::string_view json, Message *message) {
    google::protobuf::util::JsonParseOptions options;
    options.ignore_unknown_fields = true;
    const google::protobuf::StringPiece input(json.data(), static_cast<ptrdiff_t>(json.size()));
    return google::protobuf::util::JsonStringToMessage(input, message, options).ok();
}

bool JsonSemanticallyEquals(std::string_view lhs, std::string_view rhs) {
    rapidjson::Document lhs_document;
    rapidjson::Document rhs_document;
    lhs_document.Parse<rapidjson::kParseValidateEncodingFlag>(lhs.data(), lhs.size());
    rhs_document.Parse<rapidjson::kParseValidateEncodingFlag>(rhs.data(), rhs.size());
    return !lhs_document.HasParseError() && !rhs_document.HasParseError() && lhs_document == rhs_document;
}

void VerifyConformance(const Message &source) {
    std::string protobuf_json;
    std::string fast_json;
    ASSERT_TRUE(ProtobufToJson(source, &protobuf_json));
    ASSERT_TRUE(FastProtoJsonCodec::TryToJson(source, fast_json));

    // Protobuf is the independent oracle. Different escaping and object
    // member order are allowed; parsed JSON values must still be identical.
    EXPECT_TRUE(JsonSemanticallyEquals(protobuf_json, fast_json))
        << "protobuf JSON: " << protobuf_json << "\nfast JSON: " << fast_json;

    std::unique_ptr<Message> protobuf_from_fast(source.New());
    ASSERT_TRUE(ProtobufFromJson(fast_json, protobuf_from_fast.get())) << fast_json;
    EXPECT_TRUE(google::protobuf::util::MessageDifferencer::Equals(source, *protobuf_from_fast));

    std::unique_ptr<Message> fast_from_protobuf(source.New());
    ASSERT_TRUE(FastProtoJsonCodec::TryFromJson(protobuf_json, fast_from_protobuf.get())) << protobuf_json;
    EXPECT_TRUE(google::protobuf::util::MessageDifferencer::Equals(source, *fast_from_protobuf));
}

} // namespace

class FastProtoJsonCodecConformanceTest : public TESTBASE {};

TEST_F(FastProtoJsonCodecConformanceTest, TestAllRegisteredProtocolMessagesMatchProtobufJsonSemantics) {
    int verified_message_cases = 0;
    for (const FileDescriptor *file : kProtocolFiles) {
        SCOPED_TRACE(file->name());
        for (int i = 0; i < file->message_type_count(); ++i) {
            const Descriptor *descriptor = file->message_type(i);
            SCOPED_TRACE(descriptor->full_name());
            ASSERT_TRUE(FastProtoJsonCodec::Supports(descriptor));

            const Message *prototype = google::protobuf::MessageFactory::generated_factory()->GetPrototype(descriptor);
            ASSERT_NE(nullptr, prototype);

            // The empty case covers default values and absent fields. Eight
            // populated cases rotate every current oneof arm and exercise
            // boundary/escaped scalar, repeated, map, wrapper and nested values.
            std::unique_ptr<Message> empty_message(prototype->New());
            VerifyConformance(*empty_message);
            ++verified_message_cases;
            for (uint32_t test_case = 0; test_case < kPopulatedCaseCount; ++test_case) {
                SCOPED_TRACE("populated case " + std::to_string(test_case));
                std::unique_ptr<Message> message(prototype->New());
                PopulateMessage(message.get(), test_case, 0);
                VerifyConformance(*message);
                ++verified_message_cases;
            }
        }
    }
    EXPECT_GT(verified_message_cases, 0);
}

} // namespace kv_cache_manager
