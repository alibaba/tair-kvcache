#include "tools/kvcm_swarm/evidence/json_writer.h"

#include <cmath>
#include <variant>

#include "rapidjson/prettywriter.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"

namespace kvcm_swarm {

struct JsonWriter::Impl {
    rapidjson::StringBuffer buffer;
    std::variant<rapidjson::Writer<rapidjson::StringBuffer>, rapidjson::PrettyWriter<rapidjson::StringBuffer>> writer;

    explicit Impl(bool pretty)
        : writer(pretty ? decltype(writer)(std::in_place_index<1>, buffer)
                        : decltype(writer)(std::in_place_index<0>, buffer)) {}

    template <typename Fn>
    void Apply(Fn &&fn) {
        std::visit([&fn](auto &w) { fn(w); }, writer);
    }
};

JsonWriter::JsonWriter(bool pretty) : impl_(std::make_unique<Impl>(pretty)) {}
JsonWriter::~JsonWriter() = default;

void JsonWriter::BeginObject() {
    impl_->Apply([](auto &w) { w.StartObject(); });
}
void JsonWriter::EndObject() {
    impl_->Apply([](auto &w) { w.EndObject(); });
}
void JsonWriter::BeginArray() {
    impl_->Apply([](auto &w) { w.StartArray(); });
}
void JsonWriter::EndArray() {
    impl_->Apply([](auto &w) { w.EndArray(); });
}
void JsonWriter::Key(std::string_view key) {
    impl_->Apply([&key](auto &w) { w.Key(key.data(), static_cast<rapidjson::SizeType>(key.size())); });
}
void JsonWriter::String(std::string_view value) {
    impl_->Apply([&value](auto &w) { w.String(value.data(), static_cast<rapidjson::SizeType>(value.size())); });
}
void JsonWriter::Int(int64_t value) {
    impl_->Apply([value](auto &w) { w.Int64(value); });
}
void JsonWriter::Uint(uint64_t value) {
    impl_->Apply([value](auto &w) { w.Uint64(value); });
}
void JsonWriter::Double(double value) {
    // JSON has no NaN/Inf; report them as null so the schema stays parseable.
    if (std::isnan(value) || std::isinf(value)) {
        impl_->Apply([](auto &w) { w.Null(); });
        return;
    }
    impl_->Apply([value](auto &w) { w.Double(value); });
}
void JsonWriter::Bool(bool value) {
    impl_->Apply([value](auto &w) { w.Bool(value); });
}
void JsonWriter::Null() {
    impl_->Apply([](auto &w) { w.Null(); });
}
void JsonWriter::RawValue(std::string_view json) {
    if (json.empty()) {
        Null();
        return;
    }
    impl_->Apply([&json](auto &w) { w.RawValue(json.data(), json.size(), rapidjson::kObjectType); });
}

std::string JsonWriter::Take() { return std::string(impl_->buffer.GetString(), impl_->buffer.GetSize()); }

std::string JsonQuote(std::string_view value) {
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    writer.String(value.data(), static_cast<rapidjson::SizeType>(value.size()));
    return std::string(buffer.GetString(), buffer.GetSize());
}

} // namespace kvcm_swarm
