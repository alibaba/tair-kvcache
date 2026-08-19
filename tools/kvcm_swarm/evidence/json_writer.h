// Thin streaming JSON writer.
//
// The stable JSON report and the human-readable summary are rendered from the
// same report model; this writer is the only place that formats JSON.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

namespace kvcm_swarm {

class JsonWriter {
public:
    explicit JsonWriter(bool pretty);
    ~JsonWriter();

    JsonWriter(const JsonWriter &) = delete;
    JsonWriter &operator=(const JsonWriter &) = delete;

    void BeginObject();
    void EndObject();
    void BeginArray();
    void EndArray();
    void Key(std::string_view key);
    void String(std::string_view value);
    void Int(int64_t value);
    void Uint(uint64_t value);
    void Double(double value);
    void Bool(bool value);
    void Null();
    // Emits an already-serialised JSON fragment verbatim.
    void RawValue(std::string_view json);

    void KeyString(std::string_view key, std::string_view value) {
        Key(key);
        String(value);
    }
    void KeyInt(std::string_view key, int64_t value) {
        Key(key);
        Int(value);
    }
    void KeyUint(std::string_view key, uint64_t value) {
        Key(key);
        Uint(value);
    }
    void KeyDouble(std::string_view key, double value) {
        Key(key);
        Double(value);
    }
    void KeyBool(std::string_view key, bool value) {
        Key(key);
        Bool(value);
    }

    std::string Take();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace kvcm_swarm
