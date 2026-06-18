#pragma once

#include <cstdint>
#include <string>

#include "kv_cache_manager/event/base_event.h"

namespace kv_cache_manager {

// Emitted by the reader when expected_checksum != actual_checksum on a block. Used for
// auditing and reverse triage (which block / spec / host hit the most mismatches).
// source is supplied by the caller (typically the client trace_id).
class ChecksumMismatchEvent : public BaseEvent {
public:
    explicit ChecksumMismatchEvent(const std::string &source)
        : BaseEvent(source, "data_integrity", "ChecksumMismatch") {}

    void SetAdditionalArgs(const std::string &instance_id,
                           std::int64_t block_key,
                           std::int64_t expected_checksum,
                           std::int64_t actual_checksum,
                           const std::string &spec_name,
                           const std::string &storage_uri) {
        instance_id_ = instance_id;
        block_key_ = block_key;
        expected_checksum_ = expected_checksum;
        actual_checksum_ = actual_checksum;
        spec_name_ = spec_name;
        storage_uri_ = storage_uri;
    }

    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override {
        BaseEvent::ToRapidWriter(writer);
        Put(writer, "instance_id", instance_id_);
        Put(writer, "block_key", block_key_);
        Put(writer, "expected_checksum", expected_checksum_);
        Put(writer, "actual_checksum", actual_checksum_);
        Put(writer, "spec_name", spec_name_);
        Put(writer, "storage_uri", storage_uri_);
    }

    const std::string &instance_id() const { return instance_id_; }
    std::int64_t block_key() const { return block_key_; }
    std::int64_t expected_checksum() const { return expected_checksum_; }
    std::int64_t actual_checksum() const { return actual_checksum_; }
    const std::string &spec_name() const { return spec_name_; }
    const std::string &storage_uri() const { return storage_uri_; }

private:
    std::string instance_id_;
    std::int64_t block_key_ = 0;
    std::int64_t expected_checksum_ = 0;
    std::int64_t actual_checksum_ = 0;
    std::string spec_name_;
    std::string storage_uri_;
};

} // namespace kv_cache_manager
