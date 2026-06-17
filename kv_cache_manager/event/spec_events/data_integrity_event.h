#pragma once

#include <cstdint>
#include <string>

#include "kv_cache_manager/event/base_event.h"

namespace kv_cache_manager {

// 任务 82620492：读端发现 expected_hash != actual_hash 时发布的事件。
// 用于审计 / 反向追溯 (哪条 block / 哪个 spec / 哪台机器最常出现读错)。
// source 由调用方传入 (通常是 client 的 trace_id)，body 描述具体 mismatch
// 上下文。本期 client 端事件发布通路在 commit 8 接入。
class ChecksumMismatchEvent : public BaseEvent {
public:
    explicit ChecksumMismatchEvent(const std::string &source)
        : BaseEvent(source, "data_integrity", "ChecksumMismatch") {}

    void SetAdditionalArgs(const std::string &instance_id,
                           std::int64_t block_key,
                           std::int64_t expected_hash,
                           std::int64_t actual_hash,
                           const std::string &spec_name,
                           const std::string &storage_uri) {
        instance_id_ = instance_id;
        block_key_ = block_key;
        expected_hash_ = expected_hash;
        actual_hash_ = actual_hash;
        spec_name_ = spec_name;
        storage_uri_ = storage_uri;
    }

    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override {
        BaseEvent::ToRapidWriter(writer);
        Put(writer, "instance_id", instance_id_);
        Put(writer, "block_key", block_key_);
        Put(writer, "expected_hash", expected_hash_);
        Put(writer, "actual_hash", actual_hash_);
        Put(writer, "spec_name", spec_name_);
        Put(writer, "storage_uri", storage_uri_);
    }

    const std::string &instance_id() const { return instance_id_; }
    std::int64_t block_key() const { return block_key_; }
    std::int64_t expected_hash() const { return expected_hash_; }
    std::int64_t actual_hash() const { return actual_hash_; }
    const std::string &spec_name() const { return spec_name_; }
    const std::string &storage_uri() const { return storage_uri_; }

private:
    std::string instance_id_;
    std::int64_t block_key_ = 0;
    std::int64_t expected_hash_ = 0;
    std::int64_t actual_hash_ = 0;
    std::string spec_name_;
    std::string storage_uri_;
};

} // namespace kv_cache_manager
