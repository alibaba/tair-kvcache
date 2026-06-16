#pragma once

#include <cstdint>
#include <string>

#include "kv_cache_manager/event/base_event.h"

namespace kv_cache_manager {

// 多层存储迁移事件：与 cache_reclaim_event 同模式，写本地事件日志供离线分析。
// component 统一为 "migration"。

class MigrationSubmittedEvent : public BaseEvent {
public:
    explicit MigrationSubmittedEvent(const std::string &source)
        : BaseEvent(source, "migration", "MigrationSubmitted") {}

    void SetAdditionalArgs(std::int64_t block_key,
                           const std::string &src_storage,
                           const std::string &dst_storage,
                           const std::string &trace_id) {
        block_key_ = block_key;
        src_storage_ = src_storage;
        dst_storage_ = dst_storage;
        trace_id_ = trace_id;
    }

    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override {
        BaseEvent::ToRapidWriter(writer);
        Put(writer, "block_key", block_key_);
        Put(writer, "src_storage", src_storage_);
        Put(writer, "dst_storage", dst_storage_);
        Put(writer, "method", std::string("copy"));
        Put(writer, "trace_id", trace_id_);
    }

private:
    std::int64_t block_key_{0};
    std::string src_storage_;
    std::string dst_storage_;
    std::string trace_id_;
};

class MigrationCompletedEvent : public BaseEvent {
public:
    explicit MigrationCompletedEvent(const std::string &source)
        : BaseEvent(source, "migration", "MigrationCompleted") {}

    void SetAdditionalArgs(std::int64_t block_key,
                           const std::string &src_storage,
                           const std::string &dst_storage,
                           std::int64_t duration_ms,
                           std::uint64_t bytes,
                           bool success,
                           const std::string &fail_reason) {
        block_key_ = block_key;
        src_storage_ = src_storage;
        dst_storage_ = dst_storage;
        duration_ms_ = duration_ms;
        bytes_ = bytes;
        success_ = success;
        fail_reason_ = fail_reason;
    }

    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override {
        BaseEvent::ToRapidWriter(writer);
        Put(writer, "block_key", block_key_);
        Put(writer, "src_storage", src_storage_);
        Put(writer, "dst_storage", dst_storage_);
        Put(writer, "method", std::string("copy"));
        Put(writer, "duration_ms", duration_ms_);
        Put(writer, "bytes", bytes_);
        Put(writer, "success", success_);
        Put(writer, "fail_reason", fail_reason_);
    }

private:
    std::int64_t block_key_{0};
    std::string src_storage_;
    std::string dst_storage_;
    std::int64_t duration_ms_{0};
    std::uint64_t bytes_{0};
    bool success_{false};
    std::string fail_reason_;
};

class MigrationMarkAddEvent : public BaseEvent {
public:
    explicit MigrationMarkAddEvent(const std::string &source) : BaseEvent(source, "migration", "MigrationMarkAdd") {}

    void SetAdditionalArgs(std::int64_t block_key, const std::string &dst_storage) {
        block_key_ = block_key;
        dst_storage_ = dst_storage;
    }

    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override {
        BaseEvent::ToRapidWriter(writer);
        Put(writer, "block_key", block_key_);
        Put(writer, "dst_storage", dst_storage_);
    }

private:
    std::int64_t block_key_{0};
    std::string dst_storage_;
};

class MigrationMarkConsumedEvent : public BaseEvent {
public:
    explicit MigrationMarkConsumedEvent(const std::string &source)
        : BaseEvent(source, "migration", "MigrationMarkConsumed") {}

    void SetAdditionalArgs(std::int64_t block_key, const std::string &dst_storage) {
        block_key_ = block_key;
        dst_storage_ = dst_storage;
    }

    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override {
        BaseEvent::ToRapidWriter(writer);
        Put(writer, "block_key", block_key_);
        Put(writer, "dst_storage", dst_storage_);
    }

private:
    std::int64_t block_key_{0};
    std::string dst_storage_;
};

} // namespace kv_cache_manager
