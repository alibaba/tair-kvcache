#pragma once

#include <string>

#include "kv_cache_manager/event/base_event.h"
#include "kv_cache_manager/manager/cache_location.h"

namespace kv_cache_manager {
namespace optimizer_event {
using KeyType = int64_t;
using KeyVector = std::vector<KeyType>;
using TokenIds = int64_t;
using TokenIdsVector = std::vector<TokenIds>;
} // namespace optimizer_event

class CacheGetEvent : public BaseEvent {
public:
    CacheGetEvent(const std::string &source) : BaseEvent(source, "cache_manager", "GetCacheLocation") {}

    // get and set
    const std::string &query_type() const { return query_type_; }
    const optimizer_event::KeyVector &get_keys() const { return keys_; }
    const optimizer_event::TokenIdsVector &get_tokens() const { return tokens_; }
    const BlockMask &get_block_mask() const { return block_mask_; }
    int32_t sw_size() const { return sw_size_; }
    const std::vector<std::string> &location_spec_names() const { return location_spec_names_; }

    void SetAddtionalArgs(const std::string &query_type,
                          const optimizer_event::KeyVector &keys,
                          const optimizer_event::TokenIdsVector &tokens,
                          const BlockMask &block_mask,
                          int32_t sw_size,
                          const std::vector<std::string> &location_spec_names) {
        query_type_ = query_type;
        keys_ = keys;
        tokens_ = tokens;
        block_mask_ = block_mask;
        sw_size_ = sw_size;
        location_spec_names_ = location_spec_names;
    }
    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override {
        BaseEvent::ToRapidWriter(writer);

        Put(writer, "query_type", query_type_);
        Put(writer, "keys", keys_);
        Put(writer, "tokens", tokens_);
        PutBlockMask(writer, "block_mask", block_mask_);
        Put(writer, "sw_size", sw_size_);
        Put(writer, "location_spec_names", location_spec_names_);
    }

private:
    std::string query_type_;
    optimizer_event::KeyVector keys_;
    optimizer_event::TokenIdsVector tokens_;
    BlockMask block_mask_;
    int32_t sw_size_{0};
    std::vector<std::string> location_spec_names_;
};

class StartWriteCacheEvent : public BaseEvent {
public:
    StartWriteCacheEvent(const std::string &source) : BaseEvent(source, "cache_manager", "StartWriteCache") {}
    // get and set
    const std::string &write_session_id() const { return write_session_id_; }
    void set_write_session_id(const std::string &id) { write_session_id_ = id; }

    const optimizer_event::KeyVector &keys() const { return keys_; }
    const BlockMask &block_mask() const { return block_mask_; }
    const optimizer_event::TokenIdsVector &tokens() const { return tokens_; }
    const std::vector<std::string> &location_spec_group_names() const { return location_spec_group_names_; }
    int64_t write_timeout_seconds() const { return write_timeout_seconds_; }

    void SetAddtionalArgs(const std::string &write_session_id,
                          const optimizer_event::KeyVector &keys,
                          const optimizer_event::TokenIdsVector &tokens,
                          const BlockMask &block_mask,
                          const std::vector<std::string> &location_spec_group_names,
                          int64_t write_timeout_seconds) {
        write_session_id_ = write_session_id;
        keys_ = keys;
        block_mask_ = block_mask;
        tokens_ = tokens;
        location_spec_group_names_ = location_spec_group_names;
        write_timeout_seconds_ = write_timeout_seconds;
    }

    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override {
        BaseEvent::ToRapidWriter(writer);
        Put(writer, "write_session_id", write_session_id_);
        Put(writer, "keys", keys_);
        Put(writer, "tokens", tokens_);
        PutBlockMask(writer, "block_mask", block_mask_);
        Put(writer, "location_spec_group_names", location_spec_group_names_);
        Put(writer, "write_timeout_seconds", write_timeout_seconds_);
    }

private:
    std::string write_session_id_;
    optimizer_event::KeyVector keys_;
    optimizer_event::TokenIdsVector tokens_;
    BlockMask block_mask_;
    std::vector<std::string> location_spec_group_names_;
    int64_t write_timeout_seconds_{0};
};

class FinishWriteCacheEvent : public BaseEvent {
public:
    FinishWriteCacheEvent(const std::string &source) : BaseEvent(source, "cache_manager", "FinishWriteCache") {}
    // get and set
    const std::string &write_session_id() const { return write_session_id_; }
    const BlockMask &success_block() const { return success_block_; }

    void SetAddtionalArgs(const std::string &write_session_id, const BlockMask &success_block) {
        write_session_id_ = write_session_id;
        success_block_ = success_block;
    }
    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override {
        BaseEvent::ToRapidWriter(writer);
        Put(writer, "write_session_id", write_session_id_);
        PutBlockMask(writer, "success_block", success_block_);
    }

private:
    std::string write_session_id_;
    BlockMask success_block_;
};
} // namespace kv_cache_manager