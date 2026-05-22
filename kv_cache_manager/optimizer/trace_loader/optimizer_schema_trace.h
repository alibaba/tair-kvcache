#pragma once
#include <algorithm>
#include <cstdint>
#include <limits>

#include "kv_cache_manager/common/jsonizable.h"
#include "kv_cache_manager/meta/cache_location.h"

namespace kv_cache_manager {
inline bool ParseOptimizerInt64AllowUint64(const rapidjson::Value &value, int64_t &parsed_value) {
    if (value.IsInt64()) {
        parsed_value = value.GetInt64();
        return true;
    }
    if (value.IsUint64()) {
        const auto unsigned_value = value.GetUint64();
        if (unsigned_value > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
            return false;
        }
        parsed_value = static_cast<int64_t>(unsigned_value);
        return true;
    }
    if (value.IsInt()) {
        parsed_value = static_cast<int64_t>(value.GetInt());
        return true;
    }
    if (value.IsUint()) {
        parsed_value = static_cast<int64_t>(value.GetUint());
        return true;
    }
    return false;
}

inline bool ParseOptimizerInt64VectorAllowUint64(const rapidjson::Value &rapid_value,
                                                 const char *key,
                                                 std::vector<int64_t> &values,
                                                 const std::vector<int64_t> &default_values = {}) {
    if (!rapid_value.HasMember(key)) {
        values = default_values;
        return true;
    }
    const auto &array_value = rapid_value[key];
    if (!array_value.IsArray()) {
        return false;
    }
    values.clear();
    values.reserve(array_value.Size());
    for (const auto &value : array_value.GetArray()) {
        int64_t parsed_value = 0;
        if (!ParseOptimizerInt64AllowUint64(value, parsed_value)) {
            return false;
        }
        values.push_back(parsed_value);
    }
    return true;
}

// 基础的Optimizer Schema Trace
class OptimizerSchemaTrace : public Jsonizable {
public:
    OptimizerSchemaTrace() = default;
    ~OptimizerSchemaTrace() override = default;
    bool FromRapidValue(const rapidjson::Value &rapid_value) override {
        KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "instance_id", instance_id_, std::string(""));
        KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "trace_id", trace_id_, std::string(""));
        timestamp_ns_ = 0;
        if (!rapid_value.HasMember("timestamp_ns")) {
            return false;
        }
        const auto &timestamp_value = rapid_value["timestamp_ns"];
        if (!ParseOptimizerInt64AllowUint64(timestamp_value, timestamp_ns_)) {
            return false;
        }
        if (!ParseOptimizerInt64VectorAllowUint64(rapid_value, "tokens", tokens_)) {
            return false;
        }
        if (!ParseOptimizerInt64VectorAllowUint64(rapid_value, "keys", keys_)) {
            return false;
        }
        input_len_ = -1;
        if (!rapid_value.HasMember("input_len")) {
            return false;
        }
        const auto &input_len_value = rapid_value["input_len"];
        if (!ParseOptimizerInt64AllowUint64(input_len_value, input_len_)) {
            return false;
        }
        return true;
    };
    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override {
        Put(writer, "instance_id", instance_id_);
        Put(writer, "trace_id", trace_id_);
        Put(writer, "timestamp_ns", timestamp_ns_);
        Put(writer, "tokens", tokens_);
        Put(writer, "keys", keys_);
        if (input_len_ >= 0) {
            Put(writer, "input_len", input_len_);
        }
    };

public:
    const std::string &instance_id() const { return instance_id_; }
    const std::string &trace_id() const { return trace_id_; }
    int64_t timestamp_ns() const { return timestamp_ns_; }
    const std::vector<int64_t> &keys() const { return keys_; }
    const std::vector<int64_t> &tokens() const { return tokens_; }
    int64_t input_len() const { return input_len_; }
    size_t full_block_count(size_t block_size) const {
        if (input_len_ >= 0 && block_size > 0) {
            return static_cast<size_t>(std::min<int64_t>(static_cast<int64_t>(keys_.size()), input_len_ / block_size));
        }
        if (!tokens_.empty() && block_size > 0) {
            return std::min(keys_.size(), tokens_.size() / block_size);
        }
        return keys_.size();
    }
    size_t input_token_count(size_t block_size) const {
        if (input_len_ >= 0) {
            return static_cast<size_t>(std::max<int64_t>(input_len_, 0));
        }
        if (!tokens_.empty()) {
            return tokens_.size();
        }
        return keys_.size() * block_size;
    }
    void set_instance_id(const std::string &instance_id) { instance_id_ = instance_id; }
    void set_trace_id(const std::string &trace_id) { trace_id_ = trace_id; }
    void set_timestamp_ns(int64_t timestamp_ns) { timestamp_ns_ = timestamp_ns; }
    void set_keys(const std::vector<int64_t> &keys) { keys_ = keys; }
    void set_tokens(const std::vector<int64_t> &tokens) { tokens_ = tokens; }
    void set_input_len(int64_t input_len) { input_len_ = input_len; }

private:
    std::string instance_id_;
    std::string trace_id_;
    int64_t timestamp_ns_;
    std::vector<int64_t> keys_;
    std::vector<int64_t> tokens_;
    int64_t input_len_ = -1;
};
// GetCacheLocation事件的Trace
// 只包含读取缓存相关的信息
class GetLocationSchemaTrace : public OptimizerSchemaTrace {
public:
    const std::string &query_type() const { return query_type_; }
    const std::vector<std::string> &location_spec_names() const { return location_spec_names_; }
    const BlockMask &block_mask() const { return block_mask_; }
    int32_t sw_size() const { return sw_size_; }
    void set_query_type(const std::string &query_type) { query_type_ = query_type; }
    void set_location_spec_names(const std::vector<std::string> &location_spec_names) {
        location_spec_names_ = location_spec_names;
    }
    void set_block_mask(const BlockMask &block_mask) { block_mask_ = block_mask; }
    void set_sw_size(int32_t sw_size) { sw_size_ = sw_size; }
    bool FromRapidValue(const rapidjson::Value &rapid_value) override {
        // 先调用基类的FromRapidValue
        if (!OptimizerSchemaTrace::FromRapidValue(rapid_value)) {
            return false;
        }
        // 解析自己的字段
        KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "query_type", query_type_, std::string("prefix_match"));
        KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "sw_size", sw_size_, int32_t(0));
        KVCM_JSON_GET_DEFAULT_MACRO(
            rapid_value, "location_spec_names", location_spec_names_, std::vector<std::string>{});

        // 解析block_mask字段
        if (rapid_value.HasMember("block_mask")) {
            const auto &block_mask_value = rapid_value["block_mask"];
            if (block_mask_value.IsArray()) {
                BlockMaskVector block_mask_vector;
                for (const auto &val : block_mask_value.GetArray()) {
                    if (!val.IsBool()) {
                        return false;
                    }
                    block_mask_vector.push_back(val.GetBool());
                }
                block_mask_ = block_mask_vector;
            } else {
                int64_t block_mask_offset = 0;
                if (!ParseOptimizerInt64AllowUint64(block_mask_value, block_mask_offset) || block_mask_offset < 0) {
                    return false;
                }
                block_mask_ = BlockMaskOffset(block_mask_offset);
            }
        } else {
            // 默认为空的BlockMaskVector
            block_mask_ = BlockMaskVector{};
        }
        return true;
    }
    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override {
        OptimizerSchemaTrace::ToRapidWriter(writer);

        Put(writer, "query_type", query_type_);
        PutBlockMask(writer, "block_mask", block_mask_);
        Put(writer, "sw_size", sw_size_);
        Put(writer, "location_spec_names", location_spec_names_);
    }

private:
    std::string query_type_ = "prefix_match";
    BlockMask block_mask_;
    int32_t sw_size_{0};
    std::vector<std::string> location_spec_names_;
};
// WriteCache事件的Trace
// 只包含写入缓存相关的信息
class WriteCacheSchemaTrace : public OptimizerSchemaTrace {
public:
    int64_t ttl_us() const { return ttl_us_; }
    void set_ttl_us(int64_t ttl_us) { ttl_us_ = ttl_us; }

    bool FromRapidValue(const rapidjson::Value &rapid_value) override {
        if (!OptimizerSchemaTrace::FromRapidValue(rapid_value)) {
            return false;
        }
        ttl_us_ = 0;
        if (rapid_value.HasMember("ttl_us") && !ParseOptimizerInt64AllowUint64(rapid_value["ttl_us"], ttl_us_)) {
            return false;
        }
        return true;
    }
    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override {
        OptimizerSchemaTrace::ToRapidWriter(writer);
        Put(writer, "ttl_us", ttl_us_);
    }

private:
    int64_t ttl_us_ = 0; // 0 = 使用 group 默认, -1 = 禁用
};
} // namespace kv_cache_manager
