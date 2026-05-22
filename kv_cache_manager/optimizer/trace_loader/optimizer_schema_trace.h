#pragma once
#include "kv_cache_manager/common/jsonizable.h"
#include "kv_cache_manager/meta/cache_location.h"

namespace kv_cache_manager {
// 基础的Optimizer Schema Trace
class OptimizerSchemaTrace : public Jsonizable {
public:
    OptimizerSchemaTrace() = default;
    ~OptimizerSchemaTrace() override = default;
    bool FromRapidValue(const rapidjson::Value &rapid_value) override {
        KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "instance_id", instance_id_, std::string(""));
        KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "trace_id", trace_id_, std::string(""));
        // Optimizer 读 trace 入口：可含 timestamp_ns，或旧数据 timestamp_us（微秒→×1000）；之后全用纳秒
        timestamp_ns_ = 0;
        if (rapid_value.HasMember("timestamp_ns")) {
            const auto &v = rapid_value["timestamp_ns"];
            if (v.IsInt64()) {
                timestamp_ns_ = v.GetInt64();
            } else if (v.IsUint64()) {
                timestamp_ns_ = static_cast<int64_t>(v.GetUint64());
            }
        } else if (rapid_value.HasMember("timestamp_us")) {
            const auto &v = rapid_value["timestamp_us"];
            if (v.IsInt64()) {
                timestamp_ns_ = v.GetInt64() * 1000;
            } else if (v.IsUint64()) {
                timestamp_ns_ = static_cast<int64_t>(v.GetUint64()) * 1000;
            }
        }
        KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "tokens", tokens_, std::vector<int64_t>{});
        KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "keys", keys_, std::vector<int64_t>{});
        return true;
    };
    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override {
        Put(writer, "instance_id", instance_id_);
        Put(writer, "trace_id", trace_id_);
        Put(writer, "timestamp_ns", timestamp_ns_);
        Put(writer, "tokens", tokens_);
        Put(writer, "keys", keys_);
    };

public:
    const std::string &instance_id() const { return instance_id_; }
    const std::string &trace_id() const { return trace_id_; }
    int64_t timestamp_ns() const { return timestamp_ns_; }
    const std::vector<int64_t> &keys() const { return keys_; }
    const std::vector<int64_t> &tokens() const { return tokens_; }
    void set_instance_id(const std::string &instance_id) { instance_id_ = instance_id; }
    void set_trace_id(const std::string &trace_id) { trace_id_ = trace_id; }
    void set_timestamp_ns(int64_t timestamp_ns) { timestamp_ns_ = timestamp_ns; }
    void set_keys(const std::vector<int64_t> &keys) { keys_ = keys; }
    void set_tokens(const std::vector<int64_t> &tokens) { tokens_ = tokens; }

private:
    std::string instance_id_;
    std::string trace_id_;
    int64_t timestamp_ns_;
    std::vector<int64_t> keys_;
    std::vector<int64_t> tokens_;
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
                    if (val.IsBool()) {
                        block_mask_vector.push_back(val.GetBool());
                    }
                }
                block_mask_ = block_mask_vector;
            } else if (block_mask_value.IsInt64()) {
                block_mask_ = BlockMaskOffset(block_mask_value.GetInt64());
            } else {
                // 默认为空的BlockMaskVector
                block_mask_ = BlockMaskVector{};
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
        KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "ttl_us", ttl_us_, int64_t(0));
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