#pragma once
#include "kv_cache_manager/common/jsonizable.h"
#include "kv_cache_manager/manager/cache_location.h"

namespace kv_cache_manager {
// 基础的Optimizer Schema Trace
class OptimizerSchemaTrace : public Jsonizable {
public:
    OptimizerSchemaTrace() = default;
    ~OptimizerSchemaTrace() override = default;
    bool FromRapidValue(const rapidjson::Value &rapid_value) override {
        KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "instance_id", instance_id_, std::string(""));
        KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "trace_id", trace_id_, std::string(""));
        KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "timestamp_us", timestamp_us_, int64_t(0));
        KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "tokens", tokens_, std::vector<int64_t>{});
        KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "keys", keys_, std::vector<int64_t>{});
        return true;
    };
    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override {
        Put(writer, "instance_id", instance_id_);
        Put(writer, "trace_id", trace_id_);
        Put(writer, "timestamp_us", timestamp_us_);
        Put(writer, "tokens", tokens_);
        Put(writer, "keys", keys_);
    };

public:
    const std::string &instance_id() const { return instance_id_; }
    const std::string &trace_id() const { return trace_id_; }
    int64_t timestamp_us() const { return timestamp_us_; }
    const std::vector<int64_t> &keys() const { return keys_; }
    const std::vector<int64_t> &tokens() const { return tokens_; }
    void set_instance_id(const std::string &instance_id) { instance_id_ = instance_id; }
    void set_trace_id(const std::string &trace_id) { trace_id_ = trace_id; }
    void set_timestamp_us(int64_t timestamp_us) { timestamp_us_ = timestamp_us; }
    void set_keys(const std::vector<int64_t> &keys) { keys_ = keys; }
    void set_tokens(const std::vector<int64_t> &tokens) { tokens_ = tokens; }

private:
    std::string instance_id_;
    std::string trace_id_;
    int64_t timestamp_us_;
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
    bool FromRapidValue(const rapidjson::Value &rapid_value) override {
        // 调用基类的FromRapidValue解析基础字段
        return OptimizerSchemaTrace::FromRapidValue(rapid_value);
    }
    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override {
        OptimizerSchemaTrace::ToRapidWriter(writer);
    }
};
// 整个对话轮次的Trace，包含输入输出信息
// 目前直接服务于算力画像的输入
class DialogTurnSchemaTrace : public GetLocationSchemaTrace {
public:
    DialogTurnSchemaTrace() = default;
    // 添加接受GetLocationSchemaTrace参数的构造函数
    explicit DialogTurnSchemaTrace(const GetLocationSchemaTrace &other) {
        // 复制基类GetLocationSchemaTrace的成员
        set_instance_id(other.instance_id());
        set_timestamp_us(other.timestamp_us());
        set_keys(other.keys());
        set_tokens(other.tokens());
        set_query_type(other.query_type());
        set_location_spec_names(other.location_spec_names());
        set_block_mask(other.block_mask());
        set_sw_size(other.sw_size());

        // 初始化自己的成员变量
        input_len_ = 0;
        output_len_ = 0;
        total_keys_ = other.keys();
    }
    explicit DialogTurnSchemaTrace(const std::shared_ptr<GetLocationSchemaTrace> &other_ptr) {
        // 复制基类GetLocationSchemaTrace的成员
        set_instance_id(other_ptr->instance_id());
        set_timestamp_us(other_ptr->timestamp_us());
        set_keys(other_ptr->keys());
        set_tokens(other_ptr->tokens());
        set_query_type(other_ptr->query_type());
        set_location_spec_names(other_ptr->location_spec_names());
        set_block_mask(other_ptr->block_mask());
        set_sw_size(other_ptr->sw_size());

        // 初始化自己的成员变量
        input_len_ = 0;
        output_len_ = 0;
        total_keys_ = other_ptr->keys();
    }
    int64_t input_len() const { return input_len_; }
    int64_t output_len() const { return output_len_; }
    const std::vector<int64_t> &total_keys() const { return total_keys_; }
    void set_input_len(int64_t input_len) { input_len_ = input_len; }
    void set_output_len(int64_t output_len) { output_len_ = output_len; }
    void set_total_keys(const std::vector<int64_t> &total_keys) { total_keys_ = total_keys; }
    bool FromRapidValue(const rapidjson::Value &rapid_value) override {
        // 先调用基类的FromRapidValue
        if (!GetLocationSchemaTrace::FromRapidValue(rapid_value)) {
            return false;
        }
        // 解析自己的字段
        KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "input_len", input_len_, int64_t(0));
        KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "output_len", output_len_, int64_t(0));
        KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "total_keys", total_keys_, std::vector<int64_t>{});
        return true;
    }
    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override {
        GetLocationSchemaTrace::ToRapidWriter(writer);
        Put(writer, "input_len", input_len_);
        Put(writer, "output_len", output_len_);
        Put(writer, "total_keys", total_keys_);
    }

private:
    int64_t input_len_;
    int64_t output_len_;
    std::vector<int64_t> total_keys_;
};
} // namespace kv_cache_manager