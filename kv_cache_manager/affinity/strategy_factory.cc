#include "kv_cache_manager/affinity/strategy_factory.h"

#include <cmath>
#include <string>
#include <utility>

#include "kv_cache_manager/affinity/hint_suppressor.h"
#include "kv_cache_manager/affinity/local_replica_strategy.h"
#include "kv_cache_manager/affinity/noop_strategy.h"
#include "kv_cache_manager/affinity/pipeline/candidate_pipeline.h"
#include "kv_cache_manager/common/logger.h"
#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"

namespace kv_cache_manager {

namespace {

constexpr const char *kFieldType = "type";
constexpr const char *kTypeNoop = "noop";
constexpr const char *kTypeLocalReplica = "local_replica";
constexpr const char *kFieldStrategy = "strategy";
constexpr const char *kFieldEnabledAspects = "enabled_aspects";
constexpr const char *kFieldWrite = "write";       // 写一级配置块
constexpr const char *kFieldRead = "read";         // 读一级配置块
constexpr const char *kFieldEviction = "eviction"; // 淘汰一级配置块
constexpr const char *kFieldOps = "ops";           // op 链
constexpr const char *kFieldOnMiss = "on_miss";    // 读子项

void SetErr(std::string *out, const std::string &msg) {
    if (out) {
        *out = msg;
    }
}

std::string ValueToJson(const rapidjson::Value &v) {
    rapidjson::StringBuffer sb;
    rapidjson::Writer<rapidjson::StringBuffer> w(sb);
    v.Accept(w);
    return std::string(sb.GetString(), sb.GetSize());
}

// Missing fields retain defaults; malformed fields must not turn into a valid
// high-priority strategy that masks a lower-priority configuration.
template <typename T>
bool ReadField(const rapidjson::Value &obj, const char *key, T &out, std::string *err) {
    if (!obj.HasMember(key))
        return true;
    if (!obj[key].Is<T>()) {
        SetErr(err, std::string("invalid type or integer range for '") + key + "'");
        return false;
    }
    out = obj[key].Get<T>();
    return true;
}

bool ReadField(const rapidjson::Value &obj, const char *key, double &out, std::string *err) {
    if (!obj.HasMember(key))
        return true;
    const auto &value = obj[key];
    if (!value.IsNumber() || !std::isfinite(value.GetDouble()) || value.GetDouble() < 0) {
        SetErr(err, std::string(key) + " must be a finite non-negative number");
        return false;
    }
    out = value.GetDouble();
    return true;
}

bool CheckObjectField(const rapidjson::Value &obj, const char *key, std::string *err) {
    if (!obj.HasMember(key) || obj[key].IsObject())
        return true;
    SetErr(err, std::string(key) + " must be an object");
    return false;
}

bool ParseWriteOps(const rapidjson::Value &write_block, LocalReplicaAffinityStrategy::Params &p, std::string *err) {
    if (!CheckObjectField(write_block, kFieldOps, err))
        return false;
    if (!write_block.HasMember(kFieldOps))
        return true;
    p.write_pipeline = CandidatePipeline::ParseJsonString(ValueToJson(write_block[kFieldOps]), err);
    return p.write_pipeline != nullptr;
}

bool ApplyOnMiss(const rapidjson::Value &on_miss, LocalReplicaAffinityStrategy::Params &p, std::string *err) {
    return ReadField(on_miss, "enabled", p.enable_on_miss, err) &&
           ReadField(on_miss, "replication_hot_threshold", p.replication_hot_threshold, err) &&
           ReadField(on_miss, "heat_half_life_ms", p.heat_half_life_ms, err) &&
           ReadField(on_miss, "min_benefit_ratio", p.min_benefit_ratio, err) &&
           ReadField(on_miss, "prefix_bonus", p.prefix_bonus, err) &&
           ReadField(on_miss, "max_replication_bytes", p.max_replication_bytes, err) &&
           ReadField(on_miss, "caller_capacity_threshold", p.caller_capacity_threshold, err) &&
           ReadField(on_miss, "caller_capacity_buffer", p.caller_capacity_buffer, err) &&
           ReadField(on_miss, "suppression_window_ms", p.suppression_window_ms, err);
}

bool ApplyEvictionOps(const rapidjson::Value &eviction_block,
                      LocalReplicaAffinityStrategy::Params &p,
                      std::string *err) {
    if (!eviction_block.HasMember(kFieldOps))
        return true;
    if (!eviction_block[kFieldOps].IsArray()) {
        SetErr(err, "eviction.ops must be an array");
        return false;
    }
    bool legacy_critical = false;
    for (const auto &op : eviction_block[kFieldOps].GetArray()) {
        if (!op.IsObject() || !op.HasMember("op") || !op["op"].IsString() ||
            std::string(op["op"].GetString()) != "node_water_level") {
            SetErr(err, "eviction.ops requires node_water_level operations");
            return false;
        }
        if (!ReadField(op, "threshold", p.node_water_threshold, err) || !ReadField(op, "low", p.node_water_low, err))
            return false;
        legacy_critical |= op.HasMember("critical");
    }
    if (legacy_critical) {
        KVCM_LOG_WARN("affinity node_water_level.critical is deprecated and ignored; use threshold and low");
    }
    return true;
}

// 构造 LocalReplicaAffinityStrategy::Params；从 target 对象按一级行为块解析。
//   target 期望形如：
//     { "type":"local_replica",
//       "enabled_aspects": { "write":true, "read":true, "eviction":true },
//       "write":     { "ops": <5 段流水线对象> },
//       "read":      { "on_miss": { "replication_hot_threshold": 3, ... } },
//       "eviction":  { "ops": [{"op":"node_water_level","threshold":0.85}, ...] } }
bool BuildLocalReplicaParams(const rapidjson::Value &target,
                             LocalReplicaAffinityStrategy::Params &p,
                             std::string *err) {
    for (const char *key : {"replica_limits", kFieldEnabledAspects, kFieldWrite, kFieldRead, kFieldEviction}) {
        if (!CheckObjectField(target, key, err))
            return false;
    }
    if (target.HasMember("replica_limits")) {
        const auto &limits = target["replica_limits"];
        if (!ReadField(limits, "max_replicas_per_key", p.replica_limits.max_replicas_per_key, err) ||
            !ReadField(limits, "min_retained_replicas", p.replica_limits.min_retained_replicas, err) ||
            !ReadField(limits, "max_instance_bytes", p.replica_limits.max_instance_bytes, err))
            return false;
    }
    if (target.HasMember(kFieldEnabledAspects)) {
        const auto &ea = target[kFieldEnabledAspects];
        if (!ReadField(ea, "write", p.enable_write, err) || !ReadField(ea, "read", p.enable_read, err) ||
            !ReadField(ea, "eviction", p.enable_eviction, err))
            return false;
    }
    if (target.HasMember(kFieldWrite) && !ParseWriteOps(target[kFieldWrite], p, err))
        return false;
    if (target.HasMember(kFieldRead)) {
        const auto &read_block = target[kFieldRead];
        if (!CheckObjectField(read_block, kFieldOnMiss, err))
            return false;
        if (read_block.HasMember(kFieldOnMiss) && !ApplyOnMiss(read_block[kFieldOnMiss], p, err))
            return false;
    }
    if (target.HasMember(kFieldEviction) && !ApplyEvictionOps(target[kFieldEviction], p, err))
        return false;
    if (p.caller_capacity_threshold > 1 || p.caller_capacity_buffer > p.caller_capacity_threshold) {
        SetErr(err, "caller capacity requires 0 <= buffer <= threshold <= 1");
        return false;
    }
    if (p.node_water_threshold > 1 || p.node_water_low > p.node_water_threshold) {
        SetErr(err, "node water levels require 0 <= low <= threshold <= 1");
        return false;
    }
    if (p.replica_limits.max_replicas_per_key > 0 &&
        p.replica_limits.min_retained_replicas > p.replica_limits.max_replicas_per_key) {
        SetErr(err, "min_retained_replicas exceeds max_replicas_per_key");
        return false;
    }
    return true;
}

} // namespace

std::shared_ptr<AffinityStrategy> StrategyFactory::ParseJsonString(const std::string &json,
                                                                   FrequencySketch *sketch,
                                                                   HintSuppressor *suppressor,
                                                                   std::string *error_msg) {
    if (json.empty()) {
        SetErr(error_msg, "empty json");
        return nullptr;
    }

    rapidjson::Document doc;
    doc.Parse(json.c_str());
    if (doc.HasParseError()) {
        SetErr(error_msg, "json parse error");
        return nullptr;
    }
    if (!doc.IsObject()) {
        SetErr(error_msg, "json root is not object");
        return nullptr;
    }

    const rapidjson::Value *target = &doc;
    if (!CheckObjectField(doc, kFieldStrategy, error_msg))
        return nullptr;
    if (doc.HasMember(kFieldStrategy)) {
        target = &doc[kFieldStrategy];
    }

    if (!target->HasMember(kFieldType) || !(*target)[kFieldType].IsString()) {
        SetErr(error_msg, "missing or non-string 'type' field");
        return nullptr;
    }

    std::string type = (*target)[kFieldType].GetString();
    if (type == kTypeNoop) {
        return std::make_shared<NoopAffinityStrategy>();
    }
    if (type == kTypeLocalReplica) {
        LocalReplicaAffinityStrategy::Params params;
        params.sketch = sketch;
        params.suppressor = suppressor;
        if (!BuildLocalReplicaParams(*target, params, error_msg))
            return nullptr;
        return std::make_shared<LocalReplicaAffinityStrategy>(std::move(params));
    }

    SetErr(error_msg, std::string("unknown strategy type: ") + type);
    return nullptr;
}

} // namespace kv_cache_manager
