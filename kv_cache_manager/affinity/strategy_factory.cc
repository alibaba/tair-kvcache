#include "kv_cache_manager/affinity/strategy_factory.h"

#include <string>

#include "kv_cache_manager/affinity/noop_strategy.h"
#include "kv_cache_manager/affinity/local_replica_strategy.h"
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
constexpr const char *kFieldWrite = "write";        // 写一级配置块
constexpr const char *kFieldRead = "read";          // 读一级配置块
constexpr const char *kFieldEviction = "eviction";  // 淘汰一级配置块
constexpr const char *kFieldOps = "ops";            // op 链
constexpr const char *kFieldOnMiss = "on_miss";     // 读子项

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

void ReadBoolField(const rapidjson::Value &obj, const char *key, bool &out) {
    if (obj.HasMember(key) && obj[key].IsBool()) {
        out = obj[key].GetBool();
    }
}
void ReadUIntField(const rapidjson::Value &obj, const char *key, uint32_t &out) {
    if (obj.HasMember(key) && obj[key].IsUint()) {
        out = obj[key].GetUint();
    }
}
void ReadDoubleField(const rapidjson::Value &obj, const char *key, double &out) {
    if (obj.HasMember(key) && obj[key].IsNumber()) {
        out = obj[key].GetDouble();
    }
}

// 解析 write 一级配置块：{ "ops": <5 段流水线对象> }
std::shared_ptr<CandidatePipeline> ParseWriteOps(const rapidjson::Value &write_block, std::string *err) {
    if (write_block.HasMember(kFieldOps) && write_block[kFieldOps].IsObject()) {
        return CandidatePipeline::ParseJsonString(ValueToJson(write_block[kFieldOps]), err);
    }
    return nullptr;
}

// 解析 read 一级的 on_miss 子项参数（复制触发 4 gate 阈值等）。
void ApplyOnMiss(const rapidjson::Value &on_miss, LocalReplicaAffinityStrategy::Params &p) {
    // 子开关：on_miss 路径可单独关停（不影响 read.pick 本地优先）
    ReadBoolField(on_miss, "enabled", p.enable_on_miss);
    ReadUIntField(on_miss, "replication_hot_threshold", p.replication_hot_threshold);
    ReadDoubleField(on_miss, "caller_capacity_threshold", p.caller_capacity_threshold);
    ReadDoubleField(on_miss, "caller_capacity_buffer", p.caller_capacity_buffer);
}

// 解析 eviction 一级 ops 参数（v1 只有 node_water）。
void ApplyEvictionOps(const rapidjson::Value &eviction_block, LocalReplicaAffinityStrategy::Params &p) {
    if (eviction_block.HasMember(kFieldOps) && eviction_block[kFieldOps].IsArray()) {
        for (const auto &op : eviction_block[kFieldOps].GetArray()) {
            if (!op.IsObject() || !op.HasMember("op") || !op["op"].IsString()) {
                continue;
            }
            std::string name = op["op"].GetString();
            if (name == "node_water_level") {
                ReadDoubleField(op, "threshold", p.node_water_threshold);
                ReadDoubleField(op, "critical", p.node_water_critical);
                ReadDoubleField(op, "low", p.node_water_low);
            }
        }
    }
}

// 构造 LocalReplicaAffinityStrategy::Params；从 target 对象按一级行为块解析。
//   target 期望形如：
//     { "type":"local_replica",
//       "enabled_aspects": { "write":true, "read":true, "eviction":true },
//       "write":     { "ops": <5 段流水线对象> },
//       "read":      { "on_miss": { "replication_hot_threshold": 3, ... } },
//       "eviction":  { "ops": [{"op":"node_water_level","threshold":0.85}, ...] } }
LocalReplicaAffinityStrategy::Params BuildLocalReplicaParams(const rapidjson::Value &target, FrequencySketch *sketch) {
    LocalReplicaAffinityStrategy::Params p;
    p.sketch = sketch;

    // enabled_aspects
    if (target.HasMember(kFieldEnabledAspects) && target[kFieldEnabledAspects].IsObject()) {
        const auto &ea = target[kFieldEnabledAspects];
        ReadBoolField(ea, "write", p.enable_write);
        ReadBoolField(ea, "read", p.enable_read);
        ReadBoolField(ea, "eviction", p.enable_eviction);
    }
    // write 一级
    if (target.HasMember(kFieldWrite) && target[kFieldWrite].IsObject()) {
        auto pipe = ParseWriteOps(target[kFieldWrite], nullptr);
        if (pipe) {
            p.write_pipeline = std::move(pipe);
        }
    }
    // read 一级 - on_miss 子项
    if (target.HasMember(kFieldRead) && target[kFieldRead].IsObject()) {
        const auto &read_block = target[kFieldRead];
        if (read_block.HasMember(kFieldOnMiss) && read_block[kFieldOnMiss].IsObject()) {
            ApplyOnMiss(read_block[kFieldOnMiss], p);
        }
    }
    // eviction 一级
    if (target.HasMember(kFieldEviction) && target[kFieldEviction].IsObject()) {
        ApplyEvictionOps(target[kFieldEviction], p);
    }
    return p;
}

} // namespace

std::shared_ptr<AffinityStrategy>
StrategyFactory::ParseJsonString(const std::string &json, FrequencySketch *sketch, std::string *error_msg) {
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
    if (doc.HasMember(kFieldStrategy) && doc[kFieldStrategy].IsObject()) {
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
        return std::make_shared<LocalReplicaAffinityStrategy>(BuildLocalReplicaParams(*target, sketch));
    }

    SetErr(error_msg, std::string("unknown strategy type: ") + type);
    return nullptr;
}

} // namespace kv_cache_manager
