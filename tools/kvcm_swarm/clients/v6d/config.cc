#include "tools/kvcm_swarm/clients/v6d/config.h"

#include <algorithm>
#include <limits>
#include <set>

#include "tools/kvcm_swarm/scenario/duration.h"

namespace kvcm_swarm {

const char *CacheGroupKindName(CacheGroupKind kind) {
    return kind == CacheGroupKind::kFullAttention ? "full_attention" : "mamba";
}

const char *FullSelectorName(FullSelector selector) {
    return selector == FullSelector::kPrefix ? "prefix" : "coverage";
}

const char *ArrivalModeName(ArrivalMode mode) { return mode == ArrivalMode::kEven ? "even" : "poisson"; }

namespace {

void WriteKey(rapidjson::Writer<rapidjson::StringBuffer> &writer, std::string_view key) {
    writer.Key(key.data(), static_cast<rapidjson::SizeType>(key.size()), false);
}

void WriteDuration(rapidjson::Writer<rapidjson::StringBuffer> &writer, Duration value) {
    const std::string text = FormatDuration(value);
    writer.String(text.data(), static_cast<rapidjson::SizeType>(text.size()), false);
}

void WriteIntSpec(rapidjson::Writer<rapidjson::StringBuffer> &writer, const IntSpec &spec) {
    writer.StartObject();
    writer.Key("min");
    writer.Uint64(spec.min);
    writer.Key("max");
    writer.Uint64(spec.max);
    writer.EndObject();
}

void WriteDurationSpec(rapidjson::Writer<rapidjson::StringBuffer> &writer, const DurationSpec &spec) {
    writer.StartObject();
    writer.Key("min");
    WriteDuration(writer, spec.min);
    writer.Key("max");
    WriteDuration(writer, spec.max);
    writer.EndObject();
}

bool DecodeUint(const rapidjson::Value &value, uint64_t *out, std::string *error) {
    if (!value.IsUint64()) {
        *error = value.IsInt64() && value.GetInt64() < 0 ? "must not be negative" : "must be an integer";
        return false;
    }
    *out = value.GetUint64();
    return true;
}

bool DecodeDuration(const rapidjson::Value &value, Duration *out, std::string *error) {
    if (!value.IsString()) {
        *error = "must be a duration string such as \"10ms\"";
        return false;
    }
    return ParseDuration(std::string(value.GetString(), value.GetStringLength()), out, error);
}

bool DecodeIntSpec(const rapidjson::Value &value, IntSpec *out, std::string *error) {
    uint64_t scalar = 0;
    if (DecodeUint(value, &scalar, error)) {
        *out = IntSpec(scalar);
        return true;
    }
    if (!value.IsObject()) {
        *error = "must be an integer or {\"min\":..,\"max\":..}";
        return false;
    }
    for (const auto &member : value.GetObject()) {
        const std::string_view key(member.name.GetString(), member.name.GetStringLength());
        if (key != "min" && key != "max") {
            *error = "unknown field '" + std::string(key) + "' in range object";
            return false;
        }
    }
    const auto min_member = value.FindMember("min");
    const auto max_member = value.FindMember("max");
    if (min_member == value.MemberEnd() || max_member == value.MemberEnd()) {
        *error = "range object requires both min and max";
        return false;
    }
    uint64_t min_value = 0;
    uint64_t max_value = 0;
    std::string nested_error;
    if (!DecodeUint(min_member->value, &min_value, &nested_error)) {
        *error = "min " + nested_error;
        return false;
    }
    if (!DecodeUint(max_member->value, &max_value, &nested_error)) {
        *error = "max " + nested_error;
        return false;
    }
    if (max_value < min_value) {
        *error = "max must be >= min";
        return false;
    }
    *out = IntSpec(min_value, max_value);
    return true;
}

bool DecodeDurationSpec(const rapidjson::Value &value, DurationSpec *out, std::string *error) {
    Duration scalar{};
    if (value.IsString()) {
        if (!DecodeDuration(value, &scalar, error)) {
            return false;
        }
        *out = DurationSpec(scalar);
        return true;
    }
    if (!value.IsObject()) {
        *error = "must be a duration string or {\"min\":\"..\",\"max\":\"..\"}";
        return false;
    }
    for (const auto &member : value.GetObject()) {
        const std::string_view key(member.name.GetString(), member.name.GetStringLength());
        if (key != "min" && key != "max") {
            *error = "unknown field '" + std::string(key) + "' in range object";
            return false;
        }
    }
    const auto min_member = value.FindMember("min");
    const auto max_member = value.FindMember("max");
    if (min_member == value.MemberEnd() || max_member == value.MemberEnd()) {
        *error = "range object requires both min and max";
        return false;
    }
    Duration min_value{};
    Duration max_value{};
    std::string nested_error;
    if (!DecodeDuration(min_member->value, &min_value, &nested_error)) {
        *error = "min " + nested_error;
        return false;
    }
    if (!DecodeDuration(max_member->value, &max_value, &nested_error)) {
        *error = "max " + nested_error;
        return false;
    }
    if (max_value < min_value) {
        *error = "max must be >= min";
        return false;
    }
    *out = DurationSpec(min_value, max_value);
    return true;
}

class LocalCacheJson final : public JsonConfig {
public:
    bool FromRapidValue(const rapidjson::Value &value) override {
        if (!BeginObject(value, {"capacity_bytes"})) {
            return false;
        }
        Required(value, "capacity_bytes", capacity_bytes);
        return true;
    }
    uint64_t capacity_bytes = 0;
};

class SessionArrivalJson final : public JsonConfig {
public:
    bool FromRapidValue(const rapidjson::Value &value) override {
        if (!BeginObject(value, {"rate", "mode"})) {
            return false;
        }
        Required(value, "rate", rate);
        Required(value, "mode", mode);
        return true;
    }
    double rate = 0.0;
    std::string mode;
};

class LimitsJson final : public JsonConfig {
public:
    bool FromRapidValue(const rapidjson::Value &value) override {
        if (!BeginObject(value, {"max_active_sessions"})) {
            return false;
        }
        Required(value, "max_active_sessions", max_active_sessions);
        return true;
    }
    uint64_t max_active_sessions = 0;
};

bool ToUint32(uint64_t value, std::string_view path, uint32_t *out, std::vector<std::string> *errors) {
    if (value > std::numeric_limits<uint32_t>::max()) {
        errors->push_back(std::string(path) + ": must fit in uint32");
        return false;
    }
    *out = static_cast<uint32_t>(value);
    return true;
}

} // namespace

bool CacheGroupSpec::FromRapidValue(const rapidjson::Value &value) {
    if (!BeginObject(value, {"id", "kind", "block_size", "object_size", "lookup_selector", "key_presence_rate"})) {
        return false;
    }
    Required(value, "id", group_id);
    Required(value, "kind", kind_json_);
    Required(value, "block_size", block_size_json_);
    Required(value, "object_size", object_size_bytes);
    Optional(value, "lookup_selector", lookup_selector_json_);
    Optional(value, "key_presence_rate", key_presence_rate_json_);
    return true;
}

bool CacheGroupSpec::Validate(std::string_view path, std::vector<std::string> *errors) {
    const size_t before = errors->size();
    ToUint32(block_size_json_, std::string(path) + ".block_size", &block_size_tokens, errors);
    kind_valid_ = false;
    if (kind_json_ == "full_attention") {
        kind = CacheGroupKind::kFullAttention;
        kind_valid_ = true;
        if (!lookup_selector_json_.has_value()) {
            errors->push_back(std::string(path) +
                              ".lookup_selector: Full Attention groups must set 'prefix' or 'coverage' explicitly");
        } else if (*lookup_selector_json_ == "prefix") {
            lookup_selector = FullSelector::kPrefix;
        } else if (*lookup_selector_json_ == "coverage") {
            lookup_selector = FullSelector::kCoverage;
        } else {
            errors->push_back(std::string(path) +
                              ".lookup_selector: Full Attention groups must set 'prefix' or 'coverage' explicitly");
        }
        if (key_presence_rate_json_.has_value()) {
            errors->push_back(std::string(path) + ".key_presence_rate: only Mamba groups accept key_presence_rate");
        }
        key_presence_rate = 1.0;
    } else if (kind_json_ == "mamba") {
        kind = CacheGroupKind::kMamba;
        kind_valid_ = true;
        lookup_selector.reset();
        if (lookup_selector_json_.has_value()) {
            errors->push_back(std::string(path) +
                              ".lookup_selector: Mamba groups always use COVERAGE and must not set this field");
        }
        if (!key_presence_rate_json_.has_value()) {
            errors->push_back(std::string(path) + ".key_presence_rate: required for Mamba groups");
        } else {
            key_presence_rate = *key_presence_rate_json_;
            if (key_presence_rate < 0.0 || key_presence_rate > 1.0) {
                errors->push_back(std::string(path) + ".key_presence_rate: must be within [0, 1]");
            }
        }
    } else {
        errors->push_back(std::string(path) + ".kind: must be 'full_attention' or 'mamba'");
    }
    if (group_id.empty()) {
        errors->push_back(std::string(path) + ".id: must not be empty");
    }
    if (block_size_tokens == 0) {
        errors->push_back(std::string(path) + ".block_size: must be positive");
    }
    if (object_size_bytes == 0) {
        errors->push_back(std::string(path) + ".object_size: must be positive");
    }
    spec_name = "v6d_" + std::to_string(object_size_bytes);
    return errors->size() == before;
}

void CacheGroupSpec::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept {
    Put(writer, "id", group_id);
    Put(writer, "kind", std::string(CacheGroupKindName(kind)));
    Put(writer, "block_size", block_size_tokens);
    Put(writer, "object_size", object_size_bytes);
    Put(writer, "spec_name", spec_name);
    if (kind == CacheGroupKind::kFullAttention && lookup_selector.has_value()) {
        Put(writer, "lookup_selector", std::string(FullSelectorName(*lookup_selector)));
    } else if (kind == CacheGroupKind::kMamba) {
        Put(writer, "key_presence_rate", key_presence_rate);
    }
}

bool SessionClass::FromRapidValue(const rapidjson::Value &value) {
    if (!BeginObject(value,
                     {"name",
                      "weight",
                      "turns",
                      "turn_interval",
                      "initial_tokens",
                      "new_tokens_per_turn",
                      "rewrite_tail_tokens",
                      "shared_prefix_probability"})) {
        return false;
    }
    Required(value, "name", name);
    Required(value, "weight", weight);
    RequiredCustom(value, "turns", turns, DecodeIntSpec);
    RequiredCustom(value, "turn_interval", turn_interval, DecodeDurationSpec);
    RequiredCustom(value, "initial_tokens", initial_tokens, DecodeIntSpec);
    RequiredCustom(value, "new_tokens_per_turn", new_tokens_per_turn, DecodeIntSpec);
    RequiredCustom(value, "rewrite_tail_tokens", rewrite_tail_tokens, DecodeIntSpec);
    Required(value, "shared_prefix_probability", shared_prefix_probability);
    return true;
}

bool SessionClass::Validate(std::string_view path, std::vector<std::string> *errors) const {
    const size_t before = errors->size();
    if (name.empty()) {
        errors->push_back(std::string(path) + ".name: must not be empty");
    }
    if (!(weight > 0.0)) {
        errors->push_back(std::string(path) + ".weight: must be positive");
    }
    if (turns.min == 0) {
        errors->push_back(std::string(path) + ".turns: must be at least 1");
    }
    if (turn_interval.min <= Duration::zero()) {
        errors->push_back(std::string(path) + ".turn_interval: must be positive");
    }
    if (initial_tokens.min == 0) {
        errors->push_back(std::string(path) + ".initial_tokens: must be at least 1");
    }
    if (shared_prefix_probability < 0.0 || shared_prefix_probability > 1.0) {
        errors->push_back(std::string(path) + ".shared_prefix_probability: must be within [0, 1]");
    }
    return errors->size() == before;
}

void SessionClass::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept {
    Put(writer, "name", name);
    Put(writer, "weight", weight);
    WriteKey(writer, "turns");
    WriteIntSpec(writer, turns);
    WriteKey(writer, "turn_interval");
    WriteDurationSpec(writer, turn_interval);
    WriteKey(writer, "initial_tokens");
    WriteIntSpec(writer, initial_tokens);
    WriteKey(writer, "new_tokens_per_turn");
    WriteIntSpec(writer, new_tokens_per_turn);
    WriteKey(writer, "rewrite_tail_tokens");
    WriteIntSpec(writer, rewrite_tail_tokens);
    Put(writer, "shared_prefix_probability", shared_prefix_probability);
}

bool SharedPrefixPool::FromRapidValue(const rapidjson::Value &value) {
    if (!BeginObject(value, {"root_count", "prefix_tokens"})) {
        return false;
    }
    Required(value, "root_count", root_count_json_);
    RequiredCustom(value, "prefix_tokens", prefix_tokens, DecodeIntSpec);
    return true;
}

bool SharedPrefixPool::Validate(std::string_view path, std::vector<std::string> *errors) {
    return ToUint32(root_count_json_, std::string(path) + ".root_count", &root_count, errors);
}

void SharedPrefixPool::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept {
    Put(writer, "root_count", root_count);
    WriteKey(writer, "prefix_tokens");
    WriteIntSpec(writer, prefix_tokens);
}

bool V6dConfig::FromRapidValue(const rapidjson::Value &value) {
    if (!BeginObject(value, {"process_count",      "process_startup_interval", "instance_group",      "instance_id",
                             "local_cache",        "session_arrival",          "session_affinity",    "limits",
                             "heartbeat_interval", "min_replica_count",        "shared_prefix_pool",  "groups",
                             "session_classes",    "leader_poll_interval",     "write_timeout",       "turn_deadline",
                             "rpc_timeout",        "host_down_timeout",        "eviction_batch_size", "process_host_ip",
                             "process_port_base"})) {
        return false;
    }
    LocalCacheJson local_cache;
    SessionArrivalJson session_arrival;
    LimitsJson limits;
    Required(value, "process_count", process_count_json_);
    OptionalCustom(value,
                   "process_startup_interval",
                   process_startup_interval,
                   DurationSpec(Duration::zero()),
                   DecodeDurationSpec);
    Required(value, "instance_group", instance_group);
    Required(value, "instance_id", instance_id);
    Required(value, "local_cache", local_cache);
    Required(value, "session_arrival", session_arrival);
    Required(value, "session_affinity", session_affinity);
    Required(value, "limits", limits);
    RequiredCustom(value, "heartbeat_interval", heartbeat_interval, DecodeDuration);
    Optional(value, "min_replica_count", min_replica_count_json_, uint64_t{2});
    Required(value, "shared_prefix_pool", shared_prefix_pool);
    Required(value, "groups", groups);
    Required(value, "session_classes", session_classes);
    OptionalCustom(
        value, "leader_poll_interval", leader_poll_interval, Duration(std::chrono::seconds(15)), DecodeDuration);
    OptionalCustom(value, "write_timeout", write_timeout, Duration(std::chrono::seconds(30)), DecodeDuration);
    OptionalCustom(value, "turn_deadline", turn_deadline, Duration(std::chrono::seconds(10)), DecodeDuration);
    OptionalCustom(value, "rpc_timeout", rpc_timeout, Duration(std::chrono::seconds(10)), DecodeDuration);
    OptionalCustom(value, "host_down_timeout", host_down_timeout, Duration(std::chrono::seconds(3)), DecodeDuration);
    Optional(value, "eviction_batch_size", eviction_batch_size_json_, uint64_t{128});
    Optional(value, "process_host_ip", process_host_ip, std::string("10.99.0.1"));
    Optional(value, "process_port_base", process_port_base_json_, uint64_t{40000});

    local_cache_capacity_bytes = local_cache.capacity_bytes;
    session_arrival_rate = session_arrival.rate;
    arrival_mode_json_ = session_arrival.mode;
    max_active_sessions_json_ = limits.max_active_sessions;
    MergeJsonErrors("local_cache", local_cache);
    MergeJsonErrors("session_arrival", session_arrival);
    MergeJsonErrors("limits", limits);
    MergeJsonErrors("shared_prefix_pool", shared_prefix_pool);
    for (size_t i = 0; i < groups.size(); ++i) {
        MergeJsonErrors("groups[" + std::to_string(i) + "]", groups[i]);
    }
    for (size_t i = 0; i < session_classes.size(); ++i) {
        MergeJsonErrors("session_classes[" + std::to_string(i) + "]", session_classes[i]);
    }
    return true;
}

bool V6dConfig::Validate(std::vector<std::string> *errors) {
    const size_t before = errors->size();
    ToUint32(process_count_json_, "process_count", &process_count, errors);
    ToUint32(max_active_sessions_json_, "limits.max_active_sessions", &max_active_sessions, errors);
    ToUint32(eviction_batch_size_json_, "eviction_batch_size", &eviction_batch_size, errors);
    ToUint32(process_port_base_json_, "process_port_base", &process_port_base, errors);
    shared_prefix_pool.Validate("shared_prefix_pool", errors);
    if (min_replica_count_json_ > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
        errors->push_back("min_replica_count: must fit in int32");
    } else {
        min_replica_count = static_cast<int32_t>(min_replica_count_json_);
    }
    if (arrival_mode_json_ == "even") {
        arrival_mode = ArrivalMode::kEven;
    } else if (arrival_mode_json_ == "poisson") {
        arrival_mode = ArrivalMode::kPoisson;
    } else {
        errors->push_back("session_arrival.mode: must be 'even' or 'poisson'");
    }
    for (size_t i = 0; i < groups.size(); ++i) {
        groups[i].Validate("groups[" + std::to_string(i) + "]", errors);
    }
    for (size_t i = 0; i < session_classes.size(); ++i) {
        session_classes[i].Validate("session_classes[" + std::to_string(i) + "]", errors);
    }

    if (process_count == 0) {
        errors->push_back("process_count: must be at least 1");
    }
    if (instance_group.empty()) {
        errors->push_back("instance_group: must not be empty");
    }
    if (instance_id.empty()) {
        errors->push_back("instance_id: must not be empty");
    }
    if (instance_id.find('#') != std::string::npos) {
        errors->push_back("instance_id: must not contain '#': it is a location id component");
    }
    if (process_host_ip.empty() || process_host_ip.find('#') != std::string::npos ||
        process_host_ip.find(':') != std::string::npos || process_host_ip.find('/') != std::string::npos) {
        errors->push_back("process_host_ip: must be a bare host without ':', '/' or '#'");
    }
    if (process_port_base == 0 || static_cast<uint64_t>(process_port_base) + process_count > 65536) {
        errors->push_back("process_port_base: process ports must stay within 1..65535");
    }
    if (!(session_arrival_rate > 0.0)) {
        errors->push_back("session_arrival.rate: must be positive");
    }
    if (session_affinity < 0.0 || session_affinity > 1.0) {
        errors->push_back("session_affinity: must be within [0, 1]");
    }
    if (max_active_sessions == 0) {
        errors->push_back("limits.max_active_sessions: must be at least 1");
    }
    if (heartbeat_interval <= Duration::zero()) {
        errors->push_back("heartbeat_interval: must be positive");
    }
    if (min_replica_count < 1) {
        errors->push_back("min_replica_count: must be at least 1");
    }
    if (local_cache_capacity_bytes == 0) {
        errors->push_back("local_cache.capacity_bytes: must be positive");
    }
    if (eviction_batch_size == 0 || eviction_batch_size > 128) {
        errors->push_back("eviction_batch_size: must be within 1..128 (the V6D batch ceiling)");
    }
    if (turn_deadline <= Duration::zero()) {
        errors->push_back("turn_deadline: must be positive");
    }
    if (write_timeout <= Duration::zero()) {
        errors->push_back("write_timeout: must be positive");
    }
    if (leader_poll_interval <= Duration::zero()) {
        errors->push_back("leader_poll_interval: must be positive");
    }

    std::set<std::string> group_ids;
    bool has_full = false;
    for (const auto &group : groups) {
        if (!group.group_id.empty() && !group_ids.insert(group.group_id).second) {
            errors->push_back("groups: duplicate group id '" + group.group_id + "'");
        }
        has_full = has_full || (group.has_valid_kind() && group.kind == CacheGroupKind::kFullAttention);
    }
    if (!has_full) {
        errors->push_back("groups: at least one Full Attention group is required");
    }
    if (local_cache_capacity_bytes > 0 && MaxObjectSize() > local_cache_capacity_bytes) {
        errors->push_back("local_cache.capacity_bytes: must be >= the largest groups[].object_size");
    }
    if (local_cache_capacity_bytes > 0 && !groups.empty() && !session_classes.empty()) {
        const uint64_t working_set = WorstCaseTurnWorkingSetBytes();
        if (working_set > local_cache_capacity_bytes) {
            errors->push_back("local_cache.capacity_bytes: must be >= the worst-case single-turn working set (" +
                              std::to_string(working_set) +
                              " bytes, derived from session_classes and groups); otherwise a single turn cannot "
                              "keep its objects resident");
        }
    }

    std::set<std::string> class_names;
    for (const auto &session_class : session_classes) {
        if (!session_class.name.empty() && !class_names.insert(session_class.name).second) {
            errors->push_back("session_classes: duplicate name '" + session_class.name + "'");
        }
        if (session_class.shared_prefix_probability > 0.0 && shared_prefix_pool.root_count > 0 &&
            session_class.initial_tokens.min < shared_prefix_pool.prefix_tokens.max) {
            errors->push_back("session_classes[" + session_class.name +
                              "].initial_tokens.min must be >= shared_prefix_pool.prefix_tokens.max");
        }
        if (session_class.shared_prefix_probability > 0.0 && shared_prefix_pool.root_count == 0) {
            errors->push_back("session_classes[" + session_class.name +
                              "]: shared_prefix_probability > 0 requires shared_prefix_pool.root_count > 0");
        }
    }
    if (session_classes.empty()) {
        errors->push_back("session_classes: at least one session class is required");
    }
    return errors->size() == before;
}

void V6dConfig::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept {
    Put(writer, "process_count", process_count);
    WriteKey(writer, "process_startup_interval");
    WriteDurationSpec(writer, process_startup_interval);
    Put(writer, "instance_group", instance_group);
    Put(writer, "instance_id", instance_id);
    Put(writer, "process_host_ip", process_host_ip);
    Put(writer, "process_port_base", process_port_base);
    writer.Key("local_cache");
    writer.StartObject();
    Put(writer, "capacity_bytes", local_cache_capacity_bytes);
    Put(writer, "worst_case_turn_working_set_bytes", WorstCaseTurnWorkingSetBytes());
    writer.EndObject();
    writer.Key("session_arrival");
    writer.StartObject();
    Put(writer, "rate", session_arrival_rate);
    Put(writer, "mode", std::string(ArrivalModeName(arrival_mode)));
    writer.EndObject();
    Put(writer, "session_affinity", session_affinity);
    writer.Key("limits");
    writer.StartObject();
    Put(writer, "max_active_sessions", max_active_sessions);
    writer.EndObject();
    WriteKey(writer, "heartbeat_interval");
    WriteDuration(writer, heartbeat_interval);
    Put(writer, "min_replica_count", min_replica_count);
    WriteKey(writer, "leader_poll_interval");
    WriteDuration(writer, leader_poll_interval);
    WriteKey(writer, "write_timeout");
    WriteDuration(writer, write_timeout);
    WriteKey(writer, "turn_deadline");
    WriteDuration(writer, turn_deadline);
    WriteKey(writer, "rpc_timeout");
    WriteDuration(writer, rpc_timeout);
    WriteKey(writer, "host_down_timeout");
    WriteDuration(writer, host_down_timeout);
    Put(writer, "eviction_batch_size", eviction_batch_size);
    Put(writer, "shared_prefix_pool", shared_prefix_pool);
    Put(writer, "groups", groups);
    Put(writer, "session_classes", session_classes);
}

uint64_t V6dConfig::MaxObjectSize() const {
    uint64_t max_size = 0;
    for (const auto &group : groups) {
        max_size = std::max(max_size, group.object_size_bytes);
    }
    return max_size;
}

uint64_t V6dConfig::WorstCaseContextTokens() const {
    uint64_t worst = 0;
    for (const auto &session_class : session_classes) {
        uint64_t tokens = session_class.initial_tokens.max;
        tokens += session_class.turns.max * session_class.new_tokens_per_turn.max;
        worst = std::max(worst, tokens);
    }
    return worst;
}

uint64_t V6dConfig::WorstCaseTurnWorkingSetBytes() const {
    const uint64_t tokens = WorstCaseContextTokens();
    uint64_t bytes = 0;
    for (const auto &group : groups) {
        if (group.block_size_tokens == 0) {
            continue;
        }
        const uint64_t blocks = tokens / group.block_size_tokens;
        bytes += blocks * group.object_size_bytes;
    }
    return bytes;
}

V6dProcessIdentity MakeProcessIdentity(const std::string &behavior_id, const V6dConfig &config, uint32_t index) {
    V6dProcessIdentity identity;
    identity.index = index;
    identity.process_id = behavior_id + "-p" + std::to_string(index);
    identity.host_ip_port = config.process_host_ip + ":" + std::to_string(config.process_port_base + index);
    return identity;
}

bool ParseV6dConfig(const BehaviorSpec &spec, V6dConfig *config, std::vector<std::string> *errors) {
    const size_t before = errors->size();
    *config = V6dConfig();
    std::string parse_error;
    if (!config->FromJsonString(spec.config_json, &parse_error)) {
        if (!parse_error.empty()) {
            errors->push_back(parse_error);
        }
        config->AppendJsonErrors("", errors);
        return false;
    }
    config->AppendJsonErrors("", errors);
    if (errors->size() != before) {
        return false;
    }
    config->Validate(errors);
    return errors->size() == before;
}

} // namespace kvcm_swarm
