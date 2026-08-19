#include "tools/kvcm_swarm/clients/v6d/config.h"

#include <algorithm>
#include <set>

#include "tools/kvcm_swarm/scenario/config_reader.h"

namespace kvcm_swarm {

const char *CacheGroupKindName(CacheGroupKind kind) {
    return kind == CacheGroupKind::kFullAttention ? "full_attention" : "mamba";
}

const char *FullSelectorName(FullSelector selector) {
    return selector == FullSelector::kPrefix ? "prefix" : "coverage";
}

const char *ArrivalModeName(ArrivalMode mode) { return mode == ArrivalMode::kEven ? "even" : "poisson"; }

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
        // The first turn already exists at creation, so at most turns.max - 1
        // further appends happen; be conservative and count turns.max.
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

namespace {

bool ParseGroup(ConfigNode node, std::vector<std::string> *errors, CacheGroupSpec *group) {
    const size_t before = errors->size();
    ConfigReader reader(node, errors);
    if (!node.IsObject()) {
        errors->push_back(node.path() + ": group must be an object");
        return false;
    }
    group->group_id = reader.RequiredString("id");
    const std::string kind = reader.RequiredString("kind");
    if (kind == "full_attention") {
        group->kind = CacheGroupKind::kFullAttention;
    } else if (kind == "mamba") {
        group->kind = CacheGroupKind::kMamba;
    } else if (!kind.empty()) {
        reader.ErrorAt("kind", "must be 'full_attention' or 'mamba'");
    }
    group->block_size_tokens = static_cast<uint32_t>(reader.RequiredUint("block_size"));
    group->object_size_bytes = reader.RequiredUint("object_size");
    group->spec_name = "v6d_" + std::to_string(group->object_size_bytes);

    if (group->kind == CacheGroupKind::kFullAttention) {
        const std::string selector = reader.RequiredString("lookup_selector");
        if (selector == "prefix") {
            group->lookup_selector = FullSelector::kPrefix;
        } else if (selector == "coverage") {
            group->lookup_selector = FullSelector::kCoverage;
        } else {
            reader.ErrorAt("lookup_selector", "Full Attention groups must set 'prefix' or 'coverage' explicitly");
        }
        if (node.Has("key_presence_rate")) {
            reader.ErrorAt("key_presence_rate", "only Mamba groups accept key_presence_rate");
            node.Get("key_presence_rate");
        }
        group->key_presence_rate = 1.0;
    } else {
        if (node.Has("lookup_selector")) {
            reader.ErrorAt("lookup_selector", "Mamba groups always use COVERAGE and must not set this field");
            node.Get("lookup_selector");
        }
        group->key_presence_rate = reader.RequiredDouble("key_presence_rate");
        if (group->key_presence_rate < 0.0 || group->key_presence_rate > 1.0) {
            reader.ErrorAt("key_presence_rate", "must be within [0, 1]");
        }
    }

    if (group->group_id.empty()) {
        reader.ErrorAt("id", "must not be empty");
    }
    if (group->block_size_tokens == 0) {
        reader.ErrorAt("block_size", "must be positive");
    }
    if (group->object_size_bytes == 0) {
        reader.ErrorAt("object_size", "must be positive");
    }
    return errors->size() == before;
}

bool ParseSessionClass(ConfigNode node, std::vector<std::string> *errors, SessionClass *session_class) {
    const size_t before = errors->size();
    ConfigReader reader(node, errors);
    if (!node.IsObject()) {
        errors->push_back(node.path() + ": session class must be an object");
        return false;
    }
    session_class->name = reader.RequiredString("name");
    session_class->weight = reader.RequiredDouble("weight");
    session_class->turns = reader.RequiredIntSpec("turns");
    session_class->turn_interval = reader.RequiredDurationSpec("turn_interval");
    session_class->initial_tokens = reader.RequiredIntSpec("initial_tokens");
    session_class->new_tokens_per_turn = reader.RequiredIntSpec("new_tokens_per_turn");
    session_class->rewrite_tail_tokens = reader.RequiredIntSpec("rewrite_tail_tokens");
    session_class->shared_prefix_probability = reader.RequiredDouble("shared_prefix_probability");

    if (session_class->name.empty()) {
        reader.ErrorAt("name", "must not be empty");
    }
    if (!(session_class->weight > 0.0)) {
        reader.ErrorAt("weight", "must be positive");
    }
    if (session_class->turns.min == 0) {
        reader.ErrorAt("turns", "must be at least 1");
    }
    if (session_class->turn_interval.min <= Duration::zero()) {
        reader.ErrorAt("turn_interval", "must be positive");
    }
    if (session_class->initial_tokens.min == 0) {
        reader.ErrorAt("initial_tokens", "must be at least 1");
    }
    if (session_class->shared_prefix_probability < 0.0 || session_class->shared_prefix_probability > 1.0) {
        reader.ErrorAt("shared_prefix_probability", "must be within [0, 1]");
    }
    return errors->size() == before;
}

} // namespace

bool ParseV6dConfig(const BehaviorSpec &spec, V6dConfig *config, std::vector<std::string> *errors) {
    const size_t before = errors->size();
    if (!spec.config.IsObject()) {
        errors->push_back("behaviors[" + spec.id + "].config: must be an object");
        return false;
    }
    ConfigReader reader(spec.config, errors);

    config->process_count = static_cast<uint32_t>(reader.RequiredUint("process_count"));
    config->process_startup_interval =
        reader.OptionalDurationSpec("process_startup_interval", DurationSpec(Duration::zero()));
    config->instance_group = reader.RequiredString("instance_group");
    config->instance_id = reader.RequiredString("instance_id");

    ConfigReader cache = reader.Child("local_cache");
    config->local_cache_capacity_bytes = cache.RequiredUint("capacity_bytes");

    ConfigReader arrival = reader.Child("session_arrival");
    config->session_arrival_rate = arrival.RequiredDouble("rate");
    const std::string mode = arrival.RequiredString("mode");
    if (mode == "even") {
        config->arrival_mode = ArrivalMode::kEven;
    } else if (mode == "poisson") {
        config->arrival_mode = ArrivalMode::kPoisson;
    } else if (!mode.empty()) {
        arrival.ErrorAt("mode", "must be 'even' or 'poisson'");
    }

    config->session_affinity = reader.RequiredDouble("session_affinity");
    ConfigReader limits = reader.Child("limits");
    config->max_active_sessions = static_cast<uint32_t>(limits.RequiredUint("max_active_sessions"));
    config->heartbeat_interval = reader.RequiredDuration("heartbeat_interval");
    config->min_replica_count = static_cast<int32_t>(reader.OptionalUint("min_replica_count", 2));

    ConfigReader prefix = reader.Child("shared_prefix_pool");
    config->shared_prefix_pool.root_count = static_cast<uint32_t>(prefix.RequiredUint("root_count"));
    config->shared_prefix_pool.prefix_tokens = prefix.RequiredIntSpec("prefix_tokens");

    config->leader_poll_interval = reader.OptionalDuration("leader_poll_interval", std::chrono::seconds(15));
    config->write_timeout = reader.OptionalDuration("write_timeout", std::chrono::seconds(30));
    config->turn_deadline = reader.OptionalDuration("turn_deadline", std::chrono::seconds(10));
    config->rpc_timeout = reader.OptionalDuration("rpc_timeout", std::chrono::seconds(10));
    config->host_down_timeout = reader.OptionalDuration("host_down_timeout", std::chrono::seconds(3));
    config->eviction_batch_size = static_cast<uint32_t>(reader.OptionalUint("eviction_batch_size", 128));
    config->process_host_ip = reader.OptionalString("process_host_ip", "10.99.0.1");
    config->process_port_base = static_cast<uint32_t>(reader.OptionalUint("process_port_base", 40000));

    for (ConfigNode group_node : reader.RequiredArray("groups")) {
        CacheGroupSpec group;
        ParseGroup(group_node, errors, &group);
        config->groups.push_back(std::move(group));
    }
    for (ConfigNode class_node : reader.RequiredArray("session_classes")) {
        SessionClass session_class;
        ParseSessionClass(class_node, errors, &session_class);
        config->session_classes.push_back(std::move(session_class));
    }

    // ---- cross-field constraints ----
    if (config->process_count == 0) {
        reader.ErrorAt("process_count", "must be at least 1");
    }
    if (config->instance_group.empty()) {
        reader.ErrorAt("instance_group", "must not be empty");
    }
    if (config->instance_id.empty()) {
        reader.ErrorAt("instance_id", "must not be empty");
    }
    if (config->instance_id.find('#') != std::string::npos) {
        reader.ErrorAt("instance_id", "must not contain '#': it is a location id component");
    }
    if (config->process_host_ip.empty() || config->process_host_ip.find('#') != std::string::npos ||
        config->process_host_ip.find(':') != std::string::npos ||
        config->process_host_ip.find('/') != std::string::npos) {
        reader.ErrorAt("process_host_ip", "must be a bare host without ':', '/' or '#'");
    }
    if (config->process_port_base == 0 ||
        static_cast<uint64_t>(config->process_port_base) + config->process_count > 65536) {
        reader.ErrorAt("process_port_base", "process ports must stay within 1..65535");
    }
    if (!(config->session_arrival_rate > 0.0)) {
        errors->push_back(spec.config.path() + "session_arrival.rate: must be positive");
    }
    if (config->session_affinity < 0.0 || config->session_affinity > 1.0) {
        reader.ErrorAt("session_affinity", "must be within [0, 1]");
    }
    if (config->max_active_sessions == 0) {
        errors->push_back("limits.max_active_sessions: must be at least 1");
    }
    if (config->heartbeat_interval <= Duration::zero()) {
        reader.ErrorAt("heartbeat_interval", "must be positive");
    }
    if (config->min_replica_count < 1) {
        reader.ErrorAt("min_replica_count", "must be at least 1");
    }
    if (config->local_cache_capacity_bytes == 0) {
        errors->push_back("local_cache.capacity_bytes: must be positive");
    }
    if (config->eviction_batch_size == 0 || config->eviction_batch_size > 128) {
        reader.ErrorAt("eviction_batch_size", "must be within 1..128 (the V6D batch ceiling)");
    }
    if (config->turn_deadline <= Duration::zero()) {
        reader.ErrorAt("turn_deadline", "must be positive");
    }
    if (config->write_timeout <= Duration::zero()) {
        reader.ErrorAt("write_timeout", "must be positive");
    }
    if (config->leader_poll_interval <= Duration::zero()) {
        reader.ErrorAt("leader_poll_interval", "must be positive");
    }

    std::set<std::string> group_ids;
    bool has_full = false;
    for (const auto &group : config->groups) {
        if (!group.group_id.empty() && !group_ids.insert(group.group_id).second) {
            errors->push_back("groups: duplicate group id '" + group.group_id + "'");
        }
        if (group.kind == CacheGroupKind::kFullAttention) {
            has_full = true;
        }
    }
    if (config->groups.empty()) {
        errors->push_back("groups: at least one Full Attention group is required");
    } else if (!has_full) {
        errors->push_back("groups: at least one Full Attention group is required");
    }
    // A single object must always fit into a per-process cache, otherwise
    // materialisation could never make progress.
    if (config->local_cache_capacity_bytes > 0 && config->MaxObjectSize() > config->local_cache_capacity_bytes) {
        errors->push_back("local_cache.capacity_bytes: must be >= the largest groups[].object_size");
    }
    // A turn holds a short lease on every object it uses, so a cache smaller
    // than the worst-case single-turn working set would make turns thrash and
    // the resulting traffic would not represent the modelled workload.
    if (config->local_cache_capacity_bytes > 0 && !config->groups.empty() && !config->session_classes.empty()) {
        const uint64_t working_set = config->WorstCaseTurnWorkingSetBytes();
        if (working_set > config->local_cache_capacity_bytes) {
            errors->push_back("local_cache.capacity_bytes: must be >= the worst-case single-turn working set (" +
                              std::to_string(working_set) +
                              " bytes, derived from session_classes and groups); otherwise a single turn cannot "
                              "keep its objects resident");
        }
    }

    std::set<std::string> class_names;
    for (const auto &session_class : config->session_classes) {
        if (!session_class.name.empty() && !class_names.insert(session_class.name).second) {
            errors->push_back("session_classes: duplicate name '" + session_class.name + "'");
        }
        // A session that may attach to a shared prefix must be able to hold
        // the longest possible root.
        if (session_class.shared_prefix_probability > 0.0 && config->shared_prefix_pool.root_count > 0 &&
            session_class.initial_tokens.min < config->shared_prefix_pool.prefix_tokens.max) {
            errors->push_back("session_classes[" + session_class.name +
                              "].initial_tokens.min must be >= shared_prefix_pool.prefix_tokens.max");
        }
        if (session_class.shared_prefix_probability > 0.0 && config->shared_prefix_pool.root_count == 0) {
            errors->push_back("session_classes[" + session_class.name +
                              "]: shared_prefix_probability > 0 requires shared_prefix_pool.root_count > 0");
        }
    }
    if (config->session_classes.empty()) {
        errors->push_back("session_classes: at least one session class is required");
    }

    std::vector<std::string> unknown;
    spec.config.CollectUnknown(&unknown);
    for (const auto &key : unknown) {
        errors->push_back("unknown configuration field: " + key);
    }
    return errors->size() == before;
}

} // namespace kvcm_swarm
