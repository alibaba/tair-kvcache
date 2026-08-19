// v6d_deployment configuration and its cross-field validation.
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "tools/kvcm_swarm/clients/client_behavior.h"
#include "tools/kvcm_swarm/runtime/clock.h"
#include "tools/kvcm_swarm/runtime/sample_spec.h"

namespace kvcm_swarm {

enum class CacheGroupKind {
    kFullAttention,
    kMamba
};
enum class FullSelector {
    kPrefix,
    kCoverage
};
enum class ArrivalMode {
    kEven,
    kPoisson
};

const char *CacheGroupKindName(CacheGroupKind kind);
const char *FullSelectorName(FullSelector selector);
const char *ArrivalModeName(ArrivalMode mode);

struct CacheGroupSpec {
    std::string group_id;
    CacheGroupKind kind = CacheGroupKind::kFullAttention;
    uint32_t block_size_tokens = 16;
    uint64_t object_size_bytes = 4096;
    std::string spec_name; // v6d_<object_size_bytes>
    // Full Attention groups pick one explicit selector; Mamba is always
    // COVERAGE and must not carry the field.
    std::optional<FullSelector> lookup_selector;
    double key_presence_rate = 1.0; // Mamba only
};

struct SessionClass {
    std::string name;
    double weight = 1.0;
    IntSpec turns{1};
    DurationSpec turn_interval{Duration(std::chrono::milliseconds(50))};
    IntSpec initial_tokens{256};
    IntSpec new_tokens_per_turn{32};
    IntSpec rewrite_tail_tokens{0};
    double shared_prefix_probability = 0.0;
};

struct SharedPrefixPool {
    uint32_t root_count = 0;
    IntSpec prefix_tokens{0};
};

struct V6dConfig {
    uint32_t process_count = 1;
    DurationSpec process_startup_interval{Duration::zero()};
    std::string instance_group;
    std::string instance_id;
    uint64_t local_cache_capacity_bytes = 0;

    double session_arrival_rate = 1.0;
    ArrivalMode arrival_mode = ArrivalMode::kEven;
    double session_affinity = 0.5;
    uint32_t max_active_sessions = 1024;

    Duration heartbeat_interval = std::chrono::seconds(10);
    int32_t min_replica_count = 2;

    SharedPrefixPool shared_prefix_pool;
    std::vector<CacheGroupSpec> groups;
    std::vector<SessionClass> session_classes;

    // ---- advanced knobs: defaulted here and always reported ----
    Duration leader_poll_interval = std::chrono::seconds(15);
    Duration write_timeout = std::chrono::seconds(30);
    Duration turn_deadline = std::chrono::seconds(10);
    Duration rpc_timeout = std::chrono::seconds(10);
    Duration host_down_timeout = std::chrono::seconds(3);
    uint32_t eviction_batch_size = 128;
    // Simulated V6D daemon addresses: process i is
    // <process_host_ip>:<process_port_base + i>.
    std::string process_host_ip = "10.99.0.1";
    uint32_t process_port_base = 40000;

    // Derived helpers.
    uint64_t MaxObjectSize() const;
    // Largest logical context a single session can reach.
    uint64_t WorstCaseContextTokens() const;
    // Bytes a single turn may need resident at once. A turn holds a short lease
    // on every object it uses, so the per-process cache must be able to hold
    // this much or the turn could never complete.
    uint64_t WorstCaseTurnWorkingSetBytes() const;
};

bool ParseV6dConfig(const BehaviorSpec &spec, V6dConfig *config, std::vector<std::string> *errors);

// Reporter identity of one simulated V6D process.
struct V6dProcessIdentity {
    uint32_t index = 0;
    std::string process_id;   // <behavior_id>-p<index>
    std::string host_ip_port; // unique per process
};

V6dProcessIdentity MakeProcessIdentity(const std::string &behavior_id, const V6dConfig &config, uint32_t index);

} // namespace kvcm_swarm
