// Behavior contract.
//
// A behavior owns its own domain state. The common runtime knows nothing about
// V6D sessions, prefixes, local caches, location owners or reporter
// generations; `RuntimeServices` deliberately exposes none of them.
#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "tools/kvcm_swarm/evidence/json_writer.h"
#include "tools/kvcm_swarm/evidence/observation.h"
#include "tools/kvcm_swarm/evidence/sink.h"
#include "tools/kvcm_swarm/runtime/admission.h"
#include "tools/kvcm_swarm/runtime/executor.h"
#include "tools/kvcm_swarm/runtime/rng.h"
#include "tools/kvcm_swarm/scenario/config_node.h"
#include "tools/kvcm_swarm/transport/transport.h"
#include "tools/kvcm_swarm/transport/transport_provider.h"

namespace kvcm_swarm {

struct BehaviorSpec {
    std::string id;
    std::string type;
    TransportKind transport = TransportKind::kHttp;
    ConfigNode config;
};

struct RuntimeServices {
    SwarmExecutor &executor;
    AdmissionController &admission;
    TransportProvider &transports;
    EvidenceSink &evidence;
    const SeedDeriver &seeds;
    StopToken stop;
    // Behaviors read the phase but never change it.
    const PhaseSource &phase;
};

struct ValidationResult {
    bool ok = true;
    std::vector<std::string> errors;

    void Fail(std::string message) {
        ok = false;
        errors.push_back(std::move(message));
    }
    void Absorb(const ValidationResult &other) {
        if (!other.ok) {
            ok = false;
            errors.insert(errors.end(), other.errors.begin(), other.errors.end());
        }
    }
};

class ClientBehavior {
public:
    virtual ~ClientBehavior() = default;

    // Creates connections, registers, and reaches a ready barrier.
    virtual Task<bool> Initialize(TimePoint deadline) = 0;
    // Starts timer-driven long-running operations. Never blocks the caller.
    virtual void StartTraffic() = 0;
    // Idempotent. Closes in-flight work within the deadline.
    virtual Task<> Drain(TimePoint deadline) = 0;

    virtual std::string_view TypeName() const = 0;
    virtual const std::string &Id() const = 0;

    // Emits the behavior's own facts under "behaviors.<id>".
    virtual void WriteReport(JsonWriter &writer) const = 0;
    // Emits the effective configuration actually used.
    virtual void WriteEffectiveConfig(JsonWriter &writer) const = 0;
    // Contract results owned by this behavior.
    virtual std::vector<InvariantObservation> Invariants() const = 0;
    // Optional sections. Return false when the behavior has nothing to add.
    virtual bool WriteCacheReport(JsonWriter & /*writer*/) const { return false; }
    virtual bool WriteWorkloadShape(JsonWriter & /*writer*/) const { return false; }
    virtual bool WriteCleanupReport(JsonWriter & /*writer*/) const { return false; }
    // True once every asynchronous operation has finished and released state.
    virtual bool Quiesced() const = 0;
};

// Behavior-declared claims, so the common loader can enforce cross-behavior
// uniqueness and target consistency without knowing any domain type.
struct BehaviorIdentityClaims {
    // Names that must be unique across the whole run, e.g. an instance_id or a
    // reporter address. Rendered as "<kind>:<value>".
    std::vector<std::string> exclusive_names;
    // Instance groups that must already be declared in target.instance_groups.
    std::vector<std::string> required_instance_groups;
};

class BehaviorFactory {
public:
    virtual ~BehaviorFactory() = default;
    virtual ValidationResult Validate(const BehaviorSpec &spec) const = 0;
    virtual std::unique_ptr<ClientBehavior> Create(const BehaviorSpec &spec, RuntimeServices services) const = 0;
    virtual std::string_view TypeName() const = 0;
    virtual BehaviorIdentityClaims Claims(const BehaviorSpec & /*spec*/) const { return {}; }
};

} // namespace kvcm_swarm
