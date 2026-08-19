#include "tools/kvcm_swarm/test/fake_behavior.h"

#include <mutex>

#include "tools/kvcm_swarm/evidence/json_writer.h"
#include "tools/kvcm_swarm/scenario/config_reader.h"

namespace kvcm_swarm {
namespace {

struct FakeConfig {
    Duration tick_interval = std::chrono::milliseconds(10);
};

class FakeBehavior : public ClientBehavior {
public:
    FakeBehavior(BehaviorSpec spec, FakeConfig config, RuntimeServices services)
        : spec_(std::move(spec)), config_(config), services_(services) {}

    Task<bool> Initialize(TimePoint /*deadline*/) override {
        // Only the generic transport factory is used; no domain type is needed.
        ClientIdentity identity;
        identity.behavior_type = std::string(TypeName());
        identity.behavior_id = spec_.id;
        transport_ = services_.transports.CreateClientContext(identity, spec_.transport);
        initialized_ = transport_ != nullptr;
        co_return initialized_;
    }

    void StartTraffic() override {
        running_.fetch_add(1);
        TickLoop().via(&services_.executor).start([](auto &&) {});
    }

    Task<> Drain(TimePoint deadline) override {
        own_stop_.RequestStop();
        while (running_.load() > 0 && Now() < deadline) {
            co_await SleepFor(services_.executor, std::chrono::milliseconds(2), StopToken());
        }
        co_return;
    }

    std::string_view TypeName() const override { return "fake_behavior"; }
    const std::string &Id() const override { return spec_.id; }

    void WriteReport(JsonWriter &writer) const override {
        writer.BeginObject();
        writer.KeyString("type", std::string(TypeName()));
        writer.KeyUint("ticks", ticks_.load());
        writer.KeyBool("initialized", initialized_);
        writer.EndObject();
    }

    void WriteEffectiveConfig(JsonWriter &writer) const override {
        writer.BeginObject();
        writer.KeyString("tick_interval", FormatDuration(config_.tick_interval));
        writer.EndObject();
    }

    std::vector<InvariantObservation> Invariants() const override {
        InvariantObservation observation;
        observation.behavior_type = std::string(TypeName());
        observation.check_name = "fake_ticks_advance";
        observation.checked = ticks_.load();
        observation.violations = 0;
        observation.status = ticks_.load() > 0 ? CheckStatus::kPass : CheckStatus::kNotRun;
        observation.reason = "timer-driven ticks observed";
        return {observation};
    }

    bool Quiesced() const override { return running_.load() == 0; }

private:
    Task<> TickLoop() {
        TimePoint planned = Now() + config_.tick_interval;
        while (!own_stop_.StopRequested() && !services_.stop.StopRequested()) {
            if (!co_await SleepUntil(services_.executor, planned, own_stop_.Token())) {
                break;
            }
            ticks_.fetch_add(1);
            planned += config_.tick_interval;
        }
        running_.fetch_sub(1);
        co_return;
    }

    BehaviorSpec spec_;
    FakeConfig config_;
    RuntimeServices services_;
    StopSource own_stop_;
    ClientTransportContext *transport_ = nullptr;
    bool initialized_ = false;
    std::atomic<uint64_t> ticks_{0};
    std::atomic<int> running_{0};
};

class FakeBehaviorFactory : public BehaviorFactory {
public:
    std::string_view TypeName() const override { return "fake_behavior"; }

    ValidationResult Validate(const BehaviorSpec &spec) const override {
        ValidationResult result;
        std::vector<std::string> errors;
        ConfigReader reader(spec.config, &errors);
        reader.OptionalDuration("tick_interval", std::chrono::milliseconds(10));
        std::vector<std::string> unknown;
        spec.config.CollectUnknown(&unknown);
        for (const auto &key : unknown) {
            result.Fail("unknown configuration field: " + key);
        }
        for (const auto &error : errors) {
            result.Fail(error);
        }
        return result;
    }

    std::unique_ptr<ClientBehavior> Create(const BehaviorSpec &spec, RuntimeServices services) const override {
        std::vector<std::string> errors;
        ConfigReader reader(spec.config, &errors);
        FakeConfig config;
        config.tick_interval = reader.OptionalDuration("tick_interval", std::chrono::milliseconds(10));
        if (!errors.empty()) {
            return nullptr;
        }
        return std::make_unique<FakeBehavior>(spec, config, services);
    }

    BehaviorIdentityClaims Claims(const BehaviorSpec &spec) const override {
        BehaviorIdentityClaims claims;
        claims.exclusive_names.push_back("fake_behavior:" + spec.id);
        return claims;
    }
};

} // namespace

std::unique_ptr<BehaviorFactory> MakeFakeBehaviorFactory() { return std::make_unique<FakeBehaviorFactory>(); }

} // namespace kvcm_swarm
