#pragma once

#include <string>
#include <unordered_set>
#include <vector>

namespace kv_cache_manager {

// Decides which configured instances replay one offline trace event.
// Exactly one of three modes applies (fanout and override are mutually
// exclusive at the config layer):
//   fanout:   every event replays on every configured instance, in config
//             order; the trace's own instance_id is ignored
//   override: every event routes to the single named instance
//   default:  route by the trace's own instance_id; an unknown id is an error
class LiteHitTraceRouter {
public:
    LiteHitTraceRouter(bool fanout_all_instances,
                       std::string override_instance_id,
                       std::vector<std::string> instance_ids);

    // Config-level consistency: a non-empty override must name a configured
    // instance.
    bool Validate(std::string &error) const;

    // Appends the target instance ids for a trace carrying
    // trace_instance_id. Returns false only in default mode when the id has
    // no configured instance.
    bool Route(const std::string &trace_instance_id, std::vector<std::string> &targets, std::string &error) const;

private:
    bool fanout_all_instances_;
    std::string override_instance_id_;
    std::vector<std::string> instance_ids_;
    std::unordered_set<std::string> known_ids_;
};

} // namespace kv_cache_manager
