#include "kv_cache_manager/optimizer/liteHit/trace_router.h"

#include <utility>

namespace kv_cache_manager {

LiteHitTraceRouter::LiteHitTraceRouter(bool fanout_all_instances,
                                       std::string override_instance_id,
                                       std::vector<std::string> instance_ids)
    : fanout_all_instances_(fanout_all_instances)
    , override_instance_id_(std::move(override_instance_id))
    , instance_ids_(std::move(instance_ids))
    , known_ids_(instance_ids_.begin(), instance_ids_.end()) {}

bool LiteHitTraceRouter::Validate(std::string &error) const {
    if (!override_instance_id_.empty() && known_ids_.count(override_instance_id_) == 0) {
        error = "override_instance_id[" + override_instance_id_ + "] has no matching configured instance";
        return false;
    }
    return true;
}

bool LiteHitTraceRouter::Route(const std::string &trace_instance_id,
                               std::vector<std::string> &targets,
                               std::string &error) const {
    if (fanout_all_instances_) {
        targets.insert(targets.end(), instance_ids_.begin(), instance_ids_.end());
        return true;
    }
    const std::string &instance_id = override_instance_id_.empty() ? trace_instance_id : override_instance_id_;
    if (known_ids_.count(instance_id) == 0) {
        error = "unknown instance[" + instance_id + "]";
        return false;
    }
    targets.push_back(instance_id);
    return true;
}

} // namespace kv_cache_manager
