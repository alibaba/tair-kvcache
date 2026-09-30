#pragma once

#include <cstddef>
#include <string>
#include <string_view>

#include "kv_cache_manager/config/instance_info.h"

namespace kv_cache_manager {

inline constexpr std::string_view kKvMetaInternalInstancePrefix = "__kv_meta_v1__";
inline constexpr std::string_view kKvMetaLocationIdPrefix = "kvmeta:v1:";
inline constexpr std::string_view kKvMetaValueSpecName = "value";
inline constexpr std::string_view kKvMetaModelName = "__kv_meta_object__";
inline constexpr std::string_view kKvMetaDtype = "opaque_bytes";
inline constexpr std::string_view kKvMetaDeploymentExtra = "kv_meta_v1";

inline bool HasKvMetaInternalInstanceId(const std::string &instance_id) noexcept {
    if (instance_id.size() <= kKvMetaInternalInstancePrefix.size() ||
        instance_id.compare(0, kKvMetaInternalInstancePrefix.size(), kKvMetaInternalInstancePrefix) != 0) {
        return false;
    }
    const std::size_t encoded_size = instance_id.size() - kKvMetaInternalInstancePrefix.size();
    if ((encoded_size & 1U) != 0) {
        return false;
    }
    for (std::size_t i = kKvMetaInternalInstancePrefix.size(); i < instance_id.size(); ++i) {
        const char c = instance_id[i];
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) {
            return false;
        }
    }
    return true;
}

// A prefix alone is deliberately insufficient: recovery and background
// maintenance must never treat an ordinary instance as a generic-object
// instance merely because its caller-selected id happens to share a prefix.
inline bool IsKvMetaInstance(const InstanceInfo &instance) noexcept {
    if (!HasKvMetaInternalInstanceId(instance.instance_id()) || instance.block_size() != 1 ||
        instance.default_query_type() != 1 || instance.location_spec_infos().size() != 1 ||
        instance.location_spec_infos().front().name() != kKvMetaValueSpecName ||
        instance.location_spec_infos().front().size() != 1 || !instance.location_spec_groups().empty()) {
        return false;
    }
    const auto &deployment = instance.model_deployment();
    return deployment.model_name() == kKvMetaModelName && deployment.dtype() == kKvMetaDtype &&
           deployment.tp_size() == 1 && deployment.dp_size() == 1 && deployment.pp_size() == 1 &&
           deployment.extra() == kKvMetaDeploymentExtra;
}

} // namespace kv_cache_manager
