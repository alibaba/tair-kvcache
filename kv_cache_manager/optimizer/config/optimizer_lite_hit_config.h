#pragma once

#include <string>
#include <vector>

#include "kv_cache_manager/common/jsonizable.h"
#include "kv_cache_manager/optimizer/config/optimizer_instance_group.h"
#include "kv_cache_manager/optimizer/config/optimizer_instance_info.h"

namespace kv_cache_manager {

// Configuration for the offline LiteHit entry (lite_hit_main / LiteHitOfflineRunner).
//
// It deliberately reuses the SAME config objects as the online service
// (OptimizerInstanceGroup + OptimizerInstanceInfo). Capacity is given in GB on the
// instance group and the block size / bytes-per-block come from the instance, so the
// offline path drives OnlineOptimizerManager and shares the online GB->block
// conversion and LiteHit core verbatim. Only trace_file_path/output_result_path are
// offline-specific.
class OptimizerLiteHitConfig : public Jsonizable {
public:
    OptimizerLiteHitConfig() = default;
    ~OptimizerLiteHitConfig() override = default;

    bool FromRapidValue(const rapidjson::Value &rapid_value) override;
    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override;

    [[nodiscard]] const std::string &trace_file_path() const { return trace_file_path_; }
    [[nodiscard]] const std::string &output_result_path() const { return output_result_path_; }
    [[nodiscard]] const std::vector<OptimizerInstanceGroup> &instance_groups() const { return instance_groups_; }
    [[nodiscard]] const std::vector<OptimizerInstanceInfo> &instances() const { return instances_; }
    // When true (default) the trace is assumed to be ordered by timestamp_ns, so it
    // is replayed in a single streaming pass (O(1) memory). When false the whole
    // trace is loaded and globally sorted before replay.
    [[nodiscard]] bool assume_time_sorted() const { return assume_time_sorted_; }
    // When non-empty, every replayed request is attributed to this single
    // instance_id (the trace's own instance_id is ignored). Use it to pool a
    // whole per-service trace into ONE global cache, e.g. instance_id=pod files
    // replayed as a single service-wide cache. The instances() list must define
    // exactly this id.
    [[nodiscard]] const std::string &override_instance_id() const { return override_instance_id_; }
    // When true, the per-request CSV is replaced by a compact per-capacity
    // aggregate (one row per capacity: hit blocks/tokens, input tokens, hit
    // rate, plus an "inf" row for the theoretical maximum). Avoids emitting
    // billions of rows for full-day traces.
    [[nodiscard]] bool aggregate_only() const { return aggregate_only_; }

    void set_trace_file_path(const std::string &path) { trace_file_path_ = path; }
    void set_output_result_path(const std::string &path) { output_result_path_ = path; }
    void set_instance_groups(const std::vector<OptimizerInstanceGroup> &groups) { instance_groups_ = groups; }
    void set_instances(const std::vector<OptimizerInstanceInfo> &instances) { instances_ = instances; }
    void set_assume_time_sorted(bool assume_time_sorted) { assume_time_sorted_ = assume_time_sorted; }
    void set_override_instance_id(const std::string &id) { override_instance_id_ = id; }
    void set_aggregate_only(bool v) { aggregate_only_ = v; }

private:
    std::string trace_file_path_;
    std::string output_result_path_;
    std::vector<OptimizerInstanceGroup> instance_groups_;
    std::vector<OptimizerInstanceInfo> instances_;
    bool assume_time_sorted_ = true;
    std::string override_instance_id_;
    bool aggregate_only_ = false;
};

} // namespace kv_cache_manager
