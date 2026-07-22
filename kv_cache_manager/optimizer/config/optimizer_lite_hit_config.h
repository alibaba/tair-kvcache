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
// (OptimizerInstanceGroup + OptimizerInstanceInfo): instance registration is
// validated by OnlineOptimizerManager exactly like online. The replay itself
// produces one capacity-independent facts CSV
// (${output_result_path}/litehit_facts.csv); capacities are applied
// afterwards by the facts query tool.
class OptimizerLiteHitConfig : public Jsonizable {
public:
    OptimizerLiteHitConfig() = default;
    ~OptimizerLiteHitConfig() override = default;

    bool FromRapidValue(const rapidjson::Value &rapid_value) override;
    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override;

    [[nodiscard]] const std::string &trace_file_path() const { return trace_file_path_; }
    // Output DIRECTORY. The replay atomically publishes
    // ${output_result_path}/litehit_facts.csv on success.
    [[nodiscard]] const std::string &output_result_path() const { return output_result_path_; }
    [[nodiscard]] const std::vector<OptimizerInstanceGroup> &instance_groups() const { return instance_groups_; }
    [[nodiscard]] const std::vector<OptimizerInstanceInfo> &instances() const { return instances_; }
    // When non-empty, every replayed request is attributed to this single
    // instance_id (the trace's own instance_id is ignored). Use it to pool a
    // whole per-service trace into ONE global cache, e.g. instance_id=pod files
    // replayed as a single service-wide cache. The instances() list must define
    // exactly this id.
    [[nodiscard]] const std::string &override_instance_id() const { return override_instance_id_; }
    // The only public parallelism knob of the offline pipeline. Values < 1 are
    // clamped to 1; queue/window sizes are derived internally.
    [[nodiscard]] int32_t pipeline_worker_count() const { return pipeline_worker_count_; }

    void set_trace_file_path(const std::string &path) { trace_file_path_ = path; }
    void set_output_result_path(const std::string &path) { output_result_path_ = path; }
    void set_instance_groups(const std::vector<OptimizerInstanceGroup> &groups) { instance_groups_ = groups; }
    void set_instances(const std::vector<OptimizerInstanceInfo> &instances) { instances_ = instances; }
    void set_override_instance_id(const std::string &id) { override_instance_id_ = id; }
    void set_pipeline_worker_count(int32_t count) { pipeline_worker_count_ = count; }

private:
    std::string trace_file_path_;
    std::string output_result_path_;
    std::vector<OptimizerInstanceGroup> instance_groups_;
    std::vector<OptimizerInstanceInfo> instances_;
    std::string override_instance_id_;
    int32_t pipeline_worker_count_ = 1;
};

} // namespace kv_cache_manager
