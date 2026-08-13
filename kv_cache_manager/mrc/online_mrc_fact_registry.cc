#include "kv_cache_manager/mrc/online_mrc_fact_registry.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <exception>
#include <limits>

#include "kv_cache_manager/common/error_code.h"
#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/metrics/metrics_registry.h"
#include "kv_cache_manager/optimizer/config/optimizer_instance_group.h"
#include "kv_cache_manager/optimizer/config/optimizer_instance_info.h"
#include "kv_cache_manager/optimizer/config/optimizer_registry_manager.h"
#include "kv_cache_manager/optimizer/liteHit/request_preprocess.h"
#include "kv_cache_manager/optimizer/online_runtime/online_optimizer_manager.h"

namespace kv_cache_manager {
namespace {

constexpr char kMetricTheoreticalHitRate[] = "online_mrc.theoretical_hit_rate";
constexpr char kMetricFactCount[] = "online_mrc.fact_count";
constexpr char kMetricFactMemoryBytes[] = "online_mrc.fact_memory_bytes";
constexpr char kMetricLiteHitMemoryBytes[] = "online_mrc.lite_hit_memory_bytes";
constexpr char kMetricInstanceMemoryBytes[] = "online_mrc.instance_memory_bytes";
constexpr char kMetricTotalMemoryBytes[] = "online_mrc.total_memory_bytes";
constexpr char kMetricProjectionDurationUs[] = "online_mrc.projection_duration_us";
constexpr char kMetricProjectionFactScans[] = "online_mrc.projection_fact_scans";
constexpr char kMetricTrackedInstances[] = "online_mrc.tracked_instances";
constexpr char kMetricTrackedGroups[] = "online_mrc.tracked_groups";
constexpr char kMetricGroupInstances[] = "online_mrc.group_instances";
constexpr char kMetricOutOfOrderEvents[] = "online_mrc.out_of_order_events";
constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;

std::string FormatDouble(double value) {
    char buffer[32];
    snprintf(buffer, sizeof(buffer), "%g", value);
    return buffer;
}

OptimizerInstanceInfo ToInstanceInfo(const proto::optimizer::CacheEventBatch &batch) {
    const auto &meta = batch.instance_meta();
    std::vector<LocationSpecInfo> specs;
    specs.reserve(meta.location_spec_infos_size());
    for (const auto &spec : meta.location_spec_infos()) {
        specs.emplace_back(spec.name(), spec.size());
    }

    std::vector<LocationSpecGroup> spec_groups;
    spec_groups.reserve(meta.location_spec_groups_size());
    for (const auto &group : meta.location_spec_groups()) {
        spec_groups.emplace_back(group.name(), std::vector<std::string>(group.spec_names().begin(),
                                                                       group.spec_names().end()));
    }

    const auto &state = meta.optimizer_state_info();
    return OptimizerInstanceInfo(batch.instance_group(),
                                 batch.instance_id(),
                                 meta.block_size(),
                                 specs,
                                 spec_groups,
                                 meta.linear_step(),
                                 OptimizerStateInfo(state.full_location_spec_group_name(),
                                                    state.linear_location_spec_group_name()));
}

bool SameFormalGroup(const OptimizerInstanceGroup &lhs, const OptimizerInstanceGroup &rhs) {
    return lhs.name() == rhs.name() && lhs.capacity_gb() == rhs.capacity_gb() &&
           lhs.eviction_policy() == rhs.eviction_policy() && lhs.shared_group_quota() == rhs.shared_group_quota() &&
           lhs.enable_theoretical_max_cache() == rhs.enable_theoretical_max_cache() &&
           lhs.ttl_seconds() == rhs.ttl_seconds() && lhs.enable_prefix_hash() == rhs.enable_prefix_hash();
}

} // namespace

OnlineMrcFactRegistry::OnlineMrcFactRegistry(const OnlineMrcConfig &config,
                                             const std::vector<OptimizerInstanceGroup> &instance_groups,
                                             std::shared_ptr<MetricsRegistry> metrics_registry,
                                             std::shared_ptr<OnlineOptimizerManager> manager)
    : config_(config),
      instance_groups_(instance_groups),
      metrics_registry_(std::move(metrics_registry)),
      manager_(std::move(manager)),
      capacity_gb_grid_(config.capacity_gb_grid) {}

bool OnlineMrcFactRegistry::Init() {
    if (!manager_ || !manager_->registry_manager() || !ValidateCapacityGrid(config_.capacity_gb_grid) ||
        instance_groups_.empty()) {
        KVCM_LOG_ERROR("online mrc: manager, projection grid, and formal instance groups are required");
        return false;
    }

    std::vector<OptimizerInstanceGroup> desired_groups;
    desired_groups.reserve(instance_groups_.size());
    for (const auto &group : instance_groups_) {
        std::string invalid_fields;
        if (!group.ValidateRequiredFields(invalid_fields) || !group.enable_prefix_hash()) {
            KVCM_LOG_ERROR("online mrc: invalid formal group[%s] or enable_prefix_hash is not true: %s",
                           group.name().c_str(),
                           invalid_fields.c_str());
            return false;
        }
        const auto existing = manager_->registry_manager()->GetInstanceGroup(group.name());
        if (existing && !SameFormalGroup(*existing, group)) {
            KVCM_LOG_ERROR("online mrc: formal group[%s] already exists with different configuration; refusing to "
                           "modify it",
                           group.name().c_str());
            return false;
        }
        desired_groups.push_back(group);
    }

    for (const auto &group : desired_groups) {
        if (!manager_->registry_manager()->GetInstanceGroup(group.name())) {
            const ErrorCode ec = manager_->CreateInstanceGroup(group);
            if (ec != EC_OK) {
                KVCM_LOG_ERROR("online mrc: CreateInstanceGroup[%s] failed, ec=%d",
                               group.name().c_str(),
                               static_cast<int>(ec));
                return false;
            }
        }
    }
    {
        std::lock_guard<std::mutex> guard(contexts_mutex_);
        for (const auto &group : desired_groups) {
            groups_.emplace(group.name(), GroupContext{});
        }
        initialized_ = true;
    }
    return true;
}

bool OnlineMrcFactRegistry::RegisterFormalInstance(const proto::optimizer::CacheEventBatch &batch,
                                                   std::string &serialized_instance_info,
                                                   RegisterInstanceResult &result) {
    if (!manager_ || batch.instance_meta().linear_step() != 0) {
        KVCM_LOG_WARN("online mrc: instance[%s] requires full-only metadata (linear_step=0)",
                      batch.instance_id().c_str());
        return false;
    }
    const OptimizerInstanceInfo instance_info = ToInstanceInfo(batch);
    serialized_instance_info = instance_info.ToJsonString();

    bool same_active_instance = false;
    const ErrorCode state_ec = manager_->GetInstanceState(batch.instance_id(), [&](const InstanceState &state) {
        if (state.instance_info && state.instance_info->ToJsonString() == serialized_instance_info) {
            same_active_instance = true;
            result.size_full_only = state.size_full_only;
            result.size_full_linear = state.size_full_linear;
        }
    });
    if (state_ec == EC_OK && same_active_instance && result.size_full_only > 0) {
        return true;
    }

    const ErrorCode ec = manager_->RegisterInstance(instance_info, result);
    if (ec != EC_OK || result.size_full_only <= 0) {
        KVCM_LOG_ERROR("online mrc: formal RegisterInstance[%s] failed, ec=%d size_full_only=%ld",
                       batch.instance_id().c_str(),
                       static_cast<int>(ec),
                       static_cast<long>(result.size_full_only));
        return false;
    }
    return true;
}

std::shared_ptr<OnlineMrcFactRegistry::InstanceContext>
OnlineMrcFactRegistry::RegisterOrGetInstance(const proto::optimizer::CacheEventBatch &batch) {
    std::lock_guard<std::mutex> guard(contexts_mutex_);
    const auto existing = contexts_.find(batch.instance_id());
    if (existing != contexts_.end()) {
        return existing->second;
    }
    if (!initialized_ || groups_.find(batch.instance_group()) == groups_.end()) {
        KVCM_LOG_WARN("online mrc: reject instance[%s] from unconfigured formal group[%s]",
                      batch.instance_id().c_str(),
                      batch.instance_group().c_str());
        return nullptr;
    }
    if (static_cast<int32_t>(contexts_.size()) >= config_.max_instances) {
        KVCM_LOG_WARN("online mrc: instance limit[%d] reached, reject instance[%s]",
                      config_.max_instances,
                      batch.instance_id().c_str());
        return nullptr;
    }

    RegisterInstanceResult register_result;
    std::string serialized_instance_info;
    if (!RegisterFormalInstance(batch, serialized_instance_info, register_result)) {
        return nullptr;
    }
    const auto formal_group = manager_->registry_manager()->GetInstanceGroup(batch.instance_group());
    if (!formal_group) {
        return nullptr;
    }

    auto context = std::make_shared<InstanceContext>();
    context->instance_id = batch.instance_id();
    context->instance_group = batch.instance_group();
    context->serialized_instance_info = std::move(serialized_instance_info);
    context->block_size = batch.instance_meta().block_size();
    context->block_bytes = register_result.size_full_only;
    context->enable_prefix_hash = formal_group->enable_prefix_hash();
    contexts_.emplace(context->instance_id, context);
    groups_.at(context->instance_group).instance_ids.emplace(context->instance_id);
    return context;
}

void OnlineMrcFactRegistry::MoveInstanceToGroup(const std::string &instance_id,
                                                const std::string &old_group,
                                                const std::string &new_group) {
    if (old_group == new_group) {
        return;
    }
    std::lock_guard<std::mutex> guard(contexts_mutex_);
    const auto old_it = groups_.find(old_group);
    if (old_it != groups_.end()) {
        old_it->second.instance_ids.erase(instance_id);
    }
    groups_.at(new_group).instance_ids.emplace(instance_id);
}

bool OnlineMrcFactRegistry::Observe(const proto::optimizer::CacheEventBatch &batch) {
    if (batch.instance_id().empty() || batch.instance_group().empty() || !batch.has_instance_meta() ||
        batch.events().empty()) {
        return false;
    }

    auto context = RegisterOrGetInstance(batch);
    if (!context) {
        return false;
    }

    std::lock_guard<std::mutex> guard(context->mutex);
    const std::string candidate_info = ToInstanceInfo(batch).ToJsonString();
    if (candidate_info != context->serialized_instance_info) {
        {
            std::lock_guard<std::mutex> contexts_guard(contexts_mutex_);
            if (groups_.find(batch.instance_group()) == groups_.end()) {
                KVCM_LOG_WARN("online mrc: reject metadata change for instance[%s] to unconfigured group[%s]",
                              batch.instance_id().c_str(),
                              batch.instance_group().c_str());
                return false;
            }
        }
        RegisterInstanceResult register_result;
        std::string serialized_instance_info;
        if (!RegisterFormalInstance(batch, serialized_instance_info, register_result)) {
            return false;
        }
        const auto formal_group = manager_->registry_manager()->GetInstanceGroup(batch.instance_group());
        if (!formal_group) {
            return false;
        }
        MoveInstanceToGroup(context->instance_id, context->instance_group, batch.instance_group());
        context->instance_group = batch.instance_group();
        context->serialized_instance_info = std::move(serialized_instance_info);
        context->block_size = batch.instance_meta().block_size();
        context->block_bytes = register_result.size_full_only;
        context->enable_prefix_hash = formal_group->enable_prefix_hash();
        context->lite_hit.Reset();
        context->facts.clear();
        context->last_timestamp_ns = 0;
        context->out_of_order_events = 0;
        ++context->meta_generation;
    }

    struct PreparedEvent {
        int64_t timestamp_ns = 0;
        NormalizedRequest request;
    };
    std::vector<PreparedEvent> prepared;
    prepared.reserve(batch.events_size());
    try {
        for (const auto &event : batch.events()) {
            PreparedEvent item;
            item.timestamp_ns = event.timestamp_ns();
            item.request = NormalizeRequest(std::vector<int64_t>(event.block_keys().begin(), event.block_keys().end()),
                                            event.input_token_len(),
                                            static_cast<uint64_t>(context->block_size),
                                            context->enable_prefix_hash);
            prepared.push_back(std::move(item));
        }
    } catch (const std::exception &e) {
        KVCM_LOG_WARN("online mrc: reject invalid batch for instance[%s]: %s",
                      batch.instance_id().c_str(),
                      e.what());
        return false;
    }

    for (auto &event : prepared) {
        if (event.timestamp_ns < context->last_timestamp_ns) {
            ++context->out_of_order_events;
            continue;
        }
        context->last_timestamp_ns = event.timestamp_ns;
        StoredFact fact;
        fact.input_token_len = event.request.input_token_len;
        fact.fact = context->lite_hit.ProcessRequest(event.request.block_keys);
        context->facts.push_back(std::move(fact));
    }
    return true;
}

void OnlineMrcFactRegistry::ReportMetrics() {
    if (!metrics_registry_) {
        return;
    }
    const auto projection_start = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> projection_guard(projection_mutex_);
    for (const char *metric_name :
         {kMetricTheoreticalHitRate,
          kMetricFactCount,
          kMetricFactMemoryBytes,
          kMetricLiteHitMemoryBytes,
          kMetricInstanceMemoryBytes,
          kMetricGroupInstances,
          kMetricOutOfOrderEvents}) {
        const auto data = metrics_registry_->GetMetricsData(metric_name);
        if (data) {
            data->RemoveByTagFilter({});
        }
    }

    std::map<std::string, std::shared_ptr<InstanceContext>> contexts;
    std::map<std::string, size_t> group_sizes;
    {
        std::lock_guard<std::mutex> guard(contexts_mutex_);
        contexts = contexts_;
        for (const auto &[name, group] : groups_) {
            group_sizes.emplace(name, group.instance_ids.size());
        }
    }
    const MetricsTags empty_tags;
    REPORT_DYNAMIC_GAUGE_(metrics_registry_, kMetricTrackedInstances, empty_tags, contexts.size());
    REPORT_DYNAMIC_GAUGE_(metrics_registry_, kMetricTrackedGroups, empty_tags, group_sizes.size());
    for (const auto &[group, instance_count] : group_sizes) {
        const MetricsTags group_tags{{"instance_group", group}};
        REPORT_DYNAMIC_GAUGE_(metrics_registry_, kMetricGroupInstances, group_tags, instance_count);
    }

    uint64_t total_memory_bytes = 0;
    uint64_t projection_fact_scans = 0;
    for (const auto &[_, context] : contexts) {
        std::lock_guard<std::mutex> guard(context->mutex);
        MetricsTags instance_tags{{"instance_group", context->instance_group},
                                  {"instance_id", context->instance_id},
                                  {"meta_generation", std::to_string(context->meta_generation)}};
        REPORT_DYNAMIC_GAUGE_(metrics_registry_, kMetricFactCount, instance_tags, context->facts.size());
        const uint64_t fact_memory_bytes = FactMemoryBytes(context->facts);
        const uint64_t lite_hit_memory_bytes = context->lite_hit.memory_usage_bytes();
        const uint64_t instance_memory_bytes = fact_memory_bytes + lite_hit_memory_bytes;
        total_memory_bytes += instance_memory_bytes;
        REPORT_DYNAMIC_GAUGE_(metrics_registry_, kMetricFactMemoryBytes, instance_tags, fact_memory_bytes);
        REPORT_DYNAMIC_GAUGE_(metrics_registry_, kMetricLiteHitMemoryBytes, instance_tags, lite_hit_memory_bytes);
        REPORT_DYNAMIC_GAUGE_(metrics_registry_, kMetricInstanceMemoryBytes, instance_tags, instance_memory_bytes);
        REPORT_DYNAMIC_GAUGE_(
            metrics_registry_, kMetricOutOfOrderEvents, instance_tags, context->out_of_order_events);
        for (const double capacity_gb : capacity_gb_grid_) {
            projection_fact_scans += context->facts.size();
            const uint64_t capacity_bytes = static_cast<uint64_t>(capacity_gb * kGiB);
            uint64_t total_tokens = 0;
            uint64_t hit_tokens = 0;
            for (const auto &stored : context->facts) {
                const uint64_t hit_blocks = HitCurveProjector::ProjectBytes(
                    stored.fact, capacity_bytes, static_cast<uint64_t>(context->block_bytes));
                total_tokens += stored.input_token_len;
                hit_tokens += std::min<uint64_t>(
                    hit_blocks * static_cast<uint64_t>(context->block_size), stored.input_token_len);
            }
            MetricsTags tags = instance_tags;
            tags["capacity_gb"] = FormatDouble(capacity_gb);
            const double hit_rate = total_tokens == 0 ? 0.0 : static_cast<double>(hit_tokens) / total_tokens;
            REPORT_DYNAMIC_GAUGE_(metrics_registry_, kMetricTheoreticalHitRate, tags, hit_rate);
        }
    }
    const auto projection_end = std::chrono::steady_clock::now();
    const auto projection_duration_us =
        std::chrono::duration_cast<std::chrono::microseconds>(projection_end - projection_start).count();
    REPORT_DYNAMIC_GAUGE_(metrics_registry_, kMetricTotalMemoryBytes, empty_tags, total_memory_bytes);
    REPORT_DYNAMIC_GAUGE_(metrics_registry_, kMetricProjectionDurationUs, empty_tags, projection_duration_us);
    REPORT_DYNAMIC_GAUGE_(metrics_registry_, kMetricProjectionFactScans, empty_tags, projection_fact_scans);
}

uint64_t OnlineMrcFactRegistry::FactMemoryBytes(const std::deque<StoredFact> &facts) {
    // Account for live Fact payload and reserved hit-curve storage. std::deque
    // implementation slack and allocator metadata are intentionally excluded;
    // process RSS remains the authoritative OOM signal.
    uint64_t bytes = sizeof(facts) + static_cast<uint64_t>(facts.size()) * sizeof(StoredFact);
    for (const auto &stored : facts) {
        bytes += static_cast<uint64_t>(stored.fact.hit_curve.capacity()) * sizeof(HitCurveSegment);
    }
    return bytes;
}

size_t OnlineMrcFactRegistry::InstanceCount() const {
    std::lock_guard<std::mutex> guard(contexts_mutex_);
    return contexts_.size();
}

size_t OnlineMrcFactRegistry::GroupCount() const {
    std::lock_guard<std::mutex> guard(contexts_mutex_);
    return groups_.size();
}

size_t OnlineMrcFactRegistry::GroupInstanceCount(const std::string &instance_group) const {
    std::lock_guard<std::mutex> guard(contexts_mutex_);
    const auto it = groups_.find(instance_group);
    return it == groups_.end() ? 0 : it->second.instance_ids.size();
}

size_t OnlineMrcFactRegistry::FactCount(const std::string &instance_id) const {
    std::shared_ptr<InstanceContext> context;
    {
        std::lock_guard<std::mutex> guard(contexts_mutex_);
        const auto it = contexts_.find(instance_id);
        if (it == contexts_.end()) {
            return 0;
        }
        context = it->second;
    }
    std::lock_guard<std::mutex> guard(context->mutex);
    return context->facts.size();
}

uint64_t OnlineMrcFactRegistry::MetaGeneration(const std::string &instance_id) const {
    std::shared_ptr<InstanceContext> context;
    {
        std::lock_guard<std::mutex> guard(contexts_mutex_);
        const auto it = contexts_.find(instance_id);
        if (it == contexts_.end()) {
            return 0;
        }
        context = it->second;
    }
    std::lock_guard<std::mutex> guard(context->mutex);
    return context->meta_generation;
}

bool OnlineMrcFactRegistry::ValidateCapacityGrid(const std::vector<double> &capacity_gb_grid) {
    if (capacity_gb_grid.empty()) {
        return false;
    }
    double previous = 0.0;
    for (const double capacity_gb : capacity_gb_grid) {
        if (!std::isfinite(capacity_gb) || capacity_gb <= previous ||
            capacity_gb > static_cast<double>(std::numeric_limits<int64_t>::max()) / kGiB) {
            return false;
        }
        previous = capacity_gb;
    }
    return true;
}

bool OnlineMrcFactRegistry::UpdateCapacityGrid(const std::vector<double> &capacity_gb_grid) {
    if (!ValidateCapacityGrid(capacity_gb_grid)) {
        return false;
    }
    std::lock_guard<std::mutex> guard(projection_mutex_);
    if (capacity_gb_grid_ == capacity_gb_grid) {
        return true;
    }
    capacity_gb_grid_ = capacity_gb_grid;
    ++projection_generation_;
    if (metrics_registry_) {
        const auto hit_rate_data = metrics_registry_->GetMetricsData(kMetricTheoreticalHitRate);
        if (hit_rate_data) {
            hit_rate_data->RemoveByTagFilter({});
        }
    }
    return true;
}

std::vector<double> OnlineMrcFactRegistry::CapacityGrid() const {
    std::lock_guard<std::mutex> guard(projection_mutex_);
    return capacity_gb_grid_;
}

uint64_t OnlineMrcFactRegistry::ProjectionGeneration() const {
    std::lock_guard<std::mutex> guard(projection_mutex_);
    return projection_generation_;
}

} // namespace kv_cache_manager
