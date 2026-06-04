#include "kv_cache_manager/optimizer/manager/optimizer_manager.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/meta/cache_location.h"
#include "kv_cache_manager/optimizer/analysis/tracker/block_lifecycle_tracker.h"
#include "kv_cache_manager/optimizer/config/tier_config.h"
#include "kv_cache_manager/optimizer/eviction_policy/policy_factory.h"
#include "kv_cache_manager/optimizer/trace_loader/trace_util.h"
namespace kv_cache_manager {
namespace {
int64_t RequirePositiveInputLen(const char *api_name, int64_t input_len) {
    if (input_len > 0) {
        return input_len;
    }
    std::string message = std::string(api_name) + " requires positive input_len";
    KVCM_LOG_ERROR("%s", message.c_str());
    throw std::runtime_error(message);
}
} // namespace

OptimizerManager::OptimizerManager(const OptimizerConfig &config,
                                   bool enable_lifecycle_tracking,
                                   bool enable_template_analysis,
                                   HitRatePerspective hit_rate_perspective)
    : config_(config)
    , enable_lifecycle_tracking_(enable_lifecycle_tracking)
    , enable_template_analysis_(enable_template_analysis)
    , hit_rate_perspective_(hit_rate_perspective) {}

bool OptimizerManager::Init() {
    eviction_manager_.reset(new OptEvictionManager());
    if (!eviction_manager_->Init(config_.eviction_config())) {
        KVCM_LOG_ERROR("Failed to initialize eviction manager.");
        return false;
    }
    indexer_manager_.reset(new OptIndexerManager(eviction_manager_));

    if (config_.instance_groups().empty()) {
        KVCM_LOG_ERROR("No instance groups found in configuration.");
        return false;
    }

    // ---- 初始化 StatsCollector 并注册子 Tracker ----
    stats_collector_ = std::make_shared<StatsCollector>();
    hit_rate_tracker_ = stats_collector_->EmplaceTracker<HitRateTracker>(hit_rate_perspective_);

    if (enable_template_analysis_) {
        template_prefix_tracker_ = stats_collector_->EmplaceTracker<TemplatePrefixTracker>();
        KVCM_LOG_INFO("Template analysis enabled");
    } else {
        KVCM_LOG_DEBUG("Template analysis disabled (replay performance optimization)");
    }

    if (enable_lifecycle_tracking_) {
        stats_collector_->EmplaceTracker<BlockLifecycleTracker>();
        KVCM_LOG_INFO("Lifecycle tracking enabled");
    } else {
        KVCM_LOG_DEBUG("Lifecycle tracking disabled (memory optimization)");
    }

    size_t total_instances = 0;
    size_t failed_instances = 0;
    std::vector<std::string> failed_instance_ids;

    for (auto &group : config_.mutable_instance_groups()) {
        auto group_name = group.group_name();

        if (instance_group_configs_.find(group_name) != instance_group_configs_.end()) {
            KVCM_LOG_WARN("Duplicate group_name found: %s", group_name.c_str());
            continue;
        }

        if (group.storages().empty()) {
            KVCM_LOG_WARN("No storage configs found for group: %s", group_name.c_str());
            continue;
        }

        instance_group_configs_[group_name] = group;

        if (group.instances().empty()) {
            KVCM_LOG_WARN("No instances found in group: %s", group_name.c_str());
            continue;
        }

        for (auto &instance : group.instances()) {
            total_instances++;
            auto instance_config = instance;
            auto instance_id = instance_config.instance_id();

            if (instance_configs_.find(instance_id) != instance_configs_.end()) {
                KVCM_LOG_WARN("Duplicate instance_id found: %s", instance_id.c_str());
                continue;
            }

            instance_config.set_instance_group_name(group_name);

            if (instance_group_configs_.find(group_name) == instance_group_configs_.end()) {
                KVCM_LOG_WARN(
                    "Instance group '%s' not found for instance: %s", group_name.c_str(), instance_id.c_str());
                continue;
            }

            instance_configs_[instance_id] = instance_config;
            instance_group_ttl_disabled_[instance_id] = (group.default_block_ttl_seconds() == 0);
            instance_ttl_refresh_on_read_[instance_id] =
                (instance_config.eviction_policy_type() == EvictionPolicyType::POLICY_TTL) ? group.ttl_refresh_on_read()
                                                                                           : true;

            if (instance_config.eviction_policy_type() == EvictionPolicyType::POLICY_TTL &&
                group.default_block_ttl_seconds() == 0 &&
                std::holds_alternative<TtlParams>(instance_config.eviction_policy_param())) {
                const auto &ttl_params = std::get<TtlParams>(instance_config.eviction_policy_param());
                if (!ttl_params.fallback_on_pressure) {
                    KVCM_LOG_WARN("instance %s uses TTL policy with default_block_ttl_seconds=0 and "
                                  "fallback_on_pressure=false; no expiration and no capacity fallback, "
                                  "cache may grow without bound",
                                  instance_id.c_str());
                }
            }

            if (!CreateRadixTreeIndex(instance_config, group.storages())) {
                KVCM_LOG_ERROR("Failed to create RadixTreeIndex for instance: %s", instance_id.c_str());
                failed_instances++;
                failed_instance_ids.push_back(instance_id);
                instance_configs_.erase(instance_id);
                instance_group_ttl_disabled_.erase(instance_id);
                instance_ttl_refresh_on_read_.erase(instance_id);
                continue;
            }

            // 将 StatsCollector 注入到 RadixTreeIndex（替代原先的 lifecycle_tracker 注入）
            auto indexer = indexer_manager_->GetOptIndexer(instance_id);
            if (indexer) {
                indexer->SetStatsCollector(stats_collector_);
            }
        }
    }

    if (failed_instances > 0) {
        KVCM_LOG_ERROR("Failed to initialize %zu out of %zu instances", failed_instances, total_instances);
        for (const auto &id : failed_instance_ids) {
            KVCM_LOG_WARN("Failed instance: %s", id.c_str());
        }
        if (failed_instances == total_instances) {
            KVCM_LOG_ERROR("All instances failed to initialize.");
            return false;
        }
        KVCM_LOG_WARN("Continuing with %zu successful instances", total_instances - failed_instances);
    }

    if (instance_configs_.empty()) {
        KVCM_LOG_ERROR("No instances successfully initialized.");
        return false;
    }

    indexer_manager_->RegisterInstanceGroups(instance_group_configs_);
    indexer_manager_->RegisterInstances(instance_configs_);

    optimizer_runner_.reset(new OptimizerRunner(indexer_manager_,
                                                eviction_manager_,
                                                stats_collector_,
                                                instance_group_ttl_disabled_,
                                                instance_ttl_refresh_on_read_));
    return true;
}

bool OptimizerManager::CreateRadixTreeIndex(const OptInstanceConfig &instance_config,
                                            const std::vector<OptTierConfig> &storage_configs) {
    auto group_it = instance_group_configs_.find(instance_config.instance_group_name());
    if (group_it == instance_group_configs_.end()) {
        KVCM_LOG_ERROR("Instance group '%s' not found for instance: %s",
                       instance_config.instance_group_name().c_str(),
                       instance_config.instance_id().c_str());
        return false;
    }

    int64_t default_ttl_ns = group_it->second.default_block_ttl_seconds() * 1000000000;
    if (default_ttl_ns > 0 && instance_config.eviction_policy_type() != EvictionPolicyType::POLICY_TTL) {
        KVCM_LOG_WARN("default_block_ttl_seconds=%ld is set but eviction_policy_type is not TTL; "
                      "TTL will not be enforced for instance %s",
                      group_it->second.default_block_ttl_seconds(),
                      instance_config.instance_id().c_str());
    }
    if (!indexer_manager_->CreateOptIndexer(instance_config,
                                            storage_configs,
                                            group_it->second.hierarchical_eviction_enabled(),
                                            group_it->second.tier_write_mode(),
                                            default_ttl_ns,
                                            static_cast<size_t>(group_it->second.selective_write_threshold()),
                                            group_it->second.tier_access_propagation_enabled(),
                                            group_it->second.tier_flow_strategies())) {
        KVCM_LOG_ERROR("Failed to create optimizer indexer for instance_id: %s", instance_config.instance_id().c_str());
        return false;
    }
    KVCM_LOG_INFO("Created optimizer indexer for instance_id: %s", instance_config.instance_id().c_str());
    return true;
}

void OptimizerManager::DirectRun() {
    if (!optimizer_runner_) {
        KVCM_LOG_ERROR("optimizer_runner_ is not initialized");
        return;
    }
    optimizer_runner_->Run(config_);
}

WriteCacheRes OptimizerManager::WriteCache(const std::string &instance_id,
                                           const std::string &trace_id,
                                           const int64_t timestamp,
                                           const std::vector<int64_t> &block_ids,
                                           const int64_t ttl_seconds) {
    int64_t ttl_us = (ttl_seconds > 0) ? ttl_seconds * 1000000 : ttl_seconds;
    return WriteCacheWithTtlUs(instance_id, trace_id, timestamp, block_ids, ttl_us);
}

WriteCacheRes OptimizerManager::WriteCacheWithTtlUs(const std::string &instance_id,
                                                    const std::string &trace_id,
                                                    const int64_t timestamp,
                                                    const std::vector<int64_t> &block_ids,
                                                    const int64_t ttl_us) {
    WriteCacheSchemaTrace trace;
    trace.set_instance_id(instance_id);
    trace.set_trace_id(trace_id);
    trace.set_timestamp_ns(timestamp);
    trace.set_keys(block_ids);
    trace.set_ttl_us(ttl_us);
    const auto write_record = optimizer_runner_->HandleWriteCache(trace);
    stats_collector_->UpdateTimestamp(instance_id, timestamp);

    WriteCacheRes res;
    res.trace_id = trace_id;
    res.kvcm_write_length = 0;
    res.kvcm_write_hit_length = 0;
    res.kvcm_write_length = write_record.newly_inserted_blocks;
    // write_hit = 请求写入数 - 实际新插入数（即已存在、未被驱逐的 block 数）
    res.kvcm_write_hit_length = write_record.write_blocks - write_record.newly_inserted_blocks;
    res.pool_source_write_keys = write_record.pool_source_write_keys;
    res.evicted_keys = write_record.evicted_keys;
    res.tier_flow_events = write_record.tier_flow_events;
    return res;
}

WriteCacheRes OptimizerManager::FillCachePathWithTtlUs(const std::string &instance_id,
                                                       const std::string &trace_id,
                                                       const int64_t timestamp,
                                                       const std::vector<int64_t> &block_ids,
                                                       const std::vector<size_t> &materialized_indices,
                                                       const int64_t ttl_us) {
    WriteCacheSchemaTrace trace;
    trace.set_instance_id(instance_id);
    trace.set_trace_id(trace_id);
    trace.set_timestamp_ns(timestamp);
    trace.set_keys(block_ids);
    trace.set_ttl_us(ttl_us);
    const auto write_record = optimizer_runner_->HandleFillCachePath(trace, materialized_indices);
    stats_collector_->UpdateTimestamp(instance_id, timestamp);

    WriteCacheRes res;
    res.trace_id = trace_id;
    res.kvcm_write_length = write_record.newly_inserted_blocks;
    res.kvcm_write_hit_length = write_record.write_blocks - write_record.newly_inserted_blocks;
    res.pool_source_write_keys = write_record.pool_source_write_keys;
    res.evicted_keys = write_record.evicted_keys;
    res.tier_flow_events = write_record.tier_flow_events;
    return res;
}

GetCacheLocationRes OptimizerManager::GetCacheLocation(const std::string &instance_id,
                                                       const std::string &trace_id,
                                                       const int64_t timestamp,
                                                       const std::vector<int64_t> &block_ids,
                                                       const BlockMask &block_mask,
                                                       const int64_t input_len,
                                                       bool touch_local_hits,
                                                       bool local_hits_are_reads,
                                                       const std::string &query_type) {
    GetLocationSchemaTrace trace;
    trace.set_instance_id(instance_id);
    trace.set_trace_id(trace_id);
    trace.set_timestamp_ns(timestamp);
    trace.set_keys(block_ids);
    trace.set_input_len(RequirePositiveInputLen("GetCacheLocation", input_len));
    trace.set_query_type(query_type);
    trace.set_block_mask(block_mask);
    const auto read_record = optimizer_runner_->HandleGetLocation(trace, touch_local_hits, local_hits_are_reads);
    stats_collector_->UpdateTimestamp(instance_id, timestamp);

    GetCacheLocationRes res;
    res.trace_id = trace_id;
    res.kvcm_hit_length = 0;
    if (hit_rate_perspective_ == HitRatePerspective::ENGINE_LOCAL) {
        res.kvcm_hit_length = read_record.local_hit_blocks + read_record.remote_hit_blocks;
        res.hit_indices = read_record.local_hit_indices;
        res.hit_indices.insert(
            res.hit_indices.end(), read_record.remote_hit_indices.begin(), read_record.remote_hit_indices.end());
    } else {
        res.kvcm_hit_length = read_record.remote_hit_blocks;
        res.hit_indices = read_record.remote_hit_indices;
    }
    res.evicted_keys = read_record.evicted_keys;
    res.tier_flow_events = read_record.tier_flow_events;
    return res;
}

size_t OptimizerManager::PrefixMatchCount(const std::string &instance_id,
                                          const std::vector<int64_t> &block_ids,
                                          int64_t timestamp) const {
    auto indexer = indexer_manager_ ? indexer_manager_->GetOptIndexer(instance_id) : nullptr;
    if (!indexer) {
        return 0;
    }
    return indexer->PrefixMatchCount(block_ids, timestamp);
}

std::vector<TierFlowKeyEvent> OptimizerManager::TouchCacheKeysAtTier(const std::string &instance_id,
                                                                     const std::vector<int64_t> &block_ids,
                                                                     const std::string &tier_name,
                                                                     int64_t timestamp) {
    auto indexer = indexer_manager_ ? indexer_manager_->GetOptIndexer(instance_id) : nullptr;
    if (!indexer) {
        return {};
    }
    bool refresh_ttl_on_read = true;
    auto it = instance_ttl_refresh_on_read_.find(instance_id);
    if (it != instance_ttl_refresh_on_read_.end()) {
        refresh_ttl_on_read = it->second;
    }
    indexer->TouchKeysAtTier(block_ids, tier_name, timestamp, refresh_ttl_on_read);
    return indexer->ConsumeTierFlow().KeyEvents();
}

std::vector<int64_t> OptimizerManager::PoolSourceWriteTouchKeysAtLeast(const std::string &instance_id,
                                                                       const std::vector<int64_t> &block_ids,
                                                                       size_t threshold,
                                                                       int64_t timestamp) const {
    auto indexer = indexer_manager_ ? indexer_manager_->GetOptIndexer(instance_id) : nullptr;
    if (!indexer) {
        return {};
    }
    return indexer->PoolSourceWriteTouchKeysAtLeast(block_ids, threshold, timestamp);
}

void OptimizerManager::AnalyzeResults() {
    for (const auto &[instance_id, _] : instance_configs_) {
        int64_t final_timestamp = stats_collector_->GetLastTimestamp(instance_id);
        KVCM_LOG_INFO("Finalizing stats for instance %s with timestamp: %ld", instance_id.c_str(), final_timestamp);
        stats_collector_->FinalizeAll(instance_id, final_timestamp);

        // 模板前缀分析：Finalize 之后、Export 之前，从 RadixTree 提取模板
        if (template_prefix_tracker_) {
            auto indexer = indexer_manager_->GetOptIndexer(instance_id);
            if (indexer) {
                template_prefix_tracker_->AnalyzeTemplates(instance_id, indexer->GetRoot());
            }
        }

        stats_collector_->ExportAll(instance_id, config_);
        stats_collector_->ResetAll(instance_id);
    }

    KVCM_LOG_INFO("Analysis complete and memory released (all data persisted to %s)",
                  config_.output_result_path().c_str());
}

std::unordered_map<std::string, RadixTreeIndex::RadixTreeExport> OptimizerManager::ExportRadixTrees() const {
    std::unordered_map<std::string, RadixTreeIndex::RadixTreeExport> export_data;

    if (!indexer_manager_) {
        KVCM_LOG_WARN("Indexer manager not initialized");
        return export_data;
    }

    auto indexers = indexer_manager_->GetAllOptIndexers();
    for (const auto &[instance_id, indexer] : indexers) {
        if (indexer) {
            export_data[instance_id] = indexer->ExportForVisualization();
        }
    }

    KVCM_LOG_INFO("Exported %zu radix trees for visualization", export_data.size());
    return export_data;
}

bool OptimizerManager::ClearCache(const std::string &instance_id) {
    if (!indexer_manager_) {
        KVCM_LOG_ERROR("Indexer manager not initialized");
        return false;
    }
    return indexer_manager_->ClearCache(instance_id);
}

void OptimizerManager::ClearAllCaches() {
    if (!indexer_manager_) {
        KVCM_LOG_ERROR("Indexer manager not initialized");
        return;
    }
    indexer_manager_->ClearAllCaches();
}

bool OptimizerManager::ClearCacheAndResetStats(const std::string &instance_id) {
    if (!ClearCache(instance_id)) {
        return false;
    }
    stats_collector_->ResetAll(instance_id);
    KVCM_LOG_INFO("Reset statistics for instance_id: %s", instance_id.c_str());
    return true;
}

void OptimizerManager::ClearAllCachesAndResetStats() {
    ClearAllCaches();
    for (const auto &[instance_id, _] : instance_configs_) {
        stats_collector_->ResetAll(instance_id);
    }
    KVCM_LOG_INFO("Reset statistics for all %zu instances", instance_configs_.size());
}

} // namespace kv_cache_manager
