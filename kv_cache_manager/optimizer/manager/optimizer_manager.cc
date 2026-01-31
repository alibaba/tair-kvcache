#include "kv_cache_manager/optimizer/manager/optimizer_manager.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/manager/cache_location.h"
#include "kv_cache_manager/optimizer/config/tier_config.h"
#include "kv_cache_manager/optimizer/eviction_policy/policy_factory.h"
#include "kv_cache_manager/optimizer/trace_converter/converter_factory.h"
#include "kv_cache_manager/optimizer/trace_converter/trace_util.h"
namespace kv_cache_manager {

OptimizerManager::OptimizerManager(const OptimizerConfig &config) : config_(config) {}
bool OptimizerManager::Init() {
    eviction_manager_.reset(new OptEvictionManager());
    if (!eviction_manager_->Init(config_.eviction_config())) {
        KVCM_LOG_ERROR("Failed to initialize eviction manager.");
        return false;
    }
    indexer_manager_.reset(new OptIndexerManager(eviction_manager_));

    // 检查配置中是否有实例组
    if (config_.instance_groups().empty()) {
        KVCM_LOG_ERROR("No instance groups found in configuration.");
        return false;
    }

    size_t total_instances = 0;
    size_t failed_instances = 0;
    std::vector<std::string> failed_instance_ids;

    // 解析配置，创建实例组和实例的配置映射
    for (auto &group : config_.mutable_instance_groups()) {
        auto group_name = group.group_name();

        // 检查 group_name 是否重复
        if (instance_group_configs_.find(group_name) != instance_group_configs_.end()) {
            KVCM_LOG_WARN("Duplicate group_name found: %s", group_name.c_str());
            continue;
        }

        auto &storage_configs = group.mutable_storages();

        // 检查 storage_configs 是否为空
        if (storage_configs.empty()) {
            KVCM_LOG_WARN("No storage configs found for group: %s", group_name.c_str());
            continue;
        }

        std::sort(storage_configs.begin(), storage_configs.end(), [](const OptTierConfig &a, const OptTierConfig &b) {
            return a.priority() < b.priority();
        });

        instance_group_configs_[group_name] = group;

        // 检查该 group 是否有实例
        if (group.instances().empty()) {
            KVCM_LOG_WARN("No instances found in group: %s", group_name.c_str());
            continue;
        }

        for (auto &instance : group.instances()) {
            total_instances++;
            auto instance_config = instance;
            auto instance_id = instance_config.instance_id();

            // 检查 instance_id 是否重复
            if (instance_configs_.find(instance_id) != instance_configs_.end()) {
                KVCM_LOG_WARN("Duplicate instance_id found: %s", instance_id.c_str());
                continue;
            }

            instance_config.set_instance_group_name(group_name);

            // 验证 instance_group_name 是否存在
            if (instance_group_configs_.find(group_name) == instance_group_configs_.end()) {
                KVCM_LOG_WARN(
                    "Instance group '%s' not found for instance: %s", group_name.c_str(), instance_id.c_str());
                continue;
            }

            instance_configs_[instance_id] = instance_config;

            // 调用CreateRadixTreeIndex接口创建多个RadixTreeIndex实例
            if (!CreateRadixTreeIndex(instance_config, storage_configs)) {
                KVCM_LOG_ERROR("Failed to create RadixTreeIndex for instance: %s", instance_id.c_str());
                failed_instances++;
                failed_instance_ids.push_back(instance_id);
                // 移除失败的实例配置
                instance_configs_.erase(instance_id);
                continue;
            }

            result_map_[instance_id] = std::make_shared<Result>();
        }
    }

    // 检查是否有实例初始化失败
    if (failed_instances > 0) {
        KVCM_LOG_ERROR("Failed to initialize %zu out of %zu instances", failed_instances, total_instances);
        for (const auto &id : failed_instance_ids) {
            KVCM_LOG_WARN("Failed instance: %s", id.c_str());
        }
        // 如果所有实例都失败，返回 false
        if (failed_instances == total_instances) {
            KVCM_LOG_ERROR("All instances failed to initialize.");
            return false;
        }
        // 如果部分实例失败，记录警告但继续
        KVCM_LOG_WARN("Continuing with %zu successful instances", total_instances - failed_instances);
    }

    // 检查是否至少有一个实例成功初始化
    if (instance_configs_.empty()) {
        KVCM_LOG_ERROR("No instances successfully initialized.");
        return false;
    }

    indexer_manager_->RegisterInstanceGroups(instance_group_configs_);
    indexer_manager_->RegisterInstances(instance_configs_);

    optimizer_runner_.reset(new OptimizerRunner(indexer_manager_, eviction_manager_, result_map_));
    analyzer_.reset(new HitAnalysis());
    return true;
}
bool OptimizerManager::CreateRadixTreeIndex(const OptInstanceConfig &instance_config,
                                            const std::vector<OptTierConfig> &storage_configs) {
    // 创建RadixTreeIndex实例
    // 每个index实例对应一个instance_config，以及包含多层，每层有自己的驱逐策略
    // 目前只对第一层创建驱逐策略实例，后续层级不逐出

    // 再次验证 instance_group_name 是否存在
    auto group_it = instance_group_configs_.find(instance_config.instance_group_name());
    if (group_it == instance_group_configs_.end()) {
        KVCM_LOG_ERROR("Instance group '%s' not found for instance: %s",
                       instance_config.instance_group_name().c_str(),
                       instance_config.instance_id().c_str());
        return false;
    }

    if (!indexer_manager_->CreateOptIndexer(
            instance_config, storage_configs, group_it->second.hierarchical_eviction_enabled())) {
        KVCM_LOG_ERROR("Failed to create optimizer indexer for instance_id: %s", instance_config.instance_id().c_str());
        return false;
    }
    KVCM_LOG_INFO("Created optimizer indexer for instance_id: %s", instance_config.instance_id().c_str());
    return true;
}
void OptimizerManager::DirectRun() { optimizer_runner_->Run(config_); }
WriteCacheRes OptimizerManager::WriteCache(const std::string &instance_id,
                                           const std::string &trace_id,
                                           const int64_t timestamp,
                                           const std::vector<int64_t> &block_ids,
                                           const std::vector<int64_t> &token_ids) {
    WriteCacheSchemaTrace trace;
    trace.set_instance_id(instance_id);
    trace.set_trace_id(trace_id);
    trace.set_timestamp_us(timestamp);
    trace.set_keys(block_ids);
    trace.set_tokens(token_ids);
    optimizer_runner_->HandleWriteCache(trace);
    WriteCacheRes res;
    res.trace_id = trace_id;
    res.kvcm_write_length = 0;
    res.kvcm_write_hit_length = trace.keys().size();

    auto result_it = result_map_.find(instance_id);
    if (result_it != result_map_.end() && result_it->second && !result_it->second->write_results.empty()) {
        auto &last_write = result_it->second->write_results.back();
        res.kvcm_write_length = last_write.write_blocks;
        res.kvcm_write_hit_length -= last_write.write_blocks;
    }
    return res;
}

GetCacheLocationRes OptimizerManager::GetCacheLocation(const std::string &instance_id,
                                                       const std::string &trace_id,
                                                       const int64_t timestamp,
                                                       const std::vector<int64_t> &block_ids,
                                                       const std::vector<int64_t> &token_ids,
                                                       const BlockMask &block_mask) {
    GetLocationSchemaTrace trace;
    trace.set_instance_id(instance_id);
    trace.set_trace_id(trace_id);
    trace.set_timestamp_us(timestamp);
    trace.set_keys(block_ids);
    trace.set_tokens(token_ids);
    trace.set_block_mask(block_mask);
    optimizer_runner_->HandleGetLocation(trace);
    GetCacheLocationRes res;
    res.trace_id = trace_id;
    res.kvcm_hit_length = 0;

    auto result_it = result_map_.find(instance_id);
    if (result_it != result_map_.end() && result_it->second && !result_it->second->read_results.empty()) {
        auto &last_read = result_it->second->read_results.back();
        res.kvcm_hit_length = last_read.external_hit_blocks;
    }
    return res;
}
// TODO 多线程直接在后台进行结果分析，这样可以在启动manager的时候就开始分析并且不需要调用额外接口
void OptimizerManager::AnalyzeResults() { analyzer_->Analyze(result_map_, config_); }

// 导出前缀树用于可视化
std::unordered_map<std::string, RadixTreeIndex::RadixTreeExport> OptimizerManager::ExportRadixTrees() const {
    std::unordered_map<std::string, RadixTreeIndex::RadixTreeExport> export_data;

    if (!indexer_manager_) {
        KVCM_LOG_WARN("Indexer manager not initialized");
        return export_data;
    }

    // 获取所有实例的索引器
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
    // 先清空缓存
    if (!ClearCache(instance_id)) {
        return false;
    }

    // 重置统计结果
    auto result_it = result_map_.find(instance_id);
    if (result_it != result_map_.end()) {
        result_it->second = std::make_shared<Result>();
        KVCM_LOG_INFO("Reset statistics for instance_id: %s", instance_id.c_str());
    }

    return true;
}

void OptimizerManager::ClearAllCachesAndResetStats() {
    // 清空所有缓存
    ClearAllCaches();

    // 重置所有统计结果
    for (auto &[instance_id, result] : result_map_) {
        result = std::make_shared<Result>();
    }

    KVCM_LOG_INFO("Reset statistics for all %zu instances", result_map_.size());
}

} // namespace kv_cache_manager