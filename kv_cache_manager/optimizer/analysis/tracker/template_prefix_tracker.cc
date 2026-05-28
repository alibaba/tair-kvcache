#include "kv_cache_manager/optimizer/analysis/tracker/template_prefix_tracker.h"

#include <algorithm>
#include <filesystem>
#include <fstream>

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/optimizer/config/optimizer_config.h"
#include "kv_cache_manager/optimizer/config/types.h"

namespace kv_cache_manager {

const std::vector<TemplateAggregation> TemplatePrefixTracker::kEmptyAggregations;

TemplatePrefixTracker::TemplatePrefixTracker() : StatsTracker("TemplatePrefixTracker"), config_() {}

TemplatePrefixTracker::TemplatePrefixTracker(const Config &config)
    : StatsTracker("TemplatePrefixTracker"), config_(config) {}

// ============================================================================
// 回放阶段 — 记录 trace_id + keys（通过借用指针 copy）
// ============================================================================

void TemplatePrefixTracker::OnReadComplete(const std::string &instance_id, const ReadRecord &record) {
    auto &data = instance_data_[instance_id];
    size_t total = record.remote_read_blocks + record.local_read_blocks;
    size_t hit = record.remote_hit_blocks + record.local_hit_blocks;

    TraceReadInfo info;
    info.trace_id = record.trace_id;
    info.total_blocks = total;
    info.hit_blocks = hit;
    if (record.keys_ptr) {
        info.keys = *record.keys_ptr;
    }
    data.trace_reads.push_back(std::move(info));
    data.global_hit_blocks += hit;
    data.global_total_blocks += total;
}

// ============================================================================
// Phase 1 — DFS template extraction with relative pruning
//
// Return value: the maximum fan_out among all templates retained in this subtree,
//               including the current node if it qualifies as a template.
//               Returns 0 if no templates are retained in this subtree.
//
// Core strategy:
//   1. Recurse into all children first, collecting templates registered in the subtree.
//   2. If the current node is a candidate template:
//      - Prune each child template X where X.fan_out / my_fan < fan_dominance_ratio
//        (too weak relative to this node → treated as noise).
//      - Preserve child templates where X.fan_out / my_fan >= fan_dominance_ratio
//        (strong enough to have independent value, e.g. a nested system prompt).
//      - Register the current node as a new template.
//
// Cascading effect (intentional): a high-fan node will set a high threshold, causing
// its lower-fan descendants to be pruned. This is by design — when a parent already
// captures the dominant branching pattern, weaker nested patterns are considered noise.
// If absolute preservation of all nested levels is needed, set fan_dominance_ratio = 0.
// ============================================================================

size_t TemplatePrefixTracker::ExtractTemplates(const RadixTreeNode *node,
                                               std::vector<int64_t> &path_keys,
                                               size_t path_hit_sum,
                                               std::vector<TemplateInfo> &templates) const {
    for (const auto &block : node->blocks) {
        if (block) {
            path_keys.push_back(block->key);
            path_hit_sum += block->access_count;
        }
    }

    size_t templates_before = templates.size();

    size_t subtree_max_fan = 0;
    for (const auto &[key, child] : node->children) {
        size_t child_fan = ExtractTemplates(child.get(), path_keys, path_hit_sum, templates);
        subtree_max_fan = std::max(subtree_max_fan, child_fan);
    }

    size_t result_fan = subtree_max_fan;

    bool is_candidate = node->children.size() >= config_.fan_out_threshold &&
                        node->stat.access_count >= config_.access_count_threshold &&
                        path_keys.size() >= config_.min_prefix_depth;

    if (is_candidate) {
        size_t my_fan = node->children.size();
        double threshold = my_fan * config_.fan_dominance_ratio;

        // 逐个评估子树中每个模板：fan 不足 ratio 的删除，够格的保留
        auto it = templates.begin() + static_cast<ptrdiff_t>(templates_before);
        while (it != templates.end()) {
            if (static_cast<double>(it->fan_out) < threshold) {
                it = templates.erase(it);
            } else {
                ++it;
            }
        }

        // 重新计算保留下来的子树 max_fan
        result_fan = 0;
        for (auto jt = templates.begin() + static_cast<ptrdiff_t>(templates_before); jt != templates.end(); ++jt) {
            result_fan = std::max(result_fan, jt->fan_out);
        }

        TemplateInfo tmpl;
        tmpl.template_id = "T" + std::to_string(templates.size());
        tmpl.prefix_keys = path_keys;
        tmpl.prefix_depth = path_keys.size();
        tmpl.fan_out = my_fan;
        tmpl.node_access_count = node->stat.access_count;
        tmpl.path_block_hit_sum = path_hit_sum;
        tmpl.template_node = node;
        templates.push_back(std::move(tmpl));
        result_fan = std::max(result_fan, my_fan);
    }

    for (size_t i = 0; i < node->blocks.size(); ++i) {
        if (!path_keys.empty())
            path_keys.pop_back();
    }

    return result_fan;
}

// ============================================================================
// Phase 2 — 精确归属
//
// 对一条 trace 的 keys 序列沿 RadixTree 向下匹配，
// 记录匹配路径上经过的模板分叉点。返回最深匹配的 template_id。
// ============================================================================

std::string TemplatePrefixTracker::MatchTraceToTemplate(
    const RadixTreeNode *root,
    const std::vector<int64_t> &keys,
    const std::unordered_map<const RadixTreeNode *, std::string> &node_to_template) const {

    if (keys.empty())
        return "NONE";

    std::string matched = "NONE";
    const RadixTreeNode *current = root;
    size_t key_idx = 0;

    while (key_idx < keys.size()) {
        auto child_it = current->children.find(keys[key_idx]);
        if (child_it == current->children.end())
            break;

        const RadixTreeNode *child = child_it->second.get();
        size_t match_len = 0;
        while (match_len < child->blocks.size() && (key_idx + match_len) < keys.size() && child->blocks[match_len] &&
               child->blocks[match_len]->key == keys[key_idx + match_len]) {
            match_len++;
        }

        key_idx += match_len;

        // 经过这个节点——检查它是否是模板分叉点
        auto tmpl_it = node_to_template.find(child);
        if (tmpl_it != node_to_template.end()) {
            matched = tmpl_it->second;
        }

        if (match_len < child->blocks.size())
            break;
        current = child;
    }

    return matched;
}

// ============================================================================
// 聚合 — 基于精确归属结果统计
// ============================================================================

void TemplatePrefixTracker::ComputeAggregations(InstanceData &data) const {
    std::unordered_map<std::string, size_t> tmpl_idx;
    for (size_t i = 0; i < data.templates.size(); ++i) {
        tmpl_idx[data.templates[i].template_id] = i;
    }

    struct AccEntry {
        size_t matched = 0;
        size_t hit_blocks_sum = 0;
        size_t total_blocks_sum = 0;
    };
    std::vector<AccEntry> acc(data.templates.size());

    for (const auto &result : data.trace_results) {
        auto it = tmpl_idx.find(result.template_id);
        if (it == tmpl_idx.end())
            continue;
        auto &entry = acc[it->second];
        entry.matched++;
        entry.hit_blocks_sum += result.hit_blocks;
        entry.total_blocks_sum += result.total_blocks;
    }

    data.aggregations.clear();
    data.aggregations.reserve(data.templates.size());

    for (size_t i = 0; i < data.templates.size(); ++i) {
        const auto &tmpl = data.templates[i];
        const auto &entry = acc[i];
        TemplateAggregation agg;
        agg.template_id = tmpl.template_id;
        agg.prefix_keys = tmpl.prefix_keys;
        agg.prefix_depth = tmpl.prefix_depth;
        agg.fan_out = tmpl.fan_out;
        agg.node_access_count = tmpl.node_access_count;
        agg.path_block_hit_sum = tmpl.path_block_hit_sum;
        agg.matched_trace_count = entry.matched;
        agg.matched_hit_blocks_sum = entry.hit_blocks_sum;
        agg.matched_total_blocks_sum = entry.total_blocks_sum;

        data.aggregations.push_back(std::move(agg));
    }
}

// ============================================================================
// AnalyzeTemplates — Phase 1 + Phase 2 + 聚合
// ============================================================================

void TemplatePrefixTracker::AnalyzeTemplates(const std::string &instance_id, const RadixTreeNode *root) {
    if (!root) {
        KVCM_LOG_WARN("Null root for instance: %s, skip template analysis", instance_id.c_str());
        return;
    }

    auto &data = instance_data_[instance_id];

    // ---- Phase 1: DFS 最深优先提取模板 ----
    data.templates.clear();
    std::vector<int64_t> path_keys;
    for (const auto &[key, child] : root->children) {
        ExtractTemplates(child.get(), path_keys, 0, data.templates);
    }

    KVCM_LOG_INFO("Root has %zu children, extracted %zu templates (per-template-prune, ratio=%.2f)",
                  root->children.size(),
                  data.templates.size(),
                  config_.fan_dominance_ratio);

    // 按 access_count 降序排列，同时重编号
    std::sort(data.templates.begin(), data.templates.end(), [](const auto &a, const auto &b) {
        return a.node_access_count > b.node_access_count;
    });
    for (size_t i = 0; i < data.templates.size(); ++i) {
        data.templates[i].template_id = "T" + std::to_string(i);
    }

    for (size_t i = 0; i < std::min(data.templates.size(), size_t(20)); ++i) {
        KVCM_LOG_INFO("  Template[%s] depth=%zu fan=%zu access=%zu",
                      data.templates[i].template_id.c_str(),
                      data.templates[i].prefix_depth,
                      data.templates[i].fan_out,
                      data.templates[i].node_access_count);
    }

    // ---- 构建 node → template_id 映射 ----
    std::unordered_map<const RadixTreeNode *, std::string> node_to_template;
    for (const auto &tmpl : data.templates) {
        node_to_template[tmpl.template_node] = tmpl.template_id;
    }

    // ---- Phase 2: 精确归属每条 trace ----
    // template_id → prefix_depth 快速查找
    std::unordered_map<std::string, size_t> tmpl_depth;
    for (const auto &tmpl : data.templates) {
        tmpl_depth[tmpl.template_id] = tmpl.prefix_depth;
    }

    data.trace_results.clear();
    data.trace_results.reserve(data.trace_reads.size());

    for (const auto &tr : data.trace_reads) {
        TraceTemplateResult result;
        result.trace_id = tr.trace_id;
        result.total_blocks = tr.total_blocks;
        result.hit_blocks = tr.hit_blocks;

        std::string tid = MatchTraceToTemplate(root, tr.keys, node_to_template);
        result.template_id = tid;

        if (tid != "NONE") {
            result.template_depth = tmpl_depth[tid];
            result.template_ratio =
                tr.total_blocks > 0 ? static_cast<double>(result.template_depth) / tr.total_blocks : 0.0;
        }

        data.trace_results.push_back(std::move(result));
    }

    // 诊断：NONE trace 中有 hit 的统计
    size_t none_count = 0, none_with_hit = 0;
    size_t none_hit_blocks_sum = 0;
    for (const auto &r : data.trace_results) {
        if (r.template_id == "NONE") {
            none_count++;
            if (r.hit_blocks > 0) {
                none_with_hit++;
                none_hit_blocks_sum += r.hit_blocks;
            }
        }
    }
    KVCM_LOG_INFO("Matched %zu traces to templates for instance: %s", data.trace_results.size(), instance_id.c_str());
    KVCM_LOG_INFO(
        "  NONE traces: %zu, with hits: %zu (hit_blocks_sum=%zu)", none_count, none_with_hit, none_hit_blocks_sum);

    // ---- 聚合 ----
    ComputeAggregations(data);
}

// ============================================================================
// Export — 模板汇总 CSV + per-trace 归属 CSV
// ============================================================================

void TemplatePrefixTracker::Export(const std::string &instance_id, const OptimizerConfig &config) {
    auto it = instance_data_.find(instance_id);
    if (it == instance_data_.end()) {
        KVCM_LOG_WARN("No template data for instance: %s", instance_id.c_str());
        return;
    }

    const auto &data = it->second;
    std::string dir = config.output_result_path();
    std::filesystem::create_directories(dir);

    // ---- 模板汇总 CSV ----
    if (!data.aggregations.empty()) {
        std::string summary_path = dir + "/" + instance_id + "_template_prefix_summary.csv";
        std::ofstream summary(summary_path);
        if (summary.is_open()) {
            summary << "TemplateId,PrefixDepth,FanOut,NodeAccessCount,"
                       "PathBlockHitSum,MatchedTraceCount,"
                       "MatchedHitBlocksSum,MatchedTotalBlocksSum,PrefixKeys\n";

            for (const auto &agg : data.aggregations) {
                summary << agg.template_id << "," << agg.prefix_depth << "," << agg.fan_out << ","
                        << agg.node_access_count << "," << agg.path_block_hit_sum << "," << agg.matched_trace_count
                        << "," << agg.matched_hit_blocks_sum << "," << agg.matched_total_blocks_sum << ",\"";
                for (size_t k = 0; k < agg.prefix_keys.size(); ++k) {
                    if (k)
                        summary << ", ";
                    summary << agg.prefix_keys[k];
                }
                summary << "\"\n";
            }
            summary.close();
            KVCM_LOG_INFO("Template prefix summary exported to: %s", summary_path.c_str());
        }
    }

    // ---- per-trace 归属 CSV ----
    if (!data.trace_results.empty()) {
        std::string detail_path = dir + "/" + instance_id + "_template_prefix_traces.csv";
        std::ofstream detail(detail_path);
        if (detail.is_open()) {
            detail << "TraceId,TemplateId,TotalBlocks,HitBlocks,"
                      "TemplateDepth,TemplateRatio\n";
            for (const auto &r : data.trace_results) {
                detail << r.trace_id << "," << r.template_id << "," << r.total_blocks << "," << r.hit_blocks << ","
                       << r.template_depth << "," << r.template_ratio << "\n";
            }
            detail.close();
            KVCM_LOG_INFO("Template prefix traces exported to: %s", detail_path.c_str());
        }
    }

    // ---- 日志汇总 ----
    size_t total_matched = 0;
    size_t matched_hit_sum = 0;
    size_t matched_total_sum = 0;
    for (const auto &agg : data.aggregations) {
        total_matched += agg.matched_trace_count;
        matched_hit_sum += agg.matched_hit_blocks_sum;
        matched_total_sum += agg.matched_total_blocks_sum;
    }

    size_t unmatched_hit = data.global_hit_blocks - matched_hit_sum;
    size_t unmatched_total = data.global_total_blocks - matched_total_sum;
    double matched_hit_rate = matched_total_sum > 0 ? 100.0 * matched_hit_sum / matched_total_sum : 0.0;
    double unmatched_hit_rate = unmatched_total > 0 ? 100.0 * unmatched_hit / unmatched_total : 0.0;
    double global_hit_rate =
        data.global_total_blocks > 0 ? 100.0 * data.global_hit_blocks / data.global_total_blocks : 0.0;

    KVCM_LOG_INFO("=== Template Prefix Analysis for %s ===", instance_id.c_str());
    KVCM_LOG_INFO("  Templates found: %zu", data.aggregations.size());
    KVCM_LOG_INFO("  Total traces: %zu", data.trace_reads.size());
    KVCM_LOG_INFO("  Template-matched traces: %zu (%.1f%%)",
                  total_matched,
                  data.trace_reads.empty() ? 0.0 : 100.0 * total_matched / data.trace_reads.size());
    KVCM_LOG_INFO("  Global: hit_blocks=%zu total_blocks=%zu hit_rate=%.1f%%",
                  data.global_hit_blocks,
                  data.global_total_blocks,
                  global_hit_rate);
    KVCM_LOG_INFO("  Matched: hit_blocks=%zu total_blocks=%zu hit_rate=%.1f%% (contribution=%.1f%%)",
                  matched_hit_sum,
                  matched_total_sum,
                  matched_hit_rate,
                  data.global_hit_blocks > 0 ? 100.0 * matched_hit_sum / data.global_hit_blocks : 0.0);
    KVCM_LOG_INFO("  Unmatched: hit_blocks=%zu total_blocks=%zu hit_rate=%.1f%%",
                  unmatched_hit,
                  unmatched_total,
                  unmatched_hit_rate);

    for (const auto &agg : data.aggregations) {
        double tmpl_hit_rate =
            agg.matched_total_blocks_sum > 0 ? 100.0 * agg.matched_hit_blocks_sum / agg.matched_total_blocks_sum : 0.0;
        double contribution =
            data.global_hit_blocks > 0 ? 100.0 * agg.matched_hit_blocks_sum / data.global_hit_blocks : 0.0;
        KVCM_LOG_INFO("  [%s] depth=%zu fan=%zu traces=%zu "
                      "hit_rate=%.1f%% contribution=%.1f%%",
                      agg.template_id.c_str(),
                      agg.prefix_depth,
                      agg.fan_out,
                      agg.matched_trace_count,
                      tmpl_hit_rate,
                      contribution);
    }
}

void TemplatePrefixTracker::Reset(const std::string &instance_id) {
    auto it = instance_data_.find(instance_id);
    if (it != instance_data_.end()) {
        it->second = InstanceData{};
    }
}

const std::vector<TemplateAggregation> &TemplatePrefixTracker::GetTemplates(const std::string &instance_id) const {
    auto it = instance_data_.find(instance_id);
    if (it == instance_data_.end()) {
        return kEmptyAggregations;
    }
    return it->second.aggregations;
}

} // namespace kv_cache_manager
