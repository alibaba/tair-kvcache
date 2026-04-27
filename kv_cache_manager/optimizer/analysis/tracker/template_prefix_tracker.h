#pragma once
#include <string>
#include <unordered_map>
#include <vector>

#include "kv_cache_manager/optimizer/analysis/stats_tracker.h"

namespace kv_cache_manager {

struct RadixTreeNode;

// ============================================================================
// 共享前缀模板（System Prompt）识别与量化
//
// Phase 1: DFS 遍历 RadixTree，嵌套共存提取模板。
//          满足候选条件的节点一律注册。父子模板可以共存——
//          短 system prompt 与其延伸的长 system prompt 都被保留。
//          仅当子树模板的 fan_out 远弱于当前节点（低于
//          fan_dominance_ratio）时回滚子树噪声模板。
// Phase 2: 对每条 trace 的 keys 沿树匹配，归属到经过的最深模板节点。
// Phase 3: 聚合统计 + 输出 per-trace 归属 CSV 和模板汇总 CSV。
//
// 模板内容还原（原始 prompt 文本）由 Python 侧完成。
// ============================================================================

struct TemplateInfo {
    std::string template_id;
    std::vector<int64_t> prefix_keys; // 从根到分叉点的完整 block key 序列
    size_t prefix_depth = 0;          // = prefix_keys.size()
    size_t fan_out = 0;
    size_t node_access_count = 0;
    size_t path_block_hit_sum = 0;
    const RadixTreeNode *template_node = nullptr; // 分叉点节点指针
};

struct TemplateAggregation {
    std::string template_id;
    std::vector<int64_t> prefix_keys;
    size_t prefix_depth = 0;
    size_t fan_out = 0;
    size_t node_access_count = 0;
    size_t path_block_hit_sum = 0;

    size_t matched_trace_count = 0;
    size_t matched_hit_blocks_sum = 0;
    size_t matched_total_blocks_sum = 0;
};

struct TraceTemplateResult {
    std::string trace_id;
    std::string template_id; // "NONE" if unmatched
    size_t total_blocks = 0;
    size_t hit_blocks = 0;
    size_t template_depth = 0;   // 0 if unmatched
    double template_ratio = 0.0; // template_depth / total_blocks
};

class TemplatePrefixTracker : public StatsTracker {
public:
    struct Config {
        size_t fan_out_threshold = 10;
        size_t access_count_threshold = 50;
        size_t min_prefix_depth = 20;
        // 子树最大 fan_out / 当前节点 fan_out < 此比值时，回滚子树模板（视为噪声）
        double fan_dominance_ratio = 0.5;
    };

    TemplatePrefixTracker();
    explicit TemplatePrefixTracker(const Config &config);

    void OnReadComplete(const std::string &instance_id, const ReadRecord &record) override;

    void AnalyzeTemplates(const std::string &instance_id, const RadixTreeNode *root);

    void Export(const std::string &instance_id, const OptimizerConfig &config) override;
    void Reset(const std::string &instance_id) override;

    const std::vector<TemplateAggregation> &GetTemplates(const std::string &instance_id) const;

private:
    struct TraceReadInfo {
        std::string trace_id;
        std::vector<int64_t> keys;
        size_t total_blocks;
        size_t hit_blocks;
    };

    struct InstanceData {
        std::vector<TraceReadInfo> trace_reads;
        std::vector<TemplateInfo> templates;
        std::vector<TemplateAggregation> aggregations;
        std::vector<TraceTemplateResult> trace_results;
        size_t global_hit_blocks = 0;
        size_t global_total_blocks = 0;
    };

    // DFS 逐模板裁剪：返回子树中保留模板的最大 fan_out（0 = 无模板）。
    // 候选节点注册前，逐个检查子树模板：fan < my_fan * ratio 的精准删除，够格的共存保留。
    size_t ExtractTemplates(const RadixTreeNode *node,
                            std::vector<int64_t> &path_keys,
                            size_t path_hit_sum,
                            std::vector<TemplateInfo> &templates) const;

    // 沿 RadixTree 匹配 trace keys，返回经过的最深模板分叉点
    std::string
    MatchTraceToTemplate(const RadixTreeNode *root,
                         const std::vector<int64_t> &keys,
                         const std::unordered_map<const RadixTreeNode *, std::string> &node_to_template) const;

    void ComputeAggregations(InstanceData &data) const;

    Config config_;
    std::unordered_map<std::string, InstanceData> instance_data_;
    static const std::vector<TemplateAggregation> kEmptyAggregations;
};

} // namespace kv_cache_manager
