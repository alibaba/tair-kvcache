#include "kv_cache_manager/optimizer/trace_converter/converter_factory.h"

#include "kv_cache_manager/optimizer/trace_converter/optimizer_schema_converter.h"
#include "kv_cache_manager/optimizer/trace_converter/publisher_log_converter.h"
#include "kv_cache_manager/optimizer/trace_converter/qwen_bailian_converter.h"
namespace kv_cache_manager {
std::unordered_map<std::string, int32_t> ConverterFactory::block_size_map_;

std::shared_ptr<BaseConverter> ConverterFactory::CreateConverter(OptimizerConfig &config) {
    auto type = config.trace_type();
    auto instance_groups = config.instance_groups();
    for (const auto &group : instance_groups) {
        for (const auto &instance : group.instances()) {
            auto instance_id = instance.instance_id();
            auto block_size = instance.block_size();
            block_size_map_[instance_id] = block_size;
        }
    }
    switch (type) {
    case TraceType::TRACE_PUBLISHER_LOG:
        config.set_rw_separation(true); // Publisher Log 有独立的读写事件
        return std::make_shared<PublisherLogConverter>(block_size_map_);
    case TraceType::TRACE_QWEN_BAILIAN:
        config.set_rw_separation(false); // Qwen Bailian 是对话轮次格式
        return std::make_shared<QwenBailianConverter>();
    case TraceType::TRACE_OPTIMIZER_SCHEMA:
        config.set_rw_separation(false); // trace_anonymous_tool 导出的是 DialogTurnSchemaTrace
        return std::make_shared<OptimizerSchemaConverter>();
    case TraceType::TRACE_UNSPECIFIED:
    default:
        return nullptr;
    }
}
} // namespace kv_cache_manager