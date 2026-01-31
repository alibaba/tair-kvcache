#include "kv_cache_manager/optimizer/manager/optimizer_loader.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/optimizer/config/optimizer_config.h"
#include "kv_cache_manager/optimizer/trace_converter/converter_factory.h"
#include "kv_cache_manager/optimizer/trace_converter/trace_util.h"

namespace kv_cache_manager {
std::vector<std::shared_ptr<OptimizerSchemaTrace>> OptimizerLoader::LoadTrace(OptimizerConfig &config) {
    auto trace_converter = ConverterFactory::CreateConverter(config);
    auto traces = trace_converter->ConvertLogFileToTraces(config.trace_file_path());
    TraceTimeSorter::SortTracesByTimestamp(traces);
    AddTraceId(traces);
    return traces;
}
// 该函数会导出转换后的DialogTurnSchemaTrace到一个新的文件中，文件名基于输入文件名生成
// 文件中包含多个instance的trace
// 目前用来服务于算力画像
void OptimizerLoader::DumpSchemaTracesToFile(OptimizerConfig &config) {
    auto traces = LoadTrace(config);
    std::filesystem::path input_path(config.trace_file_path());
    std::filesystem::path output_path = input_path.parent_path() / (input_path.stem().string() + "_opt_schema.log");
    std::ofstream file(output_path);
    KVCM_LOG_INFO("Saving converted traces to file: %s", output_path.c_str());
    for (size_t i = 0; i < traces.size(); ++i) {
        if (auto get_trace = std::dynamic_pointer_cast<DialogTurnSchemaTrace>(traces[i])) {
            if (i > 0) {
                file << "\n";
            }
            file << get_trace->ToJsonString();
        }
    }
    file.close();
}
} // namespace kv_cache_manager