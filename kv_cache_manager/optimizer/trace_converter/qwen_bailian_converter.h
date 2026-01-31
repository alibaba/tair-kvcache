#pragma once
#include <string>
#include <unordered_map>
#include <vector>

#include "kv_cache_manager/common/jsonizable.h"
#include "kv_cache_manager/optimizer/trace_converter/base_converter.h"
namespace kv_cache_manager {

class QwenBailianConverter : public BaseConverter {
public:
    QwenBailianConverter() = default;
    ~QwenBailianConverter() = default;
    // 从日志文件路径转换为标准Trace
    std::vector<std::shared_ptr<OptimizerSchemaTrace>>
    ConvertLogFileToTraces(const std::string &log_file_path) override;

    // 从单行JSON日志转换为标准Trace
    std::shared_ptr<OptimizerSchemaTrace> ConvertLogLineToTrace(const std::string &log_line) override;
    // 从单行JSON日志转换为指定类型的Trace
    std::shared_ptr<OptimizerSchemaTrace> ConvertLogLineToTraceByType(const std::string &log_line,
                                                                      const std::string &event_type) override;

private:
    // 转换GetCacheLocation事件到GetLocationSchemaTrace
    std::shared_ptr<GetLocationSchemaTrace> ConvertGetCacheLocationEvent(const rapidjson::Document &event_doc);
    std::shared_ptr<WriteCacheSchemaTrace> ConvertWriteCacheEvent(const rapidjson::Document &event_doc);
    std::shared_ptr<DialogTurnSchemaTrace> ConvertDialogTurnEvent(const rapidjson::Document &event_doc);
};

} // namespace kv_cache_manager