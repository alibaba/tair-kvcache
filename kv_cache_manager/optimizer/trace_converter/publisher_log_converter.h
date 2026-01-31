#pragma once
#include <fstream>
#include <list>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "kv_cache_manager/common/jsonizable.h"
#include "kv_cache_manager/optimizer/trace_converter/base_converter.h"
#include "kv_cache_manager/optimizer/trace_converter/trace_util.h"
namespace kv_cache_manager {

class PublisherLogConverter : public BaseConverter {
public:
    PublisherLogConverter(const std::unordered_map<std::string, int32_t> &block_size_map)
        : block_size_map_(block_size_map) {}
    ~PublisherLogConverter() = default;
    // 从日志文件路径转换为标准Trace
    std::vector<std::shared_ptr<OptimizerSchemaTrace>>
    ConvertLogFileToTraces(const std::string &log_file_path) override;

    // 从单行JSON日志转换为标准Trace
    std::shared_ptr<OptimizerSchemaTrace> ConvertLogLineToTrace(const std::string &log_line) override;

    // 从单行JSON日志转换为指定类型的Trace
    std::shared_ptr<OptimizerSchemaTrace> ConvertLogLineToTraceByType(const std::string &log_line,
                                                                      const std::string &event_type) override {
        return nullptr;
    };

private:
    // 从JSON字符串解析事件类型
    bool ParseEventTypeFromJson(const std::string &json_str, std::string &event_type, rapidjson::Document &event_doc);

    // 转换GetCacheLocation事件到GetLocationSchemaTrace
    std::shared_ptr<GetLocationSchemaTrace> ConvertGetLocationEvent(const rapidjson::Document &event_doc);

    // 转换StartWriteCache事件,并匹配对应的GetCacheLocation事件
    std::shared_ptr<DialogTurnSchemaTrace> ConvertStartWriteCacheEvent(const rapidjson::Document &event_doc);

    // 转换FinishWriteCache事件并匹配对应的StartWriteCache事件
    std::shared_ptr<WriteCacheSchemaTrace> ConvertFinishWriteCacheEvent(const rapidjson::Document &event_doc);

    void CheckAndConvertPendingWriteEvent(std::vector<std::shared_ptr<OptimizerSchemaTrace>> &traces);
    std::shared_ptr<DialogTurnSchemaTrace>
    FindMatchingGetLocationTrace(std::shared_ptr<WriteCacheSchemaTrace> write_trace);
    void ConvertPendingGetLocationEvent(std::vector<std::shared_ptr<OptimizerSchemaTrace>> &traces);
    // 用于关联写入事件的会话ID映射
    std::unordered_map<std::string, std::shared_ptr<WriteCacheSchemaTrace>> pending_write_sessions_;

    std::list<std::shared_ptr<GetLocationSchemaTrace>> pending_get_location_traces_;
    std::unordered_map<std::string, int32_t> block_size_map_;
};

} // namespace kv_cache_manager