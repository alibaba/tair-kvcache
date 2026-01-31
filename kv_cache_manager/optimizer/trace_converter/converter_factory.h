#pragma once
#include <memory>
#include <string>
#include <unordered_map>

#include "kv_cache_manager/optimizer/config/optimizer_config.h"
#include "kv_cache_manager/optimizer/trace_converter/base_converter.h"
namespace kv_cache_manager {
class ConverterFactory {
public:
    static std::shared_ptr<BaseConverter> CreateConverter(OptimizerConfig &config);

private:
    static std::unordered_map<std::string, int32_t> block_size_map_;
};
} // namespace kv_cache_manager