#include "kv_cache_manager/optimizer/trace_loader/standard_trace_loader.h"

#include <fstream>
#include <stdexcept>

#include "kv_cache_manager/common/jsonizable.h"
#include "kv_cache_manager/common/logger.h"

namespace kv_cache_manager {

std::vector<std::shared_ptr<OptimizerSchemaTrace>>
StandardTraceLoader::LoadFromFile(const std::string &trace_file_path) {
    std::vector<std::shared_ptr<OptimizerSchemaTrace>> traces;
    std::ifstream file(trace_file_path);

    if (!file.is_open()) {
        KVCM_LOG_ERROR("Failed to open trace file: %s", trace_file_path.c_str());
        throw std::runtime_error("Failed to open trace file: " + trace_file_path);
    }

    std::string line;
    int64_t line_number = 0;
    auto fail = [&](const std::string &message) {
        std::string full_message = trace_file_path + ":" + std::to_string(line_number) + ": " + message;
        KVCM_LOG_ERROR("%s", full_message.c_str());
        throw std::runtime_error(full_message);
    };
    while (std::getline(file, line)) {
        line_number++;

        if (line.find_first_not_of(" \t\r\n") == std::string::npos) {
            continue;
        }

        rapidjson::Document doc;
        if (doc.Parse(line.c_str()).HasParseError() || !doc.IsObject()) {
            std::string line_preview = line.length() > 100 ? line.substr(0, 100) + "..." : line;
            fail("failed to parse JSON: " + line_preview);
        }

        bool has_type_field = doc.HasMember("type") && doc["type"].IsString();
        std::string type_str = has_type_field ? doc["type"].GetString() : "";
        std::shared_ptr<OptimizerSchemaTrace> trace = nullptr;

        if (!has_type_field) {
            fail("missing string field: type");
        }
        if (!doc.HasMember("instance_id") || !doc["instance_id"].IsString() ||
            std::string(doc["instance_id"].GetString()).empty()) {
            fail("missing non-empty string field: instance_id");
        }
        if (!doc.HasMember("timestamp_ns") || !(doc["timestamp_ns"].IsInt64() || doc["timestamp_ns"].IsUint64())) {
            fail("missing integer field: timestamp_ns");
        }
        if (!doc.HasMember("keys") || !doc["keys"].IsArray()) {
            fail("missing array field: keys");
        }
        if (type_str == "get" || type_str == "request") {
            if (!doc.HasMember("input_len")) {
                fail("missing integer field: input_len");
            }
            if (!(doc["input_len"].IsInt64() || doc["input_len"].IsUint64())) {
                fail("non-integer field: input_len");
            }
            if (type_str == "request") {
                auto request_trace = std::make_shared<RequestSchemaTrace>();
                if (!request_trace->FromJsonString(line)) {
                    fail("failed to parse request trace");
                }
                trace = request_trace;
            } else {
                auto get_trace = std::make_shared<GetLocationSchemaTrace>();
                if (!get_trace->FromJsonString(line)) {
                    fail("failed to parse get trace");
                }
                trace = get_trace;
            }
        } else if (type_str == "write") {
            auto write_trace = std::make_shared<WriteCacheSchemaTrace>();
            if (!write_trace->FromJsonString(line)) {
                fail("failed to parse write trace");
            }
            trace = write_trace;
        } else {
            fail("unknown trace type: " + type_str);
        }

        if (trace && ValidateTrace(*trace)) {
            traces.push_back(trace);
        } else {
            std::string line_preview = line.length() > 100 ? line.substr(0, 100) + "..." : line;
            fail("failed to validate trace: " + line_preview);
        }
    }

    file.close();
    if (traces.empty()) {
        KVCM_LOG_ERROR("No optimizer traces loaded from file: %s", trace_file_path.c_str());
        throw std::runtime_error("No optimizer traces loaded from file: " + trace_file_path);
    }
    KVCM_LOG_INFO("Loaded %zu traces from file: %s", traces.size(), trace_file_path.c_str());
    return traces;
}

bool StandardTraceLoader::ValidateTrace(const OptimizerSchemaTrace &trace) {
    if (trace.instance_id().empty()) {
        KVCM_LOG_ERROR("Validation failed: empty instance_id");
        return false;
    }
    if (trace.timestamp_ns() <= 0) {
        KVCM_LOG_ERROR("Validation failed: invalid timestamp_ns");
        return false;
    }
    if (dynamic_cast<const GetLocationSchemaTrace *>(&trace) != nullptr && trace.input_len() <= 0) {
        KVCM_LOG_ERROR("Validation failed: invalid input_len");
        return false;
    }
    return true;
}

} // namespace kv_cache_manager
