#pragma once

#include <string>
#include <vector>

#include "httplib.h"
#include "kv_cache_manager/client/include/common.h"
#include "kv_cache_manager/common/jsonizable.h"
#include "kv_cache_manager/common/logger.h"

namespace kv_cache_manager {
namespace v6d_benchmark {

// 简单的JSON类型,使用rapidjson
using JsonValue = rapidjson::Value;
using JsonDocument = rapidjson::Document;

class KVCMHttpClient {
public:
    KVCMHttpClient(const std::string &base_url, const std::string &admin_url);

    // ReportEvent API
    bool ReportEvent(const std::string &trace_id,
                     const std::string &instance_id,
                     const std::string &host_ip_port,
                     const std::vector<JsonDocument> &events,
                     JsonDocument &response);

    // GetCacheLocation API
    bool GetCacheLocation(const std::string &trace_id,
                          const std::string &instance_id,
                          QueryType query_type,
                          const std::vector<int64_t> &keys,
                          JsonDocument &response);

    // RegisterInstance API (Admin)
    bool RegisterInstance(const std::string &trace_id,
                          const std::string &instance_group,
                          const std::string &instance_id,
                          int32_t block_size,
                          JsonDocument &response);

    // AddStorage API (Admin)
    bool AddStorage(const std::string &trace_id,
                    const std::string &storage_name,
                    const std::string &cluster_name,
                    JsonDocument &response);

    // 获取最后一次请求的响应体大小
    size_t GetLastResponseSize() const { return last_response_size_; }

private:
    bool PostJson(const std::string &url, const JsonDocument &request, JsonDocument &response);
    std::string JsonToString(const JsonDocument &doc);

    std::string base_url_;
    std::string admin_url_;
    httplib::Client client_;
    httplib::Client admin_client_;
    size_t last_response_size_ = 0;
};

} // namespace v6d_benchmark
} // namespace kv_cache_manager
