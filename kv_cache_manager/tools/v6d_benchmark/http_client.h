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

    // GetCacheLocation API (单副本，旧接口)
    bool GetCacheLocation(const std::string &trace_id,
                          const std::string &instance_id,
                          QueryType query_type,
                          const std::vector<int64_t> &keys,
                          JsonDocument &response);

    // GetBatchCacheLocations API (多副本，V6D 场景推荐)
    bool GetBatchCacheLocations(const std::string &trace_id,
                                const std::string &instance_id,
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

    // 失败诊断信息（每个客户端线程独占一份；只在 PostJson 返回 false 时具有意义）
    struct LastFailureInfo {
        int http_status = 0;         // HTTP 状态码；0 表示未拿到响应（连接失败/超时）
        int httplib_error = 0;       // httplib::Error；status==0 时有意义
        std::string url;             // 请求路径
        std::string request_body;    // 请求 body（已截断，最长由调用方控制）
        std::string response_body;   // 响应 body（已截断；status==0 时为空）
        bool is_parse_error = false; // status==200 但 JSON 解析失败
    };
    const LastFailureInfo &GetLastFailureInfo() const { return last_failure_; }

    // 控制失败诊断 body 的最大长度；<=0 表示不打 body
    void SetFailLogBodyMaxBytes(int n) { fail_log_body_max_bytes_ = n; }

private:
    bool PostJson(const std::string &url, const JsonDocument &request, JsonDocument &response);
    std::string JsonToString(const JsonDocument &doc);
    static std::string Truncate(const std::string &s, int max_bytes);

    std::string base_url_;
    std::string admin_url_;
    httplib::Client client_;
    httplib::Client admin_client_;
    size_t last_response_size_ = 0;
    LastFailureInfo last_failure_;
    int fail_log_body_max_bytes_ = 1024;
};

} // namespace v6d_benchmark
} // namespace kv_cache_manager
