#include "kv_cache_manager/tools/v6d_benchmark/http_client.h"

#include <sstream>

namespace kv_cache_manager {
namespace v6d_benchmark {

KVCMHttpClient::KVCMHttpClient(const std::string &base_url, const std::string &admin_url)
    : base_url_(base_url), admin_url_(admin_url), client_(base_url), admin_client_(admin_url) {
    client_.set_connection_timeout(5, 0); // 5秒连接超时
    client_.set_read_timeout(10, 0);      // 10秒读取超时
    client_.set_keep_alive(true);

    admin_client_.set_connection_timeout(5, 0);
    admin_client_.set_read_timeout(10, 0);
    admin_client_.set_keep_alive(true);
}

std::string KVCMHttpClient::JsonToString(const JsonDocument &doc) {
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    doc.Accept(writer);
    return std::string(buffer.GetString(), buffer.GetSize());
}

bool KVCMHttpClient::PostJson(const std::string &url, const JsonDocument &request, JsonDocument &response) {
    std::string body = JsonToString(request);

    // 判断使用哪个client: addStorage和registerInstance用admin, 其余用base
    bool use_admin = (url.find("/api/addStorage") == 0 || url.find("/api/registerInstance") == 0);
    auto &http_client = use_admin ? admin_client_ : client_;

    KVCM_LOG_DEBUG("HTTP POST %s (admin=%d, body_size=%zu)", url.c_str(), use_admin, body.size());

    auto res = http_client.Post(url.c_str(), body, "application/json");

    if (!res) {
        KVCM_LOG_ERROR("HTTP request failed: %s error_code=%d", url.c_str(), static_cast<int>(res.error()));
        return false;
    }

    if (res->status != 200) {
        KVCM_LOG_ERROR("HTTP request failed with status %d: %s", res->status, res->body.c_str());
        return false;
    }

    last_response_size_ = res->body.size();

    response.Parse(res->body.c_str());
    if (response.HasParseError()) {
        KVCM_LOG_ERROR("Failed to parse JSON response: %s", res->body.c_str());
        return false;
    }

    return true;
}

bool KVCMHttpClient::ReportEvent(const std::string &trace_id,
                                 const std::string &instance_id,
                                 const std::string &host_ip_port,
                                 const std::vector<JsonDocument> &events,
                                 JsonDocument &response) {
    JsonDocument request;
    request.SetObject();
    auto &allocator = request.GetAllocator();

    request.AddMember("trace_id", rapidjson::Value(trace_id.c_str(), allocator), allocator);
    request.AddMember("instance_id", rapidjson::Value(instance_id.c_str(), allocator), allocator);
    request.AddMember("host_ip_port", rapidjson::Value(host_ip_port.c_str(), allocator), allocator);

    rapidjson::Value events_array(rapidjson::kArrayType);
    for (const auto &event : events) {
        events_array.PushBack(rapidjson::Value(event, allocator), allocator);
    }
    request.AddMember("events", events_array, allocator);

    return PostJson("/api/reportEvent", request, response);
}

bool KVCMHttpClient::GetCacheLocation(const std::string &trace_id,
                                      const std::string &instance_id,
                                      QueryType query_type,
                                      const std::vector<int64_t> &keys,
                                      JsonDocument &response) {
    JsonDocument request;
    request.SetObject();
    auto &allocator = request.GetAllocator();

    request.AddMember("trace_id", rapidjson::Value(trace_id.c_str(), allocator), allocator);
    request.AddMember("instance_id", rapidjson::Value(instance_id.c_str(), allocator), allocator);

    // QueryType
    std::string query_type_str;
    switch (query_type) {
    case QueryType::QT_BATCH_GET:
        query_type_str = "QT_BATCH_GET";
        break;
    case QueryType::QT_PREFIX_MATCH:
        query_type_str = "QT_PREFIX_MATCH";
        break;
    default:
        query_type_str = "QT_BATCH_GET";
        break;
    }
    request.AddMember("query_type", rapidjson::Value(query_type_str.c_str(), allocator), allocator);

    // Keys (应该是block_keys)
    rapidjson::Value keys_array(rapidjson::kArrayType);
    for (int64_t key : keys) {
        keys_array.PushBack(rapidjson::Value(std::to_string(key).c_str(), allocator), allocator);
    }
    request.AddMember("block_keys", keys_array, allocator);

    // Block mask (必需字段)
    rapidjson::Value block_mask(rapidjson::kObjectType);
    block_mask.AddMember("offset", 0, allocator);
    request.AddMember("block_mask", block_mask, allocator);

    return PostJson("/api/getCacheLocation", request, response);
}

bool KVCMHttpClient::RegisterInstance(const std::string &trace_id,
                                      const std::string &instance_group,
                                      const std::string &instance_id,
                                      int32_t block_size,
                                      JsonDocument &response) {
    JsonDocument request;
    request.SetObject();
    auto &allocator = request.GetAllocator();

    request.AddMember("trace_id", rapidjson::Value(trace_id.c_str(), allocator), allocator);
    request.AddMember("instance_group", rapidjson::Value(instance_group.c_str(), allocator), allocator);
    request.AddMember("instance_id", rapidjson::Value(instance_id.c_str(), allocator), allocator);
    request.AddMember("block_size", block_size, allocator);

    // Model deployment (默认配置)
    rapidjson::Value model_deployment(rapidjson::kObjectType);
    model_deployment.AddMember("model_name", rapidjson::Value("benchmark_model", allocator), allocator);
    model_deployment.AddMember("dtype", rapidjson::Value("FP8", allocator), allocator);
    model_deployment.AddMember("use_mla", false, allocator);
    model_deployment.AddMember("tp_size", 1, allocator);
    model_deployment.AddMember("dp_size", 1, allocator);
    model_deployment.AddMember("pp_size", 1, allocator);
    request.AddMember("model_deployment", model_deployment, allocator);

    // Location spec infos (应该是数组格式)
    rapidjson::Value location_specs(rapidjson::kArrayType);
    rapidjson::Value spec1(rapidjson::kObjectType);
    spec1.AddMember("name", rapidjson::Value("tp0", allocator), allocator);
    spec1.AddMember("size", 1024, allocator);
    location_specs.PushBack(spec1, allocator);
    request.AddMember("location_spec_infos", location_specs, allocator);

    return PostJson("/api/registerInstance", request, response);
}

bool KVCMHttpClient::AddStorage(const std::string &trace_id,
                                const std::string &storage_name,
                                const std::string &cluster_name,
                                JsonDocument &response) {
    JsonDocument request;
    request.SetObject();
    auto &allocator = request.GetAllocator();

    request.AddMember("trace_id", rapidjson::Value(trace_id.c_str(), allocator), allocator);

    // Storage config
    rapidjson::Value storage(rapidjson::kObjectType);
    storage.AddMember("global_unique_name", rapidjson::Value(storage_name.c_str(), allocator), allocator);

    // Vineyard config
    rapidjson::Value vineyard(rapidjson::kObjectType);
    vineyard.AddMember("cluster_name", rapidjson::Value(cluster_name.c_str(), allocator), allocator);
    storage.AddMember("vineyard", vineyard, allocator);

    storage.AddMember("check_storage_available_when_open", false, allocator);
    request.AddMember("storage", storage, allocator);

    return PostJson("/api/addStorage", request, response);
}

} // namespace v6d_benchmark
} // namespace kv_cache_manager
