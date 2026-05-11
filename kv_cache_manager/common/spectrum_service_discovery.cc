#include "kv_cache_manager/common/spectrum_service_discovery.h"

#include "json/json.h"
#include <chrono>
#include <sstream>

#include "httplib.h"
#include "kv_cache_manager/common/logger.h"

namespace kv_cache_manager {

namespace {

constexpr const char *kSpectrumPathPrefix = "/api/v1/discovery/virtual-services/";
constexpr const char *kSpectrumPathSuffix = "/instances";

} // namespace

SpectrumServiceDiscovery::SpectrumServiceDiscovery() : CachedServiceDiscovery(30) { // 默认缓存30秒
}

SpectrumServiceDiscovery::~SpectrumServiceDiscovery() {}

bool SpectrumServiceDiscovery::Init(const std::string &virtual_service_id) {
    if (virtual_service_id.empty()) {
        KVCM_LOG_ERROR("Spectrum virtual_service_id is empty");
        return false;
    }
    virtual_service_id_ = virtual_service_id;

    // 首次加载缓存
    return Refresh();
}

bool SpectrumServiceDiscovery::FetchEndpoints(std::vector<ServiceEndpoint> &endpoints) {
    endpoints.clear();

    if (virtual_service_id_.empty()) {
        KVCM_LOG_ERROR("Spectrum virtual_service_id is empty");
        return false;
    }

    // retry_count_ 表示额外重试次数；总尝试次数 = 1 + retry_count_
    const int total_attempts = retry_count_ < 0 ? 1 : retry_count_ + 1;
    for (int attempt = 0; attempt < total_attempts; ++attempt) {
        if (DoFetchOnce(endpoints)) {
            return true;
        }
        endpoints.clear();
    }
    return false;
}

bool SpectrumServiceDiscovery::DoFetchOnce(std::vector<ServiceEndpoint> &endpoints) {
    const std::string path = std::string(kSpectrumPathPrefix) + virtual_service_id_ + kSpectrumPathSuffix;

    // httplib 的 set_*_timeout(sec, usec) 第二个参数是微秒；
    // 这里用 std::chrono 重载避免单位错误。
    httplib::Client client(GetSpectrumHost().c_str(), GetSpectrumPort());
    client.set_connection_timeout(std::chrono::milliseconds(GetRequestTimeoutMs()));
    client.set_read_timeout(std::chrono::milliseconds(GetRequestTimeoutMs()));

    auto response = client.Get(path.c_str());

    if (!response) {
        KVCM_LOG_ERROR("Spectrum request failed: %s, error: %d", path.c_str(), static_cast<int>(response.error()));
        return false;
    }

    if (response->status != 200) {
        KVCM_LOG_ERROR("Spectrum request failed: %s, status: %d", path.c_str(), response->status);
        return false;
    }

    // 解析 JSON 响应。
    // 期望格式：
    //   {
    //     "virtual_service_id": "v-ad2d143d",
    //     "instances": [
    //       {"ip": "172.1.2.10", "port": 8080,
    //        "name": "ds-abdedesd-ad2d-sded",
    //        "physical_service_id": "abdedesd"}
    //     ]
    //   }
    Json::Value root;
    Json::CharReaderBuilder reader_builder;
    std::string errs;
    std::istringstream stream(response->body);

    if (!Json::parseFromStream(reader_builder, stream, &root, &errs)) {
        KVCM_LOG_ERROR("Failed to parse Spectrum JSON response: %s", errs.c_str());
        return false;
    }

    if (!root.isMember("instances")) {
        KVCM_LOG_ERROR("Spectrum response missing instances field, vsid: %s", virtual_service_id_.c_str());
        return false;
    }

    const Json::Value &items = root["instances"];
    if (!items.isArray()) {
        KVCM_LOG_ERROR("Spectrum response instances is not an array");
        return false;
    }

    for (const auto &item : items) {
        if (!item.isMember("ip") || !item.isMember("port")) {
            KVCM_LOG_WARN("Spectrum instance missing ip or port, skipping");
            continue;
        }

        ServiceEndpoint ep;
        ep.ip = item["ip"].asString();
        ep.port = item["port"].asInt();
        ep.host = ep.ip + ":" + std::to_string(ep.port);
        ep.weight = item.isMember("weight") ? item["weight"].asInt() : 100;
        ep.healthy = true;
        endpoints.push_back(ep);
    }

    if (endpoints.empty()) {
        KVCM_LOG_WARN("No valid endpoints found in Spectrum response, vsid: %s", virtual_service_id_.c_str());
        return false;
    }

    KVCM_LOG_INFO("Fetched %zu endpoints from Spectrum, vsid: %s", endpoints.size(), virtual_service_id_.c_str());
    return true;
}

} // namespace kv_cache_manager
