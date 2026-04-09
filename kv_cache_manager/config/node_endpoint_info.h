#pragma once

#include <cstdint>
#include <string>

#include "kv_cache_manager/common/jsonizable.h"

namespace kv_cache_manager {

class NodeEndpointInfo : public Jsonizable {
public:
    NodeEndpointInfo() = default;
    NodeEndpointInfo(const std::string &node_id,
                     const std::string &host,
                     int32_t meta_rpc_port,
                     int32_t meta_http_port,
                     int32_t admin_rpc_port,
                     int32_t admin_http_port,
                     const std::string &custom_info = "")
        : node_id_(node_id)
        , host_(host)
        , meta_rpc_port_(meta_rpc_port)
        , meta_http_port_(meta_http_port)
        , admin_rpc_port_(admin_rpc_port)
        , admin_http_port_(admin_http_port)
        , custom_info_(custom_info) {}

    bool FromRapidValue(const rapidjson::Value &rapid_value) override {
        KVCM_JSON_GET_MACRO(rapid_value, "node_id", node_id_);
        KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "host", host_, std::string(""));
        KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "meta_rpc_port", meta_rpc_port_, 0);
        KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "meta_http_port", meta_http_port_, 0);
        KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "admin_rpc_port", admin_rpc_port_, 0);
        KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "admin_http_port", admin_http_port_, 0);
        KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "custom_info", custom_info_, std::string(""));
        return true;
    }

    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override {
        Put(writer, "node_id", node_id_);
        Put(writer, "host", host_);
        Put(writer, "meta_rpc_port", meta_rpc_port_);
        Put(writer, "meta_http_port", meta_http_port_);
        Put(writer, "admin_rpc_port", admin_rpc_port_);
        Put(writer, "admin_http_port", admin_http_port_);
        Put(writer, "custom_info", custom_info_);
    }

    const std::string &node_id() const { return node_id_; }
    const std::string &host() const { return host_; }
    int32_t meta_rpc_port() const { return meta_rpc_port_; }
    int32_t meta_http_port() const { return meta_http_port_; }
    int32_t admin_rpc_port() const { return admin_rpc_port_; }
    int32_t admin_http_port() const { return admin_http_port_; }
    const std::string &custom_info() const { return custom_info_; }

private:
    std::string node_id_;
    std::string host_;
    int32_t meta_rpc_port_{0};
    int32_t meta_http_port_{0};
    int32_t admin_rpc_port_{0};
    int32_t admin_http_port_{0};
    std::string custom_info_;
};

} // namespace kv_cache_manager
