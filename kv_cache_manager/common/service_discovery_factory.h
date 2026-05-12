#pragma once

#include <memory>
#include <string>

#include "kv_cache_manager/common/service_discovery.h"

namespace kv_cache_manager {

/**
 * 按 URL 形式的服务发现配置创建并初始化对应的 ServiceDiscovery 实例。
 *
 * 支持的 URL：
 *   - vipserver://<domain>[?timeout=<sec>]
 *       例：vipserver://pace.meta.vipserver?timeout=10
 *   - spectrum://<virtual_service_id>[?cache_time=<sec>&retry_time=<n>&timeout=<ms>]
 *       例：spectrum://v-ad2d143d?cache_time=30&retry_time=3&timeout=5000
 *   - static://<ip:port>[,<ip:port>...]
 *       例：static://11.22.33.44:8080,33.55.66.77:8080
 *   - 空字符串：返回 nullptr，调用方按"不使用服务发现"语义降级（走静态 domain）
 *
 * 行为：
 *   - URL 为空 / 解析失败 / 子类 Init 失败：返回 nullptr 并打印 warning
 *   - URL 中的可选参数缺失时各实现使用默认值
 *
 * 调用方需要把「nullptr」当作合法的"不使用服务发现"语义来处理。
 */
class ServiceDiscoveryFactory {
public:
    static std::unique_ptr<ServiceDiscovery> CreateServiceDiscovery(const std::string &url);
};

} // namespace kv_cache_manager
