#pragma once

#include <memory>
#include <string>

#include "kv_cache_manager/common/service_discovery.h"

namespace kv_cache_manager {

/**
 * 按 URL 形式的服务发现配置创建并初始化对应的 ServiceDiscovery 实例。
 *
 * 内置支持的 URL：
 *   - static://<ip:port>[,<ip:port>...]
 *       例：static://11.22.33.44:8080,33.55.66.77:8080
 *   - 空字符串：返回 nullptr，调用方按"不使用服务发现"语义降级（走静态 domain）
 *
 * 其他 provider 由构建环境通过 provider-neutral 的 extension hook 注入，
 * 公开工厂不依赖任何特定服务发现协议或 SDK。
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
