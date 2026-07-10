#pragma once

#include <memory>

#include "kv_cache_manager/common/service_discovery.h"
#include "kv_cache_manager/common/service_discovery_url.h"

namespace kv_cache_manager {

/**
 * 构建环境提供的服务发现扩展钩子。
 *
 * 内置工厂处理 static:// 后调用本函数。扩展支持当前 scheme 时返回已经完成
 * 初始化的实例；不支持或创建失败时返回 nullptr。开源构建提供 no-op 实现。
 */
std::unique_ptr<ServiceDiscovery> CreateServiceDiscoveryExtension(const ServiceDiscoveryUrl &url_info);

} // namespace kv_cache_manager
