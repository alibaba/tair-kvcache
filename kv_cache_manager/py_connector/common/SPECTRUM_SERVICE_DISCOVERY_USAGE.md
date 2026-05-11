# Spectrum 服务发现组件使用示例

## 概述

Spectrum 服务发现组件用于通过本地 Spectrum 网关，按 `virtual_service_id`
获取服务实例列表，提供 Python 和 C++ 两个版本的实现。该组件独立于现有的
VIPServer 实现，两者可以并存使用。

> **推荐使用方式**：业务侧通过 URL 工厂统一调用，把"用哪种发现机制"完全
> 委托给配置层。详见 [SERVICE_DISCOVERY_DESIGN.md](../../../../internal_source/kv_cache_manager/common/SERVICE_DISCOVERY_DESIGN.md)
> ，本文剩余部分主要给需要直接使用 Spectrum 子类的场景作参考。
>
> ```cpp
> auto discovery = kv_cache_manager::CreateServiceDiscovery(
>     "spectrum://v-ad2d143d?cache_time=30&retry_time=3&timeout=5000");
> ```
>
> ```python
> from kv_cache_manager.py_connector.common.service_discovery_factory import create_service_discovery
> discovery = create_service_discovery(
>     "spectrum://v-ad2d143d?cache_time=30&retry_time=3&timeout=5000")
> ```

## 特性

- 调用方仅需传入 `virtual_service_id`，URL 由组件内部固定拼接
- 本地缓存机制（默认 TTL 30 秒）
- 线程安全
- 支持自动刷新和手动刷新
- 简单的负载均衡（随机选择）

## Spectrum API 接口

固定接口：

```
GET http://127.0.0.1:8880/api/v1/discovery/virtual-services/{id}/instances
```

响应格式：

```json
{
  "virtual_service_id": "v-ad2d143d",
  "instances": [
    {
      "ip": "172.1.2.10",
      "port": 8080,
      "name": "ds-abdedesd-ad2d-sded",
      "physical_service_id": "abdedesd"
    }
  ]
}
```

## Python 版本使用示例

### 基本使用

```python
from kv_cache_manager.py_connector.common.spectrum_service_discovery import (
    SpectrumServiceDiscovery,
)

discovery = SpectrumServiceDiscovery(
    "v-ad2d143d",
    cache_ttl=30,
    refresh_timeout=5,
    auto_refresh=True,
)

endpoints = discovery.get_all_endpoints()
for ep in endpoints:
    print(f"Endpoint: {ep.host}, weight={ep.weight}")
    # Endpoint: 172.1.2.10:8080, weight=100

one_endpoint = discovery.get_one_endpoint()
if one_endpoint:
    print(f"Selected: {one_endpoint.host}")

discovery.refresh()
discovery.close()
```

### 使用 Context Manager

```python
from kv_cache_manager.py_connector.common.spectrum_service_discovery import (
    SpectrumServiceDiscovery,
)

with SpectrumServiceDiscovery("v-ad2d143d") as discovery:
    endpoints = discovery.get_all_endpoints()
    for ep in endpoints:
        print(f"Endpoint: {ep.host}")
```

### 在 V6D 中发现 KVCM

```python
from kv_cache_manager.py_connector.common.spectrum_service_discovery import (
    SpectrumServiceDiscovery,
)

kvcm_discovery = SpectrumServiceDiscovery("v-ad2d143d")
try:
    endpoints = kvcm_discovery.get_all_endpoints()
    if endpoints:
        primary = endpoints[0]
        print(f"Connecting to KVCM: {primary.host}")
    else:
        print("No KVCM endpoints available")
finally:
    kvcm_discovery.close()
```

## C++ 版本使用示例

### 基本使用

```cpp
#include "kv_cache_manager/common/spectrum_service_discovery.h"

using namespace kv_cache_manager;

SpectrumServiceDiscovery discovery;

if (!discovery.Init("v-ad2d143d")) {
    KVCM_LOG_ERROR("Failed to init Spectrum service discovery");
    return;
}

std::vector<ServiceEndpoint> endpoints;
if (discovery.GetAllEndpoints(endpoints)) {
    for (const auto& ep : endpoints) {
        KVCM_LOG_INFO("Endpoint: %s, weight=%d",
                      ep.host.c_str(), ep.weight);
    }
}

ServiceEndpoint one_endpoint;
if (discovery.GetOneEndpoint(one_endpoint)) {
    KVCM_LOG_INFO("Selected: %s", one_endpoint.host.c_str());
}

discovery.Refresh();
```

### KVCM 发现 PACE Meta

```cpp
#include "kv_cache_manager/common/spectrum_service_discovery.h"

using namespace kv_cache_manager;

class KvcmService {
public:
    bool Init(const std::string& meta_vsid) {
        if (!pace_meta_discovery_.Init(meta_vsid)) {
            KVCM_LOG_ERROR("Failed to discover PACE Meta");
            return false;
        }
        return true;
    }

    bool ConnectToPaceMeta() {
        ServiceEndpoint endpoint;
        if (pace_meta_discovery_.GetOneEndpoint(endpoint)) {
            KVCM_LOG_INFO("Connecting to PACE Meta: %s",
                          endpoint.host.c_str());
            return true;
        }
        return false;
    }

private:
    SpectrumServiceDiscovery pace_meta_discovery_;
};
```

## 配置参数

### Python 版本

```python
SpectrumServiceDiscovery(
    virtual_service_id="v-ad2d143d",  # 虚拟服务 ID（必填）
    cache_ttl=30,                     # 缓存有效期（秒），默认 30 秒
    refresh_timeout=5,                # 请求超时（秒），默认 5 秒
    auto_refresh=True,                # 是否自动刷新，默认 True
    retry_count=0,                    # 单次刷新内的额外重试次数（不含首次），默认 0
)
```

### C++ 版本

```cpp
// 网关地址 / 端口由头文件常量给出，超时与重试支持运行时调整
static constexpr const char* SPECTRUM_HOST = "127.0.0.1";
static constexpr int SPECTRUM_PORT = 8880;

SpectrumServiceDiscovery discovery;
discovery.SetCacheTtlSeconds(60);   // 调整本地缓存 TTL（秒）
discovery.SetRequestTimeoutMs(3000);// 调整 HTTP 请求超时（毫秒）
discovery.SetRetryCount(2);         // 单次刷新内的额外重试次数（不含首次）
discovery.Init("v-ad2d143d");
```

> URL 工厂会自动解析对应的 query 参数并调用以上 setter，无需手工调用。

## 与 VIPServer 的关系

- **VIPServer**：基于域名的服务发现，继续使用现有实现
- **Spectrum**：基于 `virtual_service_id` 的服务发现，通过本地网关 HTTP 接口拉取
- 两者完全独立，可以并存使用

## 架构设计

```
业务组件
    |
    +-> SpectrumServiceDiscovery (缓存层)
            |
            +-> HTTP GET 127.0.0.1:8880/api/v1/discovery/virtual-services/{id}/instances
                    |
                    +-> Spectrum 本地网关
                            |
                            +-> 返回实例列表 (JSON)
```

## 缓存策略

1. **首次初始化**：立即从 Spectrum 网关拉取数据
2. **缓存有效期内**：直接返回缓存数据
3. **缓存过期后**：
   - `auto_refresh=True`：自动刷新并返回新数据
   - `auto_refresh=False`：返回旧数据，需手动调用 `refresh()`
4. **刷新失败**：保留旧缓存数据

## 线程安全

- Python 版本：使用 `threading.Lock` 保护缓存访问
- C++ 版本：使用 `std::mutex` 保护缓存访问

## 错误处理

### Python 版本

```python
discovery = SpectrumServiceDiscovery("v-ad2d143d")
try:
    endpoints = discovery.get_all_endpoints()
    if not endpoints:
        print("No endpoints available")
finally:
    discovery.close()
```

### C++ 版本

```cpp
SpectrumServiceDiscovery discovery;
if (!discovery.Init("v-ad2d143d")) {
    KVCM_LOG_ERROR("Init failed");
    return;
}

std::vector<ServiceEndpoint> endpoints;
if (!discovery.GetAllEndpoints(endpoints)) {
    KVCM_LOG_ERROR("GetAllEndpoints failed");
    return;
}
```

## 单元测试

### 运行 Python 测试

```bash
cd github-opensource
PYTHONPATH=$(pwd) python3 -m pytest \
    kv_cache_manager/py_connector/test/test_spectrum_service_discovery.py -v
```

### 测试覆盖

- 初始化和刷新（包含 URL 拼接校验）
- 缓存 TTL 机制
- 获取单个/所有端点
- 空实例列表处理
- HTTP 错误处理
- 缺少 `instances` 字段的响应
- Context Manager 支持
- 手动刷新
- 空 `virtual_service_id` 校验

## 文件位置

- C++ 头文件：`internal_source/kv_cache_manager/common/spectrum_service_discovery.h`
- C++ 实现：`internal_source/kv_cache_manager/common/spectrum_service_discovery.cc`
- C++ 构建：`internal_source/kv_cache_manager/common/BUILD`
- Python 实现：`github-opensource/kv_cache_manager/py_connector/common/spectrum_service_discovery.py`
- Python 测试：`github-opensource/kv_cache_manager/py_connector/test/test_spectrum_service_discovery.py`

## 依赖

### C++ 版本

- `@httplib`：HTTP 客户端库
- `@jsoncpp_git//:jsoncpp`：JSON 解析库
- `//kv_cache_manager/common:logger`：日志库

### Python 版本

- `requests`：HTTP 客户端库
- `threading`：线程安全支持
- `dataclasses`：数据类支持

## 注意事项

1. **virtual_service_id**：必须非空，由调用方从配置或上层服务获取
2. **网关地址**：固定为 `http://127.0.0.1:8880`，由本机 Spectrum sidecar 暴露
3. **缓存策略**：根据业务需求调整 `cache_ttl`，平衡一致性和性能
4. **错误恢复**：刷新失败时会保留旧缓存，确保服务可用性
5. **资源释放**：使用完毕后务必调用 `close()` 或使用 context manager
6. **负载均衡**：当前使用简单随机选择，后续可扩展为轮询、加权等策略
