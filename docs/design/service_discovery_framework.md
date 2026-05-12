# KVCacheManager 通用服务发现框架详解

## 目录
- [概述](#概述)
- [架构设计](#架构设计)
- [核心组件](#核心组件)
- [与原有实现的区别](#与原有实现的区别)
- [支持的协议类型](#支持的协议类型)
- [使用示例](#使用示例)
- [技术亮点](#技术亮点)

---

## 概述

KVCacheManager 的通用服务发现框架是一个**统一的、可扩展的服务端点发现抽象层**，旨在为不同的服务发现机制（VIPServer、Spectrum、静态配置等）提供一致的接口。

### 核心目标
1. **统一接口**：无论底层使用哪种服务发现机制，业务代码只需面对统一的 `ServiceDiscovery` 抽象
2. **可扩展性**：新增服务发现类型只需实现接口并注册到工厂，业务代码零改动
3. **双语言支持**：同时提供 C++ 和 Python 实现，配置格式完全一致
4. **配置简化**：通过 URL 格式统一配置，支持多种协议和参数

---

## 架构设计

### 整体架构图

```
┌──────────────────────────────────────────────────────┐
│              业务代码 (Client/Storage)                 │
│         只依赖 ServiceDiscovery 抽象接口               │
└────────────────────┬─────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────┐
│          ServiceDiscoveryFactory                      │
│     解析 URL → 创建对应的 ServiceDiscovery 实例        │
└────────────────────┬─────────────────────────────────┘
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
┌──────────┐  ┌──────────┐  ┌──────────┐
│VIPServer │  │ Spectrum │  │  Static  │
│Subscriber│  │ Discovery│  │ Discovery│
└──────────┘  └──────────┘  └──────────┘
        │            │            │
        ▼            ▼            ▼
   VIPServer     HTTP API     静态列表
    SDK         拉取端点
```

### 设计模式

采用了**工厂模式 + 策略模式**的组合：
- **工厂模式**：`ServiceDiscoveryFactory` 根据 URL scheme 创建不同的服务发现实例
- **策略模式**：不同的 `ServiceDiscovery` 实现类封装不同的服务发现策略
- **模板方法模式**：`CachedServiceDiscovery` 提供缓存框架，子类只需实现数据拉取逻辑

---

## 核心组件

### 1. 统一数据模型：ServiceEndpoint

所有服务发现实现都使用统一的服务端点结构：

```cpp
struct ServiceEndpoint {
    std::string ip;      // IP 地址
    int port;            // 端口号
    std::string host;    // ip:port 格式，方便直接使用
    int weight;          // 权重（默认 100）
    bool healthy;        // 健康状态（默认 true）
};
```

**设计优势**：
- 不同服务发现机制返回的数据格式统一
- 下游业务代码不需要关心底层实现细节
- 便于实现负载均衡和健康检查

### 2. 抽象基类：ServiceDiscovery

定义了所有服务发现实现必须遵循的接口：

```cpp
class ServiceDiscovery {
public:
    virtual bool Init(const std::string &service_address) = 0;
    virtual bool GetAllEndpoints(std::vector<ServiceEndpoint> &endpoints) = 0;
    virtual bool GetOneEndpoint(ServiceEndpoint &endpoint) = 0;
    virtual bool Refresh() = 0;
    virtual std::string GetType() const = 0;
};
```

**接口说明**：
- `Init`：初始化服务发现（传入 domain 或 URL）
- `GetAllEndpoints`：获取所有可用服务节点
- `GetOneEndpoint`：获取单个服务节点（由实现决定负载均衡策略）
- `Refresh`：强制刷新底层数据
- `GetType`：返回服务发现类型名称（用于日志）

### 3. 带缓存的基类：CachedServiceDiscovery

为"按需拉取 + 短期缓存"型的服务发现提供通用缓存框架：

```cpp
class CachedServiceDiscovery : public ServiceDiscovery {
protected:
    // 子类只需实现这个方法
    virtual bool FetchEndpoints(std::vector<ServiceEndpoint> &endpoints) = 0;
    
    // 可运行时调整缓存 TTL
    void SetCacheTtlSeconds(int cache_ttl_seconds);
    
private:
    std::mutex cache_mutex_;
    CacheEntry cache_;  // 包含端点列表、更新时间、有效性标志
    int cache_ttl_seconds_;  // 缓存过期时间（秒）
};
```

**缓存策略**：
1. **快路径**：缓存命中且未过期 → 直接返回（锁内操作）
2. **慢路径**：缓存缺失/过期 → 在锁外执行 `FetchEndpoints`（避免阻塞）
3. **写回缓存**：获取到新数据后更新缓存（统一加锁）

**设计亮点**：
- **锁粒度优化**：网络 IO 在锁外执行，只读写缓存时加锁
- **线程安全**：使用 `std::mutex` 保护缓存状态
- **可配置 TTL**：工厂层可根据 URL 参数动态设置缓存时间

### 4. URL 解析器：ServiceDiscoveryUrl

统一解析 URL 格式的配置文件：

```cpp
struct ServiceDiscoveryUrl {
    std::string scheme;  // 协议类型（vipserver/spectrum/static）
    std::string body;    // URL 主体（domain 或 ip:port 列表）
    std::map<std::string, std::string> params;  // 查询参数
    
    int GetIntParam(const std::string &key, int default_value) const;
    static bool Parse(const std::string &url, ServiceDiscoveryUrl &out);
};
```

**URL 格式**：`<scheme>://<body>[?key=value&key=value...]`

示例：
- `vipserver://pace.meta.vipserver?timeout=10`
- `spectrum://v-ad2d143d?cache_time=30&retry_time=3&timeout=5000&port=12348`
- `static://11.22.33.44:8080,33.55.66.77:8080`

### 5. 工厂类：ServiceDiscoveryFactory

根据 URL 创建并初始化对应的服务发现实例：

```cpp
class ServiceDiscoveryFactory {
public:
    static std::unique_ptr<ServiceDiscovery> CreateServiceDiscovery(const std::string &url);
};
```

**工厂逻辑**：
1. 解析 URL 获取 scheme 和参数
2. 根据 scheme 选择对应的实现类
3. 从 URL 参数中提取配置（timeout、cache_time 等）
4. 初始化实例并返回
5. 失败时返回 `nullptr`（表示不使用服务发现）

---

## 与原有实现的区别

### 原有实现（仅 VIPServer）

**架构**：
```
业务代码
    ↓
VIPServerSubscriber (硬编码)
    ↓
VIPServer SDK
```

**问题**：
1. **紧耦合**：业务代码直接依赖 `VIPServerSubscriber` 类
2. **不可扩展**：如果要支持其他服务发现机制，需要修改大量业务代码
3. **配置分散**：不同组件的配置方式不一致
4. **无统一抽象**：没有 `ServiceEndpoint` 统一数据结构

**代码示例（旧）**：
```cpp
// 旧代码：直接使用 VIPServerSubscriber
VIPServerSubscriber subscriber;
subscriber.Init("pace.meta.vipserver");
// 业务逻辑与 VIPServer 强绑定
```

### 新实现（通用服务发现框架）

**架构**：
```
业务代码
    ↓
ServiceDiscovery (抽象接口)
    ↓
ServiceDiscoveryFactory
    ↓
┌──────────────────┬──────────────────┬──────────────┐
│VIPServerSubscriber│SpectrumDiscovery│StaticDiscovery│
└──────────────────┴──────────────────┴──────────────┘
```

**优势**：
1. **解耦**：业务代码只依赖 `ServiceDiscovery` 抽象
2. **可扩展**：新增服务发现类型只需添加实现类，业务代码零改动
3. **统一配置**：通过 URL 格式统一配置，支持多种协议
4. **双语言支持**：C++ 和 Python 实现完全对齐
5. **缓存优化**：`CachedServiceDiscovery` 提供统一的缓存框架
6. **灵活的负载均衡**：不同实现可选择不同的负载均衡策略

**代码示例（新）**：
```cpp
// 新代码：通过工厂创建，业务代码不关心具体实现
auto discovery = ServiceDiscoveryFactory::CreateServiceDiscovery(
    "spectrum://v-ad2d143d?cache_time=30&timeout=5000&port=12348"
);
if (discovery) {
    std::vector<ServiceEndpoint> endpoints;
    discovery->GetAllEndpoints(endpoints);
    // 业务逻辑与服务发现机制解耦
}
```

### 对比表格

| 特性 | 原有实现 | 新实现 |
|------|---------|--------|
| 支持的协议 | 仅 VIPServer | VIPServer / Spectrum / Static / 可扩展 |
| 业务代码耦合度 | 高（直接依赖具体类） | 低（依赖抽象接口） |
| 配置方式 | 分散 | 统一 URL 格式 |
| 缓存机制 | 各自实现 | 统一 `CachedServiceDiscovery` 框架 |
| 负载均衡 | VIPServer SDK 内置 | 可自定义（随机/轮询/加权等） |
| 双语言支持 | 仅 C++ | C++ + Python（配置格式一致） |
| 扩展性 | 需修改业务代码 | 只需添加新实现类 |
| 测试友好性 | 难（依赖外部 SDK） | 易（可用 Static 模拟） |

---

## 支持的协议类型

### 1. VIPServer（阿里内部服务发现）

**适用场景**：阿里内部部署，使用 VIPServer 进行服务注册与发现

**URL 格式**：
```
vipserver://<domain>[?timeout=<sec>]
```

**示例**：
```
vipserver://pace.meta.vipserver?timeout=10
```

**特点**：
- 直接继承 `ServiceDiscovery`（不使用缓存基类）
- VIPServer SDK 内部已带订阅式缓存
- 支持异步刷新和健康检查
- 查询超时可配置

**实现细节**：
```cpp
class VIPServerSubscriber : public ServiceDiscovery {
    // 不使用 CachedServiceDiscovery
    // 直接调用 VipClientApi::QueryAllIp / QueryIp
    // SDK 内部维护订阅缓存
};
```

### 2. Spectrum（HTTP API 拉取）

**适用场景**：使用 Spectrum 网关进行服务发现，通过 HTTP API 拉取实例列表

**URL 格式**：
```
spectrum://<virtual_service_id>[?cache_time=<sec>&retry_time=<n>&timeout=<ms>&port=<custom_port>]
```

**示例**：
```
spectrum://v-ad2d143d?cache_time=30&retry_time=3&timeout=5000&port=12348
```

**参数说明**：
- `cache_time`：本地缓存有效期（秒），默认 30 秒
- `retry_time`：单次刷新内的额外重试次数（不含首次），默认 0（仅尝试一次）
- `timeout`：HTTP 请求超时时间（毫秒），默认 5000
- `port`：自定义端口号（可选）。
  - 配置后所有 endpoint 的端口都会被强制覆盖为该值，
    Spectrum 网关 JSON 响应中的 `port` 字段将被忽略（也允许缺省）。
  - 不配置时保持向后兼容，使用 JSON 响应中 `instances[i].port` 字段的端口。

**特点**：
- 继承 `CachedServiceDiscovery`（带本地缓存）
- 通过 HTTP GET 请求 Spectrum 网关
- 默认缓存 30 秒
- 支持重试和超时配置
- 支持自定义端口号覆盖（适用于 Spectrum 实例端口与业务实际端口不一致的场景）

**API 端点**：
```
GET http://127.0.0.1:8880/api/v1/discovery/virtual-services/{id}/instances
```

**响应格式**：
```json
{
  "virtual_service_id": "v-ad2d143d",
  "instances": [
    {
      "ip": "172.1.2.10",
      "port": 8080,
      "name": "ds-abdedesd-ad2d-sded",
      "physical_service_id": "abdedesd",
      "weight": 100
    }
  ]
}
```

**实现细节**：
```cpp
class SpectrumServiceDiscovery : public CachedServiceDiscovery {
protected:
    // 只需实现这个方法，缓存由基类管理
    bool FetchEndpoints(std::vector<ServiceEndpoint> &endpoints) override {
        // HTTP 请求 Spectrum API
        // 解析 JSON 响应
        // 返回端点列表
    }
};
```

### 3. Static（静态配置）

**适用场景**：
- 测试环境
- 单机部署
- 不需要动态服务发现的场景

**URL 格式**：
```
static://<ip:port>[,<ip:port>...]
```

**示例**：
```
static://11.22.33.44:8080,33.55.66.77:8080
```

**特点**：
- 直接继承 `ServiceDiscovery`
- 端点列表在初始化时固定
- `GetOneEndpoint` 使用 round-robin 负载均衡
- `Refresh` 是 no-op（端点不会变化）
- 线程安全（使用 `atomic` 计数器）

**实现细节**：
```cpp
class StaticServiceDiscovery : public ServiceDiscovery {
    std::vector<ServiceEndpoint> endpoints_;
    std::atomic<size_t> rr_counter_{0};  // round-robin 计数器
    
    bool GetOneEndpoint(ServiceEndpoint &endpoint) override {
        const size_t idx = rr_counter_.fetch_add(1) % endpoints_.size();
        endpoint = endpoints_[idx];
        return true;
    }
};
```

---

## 使用示例

### C++ 使用示例

#### 示例 1：使用 Spectrum 服务发现

```cpp
#include "kv_cache_manager/common/service_discovery_factory.h"

using namespace kv_cache_manager;

int main() {
    // 通过 URL 创建服务发现实例
    auto discovery = ServiceDiscoveryFactory::CreateServiceDiscovery(
        "spectrum://v-ad2d143d?cache_time=30&retry_time=3&timeout=5000&port=12348"
    );
    
    if (!discovery) {
        // 降级处理：使用静态配置或默认地址
        return -1;
    }
    
    // 获取所有端点
    std::vector<ServiceEndpoint> endpoints;
    if (discovery->GetAllEndpoints(endpoints)) {
        for (const auto &ep : endpoints) {
            std::cout << "Endpoint: " << ep.host 
                      << " (weight: " << ep.weight << ")" << std::endl;
        }
    }
    
    // 获取单个端点（随机负载均衡）
    ServiceEndpoint one_ep;
    if (discovery->GetOneEndpoint(one_ep)) {
        std::cout << "Selected: " << one_ep.host << std::endl;
    }
    
    return 0;
}
```

#### 示例 2：使用静态配置（测试环境）

```cpp
auto discovery = ServiceDiscoveryFactory::CreateServiceDiscovery(
    "static://127.0.0.1:8080,127.0.0.1:8081"
);

if (discovery) {
    std::vector<ServiceEndpoint> endpoints;
    discovery->GetAllEndpoints(endpoints);
    // endpoints 包含两个端点
}
```

#### 示例 3：不使用服务发现（降级）

```cpp
// 空字符串表示不使用服务发现
auto discovery = ServiceDiscoveryFactory::CreateServiceDiscovery("");

if (!discovery) {
    // 使用硬编码的默认地址
    std::string default_host = "127.0.0.1:8080";
    // 直接使用 default_host 连接服务
}
```

### Python 使用示例

```python
from kv_cache_manager.py_connector.common.service_discovery_factory import (
    create_service_discovery
)

# 使用 Spectrum 服务发现
discovery = create_service_discovery(
    "spectrum://v-ad2d143d?cache_time=30&retry_time=3&timeout=5000&port=12348"
)

if discovery:
    # 获取所有端点
    endpoints = discovery.get_all_endpoints()
    for ep in endpoints:
        print(f"Endpoint: {ep.host} (weight: {ep.weight})")
    
    # 获取单个端点
    one_ep = discovery.get_one_endpoint()
    if one_ep:
        print(f"Selected: {one_ep.host}")
    
    # 使用 context manager 自动释放资源
    with create_service_discovery("static://127.0.0.1:8080") as sd:
        endpoints = sd.get_all_endpoints()
```

### 配置集成示例

在实际项目中，服务发现 URL 通常通过配置文件传入：

```yaml
# config.yaml
meta_service:
  service_discovery_url: "spectrum://v-ad2d143d?cache_time=30"
  
storage:
  service_discovery_url: "vipserver://pace.meta.vipserver?timeout=10"
```

```cpp
// 从配置读取 URL
std::string url = config.GetString("meta_service.service_discovery_url", "");
auto discovery = ServiceDiscoveryFactory::CreateServiceDiscovery(url);
```

---

## 技术亮点

### 1. 锁粒度优化

`CachedServiceDiscovery` 将网络 IO 操作放在锁外执行：

```cpp
bool CachedServiceDiscovery::GetAllEndpoints(std::vector<ServiceEndpoint> &endpoints) {
    // 1) 快路径：锁内读缓存
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        if (cache_.valid && !IsCacheExpiredLocked()) {
            endpoints = cache_.endpoints;
            return true;
        }
    }
    
    // 2) 慢路径：锁外执行网络 IO
    std::vector<ServiceEndpoint> new_endpoints;
    bool fetched = FetchEndpoints(new_endpoints);  // 不持有锁！
    
    // 3) 写回缓存：统一加锁
    std::lock_guard<std::mutex> lock(cache_mutex_);
    if (fetched) {
        cache_.endpoints = std::move(new_endpoints);
        cache_.update_time = std::chrono::steady_clock::now();
        cache_.valid = true;
    }
    endpoints = cache_.endpoints;
    return true;
}
```

**优势**：避免在网络请求期间持有锁，减少锁竞争

### 2. 线程安全的负载均衡

`StaticServiceDiscovery` 使用 `atomic` 实现无锁 round-robin：

```cpp
std::atomic<size_t> rr_counter_{0};

bool GetOneEndpoint(ServiceEndpoint &endpoint) {
    const size_t idx = rr_counter_.fetch_add(1, std::memory_order_relaxed) 
                       % endpoints_.size();
    endpoint = endpoints_[idx];
    return true;
}
```

**优势**：无锁设计，高并发下性能更好

### 3. 随机负载均衡

`CachedServiceDiscovery` 使用 thread-local 随机数生成器：

```cpp
static thread_local std::mt19937 gen{std::random_device{}()};
std::uniform_int_distribution<size_t> dis(0, endpoints.size() - 1);
endpoint = endpoints[dis(gen)];
```

**优势**：
- thread-local 避免多线程竞争
- 随机分布更均匀

### 4. 双语言一致性

C++ 和 Python 实现完全对齐：
- 相同的 URL 格式
- 相同的参数语义
- 相同的返回值约定
- 便于配置在前后端复用

### 5. 灵活的扩展机制

新增服务发现类型只需 3 步：

1. **实现 ServiceDiscovery 接口**
```cpp
class MyServiceDiscovery : public ServiceDiscovery {
    bool Init(const std::string &service_address) override;
    bool GetAllEndpoints(std::vector<ServiceEndpoint> &endpoints) override;
    // ...
};
```

2. **在工厂中注册**
```cpp
if (url_info.scheme == "mytype") {
    return CreateMyService(url_info);
}
```

3. **配置 URL**
```
mytype://my-service?param1=value1
```

**业务代码零改动**！

### 6. 优雅的降级策略

- URL 为空 → 返回 `nullptr` → 业务代码使用静态配置
- 初始化失败 → 返回 `nullptr` → 业务代码降级
- 获取端点失败 → 返回 `false` → 业务代码重试或使用缓存

### 7. 完善的错误处理

- URL 解析失败 → 打印 warning → 返回 `nullptr`
- 网络请求失败 → 打印 error → 返回 `false`
- JSON 解析失败 → 打印 error → 返回 `false`
- 参数缺失 → 使用默认值 → 继续运行

---

## 总结

通用服务发现框架通过**抽象接口 + 工厂模式 + 缓存框架**的设计，实现了：

✅ **解耦**：业务代码不依赖具体实现  
✅ **可扩展**：新增协议只需添加实现类  
✅ **统一配置**：URL 格式统一，支持多种协议  
✅ **双语言支持**：C++ 和 Python 完全对齐  
✅ **性能优化**：锁粒度优化、thread-local、无锁设计  
✅ **易于测试**：可用 Static 模式模拟服务发现  
✅ **优雅降级**：失败时返回 nullptr，业务代码可降级处理

这个框架为 KVCacheManager 提供了灵活、高效、易用的服务发现能力，是系统架构的重要基础设施。

---

## 升级与兼容性约束

本节聚焦 `TairMemPoolStorageSpec` 从老版（`enable_vipserver` + `vipserver_domain`）到新版（`service_discovery_url`）的升级路径。

### 服务端的 JSON / proto 兼容性

- **JSON 输入**：服务端 `TairMemPoolStorageSpec::FromRapidValue` 会**自动迁移**老字段：
  - 当 `service_discovery_url` 为空、且 `enable_vipserver=true` 与非空 `vipserver_domain` 同时存在时，自动等价于 `service_discovery_url="vipserver://<vipserver_domain>"`。
  - `service_discovery_url` 已显式设置时优先级最高，老字段被忽略。
  - 老 admin 工具 / 老的持久化数据可以无缝过渡。
- **proto 二进制**：`enable_vipserver` (tag 3) / `vipserver_domain` (tag 4) 在 `admin_service.proto` 与 `meta_service.proto` 中已 `reserved`。老 admin 二进制消息中的这两个字段会被 protobuf 静默丢弃；如果客户端依赖它们，请改用 `service_discovery_url`。

### Server → Client 升级顺序的运维约束

`storage_configs` 在 RPC 上是 JSON 字符串，老版 client 完全不识别 `service_discovery_url`，会直接调用 `pace_init(domain)`。因此：

> **`TairMemPoolStorageSpec.domain` 必须始终保持原来有效的 pace 入口，不允许在迁移到 `service_discovery_url` 时把它清空或改成无效占位符。**

执行约束：
- 服务端先升级、客户端尚未升级时，`domain` 仍是 client 唯一的连接信息来源。
- `TairMemPoolStorageSpec::ValidateRequiredFields` 已强制 `domain` 非空，但「非空」不等于「有效」，运维需要保证 `domain` 解析后真实可达。
- 待所有 client 升级到识别 `service_discovery_url` 的版本之后，才可以考虑把 `domain` 退化为单纯的 fallback 兜底地址。

### Client 侧对 `service_discovery_url` 的支持

新版客户端的 `TairMempoolSdk::Init` 会优先消费 `service_discovery_url`：
- 非空时，直接把 URL 透传给 `pace_init`，由 `pace_mp` 自身识别 `vipserver://` / `spectrum://` / `static://` 等形式（pace_mp 是内部组件，已与 service_discovery_url 完全兼容）。
- 为空时使用 `domain` 作为 `pace_init` 的目标。
- 这样新 client 在面对老 server（`service_discovery_url` 为空）时，行为与历史版本一致。

