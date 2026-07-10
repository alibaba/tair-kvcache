# 服务发现扩展框架

## 目标

KVCacheManager 提供 provider-neutral 的服务发现抽象，使业务代码只依赖统一的
端点模型，不依赖具体注册中心、网关或 SDK。开源构建内置 `static://`，其他实现由
构建环境通过扩展钩子注册。

主要约束：

- 业务代码只使用 `ServiceDiscovery`，不直接依赖 provider 实现。
- URL 只约定通用形式 `<scheme>://<body>[?key=value...]`；查询参数由 provider
  自己解释。
- 工厂创建失败返回空值，调用方决定报错或使用静态地址降级。
- provider 的协议、凭据、网关和重试策略不进入公共框架。

## 统一端点模型

C++ 和 Python 均使用 `ServiceEndpoint` 表示发现结果：

| 字段 | 说明 |
| --- | --- |
| `ip` | 节点地址 |
| `port` | 服务端口 |
| `host` | 可直接使用的 `ip:port` 字符串 |
| `weight` | provider 返回的权重，默认 100 |
| `healthy` | 健康状态，默认 true |

`ServiceDiscovery` 的通用操作包括：获取全部端点、选择一个端点、强制刷新和释放
资源。C++ 实现还包含 `Init`，Python 实现通常在构造阶段完成初始化。

## 内置 static provider

静态端点适用于本地部署、测试和不需要动态订阅的场景：

```text
static://10.0.0.1:8080,10.0.0.2:8080
```

初始化时会严格校验每个 `host:port`，端口范围为 1–65535。
`GetOneEndpoint` / `get_one_endpoint` 使用线程安全的 round-robin；`Refresh` 是
无外部 IO 的成功操作。

## C++ 扩展

公共工厂先处理 `static://`，再调用构建环境提供的扩展函数：

```cpp
std::unique_ptr<ServiceDiscovery> CreateServiceDiscoveryExtension(
    const ServiceDiscoveryUrl &url_info);
```

扩展支持当前 scheme 时，返回已初始化的 `ServiceDiscovery`；不支持或初始化失败
时返回 `nullptr`。开源构建提供 no-op 实现，因此公共工厂不会链接任何私有 SDK。

新增构建专属 provider 时：

1. 实现 `ServiceDiscovery`。
2. 在构建专属目录实现 `CreateServiceDiscoveryExtension`，根据 `url_info.scheme`
   创建实例并解释参数。
3. 让构建专属 Bazel target 以
   `//stub_source/kv_cache_manager/common:service_discovery_factory_extension`
   暴露该实现。
4. 为 provider 实现和工厂装配分别补充测试。

公共 `service_discovery_factory` 显式依赖扩展 target，避免静态链接时扩展实现被
裁剪。

## Python 扩展

Python 工厂提供线程安全注册函数：

```python
register_service_discovery_provider("custom", create_custom_discovery)
```

provider factory 签名为：

```python
def create_custom_discovery(body: str, params: dict[str, str]):
    ...
```

构建专属模块
`stub_source.kv_cache_manager.py_connector.common.service_discovery_factory_extension`
应实现：

```python
def register_service_discovery_providers(register_provider):
    register_provider("custom", create_custom_discovery)
```

扩展模块只加载一次；重复注册默认报错，避免加载顺序悄悄覆盖已有 provider。
开源构建的扩展模块不注册额外 provider。

## Python Manager Client 集成

`KvCacheManagerClient` 的 `base_url` 支持两类输入：

- `http://` / `https://`：直接作为 Manager 种子地址。
- 服务发现 URL：通过工厂创建 provider，并选择一个端点作为初始种子。

启用 Leader 自动发现后，每次调用 `/api/getClusterInfo` 都会重新向 provider 选择
种子节点；服务发现暂时返回空结果时，回退到初始化成功的种子地址。收到
`SERVER_NOT_LEADER` 后的重试也复用同一 Leader 发现路径。`close()` 会停止后台
刷新线程，同时关闭 HTTP session 和服务发现实例。

这种设计把两层职责分开：provider 负责找到任意可用 Manager 种子，Leader 发现
负责从种子查询当前 Leader。

## 缓存基类

需要本地 TTL 缓存的 C++ provider 可继承 `CachedServiceDiscovery`，只实现
`FetchEndpoints`。缓存基类的慢速拉取在锁外执行，成功结果写回缓存；刷新失败时
保留上次有效缓存。选择单个端点时使用线程局部随机数生成器，避免共享锁竞争。

provider 已自带订阅和缓存时，应直接实现 `ServiceDiscovery`，避免叠加两层缓存。

## 安全与发布检查

构建专属 provider 必须位于专属源码目录。发布开源源码或 wheel 前应检查：

- 公共工厂只包含 `static://` 和通用扩展机制。
- 构建专属模块不会进入开源源码包或 wheel。
- 文档、示例、测试数据和启动脚本不包含专属协议或网关信息。
- 两种构建模式下，扩展 target 和 Python 扩展模块都存在，开源实现为 no-op。
