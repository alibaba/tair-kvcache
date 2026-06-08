# HTTP Bearer 鉴权

KVCacheManager 在两个端口上对外暴露三个 HTTP 服务：

| 服务 | 默认端口 | 鉴权 |
|---|---|---|
| Meta   | 6382 | 始终开放（数据面） |
| Admin  | 6492 | 可选 Bearer Token |
| Debug  | 6492 | 可选 Bearer Token |

Admin 与 Debug HTTP 服务可使用 HTTP Bearer 鉴权（[RFC 6750]）进行保护。
Meta HTTP 服务是面向推理引擎的数据面入口，按设计保持开放，预期仅在受
信网络内可达。

[RFC 6750]: https://datatracker.ietf.org/doc/html/rfc6750

## 快速开始

通过 `kvcm.service.admin_auth_token` 配置项设置一个或多个被接受的
Token：

```bash
# 单个 token
kv_cache_manager_bin -e 'kvcm.service.admin_auth_token=s3cret-token'

# 多个 token（逗号分隔，用于分阶段轮换）
kv_cache_manager_bin -e 'kvcm.service.admin_auth_token=tok-old,tok-new'
```

随后在调用受保护端点时携带 `Authorization: Bearer …` 头：

```bash
# Prometheus 抓取（Admin 端口 6492）
curl -H 'Authorization: Bearer s3cret-token' \
     http://<host>:6492/metrics

# 健康检查
curl -H 'Authorization: Bearer s3cret-token' \
     http://<host>:6492/api/healthy

# Debug 故障注入
curl -X POST \
     -H 'Authorization: Bearer s3cret-token' \
     -H 'Content-Type: application/json' \
     -d '{"fault":"…"}' \
     http://<host>:6492/api/injectFault
```

当 `kvcm.service.admin_auth_token` 为空（默认）时，Admin 与 Debug
服务以未鉴权方式运行，服务端启动时会输出一条 `WARN` 日志：

```
admin/debug HTTP auth disabled (kvcm.service.admin_auth_token not set);
do not expose admin/debug ports on untrusted networks
```

## 配置

| 配置项 | 默认值 | 说明 |
|---|---|---|
| `kvcm.service.admin_auth_token` | 空 | 逗号分隔的可接受 Bearer Token 列表；为空则关闭鉴权 |

该值可通过配置文件、`--env` / `-e` 启动参数或环境变量（将 `.` 替换为
`_`）设置：

```bash
# 配置文件
kvcm.service.admin_auth_token=tok-old, tok-new

# 命令行
kv_cache_manager_bin -e 'kvcm.service.admin_auth_token=tok-old,tok-new'

# 环境变量
export kvcm_service_admin_auth_token='tok-old,tok-new'
```

每个逗号分隔项前后的空白会被裁剪。空项（包括尾随逗号造成的空项）会被
静默丢弃。

### Token 轮换

通过列出多个 Token 是支持零停机轮换的方式：

1. 部署时使用 `old,new` —— 旧、新两类客户端均被接受。
2. 将客户端切换到新 Token。
3. 重新部署时只保留 `new`，下线旧 Token。

如果希望在不重启的情况下进行在线轮换，请参见下文的
[运行时 Token 管理](#运行时-token-管理)。需要注意的是，运行时变更仅保存
在内存中 —— 运维人员需要将其同步写回配置文件（或环境变量）以保证重启后
仍然生效。

## 运行时 Token 管理

Admin 服务提供三个 RPC，用于在不重启服务的情况下查看与修改当前生效的
可接受 Token 列表。这三个端点本身也由受鉴权保护的 Admin 服务提供，因此：

- 未鉴权的调用方无法将集群锁死在自己之外。
- 持有 *当前* Token 的调用方可以在一次调用中安装新 Token 并撤销旧 Token。

变更**仅在内存中生效**。它们在进程内的任何重新配置之间保持有效，但在
重启时会被重置为 `kvcm.service.admin_auth_token` 的配置值。如需持久化
有意为之的变更，请编辑配置文件。

### 端点

| HTTP 路由（POST，端口 6492） | gRPC 方法 | 用途 |
|---|---|---|
| `/api/setAdminAuthTokens` | `AdminService.SetAdminAuthTokens` | 整体替换可接受 Token 列表 |
| `/api/rotateAdminAuthToken` | `AdminService.RotateAdminAuthToken` | 原子地新增一个 Token，并（可选地）移除一个旧 Token |
| `/api/listAdminAuthTokens` | `AdminService.ListAdminAuthTokens` | 查询数量与每个 Token 的指纹 |

`Set` 传入空列表会将服务切换回**开放模式**（等价于启动时未配置任何
Token）。在此之后，第一次非空的 `Set` 又会将其切换回鉴权模式 —— 整个
过程无需重启。

### `Set` —— 整体替换

```bash
# 安装一组全新 Token（替换任何已配置的列表）
curl -X POST \
     -H 'Authorization: Bearer s3cret-token' \
     -H 'Content-Type: application/json' \
     -d '{"trace_id":"set-1","tokens":["new-token-1","new-token-2"]}' \
     http://<host>:6492/api/setAdminAuthTokens

# 关闭鉴权（开放模式）
curl -X POST \
     -H 'Authorization: Bearer s3cret-token' \
     -H 'Content-Type: application/json' \
     -d '{"trace_id":"open","tokens":[]}' \
     http://<host>:6492/api/setAdminAuthTokens
```

空字符串项（例如 `["a","","b"]`）会被静默丢弃，与配置文件解析行为
保持一致。

### `Rotate` —— 原子地先增后删

典型的零间隙轮换流程：

```bash
# 1. 新增 `new-token`（运维人员当前使用的 Token 仍然有效）
curl -X POST \
     -H 'Authorization: Bearer current-token' \
     -H 'Content-Type: application/json' \
     -d '{"trace_id":"rot-add","new_token":"new-token"}' \
     http://<host>:6492/api/rotateAdminAuthToken

# 2. 将工具切换到 `new-token`，然后下线旧 Token
curl -X POST \
     -H 'Authorization: Bearer new-token' \
     -H 'Content-Type: application/json' \
     -d '{"trace_id":"rot-drop","old_token":"current-token","new_token":"new-token"}' \
     http://<host>:6492/api/rotateAdminAuthToken
```

| `old_token` | `new_token` | 效果 |
|---|---|---|
| 空 | 非空 | 追加 `new_token`（增量添加） |
| 非空（命中已接受 Token） | 非空 | 追加 `new_token`，并移除 `old_token` |
| 非空（未命中） | 任意 | `INVALID_ARGUMENT` |
| 任意 | 空 | `INVALID_ARGUMENT` |

`Rotate` 是 `Set` 的便捷封装：它避免了 `Set([new])` 在某个节点上执行时，
该节点自身的调用方仍在使用旧 Token 而被立即断开的间隙。

### `List` —— 在不泄露密钥的前提下查询

```bash
curl -X POST \
     -H 'Authorization: Bearer s3cret-token' \
     -H 'Content-Type: application/json' \
     -d '{"trace_id":"list-1"}' \
     http://<host>:6492/api/listAdminAuthTokens
```

响应会返回一个 enforcing 标志、Token 数量，以及每个 Token 的不可逆
8 位十六进制指纹（基于原始字节计算的 32 位 FNV-1a）。指纹在多次调用之间
是稳定的，但不可还原，适合用于跨副本人工对比：

```json
{
  "header": {"status": {"code": 0}},
  "enforcing": true,
  "token_count": 2,
  "fingerprints": ["1a2b3c4d", "deadbeef"]
}
```

### 多副本部署

每个副本都各自维护一份内存中的 Token 列表。运行时变更后，需要逐个
（或脚本化地）调用所有 leader 与 follower 节点，以保证各 verifier 同步。
跨节点对比指纹集合是审计漂移的推荐方式 —— 通过 `listAdminAuthTokens`
查询。

## Prometheus 抓取

`/metrics` 端点由 Admin HTTP 服务提供，因此在启用鉴权时也受同一套 Bearer
认证保护。请在 Prometheus 抓取任务中配置 `authorization`：

```yaml
scrape_configs:
  - job_name: kvcache_manager
    metrics_path: /metrics
    static_configs:
      - targets: ["<host>:6492"]
    authorization:
      type: Bearer
      credentials: s3cret-token
      # 或：credentials_file: /etc/prometheus/kvcm_token
```

## 响应行为

通过鉴权的请求会原样转发到后端处理器。鉴权失败时会按 [RFC 7235] §4.1
与 [RFC 6750] §3 返回带 `WWW-Authenticate` 挑战的 HTTP `401 Unauthorized`：

[RFC 7235]: https://datatracker.ietf.org/doc/html/rfc7235

```http
HTTP/1.1 401 Unauthorized
WWW-Authenticate: Bearer realm="kvcm"
Content-Type: application/json

{"error":"unauthorized"}
```

当凭证存在但格式错误或被拒绝时，`WWW-Authenticate` 头会带上 `error`
参数，便于客户端区分各种情况：

| 情况 | `WWW-Authenticate` 取值 |
|---|---|
| 没有 `Authorization` 头 | `Bearer realm="kvcm"` |
| Header 不是 Bearer scheme，或格式错误 | `Bearer realm="kvcm", error="invalid_request"` |
| Bearer scheme 但 Token 未被接受 | `Bearer realm="kvcm", error="invalid_token"` |

每条被拒绝的请求都会以 `WARN` 级别记录审计日志：

```
[AUTH] denied api=/metrics outcome=3 ip=10.0.0.42
```

`outcome` 是 `AuthOutcome` 枚举的数值（`1=missing`、
`2=invalid_request`、`3=invalid_token`）。

## 设计说明

实现位于 `kv_cache_manager/service/http_service/auth/` 目录下。

### 范围：仅 Admin 与 Debug

Bearer 鉴权仅作用于 Admin 与 Debug HTTP 服务，因为这两个服务暴露了具有
副作用的操作（载入配置快照、故障注入、Debug RPC）以及可观测性数据
（`/metrics`）。Meta HTTP 服务是数据面热路径，按设计保持开放；预期部署
方将其通过防火墙限制在推理集群内可见。

### 包装顺序：`logger(auth(handler))`

`CoroHttpService::Start` 会为每个注册的 handler 套上两层中间件：外层是
日志中间件，内层是鉴权中间件：

```
client request -> logger -> auth -> handler -> response
```

将 logger 放在最外层可以保证 `401` 响应也会被请求/响应审计日志记录下来。

### 可插拔的 verifier

鉴权通过 `TokenVerifier` 接口分发（`token_verifier.h`）：

```cpp
class TokenVerifier {
public:
    virtual AuthOutcome Verify(std::string_view authz_header) const = 0;
    virtual std::string Realm() const { return "kvcm"; }
};
```

当前的具体实现 `StaticBearerTokenVerifier` 会根据一份固定的内存列表
来校验 Header。该接口为未来其他实现（例如 JWT 校验、OAuth2 introspection）
预留了空间，无需改动 HTTP 服务的接线。

### Header 解析

`StaticBearerTokenVerifier` 遵循 [RFC 7235] §2.1 与 [RFC 6750] §2：

- 对 `Authorization` Header 值的首尾可选空白（SP / HTAB）进行裁剪
- scheme 名 `Bearer` 按大小写不敏感方式匹配
- scheme 与 Token 之间至少要有一个 SP 或 HTAB 分隔；紧贴的 Token
  （例如 `BearerXYZ`）会被拒为 `invalid_request`
- Token 内部出现空白会被拒绝
- 最终的 Token 与已接受列表使用常量时间相等比较进行匹配

### 长度泄露但与首个差异位置无关的比较

`AuthUtil::ConstantTimeEquals` 在长度不一致时会立即返回 —— 因此被接受
Token 的长度会通过时序泄露 —— 长度一致时它会遍历整段字节，不会在首个
不同字节处提前结束。这可挫败那些会泄漏匹配前缀长度的简单时序侧信道。
Token 长度应控制在合理范围内，避免长度本身成为有用的侧信道。

### Admin & Debug 上始终启用 verifier

服务启动时会无条件地将 `StaticBearerTokenVerifier` 挂载到 Admin 与 Debug
HTTP 服务上 —— 即使 `kvcm.service.admin_auth_token` 为空也是如此。
"开放模式"由 verifier 自身实现：当可接受 Token 列表为空时，`Verify`
对每个请求都返回 `kOk`。其代价是每个请求多一次共享锁获取与一次空向量
判断，相对 Admin/Debug 的低 QPS 来说可以忽略。

无条件地接入 verifier 正是运行时 `Set` / `Rotate` 端点能够在不重启的
情况下将服务从开放切换到鉴权模式的关键。

Meta HTTP 服务仍然保留真正的零开销路径：不挂载任何 verifier，
`WrapWithAuth` 直接返回原始 handler。

### 不在范围内

以下内容有意不纳入本特性：

- **TLS / HTTPS 终端。** Bearer Token 在 `Authorization` 头中以明文
  传输。生产部署应在 Admin 与 Debug 端口前端的反向代理或负载均衡器上
  终止 TLS。
- **按路由的 ACL。** Admin 与 Debug 服务上所有路由共享一份 Token 列表。
  并不存在只读 Token 与管理员 Token 的区分；如果有此类需求，请使用独立
  部署，或在前面套一层执行该策略的代理。
- **运行时变更的持久化。** `SetAdminAuthTokens` 与 `RotateAdminAuthToken`
  端点只会修改内存中的 Token 列表。如需跨重启保持，应将变更同步到配置
  文件或环境变量中。
- **Token 颁发。** Token 是运维方提供的不透明字符串；本服务不负责生成
  Token。

## 文件

| 路径 | 作用 |
|---|---|
| `kv_cache_manager/service/http_service/auth/token_verifier.h` | `TokenVerifier` 接口与 `AuthOutcome` 枚举 |
| `kv_cache_manager/service/http_service/auth/static_bearer_token_verifier.{h,cc}` | 基于静态列表的 Bearer verifier |
| `kv_cache_manager/service/http_service/auth/auth_util.{h,cc}` | 常量时间比较与大小写不敏感等工具函数 |
| `kv_cache_manager/service/http_service/coro_http_service.{h,cc}` | `WrapWithAuth` 中间件与 `SetTokenVerifier` |
| `kv_cache_manager/service/server.cc` | 启动时将 verifier 接入 Admin 与 Debug 服务 |
| `kv_cache_manager/service/admin_service_impl.{h,cc}` | 基于实时 verifier 实现 `Set/Rotate/ListAdminAuthTokens` |
| `kv_cache_manager/service/server_config.{h,cc}` | 解析 `kvcm.service.admin_auth_token` |
