# KVCM 分区 Proxy

Client 从 VipServer 发现 Proxy。Proxy 用 HTTP/gRPC 接收 MetaService 请求，使用内部 gRPC 连接 KVCM；Client 拿到 CacheLocation 后直接访问存储。Proxy 可以独立部署，也可以和 KVCM 在一个容器内以两个进程运行。

## 按操作选择路由

Bucket 的输入只有业务路由键。默认策略按 `Hash64(key, seed=0) % partition_count` 选列，列号从 0 开始。`instance_id` 保留实例命名空间，`instance_group` 选择实例组配置；两者都不参与哈希。不同实例使用相同业务键会选到同一列，各实例的数据仍按命名空间隔离。

| 操作 | 目标与结果 |
| --- | --- |
| 创建 instance（RegisterInstance） | 广播到本 Proxy 管理的每一列，每列仅发给唯一就绪的 Leader。全部成功后合并客户端存储配置 |
| 节点注册、心跳、下线 | role 0 处理全局生命周期。若上报 IP 的属列不同，role 0 成功后，原始批次交给该属列，以维护该列现有的 reporter 登记、快照及生命周期检查 |
| EventReport 写入 | 从请求的 `host_ip_port` 提取并规范化 IP 后选列。同 IP 不同端口同列；保留完整端点标识不同 reporter。完整快照只在属列生效 |
| EventReport 查询 | `GetHostCacheState` 查询所有分区，按完整 reporter 端点合并 local 命中。跨列 Top-K 联合远端前缀仍需逐块证据，目前显式返回 UNSUPPORTED |
| 普通 KVCache | 查询、Start、Finish、按 key 删除携带同一业务 `session_id`。Start 返回的 `write_session_id` 仍标识后端写会话。缺少业务键返回 INVALID_ARGUMENT |
| EmbCache | `RouteKey::Embedding(emb_key)` 按对象键固定选列。MetaService 的按键操作也可携带 `emb_key`。独立 KVMeta 入口、跨列批写和子会话归并由接入方实现 |
| 独立 PACE allocate | `RouteKey::PaceAllocate()` 从有唯一就绪 Leader 的列中随机选一列。供独立 allocate 入口使用；普通 KVCache 写入仍按 session 选列 |
| 其他元数据操作 | GetInstanceInfo 去 role 0，实例级 TrimCache 广播，GetClusterInfo 由 Proxy 返回自身公共地址 |

广播先检查公共配置中的全部列。某列缺少唯一就绪 Leader 时返回 SERVICE_NOT_READY。子请求发出后若部分失败，调用方收到失败，已成功创建的实例会保留；可对相同配置重试。各列返回的存储配置按 `global_unique_name` 合并，同名配置或组的 `extra_info` 冲突会报错。

## 管控如何指定列号

每个 KVCM 的角色目标只有一个字段，例如 `{"partition": 1}`。同一列的所有 Replica 收到相同目标。集群名、版本和总列数是 Proxy 与 KVCM 一致使用的公共部署配置。管控无需计算或下发 bucket owner 表。

| 信息 | 来源 | 使用方 |
| --- | --- | --- |
| 当前列号 | 每个 KVCM 的角色目标 | KVCM 校验列号、隔离选主键、向所有 Proxy 上报 |
| cluster、route_version、partition_count | 公共部署配置 | Proxy 预先知道完整列集合；集群、版本或列数不一致的上报被拒绝 |
| Replica 的地址、就绪状态、心跳序号 | 各 KVCM 主动上报 | Proxy 为每列挑选唯一有效 Leader。每轮结束后等待 2 秒，注册 RPC 截止时间 1 秒，租约 10 秒 |
| 业务键的归属 | Proxy 和后端执行同一哈希规则 | 后端校验转发的版本、目标列与业务键。调整列数会改变映射，需要数据交接、写会话排空并切换版本 |

## 配置与部署

公共配置示例，两个 Worker 均需配置。

```ini
kvcm.proxy.cluster=example
kvcm.proxy.route_version=2
kvcm.proxy.partition_count=2
```

Proxy 进程：

```ini
kvcm.service.worker_type=proxy
kvcm.service.advertised_host=127.0.0.1
kvcm.service.rpc_port=9100
kvcm.service.http_port=9101
kvcm.proxy.control_address=127.0.0.1:9102
```

KVCM 进程：

```ini
kvcm.service.worker_type=kvcm
kvcm.service.advertised_host=127.0.0.1
kvcm.service.rpc_port=9200
kvcm.service.http_port=9201
kvcm.service.admin_rpc_port=9202
kvcm.service.admin_http_port=9203
kvcm.partition.target_file=/path/to/partition-1.json
kvcm.proxy.discovery_uri=static://127.0.0.1:9102
```

以上地址用于单机联调。部署侧发布 Proxy 的公共端点，KVCM 通过受信任网络发现所有 Proxy 的内部注册端点。每列的 Registry、索引、GC 与配额需要按部署隔离，同一列的 Replica 则需共享协调后端及可恢复的元数据。

## 代码入口

| 模块 | 职责 |
| --- | --- |
| `service/worker.h`、`service/proxy_worker.cc` | 装配两种 Worker，建立公共协议入口与内部注册监听 |
| `service/proxy/meta_route.h`、`meta_route.cc` | 从具体请求提取操作与业务键，HTTP 与 gRPC 共用 |
| `routing/routing_table.h` | 解析哈希、全列广播、role 0 或随机单选；从版本与租约快照取得目标 |
| `service/proxy/meta_forwarder.cc` | 异步转发、等待广播子请求、聚合响应，处理 role 0 与属列之间的生命周期操作 |
| `service/proxy/partition_reporter.cc` | 读取列号、上报就绪状态并校验请求归属 |

切换 Proxy 后，Client 用相同的业务键和写会话 ID 可以继续向原 Leader 提交。KVCM 进程失效后，未提交的写会话可能丢失；Proxy 不自动重放结果未知的写入。
