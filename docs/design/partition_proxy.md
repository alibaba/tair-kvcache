# KVCM 分区代理

Client 从服务发现获取 Proxy 地址。Proxy 接收现有 MetaService HTTP 和 gRPC 请求，经分区内的 Leader
执行元数据操作。CacheLocation 和写会话仍由 KVCM 分配，客户端继续直连存储传输数据。

## 模块和接口

| 代码 | 职责 |
| --- | --- |
| `service/worker.h`、`command_line.cc` | 按 `kvcm.service.worker_type` 选择进程角色。默认值 `kvcm` 启动原有 Server |
| `service/proxy_worker.cc` | 装配 Proxy 的公共 Meta 监听、内部注册监听、请求执行器、路由和转发器 |
| `routing/routing_table.h` | `BucketPolicy::GetBucket` 提取逻辑桶。`ApplyAssignment` 校验版本和完整分配，`Report` 接收 Replica 状态，`Resolve` 返回唯一的有效 Leader |
| `protocol/protobuf/proxy_control_service.proto`、`service/proxy/partition_reporter.cc` | 接收管控生成的 PartitionTarget。KVCM 发现所有 Proxy 注册端口，向每个端点独立上报目标、进程标识和就绪状态 |
| `service/proxy/meta_forwarder.cc` | HTTP/gRPC 共用按类型定义的异步 Meta 转发。下游 gRPC 完成时释放入口请求。GetClusterInfo 在 Proxy 上返回自身对外地址 |

当前策略使用 `Hash64(instance_id, seed=0) % bucket_count`。相同实例的 Register、Query、Start、
Finish 和 ReportEvent 落在同一 Partition。一次请求只选择一个后端。每个 Proxy 独立维护内存路由表，
重启后靠后端重新上报恢复。每个进程启动时生成新的注册标识，心跳使用递增序号。两个有效进程同时
宣告同一 Partition 的 Leader 时，该分区暂时返回 `SERVICE_NOT_READY`。其余分区继续服务。

## 配置示例

对外的服务发现只发布 Proxy。KVCM 另行发现 Proxy 的内部注册端口，该端口应位于受信任的服务网络中。
注册端点需要能在公共入口放量前被发现。以下是单机联调的两个进程配置片段。

```ini
# Proxy
kvcm.service.worker_type=proxy
kvcm.service.advertised_host=127.0.0.1
kvcm.service.rpc_port=9100
kvcm.service.http_port=9101
kvcm.proxy.control_address=127.0.0.1:9102
kvcm.proxy.cluster=example
kvcm.proxy.route_version=1
```

```ini
# KVCM，为每个 Partition 配置独立的 registry 和 metadata 范围
kvcm.service.worker_type=kvcm
kvcm.service.advertised_host=127.0.0.1
kvcm.service.rpc_port=9200
kvcm.service.http_port=9201
kvcm.service.admin_rpc_port=9202
kvcm.service.admin_http_port=9203
kvcm.partition.target_file=/path/to/partition-0.json
kvcm.proxy.discovery_uri=static://127.0.0.1:9102
```

管控给同一 Partition 的所有 Replica 下发相同的文件。下面的目标把桶 0、2 分给 Partition 0，
桶 1、3 分给 Partition 1。Partition 1 的文件只需把 `partition` 改为 1。每个 Replica 使用自己可直连的
地址。多 Proxy 联调时，`static://` 可列出逗号分隔的多个控制端点；部署环境也可以使用已实现的
ServiceDiscovery 提供方。独立容器与同容器使用相同的注册与转发路径。

```json
{"cluster":"example","version":1,"partition":0,"bucketOwners":[0,1,0,1]}
```

## 故障与版本

- 同一 Partition 的 Replica 共享支持选主的协调后端及可恢复的 Registry/元数据。选主键按集群和
  Partition 隔离。后端在完成恢复并打开业务门禁后才上报可服务状态。进程内协调适合单 Replica。
- Proxy 从配置获得生效版本，只接受这个版本的注册。热路径按同一份分配计算 Bucket。后端校验
  转发请求的版本和归属。首版在启动时加载目标。跨 Partition 移交所有权需要先摘流、排空写会话、
  停止旧持有者、完成数据交接，再启用新目标。心跳只更新可达性和就绪状态。
- 更换 Proxy 可以继续原写会话。后端进程故障可能丢失未完成会话，Proxy 不自动重放结果未知的写入。
  一个后端失联会在租约过期前返回下游错误，过期后返回该分区未就绪。
- Admin、Debug、Optimizer 订阅仍由 KVCM 的内部入口提供。各 Partition 的 Registry、索引、GC 和配额
  应按部署隔离。同一资源池上的跨 Partition 配额、在线迁桶和全局 Admin 聚合留给后续实现。

## 后续开发与回归

- **管控对接**。将 PartitionTarget 的文件交付接入部署系统，发布 Proxy 公共端点和内部注册端点。
  在线更新需要同时处理旧 owner 摘流、数据恢复和版本激活。
- **策略扩展**。若要拆分一个热门实例，先定义跨桶前缀查询、写会话和完整事件快照的语义，再实现
  对应 BucketPolicy 和后端归属校验。
- **运行验证**。`//kv_cache_manager/routing:RoutingTableTest` 检查版本、冲突和过期。
  `//integration_test/proxy:partition_proxy_test` 启动真实进程，检查分区归属、跨 Proxy 完成写入、
  后端故障与同目标 Replica 接管。`//integration_test/service_dispatch:service_dispatch_smoke`
  覆盖默认 KVCM 的 HTTP/gRPC 入口。
- **容量验证**。HTTP 请求在 Proxy 解析后通过 gRPC 转发。上线前需要按实际最大 ReportEvent 快照
  和查询响应验证下游消息大小限制、在途内存以及两跳延迟。
