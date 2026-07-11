# 高可用与选主机制设计

## 1. 概述

KVCacheManager 支持多节点高可用（HA）部署模式。多个 KVCM 实例以一主多备的方式运行，通过分布式选主（Leader Election）选出唯一的 Leader 节点提供数据面服务，其余节点作为 Follower 待命。当 Leader 故障时，Follower 自动接管，实现故障自动转移。

核心设计目标：

- **自动选主**：无需人工干预，节点启动后自动参与选举。
- **故障转移**：Leader 故障或主动降级后，Follower 在租约过期窗口内自动接管。
- **Leader 发现**：客户端可向任意节点查询当前 Leader 地址，然后直连 Leader 进行数据面操作。
- **后端可扩展**：选主协调层支持多种后端（内存、文件、Redis），适配不同部署场景。

## 2. 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                      推理引擎 (Client)                        │
│                                                             │
│  ① GetClusterInfo → 发往任意节点 → 获取 leader_endpoint      │
│  ② 直连 Leader 进行读写（数据面 + 管控面）                     │
│  ③ 若请求被拒绝（Leader 已变更），回到 ① 重新发现               │
└──────────┬──────────────────────────────────┬───────────────┘
           │                                  │
  ┌────────▼────────┐               ┌────────▼─────────┐
  │ Node (Follower)  │               │  Node (Leader)    │
  │                  │               │                   │
  │  不处理读写请求    │               │   可以处理所有请求   │
  │  仅提供:          │               │                   │
  │  · Leader 查询   │               │                   │
  │  · 健康检查       │               │                   │
  │  · 运维监控       │               │                   │
  │                  │               │                   │
  │  LeaderElector   │               │  LeaderElector    │
  └────────┬─────────┘               └────────┬──────────┘
           │                                  │
  ┌────────▼──────────────────────────────────▼──────────┐
  │              CoordinationBackend                      │
  │           (分布式锁 + KV 存储)                         │
  │           实现: Memory / File / Redis                  │
  └───────────────────────────────────────────────────────┘
```

数据面和管控面的读写请求均由 Leader 处理。Follower 不处理业务请求，仅提供 Leader 查询（GetClusterInfo）、健康检查（CheckHealth）和运维监控（GetManagerClusterInfo）等能力。客户端收到 Follower 的拒绝后，应重新通过 GetClusterInfo 发现当前 Leader，将请求发往正确的节点。

## 3. 核心组件

### 3.1 CoordinationBackend

协调后端是选主和节点信息存储的基础设施层，提供两类原子操作：

- **分布式锁接口（选主用）：** TryLock / RenewLock / Unlock / GetLockHolder
- **KV 存储接口（节点端点信息存储用）：** SetValue / GetValue

三种实现：

| 实现 | URI 格式 | 适用场景 |
|---|---|---|
| CoordinationMemoryBackend | `memory://` | 单进程测试 |
| CoordinationFileBackend | `file:///path/to/dir` | 单机开发/测试 |
| CoordinationRedisBackend | `redis://[password@]host:port[?params]` | 生产环境 |

### 3.2 LeaderElector

LeaderElector 是选主状态机的核心实现，负责竞选、续约、降级和状态转换。

#### 3.2.1 角色状态机

```
                   竞选成功               OnBecomeLeader
                 （获取分布式锁）              回调完成
    FOLLOWER ──────────────► PROMOTING ──────────────► LEADER
       ▲                                                 │
       │                                        锁丢失 / │
       │               OnNoLongerLeader         主动降级  │
       │                  回调完成                        │
       └──────────────── DEMOTING ◄──────────────────────┘
```

| 状态 | 含义 | 是否可对外服务 |
|---|---|---|
| `FOLLOWER` | 备节点稳定态 | 仅响应 Leader 发现等非数据面请求 |
| `PROMOTING` | 竞选成功后的晋升中间态（执行 OnBecomeLeader 回调） | 否 |
| `LEADER` | 主节点稳定态 | 是 |
| `DEMOTING` | 降级中间态（执行 OnNoLongerLeader 回调） | 否 |

中间状态（PROMOTING / DEMOTING）确保回调逻辑（数据恢复、清理）执行完毕后才切换到稳定态，避免请求在不一致状态下被处理。

LEADER → DEMOTING 的触发条件包括：租约过期、续约失败、主动调用 Demote 接口。

#### 3.2.2 线程模型

LeaderElector 内部维护 3 个线程：

| 线程 | 职责 | 周期 |
|---|---|---|
| LeaderLock 线程 | 执行选主循环：未获锁时尝试竞选，已获锁时续约 | `loop_interval_ms`（建议 < `lease_ms / 10`） |
| LeaseCheck 线程 | 独立检查租约是否过期，作为 LeaderLock 线程的安全兜底：当 LeaderLock 线程因阻塞导致未能及时感知租约过期时，LeaseCheck 线程可独立触发降级 | `loop_interval_ms` |
| StateTransition 线程 | 消费状态转换任务队列，串行执行所有状态变更及用户回调 | 事件驱动（条件变量唤醒） |

StateTransition 线程是单线程串行执行的，保证 PROMOTING → LEADER 和 DEMOTING → FOLLOWER 的回调不会并发，简化上层逻辑。

#### 3.2.3 选主流程

```
LeaderLock 线程每次循环:

1. 检查是否已持有锁:
   ├─ 未持有 → CampaignLeader()
   │   ├─ 检查 forbid_campaign_time（降级冷却期）
   │   ├─ GetLockHolder 查看当前锁状态
   │   ├─ TryLock 尝试获锁
   │   ├─ 成功 → 入队 PROMOTING 任务
   │   └─ 失败 → 保持 FOLLOWER
   │
   └─ 已持有 → HoldLeader()
       ├─ RenewLock 续约
       ├─ 成功 → 更新 lease_expiration_time
       └─ 失败 → 入队 DEMOTING 任务
```

#### 3.2.4 节点端点信息

每个节点启动时将自身连接信息（node_id、host、各服务端口）写入 CoordinationBackend 的 KV 存储。任意节点获取到 leader_node_id 后，即可查询到 Leader 的完整连接方式，用于 Leader 发现。

## 4. Server 集成

### 4.1 初始化流程

Server 启动时通过 `CreateLeaderElector()` 完成 HA 初始化：

```
Server::Init()
  └─ CreateLeaderElector()
       ├─ 1. 解析 coordination_uri
       ├─ 2. 确定 host（advertised_host > NetUtil::GetLocalIp()）
       ├─ 3. 确定 node_id（配置指定 > 自动生成）
       ├─ 4. 创建 CoordinationBackend 并 Init
       ├─ 5. 创建 LeaderElector
       ├─ 6. 注册回调: OnBecomeLeader / OnNoLongerLeader
       └─ 7. WriteNodeInfo 将本节点端点信息写入 CoordinationBackend
```

### 4.2 角色变更回调

**OnBecomeLeader（晋升为 Leader）：**

1. Registry 数据恢复（从持久化存储加载配置）
2. 启动配置加载（首次启动时加载 startup_config）
3. CacheManager 数据恢复（重建缓存索引）
4. 恢复 CacheReclaimer（缓存回收线程）
5. 启用数据面请求

**OnNoLongerLeader（降级为 Follower）：**

1. 暂停 CacheReclaimer
2. 禁用数据面请求
3. 等待所有进行中的数据面请求完成（优雅关闭）
4. CacheManager 清理
5. Registry 清理

优雅关闭保证降级过程中不会出现请求被中途丢弃的情况。

## 5. Leader 发现

### 5.1 MetaService - GetClusterInfo

```protobuf
message GetClusterInfoResponse {
    CommonResponseHeader header = 1;
    string self_node_id = 2;       // 处理该请求的节点 ID
    string leader_node_id = 3;     // 当前 Leader 节点 ID
    NodeEndpoint leader_endpoint = 4; // Leader 的完整连接信息
}
```

- 任意节点均可处理。
- 适用于推理引擎等数据面客户端发现 Leader。

### 5.2 AdminService - GetManagerClusterInfo

```protobuf
message GetManagerClusterInfoResponse {
    CommonResponseHeader header = 1;
    int64 info_updated_time = 2;            // 信息更新时间
    int64 self_leader_expiration_time = 3;  // 自身租约过期时间（仅 Leader 有效）
    string self_node_id = 4;
    string leader_node_id = 5;
    NodeEndpoint leader_endpoint = 6;       // Leader 的完整连接信息
}
```

- 任意节点均可处理。
- 额外提供租约过期时间等运维信息。

### 5.3 NodeEndpoint 消息

```protobuf
message NodeEndpoint {
    string node_id = 1;
    string host = 2;
    int32 meta_rpc_port = 3;
    int32 meta_http_port = 4;
    int32 admin_rpc_port = 5;
    int32 admin_http_port = 6;
}
```

### 5.4 客户端 Leader 发现流程

```
1. 客户端持有任意节点的地址
2. 向任意节点发送 GetClusterInfo 请求
3. 从响应中获取 leader_endpoint
4. 使用 leader_endpoint 中的地址和端口直连 Leader
5. 若 Leader 变更（数据面请求被拒绝），重新执行步骤 2
```

客户端无需预先知道哪个节点是 Leader，可自主完成 Leader 发现和连接切换。

## 6. 配置参数

### 6.1 HA 相关配置

| 配置键 | 默认值 | 说明 |
|---|---|---|
| `kvcm.coordination.uri` | 空（单节点模式） | 协调后端 URI。为空时使用内存后端（单节点）。旧配置名 `kvcm.distributed_lock.uri` 向后兼容。 |
| `kvcm.leader_elector.node_id` | 自动生成 | 节点标识。不指定则自动生成（格式：`{host}:{admin_http_port}_{随机串}`）。 |
| `kvcm.leader_elector.lease_ms` | 10000 | 租约时间（毫秒）。 |
| `kvcm.leader_elector.loop_interval_ms` | 100 | 选主循环间隔。建议 < lease_ms / 10。 |
| `kvcm.service.advertised_host` | 空（自动检测） | 对外广告的主机地址。为空时使用 NetUtil::GetLocalIp()。 |

### 6.2 部署模式配置建议

**单节点（默认）：**

```
# 不配置 coordination.uri，使用内存后端
kvcm.leader_elector.lease_ms=600000
kvcm.leader_elector.loop_interval_ms=10000
```

**多节点 HA（文件锁，开发/测试）：**

```
kvcm.coordination.uri=file:///shared/lock/dir
kvcm.leader_elector.lease_ms=10000
kvcm.leader_elector.loop_interval_ms=100
```

**多节点 HA（Redis，生产）：**

```
kvcm.coordination.uri=redis://your_password@redis-host:6379?timeout_ms=5000&retry_count=3&cluster_name=kvcm_app_0
kvcm.leader_elector.lease_ms=10000
kvcm.leader_elector.loop_interval_ms=100
kvcm.service.advertised_host=10.0.0.1
```

Redis选主key通过`cluster_name`区分：`kvcm_<cluster_name>_lock:`/`kvcm_<cluster_name>_kv:`。

## 7. 故障场景分析

### 7.1 Leader 进程崩溃

1. Leader 停止续约，租约在 `lease_ms` 后过期。
2. Follower 的 LeaderLock 线程检测到锁未被持有，`TryLock` 成功。
3. 新 Leader 执行 OnBecomeLeader 回调，恢复数据后开始服务。
4. **故障转移时间** ≈ `lease_ms`（最差情况）。

### 7.2 Leader 主动降级（Demote）

1. 调用 `LeaderDemote` 接口，直接释放分布式锁并入队 DEMOTING 任务。
2. StateTransition 线程执行 OnNoLongerLeader 回调（优雅清理）。
3. 可选配置 `forbid_campaign_time_ms` 防止原 Leader 立即重新竞选。
4. Follower 竞选成功后成为新 Leader。

### 7.3 协调后端（Redis）短暂不可用

1. Leader 续约失败，但锁可能尚未过期。
2. 若续约持续失败直到租约过期，Leader 自动降级。
3. Redis 恢复后，Follower 尝试竞选。
4. **建议**：`lease_ms` 应大于协调后端的预期最大不可用时间，避免不必要的主备切换。

### 7.4 网络分区

由于采用单一协调后端（而非多数派投票），不存在脑裂风险：
- 能连接到协调后端的节点可以正常竞选和续约。
- 不能连接的节点无法续约，自动降级。
- 同一时刻最多只有一个节点持有锁（Leader）。

## 8. 模块目录结构

```
kv_cache_manager/
├── config/
│   ├── coordination_backend.h            # 协调后端抽象接口
│   ├── coordination_memory_backend.h/cc  # 内存实现
│   ├── coordination_file_backend.h/cc    # 文件锁实现
│   ├── coordination_redis_backend.h/cc   # Redis 实现
│   ├── coordination_backend_factory.h/cc # 工厂类
│   ├── leader_elector.h/cc              # 选主核心逻辑
│   └── node_endpoint_info.h             # 节点端点信息
├── common/
│   └── net_util.h/cc                    # 网络工具（GetLocalIp）
└── service/
    ├── server.cc                        # Server 集成层
    ├── meta_service_impl.cc             # GetClusterInfo 实现
    └── admin_service_impl.cc            # GetManagerClusterInfo 实现
```

## 9. 待完善功能

- **获取所有节点的能力**：当前只能通过 Leader 发现接口获取 Leader 信息，尚不支持列出所有存活节点。后续需要支持获取全部节点列表，用于运维监控和管理场景。
- **节点信息的过期清理机制**：节点端点信息写入 CoordinationBackend 后无 TTL，节点下线后其信息会残留。后续需要引入过期或心跳机制，自动清理已下线节点的信息。
