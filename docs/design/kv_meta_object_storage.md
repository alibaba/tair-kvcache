# KVMeta 变长对象存储设计

## 1. 文档定位

本文描述 KVCM 为 EPD（Encoder/Prefill/Decode）分离场景提供的 KVMeta V1 变长对象存储：为什么需要一条
独立链路、各组件如何协作、对象如何写入和回收，以及它如何与既有 KV cache 主链路隔离。

KVCM 的系统边界、普通 KVCache 主链路、HA 和 GC 职责划分见
[KVCM 总体设计与模块架构](module_architecture.md)；协议字段和逐 RPC 契约见
[KVMeta 通用对象 API](../api/kv_meta_service.md)。上层适配器分别见：

- v6d 仓库：`docs/tair-kvcm/kvcm-emb-storage.md`；
- RTP-LLM 仓库：`docs/multimodal/kvcm_emb_storage.md`。

### 1.1 仓库与交付边界

| 仓库层次 | 本功能的代码范围 |
|---|---|
| KVCM `github-opensource` | KVMeta 协议、独立服务、manager、object client、通用 exact-size SDK 路径和开源 backend/stub |
| KVCM 内部父仓 | 固定开源子仓版本，并为真实 TairMempool/PACE SDK 实现 variable-size policy |
| v6d | 提供可选 Python `KVCMEmbeddingStore`，不改变 v6d 原有 KV cache 传输 |
| RTP-LLM `github-opensource` | 提供可选 multimodal producer/receipt/reader/release/GC 和构建开关 |
| RTP-LLM 内部父仓 | 固定包含上述实现的开源子仓版本，不另复制一套 KVCM EMB 状态机 |

因此“开源 KVCM”和“内部 KVCM”不是两套 KVMeta：通用协议与状态机只有一份；内部仓只补充无法在开源仓实现的
PACE 数据面。部署时，各父仓必须固定到包含匹配协议和 client API 的子仓 revision。

### 1.2 结论先行与实现状态

当前 V1 是适合首轮部署验证的最小安全方案：它通过独立协议、namespace、请求门和 singleton allocation 支持变长
对象，并坚持“不能证明归属就不删除”，不会为回收率牺牲数据正确性。它没有修改固定 block 接口，也不会让 KV
cache 请求获取 KVMeta 的锁。

V1 不是最终形态。设计复盘后，后续重点不应是把 `vector<size>` 直接塞进共享 `DataStorageBackend::Create`，而应在
KVMeta 侧增加可选的异构对象 capability adapter，并用 generation-aware allocation 和持久化 retired cleanup
ledger 解决安全重试。这样既能降低 singleton 调用放大，也不会改变主链路 backend ABI/语义。第 13 节给出完整
演进设计。

| 层次 | 状态 | 能力与边界 |
|---|---|---|
| V1 | 当前代码已实现 | exact-size、singleton allocation、写租约、metadata-first 单次删除、HA 恢复 |
| V1.1 | 建议下一阶段，尚未实现 | KVMeta 独立 QoS、orphan 指标/审计、backend capability 探测；不改变协议语义 |
| V2 | 目标设计，尚未实现 | generation/fencing、幂等 operation、持久化 ledger、可续约读写 lease、异构批量和 object set |

文中没有标注“V1.1/V2”的流程均描述当前实现；演进章节中的接口和字段是设计草案，不能作为现有 API 使用。
长期看，EPD 对外应优先暴露“发布并租用一组 tensor chunks”的 object-set API；exact-key API 保留为底层通用能力和
小对象兼容入口。这样生命周期与一次推理请求对齐，而不是让业务自行拼接大量 key 的回滚和 GC 状态。

## 2. 背景、目标与非目标

### 2.1 为什么不能直接复用 KV cache 链路

既有 `MetaService + CacheManager + TransferClient` 面向模型 KV cache block：instance 注册时就确定
`location_spec_infos`，同一 spec 的 block 大小固定。Encoder/ViT 输出的 embedding、position id 和额外输入则是
opaque tensor byte stream，不同请求、不同 tensor，甚至同一批次中不同对象的字节数都可能不同。

如果把任意长度 embedding 塞进固定 block 链路，会同时破坏：

- `TransferClient` 对固定 block size 的校验；
- storage backend 对同批对象等长、共享物理文件或 allocation 的假设；
- CacheReclaimer、Migration 和 Cache GC 按 KV cache 语义维护容量与生命周期的假设。

因此本设计新增 KVMeta exact-key、exact-size 侧路，而不放宽原链路约束。

### 2.2 目标

- 同一请求可以读写不同字节数的对象，且 URI、元数据和 caller buffer 的长度严格一致；
- 以完整业务 key 精确寻址，即使一级哈希碰撞也不能串读、串删；
- 用 `PutStart`/`PutFinish` 写会话保证未完成对象不可读，并在失败、超时和换主后收敛；
- 对容量、请求规模、并发 session 和字符串长度设置明确上限；
- 为 C++/Python 上层提供组合好的 `KvMetaObjectClient`，同时保留底层元数据与数据面 API；
- 默认关闭；启用后保持独立端口、namespace、请求门、锁和后台维护，并明确共享进程/存储仍需部署 QoS 才能获得
  性能隔离。

### 2.3 V1 非目标

- 不提供自动 LRU/TTL 淘汰；对象由业务 release、`Remove` 或 `Trim` 回收；
- 不承诺一个多 key 写会话在并发 `Get` 看来具有同一瞬间的原子可见性；
- 不在 KVCM 中保存 tensor shape、dtype、切片顺序或 RTP receipt；KVCM 只保存 opaque bytes；
- 不改变 `DataStorageBackend::Create(keys, object_size)` 和普通 `TransferClient` 的固定大小接口；
- 不提供 server-side read lease。V1 调用方必须保证 consumer Load 完成前不触发同对象的 Remove/Trim/GC；
- 不保证取消已经进入不可取消 backend 的 I/O。KVMeta 会在返回前等待它结束，以保护 caller-owned buffer。

## 3. 总体架构

KVMeta 把一次对象访问拆成控制面和数据面：

- **控制面**：KVMeta gRPC 服务负责 instance、exact-key metadata、物理 allocation URI、写会话和回收；
- **数据面**：调用方根据服务端返回的 URI，通过 `KvMetaTransferClient` 直接读写实际 storage backend；
- **组合层**：`KvMetaObjectClient` 编排控制面事务和 exact-size 数据搬运，是推荐入口。

```mermaid
flowchart LR
    producer["Producer：Encoder / ViT"]
    consumer["Consumer：LLM / Decode"]
    adapter["RTP 或 v6d 适配器"]
    object_client["KvMetaObjectClient"]
    meta_client["KvMetaClient"]
    transfer_client["KvMetaTransferClient"]
    grpc["独立 KVMeta gRPC Server"]
    service["KvMetaServiceImpl"]
    manager["KvMetaManager"]
    registry["Registry / MetaIndexer"]
    storage_manager["DataStorageManager"]
    sdk["SdkWrapper::InitForKvMeta"]
    backend["NFS / HF3FS / Mooncake / TairMempool ..."]
    receipt["业务控制面 receipt\nkey + size + tensor metadata"]

    producer --> adapter --> object_client
    object_client --> meta_client --> grpc --> service --> manager
    manager --> registry
    manager -->|Create / Delete| storage_manager --> backend
    object_client --> transfer_client --> sdk -->|Get / Put exact bytes| backend
    adapter --> receipt --> consumer
    consumer --> object_client
```

KVCM 不传递图中的 receipt。RTP 自己在 ViT 与 LLM 之间传递 tensor metadata；v6d 也要求调用方在独立控制面
保存这些信息。RTP 和 v6d 是两个并列的客户端适配器，RTP 生产链路不依赖 v6d。

### 3.1 组件职责

| 组件 | 职责 | 不负责 |
|---|---|---|
| `KvMetaServiceImpl` / 独立 gRPC server | 请求门控、参数边界、错误码映射、响应形状校验 | 不直接搬运对象 bytes |
| `KvMetaManager` | exact-key metadata、allocation、动态 byte quota、写会话、HA 恢复、Remove/Trim | 不进入 KV cache 写入/淘汰流程 |
| `RegistryManager` / `MetaIndexer` | 保存 KVMeta 专用 instance 和 `CacheLocation` | 不理解 tensor 语义 |
| `DataStorageManager` | 选择 backend，创建/删除物理对象 | 不执行客户端 buffer 搬运 |
| `KvMetaClient` | metadata RPC、多地址 failover、响应校验 | 不搬运数据 |
| `KvMetaTransferClient` | 按 URI 和真实长度执行同步 Get/Put | 不管理写事务 |
| `KvMetaObjectClient` | 组合注册、Get、PutStart、数据搬运、PutFinish 和回滚 | 不保存 shape/dtype/receipt |
| RTP/v6d 适配器 | tensor 校验、切片、receipt 或 buffer 封装、业务生命周期 | 不改变 KVCM 一致性语义 |

### 3.2 代码落点

| 层次 | 主要实现 |
|---|---|
| 协议 | `kv_cache_manager/protocol/protobuf/kv_meta_service.proto` |
| 服务入口 | `kv_cache_manager/service/kv_meta_service_impl.{h,cc}`、`service/grpc_service/kv_meta_service_grpc.{h,cc}` |
| 启停与 HA 装配 | `kv_cache_manager/service/server.{h,cc}`、`server_config.{h,cc}` |
| metadata 与 session | `kv_cache_manager/manager/kv_meta_manager.{h,cc}`、`kv_meta_instance.h` |
| C++/Python client | `kv_cache_manager/client/include/kv_meta_*`、`client/src/kv_meta_*`、`client/kv_meta_object_client.py`、`client/pybind/py_client_binding.cc` |
| exact-size 数据面 | `kv_cache_manager/client/src/internal/sdk/sdk_wrapper.{h,cc}` 及各 backend SDK |
| 内部 PACE 适配 | 内部仓 `internal_source/kv_cache_manager/client/src/internal/sdk/tair_mempool_sdk.{h,cc}` |

主链路隔离的防护还落在 `cache_reclaimer.cc`、`cache_garbage_collector.cc` 和普通
`transfer_client_test.cc` 回归用例中。第 14 节给出当前实现的完整测试分层。

## 4. 与 KV cache 主链路的隔离

隔离是本设计的首要约束。V1 在代码路径、元数据和生命周期上提供强制隔离；CPU、网络、内存和物理 backend 的
性能隔离仍取决于部署方式，不能仅凭独立 RPC 端口宣称“对主链路零影响”。

### 4.1 启停与网络隔离

- `kvcm.kv_meta.rpc_port=0` 为默认值。此时不创建 `KvMetaManager`、session expiry worker 或额外 gRPC server；
- 非零端口启用独立 gRPC server 和请求计数器，不向既有 MetaService 增加 RPC；
- KVMeta 端口必须与主 RPC/HTTP、Admin RPC/HTTP 和 Debug HTTP 端口不同。

独立端口是故障与部署边界，不是身份认证。V1 沿用 KVCM 的受信网络模型，知道 instance/key 的 client 可以调用
Get/Remove；生产环境必须用网络策略限制端口来源。V2 ownership token 仍需绑定经过认证的 tenant/instance，不能
用随机 key 或 token 猜测难度代替服务认证。

### 4.2 元数据与容量隔离

- 公共 `instance_id` 会编码为保留的 KVMeta 内部 instance id，并携带完整 schema marker；
- KVMeta instance 必须放入专用 Instance Group。注册时若 group 已含普通 instance，服务端拒绝；
- 如果之后通过普通接口向该 group 混入 KV cache instance，新的 KVMeta byte admission 会 fail closed；
- 普通 CacheReclaimer、Migration 和 Cache GC 跳过 KVMeta instance；
- KVMeta 使用对象真实字节数维护 group/type quota，不把 marker `block_size=1` 当作对象用量。

### 4.3 代码路径隔离

- `KvMetaManager` 不调用 `StartWriteCache` / `FinishWriteCache`；
- 变长策略只由 `KvMetaTransferClient` 调用 `SdkWrapper::InitForKvMeta` 时开启；
- KVMeta 初始化会克隆 wrapper/backend config 后写入 variable-size policy，避免复用同一 `ClientConfig` 的普通
  `TransferClient` 通过指针别名看到该策略；
- 普通 `TransferClient` 保留原来的固定大小校验、阻塞提交、超时和 backend fallback 行为。

### 4.4 Leader 生命周期隔离

升主时先完成 Registry/CacheManager 的既有恢复、启动 GC/Migration 并开放主服务，再在独立线程恢复 KVMeta。
恢复成功且 session worker 启动后才开放 KVMeta 请求门。KVMeta 恢复失败不会阻止主服务可用。

降主或 Stop 时先关闭 KVMeta 请求门、session 准入并取消 Trim，然后按既有顺序排空主服务请求、停止主 GC 和
Migration，最后 join KVMeta recovery/session worker。内存 session 被丢弃而不是在主清理线程逐个做 storage I/O；
残留 active metadata 由下一任 leader 恢复。

### 4.5 性能与故障域边界

| 资源 | V1 已隔离 | 仍可能共享的部分 | 生产建议 |
|---|---|---|---|
| RPC | 独立 gRPC server、端口和请求计数 | 同一进程 CPU、内存、调度 | 限制 KVMeta RPC 并发；高负载时使用独立进程/cgroup |
| 锁 | KVMeta group admission shard 只由 KVMeta 获取 | Registry/MetaIndexer 的底层实现 | KVMeta 使用专用 Instance Group 和 metadata namespace |
| 后台线程 | recovery、session expiry 独立，失败不阻塞主服务放流 | 进程线程数和 CPU quota | 设置独立线程/队列上限，监控 gate 和积压 |
| 容量 | 按真实 bytes 使用专用 group/type quota | backend 的真实总容量 | storage candidate 指向专用 pool/namespace |
| 数据面 | 使用独立 `KvMetaTransferClient` 策略 | NIC、PCIe、SDK connection、存储设备 | 配置 backend 侧带宽/IOPS 限流；强隔离部署使用独立资源池 |

因此部署分两档：

- **兼容部署**：KVMeta 与 KV cache 共进程，但必须使用专用 group、namespace、并发上限和存储 quota；适合灰度；
- **强隔离部署**：KVMeta 使用独立 KVCM 进程/资源组及独立 storage pool；即使复用协调服务或 Registry 的物理
  集群，也必须使用独立 namespace/leader lock，不能与 KV cache 实例争抢同一个 leader 身份；适合大规模 EPD
  流量。

当前代码已经具备兼容部署的功能隔离，但尚未提供完整的 KVMeta 独立限流器或 `kvmeta_only` server role，这两项属于
V1.1，而不是当前能力。

## 5. 对象与元数据模型

### 5.1 对外对象标识

一个对象由 `(instance_id, key)` 唯一标识。两者在协议中均为 protobuf `string`；KVCM 按序列化后的字节长度
执行边界检查，不解析 key 的业务结构。

### 5.2 exact-key 与哈希碰撞

`MetaIndexer` 的一级 key 是 `int64`，因此 KVMeta：

1. 将完整业务 key 哈希为一级 key；
2. 将完整 key 可逆编码进稳定 `location_id`；
3. 所有 Get、条件更新和删除同时匹配一级 key 与完整 `location_id`。

同一请求里一级哈希相同的不同 key 会拆到不同 metadata layer。恢复和 Trim 也会反解 `location_id` 并重新核对
完整 key 与哈希。因此哈希碰撞最多增加一次分层访问，不能造成错误命中或误删。

### 5.3 `ValueLocation`

每个可读对象的位置满足以下不变量：

| 字段 | 约束 |
|---|---|
| `type` | 必须是已注册、可识别的 storage type |
| `spec_size` | 固定为 `1` |
| `location_specs` | 恰好一个元素，名字固定为 `value` |
| `location_specs[0].uri` | scheme/hostname 必须对应所选 backend |
| `value_size` | 对象有效字节数，必须大于 0 |
| URI `size` 参数 | 必须与 `value_size` 完全相等 |

服务端、metadata client、object client 和 transfer wrapper 都会独立验证这些不变量；任一层发现损坏都 fail closed，
不会把不可信 URI 交给数据面。

### 5.4 对象状态

KVMeta 复用 `CacheLocation` 的存储格式，但不复用 KV cache 状态机：

- **active**：`status=CLS_NEW` 且 `create_time` 为带 tag 的正数，编码写租约 wall-clock deadline；Get 不可见；
- **committed**：仍为 `status=CLS_NEW`，但 `create_time` 为负数；Get 可见；
- **absent**：metadata 不存在。

滚动升级时，旧版本遗留的无 tag 正 marker 按“创建时间 + `max_write_timeout_seconds`”推导保守截止时间。
HA 节点必须保持时钟同步，并把可能的最大漂移计入写租约配置。

## 6. 端到端流程

### 6.1 注册

`KvMetaObjectClient::Create` 首先通过 `KvMetaClient::RegisterInstance` 注册专用 instance，取得服务端权威
`storage_configs`，再创建 exact-size transfer client。相同 instance/group/schema/user data 的注册幂等；身份或
schema 不一致则失败。

### 6.2 写入

```mermaid
sequenceDiagram
    participant A as Adapter / Producer
    participant O as KvMetaObjectClient
    participant M as KVMeta Service / Manager
    participant S as Storage Backend

    A->>O: SaveObjects(keys, sizes, buffers)
    O->>O: 校验完整 batch、key、size、buffer
    O->>M: PutStart(keys, value_sizes, write_timeout)
    M->>M: 查询 exact key、检查动态 byte quota
    M->>S: 对每个 miss singleton Create(size)
    M->>M: 条件写入 active metadata + Sync
    M-->>O: key_mask + compact locations + session
    O->>S: 仅向 miss location 写 exact bytes
    O->>M: PutFinish(session, all true)
    M->>M: 条件 commit + Sync
    M-->>O: OK
```

关键语义：

- `key_mask` 与原始 keys 等长；`true` 仅表示同尺寸对象已经 committed；
- `locations` 只包含 `key_mask=false` 的 miss，按它们在请求中的相对顺序紧凑排列；
- 全部命中时不创建 session，也不返回 location；
- 任一 key 已 active 时整批返回 `WRITE_IN_PROGRESS`；已 committed 但尺寸不同则返回 `SIZE_MISMATCH`；
- object client 会额外用 `Get` 确认 masked hit 已可读，以兼容已经具备 V1 字段、但仍保留早期 active-mask
  行为的滚动升级版本；这不表示缺少 V1 新字段的原始 proto 实现可以混用；
- 数据写入返回的实际 URI 必须与服务端给出的 URI 完全一致，否则整批回滚；
- `PutFinish.success_keys` 与紧凑 `locations` 对齐。任一 `false` 会回滚本 session 的全部新对象；
- commit/rollback 逐 key 执行并带失败补偿，不承诺多 key 同时可见。

### 6.3 读取

`LoadObjects` 先校验完整请求，再执行一次 request-aligned `Get`。只有所有 key 均命中、所有 location 和期望 size
均一致时才分派数据 I/O；任一 miss 或异常 location 都不会产生部分读取。

V1 `Get` 返回的是 location snapshot，不会在服务端创建 read lease 或 pin allocation。object client 可以保证本次
Load 返回前 caller buffer 不被后台 I/O 继续访问，但不能阻止另一个 client 同时 Remove/Trim 该 URI。上层必须用
ownership/release 协议协调；RTP 的约束是 consumer 完成后才 release，且 ViT GC timeout 必须覆盖最慢读取。

### 6.4 删除与 Trim

- `Remove(keys)` 精确删除 committed metadata，`Sync` 后再删除物理 allocation；不存在的 key 幂等成功；
- 任一 key 仍 active 时，整批 `Remove` 返回 `WRITE_IN_PROGRESS`，不产生删除副作用；
- `Trim(TS_REMOVE_ALL_CACHE)` 删除 metadata 和可归属的物理对象；
- `Trim(TS_REMOVE_ALL_META)` 只删 metadata，物理数据保留，仅用于明确的修复场景；
- `TS_TIMESTAMP` 在 V1 中不支持；
- 存在 active 或正在 finalization 的 session 时，Trim 整体返回 `WRITE_IN_PROGRESS`。

## 7. 不同 value size 的实现

既有 backend 接口一次 `Create(keys, object_size)` 只能接收一个 size。KVMeta 不修改它，而是为每个实际 miss
执行 singleton `Create`：

```text
PutStart keys       key-a       key-b       key-c
value_sizes          1536        4096         768
                         \          |          /
singleton Create   Create(1536) Create(4096) Create(768)
```

这样既允许同一请求内尺寸不同，也避免某些文件型 backend 把多个 key 打包进同一物理文件后，按 key 删除时误删
相邻对象。每一代 allocation 使用随机物理 object key；同一业务 key 的新写入不会复用上一代 URI。

singleton 是兼容性选择，不是理想吞吐模型。一个 64-object miss 最多触发 64 次控制面 Create 和 64 个数据面
singleton 任务；它保证行为明确，但在高 QPS、小对象场景会放大 RPC、allocator 和线程调度开销。第 13.2 节的
KVMeta capability adapter 允许支持方一次接收不同 size，同时让不支持方继续安全回退到当前路径。

数据面仍可并行处理多个对象，但 wrapper 会把每个对象作为 singleton SDK 调用，逐项验证：

- `value_sizes[i] == URI.size == sum(buffer[i].iovs[*].size)`；
- IOV 非空、非零、不 ignored，地址非空，memory type 只能是 CPU/GPU；
- URI 合法、hostname 已注册、backend scheme 与 metadata type 一致；
- 整个 batch 在任何数据 I/O 前完成校验。

`MemoryType::GPU` 还要求实际 client/backend 以 CUDA 或 MUSA 能力构建。以开源 `LocalFileSdk` 为例，CPU-only
构建会在 allocation、mmap 和数据搬运之前返回 `ER_UNSUPPORTED_MEMORY_TYPE`；不能跳过 device copy 后仍返回
成功。RTP 的 GPU ViT 镜像因此必须使用 GPU-enabled KVCM client artifact，CPU artifact 只用于 CPU contract
test。

### 7.1 内部 TairMempool/PACE 适配

KVCM 内部仓的真实 `TairMempoolSdk` 在 variable-size policy 开启时：

- 接受 `0 < URI.size <= max_object_bytes`，不再要求 size 命中固定 spec 表；
- 必须使用服务端预分配的 PACE 地址；地址无效时直接失败，不在 client 侧重新 allocation；
- 禁用 gather/scatter 分组，按 IOV 原顺序构造一个连续对象，避免按 size 分组改变逻辑 offset；
- `actual_remote_uris` 保持与输入 URI 逐项一致。

普通固定 block 模式继续使用原 size 表、lazy allocation fallback 和既有 gather/scatter 行为。开源仓中的
TairMempool 是无真实 PACE 依赖的 stub，只保留严格、无异常的 URI 字段解析；实际 TairMempool I/O 必须使用
内部构建。

## 8. 并发、一致性与容量

### 8.1 并发写入

KVMeta 按 Instance Group 分片加 admission lock，把“读取实际 usage、选择 backend、allocation、metadata
reservation”串在同一容量准入临界区内，防止不同尺寸并发写超出 group/type quota。该锁只属于 KVMeta 侧路，
普通 KV cache 不获取。

metadata reservation 使用完整旧值条件保护。跨进程 `PutStart` 竞争失败后，会重新读取赢家并验证 exact key、
状态、backend、URI 和 size：只有相同尺寸的 committed 对象可视为命中；active 赢家返回
`WRITE_IN_PROGRESS`，本请求的候选 allocation 被回收。

### 8.2 Remove/新一代写入的 ABA 防护

对 committed 对象，`Remove` 从条件删除 metadata、持久化到物理删除结束一直持有同一 group admission shard。
下一代同 key `PutStart` 只能在旧物理对象删除完成后进入，避免可复用地址型 backend 的旧 Delete 误伤新对象。

session timeout 和 `PutFinish` finalization 同样计为 in-flight。Trim 不能与它们同时删除相同 allocation。

### 8.3 固定上限

以下是生产默认 `KvMetaManager::Limits`，客户端以相同或更严格的值预校验：

| 项目 | 上限 |
|---|---:|
| 每 RPC keys | 64 |
| 单 key | 512 bytes |
| instance id / instance group / write session id | 各 512 bytes |
| `user_data` | 64 KiB |
| 单 value | 1 GiB，且不能为 0 |
| 单 batch value 总量 | 4 GiB |
| 每 KVCM 进程 active write sessions | 4096 |
| 单写会话 timeout | 1800 秒 |

`PutStart` 在 allocation 前预检 Instance Group 总容量、storage type 容量和 active session 可用性。session 在最终
登记时还会做一次原子检查；若这一步因并发达到上限，服务端会删除本次 reservation 和候选 allocation，再向调用方
返回失败，不会返回一个不可管理的写会话。

## 9. 超时、失败与 failover

### 9.1 写租约

写租约从服务端开始处理 `PutStart` 时计时，覆盖 allocation、active metadata 持久化、数据面 Put 和
`PutFinish`。如果 session 登记前租约已经耗尽，服务端回滚候选 allocation 并返回超时。

`KvMetaObjectClient` 初始化时要求：

```text
write_timeout_seconds * 1000
    > put_timeout_ms + 3 * metadata.call_timeout_ms
```

三个 metadata 窗口分别预留给 PutStart 交接、masked-hit 兼容性 Get 和 PutFinish。该检查只证明名义预算可行，
不是 backend 的强制取消保证。

V1 的正确性前提是 backend 在 write lease 到期前停止访问该 remote allocation。SDK 为保护 caller buffer 会 drain
已经开始且不可取消的 I/O，但 drain 不会自动续约服务端 session；若 provider 无视自己的 timeout 并越过 write
lease 继续 Put，expiry 物理删除可能与旧 Put 竞争。生产 backend 必须证明 I/O 有硬 deadline/cancellation，或把
write lease 配置为覆盖经过验证的最坏 drain 时间；仅满足上面的名义不等式不够。

回滚和 expiry 始终先完成 exact-value metadata 删除及 `Sync`，然后只尝试一次物理 Delete。expiry worker 会把
provider 的标准/未知异常收敛为脱敏告警并继续处理后续 session，不让可选 KVMeta 侧路异常终止进程。物理删除
失败后不进入进程内重试队列，因为现有可复用地址型 URI 没有 allocation generation；不确定结果若被重放，可能
删除已经复用同一地址的后继对象。这里选择可运维回收的 orphan，而不是数据破坏。

### 9.2 caller buffer 生命周期

exact-object worker 使用非阻塞入队，因此 `sdk_config.queue_size` 必须至少为 64。队列压力导致部分任务无法接纳
时，wrapper 会停止尚未开始的任务并等待已接纳任务结束后返回。

到达数据面 deadline 后，排队任务不再发起 I/O；已经运行的 backend I/O 若不可取消，KVMeta 会安全 drain。
因此调用耗时可能超过名义 timeout，但返回后 backend 不再访问 caller-owned buffer。普通 TransferClient 保留
原有超时行为。

### 9.3 多地址与结果不确定

`KvMetaClient` 支持最多 64 个去重 endpoint，单地址最长 1024 bytes，单次调用的总 `call_timeout_ms` 最大
600000ms。成功 endpoint 会成为后续请求的首选。

- `Get`、`GetInstanceInfo` 和同配置的幂等 `RegisterInstance` 遇到 transport error 可以尝试下一地址；
- 所有 RPC 收到服务端明确的 not-leader/not-ready 响应时可以 failover；
- `PutStart`、`PutFinish`、`Remove`、`Trim` 遇到 transport error 时不自动重放，因为无法判断服务端是否执行；
- C++ client 用 `ER_INVALID_GRPCSTATUS` 表示这类不确定结果。调用方必须查询或审计，不能盲目重试 mutation。

未提交 active allocation 最终由 session timeout 或下一任 leader 的 recovery 清理。

## 10. HA 恢复

KVMeta recovery 只扫描带完整 KVMeta schema 的保留 namespace，并执行：

1. 分批扫描 metadata；
2. 对未到期 active lease 保持请求门关闭并等待，等待可被降主/Stop 以不超过 100ms 粒度取消；
3. 对已过期、归属可确认的 active metadata 做条件删除并持久化，再对 allocation 做一次物理清理；
4. metadata 阶段失败则保持 KVMeta 请求门关闭；物理清理失败只产生脱敏 orphan 告警，不重放不确定 Delete；
5. 完成一个无 metadata 删除、无 metadata 错误的稳定扫描后，按 committed URI 的真实 `size` 重建 KVMeta
   byte usage；
6. 启动 session expiry worker，最后开放 KVMeta 请求门。

一次 recovery 从升主开始最多按 `max_write_timeout_seconds` 等待 active lease；损坏或异常远期的持久化 deadline
不能无限阻塞 KVMeta 侧路恢复。

恢复删除使用完整序列化旧值做条件保护；若无法证明 metadata 或 URI 属于当前 KVMeta 对象，宁可保留形成 orphan，
也不会冒险删除可能已被新一代引用的数据。

Trim 和 recovery 每 1000 个 key 检查取消，并把物理删除拆成最多 256 个对象的批次，避免大 namespace 长时间阻塞
降主。

## 11. 配置与部署

### 11.1 服务端

```text
kvcm.kv_meta.rpc_port=<独立端口>
```

除此之外，KVMeta 复用现有 Registry、MetaIndexer、Instance Group quota 和 storage backend 配置。部署必须提前
创建仅供 KVMeta 使用的 Instance Group。

### 11.2 对象客户端

推荐配置 `KvMetaObjectClientConfig`：

- `metadata.addresses`、`metadata.instance_id`、`metadata.call_timeout_ms`；
- `instance_group`、`user_data`；
- `transfer_client_config`；
- `transfer_init_params.role_type=WORKER`；
- `transfer_init_params.self_location_spec_name="value"`；
- `max_object_bytes`、`write_timeout_seconds`；
- Mooncake 等需要 worker memory span 的 backend，通过 `transfer_init_params.regist_span` 提供 base/size；
- TairMempool/PACE 等需要共享内存映射时，通过 `SharedMemoryRegistration` overload 提供 base/size/fd。

transfer JSON 必须同时满足：

```json
{
  "instance_group": "<与 metadata 注册一致>",
  "instance_id": "<与 metadata 注册一致>",
  "block_size": 1,
  "location_spec_infos": {"value": 1},
  "sdk_config": {
    "queue_size": 64
  }
}
```

`block_size=1` 和 `value=1` 只是隔离 schema marker，不代表对象固定为 1 byte；真实长度始终来自每次请求的
`value_sizes` 和 `ValueLocation.value_size`。

要求 caller-side memory registration 的 backend 必须使用与其注册方式匹配的客户端构造参数。RTP 当前适配器既
没有提供 `regist_span`，也没有调用 shared-memory overload，因此不能选择要求该能力的 transfer 配置；v6d 暴露了
`memory_base/memory_size/fd`。

## 12. 物理回收和 V1 运维限制

metadata 删除先 `Sync`，再调用 backend Delete，确保仍可读 metadata 不会指向已经提前释放的 URI。该顺序适用
committed Remove/Trim，也适用 active rollback、expiry 和 recovery。若 metadata 已经持久化删除、随后物理
Delete 返回错误或抛异常，该 URI 已成为不可达 orphan：同步 API 返回错误，expiry 记录一次脱敏告警，recovery
则继续完成稳定扫描和 byte usage 重建。三条路径都不自动重放不确定 Delete，因为现有 backend URI 没有
allocation generation，地址复用后重放旧删除可能破坏后继对象。运维需要依赖 backend 的 orphan 清理策略。

KVMeta 自己的 storage wrapper 会把 Create/Delete provider 的标准异常、未知异常和 Delete 结果数量不匹配转换为
明确错误码，防止异常越过可选侧路终止服务线程。批量 PutStart 的后续 singleton Create 抛异常时，已经取得 URI
的前序候选会做一次补偿删除；抛异常的调用若在 provider 端产生了未返回 URI，仍按 orphan 处理，不猜测重试。

backend 的物理回收能力沿用现有实现。例如当前开源 NFS backend 的 Delete 是幂等 no-op；KVMeta 不为它改变
共享行为。生产上需要结合显式 release、超时清理、namespace 轮换或 `Trim(TS_REMOVE_ALL_CACHE)` 管理容量。

V1 也没有 read lease：管理面 Remove/Trim、业务 release 或配置过短的上层 GC 都可能与已经取得 URI 的 Load
竞争。部署必须把 release 放在消费完成之后，并把全量 Trim 当作需要先排空 consumer 的维护操作；仅等待 KVCM
请求计数归零并不能观察客户端已经开始的数据面读取。

V1 在 metadata 删除时同步扣减 usage。若后续物理 Delete 失败，orphan 已不可寻址，也不再计入 KVMeta quota，
因此 metadata usage 仍然准确，但可能低于 backend 实际占用。没有 generation token 时保留 URI 并自动重放同样
不安全；V1 选择记录脱敏告警并交给 backend/namespace 回收。V2 用持久化 cleanup ledger 保留安全清理所需身份，
并在物理删除成功前继续计费，从根本上解决这项容量漂移。

## 13. 更优的演进设计（尚未实现）

### 13.1 方案选择

演进实现必须始终满足以下不变量；吞吐、回收速度和兼容便利不能覆盖这些条件：

1. 普通 KV cache 的 proto、backend ABI、fixed-block 校验、GC/Migration 和默认资源预算不因 KVMeta 能力而改变；
2. `Get` 只能发布已经 committed、尚未 retired 且与返回 allocation identity 一致的对象；
3. 未证明旧读写 I/O 已终止时，不得物理回收或复用对应 allocation；
4. mutation 结果不确定时，只有具备持久化 operation id 和 backend 幂等/条件执行契约才允许自动重放；
5. allocation 未被 backend 证明回收前，其 `allocated_size` 必须继续占用配额；
6. capability 缺失、降级、返回错序或版本不匹配一律 fail closed，不能在 mutation 之后切换 fallback。

| 方案 | 优点 | 主要问题 | 结论 |
|---|---|---|---|
| 放宽固定 KV cache block size | 表面改动少 | 破坏 TransferClient、backend packing、GC/Migration 假设 | 拒绝 |
| 把 `Create(keys, sizes)` 加到共享 backend 接口 | 能批量变长 allocation | 扩大公共 ABI 和主链路回归面，所有 backend 都要理解新语义 | 不作为演进主线 |
| 保持 V1 singleton | 风险最小、兼容全部 backend | 调用放大，Delete 失败只能形成 orphan | 保留为 fallback |
| KVMeta capability adapter + generation | 只影响侧路，可批量、可安全重试 | 需要内部 backend/SDK 配合 | 推荐终态 |

推荐方案继续复用 KVCM 的 registry 和 backend 配置，但把“变长批量、allocation 身份、条件删除”封装在 KVMeta 专用
adapter 中。普通 `DataStorageBackend`、`TransferClient`、CacheReclaimer、Migration 和固定 block proto 不读取这些
capability，也不改变既有调用。

实现上由 `KvMetaManager` 持有独立的 adapter registry。只有启用非零 KVMeta 端口时才创建 adapter：legacy adapter
委托现有 `DataStorageManager` 做 singleton Create/Delete；支持方通过独立 target/factory 提供新接口。不要给共享
`DataStorageBackend` 追加必选 virtual method，也不要让普通 selector 返回或缓存 KVMeta capability。

### 13.2 KVMeta 专用 storage capability

概念接口如下，名称仅用于表达契约：

```cpp
struct KvMetaObjectSpec {
    std::string object_key;
    std::uint64_t logical_size;
};

struct KvMetaAllocation {
    DataStorageUri uri;
    std::string allocation_id;   // backend 生成、不可复用
    std::uint64_t logical_size;
    std::uint64_t allocated_size;
};

CreateObjects(specs) -> vector<KvMetaAllocation>
DeleteObjectsIfMatch(vector<{uri, allocation_id}>) -> per-object result
```

adapter 对 backend 暴露显式 capability：

- `BATCH_VARIABLE_SIZE`：一次请求可接收不同 `logical_size`，并保证每项独立所有权/删除；
- `IDEMPOTENT_CREATE`：相同 backend operation id 重放时返回同一组 allocation，不重复分配；
- `GENERATION_DELETE`：删除同时匹配 URI 与不可复用 `allocation_id`，并返回可判定的逐对象终态；
- `HARD_IO_DEADLINE`：deadline 后 backend 保证不再访问 caller buffer 或 remote allocation；
- `FENCED_IO`：Get/Put 也校验 allocation generation 或 lease token；
- `ASYNC_DELETE_STATUS`：可查询不确定删除的最终状态。

服务端只在 capability 完整且返回数量、顺序、size、backend identity 全部通过校验时使用批量路径，否则在 allocation
前回退到 V1 singleton。已经发出批量 mutation 后结果不确定时不能切换路径重试。

条件删除的结果至少区分 `DELETED`、`TARGET_ABSENT`、`IDENTITY_MISMATCH` 和 `OUTCOME_UNKNOWN`。前三者只有在
backend 能证明目标 `allocation_id` 已不再占用物理资源时，才允许移除 ledger entry 并释放配额；
`OUTCOME_UNKNOWN` 必须保留 ledger 与配额并按相同 operation/allocation identity 查询或重试。只返回布尔值、仅凭
URI 不存在，或把 provider timeout 当作删除成功，都不满足 `GENERATION_DELETE` capability。

capability 必须按 `storage_name + config_epoch` 协商和持久化，不能当作整个 Instance Group 的静态布尔值。一次
allocation/session 使用创建时的 capability snapshot；运行中 backend 配置变更或 capability 降级时 fail closed，
不能让同一 generation 的 Put、Get、Delete 分别按不同契约执行。

对于文件/对象存储，随机且永久不复用的 object key 可以作为 `allocation_id`；对于 TairMempool/VCNS 这类可复用
地址，token 必须由 allocator 生成并由 SDK 在 I/O/Delete 时校验，不能由 KVCM 根据 URI 自行猜测。

V2 wire format 继续把现有 `ValueLocation.value_size` 解释为 `logical_size`，并以 additive field 增加
`allocated_size` 和 opaque `allocation_id`；不能继续假设 URI `size` 等于业务长度，也不应把
fencing/ownership token 拼进容易被普通日志记录的 URI。新 client 只有协商到对应 protocol/capability 后才解析
这些字段，旧 client/server 组合继续 fail closed。

V2 必须满足 `0 < value_size <= allocated_size`。caller buffer 和 checksum 只覆盖 `value_size`；对齐产生的 padding
由 backend 管理，Get 不得把 padding 拷回调用方，避免读取未初始化字节或把物理分配粒度误当成 tensor 长度。

### 13.3 专用记录和持久化清理账本

V2 不再用 `create_time` 正负号长期承载对象状态，而使用版本化的 KVMeta record。为了避免物理删除故障长期阻塞同
key 的新一代对象，record 不应只有一个 `DELETING` 状态，而应包含一个可选 current head 和有界 retired ledger：

- `schema_version`、完整 key identity；
- `head`：当前 `RESERVED` 或 `COMMITTED` generation，可为空；
- `retired_allocations`：等待物理回收的旧 generation；
- 每个 generation 保存 `logical_size`、`allocated_size`、storage type、URI、`allocation_id`；
- active head 额外保存 `write_session_id`、lease deadline、leader epoch；
- retired entry 保存安全级别、删除 attempt、最后错误类别和下一次调度时间；不保存 provider 异常原文。

下图描述单个 generation 的生命周期；同一 exact record 可以同时包含一个新 head 和若干旧 retired entry。

```mermaid
stateDiagram-v2
    [*] --> RESERVED: allocation + head Sync
    RESERVED --> COMMITTED: data Put success + commit Sync
    RESERVED --> RETIRED: abort / lease expiry / recovery
    COMMITTED --> RETIRED: Remove / Trim
    RETIRED --> [*]: conditional physical delete + ledger Sync
    RETIRED --> UNSAFE: legacy delete outcome uncertain
    UNSAFE --> [*]: operator/backend confirms cleanup
```

`Get` 只返回 committed head，并为高层 object client 签发有界 read lease；object-set 场景优先使用一份 set-level
consumer lease，避免逐对象 RPC。删除时在一次 exact-record CAS 中清空 head、把相同 allocation 移入 retired
ledger 并 `Sync`；从这一刻起不再接纳新读取，但已有 lease 仍受保护。worker 等 lease 释放或过期后再执行
`DeleteObjectsIfMatch(uri, allocation_id)`，成功后才移除 retired entry 并释放 `allocated_size` quota。返回错误、
进程崩溃或换主时，下一任 worker 从 ledger 继续，而不是从日志猜测 URI。

location snapshot 与 read lease 必须由同一个 `GetAndAcquire`/`AcquireObjectSet` 线性化操作返回，不能先 Get 再单独
加 lease，否则两次 RPC 之间仍存在 Remove 释放 allocation 的窗口。lease release/renew 使用 operation id 幂等，
client crash 则由有界 expiry 收敛。

generation-aware backend 可以在 ledger 持久化后允许同 key 创建新 head，因为旧 entry 的条件删除不可能命中新
generation。若 metadata backend 将清理账本做成独立 key，则“移除 head + 创建 ledger entry”必须有原子事务或
WAL；当前 MetaIndexer 没有跨 key 事务保证时，优先把有界 retired list 放在同一个 exact record 中。每 key、每
instance 的 retired entry/bytes 都必须有硬上限，达到上限时新 allocation 在 mutation 前失败，不能无限增长记录。

V2 adapter 遇到不支持 `GENERATION_DELETE` 的 legacy backend 时，仍沿用 V1 的 group lock 和单次物理 Delete；
结果不确定时把身份记为 `UNSAFE`，之后不得由服务自动重放。该 entry 继续计费，直到 namespace 轮换或 backend
工具确认回收，防止 orphan 被当作空闲容量继续超卖。因为不会再发出旧 Delete，后续新 generation 不会被 KVCM
的重试误伤。

### 13.4 重试、活跃写和 fencing

generation-aware Delete 解决“旧 Delete 误删已复用地址”的 ABA 问题，但它本身不能阻止旧 producer 在 allocation
释放后继续向同一地址写数据。因此：

- committed 对象只有在已签发 read lease 排空后，才能用 `GENERATION_DELETE` 安全重试物理删除；generation 防止
  误删后继 allocation，read lease 防止破坏仍在进行的当前 generation Load；
- active allocation 在 lease 到期前仍不得回收；
- 若希望新 leader 不等待旧 lease，backend 必须同时支持 `FENCED_IO`：新 leader 先使旧 generation/leader epoch
  失效，后续 Put 被 backend 拒绝，再进入条件删除；
- client 在数据 I/O 仍活跃时通过幂等 `RenewWriteLease`/`RenewReadLease` 续约，object client 负责在实际 drain 完成后
  停止续约；服务端限制单次扩展和累计租约，防止失控保活；
- 只有 `HARD_IO_DEADLINE` 或 `FENCED_IO` 能证明旧 I/O 已停止时才能物理回收；两者都不具备时即使 lease 到期也
  只能进入隔离/人工确认状态，不能仅凭换主或 wall-clock 提前释放地址。

leader epoch 必须由协调层单调签发，并参与 record CAS；仅使用进程内 group lock 不能处理网络分区下的双 leader。
具备 `FENCED_IO` 时同一 epoch 还要传到 backend，由 backend 拒绝旧 epoch 的 Put/Delete。legacy backend 无法提供
该保证时，系统仍依赖单 leader 协调和租约等待，文档与指标必须把它标为较弱故障域，不能宣称完成端到端 fencing。

安全重试还必须区分“服务端内部重试”和“客户端重放 RPC”：

- 内部 cleanup worker 使用 ledger 中固定的 `allocation_id`，只重放条件删除；
- `CreateObjects` 只有在 backend 声明 `IDEMPOTENT_CREATE` 且使用持久化 operation id 时才能重放；
- V2 `PutStart` 应接收 client-generated `operation_id`，同一 id 返回同一 session/allocation，terminal result 在有界
  retention 内可查询；
- `PutFinish` 的 committed/aborted terminal result 也按 session/operation id 保留；响应丢失后的重复 Finish 返回同一
  结果，不能退化为无法区分“已完成”和“从未存在”的 `SESSION_NOT_FOUND`；
- V2 `Remove` 必须携带 `operation_id + expected_allocation_id`，或改为使用 object-set ownership token。只按 key
  重放可能删除第一次请求之后创建的新 head，仍然不安全；
- 大范围 `Trim` 应创建可查询、可取消的 maintenance job 并返回 `job_id`，重试查询同一 job，而不是再次启动扫描；
- 不具备上述字段的 V1 mutation 继续保持“transport outcome 不确定时不自动重放”。

idempotency/terminal-result record 必须持久化、按 namespace 限额并有大于 client 最大重试窗口的 retention；过期后
返回明确的 `OUTCOME_UNKNOWN`，不能悄悄执行成一笔新 mutation。

所有 retry 使用有界指数退避、jitter、每 backend 并发上限和熔断器。错误日志只记录 backend type、错误类别、数量、
attempt 和延迟；key、URI、endpoint、credential 只允许进入受权限保护的审计存储，不能进入普通日志。

### 13.5 Object set、manifest-last 与服务端 lease

RTP 一个 receipt 可能跨多个 64-object 写事务。V1 依赖客户端逐批回滚和进程内 pending map；producer crash 后只能由
namespace/backend 清理。V2 可增加与 tensor 语义无关的 object-set：

1. `BeginObjectSet` 创建 `set_id` 和 bounded lease；
2. 各批对象携带同一 `set_id` 写入，但仍按 key/size 独立校验；
3. 全部对象 committed 后，服务端最后发布只含 key、size、allocation identity 的 ownership manifest；
4. RTP 只在 manifest committed 后发送业务 receipt；shape/dtype/role 仍由 RTP 保存；
5. consumer 按 `set_id` acquire/renew/release，失联后由服务端 lease 将整组转入 `DELETING`。

manifest-last 提供“整组是否可发布”的单点判断，不要求底层 metadata 支持 N-key 同时可见，也消除了大部分客户端
跨批回滚状态。lease 必须覆盖 receipt 传递和最慢 consumer，并支持显式续约；不能用一个固定短 TTL 代替消费协议。
`set_id` 不是授权凭据：Begin 必须额外返回不可猜测、绑定 instance/set generation 的 ownership token，renew/release
使用 operation id 幂等执行且不得记录 token；token 还必须绑定已认证 tenant，不能单独充当服务身份。若一个 set
允许多个 consumer，服务端应签发独立 child lease，而不是依赖无法在进程崩溃后收敛的裸引用计数。

manifest 可选保存 backend ETag 或内容 checksum 以发现静默损坏，但该能力必须协商并可关闭；KVCM 仍不解析
shape/dtype。GPU 路径只有在 backend/设备能直接计算校验值时才建议默认开启，不能为了 checksum 强制回拷主机并
影响主链路带宽。

### 13.6 QoS、背压与观测

V1.1/V2 应增加只属于 KVMeta 的资源预算：RPC in-flight、待 allocation bytes、data-plane task、cleanup queue、每
backend IOPS/bytes/s、recovery scan rate。达到上限时在 allocation 前返回明确的 `RESOURCE_EXHAUSTED`，不借用主
链路线程或无限排队。

预算采用 global → Instance Group → instance 的分层限制，并对 group 做公平调度，避免一个大 embedding 租户占满
4096 个全局 session。容量准入应维护 `reserved + committed + retired` 的聚合 byte counter，而不是每次写入扫描
group 内所有 instance；恢复时再用 durable record 校准。cleanup 执行队列可以有界，但 durable ledger 不能因队列满
而丢弃；队列过载时应暂停新的 allocation，并让 worker 从 ledger 分页续扫。

至少暴露以下脱敏指标：

- request latency/error 和 gate 状态；
- active session/object/bytes、committed logical/allocated bytes；
- deleting/orphan object/bytes、最老 retired/unsafe ledger age、retry/circuit-breaker 状态；
- singleton 与 batch capability 命中率、每对象 allocation 放大；
- recovery 扫描量、耗时、取消次数和无法识别 record 数。

### 13.7 兼容与分阶段落地

1. **V1.1**：先增加指标、独立限流/队列和部署校验；不改变 wire format；
2. **capability 协商**：Register 响应增加 additive protocol version/capability，client 对语义不兼容版本 fail closed；
3. **generation + cleanup ledger**：先接入一种内部 backend，双写审计但不重试，再开启条件删除 worker；
4. **异构批量**：在同一 adapter 上启用 `BATCH_VARIABLE_SIZE`，保留 singleton fallback 和对照指标；
5. **object set**：最后引入 server-side lease，RTP/v6d 按 capability 灰度使用。

每阶段都必须验证 capability 缺失、混合版本、返回错序/短结果、split brain、进程在每个状态转换点崩溃、地址立即
复用、旧 writer 延迟 Put、删除结果丢失以及 cleanup 熔断。任何 backend 不能证明 generation/fencing 契约时，都
必须保留 singleton fallback，并让无法证明安全的 allocation 进入隔离/人工确认；不能自动重放不确定删除，也不能
以可用性为理由降低安全条件。

## 14. 当前实现的测试边界

测试按层覆盖：

- Manager/Service UT：注册隔离、exact-key、不同 size、容量、session、Remove/Trim、HA recovery、生命周期，
  以及 rollback、Remove、Trim、expiry、recovery 中物理 Delete 返回错误/短结果/标准或未知异常时的一次性删除、
  metadata/usage 收敛、worker 存活和恢复继续放流；
- Client/SDK UT：响应对齐、URI/size/buffer 校验、CPU build 对 GPU buffer 的 fail-closed、failover、超时 drain、
  普通 TransferClient 回归；
- 内部 TairMempool UT：variable-size policy、严格 URI、禁 fallback、禁 gather/scatter；
- v6d UT/真实服务测试：CPU/CUDA buffer 封装、不同长度读写和 remove；
- RTP UT/native 测试：manifest、切片、批处理、回滚、release、GC 和 shutdown；
- 跨仓 contract test：真实 KVCM 服务 + KVCM wheel 中的 Python object client + RTP producer/receipt，覆盖同一
  receipt 内不同 size、显式 release 和 deadline GC；C++ reader、完整 RTP 进程和 GPU backend 由 RTP 对应测试层
  单独验证。

硬件相关 backend 和 RTP 全进程 GPU 测试仍需在对应部署镜像/CI 环境中执行；跨仓 contract test 不能替代它们。
