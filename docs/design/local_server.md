# TairKVCacheLocalServer 设计方案

## 1. 背景与动机

### 1.1 KVCM 的发展方向

Tair KVCache Manager (KVCM) 从**中心化元数据管理**起步，当前架构是推理引擎通过网络与远端 KVCM Server 交互，KVCache 数据存储在远端存储（3FS / Mooncake / NFS 等）。现在需要引入**本地存储**模式，原因有二：

1. **部分生产环境没有 RDMA**，更依赖本地存储 KVCache（本机内存或本机磁盘）。
2. **KVCache 亲和性分配**，即使在远端模式下，也希望优先将 KVCache 存储到离推理引擎近的存储节点（同机或同超节点），以节约大规模部署下的网络带宽。

### 1.2 LMCache 的对称发展

[LMCache](https://github.com/LMCache/LMCache) 的发展路径与 KVCM 恰好相反：先从本地存储（L1 内存 + L2 远端存储）起步，[MP 模式](https://blog.lmcache.ai/en/2026/06/15/understanding-lmcache-mp-mode-transfer-paths-a-beginners-guide/)下引入 [Coordinator](https://github.com/LMCache/LMCache/tree/714bd7a58af2/lmcache/v1/mp_coordinator)（类似中心化元数据管理）。两者的发展方向出现交汇，LMCache 的许多设计模式值得借鉴。

### 1.3 核心思路

在每台推理节点上部署一个 **TairKVCacheLocalServer** 进程，作为该节点上的**全功能 KVCache 管理器**。它不仅管理本地存储，还承接了原 TransferClient 的远端存储 I/O 职责，成为推理引擎与所有存储资源之间的唯一中间层。

**关键设计原则：**

1. **Python 框架 + C++ 内核**：整体框架用 Python 实现（与推理引擎生态一致，跨平台性好），但 local mem storage 分配和 local disk storage 读写的内核用 C++ 实现以保证性能。
2. **可独立部署**：即使没有中心化 KVCM Server，LocalServer 也能独立运行提供本地缓存服务，但预留与 KVCM 对接的接口（ReportEvent、Heartbeat 等）。
3. **吸收 TransferClient**：将原来分散在推理引擎进程内的存储 SDK 逻辑（3FS / Mooncake / NFS 等）统一收拢到 LocalServer 内部，推理引擎只需实现轻量级 Connector。
4. **借鉴 LMCache 的 Storage 接入模式**：本地存储和远端存储后端均通过 [Native Connector 框架](https://github.com/LMCache/LMCache/blob/714bd7a58af2/csrc/storage_backends/connector_interface.h)接入，支持[零侵入扩展](https://blog.lmcache.ai/zh/2026/05/26/%e5%bd%93%e5%bc%80%e6%ba%90%e9%81%87%e8%a7%81%e5%bc%80%e6%ba%90%ef%bc%9almcache-%e4%b8%8e-mooncake-%e7%9a%84%e4%b8%80%e6%ac%a1%e5%8f%8c%e5%90%91%e5%a5%94%e8%b5%b4/)。

## 2. 架构总览

### 2.1 架构对比：现状 vs 新方案

**现状：TransferClient 在推理引擎进程内**

```
Inference Engine (vLLM) Process
┌────────────────────────────────────────────┐
│  vLLM Engine                               │
│     │                                      │
│  py_connector (TairKvCacheConnector)       │
│     ├── KvCacheManagerClient (HTTP)  ──────────→  KVCM Server (remote)
│     ├── Triton gather/scatter kernels      │
│     └── TransferClient (C++ pybind)        │
│          └── SdkWrapper                    │
│               ├── Hf3fsSdk (3FS usrbio)    │
│               ├── MooncakeSdk (RDMA)       │
│               ├── LocalFileSdk (mmap)      │  <- all inside engine process
│               └── TairMempoolSdk (CXL)     │
└────────────────────────────────────────────┘
```

问题：每个推理引擎进程都要初始化全套存储 SDK，增加推理引擎的复杂度和依赖；
各引擎（vLLM / SGLang / TRT-LLM）都要重复实现 TransferClient 集成逻辑。

**新方案：LocalServer 作为统一中间层**

```
Inference Engine (vLLM) Process     TairKVCacheLocalServer Process
┌──────────────────────┐         ┌──────────────────────────────────────┐
│  vLLM Engine         │         │                                      │
│     │                │         │  Python Framework                    │
│  Lightweight         │  ZMQ/   │  ├── RequestRouter (ZMQ Server)      │
│  Connector           │  SHM    │  ├── LocalStorageManager             │
│  ├── GPU gather/     │ <---->  │  │   ├── L1MemConnector (C++ native) │
│  │   scatter (Tritn) │         │  │   └── L1DiskConnector (C++ native)│
│  ├── SHM attach      │         │  ├── RemoteStorageManager            │
│  └── ZMQ client      │         │  │   ├── KvCacheManagerClient (HTTP)  ──→ KVCM Server
│       (~300 lines)   │         │  │   ├── Hf3fsSdk (C++ native)       │   (optional)
└──────────────────────┘         │  │   ├── MooncakeSdk (C++ native)    │
                                 │  │   └── LocalFileSdk (C++ native)   │
SGLang Process                   │  ├── CacheIndex (lookup/eviction)    │
┌──────────────────────┐         │  ├── EventBus (metrics/reporting)    │
│  Lightweight    ────── ZMQ ──→ │  └── EventReporter -> KVCM (optional)│
│  Connector           │         └──────────────────────────────────────┘
└──────────────────────┘
```

### 2.2 部署拓扑

```
 Node A                                   Node B
┌─────────────────────────────────┐       ┌─────────────────────────────────┐
│  vLLM Worker 0 ─┐               │       │  SGLang Worker 0 ─┐             │
│  vLLM Worker 1 ─┤  ZMQ/SHM      │       │  SGLang Worker 1 ─┤  ZMQ/SHM    │
│  vLLM Worker N ─┘               │       │                   ─┘            │
│         │                       │       │         │                       │
│   TairKVCacheLocalServer        │       │   TairKVCacheLocalServer        │
│   ├─ L1 Mem (SHM, C++ native)   │       │   ├─ L1 Mem (SHM, C++ native)   │
│   ├─ L1 Disk (C++ native)       │       │   ├─ L1 Disk (C++ native)       │
│   ├─ Remote SDKs (C++ native)   │       │   ├─ Remote SDKs (C++ native)   │
│   └─ EventReporter (optional)   │       │   └─ EventReporter (optional)   │
│         │                       │       │         │                       │
└─────────│───────────────────────┘       └─────────│───────────────────────┘
          │        HTTP (optional)                  │
          └──────────┬─────────────────────────────┘
                     │
              ┌──────▼──────┐
              │  KVCM Server │  <- optional, works without it
              │  (central)   │
              └─────────────┘
```

### 2.3 存储配置模式

TairKVCacheLocalServer 的核心灵活性在于**存储配置的组合**。根据开启的存储层和写入策略，共有 5 种配置模式：

| 模式 | L1 本地存储 | 远端存储 | KVCM 上报 | 写入策略 | 典型场景 |
|------|-----------|---------|----------|---------|---------|
| **模式 1** | 内存/磁盘/混合 | - | ❌ | - | 单机推理、开发测试、无需全局调度 |
| **模式 2** | 内存/磁盘/混合 | - | ✅ | - | 单机推理 + 全局可观测（KVCM 可见缓存状态） |
| **模式 3** | 内存/磁盘/混合 | ✅ | ✅ | **Writeback** | 生产部署，追求最低 store 延迟 |
| **模式 4** | 内存/磁盘/混合 | ✅ | ✅ | **Writethrough** | 生产部署，追求数据安全性 |
| **模式 5** | - | ✅ | ✅ | - | 兼容现有 KVCM 部署，无需本地存储 |

**模式说明：**

- **模式 1（L1 Only，无 KVCM）**：纯本地模式。LocalServer 仅管理本机 KVCache（内存、磁盘、或两者兼有），不与 KVCM 通信。适合单机推理、开发测试等无需全局调度的场景。等价于 [LMCache MP Server](https://github.com/LMCache/LMCache/blob/714bd7a58af2/lmcache/v1/multiprocess/server.py) 的单机模式。

- **模式 2（L1 Only，有 KVCM 上报）**：本地存储与模式 1 相同（内存/磁盘/混合），但开启 KVCM 事件上报。KVCM 可知晓各节点的缓存状态用于全局调度和可观测，但不影响本地缓存功能——本地缓存的读写路径与模式 1 完全一致。

- **模式 3/4（L1 + Remote）**：混合存储模式。L1 作为热缓存层，远端存储作为容量层。两者的区别在于写入策略：
  - **Writeback（回写）**：Store 时只写 L1，L1 驱逐（eviction）时才异步将被驱逐的 block 写入远端存储。优点是 store 延迟最低，推理引擎的 commit_store 返回速度不受远端影响；缺点是驱逐前节点宕机会丢失未回写的数据。
  - **Writethrough（直写）**：Store 时同时写 L1 和远端存储（远端写入异步但立即发起，不阻塞 commit_store 返回）。优点是数据安全性高，远端始终有最新副本；缺点是 store 整体吞吐受远端写入带宽影响。

- **模式 5（Remote Only）**：不开启本地存储，LocalServer 将所有 KVCache 读写请求转发到远端存储。此模式下 LocalServer 本质上等价于现有 KVCM TransferClient，但接口统一——推理引擎仍通过轻量 Connector 与 LocalServer 通信，无需直接集成 C++ 存储 SDK。

**存储配置与部署模式的关系：**

| 配置模式 | Standalone（无 KVCM） | Managed（有 KVCM） |
|---------|---------------------|-------------------|
| 模式 1（L1 Only, 无 KVCM） | ✅ 唯一场景 | - |
| 模式 2（L1 Only, 有 KVCM） | - | ✅ 仅上报，不开远端 |
| 模式 3/4（L1 + Remote） | ❌ 需 KVCM 协调远端存储 | ✅ 主要场景 |
| 模式 5（Remote Only） | ❌ 需 KVCM 协调远端存储 | ✅ 兼容场景 |

### 2.4 通信路径

| 路径 | 协议 | 延迟 | 用途 |
|------|------|------|------|
| 推理引擎 ↔ LocalServer（控制面） | ZMQ tcp://localhost | 微秒级 | 请求路由（lookup/store/retrieve） |
| 推理引擎 ↔ LocalServer（数据面） | POSIX SHM | 零拷贝 | KVCache tensor 传输 |
| LocalServer → KVCM Server | HTTP | 毫秒级 | 元数据上报、远端存储协调（Managed 模式） |

## 3. 推理引擎侧：轻量 Connector

### 3.1 设计目标

将推理引擎侧的 Connector 做到**尽可能轻量**：

- 只保留**必须在推理引擎进程内执行**的逻辑（GPU 内存访问、Triton gather/scatter kernel）
- 所有存储相关逻辑（本地和远端）移入 LocalServer
- 各引擎的 Connector 代码量从 ~1000 行降到 ~300 行

### 3.2 Connector 保留的职责

| 职责 | 原因 | 代码量 |
|------|------|-------|
| GPU gather/scatter | 需要直接访问推理引擎的 GPU KV cache tensor，无法跨进程 | Triton kernel 复用 |
| SHM attach + tensor view | 需要在推理引擎进程内映射共享内存为 torch tensor | ~50 行 |
| ZMQ client | 与 LocalServer 的控制面通信 | ~100 行 |
| 引擎生命周期适配 | vLLM/SGLang/TRT-LLM 各自的 Connector 接口对接 | ~150 行/引擎 |

### 3.3 Connector 不再需要的职责

| 原来的职责 | 现在由谁负责 |
|-----------|-------------|
| `KvCacheManagerClient` (HTTP → KVCM) | LocalServer 内部 |
| `TransferClient` (C++ pybind, 所有存储 SDK) | LocalServer 内部 |
| 存储配置解析、SDK 初始化 | LocalServer 内部 |
| 服务发现、leader 切换 | LocalServer 内部 |
| TP Coordinator (跨 rank 协调) | 可保留在 Connector 或移入 LocalServer |

### 3.4 Connector 调用流程

```python
class LightweightConnector:
    """各引擎的轻量 Connector 基类，约 300 行代码"""

    def __init__(self, local_server_addr="tcp://localhost:5556"):
        # ZMQ client → LocalServer
        self.zmq_client = ZmqClient(local_server_addr)
        # 注册后拿到 SHM 信息
        resp = self.zmq_client.register(kv_layout=...)
        # attach 共享内存
        self.shm = SharedMemory(name=resp.shm_name, create=False)
        self.shm_buf = self.shm.buf

    def save_kv_cache(self, block_hashes, kv_tensors_gpu):
        # 1. 请求 LocalServer 分配 SHM slots
        resp = self.zmq_client.prepare_store(block_hashes)

        if not resp.slots:
            return  # 全部已缓存，无需写入

        # 2. GPU gather → SHM（在推理引擎进程内执行，1 次拷贝）
        for slot, gpu_tensor in zip(resp.slots, kv_tensors_gpu):
            shm_tensor = torch.frombuffer(
                self.shm_buf, dtype=slot.dtype,
                count=slot.length // element_size,
                offset=slot.offset)
            shm_tensor.copy_(gpu_tensor)  # GPU → SHM

        # 3. 通知 LocalServer 完成
        self.zmq_client.commit_store(resp.session_id)
        # LocalServer 后台异步：
        #   - 写入 L1 (内存/磁盘)
        #   - ReportEvent → KVCM (如果是 Managed 模式)
        #   - 异步复制到远端存储 (如果配置了)

    def load_kv_cache(self, block_hashes):
        # 1. 请求 LocalServer 查找并准备数据
        resp = self.zmq_client.prepare_retrieve(block_hashes)

        if not resp.found:
            return None  # 未命中

        # 2. SHM → GPU scatter（在推理引擎进程内执行，1 次拷贝）
        kv_tensors = []
        for slot in resp.slots:
            shm_tensor = torch.frombuffer(
                self.shm_buf, dtype=slot.dtype,
                count=slot.length // element_size,
                offset=slot.offset)
            gpu_tensor = shm_tensor.to(device='cuda', non_blocking=True)
            kv_tensors.append(gpu_tensor)

        # 3. 通知 LocalServer 完成
        self.zmq_client.commit_retrieve(resp.session_id)
        return kv_tensors
```

各引擎只需继承这个基类，适配自己的 KV cache 布局：

```python
# vLLM connector
class VLLMTairKVCacheConnector(KVConnectorBase_V1, LightweightConnector):
    """~150 行引擎适配代码"""
    def start_load(self, ...):
        # vLLM 特有的 slot_mapping 处理
        # 调用 self.load_kv_cache(block_hashes)
        ...

# SGLang connector
class SGLangTairKVCacheConnector(HiCacheStorage, LightweightConnector):
    """~150 行引擎适配代码"""
    ...
```

### 3.5 对比：现在 vs 改造后

| 维度 | 现在（TransferClient 在引擎内） | 改造后（轻量 Connector） |
|------|------|------|
| 引擎进程依赖 | C++ pybind 模块 + 全套存储 SDK | 仅 ZMQ + SHM (纯 Python) |
| 引擎 Connector 代码量 | ~1000 行/引擎 | ~300 行/引擎 |
| 新引擎接入 | 重复集成 TransferClient | 只实现 gather/scatter + ZMQ client |
| 存储后端切换 | 需重新编译 TransferClient | LocalServer 侧配置切换，引擎无感 |
| 故障隔离 | 存储 SDK 崩溃影响推理 | 存储 SDK 在独立进程，推理引擎不受影响 |

## 4. LocalServer 内部架构

### 4.1 整体结构：Python 框架 + C++ 内核

```
TairKVCacheLocalServer (Python Process)
│
├── RequestRouter (Python, ZMQ Server)
│   ├── ZMQ ROUTER socket on tcp://localhost:5556
│   ├── SyncHandler: REGISTER / LOOKUP / PING
│   └── BlockingHandler -> ThreadPool: STORE / RETRIEVE
│
├── StorageManager (Python)
│   │
│   ├── LocalStorageManager
│   │   ├── L1MemConnector --> C++ Native (SHM pool, slab allocator)
│   │   ├── L1DiskConnector --> C++ Native (O_DIRECT file I/O)
│   │   └── CacheIndex (Python dict, block_hash -> slot/path)
│   │
│   └── RemoteStorageManager (Managed mode only)
│       ├── KvCacheManagerClient (Python HTTP) -> KVCM Server
│       ├── Hf3fsConnector --> C++ Native (usrbio)
│       ├── MooncakeConnector --> C++ Native (RDMA)
│       └── LocalFileConnector --> C++ Native (mmap)
│
├── EvictionController (Python)
│   └── LRU / TTL / LeafAwareLRU policies
│
├── EventBus (Python)
│   ├── MetricsSubscriber -> Prometheus
│   ├── LoggingSubscriber -> structured logs
│   └── ReportSubscriber -> KVCM Server (Managed mode only)
│
├── EventReporter (Python, Managed mode only)
│   ├── ReportEventClient (HTTP -> KVCM Server)
│   ├── HeartbeatLoop
│   └── BatchReportQueue
│
└── HTTP Frontend (FastAPI, optional, for K8s/ops health probes)
    ├── /healthcheck  <- K8s liveness/readiness probe
    ├── /status
    └── /metrics
```

### 4.2 Python vs C++ 的分工

| 层次 | 语言 | 职责 | 理由 |
|------|------|------|------|
| **请求路由** | Python | ZMQ Server、请求分发、session 管理 | I/O bound，Python 足够 |
| **缓存索引** | Python | block_hash → location 映射、驱逐策略 | 数据结构简单，需要灵活性 |
| **元数据上报** | Python | HTTP client → KVCM Server | 网络 I/O，Python 自然 |
| **事件总线** | Python | pub/sub、订阅者管理 | 参考 LMCache 的 [EventBus](https://github.com/LMCache/LMCache/blob/714bd7a58af2/lmcache/v1/mp_observability/event_bus.py) 实现 |
| **L1 内存存储** | **C++** | SHM slab allocator、slot 分配/回收 | 性能关键路径，需要精确的内存管理 |
| **L1 磁盘存储** | **C++** | O_DIRECT 文件 I/O、buffer 管理 | 需要绕过 page cache 的底层控制 |
| **远端存储 SDK** | **C++** | 3FS usrbio / Mooncake RDMA / mmap | 复用现有 C++ SDK，性能关键 |
| **pybind 绑定** | C++→Python | 将 C++ Connector 暴露给 Python | 一行宏搞定 |

### 4.3 C++ Native Connector 接口设计

借鉴 LMCache 的 [`IStorageConnector`](https://github.com/LMCache/LMCache/blob/714bd7a58af2/csrc/storage_backends/connector_interface.h) + [`ConnectorBase<T>`](https://github.com/LMCache/LMCache/blob/714bd7a58af2/csrc/storage_backends/connector_base.h) 模式：

```
IStorageConnector (C++ pure virtual interface)
    ├── event_fd() -> int
    ├── submit_batch_get(keys, bufs, lens) -> future_id
    ├── submit_batch_set(keys, bufs, lens) -> future_id
    ├── submit_batch_exists(keys) -> future_id
    ├── submit_batch_delete(keys) -> future_id
    ├── drain_completions() -> list[Completion]
    └── close()

ConnectorBase<T> (CRTP template, common infrastructure)
    ├── subclass implements: create_connection / do_single_get / do_single_set / do_single_exists
    ├── provides for free: thread pool / SQ+CQ / auto tiling / eventfd / per-op worker pools
    └── KVCM_BIND_CONNECTOR_METHODS(Type) -- one-line pybind macro
```

**内置的 C++ Native Connector 实现：**

```
L1MemConnector : ConnectorBase<ShmConnection>
    ├── create_connection(): init SHM slab allocator
    ├── do_single_set(): slab_alloc + memcpy
    ├── do_single_get(): slab_lookup + memcpy
    └── do_single_exists(): slab_lookup

L1DiskConnector : ConnectorBase<FileConnection>
    ├── create_connection(): open O_DIRECT fd
    ├── do_single_set(): aligned_write
    ├── do_single_get(): aligned_read
    └── do_single_exists(): stat / bloom_filter

Hf3fsConnector : ConnectorBase<Hf3fsConnection>
    └── wraps existing Hf3fsSdk usrbio logic

MooncakeConnector : ConnectorBase<MooncakeConnection>
    └── wraps existing MooncakeSdk RDMA logic

LocalFileConnector : ConnectorBase<MmapConnection>
    └── wraps existing LocalFileSdk mmap logic
```

### 4.4 Connector 在 LocalServer 内的使用

```python
# Python 框架侧：通过 pybind 使用 C++ Connector
class LocalStorageManager:
    def __init__(self, config):
        # 加载 C++ native connectors (编译为 .so, pybind 暴露)
        from kvcm_native_connectors import L1MemClient, L1DiskClient

        self.mem_connector = L1MemClient(
            shm_name=config.shm_name,
            pool_size=config.mem_pool_size,
            num_workers=config.mem_workers)

        if config.disk_path:
            self.disk_connector = L1DiskClient(
                base_path=config.disk_path,
                num_workers=config.disk_workers)

    async def store(self, keys, data):
        """先写内存，溢出写磁盘"""
        future_id = self.mem_connector.submit_batch_set(
            keys, data_bufs, data_lens)
        # eventfd 通知完成 (非 polling)
        ...

    async def retrieve(self, keys):
        """先查内存，miss 查磁盘"""
        future_id = self.mem_connector.submit_batch_get(
            keys, out_bufs, out_lens)
        ...
```

### 4.5 远端存储集成（吸收 TransferClient）

在 Managed 模式下，LocalServer 还承担远端存储的读写，完全替代原来推理引擎进程内的 TransferClient：

```python
class RemoteStorageManager:
    def __init__(self, kvcm_url, storage_configs):
        # Python HTTP client → KVCM Server (复用现有 manager_client.py)
        self.manager_client = KvCacheManagerClient(kvcm_url)

        # C++ native connectors for remote storage
        self.remote_connectors = {}
        for config in storage_configs:
            if config.type == "hf3fs":
                from kvcm_native_connectors import Hf3fsClient
                self.remote_connectors[config.name] = Hf3fsClient(
                    **config.params)
            elif config.type == "mooncake":
                from kvcm_native_connectors import MooncakeClient
                self.remote_connectors[config.name] = MooncakeClient(
                    **config.params)
            # ... 或通过 plugin 机制动态加载

    async def start_write(self, block_hashes):
        """两阶段写入：向 KVCM 申请写入位置"""
        resp = self.manager_client.start_write_cache(block_hashes)
        return resp.locations  # 远端存储 URI

    async def save_to_remote(self, locations, data):
        """通过对应的 native connector 写入远端存储"""
        connector = self.remote_connectors[location.storage_name]
        connector.submit_batch_set(keys, bufs, lens)
        ...

    async def finish_write(self, session_id, success_mask):
        """提交写入结果给 KVCM"""
        self.manager_client.finish_write_cache(session_id, success_mask)
```

### 4.6 完整的数据流

数据流因存储配置模式不同而有差异（5 种模式定义见 2.3 节）。

#### 4.6.1 Store 流程

**模式 1/2（L1 Only）— 最简路径（模式 2 比模式 1 多 KVCM 上报）：**

```
Engine                             LocalServer
    │                              │
    │ 1. prepare_store(hashes) --> │
    │                              │ check local cache, alloc SHM slots
    │ <-- slots + already_cached   │
    │                              │
    │ 2. GPU -> SHM (gather)       │
    │                              │
    │ 3. commit_store -----------> │
    │ <-- done                     │
    │                              │ 4. async background:
    │   (engine returned,          │    SHM -> L1 mem/disk
    │    continues inference)      │    EventBus.publish(STORE_FINISH)
    │                              │    ReportEvent -> KVCM (if enabled)
```

**模式 3（L1 + Remote, Writeback 回写）— 驱逐时才写远端：**

```
Engine                             LocalServer                        KVCM + Remote Storage
    │                              │                                      │
    │ 1. prepare_store(hashes) --> │                                      │
    │                              │ check local cache, alloc SHM slots   │
    │ <-- slots + already_cached   │                                      │
    │                              │                                      │
    │ 2. GPU -> SHM (gather)       │                                      │
    │                              │                                      │
    │ 3. commit_store -----------> │                                      │
    │ <-- done                     │                                      │
    │                              │ 4. async background:                 │
    │   (engine returned,          │    SHM -> L1 mem/disk                │
    │    continues inference)      │    EventBus.publish(STORE_FINISH)    │
    │                              │    (no remote write at this point)   │
    │                              │                                      │
    ..... (later, L1 capacity full) .....
    │                              │                                      │
    │                              │ 5. EvictionController triggers:      │
    │                              │    select victim blocks (LRU/TTL)    │
    │                              │    read victim data from L1          │
    │                              │    start_write -> KVCM ------------> │
    │                              │    save_to_remote (C++ SDK) -------> │
    │                              │    finish_write -> KVCM -----------> │
    │                              │    EventBus.publish(WRITEBACK)       │
    │                              │    free L1 slots                     │
    │                              │                                      │
```

> Writeback 模式下，block 的生命周期：写入 L1 → 被驱逐时回写到远端 → 释放 L1 slot。对推理引擎完全透明，store 延迟 = L1 写入延迟。

**模式 4（L1 + Remote, Writethrough 直写）— 同时写 L1 和远端：**

```
Engine                             LocalServer                        KVCM + Remote Storage
    │                              │                                      │
    │ 1. prepare_store(hashes) --> │                                      │
    │                              │ check local cache, alloc SHM slots   │
    │ <-- slots + already_cached   │                                      │
    │                              │                                      │
    │ 2. GPU -> SHM (gather)       │                                      │
    │                              │                                      │
    │ 3. commit_store -----------> │                                      │
    │ <-- done                     │ 4. dual-write in background:         │
    │                              │    +-- SHM -> L1 mem/disk            │
    │   (engine returned,          │    +-- start_write -> KVCM --------> │
    │    continues inference)      │        save_to_remote (SDK) -------> │
    │                              │        finish_write -> KVCM -------> │
    │                              │    both done:                        │
    │                              │    EventBus.publish(STORE_FINISH)    │
    │                              │    ReportEvent(BLOCK_ADD) ---------> │
    │                              │                                      │
```

> Writethrough 模式下，commit_store 返回不阻塞（推理引擎无感），但 LocalServer 后台同时向 L1 和远端写入。L1 驱逐时无需回写（远端已有副本），直接释放 L1 slot。

**模式 5（Remote Only）— 无本地存储：**

```
Engine                             LocalServer                        KVCM + Remote Storage
    │                              │                                      │
    │ 1. prepare_store(hashes) --> │                                      │
    │                              │ alloc SHM slots (temp buffer)        │
    │ <-- slots                    │                                      │
    │                              │                                      │
    │ 2. GPU -> SHM (gather)       │                                      │
    │                              │                                      │
    │ 3. commit_store -----------> │                                      │
    │ <-- done                     │ 4. async background:                 │
    │                              │    start_write -> KVCM ------------> │
    │   (engine returned,          │    SHM -> remote (C++ SDK) --------> │
    │    continues inference)      │    finish_write -> KVCM -----------> │
    │                              │    free SHM slots (temp buf only)    │
    │                              │                                      │
```

> 模式 5 等价于现有 TransferClient，但推理引擎侧的接口与其他模式完全一致。SHM 仅用作 GPU↔远端存储之间的临时拷贝缓冲，不做持久化缓存。

#### 4.6.2 Retrieve 流程

**模式 1/2（L1 Only）— 模式 2 比模式 1 多 KVCM 上报：**

```
Engine                             LocalServer
    │                              │
    │ 1. prepare_retrieve(hashes)->│
    │                              │ check local index
    │                              │ hit: write L1 data to SHM slots
    │ <-- slots / miss             │ miss: return miss
    │                              │
    │ 2. SHM -> GPU (scatter)      │
    │ 3. commit_retrieve --------> │ free SHM slots
```

**模式 3/4/5（有远端存储）— L1 优先 + 远端 fallback：**

```
Engine                             LocalServer                        KVCM + Remote Storage
    │                              │                                      │
    │ 1. prepare_retrieve(hashes)->│                                      │
    │                              │ check local index (skip in mode 5)   │
    │                              │                                      │
    │                              │ -- L1 hit (mode 3/4) --              │
    │                              │ write L1 data to SHM slots           │
    │ <-- slots (local hit)        │                                      │
    │                              │                                      │
    │                              │ -- L1 miss or mode 5 --              │
    │                              │ get_cache_location -> KVCM --------> │
    │                              │ load_from_remote (C++ SDK) <-------- │
    │                              │ write remote data to SHM slots       │
    │ <-- slots (remote hit)       │                                      │
    │                              │                                      │
    │ 2. SHM -> GPU (scatter)      │                                      │
    │ 3. commit_retrieve --------> │ free SHM slots                       │
```

> 模式 3（Writeback）的 retrieve 有一个特殊情况：如果 block 尚未被驱逐到远端（只在 L1 中），则远端不会有该数据。因此 Writeback 模式下，L1 的驱逐策略需要感知 retrieve 的远端 fallback 需求。

## 5. 独立部署模式（Standalone）

### 5.1 启动示例

```bash
# 模式 1：L1 Only，无 KVCM（Standalone）
# L1 可以是内存、磁盘、或两者兼有
python -m tair_kvcache_local_server \
    --host 0.0.0.0 --port 5556 \
    --l1-mem-size 8G \
    --shm-name kvcache_pool
    # 不传 --kvcm-url，纯本地模式

# 模式 1 变体：L1 磁盘（Standalone）
python -m tair_kvcache_local_server \
    --host 0.0.0.0 --port 5556 \
    --l1-disk-path /data/kvcache \
    --l1-disk-size 100G

# 模式 1 变体：L1 内存 + 磁盘混合（Standalone）
python -m tair_kvcache_local_server \
    --host 0.0.0.0 --port 5556 \
    --l1-mem-size 8G \
    --shm-name kvcache_pool \
    --l1-disk-path /data/kvcache \
    --l1-disk-size 100G

# 模式 2：L1 Only + KVCM 上报（Managed，但不开远端存储）
python -m tair_kvcache_local_server \
    --host 0.0.0.0 --port 5556 \
    --l1-mem-size 8G \
    --shm-name kvcache_pool \
    --kvcm-url http://kvcm-server:8080 \
    --instance-id my-instance
    # 有 KVCM 上报，但不配置远端存储

# 模式 3：L1 + Remote，Writeback
python -m tair_kvcache_local_server \
    --host 0.0.0.0 --port 5556 \
    --l1-mem-size 8G \
    --shm-name kvcache_pool \
    --kvcm-url http://kvcm-server:8080 \
    --instance-id my-instance \
    --remote-storage '{"type": "mooncake", "params": {...}}' \
    --write-policy writeback

# 模式 4：L1 + Remote，Writethrough
python -m tair_kvcache_local_server \
    --host 0.0.0.0 --port 5556 \
    --l1-mem-size 8G \
    --shm-name kvcache_pool \
    --kvcm-url http://kvcm-server:8080 \
    --instance-id my-instance \
    --remote-storage '{"type": "mooncake", "params": {...}}' \
    --write-policy writethrough

# 模式 5：Remote Only（无本地存储，等价于 TransferClient）
python -m tair_kvcache_local_server \
    --host 0.0.0.0 --port 5556 \
    --shm-name kvcache_pool \
    --kvcm-url http://kvcm-server:8080 \
    --instance-id my-instance \
    --remote-storage '{"type": "mooncake", "params": {...}}'
    # 不传 --l1-mem-size / --l1-disk-path，就是 Remote Only 模式
```

### 5.2 各模式功能矩阵

| 功能 | 模式 1 (L1, 无 KVCM) | 模式 2 (L1, 有 KVCM) | 模式 3 (Writeback) | 模式 4 (Writethrough) | 模式 5 (Remote Only) |
|------|:--------------------:|:--------------------:|:------------------:|:---------------------:|:--------------------:|
| L1 内存存储 (SHM) | ✅ | ✅ | ✅ | ✅ | - |
| L1 磁盘存储 | ✅ | ✅ | ✅ | ✅ | - |
| 本地 lookup | ✅ | ✅ | ✅ | ✅ | - |
| 本地驱逐 | ✅ | ✅ | ✅ (驱逐→回写远端) | ✅ (驱逐直接释放) | - |
| 远端存储读写 | - | - | ✅ (驱逐时写) | ✅ (实时写) | ✅ |
| 远端 retrieve fallback | - | - | ✅ | ✅ | ✅ (唯一路径) |
| Prometheus 指标 | ✅ | ✅ | ✅ | ✅ | ✅ |
| HTTP 健康检查 | 可选 | 可选 | 可选 | 可选 | 可选 |
| KVCM 元数据上报 | ❌ | ✅ | ✅ | ✅ | ✅ |
| 心跳保活 | ❌ | ✅ | ✅ | ✅ | ✅ |
| 跨节点缓存查询 | - | ✅ (via KVCM) | ✅ (via KVCM) | ✅ (via KVCM) | ✅ (via KVCM) |
| Store 延迟 | 最低 (仅 L1) | 最低 (仅 L1) | 低 (仅 L1) | 中 (L1 + 异步远端) | 高 (仅远端) |
| 数据安全性 | 低 (节点宕机丢失) | 低 (节点宕机丢失) | 中 (驱逐后安全) | 高 (实时双副本) | 高 (远端持久化) |

> **关于健康检查的说明：** HTTP 健康检查端点（`/healthcheck`）是给**外部编排系统**（K8s kubelet、supervisor、运维平台）用的，不是给 KVCM 用的。KVCM 对 LocalServer 的存活判断完全依赖心跳机制（见 7.4 节），不需要额外的 HTTP 探活。

### 5.3 存储模式的条件化初始化

```python
class TairKVCacheLocalServer:
    def __init__(self, config):
        # 始终初始化
        self.event_bus = EventBus()
        self.cache_index = CacheIndex()

        # ── L1 本地存储（模式 1/2/3/4 开启）──
        if config.l1_mem_size or config.l1_disk_path:
            self.local_storage = LocalStorageManager(config)
            self.eviction = EvictionController(config.eviction_policy)
        else:
            self.local_storage = None  # 模式 5: Remote Only
            self.eviction = None

        # ── 远端存储（模式 3/4/5 开启）──
        if config.remote_storages:
            assert config.kvcm_url, "远端存储需要 KVCM 协调"
            self.remote_storage = RemoteStorageManager(
                config.kvcm_url, config.remote_storages)
        else:
            self.remote_storage = None  # 模式 1/2: L1 Only（无远端存储）

        # ── 写入策略（模式 3/4 才需要）──
        if self.local_storage and self.remote_storage:
            self.write_policy = config.write_policy  # "writeback" / "writethrough"
            if self.write_policy == "writeback":
                # 驱逐回调：L1 驱逐时写入远端
                self.eviction.set_writeback_callback(
                    self._on_eviction_writeback)
        else:
            self.write_policy = None

        # ── KVCM 对接（有 kvcm_url 就开启；模式 2/3/4/5 需要，模式 1 不配置）──
        if config.kvcm_url:
            self.event_reporter = EventReporter(config.kvcm_url)
            self.event_bus.subscribe(ReportSubscriber(
                self.event_reporter))
            self.event_reporter.start_heartbeat_loop()
        else:
            self.event_reporter = None

    def _on_eviction_writeback(self, evicted_blocks):
        """Writeback 模式下的驱逐回调：将被驱逐的 block 写入远端存储"""
        locations = self.remote_storage.start_write(evicted_blocks.hashes)
        self.remote_storage.save_to_remote(locations, evicted_blocks.data)
        self.remote_storage.finish_write(locations.session_id, success_mask)
        self.event_bus.publish(CacheEvent(
            type=EventType.WRITEBACK, block_hashes=evicted_blocks.hashes))
```

**模式判定逻辑：**

```python
@property
def storage_mode(self):
    has_l1 = self.local_storage is not None
    has_remote = self.remote_storage is not None
    has_kvcm = self.event_reporter is not None
    if has_l1 and not has_remote and not has_kvcm:
        return "l1_only"          # 模式 1: L1 Only, 无 KVCM
    elif has_l1 and not has_remote and has_kvcm:
        return "l1_with_kvcm"     # 模式 2: L1 Only, 有 KVCM 上报
    elif has_l1 and has_remote:
        return self.write_policy  # 模式 3: "writeback" / 模式 4: "writethrough"
    elif not has_l1 and has_remote:
        return "remote_only"      # 模式 5
    else:
        raise ValueError("至少需要开启一种存储")
```

## 6. KVCache 数据传输

### 6.1 传输模式

TairKVCacheLocalServer 支持两种本地传输模式：

| 传输模式 | Store 拷贝 | Retrieve 拷贝 | 适用场景 |
|---------|-----------|-------------|---------|
| **SHM 零拷贝**（推荐） | 1 (GPU→SHM) | 1 (SHM→GPU) | 默认模式 |
| **ZMQ 数据帧** | 2 (GPU→CPU→ZMQ) | 2 (ZMQ→CPU→GPU) | SHM 不可用时 |

### 6.2 SHM 零拷贝模式（推荐）

```
Store:
  Engine GPU                   Shared Memory (/dev/shm)           LocalServer
       │                           │                              │
       │ 1. prepare_store ─────────────────────────────────────→  │
       │                           │         alloc SHM slots,     │
       │ <── ShmSlotDescriptor ──────────── add write_lock        │
       │                           │                              │
       │ 2. GPU -> SHM (1 copy) ─→ │                              │
       │                           │                              │
       │ 3. commit_store ─────────────────────────────────────→   │
       │                           │         release write_lock,  │
       │                           │         update CacheIndex,   │
       │                           │         async persist+report │
       └───────────────────────────┘──────────────────────────────┘
```

关键实现要点（借鉴 LMCache [`EngineDrivenContext` SHM 实现](https://github.com/LMCache/LMCache/blob/714bd7a58af2/lmcache/v1/multiprocess/transfer_context/shm.py)）：

- **SHM 池管理**：LocalServer 启动时创建 POSIX 共享内存，推理引擎通过 `register` 拿到 `shm_name` 后 attach
- **Slot 描述符**：`ShmSlotDescriptor(offset, length, shape, dtype)`，推理引擎通过 `torch.frombuffer` 创建零拷贝 tensor view
- **Write Lock + TTL**：防止 LocalServer 在推理引擎写入 SHM 的过程中驱逐该 slot
- **已缓存优化**：`prepare_store` 返回 `already_cached` 标记，跳过已缓存 block 的拷贝

### 6.3 两阶段协议

借鉴 LMCache 的 [`PREPARE → data operation → COMMIT`](https://github.com/LMCache/LMCache/blob/714bd7a58af2/lmcache/v1/multiprocess/protocols/base.py) 设计：

- **PREPARE**：分配资源、加锁、返回传输描述符
- **数据操作**：推理引擎在两次 RPC 之间执行 GPU↔SHM 拷贝，不经过 LocalServer 进程
- **COMMIT**：释放锁、更新索引、触发后台任务

控制面走 ZMQ（简单高效），数据面走 SHM（零拷贝），两者解耦。

### 6.4 控制面协议定义

```python
# 借鉴 LMCache 的 RequestType enum（见 https://github.com/LMCache/LMCache/blob/714bd7a58af2/lmcache/v1/multiprocess/protocols/base.py）
class RequestType(IntEnum):
    # 生命周期
    REGISTER = 1          # 推理引擎注册，返回 SHM info
    UNREGISTER = 2
    PING = 3

    # 缓存操作
    LOOKUP = 10           # 查询 block 是否存在
    PREPARE_STORE = 11    # 分配 SHM slots
    COMMIT_STORE = 12     # 确认写入完成
    PREPARE_RETRIEVE = 13 # 准备数据到 SHM slots
    COMMIT_RETRIEVE = 14  # 确认读取完成

    # 管理
    EVICT = 20            # 主动驱逐
    STATUS = 21           # 状态查询
```

## 7. 与 KVCM 中心服务端的交互

### 7.1 元数据上报：复用 ReportEvent

KVCM 已有成熟的 `ReportEvent` 机制（5 种事件类型），LocalServer 在 Managed 模式下直接复用：

| 事件类型 | LocalServer 如何使用 |
|---------|-------------------|
| `NODE_REGISTER` | 启动时注册自身节点信息 |
| `BLOCK_ADD` | 本地写入成功后，异步批量上报 |
| `BLOCK_DELETE` | 本地驱逐后，异步上报 |
| `HEARTBEAT` | 周期性发送，携带存储利用率 |
| `HOST_DOWN` | 优雅关闭时上报 |

### 7.2 上报模式：异步批量 + 与本地操作解耦

```
commit_store done
    │
    ▼
EventBus.publish(STORE_FINISH, block_hashes)
    │
    ▼                              <- engine already got result and returned
ReportSubscriber (async consumer)
    │
    ├─ batch (batch_size=100 or interval=100ms)
    │
    ▼
ReportEvent HTTP -> KVCM Server     <- failure does not affect local cache
```

### 7.3 弹性注册（借鉴 [LMCache Coordinator](https://github.com/LMCache/LMCache/blob/714bd7a58af2/lmcache/v1/mp_coordinator/registry.py) 模式，[设计文档](https://github.com/LMCache/LMCache/blob/714bd7a58af2/docs/design/v1/mp_coordinator/README.md)）

```python
class EventReporter:
    async def heartbeat_loop(self):
        while self.running:
            try:
                resp = self.client.report_event(
                    event_type="HEARTBEAT",
                    system_status=self.collect_status())
                if resp.code == "NODE_NOT_REGISTERED":
                    # KVCM 重启后丢失节点信息，重新注册
                    await self.register()
            except ConnectionError:
                # 网络断开，本地缓存仍正常服务
                pass
            await asyncio.sleep(self.heartbeat_interval)
```

### 7.4 KVCM 侧的心跳超时处理

KVCM 对 LocalServer 的存活判断**完全依赖心跳机制**，不需要额外的 HTTP 探活。这套机制已在 VineyardBackend 中实现并经过生产验证，LocalServer 直接复用。

**完整的生命周期状态机：**

```
LocalServer starts
    │
    ▼
1. NODE_REGISTER -> KVCM
    KVCM: RegisterNode(), record node info, node_generation++
    │
    ▼
2. HEARTBEAT (periodic, default 5s) -> KVCM
    KVCM: OnHeartbeat()
      - update last_heartbeat_ms
      - collect system_status (storage utilization etc.) -> Prometheus
      - if previously marked unavailable -> restore to available
    │
    │  -- heartbeat normal: KVCM uses this node's cache locations
    │     normally in GetCacheLocation / StartWriteCache
    │
    ▼
3. heartbeat timeout (default 30s without heartbeat)
    KVCM: LivenessCheckerLoop detects
      - mark node unavailable
      - clear node's Prometheus gauges (ClearNodeGauges)
      - KVCM stops assigning new storage locations on this node
      - existing metadata (CacheLocation) kept temporarily
    │
    │  -- if LocalServer resumes heartbeat now:
    │     OnHeartbeat() re-marks as available, everything recovers
    │
    ▼
4. grace period timeout (default 5min from marking unavailable)
    KVCM: LivenessCheckerLoop triggers cleanup
      - call CleanupCallback: batch-delete all CacheLocations
        on this node from metadata index
      - check if node_generation still matches (ABA guard):
        - match -> UnregisterNode(), fully remove node
        - mismatch -> node already re-registered, skip cleanup
    │
    ▼
5. (if LocalServer recovers after cleanup)
    next HEARTBEAT -> KVCM returns EC_NODE_NOT_REGISTERED
    LocalServer: re-execute NODE_REGISTER
    KVCM: RegisterNode(), node_generation++ (new generation)
    local cache data still present, re-report BLOCK_ADD
```

**关键设计点：**

- **三级递进**：心跳正常 → 超时标记不可用（停止分配，保留数据）→ 宽限期后清理（删除元数据），给了网络抖动充分的恢复窗口
- **node_generation 防 ABA**：节点每次注册都递增 generation，清理回调触发时会检查 generation 是否仍匹配，防止旧的清理任务误删重新注册节点的数据
- **本地缓存不受影响**：KVCM 侧的清理只影响**中心元数据**，LocalServer 本地的 L1 缓存数据和索引不受影响，Standalone 路径仍正常工作

## 8. 存储后端扩展性

### 8.1 统一的 Native Connector 接口

LocalServer 的所有存储后端（无论本地还是远端）统一通过 `IStorageConnector` 接口接入：

```cpp
// C++ 接口 (connector_interface.h)
class IStorageConnector {
public:
    virtual int event_fd() = 0;
    virtual uint64_t submit_batch_get(keys, bufs, lens) = 0;
    virtual uint64_t submit_batch_set(keys, bufs, lens) = 0;
    virtual uint64_t submit_batch_exists(keys) = 0;
    virtual uint64_t submit_batch_delete(keys) = 0;
    virtual std::vector<Completion> drain_completions() = 0;
    virtual void close() = 0;
};
```

```cpp
// CRTP 模板 (connector_base.h)
template<typename ConnType>
class ConnectorBase : public IStorageConnector {
protected:
    // 子类只需实现这 4 个方法
    virtual ConnType create_connection() = 0;
    virtual bool do_single_get(ConnType&, key, buf, len) = 0;
    virtual bool do_single_set(ConnType&, key, buf, len) = 0;
    virtual bool do_single_exists(ConnType&, key) = 0;

    // 免费获得: 线程池, SQ/CQ, tiling, eventfd, per-op workers
};

// pybind 一行宏
KVCM_BIND_CONNECTOR_METHODS(L1MemConnector)
KVCM_BIND_CONNECTOR_METHODS(L1DiskConnector)
KVCM_BIND_CONNECTOR_METHODS(Hf3fsConnector)
```

### 8.2 内置 Connector 实现

| Connector | 类型 | 实现 |
|-----------|------|------|
| `L1MemConnector` | 本地内存 | SHM slab allocator |
| `L1DiskConnector` | 本地磁盘 | O_DIRECT file I/O |
| `Hf3fsConnector` | 远端存储 | 封装现有 Hf3fsSdk (usrbio) |
| `MooncakeConnector` | 远端存储 | 封装现有 MooncakeSdk (RDMA) |
| `LocalFileConnector` | 远端/本地存储 | 封装现有 LocalFileSdk (mmap) |

### 8.3 第三方 Connector 扩展（零侵入）

借鉴 LMCache 的 [`native_plugin` 机制](https://github.com/LMCache/LMCache/blob/714bd7a58af2/lmcache/v1/distributed/l2_adapters/native_plugin_l2_adapter.py)，第三方可以通过 `pip install` 接入新后端：

```python
# 启动时通过配置加载第三方 connector
python -m tair_kvcache_local_server \
    --remote-storage '{
        "type": "native_plugin",
        "module_path": "my_storage_connector",
        "class_name": "MyConnector",
        "params": {"endpoint": "...", "token": "..."}
    }'
```

加载链：
```
1. importlib.import_module("my_storage_connector")
2. getattr(module, "MyConnector")
3. MyConnector(**params)
4. 验证 IStorageConnector 必需方法
5. 注册到 RemoteStorageManager
```

第三方 Connector 只需：
1. 继承 `ConnectorBase<T>`，实现 4 个方法
2. 用 `KVCM_BIND_CONNECTOR_METHODS` 宏生成 pybind 绑定
3. 打包为 pip 包

**不需要修改 LocalServer 任何代码。**

### 8.4 配置透传

借鉴 [LMCache × Mooncake 接入](https://blog.lmcache.ai/zh/2026/05/26/%e5%bd%93%e5%bc%80%e6%ba%90%e9%81%87%e8%a7%81%e5%bc%80%e6%ba%90%ef%bc%9almcache-%e4%b8%8e-mooncake-%e7%9a%84%e4%b8%80%e6%ac%a1%e5%8f%8c%e5%90%91%e5%a5%94%e8%b5%b4/)的设计哲学——不替第三方做翻译，只做透传：

```python
# connector 收到的配置就是用户传入的原始 params dict
class MyConnector(ConnectorBase):
    def __init__(self, endpoint, token, batch_size=100, **kwargs):
        # 直接拿到自己关心的配置
        self.client = MyStorageClient(endpoint, token)
        self.batch_size = batch_size
```

LocalServer 不定义任何后端特有的配置 schema，新后端增删配置项时 LocalServer 一行代码都不用改。

### 8.5 Per-Op Worker Pools

借鉴 LMCache 在 [`ConnectorBase`](https://github.com/LMCache/LMCache/blob/714bd7a58af2/csrc/storage_backends/connector_base.h) 中的 `WorkerPoolConfig`，将 lookup / retrieve / store / delete 分配到独立 worker lane：

```python
# 配置示例
--remote-storage '{
    "type": "mooncake",
    "params": {...},
    "per_op_workers": {
        "lookup": 1,    # lookup 极快但延迟敏感
        "retrieve": 2,  # retrieve 较重
        "store": 4,     # store 最重但可容忍延迟
        "delete": 1
    }
}'
```

防止 store 突发拖垮 lookup 的 p99 延迟（[LMCache 实测 lookup p99 下降 35 倍](https://blog.lmcache.ai/zh/2026/05/26/%e5%bd%93%e5%bc%80%e6%ba%90%e9%81%87%e8%a7%81%e5%bc%80%e6%ba%90%ef%bc%9almcache-%e4%b8%8e-mooncake-%e7%9a%84%e4%b8%80%e6%ac%a1%e5%8f%8c%e5%90%91%e5%a5%94%e8%b5%b4/)）。

### 8.6 KVCM Server 端的存储后端扩展

对于 KVCM Server（中心）侧的存储后端注册，仍建议引入注册表模式消除 switch/if-else：

```cpp
// storage_backend_registry.h
#define REGISTER_STORAGE_BACKEND(type_name, BackendClass, SpecClass) \
    static bool _reg_##BackendClass = [] {                           \
        StorageBackendRegistry::Instance().RegisterBackend(          \
            type_name, ...);                                         \
        return true;                                                  \
    }()

// 各后端 .cc 底部一行自注册
REGISTER_STORAGE_BACKEND("vineyard", VineyardBackend, VineyardStorageSpec);
```

这是增量改进，不影响 LocalServer 的开发。

## 9. EventBus 设计

### 9.1 Python 实现（借鉴 [LMCache EventBus](https://github.com/LMCache/LMCache/blob/714bd7a58af2/lmcache/v1/mp_observability/event_bus.py)）

```python
from enum import IntEnum
from collections import deque
from dataclasses import dataclass
import threading
import time

class EventType(IntEnum):
    STORE_START = 1
    STORE_FINISH = 2
    RETRIEVE_START = 3
    RETRIEVE_FINISH = 4
    LOOKUP_HIT = 5
    LOOKUP_MISS = 6
    EVICTION = 7
    REMOTE_STORE_FINISH = 8
    REMOTE_RETRIEVE_FINISH = 9
    WRITEBACK = 10            # L1 驱逐回写到远端存储完成

@dataclass
class CacheEvent:
    type: EventType
    timestamp_ns: int
    block_hashes: list[int]
    medium: str = ""       # "mem" / "disk" / "remote"
    storage_name: str = "" # 具体后端名

class EventSubscriber:
    def get_subscriptions(self) -> dict[EventType, callable]:
        raise NotImplementedError

class EventBus:
    def __init__(self, max_queue_size=10000):
        self._queue = deque(maxlen=max_queue_size)
        self._subscribers: dict[EventType, list[callable]] = {}
        self._drain_thread = threading.Thread(
            target=self._drain_loop, daemon=True)
        self._running = True
        self._drain_thread.start()

    def subscribe(self, subscriber: EventSubscriber):
        for event_type, handler in subscriber.get_subscriptions().items():
            self._subscribers.setdefault(event_type, []).append(handler)

    def publish(self, event: CacheEvent):
        if event.type not in self._subscribers:
            return  # has_subscribers 前置检查
        self._queue.append(event)

    def _drain_loop(self):
        while self._running:
            while self._queue:
                event = self._queue.popleft()
                for handler in self._subscribers.get(event.type, []):
                    handler(event)
            time.sleep(0.001)  # 1ms drain interval
```

### 9.2 订阅者

```python
class MetricsSubscriber(EventSubscriber):
    """Prometheus 指标"""
    def get_subscriptions(self):
        return {
            EventType.STORE_FINISH: self._on_store,
            EventType.RETRIEVE_FINISH: self._on_retrieve,
            EventType.LOOKUP_HIT: self._on_hit,
            EventType.LOOKUP_MISS: self._on_miss,
            EventType.EVICTION: self._on_evict,
        }

class ReportSubscriber(EventSubscriber):
    """KVCM 元数据上报（仅 Managed 模式）"""
    def get_subscriptions(self):
        return {
            EventType.STORE_FINISH: self._on_store,
            EventType.EVICTION: self._on_evict,
        }

    def _on_store(self, event):
        self.batch_queue.append(("BLOCK_ADD", event.block_hashes))
        if len(self.batch_queue) >= self.batch_size:
            self._flush()
```

## 10. 完整落地路线图

### Phase 1：最小可用的 Standalone LocalServer（2-3 周）

1. 搭建 Python 项目框架（FastAPI HTTP + ZMQ Server）
2. 实现 `L1MemConnector` C++ native（SHM slab allocator + pybind）
3. 实现请求路由：REGISTER / PREPARE_STORE / COMMIT_STORE / PREPARE_RETRIEVE / COMMIT_RETRIEVE / LOOKUP
4. 实现 `CacheIndex`（内存索引）+ `EvictionController`（LRU）
5. 实现 vLLM 轻量 Connector
6. 端到端验证：vLLM → Connector → LocalServer → SHM 读写

**交付标准：** Standalone 模式下 vLLM 可以通过 LocalServer 完成 KVCache 的本地存储和读取。

### Phase 2：L1 磁盘 + 远端存储集成（2 周）

1. 实现 `L1DiskConnector` C++ native（O_DIRECT file I/O + pybind）
2. 将现有 C++ SDK（Hf3fs / Mooncake / LocalFile）封装为 `ConnectorBase<T>` 子类
3. 实现 `RemoteStorageManager`（Python，集成 `KvCacheManagerClient`）
4. 实现远端存储的 prepare_retrieve fallback 路径

### Phase 3：KVCM 对接 + Managed 模式（1-2 周）

1. 实现 `EventReporter`（ReportEvent + HeartbeatLoop）
2. 实现 `EventBus` + MetricsSubscriber + ReportSubscriber
3. KVCM Server 端适配：LocalServer 的 ReportEvent 处理
4. Managed 模式端到端验证

### Phase 4：扩展性 + 多引擎支持（1-2 周）

1. 实现 `native_plugin` 动态加载机制
2. 实现 SGLang / TRT-LLM 轻量 Connector
3. 编写外部 Connector 模板 `examples/external_connector_template/`
4. Per-Op Worker Pools 支持

## 附录

### A. LMCache 可借鉴的设计模式

| # | 模式 | LMCache 实现 | LocalServer 应用 |
|---|------|-------------|-----------------|
| 1 | MP 独立进程 | [`MPCacheServer`](https://github.com/LMCache/LMCache/blob/714bd7a58af2/lmcache/v1/multiprocess/server.py) | LocalServer 独立进程 |
| 2 | ZMQ 消息队列 | `MessageQueueServer` ROUTER/DEALER | 推理引擎 ↔ LocalServer 控制面 |
| 3 | SHM 零拷贝 | [`EngineDrivenContext` SHM](https://github.com/LMCache/LMCache/blob/714bd7a58af2/lmcache/v1/multiprocess/transfer_context/shm.py) + `ShmSlotDescriptor` | L1 内存传输 |
| 4 | 两阶段协议 | [`PREPARE/COMMIT`](https://github.com/LMCache/LMCache/blob/714bd7a58af2/lmcache/v1/multiprocess/protocols/base.py) | prepare_store/commit_store |
| 5 | Native Connector 框架 | [`IStorageConnector`](https://github.com/LMCache/LMCache/blob/714bd7a58af2/csrc/storage_backends/connector_interface.h) + [`ConnectorBase<T>`](https://github.com/LMCache/LMCache/blob/714bd7a58af2/csrc/storage_backends/connector_base.h) | 统一的 C++ 存储接口 |
| 6 | pybind 宏 | [`connector_pybind_utils.h`](https://github.com/LMCache/LMCache/blob/714bd7a58af2/csrc/storage_backends/connector_pybind_utils.h) | `KVCM_BIND_CONNECTOR_METHODS` |
| 7 | 动态插件加载 | [`native_plugin_l2_adapter.py`](https://github.com/LMCache/LMCache/blob/714bd7a58af2/lmcache/v1/distributed/l2_adapters/native_plugin_l2_adapter.py) | `native_plugin` 第三方扩展 |
| 8 | 配置透传 | [`mooncake_store_l2_adapter.py`](https://github.com/LMCache/LMCache/blob/714bd7a58af2/lmcache/v1/distributed/l2_adapters/mooncake_store_l2_adapter.py) | Connector params 不透明传递 |
| 9 | Per-Op Workers | [`WorkerPoolConfig`](https://github.com/LMCache/LMCache/blob/714bd7a58af2/csrc/storage_backends/connector_base.h) | 读写隔离防止 p99 劣化 |
| 10 | EventBus | [`event_bus.py`](https://github.com/LMCache/LMCache/blob/714bd7a58af2/lmcache/v1/mp_observability/event_bus.py) deque + drain thread | 观测/上报解耦 |
| 11 | Coordinator 弹性注册 | [`registrar.py`](https://github.com/LMCache/LMCache/blob/714bd7a58af2/lmcache/v1/mp_coordinator/registrar.py) 注册/重试 | HeartbeatLoop 断线重连 |

### B. 关键参考文件

**LMCache**（基于 commit [`714bd7a`](https://github.com/LMCache/LMCache/tree/714bd7a58af2)，2026-06-16）：
- [`lmcache/v1/multiprocess/server.py`](https://github.com/LMCache/LMCache/blob/714bd7a58af2/lmcache/v1/multiprocess/server.py) — MP Server 组装（`MPCacheServer` compositor）
- [`lmcache/v1/multiprocess/transfer_context/shm.py`](https://github.com/LMCache/LMCache/blob/714bd7a58af2/lmcache/v1/multiprocess/transfer_context/shm.py) — SHM 零拷贝实现（`EngineDrivenContext` + `ShmSlotDescriptor`）
- [`lmcache/v1/multiprocess/protocols/base.py`](https://github.com/LMCache/LMCache/blob/714bd7a58af2/lmcache/v1/multiprocess/protocols/base.py) — `RequestType` 枚举 + `PREPARE/COMMIT` 协议定义
- [`lmcache/v1/mp_observability/event_bus.py`](https://github.com/LMCache/LMCache/blob/714bd7a58af2/lmcache/v1/mp_observability/event_bus.py) — EventBus 实现（deque + drain thread + subscriber ABC）
- [`lmcache/v1/mp_coordinator/`](https://github.com/LMCache/LMCache/tree/714bd7a58af2/lmcache/v1/mp_coordinator) — Coordinator：fleet 级 instance 注册/心跳/驱逐/quota 管理
  - [`app.py`](https://github.com/LMCache/LMCache/blob/714bd7a58af2/lmcache/v1/mp_coordinator/app.py) — FastAPI 应用工厂 + lifespan
  - [`registry.py`](https://github.com/LMCache/LMCache/blob/714bd7a58af2/lmcache/v1/mp_coordinator/registry.py) — `InstanceRegistry`（线程安全的 instance 注册表）
  - [`registrar.py`](https://github.com/LMCache/LMCache/blob/714bd7a58af2/lmcache/v1/mp_coordinator/registrar.py) — mp server 侧的注册/心跳/断线重连 helper
  - [设计文档](https://github.com/LMCache/LMCache/blob/714bd7a58af2/docs/design/v1/mp_coordinator/README.md) — REST API、请求流程、健康检查循环
- [`csrc/storage_backends/connector_interface.h`](https://github.com/LMCache/LMCache/blob/714bd7a58af2/csrc/storage_backends/connector_interface.h) — `IStorageConnector` C++ 纯虚接口
- [`csrc/storage_backends/connector_base.h`](https://github.com/LMCache/LMCache/blob/714bd7a58af2/csrc/storage_backends/connector_base.h) — `ConnectorBase<T>` CRTP 模板 + `WorkerPoolConfig`
- [`csrc/storage_backends/connector_pybind_utils.h`](https://github.com/LMCache/LMCache/blob/714bd7a58af2/csrc/storage_backends/connector_pybind_utils.h) — pybind 绑定宏
- [`lmcache/v1/distributed/l2_adapters/native_plugin_l2_adapter.py`](https://github.com/LMCache/LMCache/blob/714bd7a58af2/lmcache/v1/distributed/l2_adapters/native_plugin_l2_adapter.py) — 第三方插件动态加载
- [`lmcache/v1/distributed/l2_adapters/mooncake_store_l2_adapter.py`](https://github.com/LMCache/LMCache/blob/714bd7a58af2/lmcache/v1/distributed/l2_adapters/mooncake_store_l2_adapter.py) — Mooncake L2 adapter（配置透传典范）
- [`examples/lmc_external_native_connector/`](https://github.com/LMCache/LMCache/tree/714bd7a58af2/examples/lmc_external_native_connector) — 外部插件模板

**KVCM:**
- `kv_cache_manager/client/` — 现有 TransferClient + SDK（待重构为 ConnectorBase 子类）
- `kv_cache_manager/py_connector/` — 现有引擎 Connector（待简化为轻量 Connector）
- `kv_cache_manager/py_connector/common/manager_client.py` — HTTP client（直接复用）
- `kv_cache_manager/data_storage/vineyard_backend.cc` — 节点注册/心跳参考
- `kv_cache_manager/manager/cache_manager.cc` — ReportEvent 处理逻辑

### C. 推荐阅读

**LMCache 博客：**

1. [Understanding LMCache MP Mode Transfer Paths: A Beginner's Guide](https://blog.lmcache.ai/en/2026/06/15/understanding-lmcache-mp-mode-transfer-paths-a-beginners-guide/) — 详解 LMCache MP 模式下 KVCache 在推理引擎和 MP Server 之间的传输路径：CUDA IPC（GPU 共享内存句柄）、SHM（共享内存 1 copy）、Pickle（通用 fallback 4 copies）。LocalServer 的 SHM 零拷贝设计直接借鉴了这里的 Non-CUDA SHM 路径。
2. [当开源遇见开源：LMCache 与 Mooncake 的一次双向奔赴](https://blog.lmcache.ai/zh/2026/05/26/%e5%bd%93%e5%bc%80%e6%ba%90%e9%81%87%e8%a7%81%e5%bc%80%e6%ba%90%ef%bc%9almcache-%e4%b8%8e-mooncake-%e7%9a%84%e4%b8%80%e6%ac%a1%e5%8f%8c%e5%90%91%e5%a5%94%e8%b5%b4/) — 记录 LMCache × Mooncake 的联合开发过程：Native Connector 框架重构 → 文件系统 adapter → 动态加载机制 → Mooncake RDMA adapter → L1 预注册 → batch 操作 → Per-Op Worker Pools。本文档中"配置透传"、"零侵入扩展"、"Per-Op Workers" 等设计均源自此博客描述的实践。

**LMCache Coordinator 设计：**

3. [MP Coordinator 设计文档](https://github.com/LMCache/LMCache/blob/714bd7a58af2/docs/design/v1/mp_coordinator/README.md) — FastAPI REST API 的 fleet 级 coordinator：instance 注册/心跳/驱逐、quota reconcile、blend-lookup routing。LocalServer 的弹性注册（§7.3）和 KVCM 心跳超时处理（§7.4）借鉴了这里的模式。
4. [MP Coordinator 代码入口](https://github.com/LMCache/LMCache/tree/714bd7a58af2/lmcache/v1/mp_coordinator) — `app.py`（FastAPI 应用）、`registry.py`（InstanceRegistry）、`registrar.py`（mp server 侧注册 helper）、`http_apis/`（auto-discovered REST routers）
