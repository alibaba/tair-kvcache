# LocalStorageBackend 设计方案

## 1. 背景与目标

### 问题

- 部分推理环境没有 RDMA 网络，不适合走 TairMempool/Mooncake 等远程存储后端
- 现有 NFS backend + local_file_sdk 仅适用于单机场景，多机部署时存在 key 冲突（各节点 MetaIndexer 隔离，无法跨机协调）
- 缺少路由感知能力，推理引擎调度器不知道哪个节点上已有匹配的 KVCache

### 目标

1. 支持多推理节点各自使用本地内存/SSD 存储 KVCache
2. KVCM 集中管理全局元数据，知道每个节点上有哪些 key
3. 提供 Router 接口，给调度器建议应该访问哪个节点
4. 支持全局 LRU 淘汰

### 核心设计原则

- 节点是"被 KVCM 寻址的内存"，不做分配/淘汰决策
- KVCM 管 free_list（逻辑分配/回收），节点只管物理读写
- 尽量复用现有 StartWriteCache/FinishWriteCache/CacheReclaimer 流程，减少协议变更

---

## 2. 架构总览

```
                 ┌────────────────────────────────────┐
                 │       Central KVCM Server          │
                 │                                    │
                 │  ┌──────────────┐  ┌───────────┐  │
                 │  │ MetaIndexer  │  │  Router   │  │
                 │  │ (全局元数据)  │  │  (路由)   │  │
                 │  └──────────────┘  └───────────┘  │
                 │                                    │
                 │  ┌──────────────────────────────┐  │
                 │  │   LocalStorageBackend         │  │
                 │  │  - 节点注册表 (NodeRegistry)   │  │
                 │  │  - per-node free_list (block级) │  │
                 │  │  - 心跳 & 存活性检查          │  │
                 │  │  - Create = pop slot          │  │
                 │  │  - Delete = push slot         │  │
                 │  └──────────────────────────────┘  │
                 │                                    │
                 │  ┌──────────────┐                  │
                 │  │CacheReclaimer│ (原有流程不变)    │
                 │  └──────────────┘                  │
                 └───────────────┬────────────────────┘
                                 │ gRPC
            ┌────────────────────┼────────────────────┐
            │                    │                    │
     ┌──────┴──────┐     ┌──────┴──────┐     ┌──────┴──────┐
     │   Node A    │     │   Node B    │     │   Node C    │
     │ local_mem   │     │ local_mem   │     │ local_mem   │
     │ _sdk        │     │ _sdk        │     │ _sdk        │
     │             │     │             │     │             │
     │ 预分配 pool │     │ 预分配 pool │     │ 预分配 pool │
     │ 按URI读写   │     │ 按URI读写   │     │ 按URI读写   │
     │ Inference   │     │ Inference   │     │ Inference   │
     └─────────────┘     └─────────────┘     └─────────────┘
```

---

## 3. 数据流

### 3.1 节点注册

```
Node A 启动:
  1. local_mem_sdk.Init(config):
     - 读取 spec 布局（从 model config 或 RegisterInstance 响应）
     - 按 spec 布局计算每个 block 占用字节数
     - 分配内存池 (mmap /dev/shm 或 SSD 文件)
     - 计算 total_blocks = pool_size / per_block_size

  2. RegisterInstance(..., node_id, node_endpoint, node_total_blocks, storage_unique_name):
     - KVCM 的 RegisterInstance handler 检测到 node_id 非空:
       - 记录节点信息 (endpoint, available=true)
       - 为该节点初始化 free_list (block 级, 所有 spec 共享)
       - free_list: {0, 1, 2, ..., total_blocks-1}
```

### 3.2 写入流程（与远程存储完全一致）

```
  1. StartWriteCache(keys, node_id="nodeA")
     → RequestContext.set_target_node_id("nodeA")    // 入口处设置
     → FilterWriteCache (node-aware): 只有 nodeA 本地有的 key 才算 exists
     → GenWriteLocation:
       → DataStorageSelector 选中 LocalStorageBackend
       → LocalStorageBackend::Create:
         - 从 RequestContext 取 target_node_id
         - 从 nodeA 的 free_list 中 pop slot
         - 生成 URI: local://local_pool_01/nodeA/TP0/42?offset=172032&size=4096
           (注: hostname 会被 DataStorageManager::Create 覆写为 unique_name,
            所以 node_id 放在 path 中而非 hostname)
     → 返回 block_mask + URIs

  2. Client (Node A):
     - 解析 URI 中的 offset 和 size
     - memcpy(pool_base + offset, gpu_kv_buffer, size)

  3. FinishWriteCache(session_id, success_blocks):
     - 原流程: CLS_WRITING → CLS_SERVING
```

### 3.3 读取流程

```
  1. GetCacheLocation(keys) 或 GetRoutingSuggestion(keys)
     → MetaSearcher 查到 location: local://local_pool_01/nodeA/TP0/42?...

  2. Client 判断 location 是否在本地节点:
     - 在本节点: 本地读 memcpy(gpu_buffer, pool_base + offset, size)
     - 不在本节点: 视为未命中 (key 不存在), 需要重新计算或走 Router 调度
```

> **设计决策**: LOCAL 模式下不支持跨节点数据拉取。数据只在产生它的节点上有意义——如果调度器把请求路由到了另一个节点，该节点的本地池中没有这份缓存，直接按 cache miss 处理。跨节点传输（Phase 3）作为可选扩展，不在基础设计中。

### 3.4 淘汰流程（原有 CacheReclaimer 不变）

```
  1. CacheReclaimer 定时触发
  2. 计算 WaterLevelExceed (基于 MetaIndexer 中的 key 数量 / 容量配额,
     与现有 CacheReclaimer 流程完全一致, 不需要节点上报 usage_ratio)
  3. 采样最冷的 keys
  4. MetaIndexer 删除元数据
  5. DataStorageManager::Delete → LocalStorageBackend::Delete
     → 将 slot 推回对应 node 的 free_list
  6. 不需要通知节点，该 slot 下次被 Create 时自然被覆盖
```

> **设计决策**: KVCM 完全负责淘汰，节点不参与淘汰决策。KVCM 从 MetaIndexer 删除元数据 + 回收 free_list slot 后，该物理内存即视为可复用——节点侧无需做任何清理，下次写入时直接覆盖。

---

## 4. 需要改动的文件清单

### 4.1 Proto 定义

**文件**: `github-opensource/kv_cache_manager/protocol/protobuf/meta_service.proto`

| 改动 | 说明 |
|------|------|
| `StorageType` 枚举新增 `ST_LOCAL = 8` | 新存储类型标识 |
| `LocalStorageSpec` 消息体填充字段 | 现有 `LocalStorageSpec` 是空消息（三个 proto 文件中均为 `message LocalStorageSpec {}`），且无任何 C++ 代码引用，直接复用 |
| `RegisterInstanceRequest` 新增节点注册字段（仅 LOCAL 模式使用） | 见下方详细定义，替代独立的 `RegisterNodeRequest` |
| `StartWriteCacheRequest` 新增 `string node_id = 7` | 标识发起写入的节点 |
| 新增 `NodeHeartbeatRequest` / `NodeHeartbeatResponse` | 心跳 |
| 新增 `GetRoutingSuggestionRequest` / `GetRoutingSuggestionResponse` | Router API |
| `MetaService` 新增 RPC: `NodeHeartbeat`, `GetRoutingSuggestion` | |

> **注意**: `LocalStorageSpec local = 2` 已存在于 `StorageConfig.storage_spec` oneof 中（三个 proto 文件都有），但 `LocalStorageSpec` 消息体为空且无任何 C++ 代码引用（`StorageFromProto` 不处理 `kLocal`，无对应 C++ StorageSpec 子类）。直接复用 field 2 并填充字段即可，**不需要新增 field 10**。三个 proto 文件（`meta_service.proto`、`admin_service.proto`、`kv_meta_service.proto`）中的 `LocalStorageSpec` 都需要保持一致。

**LocalStorageSpec 定义**:

```protobuf
message LocalStorageSpec {
    int64 heartbeat_timeout_ms = 1;    // 心跳超时阈值，超时判定节点不可用
}
```

**RegisterInstanceRequest 扩展（仅 LOCAL 模式使用）**:

节点注册信息直接放在 `RegisterInstanceRequest` 中，不需要独立的 `RegisterNodeRequest` RPC。推理节点在 `RegisterInstance` 时顺带完成节点注册。以下字段仅在 LOCAL 模式下有意义，非 LOCAL 模式下不填（空字符串 / 0）:

```protobuf
message RegisterInstanceRequest {
    // --- 现有字段 ---
    string trace_id = 1;
    string instance_group = 2;
    string instance_id = 3;
    int32 block_size = 4;
    repeated LocationSpecInfo location_spec_infos = 5;  // 已有: spec name + size
    ModelDeployment model_deployment = 6;
    repeated LocationSpecGroup location_spec_groups = 7;

    // --- 新增: LOCAL 模式节点注册 (field 8-11) ---
    string node_id = 8;                // 节点唯一标识 (e.g. "node-192-168-1-10")
    string node_endpoint = 9;          // 数据传输地址 (ip:port)
    uint32 node_total_blocks = 10;     // 该节点的 block 总数 (pool_size / per_block_size)
    string storage_unique_name = 11;   // 对应哪个 LocalStorageBackend 实例
}
```

> **设计决策**: `RegisterNodeRequest` 被合并到 `RegisterInstanceRequest` 中，原因：
> 1. `location_spec_infos` 已经携带了 spec name 和 size，与 `SpecSlotInfo` 重叠
> 2. `total_slots` 所有 spec 相同（block 级管理），只需一个 `node_total_blocks` 字段
> 3. 节点注册是 `RegisterInstance` 的自然伴随动作，减少一次 RPC 调用
> 4. KVCM 侧 `RegisterInstance` handler 检查 `node_id` 是否非空，非空则额外调用 `LocalStorageBackend::RegisterNode`

**NodeHeartbeatRequest 定义**:

```protobuf
message NodeHeartbeatRequest {
    string node_id = 1;
    int32 inference_load = 2;         // 推理负载 (可选, 用于路由均衡)
}
```

**GetRoutingSuggestionRequest/Response 定义**:

```protobuf
message GetRoutingSuggestionRequest {
    string trace_id = 1;
    string instance_id = 2;
    repeated int64 block_keys = 3;
    repeated int64 token_ids = 4;
    int32 top_k_nodes = 5;
}

message GetRoutingSuggestionResponse {
    ResponseHeader header = 1;
    repeated NodeSuggestion suggestions = 2;
}

message NodeSuggestion {
    string node_id = 1;
    string node_endpoint = 2;
    int32 matched_prefix_len = 3;      // 该节点匹配的最长前缀 block 数
    int32 total_matched_blocks = 4;    // 该节点总命中 block 数
    double node_load = 5;              // 节点负载
}
```

---

### 4.2 存储后端

#### 4.2.1 枚举与配置

**文件**: `github-opensource/kv_cache_manager/data_storage/storage_config.h`

| 改动 | 说明 |
|------|------|
| `DataStorageType` 枚举新增 `DATA_STORAGE_TYPE_LOCAL = 8` | |
| `ToString(DataStorageType)` 新增 case | 返回 `"local"` |
| 新增 `LocalStorageSpec` 类 (C++ 侧) | 继承 `StorageSpec`，字段对应 proto |

#### 4.2.2 LocalStorageBackend 实现

**新建文件**:
- `github-opensource/kv_cache_manager/data_storage/local_storage_backend.h`
- `github-opensource/kv_cache_manager/data_storage/local_storage_backend.cc`

**类设计**:

```cpp
class LocalStorageBackend : public DataStorageBackend {
public:
    LocalStorageBackend(std::shared_ptr<MetricsRegistry> metrics_registry)
        : DataStorageBackend(std::move(metrics_registry)) {}

    DataStorageType GetType() override { return DATA_STORAGE_TYPE_LOCAL; }
    bool Available() override;

    // --- 节点管理 ---
    ErrorCode RegisterNode(const std::string& node_id,
                           const std::string& endpoint,
                           const std::vector<SpecSlotInfo>& spec_slots);
    ErrorCode UnregisterNode(const std::string& node_id);
    ErrorCode OnHeartbeat(const std::string& node_id, int32_t load);

    // --- DataStorageBackend 接口实现 ---
    ErrorCode DoOpen(const StorageConfig& config, const std::string& trace_id) override;
    ErrorCode Close() override;

    // Create: 从 RequestContext 取 target_node_id, 从该 node 的 free_list 分配 slot
    // 返回值: 每个 key 对应一个 pair<ErrorCode, DataStorageUri>
    // 注: DataStorageManager::Create 会将 URI hostname 覆写为 unique_name,
    //     因此 node_id 放在 URI path 中: local://<unique_name>/<node_id>/<spec>/<slot>
    std::vector<std::pair<ErrorCode, DataStorageUri>> Create(
        const std::vector<std::string>& keys,
        size_t size_per_key,
        const std::string& trace_id,
        std::function<void()> cb) override;

    // Delete: 将 slot 归还到对应 node 的 free_list
    // 返回值: 每个 URI 对应一个 ErrorCode
    std::vector<ErrorCode> Delete(
        const std::vector<DataStorageUri>& storage_uris,
        const std::string& trace_id,
        std::function<void()> cb) override;

    // Exist: 检查 URI 对应的节点是否存活
    std::vector<bool> Exist(const std::vector<DataStorageUri>& uris) override;

    // MightExist: 同 Exist (本地存储无布隆过滤器, 精确判断)
    std::vector<bool> MightExist(const std::vector<DataStorageUri>& uris) override;

    // Lock/UnLock: 本地存储不需要分布式锁, 返回全成功
    std::vector<ErrorCode> Lock(const std::vector<DataStorageUri>& uris) override;
    std::vector<ErrorCode> UnLock(const std::vector<DataStorageUri>& uris) override;

    double GetStorageUsageRatio(const std::string& trace_id) const override;

    // --- Router 辅助 ---
    std::vector<NodeInfo> GetAvailableNodes() const;
    bool IsNodeAvailable(const std::string& node_id) const;

private:
    struct NodePool {
        std::string node_id;
        std::string endpoint;
        bool available = true;
        int32_t inference_load = 0;
        int64_t last_heartbeat_ms = 0;
        uint64_t per_block_size;    // 所有 spec size 之和
        uint32_t total_blocks;
        std::queue<uint32_t> free_blocks;
        std::mutex mutex;

        struct SpecLayout {
            std::string name;
            uint64_t size;
            uint64_t offset_in_block;
        };
        std::vector<SpecLayout> spec_layouts;
    };

    // 用 NodeInfo 别名方便文档引用
    using NodeInfo = NodePool;

    mutable std::shared_mutex nodes_mutex_;
    std::unordered_map<std::string, NodePool> nodes_;

    // 心跳超时检测
    std::unique_ptr<std::thread> liveness_checker_thread_;
    std::atomic<bool> running_{false};
    int64_t heartbeat_timeout_ms_ = 30000;

    void LivenessCheckerLoop();
    void OnNodeUnavailable(const std::string& node_id);
};
```

> **并发安全约定**: `RegisterNode`/`UnregisterNode`/`OnHeartbeat`/`OnNodeUnavailable` 使用 `unique_lock`（写）；`Exist`/`MightExist`/`GetAvailableNodes`/`IsNodeAvailable`/`GetStorageUsageRatio` 使用 `shared_lock`（读）；`LivenessCheckerLoop` 检查时用 `shared_lock`，标记不可用时升级为 `unique_lock`。

#### 4.2.3 Create 实现细节

```cpp
// ★ 关键设计: node_id 通过 RequestContext 传递
//
// 现有 DataStorageBackend::Create 接口没有 node_id 参数,
// 但 DataStorageManager::Create 接受 RequestContext* 参数。
// 在 StartWriteCache 入口处将 node_id 设置到 RequestContext,
// LocalStorageBackend::Create 通过 RequestContext 读取。
//
// 但 LocalStorageBackend::Create 签名中没有 RequestContext 参数!
// → 需要在 DataStorageManager::Create 调用后端 Create 前,
//   将 target_node_id 从 RequestContext 取出, 通过 thread_local 传递给后端。
//
// 实现方案: DataStorageManager 中增加 thread_local 变量:
//   thread_local std::string tl_target_node_id;
// 在 Create 中设置, LocalStorageBackend 读取后清空。
//
// URI 格式: local://<unique_name>/<node_id>/<spec_name>/<slot_id>?offset=...&size=...
//   例: local://local_pool_01/node-192-168-1-10/TP0/42?offset=172032&size=4096
//
// 注: hostname 会被 DataStorageManager::Create 覆写为 unique_name (见 data_storage_manager.cc L206),
//     所以 node_id 不能放在 hostname, 必须放在 path 中。

std::vector<std::pair<ErrorCode, DataStorageUri>> LocalStorageBackend::Create(
    const std::vector<std::string>& keys,
    size_t size_per_key,
    const std::string& trace_id,
    std::function<void()> cb) {

    // 1. 从 thread_local 获取目标 node_id
    std::string node_id = tl_target_node_id;
    if (node_id.empty()) {
        // 非 LOCAL 场景或异常, 返回全失败
        return std::vector(keys.size(), {EC_INVALID_ARGUMENT, {}});
    }

    std::shared_lock lock(nodes_mutex_);
    auto it = nodes_.find(node_id);
    if (it == nodes_.end() || !it->second.available) {
        return std::vector(keys.size(), {EC_STORAGE_NOT_AVAILABLE, {}});
    }

    auto& pool = it->second;
    std::lock_guard pool_lock(pool.mutex);

    // 2. 对同一 block_key 的所有 spec 只 pop 一次 free_list
    //    key 格式: "instance_id/spec_name/hex(block_key)"
    std::unordered_map<std::string, uint32_t> block_key_to_slot;
    std::vector<std::pair<ErrorCode, DataStorageUri>> results;
    results.reserve(keys.size());

    for (const auto& key : keys) {
        auto [spec_name, block_key_hex] = ParseKey(key);

        uint32_t slot;
        if (auto slot_it = block_key_to_slot.find(block_key_hex);
            slot_it != block_key_to_slot.end()) {
            slot = slot_it->second;  // 同 block_key 复用已分配的 slot
        } else {
            if (pool.free_blocks.empty()) {
                results.push_back({EC_NO_SPACE, {}});
                continue;
            }
            slot = pool.free_blocks.front();
            pool.free_blocks.pop();
            block_key_to_slot[block_key_hex] = slot;
        }

        // 3. 生成 URI (PascalCase: SetProtocol/SetHostName/SetPath/SetParam)
        auto& layout = FindSpecLayout(pool, spec_name);
        uint64_t offset = slot * pool.per_block_size + layout.offset_in_block;

        DataStorageUri uri;
        uri.SetProtocol("local");
        // hostname 会被 DataStorageManager::Create 覆写为 unique_name, 这里先设为空
        uri.SetHostName("");
        uri.SetPath("/" + node_id + "/" + spec_name + "/" + std::to_string(slot));
        uri.SetParam("offset", std::to_string(offset));
        uri.SetParam("size", std::to_string(layout.size));
        uri.SetParam("endpoint", pool.endpoint);

        results.push_back({EC_OK, uri});
    }

    if (cb) cb();
    return results;
}
```

#### 4.2.4 工厂注册

**文件**: `github-opensource/kv_cache_manager/data_storage/data_storage_manager.cc`

| 改动 | 说明 |
|------|------|
| `CreateStorageBackend` switch 新增 `DATA_STORAGE_TYPE_LOCAL` case | 返回 `make_shared<LocalStorageBackend>()` |
| `#include "local_storage_backend.h"` | |

**文件**: `github-opensource/kv_cache_manager/data_storage/storage_config.h`

| 改动 | 说明 |
|------|------|
| `ToBaseType` 映射表，LOCAL 映射到自身 | |

---

### 4.3 CacheManager 层改动

**文件**: `github-opensource/kv_cache_manager/manager/cache_manager.h`

| 改动 | 说明 |
|------|------|
| `StartWriteCache` 签名新增 `const std::string& node_id` 参数 | |
| `FilterWriteCache` 签名新增 `const std::string& node_id` 参数 | |
| 新增 `RegisterNode` 方法 | 委托给 LocalStorageBackend |
| 新增 `NodeHeartbeat` 方法 | 委托给 LocalStorageBackend |
| 新增 `GetRoutingSuggestion` 方法 | 实现路由逻辑 |

**文件**: `github-opensource/kv_cache_manager/common/request_context.h`

| 改动 | 说明 |
|------|------|
| 新增 `target_node_id()` / `set_target_node_id()` | 用于在 StartWriteCache 流程中传递 node_id |

**文件**: `github-opensource/kv_cache_manager/data_storage/data_storage_manager.cc`

| 改动 | 说明 |
|------|------|
| 新增 `thread_local std::string tl_target_node_id` | 在 `Create` 中设置/清空，供 LocalStorageBackend 读取 |

**文件**: `github-opensource/kv_cache_manager/manager/cache_manager.cc`

#### 4.3.1 FilterWriteCache 改动

`FilterWriteCache` 中的 `existsForWrite` lambda 需要根据是否有 `location_spec_group_names` 选择调用 `ExistsForWrite` 的哪个重载。**两个重载都需要新增 `node_id` 参数**：

```cpp
// 现有 lambda (cache_manager.cc ~L792-807):
auto existsForWrite =
    [&](size_t i, const CacheLocationMap &m,
        std::vector<std::string> &out_prune_loc_ids) -> bool {
    if (!instance_info || i >= location_spec_group_names.size()
        || location_spec_group_names[i].empty()) {
        return policy->ExistsForWrite(m, check_loc_data_exist, out_prune_loc_ids);
    }
    // ...
    return policy->ExistsForWrite(m, it->spec_names(),
                                  check_loc_data_exist, out_prune_loc_ids);
};

// 改为: 两个重载都新增 node_id 参数
auto existsForWrite =
    [&](size_t i, const CacheLocationMap &m,
        std::vector<std::string> &out_prune_loc_ids) -> bool {
    if (!instance_info || i >= location_spec_group_names.size()
        || location_spec_group_names[i].empty()) {
        return policy->ExistsForWrite(m, node_id,           // ← 新增
                                      check_loc_data_exist, out_prune_loc_ids);
    }
    // ...
    return policy->ExistsForWrite(m, it->spec_names(), node_id,  // ← 新增
                                  check_loc_data_exist, out_prune_loc_ids);
};
```

#### 4.3.2 GenWriteLocation 中传递 node_id 到 Create

`GenWriteLocation` 调用 `DataStorageManager::Create` 时，需要让 `LocalStorageBackend::Create` 知道目标 node_id。

**问题**: `DataStorageBackend::Create` 的签名中没有 `node_id` 参数，也不能修改这个基类接口（会影响所有后端）。

**方案 — thread_local 传递**:

在 `DataStorageManager` 中增加 `thread_local` 变量，`DataStorageManager::Create` 在调用后端 `Create` 前设置它：

```cpp
// data_storage_manager.h 或 .cc 中:
thread_local std::string tl_target_node_id;

// DataStorageManager::Create 中:
std::vector<std::pair<ErrorCode, DataStorageUri>>
DataStorageManager::Create(RequestContext *request_context, ...) {
    // ... 查找 backend ...

    // 将 node_id 从 RequestContext 传到 thread_local
    tl_target_node_id = request_context->target_node_id();

    auto results = storage_backend->Create(keys, size_per_key, trace_id, cb);

    tl_target_node_id.clear();  // 用完清空
    // ... 覆写 hostname ...
    return results;
}
```

node_id 的设置在 `StartWriteCache` 入口处完成：

```cpp
// cache_manager.cc, StartWriteCache 入口:
request_context->set_target_node_id(node_id);  // 从 request proto 取
// 后续 FilterWriteCache / GenWriteLocation 自然可用
```

#### 4.3.3 GetRoutingSuggestion 实现

```cpp
ErrorCode CacheManager::GetRoutingSuggestion(
    RequestContext* request_context,
    const std::string& instance_id,
    const KeyVector& keys,
    const TokenIdsVector& tokens,
    int32_t top_k,
    std::vector<NodeSuggestion>& suggestions) {

    // 1. 用 MetaSearcher::PrefixMatch 找匹配的 locations
    //    (需要构造 BlockMask、SelectLocationPolicy 等参数)
    // 2. 从每个 location 的 URI path 解析 node_id
    //    URI 格式: local://<unique_name>/<node_id>/<spec>/<slot>
    //    → path 第一段即为 node_id
    // 3. 按 node_id 分组统计: matched_prefix_len, total_matched_blocks
    // 4. 从 LocalStorageBackend 获取各节点负载信息
    // 5. 排序:
    //    - 主排序: matched_prefix_len DESC (前缀越长越优先)
    //    - 次排序: node_load ASC (负载越低越优先)
    // 6. 返回 top_k
}
```

---

### 4.4 SelectLocationPolicy 改动

**文件**: `github-opensource/kv_cache_manager/manager/select_location_policy.h`

| 改动 | 说明 |
|------|------|
| `ExistsForWrite` 两个重载均新增 `node_id` 参数 | |
| `SelectForMatch` 增加本地优先权重 (可选) | |

**文件**: `github-opensource/kv_cache_manager/manager/select_location_policy.cc`

```cpp
// 两个重载都新增 node_id 参数, 这里以基础版为例:
bool StaticWeightSLPolicy::ExistsForWrite(
    const CacheLocationMap& location_map,
    const std::string& node_id,             // 新增参数
    CheckLocDataExistFunc check_loc_data_exist,
    std::vector<std::string>& out_prune_loc_ids) const {

    bool exists = false;
    for (const auto& kv : location_map) {
        if (!kv.second) continue;
        if (kv.second->status() == CacheLocationStatus::CLS_NOT_FOUND) continue;

        // CLS_SERVING 需要检查数据是否真实存在
        if (kv.second->status() == CacheLocationStatus::CLS_SERVING
            && check_loc_data_exist && !check_loc_data_exist(*kv.second)) {
            out_prune_loc_ids.emplace_back(kv.first);
            continue;
        }

        if (GetWeight(kv) <= 0) continue;

        // ★ 核心改动: 对 LOCAL 类型做节点过滤
        if (IsLocalStorageType(kv.second->type())) {
            // local 类型: 只有在请求节点上才算 exists
            if (!node_id.empty() && IsLocationOnNode(kv.second, node_id)) {
                exists = true;
            }
            // node_id 为空 或 不在本节点 → 不算 exists
        } else {
            // 远程类型 (3FS/Mooncake/TairMempool): 全局可达, 直接算 exists
            exists = true;
        }
    }
    return exists;
}
```

辅助函数:

```cpp
static bool IsLocalStorageType(DataStorageType type) {
    return type == DataStorageType::DATA_STORAGE_TYPE_LOCAL;
}

static bool IsLocationOnNode(const CacheLocationConstPtr& loc, const std::string& node_id) {
    // 从 location_specs 的第一个 URI 的 path 中解析 node_id
    // URI 格式: local://<unique_name>/<node_id>/<spec>/<slot>
    // path = "/<node_id>/<spec>/<slot>"
    if (loc->location_specs().empty()) return false;
    DataStorageUri uri;
    if (!uri.Parse(loc->location_specs()[0].uri())) return false;
    std::string path = uri.GetPath();  // e.g. "/node-192-168-1-10/TP0/42"
    // 跳过首 '/', 取第一段作为 node_id
    auto pos = path.find('/', 1);
    std::string uri_node_id = (pos != std::string::npos)
        ? path.substr(1, pos - 1) : path.substr(1);
    return uri_node_id == node_id;
}
```

---

### 4.5 Service 层

**文件**: `github-opensource/kv_cache_manager/service/meta_service_impl.h`

| 改动 | 说明 |
|------|------|
| 新增 `NodeHeartbeat` 方法声明 | |
| 新增 `GetRoutingSuggestion` 方法声明 | |
| `RegisterInstance` handler 扩展 | 检查 `node_id` 字段，非空时额外调用 `cache_manager_->RegisterNode` |

**文件**: `github-opensource/kv_cache_manager/service/meta_service_impl.cc`

| 改动 | 说明 |
|------|------|
| `StartWriteCache`: 从 request 取 `node_id` 传入 `cache_manager_->StartWriteCache` | |
| `RegisterInstance`: 检查 `node_id` 字段，非空时调用 `cache_manager_->RegisterNode` | 节点注册与 Instance 注册合并 |
| 新增 `NodeHeartbeat` 实现: 调用 `cache_manager_->NodeHeartbeat` | |
| 新增 `GetRoutingSuggestion` 实现: 调用 `cache_manager_->GetRoutingSuggestion` | |

**文件**: `github-opensource/kv_cache_manager/service/grpc_service/meta_service_grpc.cc`

| 改动 | 说明 |
|------|------|
| 新增 RPC handler 绑定 | NodeHeartbeat, GetRoutingSuggestion |

**文件**: `github-opensource/kv_cache_manager/service/http_service/meta_service_http.cc`

| 改动 | 说明 |
|------|------|
| 新增 HTTP 路由 (可选) | /api/heartbeat, /api/routing_suggestion |

---

### 4.6 Client SDK

#### 4.6.1 LocalMemSdk（与现有 LocalFileSdk 的关系）

现有 `LocalFileSdk`（`SdkType::LOCAL_FILE`）基于本地文件系统路径读写，适用于 NFS backend 的单节点场景。`LocalMemSdk` 是基于预分配内存池 + URI offset 的新实现，面向多节点本地内存场景。

两者**并存**，通过不同的 `SdkType` 区分：

| | LocalFileSdk | LocalMemSdk |
|---|---|---|
| SdkType | `LOCAL_FILE = 3` | `LOCAL_MEM = 4`（新增） |
| URI 协议 | `local_file://` | `local://` |
| 内存管理 | 依赖文件系统 | 预分配 mmap 内存池 |
| 多节点感知 | 无 | 有（URI 中包含 node_id） |

**新建文件**:
- `github-opensource/kv_cache_manager/client/src/internal/sdk/local_mem_sdk.h`
- `github-opensource/kv_cache_manager/client/src/internal/sdk/local_mem_sdk.cc`

**职责**:

```cpp
class LocalMemSdk : public SdkInterface {
public:
    // 初始化: 分配内存池, 计算 spec 布局
    ClientErrorCode Init(const std::shared_ptr<SdkBackendConfig>& config,
                         const std::shared_ptr<StorageConfig>& storage_config) override;

    SdkType Type() override { return SdkType::LOCAL_MEM; }

    // 读: 根据 URI 中的 offset+size, 从 pool 读到 buffer
    ClientErrorCode Get(const std::vector<DataStorageUri>& uris,
                        const BlockBuffers& local_buffers) override;

    // 写: 根据 URI 中的 offset+size, 从 buffer 写到 pool
    ClientErrorCode Put(const std::vector<DataStorageUri>& uris,
                        const BlockBuffers& local_buffers,
                        std::shared_ptr<std::vector<DataStorageUri>> actual_remote_uris) override;

protected:
    ClientErrorCode Alloc(const std::vector<DataStorageUri>& remote_uris,
                          std::vector<DataStorageUri>& alloc_uris) override;

private:
    void* pool_base_ = nullptr;      // mmap 基址
    uint64_t pool_size_ = 0;
    std::string node_id_;
};
```

核心读写逻辑:

```cpp
ClientErrorCode LocalMemSdk::Get(const std::vector<DataStorageUri>& uris,
                                 const BlockBuffers& local_buffers) {
    for (size_t i = 0; i < uris.size(); i++) {
        uint64_t offset = std::stoull(uris[i].GetParam("offset"));
        uint64_t size = std::stoull(uris[i].GetParam("size"));
        void* src = static_cast<char*>(pool_base_) + offset;

        // 拷贝到 buffer (可能是 GPU 内存, 需要对应的 memcpy)
        CopyToBuffer(local_buffers[i], src, size);
    }
    return ClientErrorCode::OK;
}

ClientErrorCode LocalMemSdk::Put(const std::vector<DataStorageUri>& uris,
                                 const BlockBuffers& local_buffers,
                                 std::shared_ptr<std::vector<DataStorageUri>> actual_remote_uris) {
    for (size_t i = 0; i < uris.size(); i++) {
        uint64_t offset = std::stoull(uris[i].GetParam("offset"));
        uint64_t size = std::stoull(uris[i].GetParam("size"));
        void* dst = static_cast<char*>(pool_base_) + offset;

        CopyFromBuffer(dst, local_buffers[i], size);
    }
    *actual_remote_uris = uris;
    return ClientErrorCode::OK;
}
```

#### 4.6.2 SDK 注册与工厂

**文件**: `github-opensource/kv_cache_manager/client/src/internal/sdk/sdk_type.h`

| 改动 | 说明 |
|------|------|
| `SdkType` 枚举新增 `LOCAL_MEM = 4` | 与现有 `LOCAL_FILE = 3` 区分 |

**文件**: SDK 工厂（`sdk_interface.h` 或 `sdk_factory`）

| 改动 | 说明 |
|------|------|
| SDK 工厂新增 `SdkType::LOCAL_MEM` 的分发 | 返回 `LocalMemSdk` 实例 |

#### 4.6.3 MetaClient 新增方法

**文件**: `github-opensource/kv_cache_manager/client/include/meta_client.h`

| 改动 | 说明 |
|------|------|
| `StartWrite` 新增 `node_id` 参数 | 透传到 StartWriteCacheRequest |
| 新增 `SendHeartbeat(...)` | |
| 新增 `GetRoutingSuggestion(...)` | |

---

### 4.7 DataStorageSelector 改动

**文件**: `github-opensource/kv_cache_manager/manager/data_storage_selector.h` / `.cc`

| 改动 | 说明 |
|------|------|
| `Select()` 逻辑新增 `CPS_ALWAYS_LOCAL` / `CPS_PREFER_LOCAL` 分支 | 选中 LOCAL 类型后端 |

**文件**: `github-opensource/kv_cache_manager/config/cache_config.h`

| 改动 | 说明 |
|------|------|
| `CachePreferStrategy` 枚举新增 `CPS_ALWAYS_LOCAL` / `CPS_PREFER_LOCAL` | |

---

### 4.8 StorageTypeWeights 更新

**文件**: `github-opensource/kv_cache_manager/manager/select_location_policy.h`

| 改动 | 说明 |
|------|------|
| `StaticWeightSLPolicy::StorageTypeWeights` 新增 `LOCAL = 8` | 高于 NFS=5，因为是零拷贝本地内存访问 |
| `default_storage_weights_` 数组扩展 | 新增 `DATA_STORAGE_TYPE_LOCAL` 索引位（`COUNT` 从 8 变为 9） |

> **注意**: `meta_searcher.h` 中也有一份 `StorageTypeWeights` 定义（`size_t` 类型，缺少 VCNS_HF3FS/VINEYARD），疑似早期遗留死代码。建议一并更新或清理，避免后续维护者混淆。

---

### 4.9 BUILD 文件

**文件**: `github-opensource/kv_cache_manager/data_storage/BUILD`

| 改动 | 说明 |
|------|------|
| 新增 `local_storage_backend` cc_library target | |
| `data_storage_manager` 依赖中增加 `local_storage_backend` | |

**文件**: `github-opensource/kv_cache_manager/client/src/internal/sdk/BUILD` (或对应位置)

| 改动 | 说明 |
|------|------|
| 新增 `local_mem_sdk` cc_library target | |

---

### 4.10 单元测试

**新建文件**:
- `github-opensource/kv_cache_manager/data_storage/test/local_storage_backend_test.cc`
- `github-opensource/kv_cache_manager/client/src/internal/sdk/test/local_mem_sdk_test.cc` (如有测试目录)

**测试用例**:

| 测试 | 验证点 |
|------|--------|
| `RegisterNodeTest` | 注册节点后 free_list 正确初始化 |
| `CreateAndDeleteTest` | Create pop slot → Delete push slot → 循环不泄漏 |
| `CreateNoSpaceTest` | free_list 耗尽时返回 EC_NO_SPACE |
| `MultiNodeTest` | 不同节点独立的 free_list，互不影响 |
| `MultiSpecTest` | 不同 spec 独立分配，offset 正确 |
| `NodeUnavailableTest` | 心跳超时 → MightExist 返回 false |
| `ExistsForWriteNodeAwareTest` | 同 key 在 nodeA 有 → nodeB 请求不算 exists |
| `ExistsForWriteRemoteTypeTest` | 远程类型 location 不受 node_id 过滤影响 |
| `LocalMemSdkGetPutTest` | 写入后读取一致 |

---

## 5. 节点侧内存布局详细设计

### 5.1 内存布局计算

```
输入:
  pool_size = 10GB
  location_spec_infos = [("TP0", 4096), ("TP1", 8192)]

计算:
  per_block_size = sum(spec.size for spec in specs) = 4096 + 8192 = 12288 bytes
  total_blocks = pool_size / per_block_size = 10GB / 12288 ≈ 873,813 blocks

布局:
  Block 0: [TP0: 4096B][TP1: 8192B]   offset 0 ~ 12287
  Block 1: [TP0: 4096B][TP1: 8192B]   offset 12288 ~ 24575
  Block 2: [TP0: 4096B][TP1: 8192B]   offset 24576 ~ 36863
  ...

TP0 of Block N: offset = N * per_block_size + 0,           size = 4096
TP1 of Block N: offset = N * per_block_size + 4096,        size = 8192
```

### 5.2 Per-Spec Base Offset

```
spec_offsets (block 内偏移):
  "TP0" → 0
  "TP1" → 4096

URI 中的绝对 offset 计算:
  offset = slot_id * per_block_size + spec_offset
```

### 5.3 向 KVCM 注册

```
RegisterInstance (携带 LOCAL 模式字段):
  node_id = "node-192-168-1-10"
  node_endpoint = "192.168.1.10:9100"
  location_spec_infos = [
    { name: "TP0", size: 4096 },
    { name: "TP1", size: 8192 },
  ]
  node_total_blocks = 873813
  storage_unique_name = "local_pool_01"
```

注意: 所有 spec 的 `total_slots` 相同（因为按 block 整体管理），KVCM 侧用一个 free_list 管所有 spec，Create 时为同一个 block_key 的各 spec 分配相同的 slot_id。

---

## 6. KVCM 侧 Free List 管理

### 6.1 实际只需要一个 free_list per node

因为同一个 block_key 的所有 spec 总是同生同灭（一起 Create、一起 Delete），所以**不需要** per-spec free_list，只需要 per-node 一个 block free_list:

```cpp
struct NodePool {
    std::string node_id;
    std::string endpoint;
    uint64_t per_block_size;    // 所有 spec size 之和
    uint32_t total_blocks;
    std::queue<uint32_t> free_blocks;  // 一个就够
    std::mutex mutex;

    // spec 布局信息 (用于生成 URI 中的 offset)
    struct SpecLayout {
        std::string name;
        uint64_t size;
        uint64_t offset_in_block;  // 该 spec 在一个 block 内的偏移
    };
    std::vector<SpecLayout> spec_layouts;
};
```

### 6.2 Create 流程 (GenWriteLocation 调用)

`GenWriteLocation` 为一个 block_key 生成多个 spec 的 URI。现有两条路径：

- **CreateInSingleBatch**: 所有 spec 的 key 合并后一次调用 `Create`，`block_key_to_slot` map 可以在单次调用内去重 ✅
- **CreateBySpec**: 每个 spec **单独调用** `Create`，`block_key_to_slot` map 在调用间不共享 ❌

> **关键约束**: LOCAL 类型必须走 `CreateInSingleBatch` 路径。否则 `CreateBySpec` 会导致同一个 block_key 的 TP0 和 TP1 在不同 `Create` 调用中分配到不同的 slot，内存布局错乱。

**实现方式**: 在 `GenWriteLocation` 中，当 `DataStorageSelector` 选中的后端类型是 `DATA_STORAGE_TYPE_LOCAL` 时，**强制合并所有 spec 的 key**，走 `CreateInSingleBatch` 路径（即使各 spec size 不同）。`LocalStorageBackend::Create` 内部根据 spec_layout 计算每个 spec 各自的 offset，不依赖 `size_per_key` 参数。

```cpp
// key 格式: "instance_id/TP0/abc123", "instance_id/TP1/abc123"
// 解析出 block_key_hex，相同 block_key_hex 复用同一个 slot_id

std::unordered_map<std::string, uint32_t> block_key_to_slot;
std::vector<std::pair<ErrorCode, DataStorageUri>> results;

for (const auto& key : keys) {
    auto [spec_name, block_key_hex] = ParseKey(key);

    uint32_t slot;
    if (auto it = block_key_to_slot.find(block_key_hex);
        it != block_key_to_slot.end()) {
        slot = it->second;  // 同 block_key 复用已分配的 slot
    } else {
        if (pool.free_blocks.empty()) {
            results.push_back({EC_NO_SPACE, {}});
            continue;
        }
        slot = pool.free_blocks.front();
        pool.free_blocks.pop();
        block_key_to_slot[block_key_hex] = slot;
    }

    auto& layout = FindSpecLayout(pool, spec_name);
    uint64_t offset = slot * pool.per_block_size + layout.offset_in_block;
    // 生成 URI ...
}
```

### 6.3 Delete 流程

```cpp
// 从 URI 中解析 slot_id，每个 block_key 只需回收一次
std::unordered_set<uint32_t> freed_slots;
for (auto& uri : uris) {
    uint32_t slot = ParseSlotFromUri(uri);
    if (freed_slots.insert(slot).second) {
        pool.free_blocks.push(slot);
    }
}
```

### 6.4 KVCM 重启恢复

KVCM 重启后 free_list 丢失。恢复策略:

**方案: 节点重新注册 + MetaIndexer 恢复**

1. KVCM 重启后，等待各节点重新 `RegisterInstance`（携带 node_id 字段，心跳断连后会触发重连注册）
2. 初始化时 free_list = {0, 1, ..., total_blocks-1}（全部可用）
3. 从 MetaIndexer（有持久化）扫描该节点的所有 SERVING/WRITING 状态的 location
4. 解析 URI 中的 slot_id，从 free_list 中剔除
5. 恢复完成

```cpp
void LocalStorageBackend::RecoverNodeFreeList(const std::string& node_id,
                                              const std::vector<uint32_t>& occupied_slots) {
    auto& pool = nodes_[node_id];
    std::unordered_set<uint32_t> occupied(occupied_slots.begin(), occupied_slots.end());
    std::queue<uint32_t> new_free;
    for (uint32_t i = 0; i < pool.total_blocks; i++) {
        if (!occupied.count(i)) {
            new_free.push(i);
        }
    }
    pool.free_blocks = std::move(new_free);
}
```

---

## 7. Router 实现细节

### 7.1 路由算法

```
GetRoutingSuggestion(instance_id, keys/tokens, top_k):
  1. 将 tokens 转为 block_keys (GenKeyVector)
  2. MetaSearcher::PrefixMatch(request_context, block_keys, mask,
                               out_locations, policy)
     → 返回匹配的 CacheLocationVector
  3. 按 node_id 聚合:
     for each matched location:
       从 URI path 解析 node_id (path 第一段)
       node_stats[node_id].matched_blocks++
       node_stats[node_id].prefix_len = max(prefix_len, contiguous_match_count)
  4. 从 LocalStorageBackend 获取各节点负载信息:
     for each node:
       suggestion.node_load = node.inference_load
  5. 排序:
     - 主排序: matched_prefix_len DESC (前缀越长越优先)
     - 次排序: node_load ASC (负载越低越优先)
  6. 返回 top_k
```

### 7.2 无匹配时的 fallback

如果没有任何节点匹配（新 prompt，从未缓存过），Router 返回按负载排序的节点列表，让调度器选最空闲的节点。

---

## 8. 配置示例

### 8.1 InstanceGroup 配置

```json
{
    "name": "local_cache_group",
    "storage_candidates": ["local_pool_01"],
    "cache_config": {
        "cache_prefer_strategy": "CPS_ALWAYS_LOCAL",
        "cache_reclaim_strategy": {
            "strategy": "LRU",
            "trigger_usage_ratio": 0.85,
            "target_usage_ratio": 0.75,
            "sampling_size_total": 1000
        }
    },
    "quota": {
        "max_capacity_bytes": 0
    }
}
```

注: `max_capacity_bytes = 0` 表示不做全局容量限制（由各节点实际容量决定），淘汰完全由 KVCM 的 `CacheReclaimer` 驱动（基于 MetaIndexer 中的 key 数量与 quota 的比较），与现有淘汰流程完全一致，不需要节点侧额外配合。

### 8.2 StorageConfig 注册

```json
{
    "global_unique_name": "local_pool_01",
    "type": "LOCAL",
    "local_storage_spec": {
        "heartbeat_timeout_ms": 30000
    }
}
```

---

## 9. 实现分期

### Phase 1 — 最小可用 (单机验证)

1. Proto 改动: `StorageType` 枚举、`LocalStorageSpec`、`StartWriteCacheRequest.node_id`
2. `LocalStorageBackend` 基本实现: 单节点 Create/Delete/free_list
3. `ExistsForWrite` node-aware 改动
4. `local_mem_sdk` 实现
5. 单元测试
6. **不含**: 多节点、Router、心跳

### Phase 2 — 多节点 + Router

1. `RegisterInstance` handler 扩展（节点注册）+ `NodeHeartbeat` RPC
2. 心跳超时检测 + 标记节点不可用 + 清理该节点的 free_list
3. `GetRoutingSuggestion` RPC 实现
4. `DataStorageSelector` 支持 LOCAL 类型 + `CachePreferStrategy` 新增值
5. MetaClient 新增方法

### Phase 3 — 跨节点数据传输 (可选)

1. 节点侧 DataAgent gRPC 服务 (Read RPC)
2. `local_mem_sdk` 增加远程读取能力 (解析 endpoint, TCP 拉取)
3. `SelectForMatch` 本地优先 + 远程 fallback 权重

---

## 10. 风险与注意事项

| 风险 | 缓解 |
|------|------|
| KVCM 单点故障导致所有节点无法写入 | Phase 2+ 引入 HA (LeaderElector + Redis MetaStorageBackend) |
| 节点重启丢失所有本地 cache | KVCM 检测心跳超时 → 标记不可用 + 清理 free_list + 删除该节点所有元数据; 节点重启后重新 RegisterInstance (携带 node_id) |
| free_list 与实际内存不一致 (KVCM 重启) | RecoverNodeFreeList 从 MetaIndexer 恢复 |
| Create 时节点已下线 | `DataStorageSelector` 选后端前检查 `IsNodeAvailable` |
| 同一 block_key 跨 spec 分到不同 slot | LOCAL 类型强制走 CreateInSingleBatch + Create 内部 block_key_hex dedup，保证同 block 同 slot |
| DataStorageManager::Create 覆写 URI hostname | node_id 放在 URI path 中（而非 hostname），不受覆写影响 |
