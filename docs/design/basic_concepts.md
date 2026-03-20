# 基本概念

## 管控面概念

### 1. Storage

一套存储系统。

- 有独立的存储配置（连接地址等）。
- 支持多种不同存储类型，如 NFS、3FS、TairMemPool、Mooncake 等。
- 允许不同的 Instance Group 和 Instance 共享同一 Storage。

### 2. Instance Group

- 同 Group 内所有 Instance 共享一套配额。
- 每个 Group 可以单独配置可用的 Storage 列表。
- 常见用途：
  - 对应一个业务团队，团队内的多个模型共享存储配额。
  - 对应一个模型，模型的多个版本共享存储配额。
  - 为重要模型单独配置 Instance Group，保证独占存储资源。
  - 多个长尾模型共用一个 Instance Group，共享存储资源的同时满足个别模型的突发容量需求。

### 3. Instance

一个 KVCache 实例。

- **仅单个 Instance 内部会复用 KVCache。需要互相复用 KVCache 的推理实例应当配置为使用同一个 Instance。跨 Instance 不复用 KVCache。**
- 对应的模型和 KVCache 配置（比如 fp8/bf16、block_size 等）唯一且不变。
- 属于且仅属于一个 Instance Group。
- 无需单独配置容量配额，使用所属 Instance Group 的配额。

通过以上抽象，将存储、配额和实例解耦，允许灵活控制 KVCache 存储方式和容量，便于统一管理大量 KVCache 实例。在 Instance Group 上配置容量配额，避免了为 Instance 单独配置存储容量，简化了业务侧的接入流程，也便于模型版本切换时的容量转移。

---

## 数据面概念

### 1. Block

一段定长的连续 Token 组成 1 个 Block。

- 单个 Block 对应的 token ids 数量在 Instance 初始化时指定。用户传入的 token id 序列会被切分为多个 Block。
- 有前缀依赖关系。自身 Token 序列相同但前缀不一致的两个 Block 是不同的。
- 每个 Block 可以有多个 CacheLocation，对应多个存储位置/层级。

### 2. CacheLocation

单个 Block 的一个存储位置。

- 单个 CacheLocation 内的所有数据必须全部位于同一种存储类型（type）。
- 状态流转：**writing → serving → deleting**
  - **writing**：KVCache 正在写入，不可服务。暂时无需再次写入。
  - **serving**：已写入完成，可正常读取。
  - **deleting**：正在删除中，不可读取。

### 3. LocationSpec

单个 CacheLocation 的一部分数据。

- 组织存储位置格式统一为 URI，但允许不同表达方式：
  - 对于内存池可能是地址。
  - 对于文件系统可能是文件路径。
  - 对于 KV 存储可能是 Key。
- 统一了格式，同时避免了强制映射到地址偏移或 KV 等导致的底层位置语义错位。
- 在 URI 中记录 size 以简化容量统计。
- 通过 blkid + size 支持单个存储位置（如单个 3fs 文件）内的分块存储。
- spec name 允许用户配置，灵活支持不同的 TP、PP、混合注意力。
- 允许 Location 内仅有部分 Spec：混合注意力场景下很多 Block 不需要存储线性注意力。

---

## 设计考量

基于以上抽象，既满足了推理引擎的 KVCache 查询需求（混合注意力的复合匹配等），也在 Tair KVCM 内保留了 LLM 的相关业务特征和语义：多 Block 间的前缀关系、同 Location 多 Spec 的关联关系等。由于感知并保留了 LLM 相关特征，Tair KVCM 还可以针对 KVCache 存储场景实现更多优化：

- 通过前缀对请求进行更细致的分类，针对性调整逐出策略。
- 利用前缀依赖的数据关系，避免后缀的无效存储。
- 为线性注意力选择最需要保留的 Block，在不牺牲命中率的情况下优化存储容量。

从底层存储视角看，降低了接入的复杂度（不需要了解任何推理概念，完全透明），同时还可以获取到提炼并翻译后的存储特征（存储对象间的生命周期关联等），为后续更进一步的优化和专用存储系统的开发留出了充足空间。
