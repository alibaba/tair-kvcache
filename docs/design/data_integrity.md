# KVCache 读写链路数据完整性校验

## 1. 评审结论、目标与边界

本文描述当前实现，覆盖以下能力：

- KVCM 在写前计算 checksum，并在读后校验；
- 调用方自行计算 checksum，KVCM 负责保存并在查询时原样返回；
- 调用方使用与 KVCM 相同的算法时，可显式要求 KVCM 在写入、元数据往返和读取三个阶段校验；
- checksum 随独立存储的 `LocationSpec` 持久化，并在 spec 合并、状态更新和分层迁移时保留；
- 老客户端、老元数据和未开启校验的合法调用保持原行为。

当前只实现 meta checksum 方案。inline header 的协议和配置槽位已预留，但本版本拒绝启用。
完整性逻辑位于共享的 client、manager 和 meta 层，与具体数据存储后端无关，不依赖后端专用校验 target。

### 1.1 生产结论

本实现的 CPU/协议/元数据路径已具备生产代码应有的 fail-closed 语义：显式校验无法执行时返回错误，
不会静默当作成功；默认未开启的合法历史调用仍保持原行为。外源、内源、HTTP/gRPC 集成、ASAN、配置工具和
CPU 协议性能均有自动化或可重复测试覆盖。

但“代码可合入”和“生产可开启 GPU 校验”是两个门槛。当前开发环境没有 CUDA/MUSA 编译器和设备，
因此真实 kernel、设备资源释放以及 write → 篡改存储字节 → read 的硬件闭环仍是**上线前强制门禁**，
不能用 CPU UT 或远端普通 CI 代替。第 12 节给出逐项准入条件；任一条件未满足时只能合入并保持功能关闭，
不能在生产 storage 上设置 `enable_meta_checksum=true`。

再次按 RTP 全链路责任归因目标审查后的结论是：当前 `CA_CRC32_XOR_INT64` 可以作为兼容的快速抽样探针，
但还不能单独作为 `rtp_strict` 的生产契约。它没有在 checksum 中绑定 iov 长度，也没有通过 API/Meta 携带
resolved 采样窗口和算法版本；仅靠各进程约定同一个环境变量，既无法在启动时证明契约一致，也会把契约漂移
误报成数据损坏。严格模式还必须落实第 7.2 节的版本化 contract，以及第 13.4.4 节的 buffer 所有权和 I/O
完成性约束。在这些能力实现前，当前 PR 的正确发布姿势仍是“代码合入、功能默认关闭”，不能把设计中的待实现
链路当成已经具备的生产保证。

### 1.2 保证范围

```text
caller/KVCM trusted checksum
          │
          ├─ WRITE_INPUT: buffer ──compare──> trusted checksum ──> Put
          │
          └──────────────────────────────> FinishWrite ──> Meta
                                                             │
query ──> META_ROUND_TRIP: returned checksum ──compare────────┘
  │
  └─> Get ──> READ_OUTPUT: loaded buffer ──compare──> trusted checksum
```

- 三阶段全部执行并通过时，能够把故障区间缩小到写入前、元数据往返或 Put/介质/Get 链路；
- 仅保存并查询调用方 checksum 时，KVCM 只保证 opaque `int64_t` 的持久化和返回，不证明数据字节正确；
- 当前算法默认只采样每个 iov 的头尾，不能证明未采样区域完整，也不是防恶意篡改的认证机制；
- Manager/Meta 不持有 payload，`FinishWrite` 成功只表示元数据状态提交成功，不等价于重新读取并验证数据；
- 所有保证都限定在同一 `instance_id`，不允许跨 Instance 复用或比较 cache。

### 1.3 二次设计裁决：能够承诺什么

| 目标 | 裁决 | 必要条件 |
|---|---|---|
| 低开销覆盖每个 block | 接受按 iov 头尾采样 | 明确受保护字节集合和 coverage，默认 `W=4` 不作为 RTP 生产参数 |
| 发现任意位置的单点损坏 | 头尾采样不能承诺 | 使用全量算法、存储原生全量校验，或接受未采样区风险 |
| 全链路结果可比较 | “同叫 CRC32”不够 | 位级 contract、resolved `W`、iov 顺序/长度、结果编码和 golden vectors 全部一致 |
| 判断谁写坏 | 只能先定位“首个失败责任区间” | 不可变 `C0`、sealed generation、相邻边界比较和基于已提交字节的 storage 证据 |
| 判断 producer 计算语义正确 | checksum 不解决 | 模型/算子级正确性验证；`C0` 只保护生成后的 bytes |
| 防恶意篡改 | CRC 不解决 | 带密钥的 MAC/认证存储和独立信任根 |

因此，stage 数量增加不会弥补采样盲区：所有 stage 使用相同头尾算法时，它们会一致地看不见同一个未采样
中段。责任归因也不是“最后看到 mismatch 的组件就是责任方”，而是记录**第一个与同一 `C0` 分叉的边界**；
只有边界两侧的所有权和完成性都成立，才可以把区间继续收窄。

## 2. 数据模型与不变量

一个 checksum 对应“一个 block 的一个 `LocationSpec` 所实际保存的 payload”。一次
`TransferClient::SaveKvCaches` 处理当前 worker 的一个 `self_location_spec_name`，每个输入 `BlockBuffer`
按 iov 顺序计算一个值。核心不变量如下：

1. checksum 与 `LocationSpec` 绑定。TP/PP 分片可保存不同 payload，因此同一 block 的 `tp0`、`tp1`
   必须能够携带不同值；同名 spec 被复制到另一 storage 时才继承原值。
2. Manager/Meta 把 checksum 当作 opaque `int64_t`，只保存、复制和返回，不推断其来源或算法。
3. checksum 的数值和“是否存在”分开表示；`0`、负数以及其他任意 `int64_t` 都是合法值。
4. 写入和查询 API 都先按 spec name 分组，再让组内 vector 与 block 位置一一对应，避免跨分片错配。
5. 校验按 block 逐项比较，不使用 batch 聚合值，因此能够定位错误下标并识别 block 交换。
6. checksum 仍受 Instance 隔离约束；不同 `instance_id` 之间不会共享元数据或校验基准。

元数据中的表示为：

```text
CacheLocation
└── location_specs[]
    ├── name / uri
    ├── checksum: int64
    └── checksum_present: bool
```

持久化 JSON 只在 `checksum_present=true` 时写入 `checksum` 字段。老 JSON 没有该字段时，读取结果为
`checksum=0, checksum_present=false`；合法的 checksum `0` 则为
`checksum=0, checksum_present=true`。

## 3. 两种 checksum 来源

checksum 来源与是否由 KVCM 校验是两个独立选择：

| checksum 来源 | KVCM 行为 | 适用场景 |
|---|---|---|
| KVCM | 写前计算、保存，按需读后校验 | 使用内置算法的 KVCM 数据面往返检查 |
| 调用方，自定义算法 | 仅保存并返回 | 调用方维护自己的算法和校验逻辑 |
| 调用方，与 KVCM 同算法 | 显式执行 KVCM 三阶段 fail-closed 校验 | 既保留调用方可信基准，又定位故障区间 |

调用方自行提供 checksum 不要求 storage 开启 KVCM 计算能力。只有调用方要求 KVCM 重新计算或校验时，
才需要启用内置算法和 GPU checksum pool。

### 3.1 KVCM 计算

```cpp
auto [save_ec, save_result] = transfer_client->SaveKvCaches(
    uris,
    buffers,
    SaveKvCachesOptions::WithChecksums(trace_info));
if (save_ec != ER_OK) {
    return save_ec;
}

// 示例假定 StartWrite 返回的紧凑 session batch 全部写成功；部分失败时按第 5 节保留占位。
auto finish_ec = meta_client->FinishWrite(
    trace_id,
    write_session_id,
    success_mask,
    locations,
    FinishWriteOptions::WithChecksumBatches({save_result.ToChecksumBatch()}));
if (finish_ec != ER_OK) {
    return finish_ec;
}
```

`SaveKvCachesResult::location_spec_name` 来自 TransferClient 初始化时的 `self_location_spec_name`；
`checksums` 仅在 Put 成功后返回，且与传入的 `block_buffers` 等长。Put 失败时 vector 为空，避免调用方把
未落盘数据的 checksum 提交给 Meta。多个 worker/spec 的结果汇总后一次传入 `WithChecksumBatches`。

查询并校验读取结果：

```cpp
auto [match_ec, match] = meta_client->MatchLocation(
    trace_id,
    QueryType::QT_PREFIX_MATCH,
    keys,
    tokens,
    block_mask,
    location_spec_names,
    MatchLocationOptions::WithChecksums());
if (match_ec != ER_OK) {
    return match_ec;
}

// 此示例查询与写入使用完全相同的 block 集合和顺序。真实调用若做了 mask、prefix
// 匹配或重排，必须先按查询返回位置构造 trusted_query_order_checksums。
const auto &trusted_query_order_checksums = save_result.checksums;

// 先用写端保留的可信值做严格元数据往返校验；任一位置缺失 checksum 也会失败。
auto meta_verify = match.VerifyChecksums(
    save_result.location_spec_name,
    trusted_query_order_checksums);
if (meta_verify.mismatch) {
    return ER_CHECKSUM_MISMATCH;
}

// 读后仍使用写端可信值，不能把未经比较的查询值直接升级成可信根。
auto load_ec = transfer_client->LoadKvCaches(
    uris,
    buffers,
    LoadKvCachesOptions::VerifyWith(trusted_query_order_checksums, trace_info, trace_id));
```

`MatchMetaOptions::WithChecksums()` 与 `MatchMetaResult` 提供相同能力。HTTP/proto 的
`GetCacheLocationRequest`、`GetCacheLocationsByBackendRequest` 和 `GetCacheMetaRequest` 均通过
`include_checksums=true` 显式请求；默认值 `false` 不复制或发送已保存的 checksum。C++ client 在未请求时
返回空的 `checksum_results`；请求后可通过 `FindChecksums(location_spec_name)` 取得对应组。

`LoadKvCachesOptions::VerifyWith(match, spec_name)` 是为存量数据滚动迁移提供的**宽松 helper**：它会跳过
`checksum_present=false` 的位置。生产强校验不能只调用该 helper；必须先用独立可信值调用
`match.VerifyChecksums(spec_name, trusted)`，或显式确认目标位置全部 presence=true，再执行读后校验。
否则老 server/老 metadata 返回全 false 时，读取会兼容通过但没有形成端到端完整性保证。

HTTP JSON 响应为保持项目既有的 protobuf JSON 输出格式，会打印标量默认值；因此未请求时仍可能看到
每个 location spec 中的 `checksum="0"` 和 `checksum_present=false`，它们只表示“未返回”，并非已保存值。
gRPC protobuf binary 不会编码这些默认标量。

### 3.2 调用方计算，仅透传

调用方可直接把某个 spec 在当前 StartWrite session 中的 checksum 交给 FinishWrite；Manager 不重算，
也不要求它使用 KVCM 算法：

```cpp
auto finish_ec = meta_client->FinishWrite(
    trace_id,
    write_session_id,
    success_mask,
    locations,
    FinishWriteOptions::WithChecksums("tp0", caller_checksums));

auto [match_ec, match] = meta_client->MatchLocation(
    trace_id,
    query_type,
    keys,
    tokens,
    block_mask,
    location_spec_names,
    MatchLocationOptions::WithChecksums());

// match.FindChecksums("tp0") 是 Meta 保存并返回的原值与 presence vector。
```

Meta 当前不保存算法 ID 或来源。使用自定义算法的 writer 和 reader 必须在 KVCM 之外约定算法、版本和
iov 布局，且不应调用 KVCM 的 `VerifyCallerChecksums` 或 `LoadKvCachesOptions::VerifyWith`。

### 3.3 调用方计算，并由 KVCM 分阶段校验

当调用方 checksum 与第 7 节的 KVCM 算法完全一致时，可保留调用方值作为可信基准，并显式执行：

```cpp
// 1. WRITE_INPUT：KVCM 在 Put 前重算；不一致时不写数据。
auto [save_ec, save_result] = transfer_client->SaveKvCaches(
    uris,
    buffers,
    SaveKvCachesOptions::VerifyCallerChecksums(trusted_write_checksums, trace_info));
if (save_ec != ER_OK) {
    return save_ec;
}

// 2. 写成功后提交调用方的可信值。
auto finish_ec = meta_client->FinishWrite(
    trace_id,
    write_session_id,
    success_mask,
    locations,
    FinishWriteOptions::WithChecksums("tp0", trusted_session_checksums));
if (finish_ec != ER_OK) {
    return finish_ec;
}

// 3. META_ROUND_TRIP：查询值必须与调用方可信值一致。
auto [match_ec, match] = meta_client->MatchLocation(
    trace_id,
    query_type,
    keys,
    tokens,
    block_mask,
    location_spec_names,
    MatchLocationOptions::WithChecksums());
if (match_ec != ER_OK) {
    return match_ec;
}
auto meta_verify = match.VerifyChecksums("tp0", trusted_query_order_checksums);
if (meta_verify.mismatch) {
    return ER_CHECKSUM_MISMATCH;
}

// 4. READ_OUTPUT：Get 完成后，KVCM 再计算目标 buffer 并与可信值比较。
auto load_ec = transfer_client->LoadKvCaches(
    uris,
    buffers,
    LoadKvCachesOptions::VerifyWith(trusted_query_order_checksums, trace_info, trace_id));
```

三个阶段的含义为：

| 阶段 | 比较内容 | 失败所定位的区间 |
|---|---|---|
| `WRITE_INPUT` | Put 前 buffer vs 调用方可信 checksum | 调用方算法/参数不一致，或进入 KVCM 前数据已变化 |
| `META_ROUND_TRIP` | 查询返回值 vs 调用方可信 checksum | FinishWrite、Meta 持久化或查询返回链路 |
| `READ_OUTPUT` | Get 后 buffer vs 可信 checksum | Put、存储介质或 Get 链路 |

只有三个阶段均通过，才能形成当前 KVCM 的基本诊断链。Manager/Meta 不持有数据，不能在 FinishWrite 内证明
实际存储内容正确；该链把 Put、介质和 Get 合并为 `READ_OUTPUT` 故障区间。若业务还要求观察初次 Put 后的
状态，必须增加第 13.4.3 节的 `STORAGE_COMMIT` backend checksum 或立即 read-back；其中只有 backend 在
commit 边界生成的证据才能直接归因写侧，read-back 仍包含一次 Get，不能仅凭三个阶段或一次回读指认组件。

## 4. strict 语义与错误

显式请求计算或校验采用 strict 语义：

- checksum 不一致：`ER_CHECKSUM_MISMATCH`；
- batch/presence 数量不一致：`ER_CHECKSUM_MISMATCH`；
- 非 CUDA/MUSA build、pool 未初始化、buffer 不是完整 GPU iov 或计算失败：
  `ER_CHECKSUM_UNAVAILABLE`；
- `WRITE_INPUT` 失败发生在 Put 之前；
- `READ_OUTPUT` 校验发生在 Get 成功之后。

不显式传入 checksum options 时，历史调用不计算、不返回、不校验 checksum。已有环境变量
`KVCM_SDK_CHECK=true` 仍只用于 best-effort 日志，不会持久化 checksum，也不会因 mismatch 让请求失败。
该环境变量会初始化同一 GPU pool，因此显式 checksum options 可以复用它；生产配置仍应使用
`enable_meta_checksum=true` 明确声明能力，不能把日志开关当作发布开关。

TransferClient 的 `WRITE_INPUT`/`READ_OUTPUT` 错误日志使用统一的 `ChecksumValidationLog`，包含可用的
`stage`、`trace_id`、`block_index`、expected/actual checksum、URI 和 block id。Options 中的 `trace_id` 与
`TransferTraceInfo::block_ids` 用于补齐诊断上下文；`TransferTraceInfo` 本身保持原始布局，避免破坏已发布
C++ Client 的二进制兼容性。`META_ROUND_TRIP` 是纯比较 helper，只返回 stage、mismatch 和 faulty indices，
业务接入层必须检查结果并补充同等结构化日志，不能忽略返回值。

### 4.1 失败、副作用与重试契约

| 失败点 | 对外结果 | 数据/元数据副作用 | 正确处理 |
|---|---|---|---|
| Save 的 URI/buffer 数量非法 | `ER_INVALID_PARAMS` | 未执行 Put | 修正请求后重试 |
| `WRITE_INPUT` batch shape 错误或 checksum 不同 | `ER_CHECKSUM_MISMATCH` | 未执行 Put | 检查 block 顺序、iov 布局、采样参数和算法 |
| `WRITE_INPUT` 无 kernel/pool、buffer 不可计算 | `ER_CHECKSUM_UNAVAILABLE` | 未执行 Put | 不得降级成成功；修正能力或关闭显式校验 |
| 存储 Put 失败 | 透传后端错误；结果中的 checksums 为空 | Put 已尝试，Meta 尚未 Finish；后端可能有部分副作用 | 用失败 mask 完成会话，或等待超时清理；不得提交 checksum |
| FinishWrite 的 instance/mask/batch/spec 非法 | Manager 为 `EC_BADARGS`，gRPC C++ client 映射为 `ER_SERVICE_INVALID_ARGUMENT` | session 未消费，Meta 未更新 | 在 session 超时前修正并用同一 session 重试 |
| FinishWrite 的 instance 不存在 | `EC_INSTANCE_NOT_EXIST` / `ER_SERVICE_INSTANCE_NOT_EXIST` | session 未消费 | 修复路由/实例；不要跨 Instance 完成会话 |
| FinishWrite 通过前置校验后 Meta 更新失败 | 服务端内部错误 | **session 已消费**；数据已 Put，Meta 可能仍是 WRITING 或部分更新 | 不可重试原 session；查询确认状态后重新 StartWrite，必要时走清理流程 |
| `META_ROUND_TRIP` 缺失或不同 | `ChecksumVerifyResult.mismatch=true` | 无新增副作用 | 严格模式停止读取；按 faulty index 检查 FinishWrite/持久化/查询链路 |
| Load 的 shape 错误或校验能力不可用 | `ER_CHECKSUM_MISMATCH` / `ER_CHECKSUM_UNAVAILABLE` | **未执行 Get** | 修正参数或能力后重试 |
| 存储 Get 失败 | 透传后端错误 | Get 已尝试，未做 checksum 比较 | 按后端错误处理 |
| `READ_OUTPUT` checksum 不同 | `ER_CHECKSUM_MISMATCH` | Get 已完成，目标 buffer 含**不可信数据** | 丢弃该批输出，不得交给推理；按 stage/index/URI 排障 |

### 4.2 write session 状态机

```text
StartWrite 成功
      │
      ▼
 ACTIVE ──非法 FinishWrite──> ACTIVE（可修正重试，超时回调仍有效）
      │
      ├─超时──> CONSUMED ──> 删除本次 WRITING locations
      │
      └─合法 FinishWrite（校验与消费在同一把锁内）
                         │
                         ▼
                     CONSUMED
                         ├─Meta 更新成功──> 成功项 SERVING；失败项异步删除
                         └─Meta 更新失败──> 状态不确定，原 session 不可重放
```

“非法请求不消费”只覆盖 session 仍在 Manager 内时能够完成的前置校验。会话取出后执行的 Meta COW/持久化
不是事务的一部分；这沿用现有 FinishWrite 语义，也是调用方必须区分“修正原请求”和“重新发起写入”的原因。

## 5. batch 对齐与部分成功

每个 `FinishWriteOptions::checksum_batches[i].checksums` 必须与服务端为这次 `StartWriteCache` 捕获的
**紧凑 session keys** 等长，而不是与原始请求 batch、某个 spec 实际写成功的子集或最终成功子集等长：

```text
StartWrite request keys = [cached0, k1, cached2, k3]
session/response keys   = [k1, k3]                 # 已命中 key 被压缩掉
success_blocks          = [T, F]
tp0 checksums           = [c1, x]
tp1 checksums           = [d1, x]
```

失败或该 block 不含该 spec 时对应值不会被读取，但必须保留 slot；`x` 可以是任意 `int64_t`。不提交任何
batch 表示本次不更新 checksum。允许只提交本次确实写入的 spec 子集，但 batch name 必须非空、不可重复，
且必须属于 StartWrite 分配出的 spec 集合。

长度错误、空/重复/未知 spec name、非法 success mask，以及使用其他 `instance_id` 完成 session 均返回
`EC_BADARGS`。write session 在 StartWrite 时绑定 instance；上述校验和 session 消费在同一临界区完成，失败
不会消费 session。调用方可以用同一个 session 修正后重试，未重试时仍由原超时回调清理。若请求中的
instance 本身不存在，则返回 `EC_INSTANCE_NOT_EXIST`，同样不消费 session。

`StartWriteCache` 返回的 `locations` 正是 session 的紧凑顺序；对这些 location 调用 `SaveKvCaches` 得到的
checksum 已与 session 对齐，不应再扩展回原始 StartWrite 请求。若某个 worker 只处理紧凑 session 的子集，
仍须在 FinishWrite 前补齐为 session 长度。查询侧每个 `LocationSpecChecksumResult` 也与返回的 `locations`
顺序对齐；缺少该 spec 或老元数据无 checksum 的位置由 `checksum_present=false` 表示。

`LoadKvCachesOptions::VerifyWith(checksums, checksum_present)` 会跳过 presence=false 的位置；可信值存在但
查询结果缺失时，`MatchLocationResult::VerifyChecksums(spec_name, expected)` 会报告
`META_ROUND_TRIP` mismatch。

## 6. 配置与运维入口

内部 `StorageConfig` JSON 示例：

```json
{
  "type": "file",
  "global_unique_name": "nfs_01",
  "storage_spec": {
    "root_path": "/tmp/nfs/",
    "key_count_per_file": 8
  },
  "integrity": {
    "enable_meta_checksum": true,
    "enable_inline_header": false,
    "inline_header_version": 0,
    "algo": "crc32_xor_int64"
  }
}
```

Admin HTTP/proto 使用枚举名 `CA_CRC32_XOR_INT64`。`kvcm_ops add_storage` 和
`update_storage` 支持：

- `--enable_meta_checksum`；
- `--disable_meta_checksum`（仅 update）；
- `--checksum_algo crc32_xor_int64`。

update 未指定这些参数时，会先读取现有 storage 并完整保留 `integrity` 对象。

配置规则：

- 缺少 `integrity`：全部关闭，兼容老配置；
- `enable_meta_checksum=true`：CUDA/MUSA TransferClient 初始化共享 `SdkBufferCheckPool`；
- 任一可见 storage 开启该字段，就为当前 TransferClient 提供内置计算能力；运行时不按单条 URI 再判断；
- 内部 JSON 省略 `algo` 时使用当前默认算法；显式写未知算法或字段类型错误时拒绝配置；Admin proto/HTTP
  开启 `enable_meta_checksum` 时必须同时传 `CA_CRC32_XOR_INT64`；
- `enable_inline_header=true` 或孤立的非零 `inline_header_version`：拒绝配置；
- Add、Update、Registry 恢复以及 TransferClient Init 都执行校验。

pool 参数沿用已有环境变量：

- `KVCM_SDK_CHECK_CELL_NUM`：pool cell 数，默认 `4`，合法范围 `[1, 1024]`；
- `KVCM_SDK_MAX_CHECK_IOV_NUM`：单次计算 chunk 的最大 iov 数，默认 `500000`；
- `KVCM_CHECK_IOV_BYTE_SIZE`：每个 iov 头部和尾部各参与 CRC 的最大字节数，默认 `4`。

cell 数、最大 iov 数或采样字节数小于等于 `0` 时 TransferClient 初始化失败；最大 iov 数超过 GPU kernel
的 `int` 索引上限或内存大小计算上限时同样拒绝，避免形成永远无法取得 cell、缓冲区大小回绕或“请求成功
但无法计算”的配置。cell 数在构造阶段只记录、不分配，`Init` 完成范围校验后才创建 vector，防止异常环境值
在校验前触发无界宿主内存分配。CUDA 与 MUSA 都会对每个 cell 运行固定 CRC fixture warm-up；任一分配、
stream 创建或 warm-up 失败都会让 TransferClient 初始化失败。

`KVCM_CHECK_IOV_BYTE_SIZE` 由 kernel translation unit 在进程加载时读取，必须在进程启动前设置；运行中修改
环境变量不构成受支持的动态配置。参与同一份 checksum 的 writer、reader 和外部调用方必须使用完全相同的
值。CPU-only build 即使看到 storage 开启了 `enable_meta_checksum` 也会告警后继续初始化，以便 opaque
checksum 的 FinishWrite/查询链路可用；任何显式 KVCM 计算或校验仍返回 `ER_CHECKSUM_UNAVAILABLE`。

### 6.1 pool 资源模型与设备约束

在 64 位平台 `sizeof(IovDevice)=16`。令 `C=KVCM_SDK_CHECK_CELL_NUM`、
`N=KVCM_SDK_MAX_CHECK_IOV_NUM`，不计 stream/runtime 固定开销：

| 资源 | 单 cell | 总量 |
|---|---:|---:|
| pinned host `h_iovs` | `16N` bytes | `16NC` bytes |
| GPU `d_iovs + d_crcs` | `20N` bytes | `20NC` bytes |

默认 `N=500000, C=4` 时约占 32 MB pinned host memory 和 40 MB GPU memory（十进制）。每个 cell 同时只服务
一个 checksum 请求；并发超过 `C` 时在 pool 条件变量上等待，因此 `C` 是吞吐/显存/pinned memory 的显式
权衡，不能为规避排队盲目放大。

pool 绑定到 TransferClient 初始化时的 GPU device。取得 cell 时会把当前线程切到该 device，归还时恢复；
pool 销毁也会切回绑定 device 释放 stream 和 buffer，再恢复调用线程原 device。传入的所有 GPU iov 必须
属于该 device 或被运行时明确支持跨设备访问；把另一张卡的普通 device pointer 交给本 pool 会计算失败并
返回 `ER_CHECKSUM_UNAVAILABLE`，不属于受支持的数据路径。

## 7. 内置算法的精确定义与边界

当前唯一算法为 `CA_CRC32_XOR_INT64` / `crc32_xor_int64`，与
`SdkBufferCheckUtil::GetBlocksHash` 保持一致：

1. 对每个 iov 计算 `sample_size = min(KVCM_CHECK_IOV_BYTE_SIZE, iov.size / 2)`；
2. 按“头部 `sample_size` 字节 + 尾部 `sample_size` 字节”的顺序计算标准 reflected CRC32
   （多项式 `0xEDB88320`，初值 `0xFFFFFFFF`，最终按位取反）；
3. 按 iov 原始顺序聚合 CRC32，初始 `hash=0`，每一步执行
   `hash ^= uint64(crc) + 0x9e3779b97f4a7c15 + (hash << 12) + arithmetic_shift_right(hash, 32)`；
   运算采用 modulo-2^64，最后按相同 bit pattern 解释为 `int64_t`；该过程即
   `HashUtil::HashIntArray` 对 `uint32_t` 的稳定语义；
4. 每个 block 输出一个 checksum，block 顺序与输入 `BlockBuffers` 相同。

算法名是兼容已有命名；聚合不是把 CRC32 简单异或一次。调用方要获得完全相同的值，必须同时保持：

- 相同的 `KVCM_CHECK_IOV_BYTE_SIZE`；
- 相同的 iov 数量、顺序、边界和 size；
- 相同的 CRC32 与 `HashUtil::HashIntArray` 64 位回绕、算术右移语义。

默认只检查每个 iov 的头尾各 4 字节，中间字节变化不会被发现。把
`KVCM_CHECK_IOV_BYTE_SIZE` 配置到不小于最大 iov size 的一半时，偶数长度 iov 会全部参与 CRC；当前
`iov.size / 2` 取整规则仍会跳过奇数长度 iov 的正中间一个字节。扩大采样范围前应评估 GPU 开销。该算法
用于发现偶发 bit flip、错误 URI 和 block 串位，不是密码学完整性或安全认证机制。

当前算法还没有把 iov 长度作为独立字段纳入 checksum。这个限制不是只有理论上的 CRC 碰撞：当 `W=4` 时，
8-byte payload `abcdefgh` 的 CRC 输入是 `abcdefgh`；9-byte payload `abcdXefgh` 的 CRC 输入仍是
`abcdefgh`，两者会得到**确定相同**的 iov CRC，且 block 聚合值也相同。实际传输通常另有 size 校验，但这说明
现有值不能独立证明 layout/长度正确，也不能把“使用不同 iov shape 仍算出相同值”描述成概率仅为 `2^-32`。
严格链路必须另外核对 layout，或使用第 7.2 节把长度绑定进 checksum 的新 contract。

`KVCM_CHECK_IOV_BYTE_SIZE` 在 checksum kernel 的静态初始化阶段读取一次（通常是进程启动或动态库加载），
运行中修改环境变量不会改变当前进程的结果。该值事实上是算法身份的一部分：Meta 目前不保存算法版本或
采样参数，因此所有参与该 KVCM 算法重算/校验的 writer、reader、外部 checksum 生产者必须在建立 baseline
前固定相同值。已经存在 checksum baseline 时，不能在滚动发布中静默修改默认值或配置值，否则新旧进程会
为相同数据计算出不同结果。调整后必须按新算法重新建立 baseline，或者先引入携带版本的迁移协议。

把固定值从 4 调到 64/128 仍然只是扩大头尾窗口，不能保证覆盖更大 iov 的中间区域；kernel 的逐字节 CRC
工作量近似随 `sample_size × iov_count` 增长，也不能在没有目标 GPU 数据的情况下假定代价可忽略。如果业务
要求覆盖整个 block，应新增明确命名、参数稳定并可版本化的全量算法，而不是改变
`crc32_xor_int64` 的既有结果。

内置 kernel 当前只接受完整 GPU `BlockBuffer`：每个 iov 必须为 GPU memory、非空、非 ignore、
base 非空且 size 至少为 2 字节。采样长度按每个 iov 独立计算；iov 数量相同但各 iov size 不同的 block
可合并到同一 GPU chunk，iov 数量不同或超过单 chunk 上限时拆批，输出顺序保持不变。

### 7.1 头尾采样的适用性与生产参数

对大体积 KVCache 使用头尾采样是可行的性能折中，也是当前算法已经采用的方式；但必须把它准确描述为
**采样完整性检查**，不能称为全 payload CRC。令第 `i` 个 iov 的长度为 `S_i`、每端窗口为 `W`，则：

```text
sampled_bytes = Σ_i 2 × min(W, floor(S_i / 2))
coverage      = sampled_bytes / Σ_i S_i
```

若只有一个位置均匀随机的坏字节，对采样区内随机错误按 hash 近似的 CRC32 漏检概率约为 `2^-32`，因此整体
发现概率近似为
`coverage × (1 - 2^-32)`。实际故障并不一定均匀随机：错 URI、错 block、错 base/size、边界覆盖和整段串位通常
会改变头尾，采样对这类工程故障很有效；固定发生在未采样中段的 bit flip 则必然漏检。对采样方案而言，主要
风险是“没有读到坏字节”，而不是 CRC32 自身碰撞。

例如单个 iov 为 1 MiB、`W=128` 时只检查 256 bytes，字节覆盖率约 `0.0244%`；`W=4` 时约
`0.00076%`。因此头尾方案的主要价值是发现结构性/边界类错误，不应拿它承诺高概率发现均匀随机的中段单点
翻转。iov 数量增加但各 iov 尺寸分布相近时，总覆盖率仍大致保持该比例，不会因为 iov 多就自动接近全量。

生产参数遵守以下规则：

- 以 **iov 为粒度**分别取头尾，不能只取整个 block 的最前和最后一段；否则中间 layer、KV scale 或其他
  group 的错误完全没有采样机会；
- `W` 没有脱离真实 iov 分布的通用正确值。当前默认 `4` 追求最低开销，但不能提供有意义的随机单点翻转覆盖率；
  RTP 接入应至少用候选 `64/128/256` 在目标 GPU、模型和 block 布局上同时测 coverage 与 P99，再由业务选择；
- 增大 `W` 只线性扩大两端窗口，不会覆盖大 iov 的中段。若目标是用固定预算提高空间覆盖，应新增带版本的
  多窗口算法（例如头、尾和若干确定性内段），不能在原算法名下静默改变 offset；
- writer、KVCM、storage commit verifier 和 reader 必须使用同一个 checksum contract：CRC 多项式、初值、
  final xor、每端窗口、按 iov 还是按 block、iov 顺序/边界、聚合公式和结果编码都必须完全相同；“都叫 CRC32”
  不代表结果兼容；
- contract 必须显式携带版本和 resolved `W`，进入 `instance_id`/TP RPC，并在初始化时跨 rank 核对。生产链路
  不能只依赖各进程恰好设置了相同环境变量。

如果业务要求每个字节都参与检查，只能选择全量算法或由存储/硬件提供等价的全量校验；头尾采样无法给出
这个覆盖。即使全量 CRC32 仍有有限 checksum 的碰撞边界；若要求更低残余概率，应使用更宽的 checksum/hash，
不能把 32-bit CRC 描述成绝对无碰撞。本文后续的“校验通过”均只覆盖 checksum contract 定义的采样字节。

### 7.2 RTP strict 的版本化生产 contract

已有 `CA_CRC32_XOR_INT64` 的位级结果必须保持稳定：一旦 checksum 被持久化，直接修正奇数长度规则、把默认
窗口从 4 改成 128，或把 size 混入聚合，都会让相同 payload 的新旧结果不兼容。因此这些修正不能偷偷放在
原枚举名下；兼容算法继续按本节前半部分解释，并在 contract 中标为
`kvcm.crc32_xor_int64.legacy-v0;W=<resolved>`。

RTP 严格模式建议新增独立算法/枚举 `sampled_crc32_layout_v1`。以下定义固定实现语义，后续实现若改变任何一项
必须增加版本，不能复用名字：

1. `W` 是大于 0 的无符号整数，由部署配置解析一次；iov 数为 `N`，第 `i` 个长度为 `S_i`。空 block、
   `ignore=true`、null pointer 或 `S_i=0` 在 strict 路径 fail-closed；`S_i=1` 合法。
2. 若 `S_i <= 2W`，`sample_i` 为整个 iov 且每个字节只出现一次；否则为
   `payload[0:W] || payload[S_i-W:S_i]`。这样小型奇数 iov 不再永久漏掉中心字节。
3. `crc_i` 是对 `LE64(S_i) || sample_i` 计算的 IEEE reflected CRC32：多项式 `0xEDB88320`、初值
   `0xFFFFFFFF`、final xor `0xFFFFFFFF`。固定 `LE64` 使语言和主机端序不影响结果，并把长度纳入保护。
4. 定义
   `mix(h,x) = h xor (x + 0x9e3779b97f4a7c15 + (h << 12) + arithmetic_shift_right(h,32))`
   （所有运算 modulo `2^64`）。令 `h=0`，依次 mix 固定 domain
   `0x4b56434d43524331`（ASCII `KVCMCRC1`）、`uint64(W)`、`uint64(N)`，再对 `i=0..N-1` 按顺序 mix
   `uint64(i)`、`uint64(S_i)`、zero-extended `uint32(crc_i)`；最终 64-bit pattern 按 two's-complement
   解释为 `int64_t`。`arithmetic_shift_right` 以 `h` 的 bit 63 为符号位补位。
5. contract descriptor 至少包含 `algo_id/version/resolved_W/layout_signature`。layout signature 描述有序
   spec/group、layer、每层 BlockInfo size 和 scale/layout 版本，不包含地址、pool 容量或瞬时 block id。

第 3、4 步有意同时绑定 size：CRC preimage 使单 iov 自包含，block 聚合再次绑定 count/index/size，避免调用方
只复用 CRC vector 却丢掉结构信息。该算法仍是抽样且非密码学方案；它修复的是 contract 和结构歧义，不会让
未采样中段自动获得保护。新算法的 payload coverage 为
`Σ_i min(S_i, 2W) / Σ_i S_i`；第 7.1 节的 `2 × min(W, floor(S_i/2))` 只描述 legacy-v0。

contract 不能只存在于设计文档或环境变量中。生产实现必须满足：

- KVCM 暴露实际运行的 descriptor；RTP、KVCM 和 storage verifier 在初始化及每次跨进程请求时做精确相等
  检查，contract 不同返回独立的 `CONTRACT_MISMATCH/UNSUPPORTED`，不能记录成 payload checksum mismatch；
- contract/layout 是 Instance 数据身份的一部分。变更算法、`W` 或 layout 时创建新 instance/baseline，禁止
  新旧 writer 在同一 instance 中滚动混写；迁移后只带 value/presence、没有可证明 contract 的旧条目按
  `contract_unknown` 安全 miss，不能因为 presence=true 就沿用；
- descriptor 在 batch/RPC 中只需携带一次，不应为每个 block 重复字符串。若通过 Instance 元数据提供，必须
  保证注册后不可变，并让 writer/reader 能够查询和确认；Finish/query 的 capability acknowledgement 必须同时
  确认 checksum value 和 effective contract，不能只看到 `checksum_present=true` 就推断算法兼容；
- caller 自定义 opaque checksum 可以继续只存值，但只有明确匹配 KVCM built-in descriptor 的 batch 才允许
  调用 KVCM 重算。当前只存 `int64_t + presence`、由调用方口头断言同算法的接口不满足 strict 能力确认；
- 为 legacy-v0 和新算法分别发布 CPU reference/golden vectors，覆盖大小端编码、正负/零结果、1-byte、奇偶
  size、异构 iov、顺序交换和上述 `abcdefgh`/`abcdXefgh` 反例。RTP calculator 保持独立实现，但 CI 共用
  golden vectors。

在 `sampled_crc32_layout_v1`、descriptor 握手和 baseline 迁移落地前，RTP 可以在受控部署中使用 legacy 算法
做 `record/verify`，但 `rtp_strict` 必须拒绝启动，不能把环境变量碰巧一致当作已验证的全链路契约。

### 7.3 性能与覆盖率的最终选择

`W=128` 只是首轮压测候选，不是未经数据即可写死的生产默认。选择参数时至少输出每种模型/布局的 iov size
分布、每 block 的受保护字节数与 coverage P50/P95/P99、kernel/D2H/host aggregate 时间、pool wait 和端到端
P99。当前实现每个 iov 由一个 GPU thread 串行执行逐 bit CRC，并把每 iov 的 32-bit CRC 拷回 host 聚合；
因此开销不只来自读取的 `2W` 字节，iov descriptor H2D、CRC D2H、host 同步和大 batch 元数据也必须实测。
生产实现必须给 `W` 和单请求 `Σ min(S_i,2W)` 设置显式、overflow-safe 的上限，并在 launch 前 fail-closed；
`KVCM_SDK_MAX_CHECK_IOV_NUM` 只限制 descriptor 数量，不能阻止一个错误的超大 `W` 把抽样算法退化成无界全量
扫描。上限是资源保护参数，不进入 checksum 值；resolved `W` 才进入 contract。需要全量扫描时选择独立算法和
独立限流队列，不能通过把 sampled 模式的环境变量设置成极大值绕过容量治理。

推荐的生产防线是分层而非夸大头尾采样：

1. 每个 block 热路径执行版本化头尾 checksum，用于快速发现边界、错 URI、串 block 和传输责任区间；
2. 启用 GPU ECC、存储介质/协议自带的全量 checksum；它们算法可以不同，但必须作为独立证据报告，不能冒充
   与 `C0` 可直接比较的同 contract 值；
3. 对确定性 canary 子集执行全 payload audit，量化“头尾通过但全量失败”的实际发生率。若 audit 也用于逐段
   归因，则该子集每个 stage 都必须携带独立的 full-checksum contract/baseline；只在最终读取时全量计算只能
   发现问题，不能定位此前哪一段首次损坏。

扩大 `W`、增加固定内段窗口和提高 full-audit 比例是不同的 coverage/成本旋钮。若在相同总字节预算下需要减少
对固定中段故障的偏盲，可另增“头尾 + 确定性内段”算法版本；它对均匀随机单点错误的发现概率仍由总 coverage
决定，不能宣传成等价于全量 CRC。

## 8. 元数据生命周期

- FinishWrite 只给成功 block 的对应 spec 写 checksum；失败 block 的 location 按原流程删除。
- 不带某个 spec checksum 的状态更新使用“保持旧值”语义，避免重试或兼容调用清掉已有基准。
- location spec 合并、查询选择与 COW 更新保留 checksum/presence。
- ReportEvent 的 BLOCK_ADD 与 BLOCK_SNAPSHOT 接受调用方 checksum，并在 URI 校验、版本化和
  snapshot 替换过程中原样保留；未携带 presence 的非零值仅作为早期客户端兼容输入。
- BLOCK_ADD 是 patch 语义：同名 spec 未携带 checksum 时保留旧基准。若 reporter 实际覆盖了该 payload，
  必须同时提交新 checksum；否则后续 strict read 会用旧基准报 mismatch。需要明确清除基准时应通过完整
  snapshot 提交不含 checksum 的该 spec，不能依赖 ADD 的字段省略。同一 ReportEvent RPC 内多个 ADD 被
  折叠时也遵守该规则：后续省略 checksum 的 URI 刷新会保留本批次较早 ADD 提交的新值。
- 分层迁移按 spec 把源 checksum/presence 复制到目标 location；复制字节本身不在 Manager 内重算，
  后续读校验可发现迁移链路损坏。
- 删除 location/spec 时，其 checksum 随所属元数据一同删除；部分删除 spec 不影响其余 spec 的 checksum。

## 9. 协议兼容性

新增字段只使用新 tag，不复用已有字段：

| Message | 字段 | tag |
|---|---|---:|
| `FinishWriteCacheRequest` | `locations`（保留、deprecated） | 5 |
| `FinishWriteCacheRequest` | `checksum_batches` | 6 |
| `GetCacheLocationRequest` | `include_checksums` | 9 |
| `GetCacheLocationsByBackendRequest` | `include_checksums` | 10 |
| meta `GetCacheMetaRequest` | `include_checksums` | 7 |
| admin `GetCacheMetaRequest` | `include_checksums` | 7 |
| meta/admin `LocationSpec` | `checksum` | 3 |
| meta/admin `LocationSpec` | `checksum_present` | 4 |
| meta/kv-meta `StorageConfig` | `integrity` | 11 |
| admin `StorageConfig` | `integrity` | 12 |

gRPC protobuf binary 天然忽略未知字段；项目的通用 HTTP protobuf JSON parser 和 ReportEvent 快速 parser
也显式忽略未知字段。因此兼容矩阵为：

| client/server 组合 | 行为 | 完整性结论 |
|---|---|---|
| 老 client → 老 server | 历史行为 | 无 checksum 能力 |
| 老 client → 新 server | 新字段保持默认值，历史行为 | 无额外开销；兼容 |
| 新 client → 新 server | 显式 opt-in 后保存/返回 | `checksum_present` 可作为逐位置确认 |
| 新 client → 老 server | 新请求字段被静默忽略，RPC/HTTP 仍可能成功 | **不能**把 FinishWrite 成功当作能力确认；查询 presence 为 false |

具体滚动行为：

- 老 client → 新 server：不携带 checksum batches，按历史路径执行；
- 新 client 仍发送 legacy `locations` 字段，以兼容可能读取 tag 5 的旧 server；新 server 的 checksum
  对齐只使用 session keys 和 tag 6，不信任该字段；
- 新 client → 老 server：gRPC 与 HTTP 都会忽略未知 `checksum_batches`，因此不会建立完整性基准；调用方
  必须通过后续 query 的 `checksum_present` 确认能力，不能仅以 FinishWrite 成功作为依据；
- 新 client 读取老 metadata/server：checksum presence 为 false；
- MetaService 与 AdminService 只在 query 的 `include_checksums=true` 时携带已保存值；老 client 不设置该字段，
  维持历史响应；StartWrite 响应始终不返回 checksum。

“历史行为兼容”只覆盖合法请求。新实现会在任何 Put 前拒绝 URI/buffer 为空或数量不等的畸形
`SaveKvCaches` 调用并返回 `ER_INVALID_PARAMS`；这比旧实现更严格，是防止 URI 与 block 错位写入的主动
收紧，不应依赖旧版本对畸形输入的未定义行为。

公共 C++ 接口的兼容范围需要精确区分：本次保持所有既有 virtual 方法的声明顺序，并把新 overload 追加到
末尾；`TransferTraceInfo` 的字段和布局没有变化。因此，下游 subclass **重新编译**时保留源码兼容性，
既有 virtual slot 的序号也不变。但这不承诺“新代码对旧的预编译 subclass 调用新增 virtual”具备二进制
安全性；同一进程内的 client 库和 subclass 必须用同一版本重新构建后，才能使用 checksum 新接口。

### 9.1 发布顺序与回滚

必须采用 server-first：先把所有可能承接该 instance 的 Manager 升级完成，再升级/开启 writer，最后开启
reader strict 校验。混部期间不能依赖负载均衡恰好命中新 server，必须以查询得到每个目标位置
`checksum_present=true` 作为能力确认。推荐在 presence 覆盖率达到 100% 前只采集、不阻断业务，达到后再
切 strict；但任何已经显式请求的 KVCM 校验本身仍保持 fail-closed。

回滚顺序相反：先停用 writer/reader 的 checksum options，再回滚 server。老 server 能忽略新 wire 字段和
持久化 JSON 的未知字段，因此 payload 仍可读；但老代码读后重写 `CacheLocation` 时不能保留它不认识的
checksum，完整性基准可能逐步丢失。再次升级后这些位置会表现为 presence=false，必须由后续成功写入重新
建立，不能宣称无损回滚完整性状态。

## 10. 基础性能验证

仓库提供手工运行的 CPU/协议基准，覆盖 4096-block checksum 比较，以及每个 block 含两个独立 checksum
spec 时 gRPC protobuf binary 的序列化/反序列化和 HTTP 快速 JSON 序列化；同时输出 opt-in 前后的
字节数和耗时：

```bash
bazelisk run -c opt //kv_cache_manager/service/util/test:data_integrity_benchmark
```

本地 `-c opt` 连续 5 次运行的中位数（4096 blocks，每个 response location 含两个 checksum spec）如下；
它用于量级评审，不是跨机器 SLA：

| 项目 | opt-out / match | opt-in / one mismatch | 折算/增量 |
|---|---:|---:|---:|
| CPU batch compare | 5.37 µs/batch | 6.39 µs/batch | 约 1.31 / 1.56 ns per block |
| protobuf binary bytes | 335899 | 438299 | +102400（+30.49%），25 bytes/block |
| protobuf serialize | 265.26 µs | 290.24 µs | +24.98 µs/batch（+9.42%） |
| protobuf parse | 1507.75 µs | 1595.38 µs | +87.63 µs/batch（+5.81%） |
| HTTP JSON bytes | 970868 | 1106036 | +135168（+13.92%），33 bytes/block |
| HTTP JSON serialize | 2096.59 µs | 2445.76 µs | +349.17 µs/batch（+16.65%） |

基准会先断言 fixture 的错误下标和序列化正确性，再报告数值；不设置容易受共享机器负载影响的硬延迟阈值。
最终发布应保存目标机多轮分位数，并把同版本 opt-out 作为对照。

### 10.1 常驻内存与持久化体积

在本仓库 x86-64/GCC 10 编译探针中，`sizeof(LocationSpec)` 从 main 的 72 bytes 增至 88 bytes，即每个内存中
spec 固定增加 16 bytes（约 22.2%）；这部分即使 presence=false 也存在，且不包含两个 `std::string` 的动态
payload。该数值依赖 ABI，换编译器/标准库必须重测。

仅按已构造对象估算，固定增量下界为 `16 × live LocationSpec count`；容器预留容量、COW 副本和 allocator
碎片会使实际 RSS 更高。例如 1000 万个活跃 spec 至少约增加 160 MB；若是 1000 万个 location、平均两个
spec，则至少约增加 320 MB。该成本在 checksum 功能 opt-out 时仍然存在，因此合入/发布前必须用生产峰值
location/spec 数量验证 Manager RSS 与回滚余量，不能只看协议和 CPU benchmark。

不能用 checksum 数值哨兵或窃取 checksum 高位压缩 presence，因为任意 `int64_t`（包括 0、负数和所有 bit
pattern）都是合法 opaque 值；`spec_size`/`create_time` 又属于整个 `CacheLocation`，无法表达逐 spec
presence。若固定增量不可接受，应单独设计按需 sidecar/bitmap 等布局，并验证 JSON、COW、迁移和并发读写，
不能把 bit-pack 当作无语义成本的局部优化。

持久化 JSON 仅在 presence=true 时增加 `checksum` 字段；query 默认 opt-out 不复制 checksum 到 protobuf，
protobuf binary 也不编码默认标量。HTTP JSON 为保持现有 protobuf JSON 形状可能打印默认的 `0/false`，
因此 HTTP 的 opt-out 不等于完全没有字段文本，容量规划应以实际 benchmark 输出为准。

GPU kernel 的吞吐、pool 排队、真实存储读写故障注入以及 CUDA/MUSA 多设备资源释放仍须在目标硬件单独测量；
在第 12 节硬件门禁完成前，本节不能作为允许生产开启的依据。

## 11. 需求—代码—测试追踪

下表是本设计的可审计索引；任何后续修改若改变行为，必须同时更新对应测试和本文：

| 需求/不变量 | 主要实现 | 直接测试 |
|---|---|---|
| checksum 值与 presence 分离，0/负数合法 | `meta/cache_location.h`；proto `LocationSpec`；转换工具 | `CacheLocationTest`、`ManagerMessageProtoUtilTest` |
| 每 block、每 spec 独立保存 | `FinishWriteCacheOptions`、`MetaSearcher::LocationUpdateTask` | `CacheManagerTest.TestFinishWriteCacheWithBlockMask`、`MetaSearcherTest.TestBatchUpdateLocationStatusPersistsPerSpecChecksum` |
| 调用方 checksum opaque 透传并查询返回 | FinishWrite proto/client；query opt-in；`GenLocations` | `GrpcStubTest`、`MetaClientTest`、HTTP/gRPC `test_caller_checksum_retry_and_query_opt_in` |
| 三阶段比较、全错误下标、shape/presence fail-closed | `common.h::VerifyBatchChecksums`；`TransferClientImpl` | `ChecksumVerifyUtilTest`、`TransferClientTest` |
| 写前失败不 Put；Get 前能力检查；读后失败丢弃 | `TransferClientImpl::SaveKvCaches/LoadKvCaches` | `TransferClientTest`；目标 GPU 故障注入门禁 |
| 非法 FinishWrite 不消费 session，instance 隔离 | `WriteLocationManager::GetAndDeleteForFinish` | `WriteLocationManagerTest`；HTTP/gRPC 集成 malformed/retry/wrong-instance 用例 |
| JSON/COW/合并/迁移不丢 checksum | `CacheLocation`、`MetaSearcher`、`MigrationManager` | `CacheLocationTest`、`MetaSearcherTest`、`MigrationManagerTest` |
| ReportEvent ADD patch、SNAPSHOT replace，快速 JSON parser 不丢字段 | event 校验与 delta/snapshot 合并；`ReportEventJsonParser` | `CacheManagerTest.TestReportEventPreservesCallerProvidedChecksum`、`ProtoMessageJsonUtilTest.TestReportEventFastJsonParserMatchesGenericParser` |
| 默认查询不返回；Meta/Admin 显式 opt-in | service handlers、`CacheLocationViewToProto` | `ManagerMessageProtoUtilTest`、`AdminServiceImplTest.TestGetCacheMetaChecksumIsOptInAndZeroRemainsPresent`、集成测试 |
| 配置非法值拒绝，update 保留旧 integrity | `StorageConfig`、`RegistryManager`、`kvcm_ops` | `StorageConfigTest`、`RegistryManagerLocalBackendTest`、`storage_util_test` |
| 算法位级稳定且避免 signed-shift UB | `HashUtil::HashIntArray`、CUDA/MUSA kernel | `HashUtilTest`、`SdkBufferCheckUtilTest`（GPU） |
| pool 边界、warm-up、多设备资源生命周期 | `SdkBufferCheckPool` CUDA/MUSA 实现 | `SdkBufferCheckUtilTest`；目标硬件门禁 |
| 协议字段只追加、老调用默认不变 | 三份 proto；旧 virtual slot 保序和 fallback | `GrpcStubTest.FinishWriteChecksumUsesAdditiveFieldNumber`、公共 C++17 源码兼容编译探针、全量 UT |
| CPU/协议开销可量化 | `data_integrity_benchmark` | `//kv_cache_manager/service/util/test:data_integrity_benchmark` |

### 11.1 本次验证记录

以下是 2026-09-16 对 PR #338 候选树执行的本地验证快照。计数用于把本次评审边界固定下来；后续代码变化
必须重新执行，不能沿用本表结论。GPU target 在无工具链或无设备平台显示 incompatible/skip 不算执行成功。

| 验证项 | 结果 | 已覆盖范围与明确排除项 |
|---|---|---|
| 外源 ASAN `//kv_cache_manager/...` | 126 个 target 通过，1 个跳过 | 共享代码 CPU/协议/元数据路径；跳过项为需要真实 GPU 的 `SdkBufferCheckUtilTest` |
| 内源 ASAN | 129/129 个可运行 target 通过 | 覆盖共享代码在内源依赖图下的编译和 UT；排除 GPU-only target，以及已不维护且非本功能依赖的 `VcnsHf3fsBackendTest`、`VcnsHf3fsAllocatorTest` |
| 外源集成测试 | 17/17 通过 | ASAN 下启动真实 Manager 进程，覆盖 HTTP/gRPC、调用方 checksum、query opt-in、非法 FinishWrite 后重试和 instance 隔离 |
| `kvcm_ops` package 测试 | 22/22 通过 | storage integrity 配置新增与 update 保留语义 |
| CUDA/MUSA host 侧编译探针 | 通过 | 用 runtime stub 编译两套 pool host 实现；仅证明 C++ 分支可编译，**不包含 kernel 编译或设备执行** |
| 公共 C++17 兼容探针与 header diff 审计 | 通过 | 探针确认 legacy subclass 仍可实例化、旧调用不歧义；diff 确认既有 virtual 声明顺序和 `TransferTraceInfo` 字段未变 |
| proto 兼容审计 | 通过 | 三份 proto 仅追加新 tag；tag 5 的 `locations` 只增加 deprecated 标记，没有删除或复用 wire 字段 |
| 静态检查 | 通过 | `git diff --check`、变更 BUILD 的 buildifier、除大段既有 raw-string corpus 外的 changed-line clang-format；该 corpus 的新增 case 人工核对 |
| CPU/协议性能基准 | 连续 5 次完成 | 第 10 节记录中位数；只用于 CPU/协议量级评审，不代表目标 GPU/storage SLA |

最终 push 对应的 GitHub normal/ASAN/CodeQL/CLA 是动态合入条件，应以
[PR #338](https://github.com/alibaba/tair-kvcache/pull/338) 的**最终 head commit**检查结果为准；它不能由
设计文档中的静态勾选替代。真实 CUDA/MUSA 与 storage 故障注入不在本地通过范围内，仍由第 12.2 节阻止
生产开启。

## 12. 生产开启清单与硬门禁

### 12.1 合入门禁

- [x] 第 11.1 节本地软件验证已通过；候选分支已 rebase 到 2026-09-18 评审时最新 `origin/main`
  (`fa79fbe94510483aa6cbd98ed999d6da26486376`)，且相对该 base 只有一个提交；
- [x] proto compatibility 检查确认没有 tag 复用，C++ header diff 审计确认 legacy virtual slot 顺序与
  `TransferTraceInfo` 布局未改变，并按第 9 节限定其兼容范围；
- [x] 设计、API、configuration 文档与候选代码 diff 已逐项核对。

合并按钮的额外动态门禁是：PR 最终 head 的 normal/ASAN/CodeQL/CLA 全绿且 review policy 满足。若 push、
rebase 或修复产生新 head，必须重新等待该 head 的检查，不能引用旧 commit 的绿色结果。

### 12.2 生产启用门禁

- [ ] 对每个计划启用的 GPU 后端，在对应的实际 CUDA/MUSA 发布镜像和设备上编译、运行并确认非空执行
  checksum 测试，覆盖 warm-up、最小 iov、异构 iov、chunk、多线程、多 device 和跨 device 失败；当前
  CUDA test body 覆盖这些场景，MUSA test body 只覆盖配置边界与 warm-up，启用 MUSA 前必须先补齐等价用例；
- [ ] 在目标 storage 上完成 write → 精确篡改采样区字节 → read，确认分别产生
  `WRITE_INPUT` / `META_ROUND_TRIP` / `READ_OUTPUT` 的预期结果，mismatch 后 buffer 不进入推理；
- [ ] 若 RTP/其他调用方要观察初次 Put 后的数据，为每个目标 backend 实现并验证第 13.4.3 节的
  `STORAGE_COMMIT` checksum 或 100% immediate read-back；若还要把责任直接归到写侧，必须使用 backend
  对目标提交字节计算的 checksum/等价 instrumentation，并落实第 13.4.4 节 sealed write lease；单次
  read-back 只能报告包含源并发写、Put、介质及其 Get 的故障区间；
- [ ] `rtp_strict` 使用第 7.2 节的版本化 descriptor，在任何 I/O 前完成跨 rank、RTP/KVCM/backend 的
  contract/layout 精确握手；当前只有裸 `int64_t` 且采样窗口来自隐藏环境变量时不得开启 strict；
- [ ] 对 producer/checksum/transfer stream 建立 event happens-before，验证 block generation、pin/write lease、
  Put/Get 成功的 I/O quiescence，以及 timeout 后 DMA 未停止时的 quarantine；否则 stage 只能用于发现问题，
  不能用于归责；
- [ ] 用生产 iov size/数量分布量化实际采样字节比例，由业务明确接受“采样校验”而非“全 block 校验”的
  能力边界；记录 coverage 分位数并用确定性 full-payload canary 监测采样盲区。若要求全覆盖，先引入
  可版本化的新算法和 baseline 迁移方案，不能直接调大现有默认值；
- [ ] 用生产 block/iov 分布压测 GPU kernel、pool 排队、pinned host/GPU memory 和端到端 P99，按第 6.1 节
  公式审定 `CELL_NUM`/`MAX_CHECK_IOV_NUM`，并落实 `W` 与单请求 sampled bytes 的 overflow-safe 硬上限；
- [ ] 按第 10.1 节用生产峰值 location/spec 数和真实编译 ABI 测量 Manager RSS、COW 峰值及回滚余量，
  明确接受 checksum opt-out 时仍存在的固定常驻内存增量；
- [ ] 所有 Manager server 先升级完成；writer canary 后查询 checksum presence 覆盖率达到 100%，再开启
  reader strict 校验；
- [ ] writer、reader、外部 checksum 生产者固定同一算法版本、iov 布局和 resolved `W`；legacy-v0 必须在
  进程启动前固定 `KVCM_CHECK_IOV_BYTE_SIZE`，新 contract 则从 descriptor 取得并核对；
- [ ] 为 KVCM 三个 stage 建立有丢失率、延迟和告警 SLA 的观测闭环；接入 RTP 全链路诊断时再覆盖
  `PRODUCER_OUTPUT`、`STORAGE_COMMIT` 和 `CONSUMER_INPUT`：业务接入层把 `META_ROUND_TRIP` 的
  `ChecksumVerifyResult` 写成结构化事件，日志平台对它和 `ChecksumValidationLog` 按 stage/error 做计数，
  或接入等价的 client telemetry hook/counter。metric label 必须是稳定低基数；trace、URI、block id 保留在
  日志中用于关联，不得直接作为 Prometheus label，`spec_name` 也只有在配置保证有界时才可作为 label。当前
  实现没有新增专用 Prometheus checksum counter，不能假设自动可观测；
- [ ] 演练第 9.1 节回滚顺序，并接受回滚期间 checksum baseline 可能丢失、需要重写恢复的事实。

只要生产启用门禁还有未完成项，本功能必须保持 storage 配置关闭；调用方 opaque checksum 的保存/查询可
独立灰度，但不得宣称 KVCM 已提供完整三阶段数据校验。

## 13. RTP-LLM 接入设计（待实现）

本节是基于 2026-09-17 的 RTP-LLM 主干代码完成的接入设计，审计基线为：

- 外源 `RTP-LLM/github-opensource`：`main@f9dc4e3d7a1393fe36ddc021d4e6ead54e29ab33`；
- 内源 `RTP-LLM`：仓库约定的内源主干 `main-internal@ae067e26a35c139054c419188b3140b11c6345d2`。

本节描述的是**后续实现契约**，不表示上述 RTP-LLM 主干已经启用 checksum。当前主干仍使用 legacy
`MatchLocation`、`SaveKvCaches`、`LoadKvCaches` 和 `FinishWrite` 接口，不请求、传递或校验 checksum。
实现时若 RTP-LLM 的 BlockTree/KVCM 路径已变化，必须重新做代码对照并更新本节，不能机械套用文件名。

### 13.1 接入结论

RTP-LLM 应提供四个递进模式，而不是用一个布尔开关同时改变写入、读取和失败语义：

| `kvcm_checksum_mode` | 写路径 | 读路径 | 用途 |
|---|---|---|---|
| `off` | legacy，不计算、不保存 | legacy，不请求、不校验 | 默认值和紧急回滚 |
| `record` | KVCM 计算并随 `FinishWrite` 保存 | 不改变读取 | server-first 后建立 baseline |
| `verify` | 同 `record` | 查询 checksum；缺失按 cache miss；Get 后由 KVCM 校验 | 常规生产模式 |
| `rtp_strict` | RTP 独立计算；KVCM 先确认 contract 再写前重算；验证 storage commit；写后校验 Meta 往返 | KVCM 校验后，RTP 再独立重算比较 | 全链路诊断；要求版本化 contract、sealed ownership，以及 backend checksum 或 read-back 成本 |

核心约束如下：

1. 一个值的身份必须是 `(instance_id, compact_session_index, location_spec_name)`，不能只按 block 或 URI
   保存。RTP 的 TP 分片和不同 cache group payload 不同，checksum 本来就可以不同。
2. `checksum=0` 是合法值。所有 RTP 内部结构和 TP RPC 都必须单独携带 presence/capability，不能用空值、
   `0` 或 response vector 是否非空猜测是否支持。
3. `verify`/`rtp_strict` 对缺少 checksum 的老数据采用“安全 miss”：从首个不完整 block 起截断远端前缀，
   重新计算而不是加载未校验数据。真正的 mismatch/unavailable/malformed 则让该次远端 load 失败。
4. checksum mismatch 是“远端 cache fail-closed”，不是无条件杀死推理。当前 BlockTree 行为是：目标 block
   不发布到复用树；PREFILL 丢弃这段远端复用并继续计算，其他角色按现有语义向上返回执行失败。
5. RTP 自算必须是独立实现且与协商出的第 7 节 contract 位级一致。受控兼容模式可实现 legacy-v0；生产
   `rtp_strict` 应使用第 7.2 节的新 contract。历史分支中的全 payload CRC32C/footer 方案只能复用测试、
   metrics 和故障转储思路，不能复用 checksum 值或 kernel。
6. 普通读取从 KVCM Meta 获得的值可以作为检测 Put/介质/Get 损坏的 baseline，但它不是独立的
   `META_ROUND_TRIP` 信任根。只有与写端仍持有的 RTP checksum 比较，才能声明完成该阶段。
7. RTP 首次产出的 `C0` 是只读信任根；后续阶段只能生成 `actual` 与 `C0` 比较，不能用自己的计算结果覆盖
   `C0` 后继续传递，否则每一段都可能为自己的坏数据生成一个“正确”的新 checksum，最终无法定位责任区间。
8. 计算 `C0` 前必须等待 KV producer 的 GPU stream/event 完成并把 block 标记为 sealed；直到 Put/commit
   校验完成前不得再修改这些 bytes。refcount 只能防止释放，不能防止另一个 kernel 改写 buffer。
9. 一个可归因的 `C0` 不是裸 `int64_t`，而是逻辑 envelope：
   `(value, contract_id, layout_signature, instance/key/spec, block_generation, write_session/trace)`。其中 value
   进入 KVCM Meta；其余字段可由不可变 Instance 配置和操作上下文提供，但任何边界都不能靠 GPU 地址推断身份。
   地址会复用，缺少 generation 会把前一代 block 的日志错误关联到后一代。

### 13.2 当前 RTP-LLM 数据路径与改造点

当前实现由 rank 0 负责 Meta，所有 TP rank 负责本卡 payload：

```text
rank 0 KVCMStorageBackend
  ├─ MatchLocation / StartWrite / FinishWrite
  ├─ spec_name -> {tp_rank, group_id, group_tag}
  └─ BroadcastManager ───────────────┐
                                      ▼
each TP rank KVCMStorageBackend::execute
  group_tag + block_id -> GroupPolicy::genBlockBuffersByTag
                                      │
                                      ▼
                         TransferClient Save / Load
```

代码对应关系和必要改造如下：

| 当前节点 | 当前行为 | checksum 改造 |
|---|---|---|
| `KVCMStorageBackend::match` | legacy `MatchLocation`，`KVCMMatchMeta` 只保存 `Locations` | `verify` 以上改用 `MatchLocationOptions::WithChecksums`，保存完整 `MatchLocationResult` |
| `KVCMStorageBackend::read` | rank 0 按 spec 分发 `group_tags/block_ids/uris` | 同序附带 expected checksum/presence；分发前做 shape、spec 和 presence 校验 |
| `KVCMStorageBackend::write` | `StartWrite` 后按 TP 分发，收集 actual URI，再 legacy `FinishWrite` | 收集每个 request item 的 checksum，按原始 spec/session 下标组装 batch 后调用新 `FinishWrite` |
| `KVCMStorageBackend::execute` | worker 生成 GPU `BlockBuffers` 后 legacy Save/Load | 根据 mode 选择 `WithChecksums`、`VerifyCallerChecksums` 或 `VerifyWith`；严格校验 RPC shape |
| `ClientWrapper` | 把 KVCM 错误压成 `bool` | 新增 options/result overload，并保留 `ClientErrorCode` 或结构化错误，不能丢失 mismatch/unavailable 分类 |
| `GroupPolicy::genBlockBuffers` | 按 layer 顺序、每层 `BlockInfo` 顺序生成 iov | 此顺序就是算法身份；RTP 与 KVCM 必须消费同一批 `BlockBuffers`，禁止各自重新推断布局 |
| `model_rpc_service.proto` | TP RPC 只有 tag、block、URI 和 actual URI | 只追加 checksum action、values、presence、contract ID 和 response capability 字段 |
| `StorageBackend::write` | 后台 best-effort，异常被吞掉 | checksum 写失败不影响本次推理，但必须 abort session、禁止发布 Meta，并独立计数告警 |
| `LoadAsyncContext` | backend read 失败时不提交目标 block | 沿用该 fail-closed 边界；不得在 mismatch 后把已写入的目标 buffer 标记为可复用 |

`GroupPolicy::genBlockBuffers` 当前对每个 `(group_tag, block_id)` 产生一个 `BlockBuffer`；iov 顺序为
`layer_ids` 顺序，再按该层 `buffer_resolver_` 返回的 `BlockInfo` 顺序（有 scale 时通常为 KV 后 scale）。
checksum 计算必须发生在该函数成功返回之后，不能仅对 pool 的整块连续 stride 做 hash，否则 group、padding、
scale 或分层布局改变时会与 KVCM 算法不一致。

### 13.3 TP RPC 的加法协议

建议在现有 `RemoteOperationRequestPB` 的 tag 7 起、`RemoteOperationResponsePB` 的 tag 3 起追加字段；既有 tag
和已 reserved 的 tag 3/name `group_ids` 不得复用。字段的语义应等价于：

```proto
enum RemoteChecksumActionPB {
    REMOTE_CHECKSUM_NONE = 0;
    REMOTE_CHECKSUM_SAVE_COMPUTE = 1;       // KVCM 计算并返回
    REMOTE_CHECKSUM_SAVE_RTP_STRICT = 2;    // RTP 先算，KVCM 再算并比较
    REMOTE_CHECKSUM_LOAD_VERIFY = 3;        // Get 后校验
    REMOTE_CHECKSUM_LOAD_RTP_STRICT = 4;    // KVCM 与 RTP 都校验
}

// RemoteOperationRequestPB additive fields
RemoteChecksumActionPB checksum_action = 7;
repeated int64 expected_checksums = 8;
repeated bool expected_checksum_present = 9;
string checksum_contract_id = 10;  // algorithm/version/sample/layout fingerprint
repeated uint64 block_generations = 11; // item-aligned pool reuse/ownership epoch

// RemoteOperationResponsePB additive fields
repeated int64 checksums = 3;
repeated bool checksum_present = 4;
bool checksum_action_handled = 5;
```

最终字段名可以按 RTP-LLM 命名规范调整，但以下 wire 不变量不能改变：

- `group_tags[i]`、`block_ids[i]`、`uris[i]`、`block_generations[i]`，以及存在时的 expected
  checksum/presence 和 response checksum/presence，全部按同一个 request item 下标对齐；
- `checksum_action_handled` 是协议能力确认。新 worker 识别并成功处理非 `NONE` action 后必须设置为 true；
  即使当前 rank 的 item 数为 0，或 read verify 成功但本来无需返回计算值，也能区分“新 worker 的合法空结果”
  与“老 worker 忽略了未知字段”；
- 请求 action 非 `NONE` 时，老 worker 的空 response 必须被 rank 0 判为 unsupported，不能静默降级；
- shape 错误、presence 缺失或 contract ID 不一致均使整次远端 transfer 失败；
- `actual_uris` 仍保持原语义，checksum 字段不改变 URI override 的判断。

同一 RTP deployment 的 TP worker 本来就要求 same-build，但仍必须在运行时检查上述 capability 和 vector shape，
避免灰度、误路由或异常 response 把 checksum 与另一个 block 错配。

### 13.4 写路径

#### 13.4.1 `record` / `verify`

1. rank 0 调用 `StartWrite`，得到 compact session 的 `locations` 和原请求 `block_mask`；
2. rank 0 为每个 location/spec 建立不可丢失的索引映射
   `{session_index, spec_index, spec_name, tp_rank, group_tag, block_id}`，再按 TP 广播；不得依赖 vector 扩容后
   可能失效的裸指针；
3. worker 用 `GroupPolicy` 生成 `BlockBuffers`，调用
   `SaveKvCachesOptions::WithChecksums(trace_info, trace_id)`；
4. worker 只在 Put 全部成功且 result shape 正确时返回 checksums；
5. rank 0 按步骤 2 的映射把值还原成 `LocationSpecChecksumBatch`。每个 batch 必须与 compact session
   等长；该 block 不含此 spec 的位置也保留占位；
6. rank 0 把 actual URI 和所有 checksum batch 在同一次新 `FinishWrite` 中提交。任一 TP 缺结果、重复结果或
   shape 不符时走现有 abort path，不能用 legacy `FinishWrite` 补救。

不能直接使用 `SaveKvCachesResult::location_spec_name` 给结果分组。当前 RTP 会把指向同一物理
`DeviceBlockPool` 的多个 group 去重成一个 TransferClient registration，registration 的 spec name 只是该 pool
排序后的第一个 group；而一次 pool batch 可能混合多个 group tag。正确 spec 只能取自 rank 0 在步骤 2 保存的
原始 `LocationSpecUnit::spec_name`，checksum 只按 response item 下标回填。

`executePoolTransfers` 会再次按物理 pool 对 item 分组。实现必须把 URI、buffer、block id、expected checksum
和 presence 作为一个不可拆散的 tuple 一起 partition，并把 actual URI、computed checksum 和 presence 一起
scatter 回原 request 顺序。只对 URI 做 scatter 会在 shared-pool/independent-pool 场景产生静默错配。

#### 13.4.2 `rtp_strict`

严格写在上述流程中增加多个责任边界比较：

1. worker 等待 producer event、seal block，再调用 RTP 自己的 `RtpKvcmChecksumCalculator`，生成不可覆盖的
   `trusted_write_checksums`（下文记为 `C0`）；
2. worker 将 `C0` 传给 `SaveKvCachesOptions::VerifyCallerChecksums`。KVCM 在任何 Put 之前独立重算 `C1`；不同则
   返回 `CVS_WRITE_INPUT/ER_CHECKSUM_MISMATCH`，本批不写；
3. Put 成功后必须验证 storage commit：优先由 backend 在其承诺的持久化/提交边界对实际字节按同一 contract
   返回 `C2`；这里的 checksum 必须来自 backend 已接受并可读的目标字节，不能只是再次读取 caller source
   buffer 或 DMA 入队前的 staging buffer。backend 不支持时，KVCM 用独立 scratch buffer 对新 URI 做立即
   read-back 并计算 `C2`。
   `C2 != C0` 时本次 session 失败且不得 `FinishWrite`；
4. 校验成功后，rank 0 提交的必须始终是 RTP 的 `C0`。`C1/C2` 只用于比较和诊断，不能替换信任根；
5. `FinishWrite` 成功后，rank 0 在释放 `C0` 前，用相同 compact keys/spec 集合立即执行一次
   `MatchLocationOptions::WithChecksums`，逐 spec 调用 `VerifyChecksums`，完成 `CVS_META_ROUND_TRIP`；
6. round-trip 缺失或不同均记录为生产故障，并对本 session 新写入的 compact keys 执行 best-effort
   `RemoveCache`/隔离，禁止当前进程继续使用这些位置。

当前 KVCM `SaveKvCaches` 只实现 Put 前计算，没有第 3 步的 backend checksum/read-back，因此仅凭现有三个
stage 不能宣称已经精确区分 Put、介质和 Get。RTP 的 `rtp_strict` 在 storage commit 机制实现前必须拒绝启动，
或使用明确的非全链路模式名；不能静默跳过后仍对外称为 strict。read-back 若只能执行完整 Get，其 I/O 成本
通常远高于头尾 CRC 本身，可提供 `backend`、`readback` 和 canary `sampled_readback` 策略，但只有 backend
逐写确认或 100% read-back 才满足“每个 block 都观察 commit 后状态”的要求；若还要直接归因写侧，则必须有
backend commit 边界证据，单次 read-back 只能收窄区间。

storage verifier 还必须获得原始有序 iov size/sample plan。backend 如果只看到 flatten 后的连续 blob，就无法
从总长度唯一恢复 iov 边界，也不能声称执行了同一 contract；此时应由 KVCM 随 Put 传递经过 shape 校验的紧凑
descriptor，或在独立 scratch read-back 后按原 `BlockBuffer` 布局计算。存储原生的其他全量 checksum 仍可作为
独立健康证据，但不等于可与 `C0` 直接比较的 `C2`。

checksum 的 byte domain 是 TransferClient 边界看到的逻辑 payload。若 backend 内部压缩、加密或重编码，直接
对物理介质 ciphertext/compressed bytes 计算同名 CRC 不能与 `C0` 比较；backend 必须证明编码写入和解码读取
链路，或对解码后的目标逻辑字节生成 `C2`。物理层原生 checksum 仍作为另一条独立证据记录。

步骤 5 发生在 `FinishWrite` 已发布之后，当前协议不是两阶段可见性事务，因此存在“发布成功到校验/隔离完成”
的短窗口。严格 reader 仍会做 presence 和 payload 校验，不会把 checksum 不一致的 payload 交给推理；如果业务
要求 Meta 往返验证前对所有 reader 完全不可见，需要另行设计 `VERIFYING -> SERVING` 的 session commit 协议，
不能把客户端立即 query 描述成原子发布。

当前 RTP 写任务是 best-effort 后台任务，`StorageBackend::write` 会吞掉异常。实现必须保证所有校验失败都：

- 在 `FinishWrite` 前失败时完成 session abort，不发布 checksum/locations；
- 在 `FinishWrite` 后 round-trip 失败时执行隔离并告警；
- 释放由 `StorageBackend::prepareWrite` pin 住的 device blocks；
- 不影响已经使用本地 KV 完成的当前推理，但不能静默只写一行 debug 日志。

#### 13.4.3 责任边界与可定位范围

全链路使用同一算法是必要条件，但只有在责任边界保留同一个 `C0` 并逐段比较，才能定位“哪一段写坏”。建议
使用以下诊断链：

| 边界 | 比较 | mismatch 可定位的责任区间 | 备注 |
|---|---|---|---|
| `PRODUCER_OUTPUT` | producer 完成后 RTP 生成 `C0` | 建立信任根，不做归因 | CRC 只能证明此后 bytes 未变；若 producer 一开始就算错 KV，CRC 也会把错误 bytes 当基准 |
| `WRITE_INPUT` | KVCM 的 `C1` vs `C0` | producer seal 后到 KVCM Put 前：stream 同步、buffer 生命周期、iov 布局或调用边界 | 失败时保证未 Put |
| `STORAGE_COMMIT` | backend/read-back 的 `C2` vs `C0` | 在 sealed lease 成立且 backend 对目标提交字节取证时，可定位写入路径；否则还包含源 buffer 并发改写。read-back 只能定位 Put/介质/本次 Get 区间 | 没有该边界时只能等到正常读取时发现 |
| `META_ROUND_TRIP` | query 的 `Cmeta` vs `C0` | FinishWrite、Meta 持久化和查询路径 | URI 与 checksum 都要按原 session/spec 对齐 |
| `READ_OUTPUT/KVCM` | Get 后 KVCM 的 `C3` vs 可信 baseline | commit 之后的介质、Get 或目标 buffer 写入 | 若已有可信 backend commit 证据，可排除初次 Put |
| `CONSUMER_INPUT/RTP` | KVCM 返回后 RTP 的 `C4` vs 可信 baseline | KVCM 校验返回到推理消费前，或 KVCM verifier 自身异常 | 通过后才允许发布 block |

每个比较事件都记录同一个 `trace_id`、`contract_id`、instance/key/spec、TP rank、block id、URI、expected `C0`、
actual 和 stage。诊断结论应表述为“首个失败边界对应的责任区间”，而不是在没有 read-back/backend 证据时直接
指认某个组件。跨进程长期读取若拿不到独立 `C0`，仍受第 13.8 节信任根生命周期限制。

#### 13.4.4 Buffer 所有权、TOCTOU 与 I/O 完成性

checksum 比较只能观察两个时间点。若校验与实际传输之间允许 buffer 被改写，就存在 TOCTOU：`C1 == C0`
之后 producer 仍可覆盖显存，Put 把新字节写入 storage，最终 `C2 != C0`；此时不能把故障直接归到 backend。
RTP strict 必须把 block 生命周期实现成可审计状态机，而不是只约定“调用方应该别改”：

```text
MUTABLE(producer owns)
  -- producer event complete + seal(generation) + C0 --> SEALED
  -- KVCM contract/C1 pass + acquire write lease ------> TRANSFERRING
  -- backend I/O quiesced + committed-byte C2 pass ----> COMMITTED
  -- FinishWrite + Meta round-trip pass ----------------> SERVING

read target:
ALLOCATED -- Get fully quiesced --> KVCM C3 pass --> RTP C4 pass --> CONSUMABLE
```

实现不变量：

- `seal` 必须绑定 pool block 的 monotonic generation/epoch；同一 GPU 地址释放并复用后属于新 generation。
  checksum、TP RPC、trace 和 pin/lease 均核对 generation，不能只靠 pointer 或可复用 block id；
- producer 在自己的 stream 上记录 event。RTP checksum stream 通过 `cudaStreamWaitEvent`/MUSA 等价机制建立依赖，
  生成 `C0` 后取得不可写 lease；仅 `cudaStreamSynchronize` checksum 自己的 stream 不能等待另一个 producer
  stream。当前 KVCM options 没有 producer event，因此该同步与 seal 必须由 RTP 在调用前完成，或后续扩展 API；
- 从 `C0` 到 Put/commit 验证结束，pool 不得回收，任何 kernel 不得取得写 lease。refcount/pin 只防 free/reuse，
  仍需所有权状态阻止合法地址上的并发写；
- backend `Put` 成功必须表示源 buffer 已不再被 DMA/线程读取；backend checksum 必须基于目标提交字节。
  若 wrapper timeout 后任务或 DMA 仍可能继续，block 保持 pinned/quarantined 直到 backend 明确 quiesced，不能因
  API 已返回就释放或复用；这种 timeout 也不能产生有效 `C2`；
- backend `Get` 成功必须表示对目标 buffer 的写入已经完成并对 checksum stream 可见。失败/timeout 后若仍可能
  有 DMA 写入，目标 generation 永不发布，并在 I/O quiesced 前不得交还 pool；
- `C3` 通过到 `C4` 通过之间目标 buffer 同样保持 sealed。只有 `C4` 成功后才把 block 发布给 BlockTree/推理；
- read-back 要从新 URI 对应的权威介质读取，绕过会直接返回 source/staging bytes 的 client cache，并使用独立
  scratch generation。否则 `C2` 通过不能证明 storage 中的字节正确。

只有这些不变量都能由状态、event/lease 和 backend completion contract 证明时，首个失败 stage 才具有稳定的
责任含义。日志里写一句 sealed 而代码没有阻止写 lease，不构成归因证据。

### 13.5 读路径

`verify` 和 `rtp_strict` 的 rank 0 流程如下：

1. `match` 调用带 `WithChecksums()` 的新 API，并把完整 `MatchLocationResult` 保存到 `KVCMMatchMeta`；
2. 按 `GroupPolicy` 选中的 spec 检查每个 block 的 result vector 长度、spec 集合和 presence；
3. prefix cache 必须保持连续：从第一个缺少任一必需 spec checksum 的 block 起截断远端命中，而不是跳过中间
   block 后继续使用后缀；返回 matched block 数前，必须把 `locations` 和每个 checksum/presence vector 按同一
   prefix 一起裁剪。老 metadata、老 server 和滚动期间未覆盖的位置因此表现为 cache miss；
4. 为每个发送给 TP worker 的 URI 附加同下标 expected checksum/presence；
5. worker 在任何 Get 前校验 shape、contract 和本机计算能力，再调用
   `LoadKvCachesOptions::VerifyWith(expected, presence, trace_info, trace_id)`；
6. KVCM Get 成功后计算并比较，mismatch 返回 `CVS_READ_OUTPUT`。`rtp_strict` 随后再用 RTP calculator 对目标
   `BlockBuffers` 重算一次；任一 verifier 不通过都返回失败；
7. 只有所有 TP rank 成功，`LoadAsyncContext` 才提交并把远端 blocks 发布到 BlockTree。

必须在 rank 0 对 presence 做严格判断，不能仅调用 KVCM 的宽松
`LoadKvCachesOptions::VerifyWith(match, spec_name)` helper；该 helper 会跳过 `presence=false`，适合迁移工具，
不满足 RTP 生产读的“没有基准就不读”要求。

当前失败语义已经适合作为安全边界：backend read exception 会令 `LoadAsyncContext` 失败，`matched_blocks_`
不会扩大到 backend 命中，目标 block 不会进入复用树。PREFILL 且不是 allocator 错误时，
`StreamCacheResource::finalizeAllocatorLoad` 会保留合法本地复用、丢弃远端结果并继续正常计算；非 PREFILL
维持现有向上报错行为。实现和测试必须锁定这一行为，不能在 checksum 分支单独发布部分成功 TP 的 block。

普通 reader 使用 query 返回值校验 payload，完成的是 `READ_OUTPUT`。它不能再拿同一个 query value与自身比较
并宣称完成 `META_ROUND_TRIP`；后者只在写端仍持有独立 RTP 值时成立。

### 13.6 RTP 独立计算器

建议在 RTP KVCM backend 下新增独立组件 `RtpKvcmChecksumCalculator`，输入直接使用已经由 `GroupPolicy`
生成的 `BlockBuffers`。不要直接 include KVCM 的 internal `SdkBufferCheckUtil`：那会把 RTP 绑定到 KVCM 私有实现，
也失去调用方独立计算对共享实现缺陷的发现能力。

RTP calculator 按协商出的 contract 选择实现。若受控兼容模式选择
`kvcm.crc32_xor_int64.legacy-v0`，必须精确复现当前第 7 节的历史语义：

1. 展平每个 block 的 iov，同时保存 block 的起止 offset；
2. 每个 iov 使用 IEEE reflected CRC32 `0xEDB88320`，只处理
   `head[min(sample_bytes, size/2)] + tail[same]`；不是 CRC32C，也不写 payload footer；
3. 按原始 iov 顺序用同一 `HashIntArray` 64 位回绕和算术右移语义聚合；
4. 输出与输入 `BlockBuffers` 一一对应的 `int64_t`，不把 `0` 当缺失；
5. CUDA 与 MUSA 使用等价实现；pool 固定绑定本 rank 的 device，复用 pinned host/GPU scratch，不能在每个
   block 上同步分配；
6. 非 GPU、空/ignore/null/小于 2 字节的 iov、跨 device pointer、kernel 或 D2H 失败全部 fail-closed。

若选择第 7.2 节的 `sampled_crc32_layout_v1`，则执行其小 iov、长度绑定、domain 和聚合规则；该版本允许
1-byte iov，但仍拒绝空/ignore/null。两个算法的行为不能由同一个模糊的 `crc32` 名称分支，RTP 收到未知
version、`W` 或 layout signature 时必须在任何数据 I/O 前返回 `CONTRACT_MISMATCH/UNSUPPORTED`。

为了防止两个仓库各自“看起来相同”但逐步漂移，KVCM 应公开稳定的 checksum contract descriptor 和 golden
vectors（算法名、版本、resolved sample bytes、边界/聚合样例）；RTP 初始化时读取/比较 descriptor，CI 对同一批
golden `BlockBuffers` 同时运行 KVCM 与 RTP 实现。descriptor 不等于共享计算实现：严格模式仍保留两套 calculator。
contract mismatch 是配置/协议故障，必须与 payload mismatch 分开计数；否则会错误指控 writer 写坏数据。

`rtp_strict` 的基础 GPU 次数是写两次（RTP + KVCM）和读两次（KVCM + RTP），另加 storage commit verifier；
若 commit 使用 read-back，还会增加一次 storage read 及一次 checksum。`record/verify` 各一次。不能在没有目标
GPU/storage 压测前默认严格模式开销可忽略。若性能评估后只需要发现合并的 Put/介质/Get 区间损坏，生产可选择
`verify`，但文档和指标必须如实标识它没有独立 RTP verifier，也没有 storage commit 归因能力。

### 13.7 算法身份与 Instance 隔离

当前 RTP 生成的 KVCM `instance_id` 已包含 block size、model、dtype、TP/DP、MLA、location spec size 等信息，
但没有 checksum 采样参数，也没有完整 iov 边界/顺序。开启任一 checksum 模式时，identity 输入必须追加：

```text
checksum_contract = sampled_crc32_layout/v1
checksum_sample_bytes = <进程启动时解析后的值>
iov_layout_signature = hash(
  ordered spec/group tag,
  ordered layer ids,
  each layer's ordered BlockInfo sizes,
  scale inclusion/layout version)
```

地址、block id 和 pool 容量不能进入 layout signature。使用自定义 `KVCM_CLIENT_CONFIG` 时，RTP 无权重写其中的
`instance_id`，因此 checksum 模式必须额外要求调用方提供并确认相同的 contract/layout identity；不能在未知
identity 上开启 strict。

`KVCM_CHECK_IOV_BYTE_SIZE` 当前在 KVCM kernel translation unit 静态初始化时读取。legacy-v0 下所有 rank 必须
在进程启动前得到同一值，RTP calculator 也只能读取一次；初始化时将 resolved 值跨 rank 校验，并与 KVCM
descriptor 比较。新生产 contract 应把 `W` 变成 descriptor 的显式配置，不再把隐藏环境变量当作协议。运行中
修改环境变量不生效。改变算法、采样字节数或 iov layout 必须产生新 instance/baseline，不能让新旧值在同一
instance 滚动混用。

### 13.8 信任根的生命周期

RTP 在 strict write 中生成的值是独立信任根，但默认只需保留到：

```text
RTP C0 -> KVCM WRITE_INPUT -> STORAGE_COMMIT -> FinishWrite -> immediate query/META_ROUND_TRIP
```

完成后可以释放，后续 reader 使用 KVCM Meta 中的 baseline 检查 payload。可信 backend `STORAGE_COMMIT`
证据通过后，后续 `READ_OUTPUT` mismatch 可以排除初次 Put，把范围收窄到 commit 之后的介质/Get/目标 buffer；
只有客户端 read-back 时仍应把该次 Get 纳入诊断区间。没有 commit verifier 时只能报告 Put/介质/Get 合并区间。
该方案不能发现“在写端释放信任根之后，Meta 中 URI 与 checksum 被一致篡改”的情况，也不是防攻击设计。

如果业务要求任意未来 reader 都能重新证明 Meta 本身正确，必须把
`(instance_id, key, spec_name, checksum_contract, checksum)` 保存到 KVCM 之外的可信 manifest，或引入带密钥的
认证值/内容寻址。把第二份值仍放在同一个 KVCM Meta backend，或查询后原样回传给 KVCM，不会形成独立信任根。

### 13.9 可观测性

RTP 现有 remote match/read/write/SDK metrics 不能区分 checksum failure。应新增独立低基数指标组，至少包含：

- event QPS/count：labels 至少覆盖
  `stage=producer_output|write_input|storage_commit|meta_round_trip|read_output|consumer_input`、
  `outcome=success|mismatch|contract_mismatch|lease_violation|missing|unavailable|malformed|unsupported`、
  `verifier=kvcm|rtp|backend`；
- processed block 数和 checksum compute/verify latency；
- query presence 覆盖率，以及因缺失而截断的 block 数；
- strict 模式 GPU pool wait/timeout（若 calculator pool 暴露）；
- write abort、post-Finish quarantine/remove 成功与失败数。

`trace_id`、write session、rank、group tag、spec name、block id、block generation、contract/layout、URI、
expected/actual 值写入受采样/限流保护的结构化日志，用于关联 KVCM 的 `ChecksumValidationLog`，不得作为常规
metric label。spec/group 只有确认配置集合有界时才允许成为 label。由于后台 write 不向推理调用方返回失败，
write checksum 指标和告警是上线硬门禁，不是可选增强。

### 13.10 灰度、兼容与回滚

发布顺序必须为：

1. 先升级全部 KVCM Manager；
2. 升级 RTP 使用的新 KVCM client RPM，但保持 `off`；
3. canary 开启 `record`，确认 Save、Finish、query presence 和资源开销；
4. presence 覆盖率达到目标后开启 `verify`；老数据缺失只形成 cache miss；
5. 完成 RTP/KVCM bit-exact 与性能门禁后，再小流量开启 `rtp_strict`；
6. 最后扩大 strict 范围，持续按 stage 观察。

回滚反向执行：先退到 `verify`/`record`，再 `off`，最后才回滚 client/server。任一非 `off` 模式遇到老 TP worker
或不支持 checksum 的 client subclass 都必须由 capability/shape 检查识别；不得因 proto 未知字段被忽略而假成功。

默认 mode 必须保持 `off`。mode、算法 contract 和 sample bytes 要加入 Python server args、C++ `KVCacheConfig`、
pybind 及配置打印；日志不得输出含密钥的 storage 配置。内源和外源都使用共享的 BlockTree/KVCM 改动，必须分别
验证；本功能不依赖已停止维护的 VCNS 专用 test target。

### 13.11 测试与性能门禁

实现至少覆盖以下测试矩阵：

| 层级 | 必测内容 |
|---|---|
| calculator UT | 0/负数结果、legacy 拒绝而新 contract 接受 1-byte iov、奇偶 iov、异构 size、KV+scale、不同 iov 顺序、跨 block 边界、legacy 确定碰撞反例、新 contract 的 size/中心字节区分、KVCM/RTP golden vectors |
| `ClientWrapper` UT | 新 options/result 原样转发；错误码不被压平；old subclass 返回 unavailable；0 与 presence 分离 |
| KVCM backend mock | 单/多 TP、FULL/LINEAR、多 group、shared/independent pool、compact StartWrite mask、actual URI 与 checksum 同时返回 |
| shape/fault UT | response reorder、少/多 value、duplicate/missing spec、老 worker capability 缺失、contract/layout 不同、generation/lease 不同、部分 TP 失败、abort/释放 pin |
| proto compatibility | 新字段只追加；旧序列化可被新代码读取；新请求到旧 worker 必须 fail-closed 而非假成功 |
| load 语义 | presence 缺失截断前缀；sampled byte 篡改不发布 block；PREFILL 安全重算；非 PREFILL 保持现有报错 |
| write 语义 | RTP/KVCM 写前不同不 Put；C1 后并发改写被 lease 阻止；backend checksum/read-back 发现 commit 损坏；timeout 后 DMA quiesce 前不复用；Finish 前失败 abort；Meta round-trip mismatch 隔离；后台失败有 metrics |
| GPU 集成 | CUDA/MUSA 真机、多 device、多线程、pool 饱和、跨 device 拒绝、write→篡改采样区→read；中段未采样篡改作为已知边界 |
| 内外源回归 | 外源全部相关 UT/集成；内源依赖图下同等 UT/集成；不把 VCNS 非必要 target 当门禁 |

性能报告必须在相同模型、block/group/iov 分布上同时给出 `off`、`record`、`verify`、`rtp_strict`：

- write/read 吞吐与 P50/P95/P99；
- checksum kernel、D2H/host aggregate、pool wait 和端到端 broadcast latency；
- 每 block、每 iov 的增量，以及 TP/group/pool 数变化后的扩展性；
- GPU/pinned host 常驻与峰值内存；
- presence 协议带来的 Meta/broadcast 字节增量；
- mismatch 路径的隔离、fallback 和重算时延。

任何模式都不得只用全零 fixture 做性能测试；必须先断言结果与 KVCM golden vector 一致。`rtp_strict` 上线还需
证明双计算不会破坏目标服务 P99/SLA。未完成 CUDA/MUSA 真机和故障注入时，只能合入保持 `off`，不能生产开启。

### 13.12 预计变更边界

RTP-LLM 侧预计修改：

- `KVCMStorageBackend.{h,cc}`：模式、match metadata、batch 对齐、TP gather/scatter、失败语义；
- `ClientWrapper.{h,cc}` / `ClientFactory` 及 mocks：新 KVCM API 和结构化错误；
- `GroupPolicy` 附近：稳定 layout descriptor/signature，不改变现有 iov 顺序；
- `model_rpc_service.proto`：第 13.3 节加法字段；
- 新的 `RtpKvcmChecksumCalculator` CUDA/MUSA 实现、pool、golden tests；
- `KVCacheConfig`、server args、pybind、配置打印和 production metrics；
- KVCM backend 的 shared/independent pool、FULL/LINEAR 和端到端集成测试；
- KVCM client/server RPM 升级到包含本设计 API 的版本。

KVCM 侧在 RTP strict 开启前还需提供第 7.2 节的新算法、稳定 checksum contract descriptor/golden vector
接口，以及 storage commit verifier 所需的 backend 能力；若 API 以新增 virtual 形式提供，继续遵守第 9 节的
slot 追加和老 subclass fail-closed fallback 约束。RTP 接入完成后，应把本节的“待实现”改为实际 commit/测试
追踪表，并删除不再成立的计划性描述。

## 14. 当前限制与后续工作

- Python binding 暂时保留 legacy Save/Load 签名，尚未暴露 Options/Result；vLLM、SGLang、TRT-LLM
  connector 因此尚未自动串起 checksum 链路。
- `RTPLLMClient` 继续走默认不请求 checksum 的兼容路径；其内部调用新 overload，但显式保持
  `include_checksums=false`。
- RTP-LLM BlockTree KVCM backend 尚未实现第 13 节的 mode、TP checksum 协议或独立 calculator；在完成实现和
  门禁前，不能把 KVCM client 自身能力描述为 RTP-LLM 已具备端到端校验。
- Meta 未记录 checksum 算法和版本；引入第二种算法前必须先设计逐值版本化与迁移策略。
- 当前 legacy 算法不把 iov length/count/domain 纳入结果，小型奇数 iov 会漏掉中心字节；这些行为已经构成
  persisted checksum 语义，不能在原枚举下静默修正。RTP strict 依赖的 `sampled_crc32_layout_v1`、descriptor
  握手和新 baseline 迁移尚未实现。
- 当前 KVCM options 不接收 producer event 或 block generation，也不管理 RTP pool 的 write lease；严格接入
  必须由 RTP 先实现 seal/happens-before，并在 backend timeout 仍可能有 I/O 时延迟复用，不能仅凭 refcount。
- 当前实现只拒绝 `KVCM_CHECK_IOV_BYTE_SIZE <= 0`，没有 sampled bytes 总预算或正值上限；错误的极大配置会把
  抽样退化成高延迟扫描。第 7.3/12.2 节的资源上限实现前不能把可配置 `W` 视为安全的生产参数。
- inline header 尚未实现，配置会被明确拒绝。
- Manager 不主动比较不同 spec 的值；它们通常对应不同 TP/PP payload，本来就可以不同。同名 spec 在多个
  storage 副本间应保持一致，当前依靠写端/迁移流程和读端校验保证。
- 需要在真实 CUDA/MUSA 环境补充 write → 篡改 storage byte → read 的端到端故障注入测试。
- 当前 MUSA 自动测试只执行 pool 配置边界和 warm-up；其余 `SdkBufferCheckUtilTest` 用例由
  `USING_CUDA` 条件保护。MUSA 生产启用前必须补齐并执行同等强度的 MUSA 测试，不能把 target 构建成功
  当作功能覆盖。
- 如需默认覆盖完整 block，应新增明确命名、参数稳定的全量 checksum 算法，而不是静默改变现有
  `crc32_xor_int64` 的结果。
