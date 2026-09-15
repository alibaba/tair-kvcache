# KVCache 读写链路数据完整性校验

## 1. 状态与目标

本文描述当前实现，覆盖以下能力：

- KVCM 在写前计算 checksum，并在读后校验；
- 调用方自行计算 checksum，KVCM 负责保存并在查询时原样返回；
- 调用方使用与 KVCM 相同的算法时，可显式要求 KVCM 在写入、元数据往返和读取三个阶段校验；
- checksum 随独立存储的 `LocationSpec` 持久化，并在 spec 合并、状态更新和分层迁移时保留；
- 老客户端、老元数据和未开启校验的调用保持原行为。

当前只实现 meta checksum 方案。inline header 的协议和配置槽位已预留，但本版本拒绝启用。
完整性逻辑位于共享的 client、manager 和 meta 层，与具体数据存储后端无关，不依赖后端专用校验 target。

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
| KVCM | 写前计算、保存，按需读后校验 | 使用内置算法的端到端检查 |
| 调用方，自定义算法 | 仅保存并返回 | 调用方维护自己的算法和校验逻辑 |
| 调用方，与 KVCM 同算法 | 显式执行三阶段 strict 校验 | 既保留调用方可信基准，又定位故障区间 |

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

auto load_ec = transfer_client->LoadKvCaches(
    uris,
    buffers,
    LoadKvCachesOptions::VerifyWith(match, save_result.location_spec_name));
```

`MatchMetaOptions::WithChecksums()` 与 `MatchMetaResult` 提供相同能力。HTTP/proto 的
`GetCacheLocationRequest`、`GetCacheLocationsByBackendRequest` 和 `GetCacheMetaRequest` 均通过
`include_checksums=true` 显式请求；默认值 `false` 不复制或发送已保存的 checksum。C++ client 在未请求时
返回空的 `checksum_results`；请求后可通过 `FindChecksums(location_spec_name)` 取得对应组。

HTTP JSON 响应为保持项目既有的 protobuf JSON 输出格式，会打印标量默认值；因此未请求时仍可能看到
每个 location spec 中的 `checksum="0"` 和 `checksum_present=false`，它们只表示“未返回”，并非已保存值。gRPC protobuf binary
不会编码这些默认标量。

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

// 2. 写成功后提交调用方的可信值。
auto finish_ec = meta_client->FinishWrite(
    trace_id,
    write_session_id,
    success_mask,
    locations,
    FinishWriteOptions::WithChecksums("tp0", trusted_session_checksums));

// 3. META_ROUND_TRIP：查询值必须与调用方可信值一致。
auto [match_ec, match] = meta_client->MatchLocation(
    trace_id,
    query_type,
    keys,
    tokens,
    block_mask,
    location_spec_names,
    MatchLocationOptions::WithChecksums());
auto meta_verify = match.VerifyChecksums("tp0", trusted_query_order_checksums);

// 4. READ_OUTPUT：Get 完成后，KVCM 再计算目标 buffer 并与可信值比较。
auto load_ec = transfer_client->LoadKvCaches(
    uris,
    buffers,
    LoadKvCachesOptions::VerifyWith(trusted_query_order_checksums, trace_info));
```

三个阶段的含义为：

| 阶段 | 比较内容 | 失败所定位的区间 |
|---|---|---|
| `WRITE_INPUT` | Put 前 buffer vs 调用方可信 checksum | 调用方算法/参数不一致，或进入 KVCM 前数据已变化 |
| `META_ROUND_TRIP` | 查询返回值 vs 调用方可信 checksum | FinishWrite、Meta 持久化或查询返回链路 |
| `READ_OUTPUT` | Get 后 buffer vs 可信 checksum | Put、存储介质或 Get 链路 |

只有三个阶段均通过，才能形成完整诊断链。Manager/Meta 不持有数据，不能在 FinishWrite 内证明实际存储
内容正确。

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

校验错误日志使用统一的 `ChecksumValidationLog`，包含可用的 `stage`、`trace_id`、`block_index`、
expected/actual checksum、URI 和 block id。Options 中的 `trace_id` 与 `TransferTraceInfo::block_ids` 用于补齐
诊断上下文；`TransferTraceInfo` 本身保持原始布局，避免破坏已发布 C++ Client 的二进制兼容性。

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

- `KVCM_SDK_CHECK_CELL_NUM`：pool cell 数，默认 `4`；
- `KVCM_SDK_MAX_CHECK_IOV_NUM`：单次计算 chunk 的最大 iov 数，默认 `500000`；
- `KVCM_CHECK_IOV_BYTE_SIZE`：每个 iov 头部和尾部各参与 CRC 的最大字节数，默认 `4`。

cell 数、最大 iov 数或采样字节数小于等于 `0` 时 TransferClient 初始化失败；最大 iov 数超过 GPU kernel
的 `int` 索引上限或内存大小计算上限时同样拒绝，避免形成永远无法取得 cell、缓冲区大小回绕或“请求成功
但无法计算”的配置。

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

内置 kernel 当前只接受完整 GPU `BlockBuffer`：每个 iov 必须为 GPU memory、非空、非 ignore、
base 非空且 size 至少为 2 字节。采样长度按每个 iov 独立计算；iov 数量相同但各 iov size 不同的 block
可合并到同一 GPU chunk，iov 数量不同或超过单 chunk 上限时拆批，输出顺序保持不变。

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
| `GetCacheMetaRequest` | `include_checksums` | 7 |
| meta/admin `LocationSpec` | `checksum` | 3 |
| meta/admin `LocationSpec` | `checksum_present` | 4 |
| meta/kv-meta `StorageConfig` | `integrity` | 11 |
| admin `StorageConfig` | `integrity` | 12 |

滚动升级行为：

- 老 client → 新 server：不携带 checksum batches，按历史路径执行；
- 新 client 仍发送 legacy `locations` 字段，以兼容可能读取 tag 5 的旧 server；新 server 的 checksum
  对齐只使用 session keys 和 tag 6，不信任该字段；
- 新 client → 老 server：老 server 忽略未知 checksum_batches 字段，因此不会建立完整性基准；调用方必须通过后续
  query 的 `checksum_present` 确认能力，不能仅以 FinishWrite 成功作为依据；
- 新 client 读取老 metadata/server：checksum presence 为 false；
- 新 server 仅在 query 的 `include_checksums=true` 时携带已保存值；老 client 不设置该新字段，维持历史响应。

## 10. 基础性能验证

仓库提供手工运行的 CPU/协议基准，覆盖 4096-block checksum 比较，以及每个 block 含两个独立 checksum
spec 时 gRPC protobuf binary 的序列化/反序列化和 HTTP 快速 JSON 序列化；同时输出 opt-in 前后的
字节数和耗时：

```bash
bazelisk run -c opt //kv_cache_manager/service/util/test:data_integrity_benchmark
```

基准只做 fixture 正确性断言并报告数值，不设置容易受机器负载影响的硬延迟阈值。GPU kernel 的吞吐与真实
存储读写故障注入仍须在目标 CUDA/MUSA 机器上单独测量。

## 11. 当前限制与后续工作

- Python binding 暂时保留 legacy Save/Load 签名，尚未暴露 Options/Result；vLLM、SGLang、TRT-LLM
  connector 因此尚未自动串起 checksum 链路。
- `RTPLLMClient` 继续走默认不请求 checksum 的兼容路径；其内部调用新 overload，但显式保持
  `include_checksums=false`。
- Meta 未记录 checksum 算法和版本；引入第二种算法前必须先设计逐值版本化与迁移策略。
- inline header 尚未实现，配置会被明确拒绝。
- Manager 不主动比较不同 spec 的值；它们通常对应不同 TP/PP payload，本来就可以不同。同名 spec 在多个
  storage 副本间应保持一致，当前依靠写端/迁移流程和读端校验保证。
- 需要在真实 CUDA/MUSA 环境补充 write → 篡改 storage byte → read 的端到端故障注入测试。
- 如需默认覆盖完整 block，应新增明确命名、参数稳定的全量 checksum 算法，而不是静默改变现有
  `crc32_xor_int64` 的结果。
