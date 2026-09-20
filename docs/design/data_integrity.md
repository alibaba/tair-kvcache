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

当前 KVCM 候选树已实现显式 checksum 请求的 fail-closed 语义，包括 GPU 计算能力探测、
`W`/工作量上限、写前校验、读后校验、Meta 往返比较和 FinishWrite 异常收敛。默认未开启的
合法历史调用仍保持原行为。当前 RTP-LLM 候选树也已接入 `off` / `record` / `verify`，但可发布范围
必须限定为：

- `off`：不启用 checksum/身份 contract；URI 未变化时保持 legacy Finish 和原有 TP 形态。若 backend 返回
  dynamic actual URI，仍必须使用 client v4/tag 7 提交并精确回读；transfer-status v1 只能把部分成功 URI
  安全带回 coordinator，不能代替 v4 提交/回收能力；
- `record` / `verify`：当前只允许 `tp_size=1` 初始化；
- `rtp_strict`：当前无条件拒绝初始化，不存在“开启后再降级”的路径。

原因不是 checksum 协议字段缺失：RTP TP 协议 v3 已携带 request tag 11 generation、tag 12 target rank
和 tag 13 pool epoch。当前安全实现不做 transfer-time 当前 token 查询：查询当下 token 无法证明 block id
在此前没有被回收并复用，反而会把新占用者错认为原请求。TP=1 时，
coordinator 从同一进程中已被 `StorageBackend` 外层 pin 住的 pool 直接捕获 `{pool_epoch,generation}`，worker 在
解析 GPU buffer 前原子验证 token 并增加 pin。TP>1 必须等待 distributed allocator 携带 allocation-time logical lease；
在此之前 non-off 初始化 fail-closed。

request tag 14 / response tag 9 另行协商 transport business status，用于在 gRPC OK response 中回收部分失败前
已经产生的稀疏 actual URI；它不是身份 token，也不能替代上述 allocation-time lease。

另一个关键边界是：RTP 当前的 `C0` / `C3` 不是独立算法实现，而是通过 KVCM client v4 公开的
`TransferClient::CalculateChecksums` 调用同一套 calculator/kernel；`C1` / `C2` 分别由 KVCM Save/Load 内部调用同一
calculator 完成。`record` 每次写入计算两次（`C0` + `C1`）；`verify` 写入也是两次，读取再计算
两次（`C2` + `C3`）。这些相邻边界比较可以发现期间的数据变化，但不能发现共享 calculator 的系统性缺陷，
也不是独立信任根。

因此“KVCM 代码路径可用”、“RTP `tp_size=1` 的 `verify` 目标可验证”和“生产可开启”是三个门槛。
当前已有 CUDA 真机 calculator 测试和采样基准，但 RTP 官方依赖仍固定在不含 v4 API 的 2026-04-29 RPM，
完整 RTP 端到端、storage 故障注入和业务 P99 门禁尚未完成。在第 12 节的未完成项清零前，生产默认仍必须
保持 `off`；不能用本地 override RPM 的开发验证替代可复现的官方制品发布。

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

上图是 KVCM API 的通用三阶段语义。RTP 当前实现在这个链路上额外定义了四个计算点，但必须注意
它们都调用同一 KVCM calculator/kernel：

| RTP 计算点 | 实际调用 | 发生时机 | 模式 |
|---|---|---|---|
| `C0` | `TransferClient::CalculateChecksums` | worker 对写源 buffer 做 snapshot | `record` / `verify` |
| `C1` | `SaveKvCaches(VerifyCallerChecksums)` 内部 calculator | Put 之前重算并与 `C0` 比较 | `record` / `verify` |
| `C2` | `LoadKvCaches(VerifyWith)` 内部 calculator | Get 完成后与 Meta baseline 比较 | `verify` |
| `C3` | `TransferClient::CalculateChecksums` | `C2` 通过后对读目标再做 snapshot | `verify` |

写侧显式比较 `C1` 与 `C0`；读侧 `C2` 和 `C3` 分别与同一个 Meta baseline 比较（`C2` 不作为独立值暴露给
RTP 再与 `C3` 直接比较）。这些是相邻时间点的双重检查，不是两套独立实现。FinishWrite 后还会立即查询并
比较 exact URI/spec/checksum，该 `META_ROUND_TRIP` 是 CPU 元数据比较，不额外启动 checksum kernel。

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
5. 校验按 block 逐项比较，不使用 batch 聚合值，因此能定位 checksum 不同的下标，也通常能发现 checksum
   不同的 block 交换；相同采样 CRC 或其他碰撞仍无法靠该值证明 block/iov 身份。
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

protobuf 输入为兼容早期只写 value 的客户端，统一按
`checksum_present || checksum != 0` 推断 presence：非零 legacy value 会被视为存在，合法值 `0` 仍必须显式
携带 `checksum_present=true`。JSON 持久化则只以 `checksum` 字段是否存在恢复 presence，不能把数值 `0`
当作缺省哨兵。

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
状态，必须补齐第 13.8 节列为缺口的 backend committed-byte checksum 或立即 read-back；其中只有 backend 在
commit 边界生成的证据才能直接归因写侧，read-back 仍包含一次 Get，不能仅凭三个阶段或一次回读指认组件。

### 3.4 RTP 当前的 caller-checksum 用法

RTP 当前不再声称有一套独立 calculator。worker 先通过 client v4 的
`CalculateChecksums` 计算 `C0`，再将它传给 `VerifyCallerChecksums`；Save 内部另一次调用同一 kernel 得到
`C1`。Put 成功后，RTP 始终把 `C0` 作为 checksum batch 提交，并要求 Save 返回的 checksum 与 `C0`
相等。这使 `record` 的每个写 batch 执行两次 checksum，而不是“只计算后保存一次”。

`verify` 的写路径与 `record` 相同。读路径先由 `LoadKvCaches(VerifyWith)` 在 Get 完成后计算 `C2`
并与 Meta baseline 比较；通过后 RTP 再调用 `CalculateChecksums` 计算 `C3` 并做第二次比较。因此
`verify` 是写两次+读两次。四次都受相同的 GPU pool、算法、resolved `W` 和 256 MiB 默认请求预算限制。

这个实现是“同一个 calculator 在多个边界重算”，可以检出两次计算之间的变化；它不是第二个独立算法
信任根。`rtp_strict` 预留名称仅用于明确拒绝：当前 init 会直接失败，不会执行上述路径后再宣称 strict。

## 4. KVCM 显式校验的 strict 语义与错误

本节的 strict 指“KVCM 显式请求不允许静默跳过”，不等于 RTP 配置值 `rtp_strict`。后者当前拒绝初始化。
显式请求计算或校验采用以下 fail-closed 语义：

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
`stage`、`trace_id`、`block_index`、expected/actual checksum、脱敏后的 `storage_endpoint`（仅 scheme 和
host/port）及 block id，不记录可能携带凭据、对象名或 allocation capability 的完整 URI。Options 中的 `trace_id` 与
`TransferTraceInfo::block_ids` 用于补齐诊断上下文；`TransferTraceInfo` 本身保持原始布局，避免破坏已发布
C++ Client 的二进制兼容性。`META_ROUND_TRIP` 是纯比较 helper，只返回 stage、mismatch 和 faulty indices，
业务接入层必须检查结果并补充同等结构化日志，不能忽略返回值。

通用 KVCM client 当前没有内置 checksum mismatch/latency Prometheus counter；结构化日志是它自身唯一的
直接故障证据。RTP 接入会把失败汇总到第 13.6 节的 event/affected-block QPS，但这仍不包含计算延迟、pool
等待或明确的 abort outcome。非 RTP 调用方必须自行接入日志告警或外层低基数指标，不能把“有日志”当作
完整 SRE 闭环。

### 4.1 失败、副作用与重试契约

| 失败点 | 对外结果 | 数据/元数据副作用 | 正确处理 |
|---|---|---|---|
| Save 的 URI/buffer 数量非法 | `ER_INVALID_PARAMS` | 未执行 Put | 修正请求后重试 |
| `WRITE_INPUT` batch shape 错误或 checksum 不同 | `ER_CHECKSUM_MISMATCH` | 未执行 Put | 检查 block 顺序、iov 布局、采样参数和算法 |
| `WRITE_INPUT` 无 kernel/pool、buffer 不可计算 | `ER_CHECKSUM_UNAVAILABLE` | 未执行 Put | 不得降级成成功；修正能力或关闭显式校验 |
| 存储 Put 失败 | 透传后端错误；结果中的 checksums/actual URI 为空 | Put 已尝试，Meta 尚未 Finish；单次 SDK Put 内可能已有上层不可见的部分分配/写入 | 用失败 mask 完成会话或等待超时清理；不得提交 checksum。只有调用方实际拿到的 URI 才能由 v4 abort 回收；backend 必须另有原子性/失败回滚保证，否则需人工对账 |
| FinishWrite 的 instance/mask/batch/spec/URI 结构非法 | Manager 为 `EC_BADARGS`，gRPC C++ client 映射为 `ER_SERVICE_INVALID_ARGUMENT` | session 未消费，Meta 未更新 | 在 session 超时前修正并用同一 session 重试 |
| FinishWrite 的 instance 不存在 | `EC_INSTANCE_NOT_EXIST` / `ER_SERVICE_INSTANCE_NOT_EXIST` | session 未消费 | 修复路由/实例；不要跨 Instance 完成会话 |
| FinishWrite URI 可解析但违反 safe-replacement 策略 | Manager 主 RMW 槽失败，整体返回内部错误 | **session 已消费**；metadata 不会被重定向；同一 unsafe URI 也不能被 cleanup 认领，可能留下未绑定 backend 分配并触发终态 cleanup 指标 | 不可重试原 session；隔离/查询后重新 StartWrite，并按 backend allocation identity 人工回收 |
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

“非法请求不消费”只覆盖 session 仍在 Manager 内时能够完成的结构/预算/归属前置校验。会话取出后执行的
safe URI replacement、Meta COW 和持久化不是事务的一部分；这些阶段失败时 session 已消费。这沿用现有
FinishWrite 语义，也是调用方必须区分“修正原请求”和“重新发起写入”的原因。

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

client v4 的 `LocationSpecUriBatch` 与 checksum batch 遵循相同的 compact-session 对齐，并额外携带与 URI
等长的 `uri_present`。`false` 表示该 block 没有这个 spec，对应 URI 必须为空占位；`true` 必须对应
非空且可解析的 URI。以上结构校验发生在 session 消费前；随后单 location RMW 还会把新 URI 与 StartWrite
创建的旧 URI 做 safe-replacement 校验：

- 非 `pace` scheme 只接受 canonical URI 完全相同，不能借 FinishWrite 改写 path/key 或 query；
- `pace` 只允许 allocation 输出的 path/offset、`node_id` 和 `range_id` 改变；protocol、userinfo、host、port、
  path 是否为空、`size`、`media_type` 和其他 query parameter 必须保持，且 offset/size/node/media/range 必须满足
  当前无符号数值范围；不得注入新参数；
- 可解析但不满足上述策略的 URI 在 session 消费后的 RMW 中返回 mismatch，不能用同一 session 修正重试。

同一请求里**实际携带的** URI、checksum 和 location status 在一次 location RMW 中更新；跨 key batch 不是
存储事务。字段省略仍是 patch 语义：只提交 URI 会保留该 spec 已有 checksum，因此原子 RMW 只保证没有
中间态，不保证旧 checksum 仍与新 payload 匹配。调用方覆盖 payload 时必须同时提交对应的新 checksum，或明确
运行在无 checksum baseline 的模式，不能把“省略字段会保留旧值”描述成自动一致性。

部分 TP/spec 在 Put 后失败时，已成功后端返回的 actual URI 仍是必须回收的资源身份。RTP 先按
successful-response mask 收集已知 URI，再组成“全 session 长度+稀疏 presence”的 URI batch；收集后的异常路径调用
`FinishWriteWithIntegrity(success_mask=0, locations={}, uri_batches=known)`。Manager 因此能把 actual URI 与
`CLS_DELETING` 在同一 RMW 中绑定，避免只删除 StartWrite 的旧 URI 而泄漏后端新分配。如果没有任何已知
actual URI，则可使用 legacy 的全失败 Finish 消费 session。一旦 Finish 已发送或结果不确定，调用方不得盲目
第二次 Finish 同一 session，而应查询确认或隔离。若 v4 abort 在本地明确返回 `NOT_SENT`，才允许退回 legacy
zero-mask Finish 以消费 session；`SENT_OR_UNCERTAIN` 绝不重试。该 legacy 降级无法携带 dynamic URI，因而只能
释放 session，不能证明新分配对象已被完整回收。

上述回收能力只覆盖“成功子调用已经返回给 RTP”的 known actual URI。若单个 SDK Put 在报错/超时前完成了
部分动态分配，却没有把这些 URI 作为可用结果返回，TransferClient 的失败结果不会暴露它们，Manager cleanup
也无法凭 StartWrite 的旧 URI 推导真实 allocation identity；这类副作用必须由 backend 自身的原子性、lease 或
失败回滚机制收敛，是当前生产故障注入必须单独验证的边界。

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
- `KVCM_CHECK_IOV_BYTE_SIZE`：每个 iov 头部和尾部各参与 CRC 的最大字节数，默认 `4`，硬上限 `4096`；
- `KVCM_CHECKSUM_MAX_SAMPLED_BYTES_PER_REQUEST`：一次 calculator/Save/Load 显式校验允许的总采样字节数，
  默认 `256 MiB`。

cell 数、最大 iov 数、采样窗口或请求预算小于等于 `0` 时 TransferClient 初始化失败；`W > 4096`
也在任何 I/O 前拒绝。最大 iov 数超过 GPU kernel 的 `int` 索引上限或内存大小计算上限时同样拒绝，
避免形成永远无法取得 cell、缓冲区大小回绕或“请求成功但无法计算”的配置。每次显式计算在获取 cell
和发射 kernel 之前，以 overflow-safe 加法累计 `Σ 2×min(W,floor(S_i/2))`；超过当前进程的预算就返回
`ER_CHECKSUM_UNAVAILABLE`。默认 256 MiB 是资源保护值，不参与 checksum 结果；显式调大后必须重做容量和 P99 评审。

cell 数在构造阶段只记录、不分配，`Init` 完成范围校验后才创建 vector，防止异常环境值
在校验前触发无界宿主内存分配。CUDA 与 MUSA 都会对每个 cell 运行固定 CRC fixture warm-up；任一分配、
stream 创建或 warm-up 失败都会让 TransferClient 初始化失败。

`KVCM_CHECK_IOV_BYTE_SIZE` 由 kernel translation unit 在进程加载时只读取一次，必须在进程启动前设置。TransferClient
初始化还会将当时环境值与 kernel 已捕获的 resolved `W` 比较；如果动态库加载后又修改环境变量，初始化
直接失败，避免对外宣告的 contract 与实际 kernel 不一致。参与同一份 checksum 的 writer、reader 和外部调用方
必须使用完全相同的值。CPU-only build 即使看到 storage 开启了 `enable_meta_checksum` 也会告警后继续初始化，以便 opaque
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
可用于发现采样区域内的偶发 bit flip，以及会改变采样结果的错误 URI/block 串位；它不是密码学完整性或
安全认证机制。

当前算法还没有把 iov 长度作为独立字段纳入 checksum。这个限制不是只有理论上的 CRC 碰撞：当 `W=4` 时，
8-byte payload `abcdefgh` 的 CRC 输入是 `abcdefgh`；9-byte payload `abcdXefgh` 的 CRC 输入仍是
`abcdefgh`，两者会得到**确定相同**的 iov CRC，且 block 聚合值也相同。实际传输通常另有 size 校验，但这说明
现有值不能独立证明 layout/长度正确，也不能把“使用不同 iov shape 仍算出相同值”描述成概率仅为 `2^-32`。
严格链路必须另外核对 layout，或使用第 7.2 节把长度绑定进 checksum 的新 contract。

`HashIntArray` 按输入顺序递推，所以一般的 iov 置换会改变结果；但 checksum preimage/聚合值没有显式编码
iov index、layer/spec identity 或 size。若被置换项的采样 CRC 相同，或出现其他 CRC/hash 碰撞，checksum
仍不能证明身份。RTP 的 rank/tag/ordered-size/layout contract 只是外部结构约束，不是 checksum 内生身份绑定。

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
- `W` 没有脱离真实 iov 分布的通用正确值。当前默认 `4` 是已有
  `legacy-v0` checksum 的兼容性基线，不能在原算法 ID 下静默改动，也不能提供有意义的随机单点
  翻转覆盖率。RTP 生产接入必须显式配置并锁定 `W`；当前用于真机回归的生产参数候选是 `128`，但仍需在
  目标 GPU、模型和 block 布局上与 `64/256` 等配置同时测 coverage 与 P99 后才能审定；
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

### 7.2 RTP 当前 contract 与未来 strict 演进

RTP `record` / `verify` 当前使用兼容算法，但不再仅靠多进程“恰好设置了相同环境变量”。源码生成的
contract base descriptor 精确格式为：

```text
kvcm.remote_tp;protocol_version=3;algorithm=crc32_xor_int64.legacy-v0;sample_bytes=<resolved W>;layout=<16 lowercase hex>
```

每个 TP request 在 base 后精确追加 `;target_tp_rank=<decimal rank>`。layout 的 16 位十六进制值是对有序
group id/name/spec tag/block size、layer 顺序，以及每层 KV stride 和可选 scale stride 构造出的 descriptor 做
稳定 FNV-1a 64-bit hash；
因此 iov 数量、顺序或任一 size 改变都会改变 contract。base descriptor 被加入 RTP 生成的 KVCM
`instance_id` 身份，每个 TP request 再追加 `target_tp_rank`。worker 必须精确匹配整个字符串，并在执行前
核对实际 `BlockBuffers` 的 iov count/顺序/size。RTP 还逐 transfer pool 读取 KVCM
`GetChecksumCapability()`，核对 `available` / algorithm / resolved `W`。非 off 且使用自定义
`KVCM_CLIENT_CONFIG` 时，RTP 无法证明 instance identity 含有该 contract，因此拒绝初始化。

这些是已实现的 `record` / `verify` contract，当前只适用 TP=1。它们防止了 `W` 和 layout 漂移被误报为
payload mismatch，但没有改变 legacy checksum 值本身不绑定 size 的事实；Meta 也仍只存 value/presence，不存
descriptor。所以该 contract 是部署和路由安全约束，不能被扩大解释为全新 checksum 算法。

已有 `CA_CRC32_XOR_INT64` 的位级结果必须保持稳定：一旦 checksum 被持久化，直接修正奇数长度规则、把默认
窗口从 4 改成 128，或把 size 混入聚合，都会让相同 payload 的新旧结果不兼容。因此这些修正不能偷偷放在
原枚举名下；兼容算法继续按本节前半部分解释，并在 contract 中标为
`algorithm=crc32_xor_int64.legacy-v0;sample_bytes=<resolved W>`。`kvcm.` 前缀和 `W=` 别名都不是当前
wire contract 的一部分，调用方不得自行改写字段名。

`rtp_strict` 的未来设计可新增独立算法/枚举 `sampled_crc32_layout_v1`。以下定义是保留的演进方向，
不是当前代码已经支持的算法；后续实现若改变任何一项
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
  size、异构 iov、顺序交换和上述 `abcdefgh`/`abcdXefgh` 反例。未来 strict 的 RTP calculator 必须是独立
  实现，并与 KVCM 在 CI 中共用 golden vectors；当前 `record`/`verify` 调用的是 KVCM calculator，不满足此项。

当前 RTP 已落地自己生成的完整 legacy descriptor：TP worker 对整个字符串精确匹配；初始化时 RTP 另逐 pool
通过 KVCM `GetChecksumCapability()` 核对 algorithm/resolved `W`。KVCM 当前并不返回 RTP layout descriptor，
Meta 也不保存它，因此这里是“两层检查”，不是 KVCM 对完整 descriptor 的单一协商接口。当前还没有独立
calculator、storage commit 证据、sealed write lease 和多 TP allocation-time logical lease。因此 `rtp_strict`
仍必须拒绝启动；这个决定不会因 `sampled_crc32_layout_v1` 文档已存在而改变。

### 7.3 性能与覆盖率的最终选择

`W=128` 只是首轮压测候选，不是未经数据即可写死的生产默认。选择参数时至少输出每种模型/布局的 iov size
分布、每 block 的受保护字节数与 coverage P50/P95/P99、kernel/D2H/host aggregate 时间、pool wait 和端到端
P99。当前实现每个 iov 由一个 GPU thread 串行执行逐 bit CRC，并把每 iov 的 32-bit CRC 拷回 host 聚合；
因此开销不只来自读取的 `2W` 字节，iov descriptor H2D、CRC D2H、host 同步和大 batch 元数据也必须实测。
当前实现已给 `W` 设置 `[1,4096]` 硬范围，并在取 pool cell/kernel launch 之前用 overflow-safe 累加核对
单请求 `Σ 2×min(W,floor(S_i/2))`；默认预算是 256 MiB。`KVCM_SDK_MAX_CHECK_IOV_NUM` 仍只限制
KVCM calculator 的 chunk/scratch 容量；因为单个 block 不跨 chunk，它同时构成 checksum 单 block 的 iov 上限，
但不是 RTP RPC 的 request-wide 上限。RTP 对所有模式另设固定 `500000` 个 derived iov 的单请求硬上限，
并在 pin/展开前预测、展开后复核。`record` 的 read action 为 `NONE`，只受 derived-iov 上限，不消耗 sampled-byte
预算。这些上限是资源保护参数，不进入 checksum 值；
resolved `W` 才进入 RTP contract。需要全量扫描时应选择独立算法和独立限流队列，不能通过调大
sampled 模式的预算或窗口来绕过容量治理。

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

- FinishWrite 只给成功 block 中实际存在的对应 spec 写 checksum；对单个 location，同一请求实际携带的
  checksum、通过 v4/tag 7 提交且通过 safe-replacement 校验的 actual URI 和 `CLS_SERVING` 状态在同一
  metadata RMW 中发布。省略 checksum 时按 patch 语义保留旧值，不应表述为自动保证新 URI/payload 与旧值一致。
- 原始 failure-mask 位置始终进入 authoritative cleanup。若主发布的聚合调用、结果 shape 或任一 slot 失败，
  **本次所有 intended-success location**（包括可能已经发布成 `SERVING` 的 slot）也全部进入同一 cleanup，
  避免跨 key 非事务部分成功后留下可见子集。cleanup 把 `{WRITING,SERVING,DELETING}` 中当前实际状态转为
  `DELETING`，并在该 RMW 中持久已校验的 actual URI。它使用 authoritative persistent pre-read 和 post-write
  `Sync`，只有聚合调用与单槽均成功才能认定该 exact post-image 已被持久 fence。
- batch 不是跨 key 事务。主发布或 cleanup 部分成功时，Manager 最多执行 3 次全批 authoritative cleanup；仍有
  多个未确认槽时，再执行最多 32 次递归分治 salvage，逐槽确认能够独立建立 durable fence 的 post-image；
  仍无法建立 durable fence 的位置会让 FinishWrite 整体返回失败，记录终态指标，并要求调用方隔离/人工对账；
  文档不将这种不确定结果描述成原子回滚。
- 只有已确认的 `DELETING` exact serialized post-image 会被交给 `pre_fenced_deleting` 删除路径。调度接纳和最终 metadata
  CAD 都比较这份 exact value，不会在 fence 后跟随同 id 的新一代 location。物理 DELETE 失败时保留
  `CLS_DELETING` metadata；在 backend 没有 lease/conditional-delete 之前，禁止 blind automatic retry，避免 URI 复用后删掉新对象。
- 不带某个 spec checksum 的状态更新使用“保持旧值”语义，避免重试或兼容调用清掉已有基准。
- location spec 合并、查询选择与 COW 更新保留 checksum/presence。
- ReportEvent 的 BLOCK_ADD 与 BLOCK_SNAPSHOT 接受调用方 checksum，并在 URI 校验、版本化和
  snapshot 替换过程中原样保留；未携带 presence 的非零值仅作为早期客户端兼容输入。
- BLOCK_ADD 是 patch 语义：同名 spec 未携带 checksum 时保留旧基准。若 reporter 实际覆盖了该 payload，
  必须同时提交新 checksum；否则后续启用 checksum 的读校验会用旧基准报 mismatch。需要明确清除基准时应通过完整
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
| `FinishWriteCacheRequest` | `uri_batches` | 7 |
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
| 新 client → 老 server | 新请求 tag 6/7 被静默忽略，RPC/HTTP 仍可能成功 | **不能**把 FinishWrite 成功当作能力确认；checksum 提交由 query presence/value 确认，URI-only 提交由 exact spec/URI 确认 |

具体滚动行为：

- 老 client → 新 server：不携带 checksum batches，按历史路径执行；
- 既有 v3 `FinishWrite(..., FinishWriteOptions)` 为保持已发布行为，仍序列化 legacy tag 5，并在有 checksum 时携带
  tag 6。当前 server 不用 tag 5 做 checksum 对齐，也不用它更新 actual URI；
- client v4 新增独立的 `FinishWriteIntegrityOptions` 和追加在 vtable 末尾的
  `FinishWriteWithIntegrity`。该路径发送 tag 6 checksum batch 和 tag 7 URI batch，故意不序列化 tag 5；
  这使新 server 只有一个 actual-URI 事实来源，也避免把 legacy `Locations` 的非精确语义带入原子提交；
- `KVCM_STAGED_CHECKSUM_API_VERSION=4`，动态库导出 C 符号 `KVCMStagedChecksumRuntimeApiVersion()`。可能加载旧 DSO 的
  嵌入方必须先 weak-probe 该符号；缺失视为 version 0，禁止调用新 virtual slot。对老 subclass，v4 默认实现
  只能在 `uri_batches` 为空时回退到 v3 checksum path；带 URI 必须返回 `ER_CHECKSUM_UNAVAILABLE`，不得丢弃 URI 后假成功；
- 新 client → 老 server：gRPC 与 HTTP 都会忽略未知 tag 6/7，因此不会建立完整性基准，也不会持久
  actual URI。携带 checksum 的提交必须通过后续 query 的 presence/value 与 URI 全部精确确认；URI-only v4
  提交不应期待 `checksum_present=true`，而应比较完整 location/spec/URI。两种路径都不能仅以 FinishWrite
  RPC 成功作为依据；
- 新 client 读取老 metadata/server：checksum presence 为 false；
- MetaService 与 AdminService 只在 query 的 `include_checksums=true` 时携带已保存值；老 client 不设置该字段，
  维持历史响应；StartWrite 响应始终不返回 checksum。

“历史行为兼容”只覆盖合法请求。新实现会在任何 Put 前拒绝 URI/buffer 为空或数量不等的畸形
`SaveKvCaches` 调用并返回 `ER_INVALID_PARAMS`；这比旧实现更严格，是防止 URI 与 block 错位写入的主动
收紧，不应依赖旧版本对畸形输入的未定义行为。

公共 C++ 接口的兼容范围需要精确区分：本次保持所有 v3 virtual 方法的声明顺序和
`FinishWriteOptions` / `TransferTraceInfo` 布局，v4 overload 只追加到末尾。下游 subclass **重新编译**时
保留源码兼容和既有 slot 序号；这不承诺新代码可以不经 runtime probe 就对旧 DSO/预编译 subclass 调用
新 slot。RTP non-off 路径以 runtime API v4 为强制能力，不做静默 ABI 降级。

### 9.1 发布顺序与回滚

必须采用 server-first：先把所有可能承接该 instance 的 Manager 升级完成，再升级/开启 writer，最后开启
reader payload 校验。混部期间不能依赖负载均衡恰好命中新 server。checksum rollout 以查询得到每个目标位置
`checksum_present=true` 且 value/URI 精确匹配作为确认；OFF 的 URI-only v4 路径则以完整 spec/URI 精确匹配
作为确认，不能要求并不存在的 checksum presence。推荐在 checksum presence 覆盖率达到 100% 前只用
`record` 建立/观察 baseline，达到后再切 RTP `verify`；这不表示可以开启当前会在 init 时拒绝的
`rtp_strict`。任何已经显式请求的 KVCM 校验本身仍保持 fail-closed。

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

本候选树已有 CUDA 单机 kernel 正确性和微基准证据，但真实业务的 pool 排队、存储读写故障注入、端到端 P99，
以及 MUSA 对等覆盖仍须在目标硬件单独完成；在第 12 节门禁完成前，本节不能作为允许生产开启的依据。

### 10.2 2026-09-20 CUDA 采样微基准

当前候选树在 NVIDIA L20 上编译并运行了真实 CUDA kernel 微基准：64 blocks、每 block 8 iov、
每 iov 1 MiB，逻辑 batch 总量 512 MiB，100 次采样迭代（full 参考为 3 次）。结果如下：

| 模式 | 实际采样字节 | avg | P95 | 相对 `W=4` |
|---|---:|---:|---:|---:|
| 头尾 `W=4` | 4 KiB | 24.112 µs | 25.272 µs | 1.000× |
| 头尾 `W=64` | 64 KiB | 48.748 µs | 49.864 µs | 2.022× |
| 头尾 `W=128` | 128 KiB | 73.450 µs | 74.376 µs | 3.046× |
| full-iov 参考 | 512 MiB | 213427.343 µs | 213501.308 µs | 8851.378× |

该数据证明了在这个 layout 上头尾采样与全量 CRC 存在数个数量级的成本差异，但不是业务端到端 SLA。
RTP `record` 的写路径对同一 batch 调用两次 calculator，`verify` 还在读路径调用两次；上表不包含
pool 竞争、broadcast、storage I/O、Meta round-trip 或双调用累计成本。所以它支持“采样方案值得继续验证”，
不支持“已可在生产开启”。

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
| v4 API 不破坏 v3 ABI；runtime DSO 能力可探测 | 独立 `FinishWriteIntegrityOptions`；`KVCMStagedChecksumRuntimeApiVersion` | `TransferClientApiTest`、`GrpcStubTest`、RPM header/symbol/runtime probe |
| actual URI/checksum/status 单 location patch RMW；actual URI 不越权重定向 | FinishWrite tag 6/7；`MetaSearcher::LocationUpdateTask`；`IsSafeFinishUriReplacement` | `CacheManagerTest`、`MetaSearcherTest`、`GrpcStubTest` 的 URI/checksum 原子更新、字段省略保留和 safe-replacement 用例 |
| 部分发布后 authoritative fence 与 exact delete | `CacheManager::FinishWriteCache`；`SchedulePlanExecutor` | `CacheManagerTest`、`SchedulePlanExecutorTest`的 partial-RMW/sync/exact-value/不盲重试用例 |
| 删除终态失败可观测且 label 有界 | `metrics/delete_cleanup_observer.*` | `DeleteCleanupObserverTest`；`KMonitorMetricsReporterTest` |
| JSON/COW/合并/迁移不丢 checksum | `CacheLocation`、`MetaSearcher`、`MigrationManager` | `CacheLocationTest`、`MetaSearcherTest`、`MigrationManagerTest` |
| ReportEvent ADD patch、SNAPSHOT replace，快速 JSON parser 不丢字段 | event 校验与 delta/snapshot 合并；`ReportEventJsonParser` | `CacheManagerTest.TestReportEventPreservesCallerProvidedChecksum`、`ProtoMessageJsonUtilTest.TestReportEventFastJsonParserMatchesGenericParser` |
| 默认查询不返回；Meta/Admin 显式 opt-in | service handlers、`CacheLocationViewToProto` | `ManagerMessageProtoUtilTest`、`AdminServiceImplTest.TestGetCacheMetaChecksumIsOptInAndZeroRemainsPresent`、集成测试 |
| 配置非法值拒绝，update 保留旧 integrity | `StorageConfig`、`RegistryManager`、`kvcm_ops` | `StorageConfigTest`、`RegistryManagerLocalBackendTest`、`storage_util_test` |
| 算法位级稳定且避免 signed-shift UB | `HashUtil::HashIntArray`、CUDA/MUSA kernel | `HashUtilTest`、`SdkBufferCheckUtilTest`（GPU） |
| pool 边界、warm-up、多设备资源生命周期 | `SdkBufferCheckPool` CUDA/MUSA 实现 | `SdkBufferCheckUtilTest`；目标硬件门禁 |
| 协议字段只追加、老调用默认不变 | 三份 proto；tag 5/6/7 分工；旧 virtual slot 保序和 fallback | `GrpcStubTest`字段号/序列化用例、公共 C++17 源码兼容编译探针、全量 UT |
| CPU/协议开销可量化 | `data_integrity_benchmark` | `//kv_cache_manager/service/util/test:data_integrity_benchmark` |
| 头尾采样与 full 成本对照 | `checksum_sampling_benchmark` | CUDA L20 真机微基准（第 10.2 节） |

### 11.1 本次验证记录

以下是 2026-09-21 当前候选树的证据边界。“通过”只表示表中明确列出的配置/硬件/制品；
它不会被外推成 RTP 端到端或生产 SLA。候选代码再变化后必须重跑对应项，跳过/incompatible 不算通过。
KVCM 结果来自已整合 `origin/main@6779a18e` 的单提交候选代码，不再是 rebase 前证据；
本次证据同步和 packaging/dependency 规则调整没有改变下表受测的运行时实现；制品规则本身的验证边界另列在表中。

| 验证项 | 已有证据 | 边界 |
|---|---|---|
| KVCM 外源 ordinary 全量 `//kv_cache_manager/...` | 127 个可运行 target 通过，1 个非 CUDA invocation 中的 `SdkBufferCheckUtilTest` 跳过 | 覆盖共享 CPU/协议/Meta/cleanup 路径；skip 不算 GPU 证据 |
| KVCM 外源 ASAN 全量 | 127 个可运行 target 通过，同一 GPU-only target 跳过 | ASAN 未报问题；不替代 CUDA/MUSA 设备运行 |
| KVCM 外源受影响 target | 61 个可运行 target 通过，1 个 GPU-only target 在非 CUDA invocation 中跳过 | 聚焦验证 client/manager/meta/metrics/cleanup；skip 不算 GPU 证据 |
| KVCM 内源 ordinary 全量（明确排除 2 个 VCNS target） | 130 个可运行 target 通过，1 个非 CUDA invocation 的 GPU-only target 跳过；PACE/Tair target 通过 | 验证共享改动在内源依赖图下可用；VCNS 已不维护且不是本功能门禁 |
| KVCM 内源 ASAN（同样排除 VCNS） | 130 个可运行 target 通过，同一 GPU-only target 跳过 | ASAN 未报问题；不把 VCNS 环境状态混入本功能结论 |
| KVCM 内源受影响 target | manager/metrics + TairMempool + PACE CopyGA/Response 共 29/29 通过 | 覆盖内源 dynamic URI 相关依赖图；不是 RTP E2E |
| KVCM `--config=client` 全量 | 127 个可运行 target 通过，同一 GPU-only target 跳过 | 验证 client 构建配置；非 CUDA invocation 不提供 kernel 证据 |
| KVCM 真实 Manager 集成 | ordinary HTTP+gRPC 2/2 通过；ASAN HTTP+gRPC 2/2 通过 | 启动真实 Manager 进程验证 API；不含 RTP、真实 storage 或 GPU payload 闭环 |
| KVCM CUDA 正确性 | `SdkBufferCheckUtilTest`、`TransferClientTest` 2/2 通过，`W=128`，真实 GPU | 已执行真 kernel；尚不是 RTP/storage 故障注入闭环 |
| KVCM CUDA 性能 | NVIDIA L20 在 post-rebase 候选上完成 `W=4/64/128` 与 full 参考，见第 10.2 节 | 单机微基准，不包含业务 I/O/broadcast/Meta |
| KVCM v4 本地 packaging 探针 | 去掉 `client_only_header` 后，自包含 CUDA 12 RPM 的精确包内 DSO 通过 v4 header/symbol、ELF/build-id、`ldd -r`、`RTLD_NOW` 和 runtime API=4；L20 上从该 DSO 初始化成功并返回 `available=true`、`legacy-v0`、`W=128`。server tar/RPM 的提交戳、必需文件和 x86_64 ELF 门禁也通过 | 这是 post-rebase 候选工作树的本地制品证据；尚不是由精确最终提交发布、可下载并被 RTP 官方 pin 的制品 |
| KVCM 正式 CUDA RPM/server 发布 | pending，本文不记为通过 | 仍缺 6-job 正式流水线、不可变 URL/SHA、目标 GPU 对正式下载制品的重复探针及 RTP 官方 pin |
| RTP 最终 actual-URI 返回预算 patch phase-1 | 2/2 通过，禁用 test cache 后真实执行 | 使用本地 KVCM v4 override；覆盖 independent-pool 和 mock-only full 直接变更面 |
| RTP focused ordinary | 14/14 通过，禁用 test cache 后真实执行 | 使用本地 KVCM v4 override；包含 block pool、KVCM backend/wrapper/observability、transfer、P2P/broadcast 相关 target |
| RTP Python/config/server smoke | 4/4 通过，禁用 test cache 后真实执行 | 使用本地 KVCM v4 override；覆盖 config pickle/server setup/server args/remote KVCM server smoke，不是真实 storage E2E |
| 候选变更静态检查 | KVCM post-rebase 实现已做 format/diff 检查 | 仍需在最终 packaging/pin 变更后重做并通过 PR/MR policy |

明确**尚未完成**的证据：

- RTP 当前官方内外源 pin 仍是 2026-04-29 的 KVCM client/server 制品，不含 staged-checksum API v4。
  本地 override 只是开发输入，必须发布新制品并同时更新外源 `deps/http.bzl` 与内源
  `internal_source/deps/http.bzl` 的 URL/SHA 后，才具备可复现构建证据；
- RTP 最新 actual-URI 返回预算 patch 的 phase-1 2/2、focused ordinary 14/14 和 Python 4/4
  已按上表真实执行通过；但当前代码的最终 ASAN 仍没有可用的 green 证据，更早的日志不得代替；
- 尚未在真实 RTP `tp_size=1 + verify` 业务流程中完成从 KV 产出、写 KVCM、Finish/Meta reconcile、
  读 KVCM 到推理消费的端到端闭环；
- 尚未完成目标 storage 采样区精确篡改、Meta 篡改、部分 TP/backend 失败的真环境故障注入，也没有
  RTP off/record/verify 的业务级 P50/P95/P99 对比；
- MUSA 还没有与 CUDA 对等的 kernel 正确性、多设备和故障注入证据。

历史 PR/head 的 green 结果只能作为回归参考，不能代替最终 squash/rebase 后新 head 的 CI/review 结果。

## 12. 生产开启清单与硬门禁

### 12.1 合入门禁

- [x] KVCM 外源 ordinary/ASAN 和内源 ordinary/ASAN（按仓库约定排除不维护 VCNS）已有第 11.1 节记录；
- [x] KVCM `--config=client` 全量及真实 Manager HTTP/gRPC ordinary/ASAN 集成已有第 11.1 节记录；
- [x] KVCM CUDA 真 kernel 正确性、`W=128` TransferClient 回归和 L20 采样微基准已有证据；
- [x] proto tag 5/6/7 未复用，v3 ABI 对象布局/旧 virtual slot 保持，v4 CPU/debug runtime probe 与其本地 RPM 自洽；
- [x] 自包含本地 CUDA RPM 的精确包内 DSO 已完成依赖解析、runtime API 和真实 GPU capability 运行探针；
- [ ] 对正式发布并准备交给 RTP 的精确 CUDA RPM 重复上述探针，记录不可变 URL/SHA，并以官方 pin 无 override 构建；
- [x] KVCM 已对整合 `origin/main@6779a18e` 的 post-rebase 实现重跑外源/内源、client、真实 Manager、CUDA 和微基准矩阵；
- [ ] 最终 packaging/pin 调整后的新 head 重做 diff/静态/制品检查，确认本文/API 文档与最终 diff 一致；
- [ ] 最终 PR/MR head 的 normal/ASAN/静态分析/CLA/review policy 满足仓库当时的合入规则；
- [x] RTP 最新 actual-URI 返回预算 patch 的 phase-1 2/2、focused ordinary 14/14 和 Python 4/4 均以禁用 test cache 的方式真实执行通过；
- [ ] RTP 最终 ASAN、官方 pin 下的无 override 重复构建，以及内外源依赖图验证全部结束。

不得用旧 PR/head 的 green 状态勾选任何仍未完成项。KVCM 与 RTP 是两个仓库；KVCM 自身回归通过不等于
RTP 接入已可合入。

### 12.2 生产启用门禁

- [ ] 发布包含 API v4 且运行时 capability 可用的 KVCM CUDA client RPM/server tar，更新 RTP 内外源官方 pin 的
  URL/SHA，在无本地 override 时可重复构建；不能把仅有 v4 ABI、capability 不可用的 CPU/debug override 当作完成；
- [ ] 以 `tp_size=1 + verify` 作为当前唯一真实目标，完成 KV producer → `C0/C1` → Put → v4 Finish → exact
  Meta reconcile → Get → `C2/C3` → 推理消费的端到端测试；
- [ ] 在目标 storage 做可重复故障注入：写前源 buffer、Meta checksum/URI、已采样的 storage 头/尾字节、
  未采样中段（应记录为已知盲区）、部分 backend 成功后其他项失败，以及**单次 SDK Put 报错但已产生未返回
  dynamic URI/部分写入**，并断言 mismatch 数据不发布给推理、未知副作用由 backend 契约或对账闭环收敛；
- [ ] 验证 RTP transfer-status v0/v1 滚动兼容、OFF multi-pool 部分失败的 gRPC OK + FAILED + 稀疏 URI、
  v4 actual-URI abort、Manager authoritative pre/post-sync cleanup、exact pre-fenced delete、物理删除失败时
  不 blind retry，以及四个固定 `stage` 的零基线/增量指标；
- [ ] 用生产 block/iov/group 分布量化 coverage，明确批准头尾采样的盲区；按 `W=4/64/128/...`
  比较 off/record/verify 的 kernel、pool wait、broadcast、storage I/O 和端到端 P50/P95/P99，并审定
  `W∈[1,4096]`、iov 上限和默认 256 MiB 请求预算；
- [ ] 在计划上线的 CUDA 发布镜像/设备上补齐多线程、pool 饱和、多 device、跨 device 拒绝和资源释放；
  如果计划启用 MUSA，必须先补齐与 CUDA 对等的 kernel 用例和证据；
- [ ] 使用生产峰值 location/spec 数和真实 ABI 测量 Manager RSS/COW 峰值，接受 checksum opt-out 时仍存在的
  `LocationSpec` 固定内存增量；
- [ ] server-first 升级所有 Manager，用 record canary 建立 baseline 并验证 query presence/exact URI，覆盖率达标后
  才切 verify；演练 verify → record → off → client/server 的反向回滚；
- [ ] 对 RTP integrity 低基数 event/status/stage 指标、KVCM `ChecksumValidationLog`、Meta reconcile/quarantine 以及
  `cache_cleanup.permanent_failure_location_count{stage=...}` 配置丢失率、延迟和告警 SLA；补齐或另行覆盖 OFF
  dynamic-URI reconcile/quarantine（当前 integrity reporter 在 OFF 下不发指标）；URI/key/trace/block id 仅进受控
  结构化日志，不作为 Prometheus label。

`rtp_strict` 不是当前可通过灰度打开的模式，而是在 init 时硬拒绝。未来要解除拒绝，至少还需要独立
RTP calculator、sealed producer/write lease、backend committed-byte 证据或等价机制，以及多 TP allocation-time distributed
logical lease。在这些能力完成前，任何配置都不得把现有 `record`/`verify` 重命名或宣传为 strict。

只要本节仍有未完成项，RTP 生产默认必须保持 `off`。调用方 opaque checksum 的保存/查询可以独立灰度，
但不得宣称 RTP 端到端校验已达到生产准入。

## 13. RTP-LLM 当前接入实现与边界

本节对照的是独立工作区 `gitlab-rtpllm2/RTP-LLM/github-opensource` 中
`codex/kvcm-data-integrity-production` 的候选工作树，包括尚待 amend 的最终 actual-URI 返回预算补丁。预期合入形态是相对
`origin/main@6aebdf0a65` 的单一 squash commit，但尚未合入 RTP `main`。因此本节描述“当前候选代码行为”，
不代表已发布、已合入或已通过生产门禁。

### 13.1 运行模式与真实可用范围

| `kvcm_checksum_mode` | 写路径 | 读路径 | 当前状态 |
|---|---|---|---|
| `off` | legacy Save；URI 未变化时 legacy Finish，dynamic actual URI 时 URI-only v4 Finish + exact URI 回读 | legacy Match/Load | 默认；保持原有 TP 支持，另有 transfer-status v1 资源回收扩展 |
| `record` | `C0` 计算 + KVCM `C1` 写前校验，v4 Finish 保存并立即 Meta reconcile | 不校验 payload checksum；仍执行 v3 结构/token 边界 | 仅 `tp_size=1` 允许 init |
| `verify` | 与 `record` 相同 | query presence；KVCM `C2` + RTP 边界 `C3`；全部通过才发布 block | 仅 `tp_size=1` 允许 init；当前唯一真实 E2E 目标 |
| `rtp_strict` | 不执行 | 不执行 | init 无条件拒绝 |

Python server args、C++ `KVCacheConfig`、pybind/stub 和配置打印已携带 mode 与
`kvcm_checksum_sample_bytes`。非 off 初始化会同时检查：

- `1 <= W <= 4096`，且 RTP 配置的 `W` 等于进程启动前的 `KVCM_CHECK_IOV_BYTE_SIZE`；
- 每个 transfer pool 的 KVCM capability 为 available，algorithm 是
  `crc32_xor_int64.legacy-v0`，resolved `W` 一致；
- `KVCM_SDK_MAX_CHECK_IOV_NUM` 和 `KVCM_CHECKSUM_MAX_SAMPLED_BYTES_PER_REQUEST` 为正数，前者是
  calculator chunk/单 block 上限、后者默认 256 MiB；RTP request-wide derived iov 另有固定 `500000` 上限；
- client DSO 通过 weak C symbol 探测到 staged-checksum API 至少为 v4；旧 DSO 缺少符号时按 version 0 处理，
  任何新 virtual 调用 fail-closed；
- 由 RTP 掌控 KVCM instance identity；非 off 不接受无法证明 contract 身份的自定义
  `KVCM_CLIENT_CONFIG`。

### 13.2 `C0` 到 `C3`：两次写计算和两次读计算

worker 对同一批 `GroupPolicy::genBlockBuffersByTag` 输出执行：

```text
record / verify write
  RTP boundary: TransferClient::CalculateChecksums(source)       -> C0
  KVCM Save:    VerifyCallerChecksums(C0), recompute before Put  -> C1
  require C0 == C1; Put success returns C0-shaped result
  FinishWriteWithIntegrity(checksum=C0, actual_uri=...)
  immediate exact Meta query: URI/spec/checksum must equal write session

verify read
  Meta baseline + presence
  KVCM Load: Get then VerifyWith(baseline)                        -> C2
  RTP boundary: TransferClient::CalculateChecksums(destination)   -> C3
  require C2 comparison and C3 comparison both pass
```

`record` 每次写计算两次；`verify` 每次写两次、每次读两次。`C0` 与 `C3` 是通过
KVCM client v4 公开的 `CalculateChecksums` 发起，`C1` 与 `C2` 由 Save/Load 内部调用；四者都使用同一
KVCM calculator/kernel。因此它能检测两个调用边界之间的变化，但不能检测该共享实现中的系统性
缺陷，不能被称为独立 RTP verifier/信任根。

legacy-v0 先对每个 iov 的头尾采样计算 CRC32，再按 iov 顺序递推
`HashUtil::HashIntArray`。这个聚合对一般置换是顺序敏感的，但 checksum 值没有显式编码 iov identity 或
size；当被置换 iov 的采样 CRC 相同，或发生其他碰撞/采样盲区时，该值不能单独证明 layer/iov 身份。
RTP v3 的 rank/tag/ordered-size/layout contract 是额外的结构约束，不是 checksum 内生的身份绑定，更不是密码学认证。

### 13.3 TP 协议 v3 和 block 身份

RTP `RemoteOperationRequestPB` 在保留既有 tag 的前提下追加了 checksum action/value/presence/contract，并使用：

| request 字段 | tag | 语义 |
|---|---:|---|
| `block_generations` | 11 | 与 item 对齐的 block allocation generation |
| `target_tp_rank` | 12 | 显式目标 rank，用 wrapper 区分“缺失”和合法 rank 0 |
| `block_pool_epochs` | 13 | 与 generation 组成原子 token 的 pool incarnation |

与 block 身份/token 相关的 protocol v3 加法字段只有上表三个 request 字段；response 不返回 allocation token。
这是有意的安全约束：transfer-time 当前 token 查询只能证明“现在这个 block id 属于某对象”，不能证明它仍是
请求创建时的对象。

另有一组与身份 token 正交的传输结果协商字段：

| message | 字段 | tag | 语义 |
|---|---|---:|---|
| request | `transfer_status_version` | 14 | `0` 保持 legacy transport；`1` 请求显式业务结果 |
| response | `transfer_status` (`RemoteTransferStatusPB`) | 9 | `UNSPECIFIED` / `OK` / `FAILED` |

当前 coordinator 对新请求发送 version 1。worker 成功时返回 `OK`；失败时仍以 gRPC OK 返回业务
`FAILED`。对 WRITE，失败 response 的 `actual_uris` 保持与 request item 等长，未产生 URI 的位置用空字符串，
从而允许前一个物理 pool 已动态分配 URI、后一个 pool 失败时，coordinator 仍收集稀疏 URI 并走 v4 abort/cleanup。
这项机制在 `off` 模式同样启用，不代表 checksum 成功；coordinator 必须把 `FAILED` 判为整个 transfer 失败。

`transfer_status_version=0` 仅表示通用 transfer status 未协商：`off`/generic transfer 失败继续返回 legacy
gRPC error，不能以 gRPC OK 携带只有新字段才能解释的业务失败；已经通过 non-`NONE` action 或 contract
协商 checksum 的请求，仍可用 gRPC OK + `checksum_status` 返回结构化 checksum 失败，同时
`transfer_status=UNSPECIFIED`。滚动期间，新 coordinator 在 `off` 模式允许旧 worker 的成功 response 保持
`UNSPECIFIED`；non-off 仍要求完整 checksum capability/status，不能借此降级。未知
`transfer_status_version > 1` 被拒绝。version 0 无法在 gRPC error 中回收部分失败前已产生的动态 URI，只有
version 1 双端路径闭合了这一资源回收窗口。

当前只有 TP=1 安全路径：`StorageBackend` 外层已经 pin 住本地 handle，coordinator 从同一进程/同一 pool
读取当前 `{pool_epoch,generation}` 并填入 request；worker 在把 block id 解析成 GPU iov 前，在 pool 锁内原子校验
epoch+generation 并增加 `LOAD` pin。pool epoch 在每个 pool incarnation 中固定且非零，generation 递增；如果 generation
将溢出，在修改 free-list 前拒绝回收。

TP>1 的 coordinator 不能从 rank 0 pool 推断其他 worker 的 token，也不能用 transfer-time 当前 token 查询补救。
在分布式 allocator 把 allocation-time logical lease 与 handle 一起传到 coordinator 前，任何 non-off
`tp_size>1` 都在 init 时拒绝。

non-off request 必须原子携带 target rank、contract、完整 epoch/generation vector；缺任意一项都是 contract/request
失败。`off` 的 checksum/身份部分只接受 legacy envelope（`NONE`，且无 contract/token）；tag 14 的 transfer-status
协商与此正交，可以为 version 1。发送一半 checksum/身份 v3 字段的请求仍会被拒绝。worker 本地 action 矩阵为：

- `off`：仅 `NONE`；
- `record`：WRITE 必须 `SAVE_VERIFY_SOURCE`，READ 允许 `NONE` 或 `LOAD_VERIFY`；
- `verify`：WRITE 必须 `SAVE_VERIFY_SOURCE`，READ 必须 `LOAD_VERIFY`；
- `rtp_strict` / 非法模式：不接受任何 action。

contract 字符串包含 protocol v3、legacy-v0 algorithm、resolved `W`、有序 group/tag/layer/每层 iov size 布局的
稳定 hash，以及 target rank。base contract 还进入 RTP 生成的 KVCM instance identity。worker 在 I/O 前精确匹配
contract，并用实际 `BlockBuffers` 再核对 iov count/顺序/size。

### 13.4 写路径、v4 Finish 和不确定结果收敛

rank 0 从 StartWrite 返回的 compact session 建立
`{session_index,spec_index,spec_name,tp_rank,group_tag,block_id}` 映射，严格校验每个 location 的 spec 集合等于对应
`location_spec_group` 配置，且没有重复。worker 按物理 pool 分组时，URI、buffer、block id、token 和 checksum
作为不可拆分 tuple gather/scatter；不从 TransferClient registration 名字推断逻辑 spec。

写成功后，rank 0 把 worker checksum 恢复为每 spec 的全 compact-session batch，并把已知 actual URI 恢复为等长
稀疏 URI/presence batch。`record`/`verify` 通过 API v4 `FinishWriteWithIntegrity` 在一次请求中提交 success mask、
checksum 和 actual URI；`off` 在 URI 未变化时保持 legacy Finish，出现 dynamic actual URI 时也必须改走 URI-only
v4/tag 7，并在允许复用前 query 校验 exact URI。v4/tag 7 是 dynamic URI 的唯一有效提交路径；legacy tag 5
不参与新 server 的对齐或提交。v4 无法发送，或新 client 对老 server 发 tag 7 后 exact URI 未落库，均不得按成功发布。

部分 rank/pool 失败时，coordinator 仍从 broadcast successful-response mask 中提取所有已知 actual URI，不因后续
checksum/shape/transport 异常而丢弃。对 negotiated v1 业务失败，worker 以 gRPC OK +
`REMOTE_TRANSFER_STATUS_FAILED` 返回等长稀疏 URI，因此 response 会进入 successful mask，但 transfer 整体仍失败。
若 Finish 尚未发送且已知 URI 非空，abort 使用 v4、全失败 mask、空
legacy locations 和稀疏 full-session URI batches；没有已知 URI 时才可使用 legacy 全失败 Finish。一旦请求可能
已发送，就不对同 session 盲目再 Finish。只有 v4 明确 `NOT_SENT` 时才退回 legacy zero-mask abort；该降级
可消费 session，但不能携带 dynamic URI，也不能声称实际分配已完整回收。

checksum Finish 和 OFF URI-only Finish 都在首次非幂等请求前，对本进程的 committed keys 建立 provisional deny。
wrapper 区分 `NOT_SENT` 与 `SENT_OR_UNCERTAIN`；只要请求可能已发送，不论 RPC 返回 OK、非 OK 或抛异常，
都立即 query。`record`/`verify` 必须同时确认 key、location/spec 集合、exact URI、checksum presence/value；
OFF URI-only 路径只确认 key、location/spec 集合和 exact URI，不要求并未提交的 checksum presence：

- exact metadata 已观察到：即使 Finish RPC 回包丢失/非 OK，也可按“可观测已提交”处理并解除 provisional deny；
- 无法确认 exact metadata：永久本地 deny 这批 key，请求 `RemoveCache` 并上报 reconcile/quarantine 事件；
- 本地 deny 内存分配自身失败：把整个 remote tier 切成本进程 fail-closed，不继续当作安全命中。

Manager 侧再按第 8 节完成 checksum+URI+status 单 location RMW、authoritative cleanup、exact pre-fenced delete
和不 blind retry。这个组合用来收敛部分发布和丢回包，但不会把跨 key batch 包装成不存在的原子事务。

### 13.5 读路径与推理发布边界

`verify` 的 Match 显式请求 checksums，先校验每个 spec 的 value/presence vector 与 location 数目对齐、spec 不重复，
再与 `GroupPolicy` 选中的 spec 集合对齐。从第一个缺少必需 checksum 或本地 deny 的 block 起截断整个远端前缀；
不跳过中间缺口后继使用后缀。裁剪 locations 时同步裁剪每个 checksum/presence vector。因此老 metadata/server
或灰度期未建立 baseline 的位置是安全 miss，不是无校验 hit。

worker 在 Get 前先校验 request shape、contract、target rank、epoch/generation 和实际 iov layout；调用
`LoadKvCachesOptions::VerifyWith` 时每个已分发 item 都必须 presence=true。KVCM `C2` 校验失败或随后 `C3`
比较失败都使整个远端 transfer 失败；已写入目标 GPU buffer 被视为不可信，不发布到 BlockTree/推理。
普通 broadcast 失败不会遮蔽另一 rank 已确认的 storage-output mismatch；后者保持为聚合失败原因并触发
storage quarantine。PREFILL 沿用现有语义：丢弃远端结果并本地重算；非 PREFILL 沿用向上返回失败。

### 13.6 输入预算、日志与指标

为避免攻击性/损坏 RPC 在 pin GPU block 或展开 buffer 后才被发现，worker 分阶段检查：

- 复制 repeated fields 前限制 item/vector 最多 `2^20`、单 group tag 最多 1024 bytes、tag 总量最多 4 MiB，
  trace/contract 各最多 4096 bytes、单 URI 最多 16 KiB、URI 总量最多 48 MiB；
- 复制 tags/blocks/URI 后、pin/展开前，按 topology 用 overflow-safe 乘加预测所有模式的 request-wide derived iov，
  固定不得超过 `500000`；只有 checksum action 非 `NONE` 时才同时预测 sampled bytes，默认预算 256 MiB；
- 展开后再次核对实际总 iov 不超过 `500000`；checksum action 还要求每个 iov **声明为** `MemoryType::GPU`、
  非 ignore、非 null、size 至少 2 且等于 contract 的 ordered expected size，并要求单 block iov 不超过
  `KVCM_SDK_MAX_CHECK_IOV_NUM`。该环境变量控制 KVCM chunk/scratch，不是 request-wide RPC 上限。当前这层
  precheck 不调用 CUDA pointer attribute，也不显式验证物理 device ownership/accessibility；“指针属于当前
  worker 可访问 device”仍是部署假设，foreign/inaccessible pointer 只会在后续 KVCM runtime/kernel 路径失败并
  fail-closed。补齐显式 device probe 和跨 device 拒绝测试仍是生产门禁。

Local RPC 日志只记录操作摘要，不打印包含 URI 的整个 protobuf request。broadcast/P2P 结果按
successful-response mask 过滤，结果 shape 不对齐则失败；多个 callback 逐个做异常隔离，一个用户 callback 抛异常
不会阻止其他 callback 被通知。

RTP 当前只新增两个 KMonitor 指标：

- `rtp_llm_kvcm_integrity_event_qps`；
- `rtp_llm_kvcm_integrity_affected_block_qps`。

它们使用低基数 `event/status/stage/mode` tag。当前 reporter 在 `checksum_mode=off` 时直接返回，因此这两个指标
只覆盖 `record`/`verify` 的 capability、operation failure、Meta reconcile、fail-closed 和 quarantine 等事件；
OFF dynamic-URI reconcile/quarantine 仍会执行本地 deny、`RemoveCache` 并写日志，但没有对应 integrity QPS。
这一缺口必须在生产启用前由新增指标或等价告警补齐。当前也**没有** checksum compute/verify latency、pool wait 或明确的
write-abort outcome 指标，所以已有 QPS 只是基础故障可观测，不是完整性能/发布门禁。KVCM Manager 内部指标
`cache_cleanup.permanent_failure_location_count{stage}` 导出为 Prometheus
`kvcm_cache_cleanup_permanent_failure_location_count{stage}`；四个固定 stage 为 `physical_delete`、
`metadata_cad`、`authoritative_fence`、`dispatch_or_worker`，均在启动时预置零基线，用于观测终态 cleanup 失败，
不代表当前积压 gauge。

### 13.7 已覆盖的代码面与尚未证明的系统面

当前 RTP 候选树已包含针对以下行为的 UT/mock/smoke 用例或 fixture：配置序列化，v4 wrapper 能力与错误码，
FULL/LINEAR、shared/independent pool、spec/compact-mask 对齐，0/presence 分离，contract/action/shape/token 拒绝，
generation/pool-epoch ABA，actual URI/checksum 同时回填，部分成功 URI abort，Meta exact reconcile/quarantine，prefix
truncation，`C0/C1/C2/C3` 调用及 mismatch，callback/P2P 异常边界，以及 integrity metrics。

上述是“候选代码中存在相应测试面”，不是“当前所有 target 已全绿”的替代说法。截至本文快照，
包含最终 actual-URI 返回预算补丁的 phase-1 2/2、focused ordinary 14/14 和 Python/config/server
smoke 4/4 已在 KVCM v4 本地 override 下禁用 test cache 真实执行通过。该结果只证明当前源码与
开发 override 的组合，不能被写成官方制品或无 override 可重复构建。最终 ASAN、官方 RPM pin、
真实 TP=1 verify E2E、生产 storage/Meta/部分成功故障注入、端到端性能和 MUSA 证据仍为未完成门禁。

### 13.8 为什么 `rtp_strict` 仍必须拒绝

当前 `record`/`verify` 已能在多个时间边界重算同一采样 checksum，但 strict 责任归因还缺四个必要证据：

1. 独立的 RTP calculator/reference，否则 `C0--C3` 会共享同一 KVCM 实现缺陷；
2. producer event happens-before 和从 `C0` 到 Put/commit 完成的 sealed write lease，否则 checksum/Put 之间存在 TOCTOU；
3. 基于 backend committed target bytes 的 storage-commit 证据，或语义和成本均明确的 read-back；否则只能把
   Put/介质/Get 合并为一个责任区间；
4. TP>1 allocation-time distributed logical lease，否则无法证明远端 block id 在调用生命期中没有复用。

未来 strict 还必须定义 timeout 后 DMA/I/O 真正 quiesced 之前的 quarantine，以及是否使用第 7.2 节的新版本
layout-bound 算法。在这些能力落地并通过门禁前，init 硬拒绝是唯一允许的行为。

## 14. 当前限制与后续工作

- Python connector 和 `RTPLLMClient` 的通用 KVCM 调用仍保持 legacy/opt-out 默认；本次实现位于 RTP
  BlockTree KVCM backend，不应宣称 vLLM、SGLang、TRT-LLM 的所有其他接入自动获得同等校验。
- RTP 官方内外源 KVCM pin 还是 2026-04-29 制品；本地 v4 override 不是可发布依赖。
- 当前只有 `tp_size=1` 可进入 `record`/`verify`；TP>1 需要 allocation-time distributed logical lease，
  不能用 transfer-time 当前 token 查询替代。
- `rtp_strict` 当前拒绝初始化；独立 calculator、producer event/sealed write lease、storage committed-byte 证据和
  timeout I/O quiescence 都尚未实现。
- Meta 只保存 checksum value/presence，不保存算法/contract。RTP contract 被绑到 instance identity 和 TP request，
  但不能证明一个脱离该 deployment 的裸 Meta value 的算法身份。
- legacy-v0 不把 iov size/identity/domain 显式编进 checksum，奇数长度 iov 永久漏掉中心字节；
  这是已持久化语义，不能在原算法 ID 下静默修改。
- 头尾采样不能发现未采样中段损坏。如需完整 block 覆盖，必须新增名称/版本稳定的全量算法和 baseline
  迁移，不能静默调大或改写 legacy-v0。
- inline header 尚未实现，配置会明确拒绝。
- Manager 不比较不同 spec 的 checksum；它们对应不同 payload，本来可以不同。同名 spec 的副本一致性
  依靠写/迁移流程和读端校验。
- 通用 KVCM client 当前只有 `ChecksumValidationLog`，没有内置 checksum mismatch/latency Prometheus counter；
  RTP 在 `record`/`verify` 下也只有 integrity event/affected-block QPS，没有 checksum latency、pool wait 和 abort
  outcome 指标；OFF dynamic-URI reconcile/quarantine 连这两个 integrity QPS 也不会发出。完整性能与 SRE
  闭环仍是生产门禁。
- RTP transfer-status version 0 下，OFF/generic failure 为保护旧 coordinator 保留 legacy gRPC error，不能从错误
  回包带回部分动态 URI；checksum contract/action 的结构化失败是独立通道。OFF multi-pool 的完整稀疏 URI
  回收只在 version 1 双端协商成功时成立。
- 真实 TP=1 verify E2E、storage/Meta/部分成功故障注入、业务性能、MUSA 对等覆盖和生产 RSS 验证
  尚未完成；在完成前生产保持 `off`。
