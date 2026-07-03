# 读写链路数据校验

> 任务: [Aone 82620492](https://project.aone.alibaba-inc.com/v2/project/2163404/task/82620492)
> 状态: 方案 A 已落地；方案 B 接口预留 + 启动期拒绝。

## 背景与目标

KVCM 读写链路上目前没有体系化的数据校验。SDK 端虽然有 `SdkBufferCheckUtil::GetBlocksHash`
(GPU 上算 CRC32 → 聚合为 `int64` checksum 的能力)，但仅由 `KVCM_SDK_CHECK` 环境变量打开
后 **print 到日志**，需要人工 grep 比对 Put / Get 两端结果 —— 线上发现不了「读错」「读串」。

本期目标分两层：

- **方案 A (完整实现)**: 把 client 端算出的 `checksum` 写入 KVCM meta，Get 端取回 expected
  checksum → 算 actual → 不匹配则返回错误码 + 发布事件。**闭环已有的 buffer_check 能力**，
  覆盖「读错 (数据脏 / 落盘错)」场景。
- **方案 B (接口留位 + 启动期拒绝)**: proto 字段、配置开关、错误码全部加上，运行期看到
  `enable_inline_header=true` 直接拒绝启动 / 拒绝 Init，避免「半实现」。等后续 P1 实现
  `sdk_wrapper` 的 inline header layout 时直接删拒绝分支即可，不动 proto 兼容性。

「读错」「读串」的区别 ——

- **读错**: 拿到的数据本身被改写 (磁盘比特翻转、落盘缓冲被踩、传输出错)。方案 A 即可挡。
- **读串**: meta 路由错误，把另一个 block 的数据拿成自己的 (location id 冲突、缓存串位)。
  方案 A 只能在 checksum 不一致时发现，但分不清「读错」还是「读串」；要从体系上挡住「读串」
  需要数据块自描述 (block_key 跟数据一起写进存储介质) —— 这是方案 B 的目标。

## 方案 A: meta 校验通路

### 流程图

```
Write path
──────────
StartWriteCache  ──▶  client allocates locations
sdk_wrapper.Put  ──▶  data hits storage backend
GetBlocksHash    ──▶  checksums[i]
FinishWriteCache ──▶  request.locations[i].checksum = checksums[i]
                        │
                        ▼
   MetaServiceImpl ──▶ CacheManager ──▶ MetaSearcher
                        │
                        ▼
   CacheLocation.checksum persisted in MetaIndexer (Redis / Local)

Read path
──────────
GetCacheLocation ──▶ server returns CacheLocation (including checksum)
sdk_wrapper.Get  ──▶ buffer
GetBlocksHash    ──▶ actual[i]
                        │
                        ▼
   VerifyBatchChecksums(expected, actual, strict_mode):
     fast (default) ── XOR aggregate with position-dependent odd multiplier,
                       single compare. Order-sensitive: block swaps and same-
                       delta paired mutations both change the aggregate.
                       └─ mismatch → fallback per-block to pinpoint faulty indices
     strict         ── KVCM_CHECKSUM_STRICT_MODE=1 forces per-block always
                       (diagnostic knob for on-call triage)

     all match  → ER_OK
     mismatch   → KVCM_LOG_ERROR per faulty block + ER_CHECKSUM_MISMATCH
```

### 关键开关 (`StorageConfig.integrity`)

```jsonc
{
  "type": "file",
  "global_unique_name": "test_nfs",
  "storage_spec": { "root_path": "/tmp/test/", "key_count_per_file": 5 },
  "integrity": {
    "enable_meta_checksum": true,    // 方案 A 开关
    "enable_inline_header": false,   // 方案 B 开关，本期必须 false
    "inline_header_version": 0,      // 方案 B，本期必须 0
    "algo": "crc32_xor_int64"        // 当前唯一算法 (与 SdkBufferCheckUtil 对齐)
  }
}
```

- spec 不带 `integrity` → 视为全关 (向后兼容)
- `enable_meta_checksum=true` → client 自动初始化 `SdkBufferCheckPool` (即使没设
  `KVCM_SDK_CHECK` 环境变量)
- `enable_inline_header=true` → 启动期被 `StorageConfig::ValidateRequiredFields` 与
  `TransferClientImpl::Init` **两层拒绝** (`EC_UNIMPLEMENTED` / `ER_INLINE_HEADER_INVALID`)
- `inline_header_version != 0` 但 `enable_inline_header=false` → 同样被拒绝 (孤儿 version)
- `KVCM_CHECKSUM_STRICT_MODE=1` 环境变量 → 读端强制走 per-block 严格校验，跳过 fast XOR
  聚合 (排查用)

### API 变化

| 层 | 接口 | 新增参数 | 兼容性 |
|---|---|---|---|
| TransferClient | `SaveKvCaches` | `std::vector<int64_t> *out_checksums` | default `nullptr`，老 caller 不变 |
| TransferClient | `LoadKvCaches` | `const std::vector<int64_t> *expected_checksums` | default `nullptr`，老 caller 不变 |
| ManagerClient / MetaClient | `FinishWrite` | `const std::vector<int64_t> &checksums` | default `{}` |
| Stub / GrpcStub | `FinishWriteCache` | same | `{}` |
| CacheManager | `FinishWriteCache` | same | `{}` |
| MetaSearcher::LocationUpdateTask | `checksum` field | — | default `0`，0 视为「不更新」 |
| CacheLocation | `checksum_` | 字段 + JSON 字段 | default `0`，JSON 兼容 |

### 兼容性 sentinel

`checksum == 0` 在三个位置都表示「未设置 / 跳过」:

1. `LocationUpdateTask.checksum = 0` → `BatchUpdateLocationStatus` 不覆盖 CacheLocation 已有 checksum
2. `CacheLocation.checksum = 0` → 老数据 / 老 client，读端跳过校验
3. `FinishWriteCacheRequest.locations[i].checksum = 0` → server 写入 0，行为同上

`int64_t` 全 0 的概率虽然小但非零 (CRC32 hash 偶然全 0 的概率约 `2^-32` per block，
再聚合 xor 后概率更低)。当前接受这个 false-negative，不引入 3-state 语义 (避免
proto 字段类型升级)。

### Validate 四层防线

`DataIntegrityConfig::ValidateRequiredFields` 同时做三件事:

1. `enable_inline_header=true` → 拒绝 (方案 B 未实现)
2. `inline_header_version != 0` 但开关没开 → 拒绝 (配置矛盾)
3. `enable_meta_checksum=true` 但 `algo == CA_UNSPECIFIED` → 拒绝 (必须显式设算法)

- **StorageConfig 层**: `StorageConfig::FromRapidValue` 里 integrity 子对象解析失败时
  整段 JSON 解析失败 (不允许静默降级为「全关」)；`StorageConfig::ValidateRequiredFields`
  把 DataIntegrityConfig 的拒绝合并到自身。
- **Registry 层**: `RegistryManager::AddStorage` 与 `RegistryManager::RecoverStorageUnsafe`
  在写入 registry 之前都调 `ValidateRequiredFields`，防止半合法的配置落盘后每次重启都
  被 recover 进来，最终让 client 端 Init 说不 —— server 与 client 视图不一致。
- **Client 层**: `TransferClientImpl::Init` 解析 `init_params.storage_configs` 时做相同
  检查 + 返回 `ER_INLINE_HEADER_INVALID` / `ER_INVALID_STORAGE_CONFIG`。

### 读端校验策略：fast + per-block fallback

Reviewer 反馈：「一个 location 一个 checksum 就够了，因为每次都是一起读的；
但每个 block 都 check 的模式可以保留，用于排查比较复杂的数据错误问题。」

落地为两段式校验 (`client/src/internal/util/checksum_verify_util.h`):

**Stage 1 — fast (默认)**: 把整个 batch 的 expected 和 actual 各自聚合成一个 `uint64`
比一次。每个 block 的贡献先乘以 `(2*i+1) * kIndexSalt` (kIndexSalt = 2^64/phi，
奇数，模 2^64 乘法可逆) 再 XOR，聚合是 order-sensitive 的 —— 整数乘法不是
GF(2)-线性，`(A*m0)^(B*m1) != (B*m0)^(A*m1)`，所以 block 顺序错乱 (读串) 直接被抓；
同一 delta 成对突变 (`expected=[A,B]`, `actual=[A^D, B^D]`) 也因非线性无法对消。
批量越大，相对节省越多 (避免 N 次 cache line read + 分支)。

**Stage 2 — slow (仅在 fast 检测到 mismatch，或 strict_mode=true 时进入)**: 遍历
batch，回填每个错位 block 的 index，让上层逐块 log + 后续发布
`ChecksumMismatchEvent`。

**strict mode (`KVCM_CHECKSUM_STRICT_MODE=1`)**: 跳过 fast 聚合，直接走 per-block。
用途：诊断工具 —— 排查阶段想要 per-block 结果而不想经过 fast fallback 时使用；
`ChecksumVerifyUtilTest.FastPathCatchesBlockSwap` /
`ChecksumVerifyUtilTest.DetectsSameDeltaPairedMutation` 分别覆盖两种历史 fast 漏检
模式，防止将来实现回退到纯 XOR 聚合时静默失效。

## 方案 B: 接口预留 (本期未实现)

设计目标 (待后续 P1 落地):

- 每块 KV data 在存储介质上前置 16-32B header: `magic + block_key + data_len + checksum + version`
- `sdk_wrapper::Put` 写时拼 header，`Get` 时读完先校验 header.block_key 是否等于自己请求的
  block_key —— 直接挡住「读串」(meta 串位 → 读到别人的 block)
- 容量统计 (`DataStorageSelector` / `CacheReclaimer` / `GetStorageUsageRatio`) 必须把
  `inline_header_size` 算进去；Spectrum 参数生成也要反映
- 滚动升级:`StorageSpec.inline_header_version=0` → 当成老数据放过；≥1 → 必须校验；
  Reclaimer 不主动清老数据，靠 LRU 自然过期

**当前的拒绝防线** 让任何尝试启用方案 B 的部署在启动期立刻报错，避免「proto 开了开关但实际
没生效」的静默故障。P1 实现完成后只需删 `DataIntegrityConfig::ValidateRequiredFields` 与
`TransferClientImpl::ValidateStorageConfigsForIntegrity` 中两条 inline_header 拒绝分支，
不需要动 proto / 配置 / 接口签名。

## 容错与故障表现

| 场景 | 行为 |
|---|---|
| client 写时 checksum != 读时 checksum | fast 聚合检测到不一致 → fallback 找出哪个 block 错 → 每个错位块 KVCM_LOG_ERROR → `LoadKvCaches` 返回 `ER_CHECKSUM_MISMATCH`，上层应丢弃 buffer |
| `expected_checksums` 长度与 `block_buffers` 不一致 | `ER_CHECKSUM_MISMATCH` (视为 client bug) |
| `FinishWriteCacheRequest.locations` 长度与 `keys` 不一致 | `EC_BADARGS` (server 校验) |
| 老 client 不传 checksum | server 收到 `checksums={}`，不修改 CacheLocation 已有 checksum |
| 老数据 (meta 里 `checksum=0`) 被新 client 读 | 读端 sentinel 跳过，不当 mismatch |
| 非 CUDA/MUSA build 的 client 收到 `expected_checksums` | warn 日志 + 整条校验路径 no-op (fail-open) |
| spec 开了 `enable_meta_checksum` 但 client 没传 `expected_checksums` | 不校验 (兼容增量接入) |
| spec 开了 `enable_meta_checksum` 但 `sdk_buffer_check_pool` init 失败 | `Init` 返回 `ER_INIT_CHECK_BUFFER_ERROR` |
| Block swap / 同 delta 成对突变 | 引入 position-dependent 奇数乘子后 fast path 直接抓；strict_mode 一致 |
| RegistryManager 收到 `enable_inline_header=true` 的持久化配置 | `AddStorage` / `RecoverStorageUnsafe` 都 `ValidateRequiredFields` 拒 (`EC_BADARGS`，recover 时 `++error_count` skip) |
| StorageConfig JSON 里 `integrity` 字段类型错乱 | `FromJsonString` 直接返回 false，`RegistryManager` / `TransferClient` 拿不到降级后的配置 |
| `SaveKvCaches` IOV 数超 `KVCM_SDK_MAX_CHECK_IOV_NUM` | Put 前拒 (`ER_INVALID_PARAMS`)，不写入介质 |
| `SaveKvCaches` 之后 `Put` 失败 | `out_checksums` 保持空，caller 不会把 hash 透传给 FinishWrite |
| merged CacheLocation (batch_get / prefix / reverse-window) | winner 的 checksum 透传到 merged location，读端能校验 |

mismatch 时**当前不自动调用 RemoveCache 清理 meta** —— 留给上层 (ManagerClient 或推理引擎
connector) 根据业务策略决定 (重试 / drop / 上报)。未来可以加一个 `RemoveCacheByMismatch`
专用接口让 server 同时打 metrics + 发 `ChecksumMismatchEvent`。

## 实现拆分

落地共 20 个 commit (跳过原 plan 中的 commit 10 chaos test，理由见 follow-up):

| Commit | 范围 |
|---|---|
| 1 | `[protocol] add DataIntegrityConfig and CacheLocation.block_hash` (3 份 proto) |
| 2 | `[common] add CHECKSUM_MISMATCH / INLINE_HEADER_INVALID error codes` (4 份 proto + C++) |
| 3 | `[data_storage] persist DataIntegrityConfig and reject inline_header` |
| 4 | `[meta] carry block_hash on CacheLocation with backward-compatible JSON` |
| 5 | `[manager] plumb block_hash through MetaSearcher and CacheManager::FinishWriteCache` |
| 6 | `[service] parse block_hash from FinishWriteCacheRequest.locations and forward to manager` |
| 7 | `[event] add ChecksumMismatchEvent class for data integrity reporting` |
| 8 | `[client] expose block_hash on TransferClient and reject inline_header at Init` |
| 9 | `[client] plumb block_hash through ManagerClient and MetaClient FinishWrite` |
| 10 | `[docs] add design doc for data integrity check` |
| 11 | `[manager] update storage JSON literal assertions for new integrity field` |
| 12 | `[refactor] rename block_hash to checksum across protocol and C++ layers` |
| 13 | `[client] add fast batch verify with per-block fallback on Load path` |
| 14 | `[client] address codex review feedback (read-path round-trip + 6 fixes)` |
| 15 | `[manager] copy checksum into merged CacheLocation for read path` |
| 16 | `[client] make VerifyBatchChecksums fast path order-sensitive` |
| 17 | `[client] address 4 more review points on Save/Load checksum path` |
| 18 | `[config] validate storage config at RegistryManager Add / Recover` |
| 19 | `[data_storage] propagate integrity JSON parse failure in StorageConfig` |
| 20 | `[docs] update data_integrity design doc for round-2 review feedback` |

每个 commit 在 `github-opensource/` 下用 `bazelisk test --config=debug --config=asan
--test_env ASAN_OPTIONS=detect_odr_violation=0 //kv_cache_manager/<改动 package>/test:all`
跑相关 UT，串完最后切内源跑一次全量。详见 `.claude/skills/kvcm-build-test/SKILL.md`。

## 后续工作 (Follow-ups)

1. **端到端 chaos test**: 原 plan 的 commit 10 已显式跳过。补一个真实 GPU 环境下的
   integration test:write → 翻 storage 文件一个 byte → read 期望 `ER_CHECKSUM_MISMATCH`。
   需要 CUDA/MUSA runtime + 实际 GPU 设备。建议加在 `integration_test/client_test/` 下。
2. **py_connector 接入**: 当前 Python binding 暂时用 lambda 截断到老 3 参数，
   vLLM / SGLang / TRT-LLM connector 都还没真正算 checksum 上报。需要:
   - Python binding 加 4 参数 (用 list 表示 checksum)
   - 各 connector 在 Save 时收 checksum → Finish 时传回 server；Get 时拿 expected → Load
3. **metrics 接入**: 当前没加 `checksum_mismatch_counter`，因为信号源在 client 端且 client
   没有 MetricsCollector。后续可在 client 加新 collector，或加 server 端 `RemoveCacheByMismatch`
   专用接口触发现有 ServiceMetricsCollector。
4. **方案 B (inline header) 真实现**: 是独立的 P1 工作，工作量比方案 A 大约 1 倍。
   重点是 `sdk_wrapper` 的 buffer layout 改造、容量统计感知、滚动升级 lazy migration。
5. **多副本一致性**: 同一 block 多副本必须有相同 checksum。当前假设 client 端为每个 block
   算一次唯一 checksum，所有副本共享。多 spec / 多 location backend 路径上理论上应该满足，
   实际部署时需要 chaos test 覆盖。
6. **client 端 `ChecksumMismatchEvent` 发布通路**: 类已就位 (commit 7)，但 client SDK
   当前没有持有 `EventManager`。需要决定 client 端事件发布机制 (新加 publisher
   依赖、或者通过 `RemoveCacheByMismatch` 让 server 端发布)。

## 关键文件索引

| 文件 | 角色 |
|---|---|
| `kv_cache_manager/protocol/protobuf/meta_service.proto` | `DataIntegrityConfig` + `ChecksumAlgo` + `CacheLocation.checksum` + `FinishWriteCacheRequest.locations` |
| `kv_cache_manager/protocol/protobuf/admin_service.proto` / `kv_meta_service.proto` | 同上 (各自 package 独立定义) |
| `kv_cache_manager/data_storage/storage_config.{h,cc}` | C++ `DataIntegrityConfig` + Validate 拒绝防线 |
| `kv_cache_manager/meta/cache_location.h` | `CacheLocation.checksum_` + JSON 序列化 |
| `kv_cache_manager/manager/meta_searcher.{h,cc}` | `LocationUpdateTask.checksum` 透传 |
| `kv_cache_manager/manager/cache_manager.cc:FinishWriteCache` | 接收 `checksums` 参数 |
| `kv_cache_manager/service/meta_service_impl.cc:FinishWriteCache` | 解析 `request->locations()` 中的 checksum |
| `kv_cache_manager/event/spec_events/data_integrity_event.h` | `ChecksumMismatchEvent` 类 |
| `kv_cache_manager/client/include/transfer_client.h` | SDK 接口加 checksum 重载 |
| `kv_cache_manager/client/src/transfer_client_impl.cc` | `Init` 拒绝 inline_header + Save/Load 算 / 校验 checksum |
| `kv_cache_manager/client/src/internal/util/checksum_verify_util.h` | `VerifyBatchChecksums` fast/slow 校验逻辑 (header-only，独立 UT) |
| `kv_cache_manager/client/src/manager_client_impl.cc` / `meta_client_impl.cc` | 透传 checksum 到 stub |
| `kv_cache_manager/client/src/internal/stub/grpc_stub.cc:FinishWriteCache` | 把 checksum 填到 proto `locations[i].checksum` |
| `kv_cache_manager/common/error_code.h` | `EC_CHECKSUM_MISMATCH` / `EC_INLINE_HEADER_INVALID` |
| `kv_cache_manager/client/include/common.h` | `ER_CHECKSUM_MISMATCH` / `ER_INLINE_HEADER_INVALID` |
| `kv_cache_manager/client/src/internal/sdk/sdk_buffer_check_util.{cc,cu,mu}` | 已存在的 checksum 计算实现 (CRC32 xor int64) |
