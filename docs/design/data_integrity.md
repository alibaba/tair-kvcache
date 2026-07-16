# 读写链路数据校验

> 任务: [Aone 82620492](https://project.aone.alibaba-inc.com/v2/project/2163404/task/82620492)
> 状态: meta checksum 通路已落地；inline header 方案仅保留配置和错误码，启动期拒绝。

## 背景与目标

KVCM 原有链路只能通过 `KVCM_SDK_CHECK` 打印 Put / Get 两端 checksum，再人工 grep
对比。这个方式无法让线上请求在读到错误 KVCache 时自动失败，也无法把 writer 算出的
校验信息沉淀到 meta。

本期目标是把已有的 buffer checksum 能力变成一条清晰的读写契约：

- 写入成功后，client 把每个 block 的 checksum 随 `FinishWriteCache` 提交到 meta。
- 读取时，server 把 checksum 随 `CacheLocation` 返回，client 读取数据后重新计算并比对。
- mismatch 时，`LoadKvCaches` 返回 `ER_CHECKSUM_MISMATCH`，上层必须丢弃本次 buffer。
- 老 client、老数据、非 CUDA/MUSA build 都保持兼容，不因为缺少 checksum 影响基本读写。

## 核心契约

checksum 的对齐对象始终是 **StartWriteCache 捕获的完整 keys batch**，不是
`success_blocks` 子集，也不是 caller 传入的 `locations` 参数。这样写入过程中的成功掩码、
location spec 展开、以及读端返回的 location 数量不会混在同一个参数语义里。

关键不变量：

- `FinishWriteOptions::checksums.size() == StartWriteCache.keys.size()`，或者为空表示未上报。
- 失败 block 或不可用 checksum 的位置填 `0`。
- server 在 `CacheManager::FinishWriteCache` 统一校验 checksum 数量。
- `FinishWriteCacheRequest` 使用独立 `repeated int64 checksums` 字段传输，不复用
  `CacheLocation`。
- `MatchLocationResult::checksums` / `MatchMetaResult::checksums` 是输出结果，不放进
  options。
- `LoadKvCachesOptions::expected_checksums` 是读端输入，必须与 `block_buffers` 一一对应。

## 写入链路

```
StartWriteCache
  -> server 为完整 keys batch 分配 WriteLocation

SaveKvCaches(..., SaveKvCachesOptions::WithChecksums())
  -> Put 成功后返回 SaveKvCachesResult{uri_str_vec, checksums}
  -> Put 失败时 checksums 保持空，避免给未落盘数据提交 checksum

FinishWrite(..., FinishWriteOptions::WithChecksums(checksums))
  -> GrpcStub 填充 FinishWriteCacheRequest.checksums
  -> MetaServiceImpl 转发 vector<int64_t>
  -> CacheManager 按 StartWrite keys 下标写入成功 block 的 CacheLocation.checksum
```

`FinishWrite` 仍保留普通重载；没有校验需求的调用方可以继续只传
`trace_id / write_session_id / success_block / locations`。

## 读取链路

```
MatchLocation(..., MatchLocationOptions::WithChecksums())
  -> 返回 MatchLocationResult{locations, checksums}

LoadKvCaches(..., LoadKvCachesOptions::VerifyWith(result.checksums))
  -> 先校验 expected_checksums.size() == block_buffers.size()
  -> sdk_wrapper.Get 读取数据
  -> CUDA/MUSA build 下计算 actual checksums 并逐块比较
  -> mismatch 记录 ChecksumMismatchLog 并返回 ER_CHECKSUM_MISMATCH
```

读端校验始终逐块比较。`expected_checksums[i] == 0` 表示该 block 没有可用 checksum，
读端跳过该位置。包含 `ignore=true`、空 iov、空 block、非法 iov 的 block 无法安全 hash，
也会被降级为跳过。

非 CUDA/MUSA build 会先执行参数长度校验；长度正确但需要实际 hash 时，记录 warn 并
fail-open 跳过计算。

## 配置

`StorageConfig.integrity` 控制能力开关：

```jsonc
{
  "type": "file",
  "global_unique_name": "test_nfs",
  "storage_spec": { "root_path": "/tmp/test/", "key_count_per_file": 5 },
  "integrity": {
    "enable_meta_checksum": true,
    "enable_inline_header": false,
    "inline_header_version": 0,
    "algo": "crc32_xor_int64"
  }
}
```

- 缺少 `integrity` 子对象时视为全关，兼容老配置。
- `enable_meta_checksum=true` 时，client 初始化 `SdkBufferCheckPool`。
- `algo` 目前只支持 `crc32_xor_int64`，与 `SdkBufferCheckUtil` 对齐。
- `enable_inline_header=true` 或 `inline_header_version != 0` 会在配置校验和
  `TransferClientImpl::Init` 阶段被拒绝，避免半实现配置进入运行期。
- `StorageConfig::FromRapidValue` 对 `integrity` 解析失败会整体失败，不静默降级。
- `RegistryManager::AddStorage` / `RecoverStorageUnsafe` 在写入或恢复前都会调用
  `ValidateRequiredFields`。

## API

| 层 | 接口 | checksum 表达 |
|---|---|---|
| TransferClient | `SaveKvCaches` | `SaveKvCachesOptions::WithChecksums()` 打开采集；`SaveKvCachesResult::checksums` 返回结果 |
| TransferClient | `LoadKvCaches` | `LoadKvCachesOptions::VerifyWith(checksums)` 提供 expected checksums |
| MetaClient / ManagerClient | `MatchLocation` | `MatchLocationOptions::WithChecksums()` 打开查询；`MatchLocationResult::checksums` 返回结果 |
| MetaClient / ManagerClient | `MatchMeta` | `MatchMetaOptions::WithChecksums()` 打开查询；`MatchMetaResult::checksums` 返回结果 |
| MetaClient / ManagerClient | `FinishWrite` | `FinishWriteOptions::WithChecksums(checksums)` 提交完整 batch checksums |
| Stub / GrpcStub | `FinishWriteCache` | `std::vector<int64_t> checksums`，默认空 |
| protocol | `FinishWriteCacheRequest` | `repeated int64 checksums` |
| meta | `CacheLocation` | `int64 checksum`，默认 0 |

Options 使用值语义持有 checksum vector，避免 caller 传入临时对象后出现悬空指针。

## Sentinel

`checksum == 0` 表示「未设置 / 不校验」：

1. writer 没有上报 checksum 时，`FinishWriteCacheRequest.checksums` 为空。
2. 某个 block 不可用或失败时，该位置填 `0`。
3. meta 中 `CacheLocation.checksum == 0` 时，读端跳过该 block。

这会带来极低概率的 false negative：真实 checksum 偶然等于 0 时也会被跳过。当前接受
这个兼容性权衡，不引入额外 proto 状态位。

## 故障表现

| 场景 | 行为 |
|---|---|
| 写端 checksum 与读端 actual 不一致 | 记录 `ChecksumMismatchLog`，返回 `ER_CHECKSUM_MISMATCH` |
| `expected_checksums` 长度与 `block_buffers` 不一致 | 返回 `ER_CHECKSUM_MISMATCH`，不读取 storage |
| `FinishWriteCacheRequest.checksums` 长度与 StartWrite keys 不一致 | 返回 `EC_BADARGS` |
| 老 client 不传 checksum | server 不更新 checksum，读端按老数据处理 |
| 老数据 checksum 为 0 | 读端跳过对应 block |
| 非 CUDA/MUSA build 收到 expected checksums | 长度先校验；实际 hash 路径 warn 后 no-op |
| `SaveKvCaches` Put 失败 | 返回错误，`SaveKvCachesResult::checksums` 为空 |
| `SaveKvCaches` 遇到不可 hash block 且 caller 请求 checksums | Put 前返回 `ER_INVALID_PARAMS` |
| merged CacheLocation | winner 的 checksum 透传到 merged location |

当前 mismatch 不自动删除 meta。上层可以根据业务策略选择重试、drop、上报或调用
`RemoveCache`。后续如果需要统一 metrics / event，可以新增一个 server 端 mismatch
处理接口。

## 关键文件

| 文件 | 角色 |
|---|---|
| `kv_cache_manager/protocol/protobuf/meta_service.proto` | `DataIntegrityConfig`、`ChecksumAlgo`、`CacheLocation.checksum`、`FinishWriteCacheRequest.checksums` |
| `kv_cache_manager/data_storage/storage_config.{h,cc}` | integrity 配置解析与 inline header 拒绝 |
| `kv_cache_manager/config/registry_manager.cc` | registry 写入 / 恢复前校验 storage config |
| `kv_cache_manager/meta/cache_location.h` | `CacheLocation.checksum_` 与 JSON 序列化 |
| `kv_cache_manager/manager/cache_manager.cc` | FinishWrite 时按完整 batch 校验并写入 checksum |
| `kv_cache_manager/manager/meta_searcher.{h,cc}` | `LocationUpdateTask.checksum` 透传与 merged location checksum |
| `kv_cache_manager/service/meta_service_impl.cc` | 解析 `FinishWriteCacheRequest.checksums` |
| `kv_cache_manager/client/include/common.h` | checksum Options / Result 定义 |
| `kv_cache_manager/client/src/transfer_client_impl.cc` | Save 采集 checksum，Load 校验 checksum |
| `kv_cache_manager/client/src/internal/util/checksum_verify_util.h` | 逐块 checksum 比对 helper |
| `kv_cache_manager/client/src/internal/stub/grpc_stub.cc` | gRPC request/response checksum 编解码 |

## 后续工作

- 补真实 GPU 环境的端到端 chaos test：write -> 篡改 storage byte -> read 返回
  `ER_CHECKSUM_MISMATCH`。
- Python binding 和推理引擎 connector 接入：Save 收 checksum，Finish 提交 checksum，
  MatchLocation 后 Load 校验 checksum。
- 客户端 metrics / event 发布通路：当前先使用结构化日志，后续决定通过 client publisher
  还是 server 端 mismatch 接口统一上报。
- inline header 方案真实现：写入数据自描述 header，用于从体系上识别 meta 串位导致的
  读串问题。
