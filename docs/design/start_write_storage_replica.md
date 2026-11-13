# StartWriteCache 按存储类型补写副本

相关文档：[基本概念](basic_concepts.md)、[分层存储](tiered_storage.md)、[模块关系](module_architecture.md)、[Proto 修改指南](../develop/proto_modification_guide.md)。

## 1. 问题与设计范围

同一个 Instance 的同一个 BlockKey 可以有多个 Location。现有 `min_replica_count` 按符合条件的 Location 总数决定是否跳过写入，不能表达“某种存储上至少需要一份”。当一个 key 已有 100 个 EventReport Location、但没有 TairMempool Location 时，总数门槛可能使共享存储补写被跳过。

本方案增加一个请求级参数，支持指定一种存储类型，或排除一组存储类型后统计期望 Location 数量。数量不足时，`StartWriteCache` 返回符合类型范围的新写入 URI，由客户端完成写入并调用 `FinishWriteCache`。

日常使用可排除 EventReport，使 DRAM、SSD 等其余类型合计达到数量门槛，并保留 Mark 引导分层。数据迁到 SSD、DRAM 副本被回收后，SSD 仍参与计数，不会仅因 DRAM 缺失而补回热层。指定类型模式用于确实需要在某一层补写的场景，继续以显式目标为准。

范围限定为：

- 指定模式按现有 `StorageType` 精确匹配；`ST_TAIRMEMPOOL` 与 `ST_TAIRMEMPOOL_SSD` 分别计数。排除模式对所有未被排除的类型合计计数；两种模式都允许跨同类型后端计数。
- 一个请求指定一种类型或排除一组类型，两者互斥；新参数与非 0 的旧顶层 `min_replica_count` 互斥。
- 每次为每个待写 key 分配一个 Location；目标大于 1 时通过后续调用逐步补足。数量只计 Location，不保证分布在不同后端、节点或故障域。
- 指定模式的显式目标优先于 Mark；排除模式中，范围内尚未满足且可执行的 Mark 优先引导写入，即使数量已满足也可补写。期望数是写入门槛，允许因 Mark 或并发超过目标。
- 新参数只作用于本次请求，不持久化为 Instance 策略，也不触发后台自动补副本。

实现以新增参数是否存在作为行为边界：未设置时保持现有路径，不顺带修改公共计数规则、读路径、迁移、回收或客户端数据传输。

## 2. 请求协议

在 [meta_service.proto](../../kv_cache_manager/protocol/protobuf/meta_service.proto) 增加一个小 message，并为 `StartWriteCacheRequest` 追加字段 8：

```protobuf
message StorageReplicaRequirement {
    StorageType storage_type = 1;
    int32 min_location_count = 2;
    repeated StorageType excluded_storage_types = 3;
}

message StartWriteCacheRequest {
    // 原有字段 1～7 保留，包含顶层 min_replica_count = 7。
    StorageReplicaRequirement storage_replica_requirement = 8;
}
```

使用 message 的存在性区分新旧模式，不把旧字段移入 oneof，也不改旧 proto 字段的默认值。`storage_type` 与非空 `excluded_storage_types` 必须且只能设置一种；`min_location_count` 是两种模式共用的数量门槛。不引入通用写入策略对象。

保留 message 避免缺省类型意外进入旧模式。HTTP 的整个 `storage_replica_requirement: null` 按未设置处理，与空对象 `{}` 不同。[HTTP JSON 解析器](../../kv_cache_manager/service/util/fast_proto_json_codec.cc) 沿用忽略未知枚举名称的行为：指定模式拼错类型或排除列表全部为未知名称时，解析后没有合法模式，因此拒绝；若列表中同时有已知和未知名称，仅已知项会保留。调用方须使用正确的枚举名或数值，不能将该接口的校验视为 JSON 原文的严格校验。本次不修改公共 JSON 解析器；解析后可见的未知枚举数值，以及排除列表中的 `ST_UNSPECIFIED` 均拒绝。

HTTP 请求示例：

```json
{
  "trace_id": "write-shared-copy",
  "instance_id": "example_instance",
  "block_keys": ["42"],
  "write_timeout_seconds": 30,
  "storage_replica_requirement": {
    "storage_type": "ST_TAIRMEMPOOL",
    "min_location_count": 1
  }
}
```

排除模式请求示例（`storage_type` 不设置或为 `ST_UNSPECIFIED`）：

```json
{
  "storage_replica_requirement": {
    "excluded_storage_types": ["ST_EVENT_REPORT_L1P5", "ST_EVENT_REPORT_L2"],
    "min_location_count": 1
  }
}
```

排除类型同时约束计数与本次新分配。重复的排除项不改变结果。只有被排除类型的副本不会满足数量门槛；没有 EventReport、只有一份有效 SSD 也可以满足上述请求。排除条件不持久化、不删除已有副本，也不限制独立 Copy。

仅在新 message 存在时增加以下校验；参数错误使用现有 `EC_BADARGS / INVALID_ARGUMENT`。此处以已有入口检查通过为前提；HTTP/gRPC 对未注册实例的前置检查仍可先返回 `INSTANCE_NOT_EXIST`，不调整错误优先级：

| 输入 | 处理 |
|---|---|
| 未设置新 message | 完全沿用旧参数校验和写入行为 |
| HTTP 中新 message 为 `null` | 按未设置处理，走旧模式 |
| 新 message 存在，包括空对象 | 校验类型和数量；空对象不视为“未设置” |
| 指定类型与非空排除列表同时设置，或两者均未设置 | 拒绝；空列表不能单独启用排除模式 |
| 指定类型为未知枚举或 EventReport 类型 | 拒绝；EventReport 不支持通过 Create 主动分配位置 |
| 排除列表含 `ST_UNSPECIFIED` 或未知枚举数值 | 拒绝；EventReport 类型可以作为排除项 |
| 新数量未设置（解析为 0）或小于等于 0 | 拒绝，不将缺省数量解释为 1 |
| 新 message 存在，旧顶层数量非 0 | 拒绝；显式传旧值 1 也算冲突 |
| 新 message 合法，旧顶层数量为 0／未设置 | 使用新模式 |

旧字段是普通 proto3 scalar，无法区分“未设置”和“显式设置 0”，两者都表示未启用旧模式。Manager 直调也要校验新参数，避免只在 HTTP/gRPC 入口限制；不借此统一旧入口对负数等输入的处理。

为配合新增参数，将 [Manager 声明](../../kv_cache_manager/manager/cache_manager.h) 中旧 `min_replica_count` 的默认实参从 1 改为 0，与 proto 和 SDK 对齐。现有代码只在值大于 1 时进入 min-replica 分支，因此未带新参数时，省略、显式 0、显式 1 的原有行为均不变。新参数追加在旧参数之后，Manager 直调启用新模式时须显式传旧参数 0，即 `StartWriteCache(..., write_timeout_seconds, 0, requirement)`；修改默认实参不会让位置参数自动省略。不要将显式 1 自动归一化为 0，否则会绕过冲突校验。

新数量使用正的 `int32`，本次不增加额外的业务上限。需要一份时显式填写 1；误配很大的值会使后续请求持续尝试补写，直到达到门槛或遇到配额等限制，服务端不自动缩小该值。

## 3. 新模式的计数口径

计数限定在当前 `instance_id + block_key`，每个满足条件的 Location 贡献 1：

| 条件 | 新模式规则 |
|---|---|
| 存储类型 | 指定模式只计匹配类型；排除模式只计不在排除列表中的类型；范围外类型不参与计数及本次存在性检查 |
| 状态 | 沿用 `ExistsForWriteWithMinCount`：直接排除 `NOT_FOUND`，其余状态继续按已有可用性、存在性和 spec 条件判断；不新增状态白名单 |
| 可用性及数据存在性 | 复用当前选择策略及 `GetCheckLocDataExistFunc`，不改它们的公共实现 |
| specs | 沿用现有 min-replica 规则：指定非空 spec group 时，按名称检查单个 Location 是否覆盖该组；未指定时不额外检查 spec 覆盖 |

这是判断某个 Location 是否计入本次写入门槛的条件，不是“一次必须写齐 Instance 全部 specs”的要求。请求指定 `location_spec_group_names` 时，只检查对应 group 的 spec 名称集合，不能仅比较 `spec_size`；未指定或对应 group 名为空时，沿用原来的空 spec 集合和 block 级计数，不增加“必须覆盖全部注册 specs”的规则。实际分配哪些 specs 仍按已有写入逻辑执行。

现有 `StartWriteCache` 支持按已注册的 spec group 分次写入：同一个 key 可以先写 Full 组形成 Location A，再写 Linear 组形成 Location B。后一次申请生成新的 Location，不会把 specs 追加到 A；现有普通查询可合并同一具体存储后端上的有效 Location，返回包含两组 specs 的视图，元数据中的两个 Location 仍独立存在。参见 [分组写入与读取回归用例](../../kv_cache_manager/manager/test/cache_manager_test.cc) 的 `TestStartWriteCacheSpecGroupDedup` 和 `TestWriteThenReadRoundTripWithSpecGroups`。

副本计数与查询合并是两种现有逻辑：若本次明确请求 Full + Linear 组，计数函数逐个检查 Location，不把分别只有 Full 和 Linear 的两个 Location 合成一个完整副本；若分两次分别请求 Full、Linear，则分别判断各组已有的 Location。此次不修改这一区别，也不扩展 specs 合并能力。

同一个 write session 内，数据也可由各 TP rank 分别传输；客户端应在本次 Location 的 specs 全部写完后才报告该 block 成功。EventReport 的增量 ADD 则有独立的按 spec name 合并机制，详见 [ReportEvent 行为](../api/report_event.md)；本方案不改变这两条写入路径。

`WRITING` 按当前元数据中可见的在写状态计数，不检查尚未写完的数据是否已经存在；其失败、超时和清理继续由现有 write session 流程负责，不增加新的租约或过期规则。`SERVING` 的存在性检查及失效 Location 清理复用现有逻辑。

`NEW` 是 `CacheLocation` 的默认构造态，正常 `BatchAddLocation` 会先把它转为 `WRITING` 再登记元数据。现有 min-replica 计数未专门排除 `NEW` 或 `DELETING`，本次也不额外排除；若要调整这些状态是否应当阻止补写，需单独讨论，不与按存储类型计数捆绑。

**N 是补写判断的门槛，不是可读副本数的下限。** `WRITING` 尚未完成，`DELETING` 也可能在删除完成前继续计数并阻止本次补写；本参数不改变回收行为、不保护最后 N 份数据，也不承诺持续维持 N 份。

**两种新模式与原 min-replica 共用有效 Location 的判断规则，新增的计数条件仅为类型范围。** 状态、spec group 解析及缺省值、可用性、数据存在性和逐 Location 计数均沿用原逻辑；不单独定义一套“新模式有效性”规则。旧默认写入、旧 min-replica 的当前处理保持原样，不修改公共 `ExistsForWrite*` 的行为。

## 4. 最小实现路径

**开发时优先扩展现有 `FilterWriteCacheWithMinReplica` 调用链，直接复用计数和后续写入流程。** 对计数部分，旧模式传入原 Location map，新模式传入按请求类型范围筛选的 map，随后调用同一个 `ExistsForWriteWithMinCount`；不要复制状态、spec 覆盖或有效性判断代码。

新增行为集中在参数校验与透传、计数前的类型筛选、普通分配的候选类型筛选，以及 Mark 的模式区分和排除检查。其余复用 [CacheManager](../../kv_cache_manager/manager/cache_manager.cc) 的 `StartWriteCache 申请位置 → 客户端写数据 → FinishWriteCache 确认结果` 流程，不增加新的 Manager、元数据字段或后台任务。

复用不要求抽取通用策略框架、扩展公共 `SelectLocationPolicy` 接口或重构旧默认写入路径；若实现中遇到原逻辑问题，单独记录，不作为本需求的前置修复。

1. `MetaServiceImpl` 解析新 message，转换存储类型并透传到 `CacheManager::StartWriteCache`。HTTP/gRPC 共用这一路径。
2. `FilterWriteCache` 在原有 `min_replica_count > 1` 分流前识别新模式，将 `storage_replica_requirement.min_location_count` 作为门槛传入现有 `FilterWriteCacheWithMinReplica`。旧顶层参数 0 只用于表达未启用旧模式，不能将它当作新门槛。目标为 1 时同样走新模式，不能退回普通“有任意副本就跳过”的路径。
3. 在该函数内仅对新模式按类型筛选 Location，构造临时 `CacheLocationMap`，复用 [ExistsForWriteWithMinCount](../../kv_cache_manager/manager/select_location_policy.cc) 完成计数、状态判断、spec 覆盖和已有可用性检查，不在调用前增加状态过滤。临时 map 只复制共享指针，不复制 Location 数据。两种模式共用现有的请求 spec group 解析和缺省值；旧调用仍传原 map。
4. 指定模式跳过 StartWrite 的 Mark 引导；排除模式复用原 min-replica 的 Mark 联合判断，只过滤本次目标类型被排除的 Mark，不因此清除元数据中的 Mark。继续复用已有 `block_mask`、待写 key 和 spec group 的组织逻辑。
5. 数量不足或有可执行且未满足的 Mark 的 key 进入 `GenWriteLocation`；Mark 目标沿用具体后端分配路径，普通分配入口传入目标类型或排除列表，复用已有 `GenWriteLocationOnStorage`。
6. 后续 `BatchAddLocation → WRITING → write session → FinishWriteCache` 原样执行，复用失败回滚、超时清理、用量记录和响应结构。

### 4.1 后端选择只增加一个有明确用途的过滤条件

为 [SelectCacheWriteDataStorageBackend](../../kv_cache_manager/manager/data_storage_selector.cc) 追加可选目标类型；未指定时保留原来的 `Select(candidates, preference)` 路径。指定时，在现有候选构造后调用已有 `SelectByType(candidates, target_type, false)`，不允许跨类型回退。

排除模式在原有候选构造后移除排除类型，再调用原 `Select(candidates, preference)`。排除列表默认空，旧调用不受影响。原 Group 的 PREFER 回退仅在剩余候选内进行；若 ALWAYS 要求的类型已被排除，则返回原有无候选错误，不绕过排除条件或改写 Group 策略。

仍然执行 Group `storage_candidates`、后端可写性、总配额和类型配额检查；同类型多个后端沿用当前候选顺序。显式类型取代本次请求的 Group 写入偏好，包括 `CPS_PREFER_*` 和 `CPS_ALWAYS_*`；即使 Group 配置了另一类型的 ALWAYS，本次仍按请求类型选择，但必须满足候选范围和配额等条件。不修改 Group 配置，不扩展通用策略枚举或后端能力注册机制。

`SelectByType` 返回第一个类型匹配的候选，不根据已有 Location 做后端分散。候选及可用状态不变时，多次调用补到 N 份通常仍选择同一后端；本参数不提供 N 个独立存储后端的保证。

后端分配选择只在存在待写 key 时执行：在元数据查询、选择策略生成及有效性检查成功，全部 key 计数已达标且没有可执行的 Mark 需要补写时，走现有“无新写入”的返回流程，不执行普通新分配的配额及 Writable 检查。排除模式仍先沿用 Mark 的准入检查。例如后端仍 `Available()`、已有数据有效，但已只读或配额已满，且没有其他 Mark 引导时，可以正常跳过。若 Group 中已注册候选全部不可用，`genSelectLocationPolicy` 可能先返回空，计数前就报错；本次不绕过这一已有检查。

全部跳过时仍沿用现有 session 和响应流程：`locations` 为空、`block_mask` 标记已跳过的 key，`write_session_id` 仍可非空。不能仅凭 session ID 判断本次是否分配了新 Location。

普通数量补写不改用迁移专用的 `CheckExplicitWriteTargets`，继续遵守 `storage_candidates`。排除模式的 Mark 引导保留其原有迁移准入路径，目标可以位于普通候选名单之外，但类型不能被本次请求排除。不把两种候选范围统一，以免改变既有迁移行为。

#### 分配失败：沿用既有错误处理

普通写入和旧 min-replica 在需要分配新 Location 时，也会经过相同的后端选择和错误转换。“无可写后端却返回 INSTANCE_NOT_EXIST”是既有问题，不属于本功能新增的缺陷，也不作为本需求的前置修复。

**选定目标后的分配失败继续报错，不返回“成功但本次跳过”。** 指定模式不跨类型回退；排除模式只在未被排除的普通候选内执行原策略。Mark 的前置准入失败仍按 §4.2 退回数量判断，不等于选定目标后 Create 失败再换后端。旧模式可能因 EventReport 副本已足够而成功跳过；新模式缺副本且无法分配时，整个 StartWriteCache 请求失败，不返回部分成功结果。

下表记录新模式复用的既有失败行为，用于接口说明及兼容性验证，不要求借此修正错误码语义。分配阶段各行均假设输入、实例、元数据查询和此前检查成功；更早的错误优先返回。对外状态指 HTTP/gRPC 响应体的 `header.status.code`，不是传输层状态码；映射沿用 [ToPbError](../../kv_cache_manager/common/error_code.h)。

| 场景 | 内部错误 | 对外状态 |
|---|---|---|
| 目标类型配额达到上限，过滤后没有该类型候选 | `EC_NOENT` | `INSTANCE_NOT_EXIST` |
| 目标类型后端全部只读，过滤后没有该类型候选 | `EC_NOENT` | `INSTANCE_NOT_EXIST` |
| 非空 `storage_candidates` 中没有目标类型后端，或目标均不可用，但前置选择策略仍可生成 | `EC_NOENT` | `INSTANCE_NOT_EXIST` |
| Group 总配额达到上限 | `EC_ERROR` | `INTERNAL_ERROR` |
| 分配入口发现 `storage_candidates` 为空 | `EC_CONFIG_ERROR` | `INTERNAL_ERROR` |
| 后端 Create 返回失败，例如物理空间不足 | `EC_ERROR`，沿用已有分配回滚 | `INTERNAL_ERROR` |
| 计数前无法生成选择策略，例如 Group 中已注册候选全部不可用 | `EC_ERROR` | `INTERNAL_ERROR` |

`INSTANCE_NOT_EXIST` 在这些场景下并不表示实例真的不存在，这是现有错误映射的限制，本次不修改公共错误码。C++ SDK 进一步分别映射为 `ER_SERVICE_INSTANCE_NOT_EXIST`、`ER_SERVICE_INTERNAL_ERROR`；不增加新的 SDK 错误转换规则。

如果业务允许缓存补写失败，业务层可记录失败、放弃本次补写并继续推理；服务端和 SDK 仍返回错误，不能在通用客户端中把上述状态一律转为“跳过成功”，否则会掩盖真实实例或配置错误。确定为临时容量或只读问题时，不立即循环重试，可等待恢复或采用业务已有的退避策略；配置问题应先修正配置。本次不扩展重试框架，也不承诺这些通用错误码足以自动区分原因。

### 4.2 与分层 Mark 的关系

指定模式维持显式目标优先：StartWrite 只依据指定类型与数量决定补写，即使另一层 Mark 未满足也不额外补写。例如 Mark 指向 SSD、请求要求 DRAM 至少 1 份，仍按 DRAM 目标处理并保留 Mark；不增加同类型时优先选 Mark 后端的优化。

排除模式复用旧 min-replica 与 Mark 的联合判断：

```text
需要数量补写 = 范围内符合原计数规则的 Location 数 < N
需要 Mark 补写 = Mark 有效、目标通过原准入检查、类型未被排除，且目标覆盖不足
本次需要写入 = 需要数量补写 或 需要 Mark 补写
```

两项都需要时只写一份到 Mark 后端；只有数量不足时按 Group 策略选择未被排除的普通候选。Mark 覆盖检查继续使用具体后端上 SERVING/WRITING 的联合 specs，不改成副本计数的单 Location 覆盖规则。

| 排除 EventReport、N=1 时已有副本 | Mark | 本次行为 |
|---|---|---|
| 只有 EventReport | 无 | 按 Group 策略补一份 |
| 只有 EventReport | SSD，目标可写且未覆盖 | 写 SSD，同时推进数量目标 |
| DRAM | SSD，目标可写且未覆盖 | 即使数量足够，仍写 SSD |
| SSD，DRAM 已回收 | 无或 SSD 目标已覆盖 | 跳过，不补回 DRAM |
| DRAM | SSD 暂时不可用或配额已满 | 原 Mark 准入退回数量判断；数量已够则跳过，保留 Mark |

若 Mark 目标类型被排除，仅忽略本次引导，不因此清标；数量不足仍尝试允许范围内的普通后端。目标注销、Mark 过期等原有条件清理保持原样。前置准入通过后分配失败沿用错误与回滚，不新增同请求内重选机制。

`FinishWriteCache` 沿用当前规则：仅本次成功 Location 所在的具体 storage 与 Mark 目标一致时，才按已有清标策略处理；类型相同但具体后端不同不等于满足 Mark。数量已达标本身也不触发清标。不扩展 `MigrationManager`，不增加 session 标志或新的清标策略。

日常请求应使用排除模式与分层协同；若所有请求都指定 DRAM，仍不能依靠 Mark 引导它们去 SSD，且 DRAM 被回收后会再次补写。Mark 本身不会自动派生 Copy。两种模式均不改变独立 Copy、源副本回收和 Mark 的有效期，适用范围同步说明于 [分层存储文档](tiered_storage.md)。

### 4.3 并发语义

过滤、分配、登记 Location 仍是现有的分步流程。两个请求同时查到 0 份时，即使目标为 1，也可能各自创建一份；已登记且可见的 `WRITING` 可以让后续请求跳过，但不能消除同时读取旧值的竞争。

本次把数量定义为补写门槛，既不主动删除超过目标的副本，也不增加锁、原子名额预留或条件插入。后续若需要严格并发上限，应单独设计，不能把它作为本次功能已经提供的保证。
