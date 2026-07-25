# Breaking Changes

本文档记录需要调用方、部署配置或持久化状态协同升级的不兼容变更。升级前必须按对应条目的前置条件处理，不能仅替换 KVCM 二进制。

## Vineyard 事件上报升级为 EventReportBackend

Introduced by: PR #249

PR #249 将 Vineyard 专用事件上报泛化为 `EventReportBackend`，并区分 `EVENT_REPORT_L1P5` 与 `EVENT_REPORT_L2`。这是一次不兼容升级，不支持新旧 KVCM 二进制、旧 Vineyard 管控客户端和新 EventReport reporter 混合部署。

### 不兼容内容

#### Protobuf 和公开 API

- `StorageType.ST_VINEYARD = 7` 被替换为 `ST_EVENT_REPORT_L1P5 = 7`，并新增 `ST_EVENT_REPORT_L2 = 8`。即使 L1.5 复用了数值 `7`，旧符号和新契约也不构成受支持的源代码或混合版本兼容性。
- Admin `StorageConfig` 的 `VineyardStorageSpec vineyard = 10` 被替换为 `EventReportStorageSpec event_report = 14`，并通过 `storage_type` 区分 L1.5/L2。旧 Admin 请求中的 oneof 字段不会被新服务识别为 EventReport 配置。
- ReportEvent 的参数消息统一改为 `*EventParams`，Block Delete 新增按 `spec_names` 删除的语义，并新增 Host Cache State 等接口。所有外部调用方必须使用 PR #249 对应的协议生成代码。

#### 持久化配置和 JSON 字段

- Storage 的规范类型从 `vineyard` 改为 `event_report_l1p5` 或 `event_report_l2`，storage spec 也从 Vineyard 命名切换为 EventReport 命名。
- Instance Group 的候选字段从 `event_reporting_storage_candidates` 改为 `event_report_storage_candidates`。
- Instance/Client 配置的实例级默认字段最终统一为 `default_query_type`，用于 GetHostCacheState 请求未指定 QueryType 时的默认查询方式。PR #249 初版使用的 `query_type` 已在后续 review 中纠偏，详见下方不兼容变更。

#### CacheLocation 元数据

- Vineyard location id 为 `kvs#v6d#<medium>#<host_ip_port>`。
- EventReport location id 为 `kvs#event_report_l1p5#<medium>#<host_ip_port>` 或 `kvs#event_report_l2#<medium>#<host_ip_port>`。
- 新服务按新的 storage type 和 location id 做匹配、删除及 host cleanup。旧 location 不会自动迁移，也不能作为新 reporter 的有效缓存事实继续使用。

### 影响范围

- 使用 Vineyard/EventReport 的 Admin 客户端、Reporter 和调度查询调用方。
- Registry 中持久化的 Vineyard storage、Instance Group 候选列表和相关 Instance 信息。
- MetaIndexer/MetaStorage 中由旧 Vineyard reporter 产生的 CacheLocation。
- 依赖旧 location id、旧 storage type 名称或旧 oneof 字段的运维脚本。

标准 NFS、3FS、Mooncake、TairMemPool 等 storage 不受此条目影响，但与 EventReport 共用同一 Instance/元数据空间时仍需确认清理范围。

### 升级前置条件和步骤

1. 备份 Registry 与 MetaStorage，并记录受影响的 storage、Instance Group 和 Instance 清单。
2. 停止受影响 Instance 的 reporter 事件写入和调度消费，避免清理期间继续生成旧 location。
3. 将所有外部 Admin/MetaService 客户端和 reporter 更新为 PR #249 对应的 PB 生成代码；确认不再发送 `vineyard = 10` 或依赖 `ST_VINEYARD`。
4. 删除并以 `event_report_l1p5`/`event_report_l2` 重建 Vineyard storage，更新 Instance Group 的 `event_report_storage_candidates`，并确认候选 storage type 与 reporter 上报的 `storage_type` 一致。
5. 清理受影响 Instance 中形如 `kvs#v6d#...` 的旧 CacheLocation。若无法精确区分，清空并重建该 Instance 的 EventReport 元数据，不能让旧、新 location 共存后直接恢复流量。
6. 在同一升级窗口部署全部 KVCM 节点；不要以新旧版本滚动混跑。
7. 启动 reporter，先发送 Node Register，再回放当前仍有效 block 的 Block Add 事件以重建元数据。`EVENT_BLOCK_SNAPSHOT` 在 PR #249 中只是占位契约，升级流程不能依赖它完成恢复。
8. 验证 Host Cache State 只返回新 reporter host，确认 L1.5/L2 指标与 cleanup 正常后再恢复调度流量。

### 回滚说明

回滚到旧 Vineyard 版本同样需要停写并清理 PR #249 生成的 `kvs#event_report_l1p5#...`/`kvs#event_report_l2#...` location，恢复旧 storage/Instance Group 配置，再由旧 reporter 重建元数据。禁止在保留新 EventReport 状态的情况下直接替换为旧二进制。

## Instance 默认查询类型重命名

Introduced by: PR #249 review follow-up

PR #249 review 进一步确认，Instance 注册信息中的 `query_type` 表示请求未显式指定查询类型时使用的 instance 级默认值，而不是某次请求实际采用的查询类型。因此，该字段在 InstanceInfo、注册协议、Registry JSON 和 Client 配置中统一重命名为 `default_query_type`。

### 不兼容内容

- Meta/Admin protobuf 的 `InstanceInfo.query_type` 与 `RegisterInstanceRequest.query_type` 重命名为 `default_query_type`。字段类型和编号 `8` 保持不变，因此 protobuf 二进制 wire 数据兼容，但生成代码的 getter/setter 与源码 API 不兼容，调用方必须重新生成并编译。
- HTTP/protobuf JSON、Registry 持久化 JSON 和 Client JSON 配置仅接受、序列化 `default_query_type`，不兼容读取旧 `query_type` key。旧配置中的值会回落为 `QT_UNSPECIFIED`。
- 请求级 `GetHostCacheStateRequest.query_type` 及其他查询、事件和 trace 中的 `query_type` 不变。显式的请求级 `query_type` 始终优先；只有请求值为 `QT_UNSPECIFIED` 时才使用 `InstanceInfo.default_query_type`。

### 升级要求

1. 将 Registry 快照、Client 配置和 RegisterInstance HTTP JSON 中的实例级 `query_type` 改为 `default_query_type`。
2. 使用新 proto 重新生成并编译所有 Meta/Admin gRPC 与 SDK 调用方，将实例注册和 InstanceInfo accessor 切换为新名称。
3. 在同一升级窗口更新 KVCM 与注册调用方。混合版本虽然可传输字段号 `8` 的 protobuf 二进制数据，但 Registry/HTTP JSON 与生成源码 API 不构成受支持的混合版本契约。
4. 升级后检查 GetInstanceInfo/Registry 回显中的 `default_query_type`；未配置时应为 `QT_UNSPECIFIED`，显式请求级 `query_type` 的行为不变。
