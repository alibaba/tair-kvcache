# Checksum 元数据存储

本功能从 PR #338 拆出 Manager 元数据能力，供已自行计算、校验 CRC 的 v6d 等调用方使用。
Manager 负责接收、持久化和返回 checksum，算法及校验由调用方管理，不需要额外的 storage 配置。

## 存储粒度与值

Checksum 属于 `(instance_id, block_key, location, location_spec_name)`。
不同 spec 对应独立 payload，不能共用一个 location 级 checksum。元数据按 instance 隔离。
值是 opaque `string`，可使用十六进制、Base64、带算法前缀的文本等形式。Manager 不解释算法、
不做数值转换、不改变大小写或前导零，也不限制为 64 位。空字符串统一表示未提供；字符串 `"0"` 是有效值。
协议 string 使用 UTF-8；二进制摘要由调用方编码为十六进制或 Base64。

`LocationSpec` 使用 `unique_ptr<char[]>` 持有独立字段的内容，并记录长度；空指针表示未设置，
非空指针表示有非空值。读取返回 `string_view`，复制 spec 时深拷贝，移动时转移所有权。
序列化使用显式长度，保留内容中的 `\0`；仅非空值输出持久化 JSON 的 `checksum` 字段。
旧记录不含该字段或存有 `"checksum":""`，读取后都归一化为无值，再次序列化时省略字段；不主动重写存量 Redis。
协议仅使用 `checksum` 字段，非空表示提供了值；空字符串或省略均视为未提供。
FinishWriteCache 的 checksums 数组中，空字符串用于未提供 checksum 的占位，不更新已有值。
HTTP 中 checksum 必须是 JSON 字符串，不能传 JSON 数字。

## 两种写入方式

### FinishWriteCache

StartWriteCache 继续返回写入 URI、`block_mask` 和 `write_session_id`。
调用方按 mask 取得实际分配的紧凑 block 序列，写入数据并计算各 spec 的 checksum，再调用：

```json
{
  "instance_id": "example",
  "write_session_id": "session-from-start-write",
  "success_blocks": {"bool_masks": {"values": [true, false, true]}},
  "checksum_batches": [
    {"location_spec_name": "tp0", "checksums": ["crc64:0000deadbeef", "", "sha256:abcdef"]}
  ]
}
```

每组 checksums 与 session 的紧凑 block 序列等长，失败位置保留占位值；没有分配该 spec 的
block 也忽略对应值，可用空字符串占位；服务端不根据占位值判断 spec 是否存在。仅成功的 block
在转为 SERVING 的同一次元数据更新中写入非空 checksum；成功位置也可填空字符串，表示未提供且不更新已有值。
省略 checksum_batches 时保持原写入行为，未提及的 spec 不更新 checksum。

服务端在消费 session 的同一把锁内检查 instance、success mask、数量及 spec 是否被分配。
空或重复的 spec name、组间长度不一致也会被拒绝。参数错误不消费 session，可修正重试；
未完成 session 继续由原超时回调清理。字段号 5 的旧 locations 保留，checksum_batches 使用 tag 6。

### ReportEvent

BLOCK_ADD / BLOCK_SNAPSHOT 的 specs 可携带 checksum。

- BLOCK_ADD 按 spec name 合并：提供非空值时更新，空字符串或省略时保留原值，同一请求内折叠多个 ADD 也遵循此规则。
- BLOCK_SNAPSHOT 是完整替换：保留本次提供的非空值，空字符串或省略时清除旧值。
- 实际 payload 变化时，调用方应同步更新 checksum，防止后续校验使用旧值。

URI 版本化、HTTP 快速解析、COW 更新、部分 spec 删除及 Copy/Mark 迁移均保留所属 spec 的 checksum。
服务端的迁移只携带值，不计算或验证 payload。

## 查询与兼容性

GetCacheLocation、GetCacheLocationsByBackend，以及 Meta/Admin 的 GetCacheMeta
均通过 `include_checksums=true` 返回值；默认不填充 checksum。StartWriteCache 只返回分配信息。
调用方按 checksum 是否非空判断本次查询是否返回了校验值。

协议均为加法扩展，旧调用不需要传新字段。应先升级服务端：旧服务端可能忽略未知字段，
因此 FinishWriteCache 成功不足以证明 checksum 已保存，接入时应通过查询验证返回值。
本功能不扩展 C++ TransferClient、GPU kernel 或推理框架连接器；v6d 可通过 HTTP/gRPC 协议接入。

LocationSpec 新增一个指针和一个 `size_t`，在当前 64 位构建中固定增加 16 字节，未设置时不分配堆内存。
设置非空值时分配 `checksum.size() + 1` 字节，包含末尾终止符；空字符串归一化为空指针，不分配缓冲。
`EstimateMemUsage` 包含此分配大小，不包含分配器额外开销。启用前应按目标 spec 数量评估常驻内存和 COW 峰值。
模块职责与依赖方向不变。

## 验证范围

单元测试覆盖 JSON/proto 往返（含内嵌 `\0`）、空字符串与旧数据、独立深拷贝、移动、内存估算、
非数值文本与前导零、查询 opt-in、COW 合并与替换、部分删除、迁移和
session 校验。HTTP/gRPC 集成测试覆盖自有 checksum 写入、紧凑 batch 对齐、非法请求重试、
跨 instance 拒绝、查询返回以及 ReportEvent 的空字符串与自定义格式。
