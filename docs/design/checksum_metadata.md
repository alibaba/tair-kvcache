# Checksum 元数据存储

本功能从 PR #338 拆出 Manager 元数据能力，供已自行计算、校验 CRC 的 v6d 等调用方使用。
Manager 负责接收、持久化和返回 checksum，算法及校验由调用方管理，不需要额外的 storage 配置。

## 存储粒度与值

Checksum 属于 `(instance_id, block_key, location, location_spec_name)`。
不同 spec 对应独立 payload，不能共用一个 location 级 checksum。元数据按 instance 隔离。
值是 opaque `string`，可使用十六进制、Base64、带算法前缀的文本等形式。Manager 不解释算法、
不做数值转换、不改变大小写或前导零，也不限制为 64 位。空字符串与字符串 `"0"` 都是合法值。
协议 string 使用 UTF-8；二进制摘要由调用方编码为十六进制或 Base64。

`LocationSpec` 将值与 presence 分开保存；持久化 JSON 中存在 `checksum` 字段即表示有值。
旧记录不含该字段，读取后保持无值。协议使用 `checksum` 和 `checksum_present` 两个字段，
ReportEvent 中的空字符串必须设置 `checksum_present=true` 才表示显式提供；非空 checksum 也会被认定为有值。
FinishWriteCache 的 checksums 数组中，每个元素本身就表示提供了值，包括空字符串，无需额外 presence 字段。
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
在转为 SERVING 的同一次元数据更新中写入 checksum，包括合法的空字符串。
省略 checksum_batches 时保持原写入行为，未提及的 spec 不更新 checksum。

服务端在消费 session 的同一把锁内检查 instance、success mask、数量及 spec 是否被分配。
空或重复的 spec name、组间长度不一致也会被拒绝。参数错误不消费 session，可修正重试；
未完成 session 继续由原超时回调清理。字段号 5 的旧 locations 保留，checksum_batches 使用 tag 6。

### ReportEvent

BLOCK_ADD / BLOCK_SNAPSHOT 的 specs 可携带 checksum 与 checksum_present。

- BLOCK_ADD 按 spec name 合并：提供值时更新，省略时保留原值，同一请求内折叠多个 ADD 也遵循此规则。
- BLOCK_SNAPSHOT 是完整替换：保留本次提供的值，省略时清除旧值。
- 实际 payload 变化时，调用方应同步更新 checksum，防止后续校验使用旧值。

URI 版本化、HTTP 快速解析、COW 更新、部分 spec 删除及 Copy/Mark 迁移均保留所属 spec 的 checksum。
服务端的迁移只携带值，不计算或验证 payload。

## 查询与兼容性

GetCacheLocation、GetCacheLocationsByBackend，以及 Meta/Admin 的 GetCacheMeta
均通过 `include_checksums=true` 返回值；默认不填充 checksum。StartWriteCache 只返回分配信息。
调用方必须检查 checksum_present，不能把空字符串当作没有值。

协议均为加法扩展，旧调用不需要传新字段。应先升级服务端：旧服务端可能忽略未知字段，
因此 FinishWriteCache 成功不足以证明 checksum 已保存，接入时应通过查询验证 presence 和值。
本功能不扩展 C++ TransferClient、GPU kernel 或推理框架连接器；v6d 可通过 HTTP/gRPC 协议接入。

LocationSpec 新增 string 和 presence，会增加每个 spec 的内存占用，长字符串还会有额外分配。
启用前应按目标 spec 数量评估 Manager 常驻内存和 COW 峰值；不传 checksum 也有结构体占用。
模块职责与依赖方向不变。

## 验证范围

单元测试覆盖 JSON/proto 往返、空字符串与旧数据、非数值文本与前导零、查询 opt-in、COW 合并与替换、部分删除、迁移和
session 校验。HTTP/gRPC 集成测试覆盖自有 checksum 写入、紧凑 batch 对齐、非法请求重试、
跨 instance 拒绝、查询返回以及 ReportEvent 的空字符串与自定义格式。
