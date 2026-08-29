# Proto文件修改指南

本文档介绍了在KVCacheManager项目中修改Protocol Buffer定义后，需要进行的相应代码调整步骤。

## 概述

当修改`.proto`文件中的消息定义（如添加、删除或修改字段）后，需要同步更新相关的C++代码，以确保proto定义与C++实现保持一致。

## 修改步骤

### 1. 修改Proto定义

在相应的`.proto`文件中进行字段的添加、删除或修改。

例如，在`kv_cache_manager/protocol/protobuf/admin_service.proto`中添加新字段：

```protobuf
message CacheReclaimStrategy {
    string storage_unique_name = 1;
    ReclaimPolicy reclaim_policy = 2;
    TriggerStrategy trigger_strategy = 3;
    int32 trigger_period_seconds = 4;
    int32 reclaim_step_size = 5;
    int32 reclaim_step_percentage = 6;
    int32 delay_before_delete_ms = 7; // 新增字段
}
```

### 2. 更新C++类定义

#### 2.1 更新头文件（`.h`）

在对应的C++头文件中：

- 添加新字段的成员变量
- 添加新字段的getter方法
- 添加新字段的setter方法
- 更新构造函数（如果需要）

```cpp
// 添加成员变量
private:
    int32_t delay_before_delete_ms_;

// 添加getter方法
int32_t delay_before_delete_ms() const { return delay_before_delete_ms_; }

// 添加setter方法
void set_delay_before_delete_ms(int32_t delay_before_delete_ms) {
    delay_before_delete_ms_ = delay_before_delete_ms;
}

// 更新构造函数（如果需要）
CacheReclaimStrategy(const std::string &storage_unique_name,
                     ReclaimPolicy reclaim_policy,
                     const TriggerStrategy &trigger_strategy,
                     int32_t trigger_period_seconds,
                     int32_t reclaim_step_size,
                     int32_t reclaim_step_percentage,
                     int32_t delay_before_delete_ms = 0)  // 添加新参数
    : storage_unique_name_(storage_unique_name)
    , reclaim_policy_(reclaim_policy)
    , trigger_strategy_(trigger_strategy)
    , trigger_period_seconds_(trigger_period_seconds)
    , reclaim_step_size_(reclaim_step_size)
    , reclaim_step_percentage_(reclaim_step_percentage)
    , delay_before_delete_ms_(delay_before_delete_ms) {}  // 初始化新成员
```

#### 2.2 更新实现文件（`.cc`）

在对应的C++实现文件中：

- 在`FromRapidValue`方法中添加新字段的解析
- 在`ToRapidWriter`方法中添加新字段的序列化

```cpp
// 在FromRapidValue方法中添加
bool CacheReclaimStrategy::FromRapidValue(const rapidjson::Value &rapid_value) {
    KVCM_JSON_GET_MACRO(rapid_value, "storage_unique_name", storage_unique_name_);
    // ... 其他字段
    KVCM_JSON_GET_MACRO(rapid_value, "delay_before_delete_ms", delay_before_delete_ms_);  // 新增字段
    return true;
}

// 在ToRapidWriter方法中添加
void CacheReclaimStrategy::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept {
    Put(writer, "storage_unique_name", storage_unique_name_);
    // ... 其他字段
    Put(writer, "delay_before_delete_ms", delay_before_delete_ms_);  // 新增字段
}
```

### 3. 更新Proto转换函数

在`kv_cache_manager/service/util/manager_message_proto_util.cc`文件中：

- 在`ProtoToCacheConfig`函数中添加proto到C++对象的转换
- 在`CacheConfigToProto`函数中添加C++对象到proto的转换

```cpp
// 在CacheConfigFromProto函数中添加
void ProtoConvert::CacheConfigFromProto(const proto::admin::CacheConfig *proto_cache_config,
                                        CacheConfig &cache_config_info) {
    // ... 其他字段设置
    reclaim_strategy->set_delay_before_delete_ms(proto_cache_config->reclaim_strategy().delay_before_delete_ms());  // 新增字段
    // ... 其他代码
}

// 在CacheConfigToProto函数中添加
void ProtoConvert::CacheConfigToProto(const CacheConfig &cache_config_info,
                                      proto::admin::CacheConfig *proto_cache_config) {
    // ... 其他字段设置
    reclaim_strategy->set_delay_before_delete_ms(cache_config_info.reclaim_strategy()->delay_before_delete_ms());  // 新增字段
    // ... 其他代码
}
```

### 4. 更新测试代码

在相关的测试文件中：

- 更新对象创建代码，设置新字段的值
- 更新断言代码，验证新字段的值

```cpp
// 在测试代码中添加
const auto reclaim_strategy = std::make_shared<CacheReclaimStrategy>();
// ... 设置其他字段
reclaim_strategy->set_delay_before_delete_ms(0);  // 设置新字段

// 在断言中添加
ASSERT_EQ(reclaim1.delay_before_delete_ms(), reclaim2.delay_before_delete_ms());  // 验证新字段
```

#### 4.1 检查 Protobuf JSON 快路径兼容性

`ProtoMessageJsonUtil` 会对 KVCM 当前协议使用的类型直接做 protobuf Reflection 与 RapidJSON
之间的转换。新增或修改字段时需要确认其类型是否在以下支持范围内：

- `int32`、`uint32`、`int64`、`uint64`、`float`、`double`、`bool`、`string` 和 enum；
- 普通嵌套 message、repeated 和 oneof；
- `map<string, string>`；
- `google.protobuf.Int32Value` 和 `google.protobuf.Int64Value`。

`ProtoMessageJsonUtilTest.TestFastCodecSupportsAllProtocolMessages` 会检查所有当前协议消息。
不在列表内的类型仍会回退到 protobuf 3.8 的通用 JSON 实现，功能不受影响，但 access log
等热点路径无法获得加速。引入例如 `bytes`、其他 map 形态或其他 well-known type 时，应同时补充
`FastProtoJsonCodec` 的实现与兼容性测试，或者在评审中明确接受相关 message 回退，并调整上述测试。

### 5. 构建和测试

完成上述修改后，执行以下步骤验证修改是否正确：

```bash
# 构建项目
bazel build //kv_cache_manager/...

# 运行相关测试
bazel test //kv_cache_manager/xxxx:xxxx
```

### 6. 更新docs中的示例文档
- AdminService的文档：docs/api/admin_service.md
- MetaService的文档：docs/api/meta_service.md

### 7. （如果是修改MetaService） 更新客户端调用
- RTP-LLM C++客户端: kv_cache_manager/client
- vLLM等Python客户端：kv_cache_manager/py_connector

### 8. （如果是修改AdminService）同步适配 kvcm_ops 运维 CLI

- 工具位置：`package/kvcm_ops`
- 若 InstanceGroup / Storage 等管理对象新增或变更字段，需同步更新对应的 Python 模型（如 `kvcm/instance_group/util.py`）的构造、校验与 JSON 序列化/反序列化，以及相关 create/update 命令参数。
- 否则 kvcm_ops 的 update 流程（GET → 整体 PUT）会静默丢弃服务端已有字段，导致配置被意外清空。

## 注意事项

1. **字段编号**：在proto文件中添加新字段时，使用递增的字段编号，避免重复使用已存在的编号。
2. **默认值**：为新字段提供合理的默认值，确保向后兼容性。
3. **数据类型**：确保proto定义中的数据类型与C++实现中的数据类型匹配。
4. **JSON序列化**：确保新字段在JSON序列化和反序列化过程中正确处理。
5. **测试覆盖**：确保测试代码覆盖新字段的设置和验证。
