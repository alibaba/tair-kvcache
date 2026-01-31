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

## 注意事项

1. **字段编号**：在proto文件中添加新字段时，使用递增的字段编号，避免重复使用已存在的编号。
2. **默认值**：为新字段提供合理的默认值，确保向后兼容性。
3. **数据类型**：确保proto定义中的数据类型与C++实现中的数据类型匹配。
4. **JSON序列化**：确保新字段在JSON序列化和反序列化过程中正确处理。
5. **测试覆盖**：确保测试代码覆盖新字段的设置和验证。