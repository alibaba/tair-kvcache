# 开发方式
## 开发镜像
- Manager开发镜像（仅包含Manager相关依赖，不包含CUDA）：
  - 镜像：ghcr.io/alibaba/tair-kvcache-kvcm-dev:latest
  - Dockerfile位置：open_source/docker/Dockerfile.dev
- Connector开发镜像：请直接基于对应推理引擎的开发镜像补充Manager依赖来构造通用开发镜像，参考：[open_source/docker/README.md](../../open_source/docker/README.md)

如果希望在同一容器内开发Manager和Connector，建议使用对应推理引擎的开发镜像。Manager的依赖安装比推理引擎更加简单。
## 编译运行
该项目采用 Bazel 作为构建系统。开发镜像中已经预装bazelisk。
```bash
bazelisk run //kv_cache_manager:main
```
## 测试
- 单元测试： ```bazelisk test //kv_cache_manager/...```
- 集成测试： ```bazelisk test //integration_test/...```
- C++客户端测试： ```bazelisk test //kv_cache_manager/... --config=client```
- 依赖Redis的测试：
  - 需要本地启动一个Redis或Valkey。
  - ```bazelisk test //kv_cache_manager/common/test:redis_client_real_service_test //kv_cache_manager/meta/test:meta_redis_backend_real_service_test //kv_cache_manager/meta/test:meta_indexer_redis_test //kv_cache_manager/manager/test:MetaSearcherRedisTest //kv_cache_manager/config/test:registry_manager_redis_backend_test --test_tag_filters=redis```
- 启用ASAN：上述命令后添加 ```--config=debug --config=asan --test_env ASAN_OPTIONS=detect_odr_violation=0```
### 测试资源清理

测试结束后会自动清理资源。测试工作目录位于 bazel runfiles 目录中，不会污染源代码目录。如果测试异常退出，可能需要手动清理：

```bash
# 清理残留进程
pkill -f kv_cache_manager_bin

# 清理 bazel 缓存（一般情况下无需执行。执行后再次运行测试需要重新拉取并编译所有依赖）
bazelisk clean --expunge
```

### 测试常见问题

#### Q: 新加测试遇到 "instance group not found"

A: 确保使用 `"default"` 作为 instance_group，或在测试前创建自定义 group。

#### Q: Bazel 使用旧的测试结果

A: 添加参数 `--cache_test_results=no` 或删除 bazel 缓存后重新运行。


## 调试
### 集成测试
集成测试涉及多个独立日志源，排查问题时通常需要交叉对比。

#### Manager Server 日志（C++）

Manager 以独立进程运行，日志写入其工作目录下的文件。

**日志位置**（在 bazel test 的 runfiles 目录中，每个测试方法独立）：

```
# 结构化日志（包含 HTTP 请求/响应、FinishWriteCache、GetCacheLocation 等）
<runfiles>/integration_test/<test_method_name>/worker_0/logs/kv_cache_manager.log

# 标准输出/错误（启动信息、signal 处理）
<runfiles>/integration_test/<test_method_name>/worker_0/stdout
<runfiles>/integration_test/<test_method_name>/worker_0/stderr

# 其他日志
<runfiles>/integration_test/<test_method_name>/worker_0/logs/access.log
<runfiles>/integration_test/<test_method_name>/worker_0/logs/event_publisher.log
```

**快速查找**：

```bash
# 找到所有 Manager 日志
find ./ -L -name "kv_cache_manager.log" -path "*integration_test*"

# 查看特定测试的 Manager 日志
find ./ -L -path "*<test_method_name>*/kv_cache_manager.log" | xargs cat
```

**日志级别控制**：Manager 启动时通过 `--env kvcm.logger.log_level=5` 设置（5=DEBUG），这由 TestBase 框架自动配置。

**关键日志模式**（排查写入/查询问题时）：

```bash
# 查看 RegisterInstance、StartWriteCache、FinishWriteCache、GetCacheLocation 关键事件
grep -E "Register|StartWrite|FinishWrite|GetCacheLocation|error|warn" kv_cache_manager.log
```

#### TransferClient 日志（C++ SDK）

TransferClient 是 C++ pybind 模块，日志写入当前工作目录的 `logs/` 子目录。

**日志位置**：

```
<runfiles>/logs/kv_cache_manager_client.log
```

**日志级别控制**：通过环境变量 `KVCM_LOG_LEVEL` 设置：

```bash
# 运行测试时启用 DEBUG 级别
bazel test //integration_test/<target> --test_env=KVCM_LOG_LEVEL=DEBUG
```

**关键日志模式**：

```bash
# 查看 SDK 初始化、文件操作、Alloc 错误
grep -E "DoPut|DoGet|Alloc failed|Init|SdkWrapper" kv_cache_manager_client.log
```

#### 排查流程建议

典型的写入-查询问题排查顺序：

1. **Connector 日志**：确认写入请求是否成功发起、是否有错误返回
2. **TransferClient 日志**：如果数据传输失败，查看 `kv_cache_manager_client.log` 中的 `Alloc failed` 或 `DoPut` 错误
3. **Manager 日志**：确认 `FinishWriteCache` 是否被处理、`GetCacheLocation` 返回了多少 locations

## 编码规范
请参考[.clang-format](../../.clang-format)

githooks中已经添加了C++等语言的格式化脚本，请确保开发环境安装了clang-format、autopep8、buildifier。（开发镜像均已预装）。
## CI
可参考```.github/workflows```目录下的配置。