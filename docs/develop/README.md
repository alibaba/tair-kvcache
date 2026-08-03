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

Mooncake 和 VCNS 后端默认不参与编译，需要时通过 Bazel 配置显式启用：

```bash
# Mooncake（包含 CUDA、HTTP 和 TCP 支持）
bazelisk run //kv_cache_manager:main --config=mooncake

# Mooncake（不启用 CUDA）
bazelisk run //kv_cache_manager:main --config=mooncake_common

# VCNS（仅内源模式提供真实实现）
bazelisk run //kv_cache_manager:main --config=vcns
```

两个后端可通过同时传入 `--config=mooncake` 和 `--config=vcns` 一起启用。`--config=client` 会继续启用客户端所需的 Mooncake 支持。

### Bazel 缓存与多 worktree 开发

Bazel 默认会按 workspace 路径生成独立的 `output_base`。因此从同一个仓库拉出新的 git worktree 后，新的 worktree 不能直接复用旧 worktree 的 `bazel-out`、analysis cache 和本地 action cache；首次构建仍需要重新完成 loading/analysis、内部 symlink/action bookkeeping，以及测试执行。

为了让新 worktree 尽量复用已有编译产物，建议在个人 `~/.bazelrc` 中配置共享缓存：

```bazelrc
# 将 Bazel output root 放到稳定、空间充足、IO 较快的本地目录。
# 这主要加速同一个 worktree 的后续增量构建；不同 worktree 仍会有不同 output_base。
startup --output_user_root=/path/to/local/bazel-output

# 共享 action output cache。不同 worktree 中 action key 相同的 C++ compile、link、proto、genrule 等动作可以直接命中。
build --disk_cache=/path/to/local/bazel-disk-cache

# 共享外部依赖下载缓存，避免每个新 output_base 重新下载 http_archive/http_file。
common --repository_cache=/path/to/local/bazel-repository-cache
```

如果同一台机器上存在多套 Bazel、编译器、系统镜像或 CUDA/MUSA 工具链，建议按工具链维度拆分 `disk_cache` 目录，避免无效缓存占用空间，也避免不同工具链之间相互干扰。`repository_cache` 缓存的是下载文件，通常可以跨工具链共享。

`bazelisk clean --expunge` 只清理当前 worktree 的 `output_base`，不会清理上面配置的共享 `disk_cache` 或 `repository_cache`。如果需要释放共享缓存占用的磁盘空间，需要手动删除对应目录；删除后首次构建会重新下载依赖或重新编译产物。

排查缓存是否生效时，可先确认 Bazel 实际读取的 rc 配置和缓存路径：

```bash
bazelisk info --announce_rc output_base
bazelisk info --announce_rc repository_cache
```

构建结束时关注 summary 中的 `disk cache hit` 和 `local` 数量。如果新 worktree 首次构建仍有大量 `local` C++ 编译，通常说明 cache 之前没有预热、构建参数/工具链发生变化、修改触发了相关 action key 变化，或启用 stamp 的目标受 stable status 变化影响。

## 测试
- 单元测试： ```bazelisk test //kv_cache_manager/...```
- 集成测试： ```bazelisk test //integration_test/...```
- C++客户端测试： ```bazelisk test //kv_cache_manager/... --config=client```
- 依赖Redis的测试：
  - 需要本地启动一个Redis或Valkey。
  - ```bazelisk test //kv_cache_manager/common/redis/test:redis_client_real_service_test //kv_cache_manager/meta/test:meta_redis_backend_real_service_test //kv_cache_manager/meta/test:meta_storage_backend_manager_real_redis_test //kv_cache_manager/meta/test:meta_indexer_redis_test //kv_cache_manager/manager/test:MetaSearcherRedisTest //kv_cache_manager/config/test:registry_manager_redis_backend_test --test_tag_filters=redis```
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

## Proto 修改

修改 `kv_cache_manager/protocol/protobuf` 下的 proto 定义时，请遵循 [Proto 文件修改指南](proto_modification_guide.md)，并同步完成其中列出的适配步骤（例如修改 AdminService 接口定义时需适配 `package/kvcm_ops` 运维 CLI）。

## 提交与 PR 要求

提交前检查和 commit message 格式见 [Commit 要求](commit_requirements.md)。

发起和参与 PR 时应遵守 [Pull Request 协作指南](pull_request_guidelines.md)，主动控制评审负担并为其他参与者准备所需的信息。

## CI
可参考```.github/workflows```目录下的配置。```test-opensrc``` 在一个 ```normal_test``` job 中运行普通单元测试和集成测试（包含默认配置下的客户端测试目标），ASAN 测试使用独立 job。
