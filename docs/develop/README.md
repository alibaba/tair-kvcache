# 开发方式
## 开发镜像
- Manager开发镜像（仅包含Manager相关依赖，不包含CUDA）：
    - 镜像：tair-kvcache-opensrc-registry.cn-hangzhou.cr.aliyuncs.com/tair-kvcache-opensrc/kv_cache_manager_dev:latest
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
- C++客户端测试： ```bazel test //kv_cache_manager/... --config=client```
- 依赖Redis的测试：
  - 需要本地启动一个Redis或Valkey。 
  - ```bazelisk test //kv_cache_manager/common/test:redis_client_real_service_test //kv_cache_manager/meta/test:meta_redis_backend_real_service_test //kv_cache_manager/meta/test:meta_indexer_redis_test //kv_cache_manager/manager/test:MetaSearcherRedisTest //kv_cache_manager/config/test:registry_manager_redis_backend_test --test_tag_filters=redis```
- 启用ASAN：上述命令后添加 ```--config=debug --config=asan --test_env ASAN_OPTIONS=detect_odr_violation=0```
## 编码规范
请参考[.clang-format](../../.clang-format)
githooks中已经添加了C++等语言的格式化脚本，请确保开发环境安装了clang-format、autopep8、buildifier。（开发镜像均已预装）。
## CI
正在将内部CI系统的流程向Github Actions迁移。
