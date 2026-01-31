# 部署方式
## 启动 Manager
### 使用容器启动（推荐）
```
docker run kv_cache_manager_prod
```
- 如果需要修改参数，可以直接添加env ```docker run kv_cache_manager_prod --env xxxx=xxxx```
### 从源码启动
```
bazel run //kv_cache_manager:main
```
### 参数配置
可配置参数请参考文档：[configuration.md](../configuration.md)
- 默认参数：package/etc/default_server_config.conf
- 默认日志参数：package/etc/default_logger_config.conf
