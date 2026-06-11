# 项目文档

### 设计文档
- [基本概念](design/basic_concepts.md) - Storage、Instance Group、Instance、Block、CacheLocation 等核心概念
- [高可用与选主机制](design/ha_leader_elector.md) - HA 架构、LeaderElector 状态机、CoordinationBackend、Leader 发现

### 开发文档
- [开发指南](develop/README.md) - 开发者入门指南和开发环境配置
- [Commit 要求](develop/commit_requirements.md) - 提交前检查和 commit message 格式约定
- [构建版本信息](develop/version_stamping.md) - Version Stamping 机制原理与使用方式
- [API 文档](api/) - API 接口说明和使用示例

### 部署文档
- [镜像文档](../open_source/docker/README.md) - Docker镜像构建和使用说明
- [部署指南](deploy/README.md) - 部署说明
- [配置指南](configuration.md) - 详细的项目配置说明和参数解释

### 模块文档
- [优化器文档](optimizer.md) - 缓存优化策略和算法说明
- Prometheus Metrics
  - [English](prometheus-en_US.md) - Prometheus metrics endpoint documentation
  - [中文](prometheus-zh_CN.md) - Prometheus 指标端点文档
- Crash Stack Trace
  - [English](crash-handler-en_US.md) - Crash signal handler & offline stack decoder
  - [中文](crash-handler-zh_CN.md) - 崩溃堆栈打印与离线解析


---

更多文档将会陆续迁移到本repo。
