# 项目文档

### 设计文档
- [模块架构与关联关系](design/module_architecture.md) - 各模块职责、依赖方向、控制流与数据流，附 Mermaid 图
- [基本概念](design/basic_concepts.md) - Storage、Instance Group、Instance、Block、CacheLocation 等核心概念
- [ReportEvent 增量上报与权威快照设计](design/report_event_snapshot_uri_version.md) - 增量/快照协同、提交屏障、故障恢复、性能取舍与 Subscriber 集成
- [ReportEvent / GetHostCacheState 小 block 性能记录](design/report_event_performance.md) - local/Redis 指标解释、锁与可见性语义、有界并发、容量基准及后续优化边界
- [高可用与选主机制](design/ha_leader_elector.md) - HA 架构、LeaderElector 状态机、CoordinationBackend、Leader 发现
- [CacheReclaimer 异步删除与过度逐出优化](design/cache_reclaimer_async_delete.md) - 异步删除生命周期、in-flight credit、反压与无进展退避
- [后台扫描 GC](design/cache_garbage_collector.md) - 基于 authoritative cursor 的后台全量巡检；V1 清理长期 orphan WRITING 和普通 SERVING storage-missing，并提供无副作用读取、精确值条件 CAS 与 HA 生命周期
- [CacheReclaimer 跨 Instance 公平逐出](design/cache_reclaimer_instance_fairness.md) - 按 Instance 用量分配采样与逐出预算，并与异步 credit 协同

### 开发文档
- [开发指南](develop/README.md) - 开发者入门指南和开发环境配置
- [Commit 要求](develop/commit_requirements.md) - 提交前检查和 commit message 格式约定
- [构建版本信息](develop/version_stamping.md) - Version Stamping 机制原理与使用方式
- [API 文档](api/) - API 接口说明和使用示例
- [ReportEvent 与查询接口行为](api/report_event.md) - 面向调用方的事件上报、全量对账、查询、错误处理和测试覆盖清单

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
