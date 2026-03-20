# AGENTS.md

项目文档见 [docs/README.md](docs/README.md)。

## 约束

- **Instance 隔离**：KVCache 仅在同一个 `instance_id` 内复用，跨 Instance 不匹配。

<!--
约束收录原则：只放会导致方向性错误的系统级约束，不放通用工程实践。
组件级实现细节放在对应模块文档中。
-->
