# Commit 要求

本页记录提交前必须遵守的最小要求,供开发者和代码助手在准备 commit 时快速确认。

## Commit message 格式

提交信息首行使用以下格式:

```text
[component] description
```

要求:

- 必须以方括号模块名开头,例如 `[manager]`、`[data_storage]`、`[docs]`。
- 提交信息长度至少 15 个字节。
- `component` 优先使用主要改动所在的顶层模块或子系统名。
- `description` 用一句话说明本次改动,避免只写 `fix`、`update`、`wip` 这类泛化描述。

常用 component:

```text
common
config
data_storage
meta
manager
optimizer
service
event
metrics
protocol
client
py_connector
integration_test
hisim
docs
build
ci
```

示例:

```text
[data_storage] enhance DataStorage::Create interface
[config] fix data race in RegistryManager write operations
[manager] optimize WriteLocationManager
[docs] document commit message requirements
```

不符合要求的示例:

```text
fix bug
manager: fix leader election
chore(ci): add tair-kvcache image sync pipeline
[manager]
```

## 提交前检查

仓库提供 `.githooks` 作为本地提交辅助检查:

- `.githooks/pre-commit` 会格式化 staged 的 C/C++、`BUILD`、`.bzl` 和 Python 文件,并重新 `git add`。
- `.githooks/commit-msg` 会校验 commit message 长度和 `[component]` 前缀。

如需启用仓库内 hooks,在仓库根目录执行:

```bash
git config core.hooksPath .githooks
```

本地 hook 只覆盖低成本检查,不能替代测试。提交前应至少运行与改动相关的单元测试;影响公共模块或用户可见行为时,应扩大到对应 package 或全量测试。

## 提交内容

- 不要提交可能包含密钥、密码、token 或内部环境信息的文件,例如 `.env`、凭证 dump、临时日志归档。
- 每个 commit 保持主题集中,避免把无关格式化、调试文件和功能改动混在一起。
