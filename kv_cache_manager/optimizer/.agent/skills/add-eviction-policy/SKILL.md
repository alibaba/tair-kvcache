# 新增驱逐策略 Skill

当需要给 Optimizer 新增 cache eviction policy 时，使用这个 skill。

## 必须先做

实现前先创建 task 和 plan，并说明为什么现有 `lru`、`random_lru`、`leaf_aware_lru` 或 `ttl` 不够。

## 步骤

1. 阅读 [../../handbook/feature_development.md](../../handbook/feature_development.md)。
2. 创建 `.agent/tasks/YYYY-MM-DD-short-title.md` 和 `.agent/plans/YYYY-MM-DD-short-title.md`。
3. 在 `config/types.h` 增加 policy enum，并在 `config/types.cc` 增加字符串转换。
4. 在 `eviction_policy/` 下实现策略文件，继承现有 policy interface。
5. 在 `eviction_policy/policy_factory.*` 中注册策略。
6. 通过现有 `eviction_policy_params` 结构或文档化扩展添加策略参数解析。
7. 添加测试：
   - 策略排序 / 选择行为
   - 容量压力下的行为
   - 非法 config 行为
8. 更新文档：
   - [../../../docs/strategy_config.md](../../../docs/strategy_config.md)
   - [../../handbook/config_decision_guide.md](../../handbook/config_decision_guide.md)
   - 如果工作流变化，更新本 skill

## 校验

- 新策略可通过 config 字符串选择。
- 既有策略仍能解析并通过测试。
- 命中率统计仍以 token 为口径。

## 回复内容

报告：

- policy 名称和 config 字符串
- task 路径
- plan 路径
- config 参数
- 执行的测试
- 是否适合开源 optimizer 仿真 PR
