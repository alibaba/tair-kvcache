# V6D max_peer_count 环境变量覆盖

## 行为

KVCM 通过进程环境变量 `KVCM_V6D_MAX_PEER_COUNT` 覆盖请求中的 `max_peer_count`。
仅作用于 ST_EVENT_REPORT_L2 的 LSS_V6D_PREFIX / LSS_V6D_COVERAGE selector。
未设置时沿用请求值；设置后完全覆盖，而非与请求值取最小值。
例如 Vineyard 请求 4，KVCM 配置 1，则按单 peer 选择。
部署时设置环境变量并重启 KVCM；撤销环境变量并重启后恢复请求配置。

## 实现

在 CacheManager::GetCacheLocationsByBackend 的 selector 处理处，复用 EnvUtil::GetEnv，
以请求值作为默认值。保留既有参数检查，不新增环境变量合法性校验、告警或配置抽象。
调用链为 MetaServiceImpl → CacheManager::GetCacheLocationsByBackend → MetaSearcher 的 V6D peer 选择。
不修改协议、Vineyard、非 V6D selector 或 peer 选择算法。

## 验证

复用现有测试 fixture 和 ScopedEnv，补充 PREFIX/COVERAGE 下覆盖为 1、覆盖为 2、
未设置时使用请求值及作用域退出后恢复请求值的测试代码，并检查非 V6D 策略不受影响。
按用户要求，不执行测试或构建；仅检查代码差异。
