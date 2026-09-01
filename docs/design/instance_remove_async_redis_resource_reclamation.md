# Instance 删除后的 async_redis 资源回收设计

## 1. 背景与结论

当前 `RemoveInstance` 只删除 Registry、清理指标并异步提交全量 Trim，没有从
`MetaSearcherManager` 和 `MetaIndexerManager` 中摘除该 Instance 对应的对象。

`async_redis` 本身的关闭链路是完整的：

```text
MetaIndexer 析构
  -> MetaStorageBackendManager 析构
    -> MetaStorageBackendManager::Close
      -> MetaAsyncRedisBackend::Close
        -> 停止并唤醒消费者
        -> join consumer_threads_
        -> 关闭 Redis client，释放队列和 client pool
```

问题不在 `MetaAsyncRedisBackend::Close`，而在 Instance 删除后仍有以下强引用长期存活：

```text
MetaSearcherManager::meta_searcher_map_
  -> MetaSearcher
    -> shared_ptr<MetaIndexer>

MetaIndexerManager::meta_indexers_
  -> shared_ptr<MetaIndexer>
```

因此 `MetaIndexer` 不会析构，属于它的 recover 线程、async_redis 消费线程、队列和
Redis client 也不会回收。进程级 `DoCleanup` 能释放这些对象，但不能解决独立 Instance
在线删除后的资源泄漏。

## 2. 目标与非目标

### 2.1 目标

- `RemoveInstance` 返回前，停止该 Instance 的新元数据访问，等待已进入的访问和删除任务退出。
- 摘除该 Instance 的 `MetaSearcher`、`MetaIndexer`，复用已有析构/`Close` 链路回收线程和后端资源。
- 保持全量 Trim 的扫描、状态流转、物理删除和任务监督链路不变。
- Instance 级回收不持有 `MetaSearcherManager` / `MetaIndexerManager` 的 map 锁执行等待、持久化或
  线程 `join`。
- 不在 backend 层增加重复 `Close`、重复状态判断或额外兜底线程。

### 2.2 非目标

- 不修改 async_redis 的队列模型、批处理、序列化和指标逻辑。
- 不改变普通 `TrimCache`、`RemoveCache` 的异步接口语义。
- 不引入新的 Instance 生命周期线程、定时扫描或 retired 对象容器。
- 不处理 `RemoveInstance` 之前已经存在的迁移语义；继续复用现有
  `MigrationManager` draining/cancel 流程。

## 3. 约束与关键风险

### 3.1 不能只删除 `MetaIndexerManager` 条目

`MetaSearcher` 持有 `shared_ptr<MetaIndexer>`。仅删除 indexer map 条目不会触发析构，
async_redis 线程仍然存活。

### 3.2 不能在现有 raw pointer 接口下直接删除 `MetaSearcher`

改造前 `GetMetaSearcher` 返回裸指针。请求取得指针后 manager 锁已经释放，若
`RemoveInstance` 并发 erase，会产生 use-after-free。需要让调用方在本次操作期间持有
`shared_ptr<MetaSearcher>`，再由删除流程等待其他 owner 退出。

### 3.3 不能在异步物理删除执行前销毁 `MetaIndexer`

`TrimCache` 的准入阶段同步执行 `CAS -> CLS_DELETING` 和 `Sync`，物理删除随后在
`SchedulePlanExecutor` worker 中异步执行。改造前 worker 再按 `instance_id` 查询 indexer；
若 manager 已摘除条目，任务会返回 `EC_NOENT`，留下未完成删除。

准入阶段已经取得并校验过 `shared_ptr<MetaIndexer>`，执行阶段应直接复用并持有这份
owner，不再重复查询或判空。

### 3.4 删除返回与同 ID 重新注册不能发生 ABA 重叠

如果 `RemoveInstance` 只触发后台释放就立即返回，同一个 `instance_id` 可在旧请求或旧
Trim 任务完成前重新注册，旧任务可能继续操作相同的持久化命名空间。删除流程必须在返回前
等待旧 `MetaSearcher` 和 `MetaIndexer` 的其他 owner 释放。同时，Registry 删除后到资源释放完成前
存在一个窗口，必须让 `RegisterInstance` 与 `RemoveInstance` 互斥，否则注册可能复用即将
被摘除的旧对象，形成 Registry 已恢复但 Searcher/Indexer 缺失的半注册状态。
该互斥直接复用现有 `MetricsLifecycle` 读写锁，不新增第二套 Instance 生命周期锁。

## 4. 方案

### 4.1 MetaSearcher 改为共享所有权

`MetaSearcherManager` 的 map value 从 `unique_ptr` 改为 `shared_ptr`，
`TryCreateMetaSearcher`、`GetMetaSearcher` 和 `GetMetaSearcherUnsafe` 均返回
`shared_ptr<MetaSearcher>`。

所有调用方在局部变量中持有返回值，既保持原调用链，也使 manager 条目被摘除后，已进入的
请求仍可安全结束。调用点只做机械类型调整，不增加下层重复校验。

### 4.2 manager 提供 Instance 级所有权摘除

为两个 manager 提供：

```cpp
std::shared_ptr<MetaSearcher> ExtractMetaSearcher(const std::string &instance_id);
std::shared_ptr<MetaIndexer> ExtractMetaIndexer(const std::string &instance_id);
```

实现使用容器 `extract(instance_id)`：

1. 仅在 manager 写锁内摘除 node；
2. 在锁外释放 node 中的 `shared_ptr`；
3. 将摘除的 owner 返回给 `RemoveInstance` 做生命周期栅栏。

Registry 删除已经完成 Instance 存在性校验，Extract 对不存在条目保持空 owner 语义，不再
重复查询 Registry、重复记录错误或增加分支式校验。

### 4.3 已准入删除任务持有 MetaIndexer

`SchedulePlanExecutor` 在删除准入阶段取得 `MetaIndexer` 后，将同一份 `shared_ptr` 传给
物理删除任务：

- `SubmitMetaDelete` 的执行 lambda 捕获已取得的 indexer；
- 通用 `PrepareDeleteTaskImpl` 把 indexer 放入 `LocationDelAdmissionResult`，
  `RunDeleteAdmission` 再移动到执行 lambda；
- `DoLocationDelTask` 接收该 indexer，删除执行阶段不再按 Instance 二次查询和判空。

这样 manager 摘除条目只会阻止新任务准入，不会破坏已经完成准入的删除任务；任务 owner
同时成为精确的完成信号，无需新增 pending 计数、future 聚合或回调链路。

### 4.4 RemoveInstance 回收顺序

```text
获取 `MetricsLifecycle` 独占锁
  -> 现有 migration draining / cancel / bounded wait
  -> RegistryManager::RemoveInstance
  -> ExtractMetaSearcher（阻止新的 MetaSearcher 获取）
  -> 等待其他 MetaSearcher owner 退出并释放摘除 owner
  -> InvalidateInstanceMetrics
  -> 复用 TrimCache 提交全量删除
  -> ExtractMetaIndexer（阻止新的删除任务准入）
  -> 等待已准入任务和其他 MetaIndexer owner 退出
  -> 释放摘除 owner
     -> MetaIndexer / backend manager 析构
     -> async_redis Close、drain、join
  -> 返回 TrimCache 的原始结果
```

`RemoveInstance` 在 CacheManager 层持有 `MetricsLifecycle` 独占锁，`RegisterInstance` 在同一层
持有共享锁，从而阻止删除与重新注册重叠。锁下沉后 AdminService 不再重复加锁，MetaService、
AdminService 和直接调用 CacheManager 的路径都复用同一约束，也无需新增 mutex 数组或动态 mutex map。
该锁是全局锁，一个 Instance 删除期间其他 Instance 的注册也会等待；但注册和删除都是低频控制面
操作，普通元数据请求不获取这把生命周期锁，且删除原本就会独占它以保证指标清理，因此接受该
取舍以减少锁体系和代码量。

等待只发生在低频管理操作中。map 已先摘除，因此 owner 数只会下降；不在热路径增加锁、计数
或校验。等待函数复用一份小型模板实现，以短间隔 sleep 避免忙等。

即使 `TrimCache` 返回错误，也必须完成 `MetaIndexer` 摘除和 owner drain，避免 Registry 已删除
但资源永久留在 manager 中；清理完成后仍向上返回原 Trim 错误，不吞掉失败。

## 5. 并发与锁分析

- manager 锁只保护 map 查找、创建和摘除；等待和对象析构均在锁外执行。
- Register 持有共享生命周期锁、Remove 持有独占生命周期锁，阻止旧资源关闭前重新注册；共享锁不
  额外串行化并发注册。
- Registry 先删除，Searcher map 随后摘除；新请求无法再获取该 Instance 的 Searcher。
- 已进入请求持有 `shared_ptr<MetaSearcher>`，不会 UAF；请求结束后 owner 自动下降。
- Searcher owner 清空后再 Trim，避免并发元数据写入发生在全量扫描之后。
- Indexer map 在 Trim 完成同步准入后摘除；已准入物理删除任务持有 indexer 并继续执行。
- Indexer owner 只剩删除线程本地 owner 时才触发析构，因此 `RemoveInstance` 返回时旧后端已关闭，
  同 ID 重新注册不会与旧任务/旧 async_redis 消费线程重叠。
- 不持有 `MetaSearcherManager` 或 `MetaIndexerManager` 锁执行 Redis flush、Close 或线程 join；
  owner 等待与 backend 关闭均发生在 map 摘除之后，不占用 manager map 锁。

## 6. 设计自审与取舍

### 6.1 已排除方案

- **只补 `DeleteMetaIndexer` 并在 RemoveInstance 中调用**：Searcher 仍持有 indexer，不能回收。
- **直接 erase Searcher/Indexer 后返回**：会产生裸指针 UAF、异步删除 `EC_NOENT` 和同 ID ABA。
- **等待 ReclaimerTaskSupervisor 中 Trim future**：只能覆盖本次 Trim，不能覆盖已进入请求和其他
  已准入任务，还需要额外 pending map、通知和竞态处理。
- **在 async_redis 增加按 Instance Stop 接口**：绕过 owner 析构会让仍在使用 indexer 的请求访问
  已关闭 backend，且与现有幂等 `Close` 重复。
- **新增后台回收线程或 retired map**：增加生命周期状态和长期维护成本，现有 `shared_ptr` owner
  已能表达准确的任务存活关系。

### 6.2 自审后保留的必要判断

- `RemoveInstance` 仍只在 Registry 删除成功后开始对象摘除，避免删除失败破坏有效 Instance。
- Trim 的错误码仍按原接口返回；资源摘除不依赖 Trim 成功。
- `DoLocationDelTask` 不再判空 indexer：上层准入已完成同一对象的获取和校验。
- Extract 不重复做 Registry 存在性判断；空 owner 可自然通过 drain helper。

## 7. 测试方案

1. 扩展 `CacheManagerTest.TestRemoveInstance`：
   - 删除前保存 `weak_ptr<MetaIndexer>`；
   - 删除后确认 Searcher/Indexer manager 均无该 Instance；
   - 确认 weak owner 已过期，证明 `MetaIndexer` 及 backend manager 已析构；
   - 删除后的元数据访问返回 `EC_INSTANCE_NOT_EXIST`，不再访问残留 Searcher。
2. 增加同 ID 并发删除/重注册回归：让删除阻塞在旧 Searcher owner 上，确认重注册在共享生命周期锁
   上等待，并在旧 Indexer 析构后创建全新的 Searcher/Indexer。
3. 扩展 `SchedulePlanExecutor` 测试：提交带 delay 的删除任务，在执行前摘除 manager owner，确认任务
   仍使用准入时持有的 indexer 成功完成。
4. 复跑 manager、meta async_redis/backend lifecycle 相关 UT；现有
   `MetaAsyncRedisBackendTest.TestOpenAndClose` 继续验证 `Close` 会 join 并清空消费者线程、队列和 client。
5. 执行项目格式检查，确保 shared_ptr 接口的所有生产和测试调用点均完成类型迁移。
