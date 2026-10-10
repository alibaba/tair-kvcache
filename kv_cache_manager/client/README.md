## 目录结构
* include: 头文件
* src: 源文件

## 元数据接口

`MetaClient::StartWrite` 增加末尾参数 `min_replica_count`，默认 0，由服务端按 1 解释；负数会返回参数错误。
`MetaClient` 新增 `GetCacheLocationsByBackend` 和 `GetHostCacheState`，经 `GrpcStub` 调用真实 MetaService。
前者每次选择一个 backend，固定 `QT_BATCH_GET`，返回与输入 key 对齐的嵌套位置数组，保留 miss／mask 的空项及 storage type、spec size、URI；spec-name 筛选若非空，须每个 key 一个名称。
后者返回主机标识、本地可复用长度、P2P 拉取块数和最终命中长度；查询模式由服务端限制为 prefix／Mamba。

新增或变更虚接口必须重新产出匹配的头文件和动态库，不能将新头文件覆盖到旧 RPM。
