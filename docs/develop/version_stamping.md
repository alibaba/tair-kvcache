# 构建版本信息（Version Stamping）

本文档介绍项目中的构建版本信息机制：如何在构建产物（Python wheel 包、C++ 二进制）中嵌入 git commit、构建时间等元数据，以及如何为新组件接入该机制。

## 原理

版本信息的注入基于 Bazel 的 [workspace status](https://bazel.build/docs/user-manual#workspace-status) 机制，整体流程如下：

```
workspace_status.sh          Bazel stamp 机制             gen_version_info.py
┌──────────────────┐    ┌──────────────────────┐    ┌──────────────────────┐
│ 执行 git 命令     │───>│ stable-status.txt    │───>│ 读取 status 文件      │
│ 输出 key-value   │    │ volatile-status.txt  │    │ 生成 .py 或 .h 文件   │
└──────────────────┘    └──────────────────────┘    └──────────────────────┘
```

### 1. workspace_status_command

`.bazelrc` 中配置了：

```
build --workspace_status_command=bazel/workspace_status.sh
```

每次 `bazel build` 时，Bazel 会执行该脚本，脚本输出 key-value 对：

| 变量名 | 示例值 | 说明 |
|--------|--------|------|
| `STABLE_GIT_COMMIT` | `2cd6c5a9` | git short commit hash (8位) |
| `STABLE_GIT_COMMIT_FULL` | `2cd6c5a9...` | git 完整 commit hash |
| `STABLE_GIT_REPO` | `git@github.com:alibaba/tair-kvcache.git` | 远程仓库地址 |
| `STABLE_KVCM_VERSION` | `0.0.1` | 语义版本号 |
| `BUILD_DATE` | `20260409` | 构建日期 |
| `BUILD_TIME` | `2026-04-09 11:38:26` | 构建时间 |

**`STABLE_` 前缀与 volatile 变量的区别**：

- `STABLE_` 前缀的变量写入 `stable-status.txt`，值变化时**会触发**依赖目标重新构建。适用于 git commit、版本号等不频繁变化的值。
- 无前缀的变量写入 `volatile-status.txt`，值变化时**不会触发**重新构建。适用于构建时间等每次都会变化的值，避免不必要的重新编译。

### 2. genrule + stamp

在 BUILD 文件中通过 `stamp = 1` 的 genrule 调用 `gen_version_info.py`，该脚本读取 Bazel 生成的 status 文件，输出格式化的版本信息文件。

- Python 组件：生成 `_version_info.py`，包含 `VERSION`、`GIT_COMMIT`、`FULL_VERSION` 等模块级变量
- C++ 组件：生成 `build_version.h`，包含 `KVCM_VERSION`、`KVCM_GIT_COMMIT`、`KVCM_FULL_VERSION` 等宏定义

### 3. py_wheel stamp

Python wheel 包的版本号通过 `py_wheel` 规则的 `stamp` 属性注入。`stamp = 1` 时，`version` 字段中的 `{VARIABLE}` 占位符会被替换为 status 文件中的实际值：

```python
py_wheel(
    name = "my_wheel",
    version = "{STABLE_KVCM_VERSION}+{BUILD_DATE}.{STABLE_GIT_COMMIT}",
    stamp = 1,
    ...
)
```

最终 wheel 内部的版本号为 `0.0.1+20260409.2cd6c5a9`（符合 [PEP 440](https://peps.python.org/pep-0440/) local version 规范）。

> **注意**：Bazel 输出路径中的文件名在分析阶段确定，仍包含字面占位符。使用 `.dist` 后缀目标（如 `bazel build :my_wheel.dist`）可在 `dist/` 目录下获得正确文件名的 wheel 文件。

## 版本号格式

最终的完整版本号格式为：

```
{STABLE_KVCM_VERSION}+{BUILD_DATE}.{STABLE_GIT_COMMIT}
```

示例：`0.0.1+20260409.2cd6c5a9`

- `0.0.1`：语义版本号，在 `bazel/workspace_status.sh` 中定义
- `20260409`：构建日期
- `2cd6c5a9`：git commit 短 hash

## 文件结构

```
bazel/
├── workspace_status.sh   # Bazel workspace status 脚本，提供 stamp 变量
├── gen_version_info.py   # 版本信息文件生成器（支持 Python / C++ 两种输出格式）
├── version.bzl           # 可复用的 Starlark 宏（version_info_py / version_info_cc）
└── BUILD                 # exports_files 声明
```

## 使用方式

### 为 Python 组件接入版本信息

1. 在 BUILD 文件中加载宏并生成版本模块：

```python
load("//bazel:version.bzl", "version_info_py")

version_info_py(name = "gen_version_info")

py_library(
    name = "my_lib",
    srcs = glob(["*.py"]) + [":gen_version_info"],
)
```

2. 在 Python 代码中导入使用：

```python
from my_package._version_info import FULL_VERSION, GIT_COMMIT, BUILD_TIME

logger.info("version: %s (commit: %s, build: %s)", FULL_VERSION, GIT_COMMIT, BUILD_TIME)
```

生成的 `_version_info.py` 包含以下变量：

| 变量名 | 类型 | 示例值 | 说明 |
|--------|------|--------|------|
| `VERSION` | str | `"0.0.1"` | 语义版本号 |
| `GIT_COMMIT` | str | `"2cd6c5a9"` | 短 commit hash |
| `GIT_COMMIT_FULL` | str | `"2cd6c5a9..."` | 完整 commit hash |
| `GIT_REPO` | str | `"git@github.com:..."` | 远程仓库地址 |
| `BUILD_DATE` | str | `"20260409"` | 构建日期 |
| `BUILD_TIME` | str | `"2026-04-09 11:38:26"` | 构建时间 |
| `FULL_VERSION` | str | `"0.0.1+20260409.2cd6c5a9"` | 完整版本号 |

### 为 C++ 组件接入版本信息

1. 在 BUILD 文件中加载宏并生成版本头文件：

```python
load("//bazel:version.bzl", "version_info_cc")

version_info_cc(name = "build_version")

cc_library(
    name = "my_lib",
    deps = [":build_version"],
    ...
)
```

2. 在 C++ 代码中包含使用：

```cpp
#include "my_package/build_version.h"

// 可用的宏定义：
// KVCM_VERSION        - "0.0.1"
// KVCM_GIT_COMMIT     - "2cd6c5a9"
// KVCM_GIT_COMMIT_FULL - "2cd6c5a9..."
// KVCM_GIT_REPO       - "git@github.com:..."
// KVCM_BUILD_DATE     - "20260409"
// KVCM_BUILD_TIME     - "2026-04-09 11:38:26"
// KVCM_FULL_VERSION   - "0.0.1+20260409.2cd6c5a9"

std::cout << "Version: " << KVCM_FULL_VERSION << std::endl;
```

### 为 py_wheel 包添加版本号

在 `py_wheel` 规则中启用 stamp：

```python
py_wheel(
    name = "my_package",
    version = "{STABLE_KVCM_VERSION}+{BUILD_DATE}.{STABLE_GIT_COMMIT}",
    stamp = 1,
    ...
)
```

构建带正确文件名的 wheel：

```bash
# 标准构建（输出文件名包含字面占位符，但 wheel 内部元数据正确）
bazel build //path/to:my_package

# 推荐：使用 .dist 目标获取正确文件名
bazel build //path/to:my_package.dist
# 输出位于 bazel-bin/path/to/my_package.dist/
# 文件名示例：my_package-0.0.1+20260409.2cd6c5a9-cp310-cp310-linux_x86_64.whl
```

## 当前接入组件

| 组件 | 类型 | 版本信息位置 |
|------|------|-------------|
| sglang connector | Python wheel + 日志 | `py_connector/common/_version_info.py`，启动时打印 |
| vllm connector | Python wheel + 日志 | `py_connector/common/_version_info.py`，启动时打印 |
| client pybind | Python wheel | wheel 包版本号 |
| optimizer pybind | Python wheel | wheel 包版本号 |
| manager (C++) | C++ 二进制 | `common/build_version.h`，`./main -v` 输出 |

## 修改版本号

语义版本号定义在 `bazel/workspace_status.sh` 中：

```bash
echo "STABLE_KVCM_VERSION 0.0.1"
```

发版时修改此值即可，所有组件会自动使用新版本号。

## 无 .git 目录时的行为

如果源码不在 git 仓库中（例如直接下载的源码包），构建流程**仍然可以正常完成**，不会失败。具体行为：

- `workspace_status.sh` 中的所有 git 命令都有 `|| echo unknown` 兜底，git 信息会退化为 `unknown`
- `STABLE_KVCM_VERSION`、`BUILD_DATE`、`BUILD_TIME` 不依赖 git，不受影响
- 最终版本号为 `0.0.1+20260409.unknown`，PEP 440 合法
- py_wheel stamp 中 `{STABLE_GIT_COMMIT}` 被替换为 `unknown`，wheel 正常构建
- C++ 宏同理，编译不受影响
