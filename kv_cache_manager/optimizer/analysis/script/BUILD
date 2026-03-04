load("@rules_python//python:py_binary.bzl", "py_binary")
load("@rules_python//python:py_library.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

PYBIND_DEP = "//kv_cache_manager/optimizer/pybind:kvcm_py_optimizer_lib"

# =========================================================================
# Library 层：imports=["."] 使 script/ 成为包根，run/ 内可直接
# from utils.csv_loader import ... / from plot.hit_rate_plot import ...
# =========================================================================

py_library(
    name = "utils_lib",
    srcs = [
        "utils/__init__.py",
        "utils/csv_loader.py",
        "utils/optimizer_runner.py",
        "utils/plot_utils.py",
    ],
    imports = ["."],
    deps = [PYBIND_DEP],
)

py_library(
    name = "plot_lib",
    srcs = [
        "plot/__init__.py",
        "plot/hit_rate_plot.py",
        "plot/lifecycle_plot.py",
        "plot/radix_tree_plot.py",
    ],
    imports = ["."],
)

py_library(
    name = "analysis_lib",
    srcs = [
        "analysis/__init__.py",
        "analysis/lifecycle_analysis.py",
    ],
    imports = ["."],
)

# =========================================================================
# 入口二进制
# =========================================================================

# 单次 optimizer 运行 + 命中率时序图
py_binary(
    name = "optimizer_run",
    srcs = ["run/optimizer_run.py"],
    main = "run/optimizer_run.py",
    deps = [":utils_lib", ":plot_lib"],
)

# Pareto 曲线分析（单策略 / 多策略自动切换）
py_binary(
    name = "tradeoff",
    srcs = ["run/tradeoff.py"],
    main = "run/tradeoff.py",
    deps = [":utils_lib", ":plot_lib"],
)

# RadixTree 导出 + 可视化
py_binary(
    name = "export_tree",
    srcs = ["run/export_tree.py"],
    main = "run/export_tree.py",
    deps = [":utils_lib", ":plot_lib"],
)

# Block Lifecycle 分析（统计 + CDF + 直方图）
py_binary(
    name = "analyze_lifecycle",
    srcs = ["run/analyze_lifecycle.py"],
    main = "run/analyze_lifecycle.py",
    deps = [":analysis_lib", ":plot_lib"],
)
