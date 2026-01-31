#!/usr/bin/env python3
# 解决使用python测试pybind11的extension时，无法通过bazel配置suppressions的问题。
# 背景：
# 1. LSAN_OPTIONS的suppressions如果给相对地址，则会以当前程序文件（python3）的目录为base目录进行查找。
# 2. bazel没办法获取项目文件的绝对地址并放入环境变量。
# 解决：包一层python，让python调用python来解决问题。
import os
import sys

if len(sys.argv) < 3:
    sys.stderr.write("Usage: asan_runner.py <suppressions.runfiles_path> <test_script> [args...]\n")
    sys.exit(1)

suppressions_runfiles_path = sys.argv[1]  # e.g. "ws/path/to/lsan_suppressions.txt"
test_script = sys.argv[2]
rest_args = sys.argv[3:]

env = os.environ.copy()
if "ASAN_OPTIONS"  in os.environ:
    # --- 构造绝对路径 ---
    suppressions_abs = os.path.join(os.environ.get("TEST_SRCDIR"), os.environ.get("TEST_WORKSPACE"), suppressions_runfiles_path)
    # --- 构造新环境 ---
    env["LD_PRELOAD"] = "libasan.so libasan.so.6"
    env["LSAN_OPTIONS"] = f"suppressions={suppressions_abs}"

# --- 构造新命令：用系统 python 启动测试脚本 ---
# 注意：用 sys.executable 确保与当前 Python 一致（避免 venv 问题）
cmd = [sys.executable, test_script] + rest_args

try:
    # exec 替换当前进程
    # stdout/stderr/in 继承自当前进程，完全透传
    os.execvpe(cmd[0], cmd, env)
except OSError as e:
    sys.stderr.write(f"ERROR: failed to exec {cmd[0]}: {e}\n")
    sys.exit(127)
