load("@rules_python//python:pip.bzl", "pip_parse")

PIP_EXTRA_ARGS = [
    "--cache-dir=~/.cache/pip",
    "--index-url=https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple",
    "--verbose",
]

def pip_deps():
    pip_parse(
        name = "pip_cpu",
        requirements_lock = "//open_source/deps:requirements_lock_cpu.txt",
        extra_pip_args = PIP_EXTRA_ARGS,
        timeout = 3600,
    )
