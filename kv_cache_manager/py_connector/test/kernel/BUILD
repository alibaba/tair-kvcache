load("@rules_python//python:py_test.bzl", "py_test")

py_test(
    name = "test_strided_gather_scatter",
    srcs = [
        "__init__.py",
        "test_strided_gather_scatter.py",
    ],
    main = "test_strided_gather_scatter.py",
    tags = [
        "no-remote-exec",
        "gpu",  # requires 1 GPU
        "exclusive",  # GPU tests run serially to avoid CUDA contention
    ],
    deps = [
        "//kv_cache_manager/py_connector/kernel",
    ],
)
