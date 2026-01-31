load("@rules_python//python:py_test.bzl", "py_test")

package(default_visibility = ["//visibility:private"])

cc_test(
    name = "TransferClientTest",
    srcs = [
        "transfer_client_test.cc",
    ],
    copts = ["-fno-access-control"],
    data = [],
    deps = [
        "//kv_cache_manager/client",
        "//kv_cache_manager/common:unittest",
    ],
)

py_binary(
    name = "asan_runner",
    srcs = ["asan_runner.py"],
    visibility = ["//visibility:private"],
)

py_test(
    name = "transfer_client_py_test",
    size = "small",
    # 主入口改为 runner
    main = "asan_runner.py",
    srcs = [
        "asan_runner.py",
        "transfer_client_py_test.py",
    ],
    deps = [
        "//kv_cache_manager/client/pybind:kvcm_py_client_lib",
    ],
    data = ["lsan_suppressions.txt"],
    args = [
        "$(location lsan_suppressions.txt)",     # → argv[1]
        "$(location transfer_client_py_test.py)", # → argv[2]
        # 后续可加 --test_arg="--verbose" 等，会传给真实测试
    ],
)