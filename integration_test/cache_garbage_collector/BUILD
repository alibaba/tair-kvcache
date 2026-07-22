package(default_visibility = ["//integration_test/cache_garbage_collector:__subpackages__"])

py_test(
    name = "cache_garbage_collector_test",
    srcs = ["cache_garbage_collector_test.py"],
    data = [
        "//kv_cache_manager:kv_cache_manager_bin",
    ],
    tags = ["no-remote-exec"],
    deps = [
        "//integration_test/admin_service:http_interface_test",
        "//integration_test/meta_service:http_interface_test",
        "//integration_test/testlib:test_base",
    ],
)
