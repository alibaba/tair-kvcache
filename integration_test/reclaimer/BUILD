package(default_visibility = ["//integration_test/reclaimer:__subpackages__"])

py_test(
    name = "reclaiming_test",
    srcs = ["reclaiming_test.py"],
    tags = ["no-remote-exec"],
    data = [
        "//kv_cache_manager:kv_cache_manager_bin",
    ],
    deps = [
        "//integration_test/testlib:test_base",
        "//integration_test/admin_service:http_interface_test",
        "//integration_test/meta_service:http_interface_test",
    ],
)
