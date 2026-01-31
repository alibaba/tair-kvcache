package(default_visibility = ["//visibility:private"])

cc_library(
    name = "client_test_lib",
    testonly = True,
    srcs = [
        "client_test_base.cc",
    ],
    hdrs = [
        "client_test_base.h",
        "service_process_controller.h",
    ],
    deps = [
        "//kv_cache_manager/common:logger",
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/client",
    ],
)

cc_test(
    name = "client_scheduler_test",
    srcs = [
        "client_scheduler_test.cc",
    ],
    copts = ["-fno-access-control"],
    data = [
        "client_test_service_logger.conf",
        "//integration_test:kv_cache_manager_install",
    ],
    deps = [
        ":client_test_lib",
    ],
)

cc_test(
    name = "manager_client_test",
    srcs = [
        "manager_client_test.cc",
    ],
    copts = ["-fno-access-control"],
    data = [
        "client_test_service_logger.conf",
        "//integration_test:kv_cache_manager_install",
    ],
    deps = [
        ":client_test_lib",
    ],
)
