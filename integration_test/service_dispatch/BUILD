package(default_visibility = ["//integration_test:__subpackages__"])

config_setting(
    name = "asan_enabled",
    values = {"copt": "-DADDRESS_SANITIZER"},
)

py_test(
    name = "service_dispatch_smoke",
    srcs = ["service_dispatch_smoke.py"],
    args = [
        "--binary",
        "$(location //kv_cache_manager:kv_cache_manager_bin)",
    ] + select({
        # LeakSanitizer scans the heap after the service has already stopped.
        ":asan_enabled": ["--exit-timeout=60"],
        "//conditions:default": [],
    }),
    data = ["//kv_cache_manager:kv_cache_manager_bin"],
    tags = ["no-remote-exec"],
    deps = [
        "//kv_cache_manager/protocol/protobuf:admin_service_py_proto",
        "//kv_cache_manager/protocol/protobuf:meta_service_py_proto",
        "@pip_cpu//grpcio",
        "@pip_cpu//protobuf",
    ],
)
