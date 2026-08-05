cc_test(
    name = "MooncakeBackendTest",
    srcs = [
        "mooncake_backend_test.cc",
    ],
    copts = ["-fno-access-control"],
    data = [],
    tags = ["manual"],
    target_compatible_with = select({
        "//kv_cache_manager/data_storage:enable_mooncake": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = [
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/data_storage",
    ],
)
