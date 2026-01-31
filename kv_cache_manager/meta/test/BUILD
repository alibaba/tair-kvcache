package(default_visibility = ["//visibility:public"])

cc_library(
    name = "meta_indexer_testlib",
    srcs = [
        "meta_indexer_test_base.cc",
    ],
    hdrs = [
        "meta_indexer_test_base.h",
    ],
    copts = ["-fno-access-control"],
    deps = [
        "//kv_cache_manager/meta",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "meta_indexer_test",
    srcs = [
        "meta_indexer_test.cc",
    ],
    copts = ["-fno-access-control"],
    data = [],
    deps = [
        "//kv_cache_manager/common:unittest",
        ":meta_indexer_testlib",
    ],
)

cc_test(
    name = "meta_indexer_manager_test",
    srcs = [
        "meta_indexer_manager_test.cc",
    ],
    copts = ["-fno-access-control"],
    data = [],
    deps = [
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/meta",
    ],
)

cc_test(
    name = "meta_search_cache_test",
    srcs = [
        "meta_search_cache_test.cc",
    ],
    copts = ["-fno-access-control"],
    data = [],
    deps = [
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/meta",
    ],
)

cc_library(
    name = "meta_storage_backend_testlib",
    srcs = [
        "meta_storage_backend_test_base.cc",
    ],
    hdrs = [
        "meta_storage_backend_test_base.h",
    ],
    copts = ["-fno-access-control"],
    deps = [
        "//kv_cache_manager/meta:meta_storage_backend",
        "//kv_cache_manager/meta:meta_redis_backend",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "meta_local_backend_test",
    srcs = [
        "meta_local_backend_test.cc",
    ],
    copts = ["-fno-access-control"],
    data = [],
    deps = [
        ":meta_storage_backend_testlib",
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/meta:meta_local_backend",
    ],
)

cc_test(
    name = "meta_redis_backend_test",
    srcs = [
        "meta_redis_backend_test.cc",
    ],
    copts = ["-fno-access-control"],
    data = [],
    deps = [
        ":meta_storage_backend_testlib",
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/common/test:redis_client_testlib",
        "//kv_cache_manager/meta:meta_redis_backend",
    ],
)

cc_test(
    name = "meta_redis_backend_real_service_test",
    srcs = [
        "meta_redis_backend_real_service_test.cc",
    ],
    copts = ["-fno-access-control"],
    data = [],
    deps = [
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/meta:meta_redis_backend",
    ],
    tags = [
        "manual",
        "redis",
    ],
)

cc_test(
    name = "meta_indexer_redis_test",
    srcs = [
        "meta_indexer_redis_test.cc",
    ],
    copts = ["-fno-access-control"],
    data = [],
    deps = [
        "//kv_cache_manager/common:unittest",
        ":meta_indexer_testlib",
    ],
    tags = [
        "manual",
        "redis",
    ],
)
