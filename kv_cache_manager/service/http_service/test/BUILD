package(default_visibility = ["//visibility:private"])

cc_test(
    name = "AuthUtilTest",
    srcs = [
        "auth_util_test.cc",
    ],
    copts = ["-std=c++20"],
    deps = [
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/service/http_service:auth",
    ],
)

cc_test(
    name = "StaticBearerTokenVerifierTest",
    srcs = [
        "static_bearer_token_verifier_test.cc",
    ],
    copts = ["-std=c++20"],
    deps = [
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/service/http_service:auth",
    ],
)
