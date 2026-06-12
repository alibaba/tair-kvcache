package(default_visibility = ["//visibility:public"])

config_setting(
    name = "using_tpu",
    values = {"define": "using_tpu=true"},
)

cc_library(
    name = "jax_utils",
    hdrs = select({
        ":using_tpu": ["jax_utils.h"],
        "//conditions:default": [],
    }),
    features = ["-use_header_modules"],
    deps = [
        "@pybind11//:pybind11",
    ] + select({
        ":using_tpu": [
            "@jax//jaxlib:py_client",
            "@xla//xla/pjrt:pjrt_client",
            "@xla//xla/pjrt/c:pjrt_c_api_hdrs",
            "@xla//xla/pjrt/c:pjrt_c_api_raw_buffer_extension_hdrs",
            "@xla//xla/pjrt/c_api_client:pjrt_c_api_client",
            "@xla//xla/python/ifrt",
            "@xla//xla/python/pjrt_ifrt",
        ],
        "//conditions:default": [],
    }),
)
