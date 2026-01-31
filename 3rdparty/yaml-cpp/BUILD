package(default_visibility = ["//visibility:public"])

cc_library(
    name = "yaml_cpp_src_contrib_headers",
    hdrs = glob([
        "src/contrib/*.h",
    ]),
    strip_include_prefix = "src/contrib",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "yaml_cpp_src_headers",
    hdrs = glob([
        "src/*.h",
        "src/**/*.h",
    ]),
    strip_include_prefix = "src",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "yaml_cpp_headers",
    hdrs = glob([
        "include/yaml-cpp/*.h",
        "include/yaml-cpp/**/*.h",
    ]),
    deps = [
        ":yaml_cpp_src_headers",
        ":yaml_cpp_src_contrib_headers",
    ],
    strip_include_prefix = "include",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "yaml_cpp",
    srcs = glob([
        "src/*.cpp",
        "src/**/*.cpp",
    ]),
    deps = [
        ":yaml_cpp_headers",
    ],
    visibility = ["//visibility:public"],
)
