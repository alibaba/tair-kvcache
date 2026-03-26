# zstd compiled from source - no system dependency required.
cc_library(
    name = "zstd",
    srcs = glob([
        "lib/common/*.c",
        "lib/compress/*.c",
        "lib/decompress/*.c",
    ]),
    hdrs = glob([
        "lib/*.h",
        "lib/common/*.h",
        "lib/compress/*.h",
        "lib/decompress/*.h",
    ]),
    includes = ["lib"],
    copts = [
        "-O3",
        "-Wno-unused-variable",
        "-Wno-maybe-uninitialized",
        "-DZSTD_DISABLE_ASM",
    ],
    visibility = ["//visibility:public"],
)
