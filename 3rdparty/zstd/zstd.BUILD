package(default_visibility = ["//visibility:public"])

licenses(["notice"])

cc_library(
    name = "zstd",
    srcs = glob([
        "lib/common/*.c",
        "lib/compress/*.c",
        "lib/decompress/*.c",
        "lib/dictBuilder/*.c",
    ]),
    hdrs = glob(["lib/**/*.h"]),
    includes = ["lib"],
    copts = ["-DZSTD_DISABLE_ASM"],
)