cc_library(
    name = "hiredis",
    srcs = glob(["*.c"], exclude=["ssl.c"]),
    hdrs = glob(["*.h"]),
    includes = ["."],
    copts = ["-Wno-unused-function"],
    visibility = ["//visibility:public"],
)
