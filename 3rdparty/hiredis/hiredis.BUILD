genrule(
    name = "hiredis_pkg_include_hiredis_h",
    srcs = ["hiredis.h"],
    outs = ["pkg_include/hiredis/hiredis.h"],
    cmd = "mkdir -p $$(dirname $@) && cp $< $@",
)

cc_library(
    name = "hiredis",
    srcs = glob(["*.c"], exclude=["ssl.c"]),
    hdrs = glob(["*.h"]) + [":hiredis_pkg_include_hiredis_h"],
    # "." keeps #include <hiredis.h> working; pkg_include matches distro layout for
    # #include <hiredis/hiredis.h> (e.g. mooncake-store redis HA code).
    includes = [
        ".",
        "pkg_include",
    ],
    copts = ["-Wno-unused-function"],
    visibility = ["//visibility:public"],
)
