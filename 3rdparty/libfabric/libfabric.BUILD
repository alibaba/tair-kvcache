# libfabric shipped by the AWS EFA installer (default /opt/amazon/efa).
# Only referenced when building with --config=efa; point elsewhere with
# --override_repository=libfabric=/path/to/libfabric/prefix.
cc_library(
    name = "libfabric",
    hdrs = glob(["include/rdma/**/*.h"]),
    includes = ["include"],
    linkopts = [
        "-L/opt/amazon/efa/lib",
        "-L/opt/amazon/efa/lib64",
        "-Wl,-rpath,/opt/amazon/efa/lib",
        "-Wl,-rpath,/opt/amazon/efa/lib64",
        "-lfabric",
    ],
    visibility = ["//visibility:public"],
)
