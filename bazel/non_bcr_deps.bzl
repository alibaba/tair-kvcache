"""Module extension for non-BCR external dependencies.

These repos were originally defined in WORKSPACE but need to be available
in bzlmod mode (--noenable_workspace) for TPU builds.
"""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def _non_bcr_deps_impl(mctx):
    http_archive(
        name = "zlib_archive",
        build_file = Label("//3rdparty/zlib:zlib.BUILD"),
        strip_prefix = "zlib-1.2.11",
        type = "tar.gz",
        urls = [
            "https://codeload.github.com/madler/zlib/tar.gz/refs/tags/v1.2.11",
            "https://github.com/madler/zlib/archive/refs/tags/v1.2.11.tar.gz",
        ],
        sha256 = "629380c90a77b964d896ed37163f5c3a34f6e6d897311f1df2a7016355c45eff",
    )

    http_archive(
        name = "rapidjson",
        sha256 = "4a76453d36770c9628d7d175a2e9baccbfbd2169ced44f0cb72e86c5f5f2f7cd",
        strip_prefix = "rapidjson-f54b0e47a08782a6131cc3d60f94d038fa6e0a51",
        type = "tar.gz",
        urls = [
            "https://codeload.github.com/Tencent/rapidjson/tar.gz/f54b0e47a08782a6131cc3d60f94d038fa6e0a51",
            "https://github.com/Tencent/rapidjson/archive/f54b0e47a08782a6131cc3d60f94d038fa6e0a51.tar.gz",
        ],
        patches = [Label("//3rdparty/rapidjson:0001-document_h.patch")],
        build_file = Label("//3rdparty/rapidjson:rapidjson.BUILD"),
    )

    http_archive(
        name = "havenask",
        sha256 = "e03d63fa06095b612c5ba77e6b668dba4102ee90fdc79f7b45df545e64893b8b",
        strip_prefix = "havenask-3c973500afbd40933eb0a80cfdfb6592274377fb",
        type = "tar.gz",
        urls = [
            "https://codeload.github.com/alibaba/havenask/tar.gz/3c973500afbd40933eb0a80cfdfb6592274377fb",
            "https://github.com/alibaba/havenask/archive/3c973500afbd40933eb0a80cfdfb6592274377fb.tar.gz",
        ],
        patches = [
            Label("//patches/havenask:havenask.patch"),
            Label("//patches/havenask:anet.patch"),
            Label("//patches/havenask:0001-fix-PrometheusSink-need-header.patch"),
        ],
        build_file = Label("//3rdparty/kmonitor:kmonitor.BUILD"),
    )

non_bcr_deps = module_extension(
    implementation = _non_bcr_deps_impl,
)
