load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive", "http_file")

# Sanitize a dependency so that it works correctly from code that includes
# TensorFlow as a submodule.
def clean_dep(dep):
    return str(Label(dep))

def git_deps():
    # NOTE: rules_cc, rules_python, com_google_googletest, com_google_absl,
    # com_google_protobuf, grpc are managed by MODULE.bazel (bzlmod).

    http_archive(
        name = "zlib_archive",
        build_file = clean_dep("//3rdparty/zlib:zlib.BUILD"),
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
        patches = ["//3rdparty/rapidjson:0001-document_h.patch"],
        build_file = clean_dep("//3rdparty/rapidjson:rapidjson.BUILD"),
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
            "//patches/havenask:havenask.patch",
            "//patches/havenask:anet.patch",
            "//patches/havenask:0001-fix-PrometheusSink-need-header.patch",
        ],
        build_file = clean_dep("//3rdparty/kmonitor:kmonitor.BUILD"),
    )

    http_archive(
        name = "nacos_sdk_cpp",
        sha256 = "7c020f763b9af9706e84da42250146eb84bfd359c7286f7c1e1aa9a5be42d72d",
        strip_prefix = "nacos-sdk-cpp-2b4104d2524776dff236a228ad2abff4676fb916",
        type = "tar.gz",
        urls = [
            "https://codeload.github.com/nacos-group/nacos-sdk-cpp/tar.gz/2b4104d2524776dff236a228ad2abff4676fb916",
            "https://github.com/nacos-group/nacos-sdk-cpp/archive/2b4104d2524776dff236a228ad2abff4676fb916.tar.gz",
        ],
        patches = [
            "//patches/nacos_sdk_cpp:nacos-compile.patch",
        ],
        build_file = clean_dep("//3rdparty/nacos_sdk_cpp:nacos_sdk_cpp.BUILD"),
    )

    http_archive(
        name = "yaml-cpp",
        sha256 = "e39f54bd2927692603378e373009e56b4891701cee8af7c27370c36978a43ffa",
        strip_prefix = "yaml-cpp-9a3624205e8774953ef18f57067b3426c1c5ada6",
        type = "tar.gz",
        urls = [
            "https://codeload.github.com/jbeder/yaml-cpp/tar.gz/9a3624205e8774953ef18f57067b3426c1c5ada6",
            "https://github.com/jbeder/yaml-cpp/archive/9a3624205e8774953ef18f57067b3426c1c5ada6.tar.gz",
        ],
        build_file = clean_dep("//3rdparty/yaml-cpp:BUILD"),
    )

    http_archive(
        name = "mooncake",
        sha256 = "eb3f3f53d873d441cbd04cebd76506b56d7526c805da25b8525ed54abc2a06ba",
        strip_prefix = "Mooncake-211b75742b6d1fee739ad9a486f2ae9ce2695847",
        type = "tar.gz",
        urls = [
            "https://codeload.github.com/openanolis/Mooncake/tar.gz/211b75742b6d1fee739ad9a486f2ae9ce2695847",
            "https://github.com/openanolis/Mooncake/archive/211b75742b6d1fee739ad9a486f2ae9ce2695847.tar.gz",
        ],
        build_file = clean_dep("//3rdparty/mooncake:mooncake.BUILD"),
        patches = [
            clean_dep("//patches/mooncake:0001-fix-spinlock-gcc10-compat.patch"),
            clean_dep("//patches/mooncake:0002-fix-missing-gflags-include.patch"),
            clean_dep("//patches/mooncake:0003-fix-linux-memfd-header-compat.patch"),
        ],
        patch_args = ["-p1"],
    )

    http_archive(
        name = "curl",
        build_file = clean_dep("//3rdparty/curl:curl.BUILD"),
        sha256 = "e9c37986337743f37fd14fe8737f246e97aec94b39d1b71e8a5973f72a9fc4f5",
        strip_prefix = "curl-7.60.0",
        urls = [
            "https://github.com/curl/curl/releases/download/curl-7_60_0/curl-7.60.0.tar.gz",
            "https://mirror.bazel.build/curl.haxx.se/download/curl-7.60.0.tar.gz",
            "https://curl.haxx.se/download/curl-7.60.0.tar.gz",
        ],
    )

    http_archive(
        name = "boringssl",
        sha256 = "1188e29000013ed6517168600fc35a010d58c5d321846d6a6dfee74e4c788b45",
        strip_prefix = "boringssl-7f634429a04abc48e2eb041c81c5235816c96514",
        type = "tar.gz",
        urls = [
            "https://codeload.github.com/google/boringssl/tar.gz/7f634429a04abc48e2eb041c81c5235816c96514",
            "https://github.com/google/boringssl/archive/7f634429a04abc48e2eb041c81c5235816c96514.tar.gz",
            "https://mirror.bazel.build/github.com/google/boringssl/archive/7f634429a04abc48e2eb041c81c5235816c96514.tar.gz",
        ],
    )

    # NOTE: native.bind() calls removed — incompatible with bzlmod.
    # Protobuf/gRPC bindings are resolved directly via bzlmod deps.
