load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive", "http_file")

# Sanitize a dependency so that it works correctly from code that includes
# TensorFlow as a submodule.
def clean_dep(dep):
    return str(Label(dep))

def git_deps():
    http_archive(
        name = "rules_cc",
        sha256 = "b87996d308549fc3933f57a786004ef65b44b83fd63f1b0303a4bbc3fd26bbaf",
        strip_prefix = "rules_cc-1477dbab59b401daa94acedbeaefe79bf9112167",
        type = "tar.gz",
        urls = ["https://codeload.github.com/bazelbuild/rules_cc/tar.gz/1477dbab59b401daa94acedbeaefe79bf9112167"],
    )

    http_archive(
        name = "rules_python",
        sha256 = "3d6fe72f1a056b3462f02afba5049210acbaec131087fb19082fa6792198a9fa",
        strip_prefix = "rules_python-084b877c98b580839ceab2b071b02fc6768f3de6",
        type = "tar.gz",
        urls = ["https://codeload.github.com/bazelbuild/rules_python/tar.gz/084b877c98b580839ceab2b071b02fc6768f3de6"],
        patches = [
            "//patches/rules_python:0001-add-extra-data.patch",
            "//patches/rules_python:0002-remove-import-from-rules_cc.patch",
            "//patches/rules_python:0001-xx.patch",
        ],
    )
    http_archive(
        name = "com_google_googletest",
        sha256 = "7ff5db23de232a39cbb5c9f5143c355885e30ac596161a6b9fc50c4538bfbf01",
        strip_prefix = "googletest-f8d7d77c06936315286eb55f8de22cd23c188571",
        type = "tar.gz",
        urls = ["https://codeload.github.com/google/googletest/tar.gz/f8d7d77c06936315286eb55f8de22cd23c188571"],
    )

    http_archive(
        name = "com_github_nanopb_nanopb",
        sha256 = "8bbbb1e78d4ddb0a1919276924ab10d11b631df48b657d960e0c795a25515735",
        build_file = "@grpc//third_party:nanopb.BUILD",
        strip_prefix = "nanopb-f8ac463766281625ad710900479130c7fcb4d63b",
        urls = [
            "http://github.com/nanopb/nanopb/archive/f8ac463766281625ad710900479130c7fcb4d63b.tar.gz",
        ],
    )

    http_archive(
        name = "six_archive",
        build_file = clean_dep("//3rdparty/six:six.BUILD"),
        sha256 = "105f8d68616f8248e24bf0e9372ef04d3cc10104f1980f54d57b2ce73a5ad56a",
        strip_prefix = "six-1.10.0",
        urls = [
            "http://mirror.bazel.build/pypi.python.org/packages/source/s/six/six-1.10.0.tar.gz",
            "http://pypi.python.org/packages/source/s/six/six-1.10.0.tar.gz",
        ],
    )

    http_archive(
        name = "zlib_archive",
        build_file = clean_dep("//3rdparty/zlib:zlib.BUILD"),
        strip_prefix = "zlib-1.2.11",
        urls = [
            "https://www.zlib.net/fossils/zlib-1.2.11.tar.gz",
        ],
        sha256 = "c3e5e9fdd5004dcb542feda5ee4f0ff0744628baf8ed2dd5d66f8ca1197cb1a1",
    )

    http_archive(
        name = "com_google_absl",
        sha256 = "62c27e7a633e965a2f40ff16b487c3b778eae440bab64cad83b34ef1cbe3aa93",
        strip_prefix = "abseil-cpp-6f9d96a1f41439ac172ee2ef7ccd8edf0e5d068c",
        type = "tar.gz",
        urls = ["https://codeload.github.com/abseil/abseil-cpp/tar.gz/6f9d96a1f41439ac172ee2ef7ccd8edf0e5d068c"],
        patch_cmds = [
            "sed -i -e 's/^#define ABSL_OPTION_USE_STD_STRING_VIEW 2/#define ABSL_OPTION_USE_STD_STRING_VIEW 0/' 'absl/base/options.h'",
            "sed 's$@bazel_tools//platforms:(linux|osx|windows|android|freebsd|ios|os)$@platforms//os:\\1$' -E -i absl/BUILD.bazel",
            "sed 's$@bazel_tools//platforms:(cpu|x86_32|x86_64|ppc|arm|aarch64|s390x)$@platforms//cpu:\\1$' -i -E absl/BUILD.bazel",
            "sed 's$@bazel_tools//platforms:(linux|osx|windows|android|freebsd|ios|os)$@platforms//os:\\1$' -E -i absl/time/internal/cctz/BUILD.bazel",
            "sed 's$@bazel_tools//platforms:(cpu|x86_32|x86_64|ppc|arm|aarch64|s390x)$@platforms//cpu:\\1$' -i -E absl/time/internal/cctz/BUILD.bazel",
        ],
    )

    native.local_repository(
        name = "com_google_protobuf",
        path = "3rdparty/protobuf",
    )

    http_archive(
        name = "grpc",
        sha256 = "ddd5c9c42bc609108c2e9494e9cfa34ea42d0efd0eb4b183db8a4124dabdc1c2",
        strip_prefix = "grpc-109c570727c3089fef655edcdd0dd02cc5958010",
        type = "tar.gz",
        urls = ["https://codeload.github.com/grpc/grpc/tar.gz/109c570727c3089fef655edcdd0dd02cc5958010"],
        patches = ["//patches/grpc:0001-Rename-gettid-functions.patch"],
    )

    http_archive(
        name = "rapidjson",
        sha256 = "4a76453d36770c9628d7d175a2e9baccbfbd2169ced44f0cb72e86c5f5f2f7cd",
        strip_prefix = "rapidjson-f54b0e47a08782a6131cc3d60f94d038fa6e0a51",
        type = "tar.gz",
        urls = ["https://codeload.github.com/Tencent/rapidjson/tar.gz/f54b0e47a08782a6131cc3d60f94d038fa6e0a51"],
        patches = ["//3rdparty/rapidjson:0001-document_h.patch"],
        build_file = clean_dep("//3rdparty/rapidjson:rapidjson.BUILD"),
    )

    http_archive(
        name = "havenask",
        sha256 = "e03d63fa06095b612c5ba77e6b668dba4102ee90fdc79f7b45df545e64893b8b",
        strip_prefix = "havenask-3c973500afbd40933eb0a80cfdfb6592274377fb",
        type = "tar.gz",
        urls = ["https://codeload.github.com/alibaba/havenask/tar.gz/3c973500afbd40933eb0a80cfdfb6592274377fb"],
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
        urls = ["https://codeload.github.com/nacos-group/nacos-sdk-cpp/tar.gz/2b4104d2524776dff236a228ad2abff4676fb916"],
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
        urls = ["https://codeload.github.com/jbeder/yaml-cpp/tar.gz/9a3624205e8774953ef18f57067b3426c1c5ada6"],
        build_file = clean_dep("//3rdparty/yaml-cpp:BUILD"),
    )

    http_archive(
        name = "mooncake",
        sha256 = "eb3f3f53d873d441cbd04cebd76506b56d7526c805da25b8525ed54abc2a06ba",
        strip_prefix = "Mooncake-211b75742b6d1fee739ad9a486f2ae9ce2695847",
        type = "tar.gz",
        urls = ["https://codeload.github.com/openanolis/Mooncake/tar.gz/211b75742b6d1fee739ad9a486f2ae9ce2695847"],
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
        urls = [
            "https://github.com/google/boringssl/archive/7f634429a04abc48e2eb041c81c5235816c96514.tar.gz",
            "https://mirror.bazel.build/github.com/google/boringssl/archive/7f634429a04abc48e2eb041c81c5235816c96514.tar.gz",
        ],
    )

    # Needed by Protobuf
    native.bind(
        name = "grpc_cpp_plugin",
        actual = "@grpc//:grpc_cpp_plugin",
    )

    native.bind(
        name = "grpc_python_plugin",
        actual = "@grpc//:grpc_python_plugin",
    )

    # Needed by gRPC
    native.bind(
        name = "libssl",
        actual = "@boringssl//:ssl",
    )

    # Needed by gRPC
    native.bind(
        name = "nanopb",
        actual = "@com_github_nanopb_nanopb//:nanopb",
    )

    # gRPC expects //external:protobuf_clib and //external:protobuf_compiler
    # to point to Protobuf's compiler library.
    native.bind(
        name = "protobuf_clib",
        actual = "@com_google_protobuf//:protoc_lib",
    )

    # Needed by gRPC
    native.bind(
        name = "protobuf_headers",
        actual = "@com_google_protobuf//:protobuf_headers",
    )

    # # Needed by Protobuf
    native.bind(
        name = "grpc_cpp_plugin",
        actual = "@grpc//:grpc_cpp_plugin",
    )
    native.bind(
        name = "grpc_python_plugin",
        actual = "@grpc//:grpc_python_plugin",
    )

    # # Needed by Protobuf
    native.bind(
        name = "six",
        actual = "@six_archive//:six",
    )

    # Needed by gRPC
    native.bind(
        name = "zlib",
        actual = "@zlib_archive//:zlib",
    )
