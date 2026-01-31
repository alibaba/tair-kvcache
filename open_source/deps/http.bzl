load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive", "http_file")
load("//bazel:tf_http_archive.bzl", "tf_http_archive")

def clean_dep(dep):
    return str(Label(dep))

def http_deps():
    http_archive(
        name = "rules_pkg",
        urls = [
            "https://github.com/bazelbuild/rules_pkg/releases/download/0.6.0/rules_pkg-0.6.0.tar.gz",
            "https://mirror.bazel.build/github.com/bazelbuild/rules_pkg/releases/download/0.6.0/rules_pkg-0.6.0.tar.gz",
        ],
        sha256 = "62eeb544ff1ef41d786e329e1536c1d541bb9bcad27ae984d57f18f314018e66",
    )

    http_archive(
        name = "bazel_skylib",
        sha256 = "97e70364e9249702246c0e9444bccdc4b847bed1eb03c5a3ece4f83dfe6abc44",
        urls = ["https://github.com/bazelbuild/bazel-skylib/releases/download/1.0.2/bazel-skylib-1.0.2.tar.gz"],
        patch_cmds = [
            "sed 's$@bazel_tools//platforms:(linux|osx|windows|android|freebsd|ios|os)$@platforms//os:\\1$' -E -i ./toolchains/unittest/BUILD",
        ],
    )

    http_archive(
        name = "io_bazel_rules_closure",
        sha256 = "5b00383d08dd71f28503736db0500b6fb4dda47489ff5fc6bed42557c07c6ba9",
        strip_prefix = "rules_closure-308b05b2419edb5c8ee0471b67a40403df940149",
        urls = ["https://github.com/bazelbuild/rules_closure.git"],
    )

    # 3fs
    # TODO: provide a document about building this rpm
    http_file(
        name = "hf3fs_rpm",
        urls = ["https://github.com/alibaba/tair-kvcache/releases/download/__binary-dependency-0.0.1/pkg_3fs_hf3fs-1.2.0-1.alios7.x86_64.rpm"],
        sha256 = "d5c9ce8474f6bf2177c11c4dc36acf633b5d4763353cd70156b0a0b2d54b8316",
    )

    # mooncake
    http_archive(
        name = "mooncake",
        urls = ["https://github.com/kvcache-ai/Mooncake/archive/refs/tags/v0.3.6.tar.gz"],
        build_file = clean_dep("//3rdparty/mooncake:mooncake.BUILD"),
        patches = ["//patches/mooncake:mooncake.patch"],
        patch_args = ["-p1"],
        strip_prefix = "Mooncake-0.3.6",
        sha256 = "97d8c17dcc0d588d99361e7e2d7a8b994117038ccc5e08f00f39bdda90379ef3",
    )

    http_archive(
        name = "boost",
        urls = [
            "https://downloads.sourceforge.net/project/boost/boost/1.70.0/boost_1_70_0.tar.gz",
            "https://archives.boost.io/release/1.70.0/source/boost_1_70_0.tar.gz",
        ],
        build_file = clean_dep("//3rdparty/boost:boost.BUILD"),
        # https://github.com/boostorg/hana/issues/446
        patches = ["//patches/boost:boost.patch"],
        strip_prefix = "boost_1_70_0",
        sha256 = "882b48708d211a5f48e60b0124cf5863c1534cd544ecd0664bb534a4b5d506e9",
    )

    # gflags
    http_archive(
        name = "com_github_gflags_gflags",
        urls = ["https://github.com/gflags/gflags/archive/refs/tags/v2.2.2.tar.gz"],
        strip_prefix = "gflags-2.2.2",
        sha256 = "34af2f15cf7367513b352bdcd2493ab14ce43692d2dcd9dfc499492966c64dcf",
    )

    tf_http_archive(
        name = "jsoncpp_git",
        build_file = clean_dep("//3rdparty/jsoncpp:jsoncpp.BUILD"),
        sha256 = "c49deac9e0933bcb7044f08516861a2d560988540b23de2ac1ad443b219afdb6",
        strip_prefix = "jsoncpp-1.8.4",
        system_build_file = clean_dep("//3rdparty/jsoncpp:jsoncpp-systemlib.BUILD"),
        urls = [
            "http://github.com/open-source-parsers/jsoncpp/archive/1.8.4.tar.gz",
            "http://mirror.bazel.build/github.com/open-source-parsers/jsoncpp/archive/1.8.4.tar.gz",
        ],
    )

    # glog
    http_archive(
        name = "glog",
        urls = ["https://github.com/google/glog/archive/refs/tags/v0.7.1.tar.gz"],
        strip_prefix = "glog-0.7.1",
        sha256 = "00e4a87e87b7e7612f519a41e491f16623b12423620006f59f5688bfd8d13b08",
    )

    # yalantinglibs
    http_archive(
        name = "yalantinglibs",
        urls = ["https://github.com/alibaba/yalantinglibs/archive/refs/tags/0.5.5.tar.gz"],
        strip_prefix = "yalantinglibs-0.5.5",
        sha256 = "7962579c1414d1ade4fd22316476723d54112c919514bf1e6015a1870e5e68f7",
    )

    # curl
    http_archive(
        name = "curl",
        build_file = clean_dep("//3rdparty/curl:curl.BUILD"),
        sha256 = "e9c37986337743f37fd14fe8737f246e97aec94b39d1b71e8a5973f72a9fc4f5",
        strip_prefix = "curl-7.60.0",
        urls = [
            "https://github.com/curl/curl/releases/download/curl-7_60_0/curl-7.60.0.tar.gz",
            "http://mirror.bazel.build/curl.haxx.se/download/curl-7.60.0.tar.gz",
            "http://curl.haxx.se/download/curl-7.60.0.tar.gz",
        ],
    )

    # numa
    http_file(
        name = "numactl-libs_rpm",
        sha256 = "78c84113dcdca65722d33fd53ddf80289ca6768262e61e228ed093425164c135",
        urls = [
            "https://mirrors.aliyun.com/alinux/2.1903/os/x86_64/Packages/numactl-libs-2.0.9-7.1.al7.x86_64.rpm",
        ],
    )
    http_file(
        name = "numactl-devel_rpm",
        sha256 = "129093692024261b099092600d35b48ee10ad56dd8d940102b370a59ee80055d",
        urls = [
            "https://mirrors.aliyun.com/alinux/2.1903/os/x86_64/Packages/numactl-devel-2.0.9-7.1.al7.x86_64.rpm",
        ],
    )

    http_archive(
        name = "rules_cuda",
        sha256 = "fe8d3d8ed52b9b433f89021b03e3c428a82e10ed90c72808cc4988d1f4b9d1b3",
        strip_prefix = "rules_cuda-v0.2.5",
        urls = ["https://github.com/bazel-contrib/rules_cuda/releases/download/v0.2.5/rules_cuda-v0.2.5.tar.gz"],
    )

    # pybind11
    http_archive(
        name = "pybind11_bazel",
        sha256 = "9c7ffea05a5f2bd9211fdf7c5c685617fa93a801ebd814aed5a32617b7193ed6",
        patches = ["//patches/pybind11_bazel:0001-replace-current_py_cc_headers.patch"],
        strip_prefix = "pybind11_bazel-3.0.0",
        urls = ["https://github.com/pybind/pybind11_bazel/archive/v3.0.0.zip"],
    )

    # We still require the pybind library.
    http_archive(
        name = "pybind11",
        build_file = "@pybind11_bazel//:pybind11-BUILD.bazel",
        sha256 = "20fb420fe163d0657a262a8decb619b7c3101ea91db35f1a7227e67c426d4c7e",
        strip_prefix = "pybind11-3.0.1",
        urls = ["https://github.com/pybind/pybind11/archive/v3.0.1.zip"],
    )
