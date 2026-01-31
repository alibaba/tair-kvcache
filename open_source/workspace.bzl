load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive", "http_file")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository", "new_git_repository")
load("//open_source/deps:http.bzl", "http_deps")
load("//open_source/deps:git.bzl", "git_deps")

def clean_dep(dep):
    return str(Label(dep))

def load_deps():
    http_deps()
    git_deps()

def kv_cache_manager_workspace():
    ## skylib
    skylib_version = "1.0.3"
    http_archive(
        name = "bazel_skylib",
        sha256 = "1c531376ac7e5a180e0237938a2536de0c54d93f5c278634818e0efc952dd56c",
        type = "tar.gz",
        urls = [
            "https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/releases/download/{}/bazel-skylib-{}.tar.gz".format(skylib_version, skylib_version),
            "https://github.com/bazelbuild/bazel-skylib/releases/download/{}/bazel-skylib-{}.tar.gz".format(skylib_version, skylib_version)
        ],
    )

    http_archive(
        name = "rules_proto",
        sha256 = "e017528fd1c91c5a33f15493e3a398181a9e821a804eb7ff5acdd1d2d6c2b18d",
        strip_prefix = "rules_proto-4.0.0-3.20.0",
        urls = [
            "https://github.com/bazelbuild/rules_proto/archive/refs/tags/4.0.0-3.20.0.tar.gz",
        ],
    )

    http_archive(
        name = "rules_pkg",
        urls = [
            "https://mirror.bazel.build/github.com/bazelbuild/rules_pkg/releases/download/0.6.0/rules_pkg-0.6.0.tar.gz",
            "https://github.com/bazelbuild/rules_pkg/releases/download/0.6.0/rules_pkg-0.6.0.tar.gz",
        ],
        sha256 = "62eeb544ff1ef41d786e329e1536c1d541bb9bcad27ae984d57f18f314018e66",
    )

    http_archive(
        name = "hiredis",
        urls = [
            "https://github.com/redis/hiredis/archive/refs/tags/v1.3.0.tar.gz",
        ],
        strip_prefix = "hiredis-1.3.0",
        sha256 = "25cee4500f359cf5cad3b51ed62059aadfc0939b05150c1f19c7e2829123631c",
        build_file = clean_dep("//3rdparty/hiredis:hiredis.BUILD"),
    )

    http_archive(
        # Hedron's Compile Commands Extractor for Bazel
        name = "hedron_compile_commands",
        patches = [
            "//3rdparty/hedron_compile_commands:support_cppm_file.patch",
        ],
        sha256 = "658122cfb1f25be76ea212b00f5eb047d8e2adc8bcf923b918461f2b1e37cdf2",
        strip_prefix = "bazel-compile-commands-extractor-4f28899228fb3ad0126897876f147ca15026151e",
        urls = [
            "https://github.com/hedronvision/bazel-compile-commands-extractor/archive/4f28899228fb3ad0126897876f147ca15026151e.tar.gz",
        ],
    )

    native.local_repository(
        name = "httplib",
        path = "3rdparty/httplib",
    )

    native.local_repository(
        name = "cpp_stub",
        path = "3rdparty/cpp_stub",
    )

    load_deps()
