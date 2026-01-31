load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive", "http_file")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository", "new_git_repository")

def clean_dep(dep):
    return str(Label(dep))

def kv_cache_manager_workspace():
    pass
