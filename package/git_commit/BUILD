load("//bazel:bundle.bzl", "bundle_files")

genrule(
    name = "git_commit_version",
    srcs = [],
    outs = ["git_commit"],
    cmd = "cat bazel-out/stable-status.txt bazel-out/volatile-status.txt > $(OUTS)",
    stamp = 1,
)

genrule(
    name = "git_commit_version_log_latest",
    srcs = [
        ":git_log.py",
        ":git_commit_version",
    ],
    outs = ["git_commit_log"],
    cmd = "python3 $(location git_log.py) --status $(location :git_commit_version) -o $(OUTS)",
)

bundle_files(
    name = "commit_version_file",
    srcs = [
        ":git_commit_version",
        ":git_commit_version_log_latest",
    ],
    prefix = "etc",
    visibility = ["//visibility:public"],
)

py_test(
    name = "git_log_test",
    srcs = ["git_log_test.py"],
    data = [
        "git_log.py",
        "//bazel:workspace_status.sh",
    ],
    python_version = "PY3",
)
