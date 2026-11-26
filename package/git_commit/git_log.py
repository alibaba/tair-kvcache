"""Generate history for the stamped workspace and stub_source revisions."""

import argparse
from pathlib import Path
import subprocess
import sys


def git(repo, *args):
    result = subprocess.run(
        ["git", "-C", str(repo)] + list(args),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    return result.stdout if result.returncode == 0 else None


def repository_id(repo):
    common_dir = git(repo, "rev-parse", "--git-common-dir")
    if common_dir is None:
        return None
    return (repo / common_dir.strip()).resolve()


def history(repo, revision):
    output = git(repo, "log", "-10", revision)
    if output is not None:
        return output
    message = "Git history unavailable for revision {}.".format(revision)
    print("WARNING: " + message, file=sys.stderr)
    return message + "\n"


def generate(workspace, status):
    revision = status.get("STABLE_GIT_COMMIT_FULL", "unknown")
    output = history(workspace, revision)
    stub = workspace / "stub_source"
    stub_revision = status.get("STABLE_STUB_GIT_COMMIT_FULL", "unknown")
    if stub_revision == "unknown":
        print("WARNING: stub_source revision unavailable.", file=sys.stderr)
        return output
    # Worktrees may share a repository but build different revisions.
    same_repository = repository_id(workspace) == repository_id(stub)
    if same_repository and revision == stub_revision:
        return output
    return output + "\n=== stub_source ===\n\n" + history(stub, stub_revision)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--out", required=True)
    parser.add_argument("--status", required=True)
    args = parser.parse_args()
    status = {}
    for line in Path(args.status).read_text().splitlines():
        key, _, value = line.partition(" ")
        status[key] = value
    workspace = Path(__file__).resolve().parents[2]
    Path(args.out).write_text(generate(workspace, status))


if __name__ == "__main__":
    main()
