"""Regression tests for generic stub_source build provenance."""

import contextlib
import importlib.util
import io
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

SOURCE_ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("git_log", str(SOURCE_ROOT / "package/git_commit/git_log.py"))
git_log = importlib.util.module_from_spec(spec)
spec.loader.exec_module(git_log)


class GitLogTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.parent = self.root / "parent repo"
        self.workspace = self.parent / "workspace"
        self.workspace.mkdir(parents=True)
        self.init_repo(self.parent, "parent")
        self.init_repo(self.workspace, "workspace")
        (self.workspace / "bazel").mkdir()
        shutil.copyfile(str(SOURCE_ROOT / "bazel/workspace_status.sh"),
                        str(self.workspace / "bazel/workspace_status.sh"))
        (self.workspace / "open_source").mkdir()
        self.stub = self.workspace / "stub_source"
        self.stub.symlink_to("open_source")

    def git(self, repo, *args):
        return subprocess.check_output(["git", "-C", str(repo)] + list(args),
            stderr=subprocess.DEVNULL, universal_newlines=True).strip()

    def init_repo(self, repo, name):
        self.git(repo, "init")
        self.git(repo, "remote", "add", "origin", "https://example.invalid/" + name)
        self.commit(repo, name)

    def commit(self, repo, message):
        self.git(repo, "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                 "commit", "--allow-empty", "-m", message)

    def status(self):
        text = subprocess.check_output(["bash", "bazel/workspace_status.sh"],
            cwd=str(self.workspace), universal_newlines=True)
        return dict(line.split(" ", 1) for line in text.splitlines())

    def test_default_stub_records_workspace_without_duplicate_history(self):
        status = self.status()
        for suffix in ("COMMIT", "COMMIT_FULL", "REPO"):
            self.assertEqual(status["STABLE_GIT_" + suffix], status["STABLE_STUB_GIT_" + suffix])
        output = git_log.generate(self.workspace, status)
        self.assertEqual(output.count("commit " + status["STABLE_GIT_COMMIT_FULL"]), 1)
        self.assertNotIn("=== stub_source ===", output)
        self.assertNotIn("https://example.invalid/parent", output)

    def test_separate_stub_uses_stamped_revisions_and_git_file(self):
        git_dir = self.root / "workspace.git"
        shutil.move(str(self.workspace / ".git"), str(git_dir))
        (self.workspace / ".git").write_text("gitdir: {}\n".format(git_dir))
        self.stub.unlink()
        self.stub.symlink_to("..")
        status = self.status()
        expected = self.git(self.parent, "rev-parse", "HEAD")
        self.assertEqual(status["STABLE_STUB_GIT_COMMIT_FULL"], expected)
        self.commit(self.parent, "later parent change")
        self.commit(self.workspace, "later workspace change")
        output = git_log.generate(self.workspace, status)
        self.assertIn("commit " + status["STABLE_GIT_COMMIT_FULL"], output)
        self.assertIn("commit " + expected, output)
        self.assertIn("=== stub_source ===", output)
        self.assertNotIn("later parent change", output)
        self.assertNotIn("later workspace change", output)
        self.stub.unlink()
        self.stub.symlink_to(self.parent)
        self.assertEqual(self.status()["STABLE_STUB_GIT_COMMIT_FULL"], self.git(self.parent, "rev-parse", "HEAD"))

    def test_worktree_with_different_revision_keeps_both_histories(self):
        worktree = self.root / "worktree"
        earlier = self.git(self.workspace, "rev-parse", "HEAD")
        self.commit(self.workspace, "second revision")
        self.git(self.workspace, "worktree", "add", "--detach", str(worktree), earlier)
        self.stub.unlink()
        self.stub.symlink_to(worktree)
        output = git_log.generate(self.workspace, self.status())
        self.assertIn("=== stub_source ===", output)
        self.assertTrue(output.split("=== stub_source ===", 1)[1].lstrip().startswith("commit " + earlier + "\n"))

    def test_missing_stub_does_not_fall_back_to_workspace(self):
        self.stub.unlink()
        status = self.status()
        self.assertEqual(status["STABLE_STUB_GIT_COMMIT_FULL"], "unknown")
        self.assertNotIn("=== stub_source ===", git_log.generate(self.workspace, status))

    def test_checkout_url_credentials_are_removed_from_both_repositories(self):
        self.stub.unlink()
        self.stub.symlink_to("..")
        for userinfo in ("user:fake-secret", "fake-token", "user:p@ss"):
            with self.subTest(userinfo=userinfo):
                for repo, name in ((self.workspace, "workspace"), (self.parent, "parent")):
                    self.git(repo, "remote", "set-url", "origin",
                             "https://{}@example.invalid/{}.git".format(userinfo, name))
                status = self.status()
                self.assertEqual(status["STABLE_GIT_REPO"], "https://example.invalid/workspace.git")
                self.assertEqual(status["STABLE_STUB_GIT_REPO"], "https://example.invalid/parent.git")
                output = git_log.generate(self.workspace, status)
                self.assertNotIn(userinfo, output)
                self.assertNotIn("STABLE_STUB_GIT_", output)

    def test_other_repository_url_forms(self):
        for remote, expected in (
            ("ssh://git@example.invalid/team/repo.git", "ssh://example.invalid/team/repo.git"),
            ("git@example.invalid:team/repo.git", "git@example.invalid:team/repo.git"),
            ("https://example.invalid/repo@ref.git", "https://example.invalid/repo@ref.git"),
        ):
            with self.subTest(remote=remote):
                self.git(self.workspace, "remote", "set-url", "origin", remote)
                status = self.status()
                self.assertEqual(status["STABLE_GIT_REPO"], expected)
                self.assertEqual(status["STABLE_STUB_GIT_REPO"], expected)

    def test_stub_without_origin_keeps_commit_history(self):
        self.stub.unlink()
        self.stub.symlink_to("..")
        self.git(self.parent, "remote", "remove", "origin")
        status = self.status()
        self.assertEqual(status["STABLE_STUB_GIT_REPO"], "unknown")
        self.assertIn("commit " + status["STABLE_STUB_GIT_COMMIT_FULL"],
                      git_log.generate(self.workspace, status))

    def test_real_submodule_repository(self):
        self.stub.unlink()
        self.git(self.workspace, "-c", "protocol.file.allow=always", "submodule", "add",
                 str(self.parent), "stub_source")
        self.assertIn(".git/modules/", (self.stub / ".git").read_text())
        status = self.status()
        self.assertEqual(status["STABLE_STUB_GIT_COMMIT_FULL"], self.git(self.parent, "rev-parse", "HEAD"))
        self.assertEqual(git_log.repository_id(self.stub), (self.workspace / ".git/modules/stub_source").resolve())
        output = git_log.generate(self.workspace, status)
        self.assertIn("=== stub_source ===", output)
        self.assertIn("commit " + status["STABLE_STUB_GIT_COMMIT_FULL"], output)

    def test_separate_repositories_with_same_sha_keep_both_histories(self):
        clone = self.root / "clone"
        self.git(self.workspace, "clone", "--no-local", str(self.workspace), str(clone))
        self.stub.unlink()
        self.stub.symlink_to(clone)
        status = self.status()
        self.assertEqual(status["STABLE_GIT_COMMIT_FULL"], status["STABLE_STUB_GIT_COMMIT_FULL"])
        output = git_log.generate(self.workspace, status)
        self.assertIn("=== stub_source ===", output)
        self.assertEqual(output.count("commit " + status["STABLE_GIT_COMMIT_FULL"]), 2)

    def test_source_archive_without_git(self):
        shutil.rmtree(str(self.parent / ".git"))
        shutil.rmtree(str(self.workspace / ".git"))
        status = self.status()
        self.assertEqual(status["STABLE_GIT_COMMIT_FULL"], "unknown")
        self.assertEqual(status["STABLE_STUB_GIT_COMMIT_FULL"], "unknown")
        warning = io.StringIO()
        with contextlib.redirect_stderr(warning):
            output = git_log.generate(self.workspace, status)
        self.assertIn("Git history unavailable", output)
        self.assertIn("WARNING: Git history unavailable", warning.getvalue())
        self.assertIn("WARNING: stub_source revision unavailable", warning.getvalue())


if __name__ == "__main__":
    unittest.main()
