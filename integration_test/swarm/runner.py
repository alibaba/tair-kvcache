"""Runs the kvcm_swarm binary and collects out-of-process facts.

The runner never influences traffic: it starts the generator, waits for it and
loads the report it produced.
"""
import json
import os
import subprocess
import time


class SwarmRun(object):
    def __init__(self, exit_code, stdout, stderr, wall_seconds, report, violations, config_path):
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr
        self.wall_seconds = wall_seconds
        self.report = report
        self.violations = violations
        self.config_path = config_path

    def describe(self):
        return ("exit_code=%s wall=%.1fs\n--- stdout ---\n%s\n--- stderr ---\n%s"
                % (self.exit_code, self.wall_seconds, self.stdout[-8000:], self.stderr[-8000:]))


def swarm_binary_path():
    """Resolves the generator binary inside the bazel runfiles tree."""
    candidates = []
    srcdir = os.environ.get("TEST_SRCDIR")
    if srcdir:
        candidates.append(os.path.join(srcdir, "kv_cache_manager", "tools", "kvcm_swarm", "kvcm_swarm"))
        candidates.append(os.path.join(srcdir, "tools", "kvcm_swarm", "kvcm_swarm"))
    candidates.append(os.path.join(os.getcwd(), "tools", "kvcm_swarm", "kvcm_swarm"))
    workspace = os.environ.get("BUILD_WORKSPACE_DIRECTORY")
    if workspace:
        candidates.append(os.path.join(workspace, "bazel-bin", "tools", "kvcm_swarm", "kvcm_swarm"))
    for candidate in candidates:
        if os.path.exists(candidate) and os.access(candidate, os.X_OK):
            return candidate
    raise AssertionError("kvcm_swarm binary not found; looked at: %s" % candidates)


def _load_report(path):
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        content = handle.read()
    if not content.strip():
        return None
    return json.loads(content)


def _load_violations(path):
    if not os.path.exists(path):
        return []
    entries = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def run_swarm(config_path, timeout_seconds=600, extra_args=None):
    binary = swarm_binary_path()
    with open(config_path) as handle:
        config = json.load(handle)
    command = [binary, "--config", config_path] + list(extra_args or [])
    started = time.time()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()
        raise AssertionError("kvcm_swarm did not finish within %ss\n%s\n%s"
                             % (timeout_seconds, stdout[-4000:], stderr[-4000:]))
    wall = time.time() - started
    report = _load_report(config["evidence"]["output_json"])
    violations = _load_violations(config["evidence"]["violations_jsonl"])
    return SwarmRun(process.returncode, stdout, stderr, wall, report, violations, config_path)


def validate_only(config_path):
    binary = swarm_binary_path()
    process = subprocess.run([binary, "--config", config_path, "--validate-only"],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=120)
    return process.returncode, process.stdout, process.stderr
