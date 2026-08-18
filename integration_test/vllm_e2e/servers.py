"""Process lifecycle for the vLLM e2e harness: the real KVCM manager
binary and the real vLLM OpenAI server, plus ScenarioEnv which owns
their lifetimes for one scenario."""

from lib_utils import (
    MODEL_PATH, assert_report_ok, block_token_hash, compare_captures,
    count_captures, find_manager_binary, find_python, find_repo_root,
    free_port, full_block_hashes, get_manager_block_size, is_hybrid_model,
    make_base_prompts, send_completions, shared_token_prefix_len,
    tokenize, wait_for_captures, wait_for_prefix_cached, wait_http,
)


import glob
import json
import logging
import os
import shutil
import socket
import subprocess
import time
import uuid
from typing import Optional

import requests

logger = logging.getLogger("vllm_e2e")
# This module only runs inside test drivers; make the orchestration evidence
# (block counts, verification report, log-scan results) visible in test.log.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s")

# Required: path to a local HF model directory (config.json + weights).
# Full-attention coverage needs a plain attention model (e.g.
# Qwen2.5-7B-Instruct); hybrid coverage needs a mamba/linear + attention model
# (e.g. Qwen3.5-4B). Hybrid models are auto-detected from config.json.
MODEL_PATH = os.environ.get("KVCM_E2E_MODEL")
if not MODEL_PATH:
    raise RuntimeError(
        "KVCM_E2E_MODEL is not set. Point it at a local model directory, e.g. "
        "--test_env=KVCM_E2E_MODEL=/path/to/Qwen2.5-7B-Instruct (full "
        "attention) or /path/to/Qwen3.5-4B (hybrid). See "
        "integration_test/vllm_e2e/README.md.")

# Model-agnostic alias the driver uses in OpenAI API requests, so the tests do
# not depend on the model directory name.
SERVED_MODEL_NAME = "e2e-model"

# --------------------------------------------------------------------------- #
class ManagerProcess:
    def __init__(self, workdir: str, storage_root: str, key_count_per_file: int = 8):
        self.workdir = workdir
        os.makedirs(workdir, exist_ok=True)
        self.rpc_port = free_port()
        self.http_port = free_port()
        self.admin_rpc_port = free_port()
        self.admin_http_port = free_port()
        self.storage_root = storage_root
        self.key_count_per_file = key_count_per_file
        self.proc: Optional[subprocess.Popen] = None
        self.config_path = os.path.join(workdir, "startup_config.json")

    def manager_uri(self) -> str:
        return f"http://127.0.0.1:{self.http_port}"

    def _write_config(self):
        cfg = {
            "storage_config": {
                "type": "file",
                "global_unique_name": "nfs_01",
                "storage_spec": {
                    # The backend concatenates root_path + key with no
                    # separator; the trailing slash keeps files inside the dir.
                    "root_path": self.storage_root.rstrip("/") + "/",
                    "key_count_per_file": self.key_count_per_file,
                },
            },
            "instance_group": {
                "name": "default",
                "storage_candidates": ["nfs_01"],
                "global_quota_group_name": "default_quota_group",
                "max_instance_count": 100,
                "quota": {
                    "capacity": 30000000000,
                    "quota_config": [
                        {"storage_type": "file", "capacity": 10000000000},
                        {"storage_type": "hf3fs", "capacity": 10000000000},
                        {"storage_type": "pace", "capacity": 10000000000},
                    ],
                },
                "cache_config": {
                    "reclaim_strategy": {
                        "reclaim_policy": 1,
                        "trigger_strategy": {"used_percentage": 0.8},
                        "delay_before_delete_ms": 1000,
                    },
                    "cache_prefer_strategy": 2,
                    "meta_indexer_config": {
                        "max_key_count": 1000000,
                        "mutex_shard_num": 16,
                        "batch_key_size": 16,
                        "meta_storage_backend_config": {
                            "storage_type": "local",
                            "storage_uri": "",
                        },
                        "meta_cache_policy_config": {
                            "type": "LRU",
                            "capacity": 10000,
                            "cache_shard_bits": 0,
                            "high_pri_pool_ratio": 0.0,
                        },
                    },
                },
                "user_data": '{"description": "vllm e2e test instance group"}',
                "version": 1,
            },
        }
        with open(self.config_path, "w") as f:
            json.dump(cfg, f, indent=2)

    def start(self, repo_root: str):
        self._write_config()
        binary = find_manager_binary(repo_root)
        cmd = [
            binary,
            "--env", f"kvcm.service.rpc_port={self.rpc_port}",
            "--env", f"kvcm.service.http_port={self.http_port}",
            "--env", f"kvcm.service.admin_rpc_port={self.admin_rpc_port}",
            "--env", f"kvcm.service.admin_http_port={self.admin_http_port}",
            "--env", f"kvcm.startup_config={self.config_path}",
            "--env", "kvcm.logger.log_level=5",
        ]
        logger.info("starting manager: %s (cwd=%s)", " ".join(cmd), self.workdir)
        self.proc = subprocess.Popen(
            cmd,
            cwd=self.workdir,
            stdout=open(os.path.join(self.workdir, "manager.stdout"), "w"),
            stderr=open(os.path.join(self.workdir, "manager.stderr"), "w"),
        )
        if not wait_http(
            f"{self.manager_uri()}/api/getClusterInfo",
            timeout=60,
            post_body={"trace_id": "probe", "instance_id": "probe"},
        ):
            raise RuntimeError("manager did not become ready; see manager.stderr")
        logger.info("manager ready at %s", self.manager_uri())

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()


# --------------------------------------------------------------------------- #
# vLLM server

# --------------------------------------------------------------------------- #
class VllmServer:
    def __init__(self, workdir: str, capture_dir: str, manager_uri: str,
                 tp_size: int, coordinator_base_port: int,
                 instance_id: str, preferred_block_size: int,
                 enable_prefix_caching: bool,
                 connector_name: str = "VerifyingConnector",
                 log_level: str = "INFO",
                 extra_config_overrides: Optional[dict] = None,
                 kv_load_failure_policy: Optional[str] = None):
        self.workdir = workdir
        os.makedirs(workdir, exist_ok=True)
        self.capture_dir = capture_dir
        os.makedirs(capture_dir, exist_ok=True)
        self.port = free_port()
        self.manager_uri = manager_uri
        self.tp_size = tp_size
        self.coordinator_base_port = coordinator_base_port
        self.instance_id = instance_id
        self.preferred_block_size = preferred_block_size
        self.enable_prefix_caching = enable_prefix_caching
        self.connector_name = connector_name
        self.log_level = log_level
        self.extra_config_overrides = extra_config_overrides or {}
        self.kv_load_failure_policy = kv_load_failure_policy
        self.proc: Optional[subprocess.Popen] = None

    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self, repo_root: str, log_suffix: str = ""):
        extra_config = {
            "manager_uri": self.manager_uri,
            "coordinator_base_port": self.coordinator_base_port,
            "instance_group": "default",
            "instance_id": self.instance_id,
            "preferred_block_size": self.preferred_block_size,
            "log_level": self.log_level,
        }
        extra_config.update(self.extra_config_overrides)
        kv_transfer_config = {
            "kv_connector": self.connector_name,
            "kv_role": "kv_both",
            "kv_connector_module_path": "test_connector",
            "kv_connector_extra_config": extra_config,
        }
        if self.kv_load_failure_policy:
            kv_transfer_config["kv_load_failure_policy"] = self.kv_load_failure_policy
        # Serving knobs, override-able via vllm_args (a scenario can widen
        # max-model-len or raise gpu-memory-utilization without editing the
        # harness); keys are vLLM CLI flags without the leading "--".
        vllm_args = {
            "max-model-len": "4096",
            "gpu-memory-utilization": "0.85",
            "enforce-eager": None,
            "max-num-seqs": "16",
            **getattr(self, "vllm_args", {}),
        }
        cmd = [
            find_python(), "-m", "vllm.entrypoints.openai.api_server",
            "--model", MODEL_PATH,
            "--served-model-name", SERVED_MODEL_NAME,
            "--host", "127.0.0.1",  # loopback only: single-machine harness
            "--port", str(self.port),
            "--tensor-parallel-size", str(self.tp_size),
        ]
        for flag, value in vllm_args.items():
            cmd += [f"--{flag}"] if value is None else [f"--{flag}", str(value)]
        cmd += ["--kv-transfer-config", json.dumps(kv_transfer_config)]
        if self.enable_prefix_caching:
            # Hybrid models need prefix caching to expose per-group block tables
            # (mamba_cache_mode="align"); align mode requires chunked prefill.
            cmd += ["--enable-prefix-caching", "--enable-chunked-prefill"]
        else:
            cmd += ["--no-enable-prefix-caching"]
        env = os.environ.copy()
        # The connector modules must resolve to THIS checkout, ahead of any
        # kv_cache_manager wheel installed in the vLLM venv.
        env["PYTHONPATH"] = os.pathsep.join([
            os.path.dirname(os.path.abspath(__file__)),
            find_repo_root(),
            env.get("PYTHONPATH", ""),
        ])
        env["KVCM_E2E_CAPTURE_DIR"] = self.capture_dir
        # Required, not cosmetic: the connector asserts token-major pages
        # (NHD memory order) at register_kv_caches; on a vLLM whose default
        # layout is HND the connector refuses to start without this, and a
        # capture comparison across layouts would be meaningless anyway.
        env.setdefault("VLLM_KV_CACHE_LAYOUT", "NHD")
        # Force FlashAttention for the full-attention layers: it produces the
        # [2, num_blocks, block_size, num_kv_heads, head_size] layout the
        # connector expects, and avoids the flashinfer backend entirely.
        env.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
        # Use the PyTorch-native sampler; the flashinfer sampler JIT-compiles
        # with ninja, which is not available in the test environment.
        env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        # The test venv ships mismatched flashinfer / flashinfer-cubin wheels;
        # skip the version check so importing vLLM's attention registry does not
        # crash before the FlashAttention backend is selected.
        env.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
        logger.info("starting vllm: %s", " ".join(cmd))
        self.proc = subprocess.Popen(
            cmd,
            cwd=self.workdir,
            env=env,
            stdout=open(os.path.join(self.workdir, f"vllm{log_suffix}.stdout"), "w"),
            stderr=open(os.path.join(self.workdir, f"vllm{log_suffix}.stderr"), "w"),
        )
        if not wait_http(f"{self.base_url()}/health", timeout=600):
            raise RuntimeError("vllm did not become ready; see vllm.stderr")
        logger.info("vllm ready at %s", self.base_url())

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()


# --------------------------------------------------------------------------- #
# Request driver

# --------------------------------------------------------------------------- #
class ScenarioEnv:
    """Owns one scenario's manager + vLLM server lifecycle and scratch dirs.

    Custom scenarios (partial-hit, full-hit, load-failure, multi-turn) share
    this; run_e2e keeps its own two-phase flow on top of the same pieces.
    """

    def __init__(self, scenario: str, tp_size: int = 1,
                 preferred_block_size: int = 0,
                 enable_prefix_caching: Optional[bool] = None,
                 connector_name: str = "VerifyingConnector",
                 log_level: str = "INFO",
                 extra_config_overrides: Optional[dict] = None,
                 key_count_per_file: int = 8,
                 kv_load_failure_policy: Optional[str] = None):
        self.scenario = scenario
        self.tp_size = tp_size
        self.hybrid = is_hybrid_model(MODEL_PATH)
        self.preferred_block_size = 0 if self.hybrid else preferred_block_size
        self.enable_prefix_caching = (self.hybrid if enable_prefix_caching is None
                                      else enable_prefix_caching)
        self.connector_name = connector_name
        self.log_level = log_level
        self.extra_config_overrides = extra_config_overrides
        self.kv_load_failure_policy = kv_load_failure_policy

        self.repo_root = find_repo_root()
        scratch_root = (os.environ.get("TEST_TMPDIR")
                        or os.environ.get("TMPDIR") or "/tmp")
        self.base_workdir = os.path.join(scratch_root, "kvcm_vllm_e2e", scenario)
        if os.path.exists(self.base_workdir):
            shutil.rmtree(self.base_workdir)
        self.storage_root = os.path.join(self.base_workdir, "nfs")
        self.capture_dir = os.path.join(self.base_workdir, "captures")
        self.vllm_dir = os.path.join(self.base_workdir, "vllm")
        os.makedirs(self.storage_root, exist_ok=True)

        self.instance_id = f"e2e-{scenario}-{uuid.uuid4().hex[:8]}"
        self.manager = ManagerProcess(
            os.path.join(self.base_workdir, "manager"), self.storage_root,
            key_count_per_file=key_count_per_file)
        self.vllm: Optional[VllmServer] = None

    def start_manager(self):
        self.manager.start(self.repo_root)

    def start_vllm(self, log_suffix: str = "") -> VllmServer:
        self.vllm = VllmServer(
            self.vllm_dir, self.capture_dir, self.manager.manager_uri(),
            self.tp_size, coordinator_base_port=free_port(),
            instance_id=self.instance_id,
            preferred_block_size=self.preferred_block_size,
            enable_prefix_caching=self.enable_prefix_caching,
            connector_name=self.connector_name,
            log_level=self.log_level,
            extra_config_overrides=self.extra_config_overrides,
            kv_load_failure_policy=self.kv_load_failure_policy,
        )
        self.vllm.start(self.repo_root, log_suffix=log_suffix)
        return self.vllm

    def restart_vllm(self, log_suffix: str = "") -> VllmServer:
        """Restart vLLM to drop its local prefix cache (KVCM state persists)."""
        if self.vllm:
            self.vllm.stop()
        return self.start_vllm(log_suffix)

    def manager_block_size(self) -> int:
        return get_manager_block_size(self.manager.manager_uri(), self.instance_id)

    def scan_connector_logs(self, pattern: str) -> list:
        """Regex-scan all vLLM std streams; returns list of match groups."""
        import re
        out = []
        for path in glob.glob(os.path.join(self.vllm_dir, "vllm*.std*")):
            with open(path, errors="replace") as f:
                for line in f:
                    m = re.search(pattern, line)
                    if m:
                        out.append(m.groups() if m.groups() else m.group(0))
        return out

    def stop(self):
        if self.vllm:
            self.vllm.stop()
        self.manager.stop()


def run_e2e(scenario: str, tp_size: int, num_prompts: int,
            preferred_block_size: int, connector_name: str = "VerifyingConnector",
            expect_verification_failure: bool = False):
    """Run one full save-then-load verification scenario.

    Full-attention models: prefix caching off, one server across both phases.
    Hybrid models: prefix caching on (align mode -> per-group block tables), the
    vLLM server is restarted between phases so phase 2 loads from KVCM instead of
    hitting the local prefix cache.

    connector_name selects the connector class inside test_connector.py; the
    mutation meta-test passes "MutatedConnector" and sets
    expect_verification_failure=True to prove the harness detects an injected
    off-by-one in the token translation.
    """
    import torch  # noqa: F401  (ensure torch importable early for clear errors)

    env = ScenarioEnv(scenario, tp_size=tp_size,
                      preferred_block_size=preferred_block_size,
                      connector_name=connector_name)
    hybrid = env.hybrid
    capture_dir = env.capture_dir
    logger.info("scenario=%s model=%s hybrid=%s tp=%d prompts=%d preferred_bs=%d",
                scenario, MODEL_PATH, hybrid, tp_size, num_prompts,
                env.preferred_block_size)

    try:
        env.start_manager()
        vllm = env.start_vllm(log_suffix="" if not hybrid else "_p1")

        base_prompts = make_base_prompts(num_prompts, hybrid)
        suffixes = [f" Now answer question {i}: what is 2+2?" for i in range(num_prompts)]
        phase2_prompts = [p + s for p, s in zip(base_prompts, suffixes)]

        # Compute per-prompt expected block counts from the actual tokenization
        # so a silently dropped prompt (or block) fails the run.
        mbs = env.manager_block_size()
        base_tokens = [tokenize(vllm.base_url(), p) for p in base_prompts]
        phase2_tokens = [tokenize(vllm.base_url(), p) for p in phase2_prompts]
        expected_save_blocks = [len(t) // mbs for t in base_tokens]
        # Loads cover the shared token prefix (the base/suffix boundary token may
        # re-merge under tokenization, shortening the shared prefix by one).
        expected_load_blocks = [
            min(shared_token_prefix_len(b, p2) // mbs, s)
            for b, p2, s in zip(base_tokens, phase2_tokens, expected_save_blocks)
        ]
        assert all(n >= 1 for n in expected_load_blocks), (
            f"prompts too short to span a manager block (mbs={mbs}): "
            f"{expected_load_blocks}")
        logger.info("mbs=%d expected save blocks=%s load blocks=%s",
                    mbs, expected_save_blocks, expected_load_blocks)

        # ---- Phase 1: fresh prefill -> connector saves -> reference capture.
        logger.info("phase 1: sending %d fresh prompts", num_prompts)
        send_completions(vllm.base_url(), base_prompts)
        wait_for_captures(capture_dir, "ref",
                          expected=tp_size * sum(expected_save_blocks), timeout=180)

        # The save is committed to the manager asynchronously after the ref
        # capture (which fires when the save is submitted). Wait until the manager
        # actually has the prefix, otherwise phase 2 would find no match.
        for toks, blocks in zip(base_tokens, expected_save_blocks):
            if not wait_for_prefix_cached(env.manager.manager_uri(), env.instance_id,
                                          toks, min_blocks=blocks):
                raise AssertionError("save was not committed to the manager in time")

        # Hybrid models keep prefix caching on, which also populates the local
        # prefix cache; restart vLLM so phase 2 loads from KVCM, not locally.
        if hybrid:
            logger.info("restarting vLLM before phase 2 (clear local prefix cache)")
            vllm = env.restart_vllm(log_suffix="_p2")

        # ---- Phase 2: same prefix + suffix -> connector loads -> loaded capture.
        logger.info("phase 2: sending %d prefix+suffix prompts", num_prompts)
        send_completions(vllm.base_url(), phase2_prompts)
        wait_for_captures(capture_dir, "loaded",
                          expected=tp_size * sum(expected_load_blocks), timeout=180)

        report = compare_captures(capture_dir, tp_size)
        min_matched = tp_size * sum(expected_load_blocks)
        if expect_verification_failure:
            try:
                assert_report_ok(report, min_matched=min_matched)
            except AssertionError as e:
                logger.info("verification failed as expected: %s", e)
                return
            raise AssertionError(
                "mutated connector passed KV verification; the harness is blind")
        assert_report_ok(report, min_matched=min_matched)
        logger.info("scenario %s PASSED: %s", scenario, json.dumps(
            {k: v for k, v in report.items() if k not in ("failures", "matched_keys")},
            default=str))
    finally:
        env.stop()
