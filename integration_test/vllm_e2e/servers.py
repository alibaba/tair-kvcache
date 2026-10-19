"""Process lifecycle for the vLLM e2e harness: the real KVCM manager
binary and the real vLLM OpenAI server, plus ScenarioEnv which owns
their lifetimes for one scenario."""

from lib_utils import (
    MODEL_PATH, CaptureVerificationError, RetentionFacts, assert_loaded_blocks,
    assert_pinned_ok, assert_report_ok, block_tables_from_logs,
    block_token_hash, compare_captures, count_captures, find_manager_binary,
    find_python, find_repo_root, free_port, full_block_hashes,
    get_manager_block_size, has_full_attention_group, is_hybrid_model,
    is_mla_model, is_multigroup_model, is_swa_model, loaded_block_count,
    make_base_prompts, model_fingerprint, retention_facts_from_logs,
    scenario_tp_size, send_completions, window_kind,
    shared_token_prefix_len, tokenize, wait_for_captures,
    wait_for_prefix_cached, wait_http, window_tail_blocks, window_tokens,
)


import glob
import json
import logging
import os
import shutil
import socket
import subprocess
import time
import unittest
import uuid
from dataclasses import dataclass
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

#: Pinned-host-RAM gate per model *fingerprint*, in GiB (see the connector's
#: staging_pool_max_bytes_per_group and the SWA pinned report). The fingerprint
#: is (architectures[0], declared window) from the checkpoint's config, not a
#: directory name: a copy, a rename or a symlinked checkpoint must not turn a
#: hard gate into a silent skip. The window separates same-architecture
#: siblings (gemma-3-4b 1024 vs gemma-3-1b/270m 512). Only models with a
#: derived budget are gated; KVCM_E2E_PINNED_GIB overrides for any model (e.g.
#: a new one whose budget was just computed).
PINNED_BUDGET_GIB = {
    ("Gemma3ForConditionalGeneration", 1024): 0.4,
    ("Gemma3ForCausalLM", 512): 0.4,
    ("Qwen3_5ForConditionalGeneration", None): 1.0,
}


def pinned_budget_gib(model_path: str = MODEL_PATH) -> Optional[float]:
    override = os.environ.get("KVCM_E2E_PINNED_GIB")
    if override:
        return float(override)
    return PINNED_BUDGET_GIB.get(model_fingerprint(model_path))


def nvidia_lib_dirs(python: str) -> list:
    """Existing nvidia cuBLAS/cuDNN lib dirs of the vLLM venv, in load order.

    The venv ships cuBLAS/cuDNN under site-packages/nvidia/*/lib, which is not
    on the dynamic loader's search path: any lazily dlopen'd consumer dies
    (gemma-3's vision tower conv2d core-dumps on libcublasLt.so.13). Only
    directories that exist are returned, so the injection is a no-op for a venv
    that does not need it.

    cuBLAS lives in ``nvidia/cu13/lib`` on CUDA-13 wheels but in
    ``nvidia/cublas/lib`` on the CUDA 11/12 ones, so both are probed (the
    missing variant is simply absent, and a venv that has both keeps the order
    below). Deliberately just these packages: widening the search path to every
    nvidia wheel (nccl, nvshmem, ...) would risk changing which shared object
    an existing model loads.
    """
    import glob as _glob
    site = os.path.join(os.path.dirname(os.path.normpath(python)), "..", "lib")
    found = []
    for package in ("cu13", "cublas", "cudnn"):
        pattern = os.path.join(site, "python*", "site-packages", "nvidia",
                               package, "lib")
        found.extend(sorted(d for d in _glob.glob(pattern) if os.path.isdir(d)))
    return [os.path.normpath(d) for d in found]

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
#: Headroom a scenario leaves inside --max-model-len for the completion it asks
#: for plus tokenizer slop when it gates a generated prompt on that limit.
MODEL_LEN_MARGIN = 16

#: The connector's worker-init line names the tp rank ("tp rank: 1/2"). The
#: pinned total does not, so the ranks come from here (same worker, same start)
#: and the report count is what says every start is covered.
RANK_LINE_RE = r"worker inited, tp rank: (\d+)/(\d+)"

#: Serving flags every scenario starts from; `overrides` (a scenario's own args
#: merged with KVCM_E2E_VLLM_ARGS) wins, then per-model additions kick in.
_BASE_SERVING_ARGS = {
    "max-model-len": "4096",
    "gpu-memory-utilization": "0.85",
    "enforce-eager": None,
    "max-num-seqs": "16",
}


def effective_serving_args(model_path: str = MODEL_PATH,
                           overrides: Optional[dict] = None) -> dict:
    """The serving flags a server is actually started with.

    Single merge point for the built-in defaults, the operator's
    KVCM_E2E_VLLM_ARGS and a scenario's own overrides (in that order of
    precedence), so anything that has to reason about a knob -- e.g. the
    scenario gates reading --max-model-len -- reads the same value the engine
    got instead of re-guessing the default.
    """
    return serving_args(model_path, {
        **(overrides or {}),
        **json.loads(os.environ.get("KVCM_E2E_VLLM_ARGS", "{}")),
    })


def serving_args(model_path: str, overrides: Optional[dict] = None) -> dict:
    """The vLLM serving flags for one model (pure: call it to inspect a run).

    Windowed models get a bounded prefill chunk so the window slides *inside*
    the prefill -- blocks produced by an earlier chunk are freed when a later
    one is scheduled, which is exactly the case the eager capture has to win
    (a single-chunk prefill would leave the window untouched until decode).

    The chunk must also fit the staging arena: a capture holds its slots until
    its write session completes, so the arena must hold the captures of the
    previous *and* the current chunk of every request that prefills
    concurrently. With the default pool (1024 slots, 128 reserved for
    transfers -> 896 shared) and up to 4 concurrent prefills x 6 windowed
    groups (gemma-3), 2 x 4 x 6 x 16 = 768 slots <= 896. A larger chunk
    overruns the arena and the capture is withheld *by design* (the blocks are
    failed rather than published, so the prefix simply stops there); the run
    fails on the historian assertion instead of silently degrading, which is
    the intended fail-closed behaviour.

    The guard is `is_swa_model`, the same predicate that selects the windowed
    orchestration -- *not* a bare window value: Qwen2.5 declares
    ``sliding_window=131072`` (``use_sliding_window: false``) and never uses
    it, and capping its chunk would change a legacy model's prefill for no
    reason (hybrid models keep the default chunk too: no window to slide).
    """
    args = {**_BASE_SERVING_ARGS, **(overrides or {})}
    if is_swa_model(model_path):
        args.setdefault("max-num-batched-tokens", 256)
    return args


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
        # KVCM_E2E_VLLM_ARGS (JSON dict, same key format) is the operator
        # escape hatch -- e.g. serving a model whose weights need
        # quantization to fit the local GPUs.
        vllm_args = effective_serving_args(MODEL_PATH,
                                           getattr(self, "vllm_args", {}))
        # Recorded for the gates that must reason about a knob afterwards
        # (--max-model-len bounds the prompts a scenario may build).
        self.serving_args = vllm_args
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
        # Multimodal checkpoints (gemma-3) dlopen cuBLAS/cuDNN lazily from the
        # venv's nvidia wheels; without these paths on the loader's search path
        # the engine core-dumps on the first conv2d. Prepended: these
        # directories win over an operator's existing entries for the same
        # library (they are the venv's own, i.e. exactly the ones this vLLM was
        # built against), while every existing entry is preserved and a
        # directory already present is not duplicated. Inert for models that
        # never dlopen those libraries.
        lib_dirs = [d for d in nvidia_lib_dirs(find_python())
                    if d not in env.get("LD_LIBRARY_PATH", "").split(os.pathsep)]
        if lib_dirs:
            env["LD_LIBRARY_PATH"] = os.pathsep.join(
                lib_dirs + ([env["LD_LIBRARY_PATH"]] if env.get("LD_LIBRARY_PATH") else []))
            logger.info("LD_LIBRARY_PATH += %s", os.pathsep.join(lib_dirs))
        # Required, not cosmetic: the connector asserts token-major pages
        # (NHD memory order) at register_kv_caches; on a vLLM whose default
        # layout is HND the connector refuses to start without this, and a
        # capture comparison across layouts would be meaningless anyway.
        env.setdefault("VLLM_KV_CACHE_LAYOUT", "NHD")
        # Force FlashAttention for the full-attention layers: it produces the
        # [2, num_blocks, block_size, num_kv_heads, head_size] layout the
        # connector expects, and avoids the flashinfer backend entirely.
        # MLA models are exempt: FLASH_ATTN has no MLA implementation, so let
        # vLLM auto-select an MLA backend (TRITON_MLA on Ampere; every MLA
        # backend registers the same 3-D (num_blocks, block, head_size) latent
        # cache, which the connector detects from the tensor shape).
        if not is_mla_model(MODEL_PATH):
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

    def __init__(self, scenario: str, tp_size: Optional[int] = None,
                 preferred_block_size: int = 0,
                 enable_prefix_caching: Optional[bool] = None,
                 connector_name: str = "VerifyingConnector",
                 log_level: str = "INFO",
                 extra_config_overrides: Optional[dict] = None,
                 key_count_per_file: int = 8,
                 kv_load_failure_policy: Optional[str] = None):
        self.scenario = scenario
        # None = per-scenario default, overridable via KVCM_E2E_TP (models
        # too large for one GPU run every scenario with tp=2).
        self.tp_size = tp_size if tp_size is not None else scenario_tp_size()
        # multigroup = hybrid OR windowed: both need prefix caching and a
        # restart between the phases. hybrid additionally selects the mamba
        # specific branches (long prompts, recurrent-state assertions);
        # window is the attention window in tokens (None = full attention) and
        # comes from the same predicate as the windowed orchestration itself:
        # a declared window a model never uses (Qwen2.5) must not switch the
        # expectations to tail-window semantics.
        self.multigroup = is_multigroup_model(MODEL_PATH)
        self.hybrid = is_hybrid_model(MODEL_PATH)
        self.window = (window_tokens(MODEL_PATH)
                       if is_swa_model(MODEL_PATH) else None)
        # Which retention rule the window follows: sliding and chunked-local
        # skip different prefixes on a hit (see lib_utils.skipped_window_tokens).
        self.window_kind = window_kind(MODEL_PATH) if self.window else None
        # A full-attention group keeps the whole matched prefix, so a hit loads
        # every matched block and not just the tail window (see
        # loaded_block_count).
        self.full_attention_group = has_full_attention_group(MODEL_PATH)
        self.preferred_block_size = 0 if self.multigroup else preferred_block_size
        self.enable_prefix_caching = (self.multigroup if enable_prefix_caching is None
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
        # Engine starts so far: each one prints one pinned report per tp rank,
        # which is what makes a missing rank detectable (see assert_pinned_ok).
        self.vllm_starts = 0
        # Parsed lazily from the connector's registration line.
        self._retention: Optional["RetentionFacts"] = None

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
        self.vllm_starts += 1
        return self.vllm

    def restart_vllm(self, log_suffix: str = "") -> VllmServer:
        """Restart vLLM to drop its local prefix cache (KVCM state persists)."""
        if self.vllm:
            self.vllm.stop()
        return self.start_vllm(log_suffix)

    def manager_block_size(self) -> int:
        return get_manager_block_size(self.manager.manager_uri(), self.instance_id)

    def max_model_len(self) -> int:
        """The server's --max-model-len, i.e. what a prompt may span.

        Read from the *effective* args: the started server's own record when
        there is one, otherwise the same merge the server would use (defaults <
        KVCM_E2E_VLLM_ARGS), so a scenario gate ("three windows must fit")
        follows an operator's override instead of the built-in default.
        """
        args = (getattr(self.vllm, "serving_args", None)
                or effective_serving_args())
        return int(args["max-model-len"])

    def retention_facts(self) -> "RetentionFacts":
        """The connector-registered retention, parsed once (see lib_utils).

        Falls back to the config-derived facts (with a log line) when the
        engine has not registered yet, e.g. before the first server start.
        """
        if self._retention is None:
            lines = self.scan_connector_logs(
                r"retention=WindowRetention\([^)]*\)")
            if lines:
                # Only the parsed registration is cached: the config-derived
                # fallback below must be re-evaluated, or a geometry built
                # before the first engine start (a scenario gate) would freeze
                # extra_retained_tokens at 0 for the expectations that follow.
                self._retention = retention_facts_from_logs(lines)
            else:
                logger.info(
                    "no WindowRetention in the connector logs yet; using the "
                    "config-derived retention (kind=%s window=%s)",
                    self.window_kind, self.window)
                return RetentionFacts(kind=self.window_kind,
                                      window_tokens=self.window)
        return self._retention

    def guard_prompt_fits(self, tokens: list, label: str,
                          max_tokens: int = 0) -> None:
        """Skip the scenario when a generated prompt exceeds the limit.

        The windowed prompts (``make_base_prompts``) exist to cross the window;
        for a window wider than roughly a third of ``--max-model-len`` they
        cannot, and the server would answer 400 before any verification. Say so
        instead: an operator can raise the limit through KVCM_E2E_VLLM_ARGS.
        ``max_tokens`` is the scenario's own completion request -- vLLM limits
        prompt plus output against the same context window, so leaving it out
        would pass the gate and still let the server reject the request.
        """
        needed = len(tokens) + max_tokens + MODEL_LEN_MARGIN
        limit = self.max_model_len()
        if needed > limit:
            raise unittest.SkipTest(
                f"{label}: prompt {len(tokens)} + output {max_tokens} + "
                f"{MODEL_LEN_MARGIN} margin = {needed} tokens (window "
                f"{self.window}) but the server accepts {limit} "
                f"(--max-model-len); raise it via KVCM_E2E_VLLM_ARGS to run "
                f"this scenario")

    def window_geometry(self) -> "WindowGeometry":
        """Expectation inputs for the loaded region, cross-checked.

        The windowed orchestration was chosen from the config *before* any
        engine started; the connector's registered retention is the
        authoritative statement of what it will actually keep. They must agree
        (a disagreement means the config-level detection missed a windowed
        group -- an R-SWA checkpoint whose window vLLM injects, say -- and the
        scenario would then verify the wrong region), so this fails loudly
        instead of quietly shrinking the assertion surface. The
        ``extra_retained_tokens`` has no config source at all and comes from
        the registration alone (0 with a log line when unknown).
        """
        facts = self.retention_facts()
        if facts.kind is not None:
            assert self.window is not None, (
                f"the connector registered a {facts.kind!r} windowed group "
                f"(window {facts.window_tokens}) but the harness treated "
                f"{MODEL_PATH} as having no window: the config-level detection "
                f"missed it, so the windowed orchestration (prefix caching, "
                f"phase restart, windowed expectations) was never applied")
            assert self.window == facts.window_tokens, (
                f"window mismatch: the config says {self.window}, the "
                f"connector registered {facts.window_tokens}")
            assert self.window_kind == facts.kind, (
                f"retention kind mismatch: the config says {self.window_kind}, "
                f"the connector registered {facts.kind}")
        return WindowGeometry(
            window=self.window,
            kind=facts.kind or self.window_kind,
            full_attention_group=self.full_attention_group,
            extra_retained_tokens=facts.extra_retained_tokens,
            manager_block_size=self.manager_block_size(),
        )

    def connector_block_tables(self) -> int:
        """How many vLLM block tables (KV-cache groups) the connector got.

        The load-failure reporting mode follows this *actual* shape, not the
        model family: the connector reports invalid blocks to vLLM only with a
        single block table (``len(load_req.all_block_ids) == 1``) and otherwise
        swallows the failure into a log line. A single-group windowed model is
        multi-group by the family predicate yet still reports. Read from the
        connector's own registration line; disagreeing registrations (e.g. a
        restart with a different layout) are a hard error.
        """
        return block_tables_from_logs(
            self.scan_connector_logs(r"block_tables=(\d+)"))

    def assert_pinned_budget(self):
        """Read the connector's pinned total out of the engine logs and gate it.

        No budget for this model's fingerprint (and no KVCM_E2E_PINNED_GIB)
        means no gate -- an invented threshold would fail runs for no reason --
        but it is a *warning*, not a debug note: the run must say out loud that
        it was not gated, otherwise a copied/renamed checkpoint silently loses
        the gate it had before (the log line is what the evidence archive
        lists).
        """
        limit = pinned_budget_gib()
        if limit is None:
            logger.warning(
                "pinned gate SKIPPED: no budget for fingerprint %s of %s; "
                "set KVCM_E2E_PINNED_GIB to gate it",
                model_fingerprint(MODEL_PATH), MODEL_PATH)
            return
        # The pinned line itself has no rank, so take the ranks from the
        # worker-init line each rank prints just before its pool plan (same
        # worker, same start) and require the report *count* to cover every
        # rank of every start: a restart prints a fresh plan per rank, and a
        # duplicated report from one rank must not cover a silent one.
        assert_pinned_ok(
            self.scan_connector_logs(r"total pinned = ([\d.]+) GiB"), limit,
            MODEL_PATH,
            expected_ranks=self.tp_size,
            expected_reports=self.tp_size * self.vllm_starts,
            reported_ranks={int(g[0]) for g in self.scan_connector_logs(
                RANK_LINE_RE)},
            declared_tp={int(g[1]) for g in self.scan_connector_logs(
                RANK_LINE_RE)})

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


@dataclass(frozen=True)
class WindowGeometry:
    """Everything a loaded-region expectation needs, in one place.

    Bundling it is the point: the loaded region depends on four inputs the
    scenarios used to thread by hand (the window, its retention kind, whether a
    full-attention group keeps everything, and the extra retained tokens), and
    forgetting one silently over- or under-counts the expected blocks. The
    token/metric inputs come from the model config for the orchestration
    decisions (they must exist before any engine starts) and are cross-checked
    against the connector's registered retention once it has.
    """

    window: Optional[int]
    kind: Optional[str]
    full_attention_group: bool
    extra_retained_tokens: int
    manager_block_size: int

    @property
    def window_blocks(self) -> int:
        """Manager blocks the window spans (vLLM's ``_contiguous_blocks_for_hit``)."""
        if self.window is None:
            return 0
        return -(-(self.window - 1 + self.extra_retained_tokens)
                 // self.manager_block_size)

    def loaded_blocks(self, matched_tokens: int) -> int:
        """Manager blocks a hit of that size really loads (union over groups)."""
        return loaded_block_count(matched_tokens, self.window,
                                  self.manager_block_size,
                                  self.full_attention_group, self.kind,
                                  self.extra_retained_tokens)

    def window_tail(self, matched_tokens: int) -> int:
        """Manager blocks the windowed groups alone keep of that hit."""
        return window_tail_blocks(matched_tokens, self.window,
                                  self.manager_block_size,
                                  self.extra_retained_tokens, self.kind)


def run_e2e(scenario: str, tp_size: int, num_prompts: int,
            preferred_block_size: int, connector_name: str = "VerifyingConnector",
            expect_verification_failure: bool = False):
    """Run one full save-then-load verification scenario.

    Single-group full-attention models: prefix caching off, one server across
    both phases. Multi-group models (hybrid mamba + attention, or windowed
    attention): prefix caching on (per-group block tables, per-layer windows),
    the vLLM server is restarted between phases so phase 2 loads from KVCM
    instead of hitting the local prefix cache.

    Expected block counts differ per retention semantics: a full-prefix group
    saves the whole prompt and loads everything it matched, while a windowed
    group saves the whole prompt (the connector copies each block in the step
    that produces it -- the historian contract) but only ever gets slots for
    the tail window on a hit. Both the load count and the save/load difference
    are asserted, so a connector that quietly loaded the whole prefix (or
    loaded nothing) fails.

    connector_name selects the connector class inside test_connector.py; the
    mutation meta-test passes "MutatedConnector" and sets
    expect_verification_failure=True to prove the harness detects an injected
    off-by-one in the token translation.
    """
    import torch  # noqa: F401  (ensure torch importable early for clear errors)

    env = ScenarioEnv(scenario, tp_size=tp_size,
                      preferred_block_size=preferred_block_size,
                      connector_name=connector_name)
    multigroup = env.multigroup
    window = env.window
    capture_dir = env.capture_dir
    logger.info("scenario=%s model=%s multigroup=%s hybrid=%s window=%s tp=%d "
                "prompts=%d preferred_bs=%d",
                scenario, MODEL_PATH, multigroup, env.hybrid, window, tp_size,
                num_prompts, env.preferred_block_size)

    try:
        env.start_manager()
        vllm = env.start_vllm(log_suffix="" if not multigroup else "_p1")

        base_prompts = make_base_prompts(num_prompts, env.hybrid, window)
        suffixes = [f" Now answer question {i}: what is 2+2?" for i in range(num_prompts)]
        phase2_prompts = [p + s for p, s in zip(base_prompts, suffixes)]

        # Compute per-prompt expected block counts from the actual tokenization
        # so a silently dropped prompt (or block) fails the run.
        mbs = env.manager_block_size()
        base_tokens = [tokenize(vllm.base_url(), p) for p in base_prompts]
        phase2_tokens = [tokenize(vllm.base_url(), p) for p in phase2_prompts]
        # Saves cover the whole prompt: every completed block is copied in the
        # step that produces it, before vLLM can free it (historian).
        expected_save_blocks = [len(t) // mbs for t in base_tokens]
        # Loads cover the shared token prefix (the base/suffix boundary token may
        # re-merge under tokenization, shortening the shared prefix by one).
        matched_blocks = [
            min(shared_token_prefix_len(b, p2) // mbs, s)
            for b, p2, s in zip(base_tokens, phase2_tokens, expected_save_blocks)
        ]
        # ... and the load covers, per group, the region vLLM gave slots for:
        # every group gets the tail window, and a full-attention group (when
        # the model has one) keeps the whole matched prefix. A connector that
        # loads less than that -- or one that quietly loaded the whole prefix
        # of a purely windowed model -- fails assert_loaded_blocks below, which
        # pins the *set*, not just the count.
        geom = env.window_geometry()
        expected_load_blocks = [geom.loaded_blocks(n * mbs) for n in matched_blocks]
        # The generated prompts are ~2.5 windows long, which only fits a server
        # whose limit accepts them: with a window wider than a third of
        # --max-model-len neither the request nor the save > 2 * window guard
        # can hold. Skip loudly instead of failing an unfixable assertion (or
        # letting the server answer 400).
        if window is not None:
            env.guard_prompt_fits(max(phase2_tokens, key=len), "run_e2e",
                                  max_tokens=4)
        assert all(n >= 1 for n in expected_load_blocks), (
            f"prompts too short to span a manager block (mbs={mbs}): "
            f"{expected_load_blocks}")
        if window is not None:
            window_blocks = geom.window_blocks
            for save, matched in zip(expected_save_blocks, matched_blocks):
                # The saved prefix must cross the window more than twice: a
                # shorter prompt would make the windowed assertions vacuous.
                assert save > 2 * window_blocks, (
                    f"prompt of {save} blocks does not cross the window "
                    f"({window_blocks} blocks) twice; the windowed assertions "
                    f"would be vacuous (raise the prompt length)")
                # ... and the phase-2 prompt must share the whole base at block
                # granularity, which is what makes `matched` the region the
                # expectations below are built from. One block may legitimately
                # be lost at the boundary: the base/suffix junction token can
                # re-merge under tokenization (the same effect
                # shared_token_prefix_len already accommodates), so the guard
                # pins "at least save - 1" and names any loss it sees.
                assert matched >= save - 1, (
                    f"phase-2 prompt shares only {matched} of {save} blocks "
                    f"with the base; it must share the whole base at block "
                    f"granularity (one boundary block may re-merge)")
        logger.info(
            "mbs=%d window=%s expected save blocks=%s matched=%s window tail=%s "
            "loaded blocks=%s",
            mbs, window, expected_save_blocks, matched_blocks,
            [geom.window_tail(n * mbs) for n in matched_blocks],
            expected_load_blocks)

        # ---- Phase 1: fresh prefill -> connector saves -> reference capture.
        logger.info("phase 1: sending %d fresh prompts", num_prompts)
        send_completions(vllm.base_url(), base_prompts)
        capture_failure: Optional[str] = None
        try:
            wait_for_captures(capture_dir, "ref",
                              expected=tp_size * sum(expected_save_blocks),
                              timeout=180)
        except CaptureVerificationError as e:
            if not expect_verification_failure:
                raise
            capture_failure = str(e)

        if capture_failure is not None:
            # The verification already failed at capture time; the rest of the
            # run (prefix wait, phase 2) would only pile unrelated timeouts
            # on top. The budget invariant still applies (design D5), then the
            # catch is the expected verdict.
            env.assert_pinned_budget()
            logger.info(
                "verification failed as expected (capture-side, phase 1): %s",
                capture_failure)
            return

        # The save is committed to the manager asynchronously after the ref
        # capture (which fires when the save is submitted). Wait until the manager
        # actually has the prefix, otherwise phase 2 would find no match.
        for toks, blocks in zip(base_tokens, expected_save_blocks):
            if not wait_for_prefix_cached(env.manager.manager_uri(), env.instance_id,
                                          toks, min_blocks=blocks):
                raise AssertionError("save was not committed to the manager in time")

        # Multi-group models keep prefix caching on, which also populates the
        # local prefix cache; restart vLLM so phase 2 loads from KVCM, not locally.
        if multigroup:
            logger.info("restarting vLLM before phase 2 (clear local prefix cache)")
            vllm = env.restart_vllm(log_suffix="_p2")

        # ---- Phase 2: same prefix + suffix -> connector loads -> loaded capture.
        logger.info("phase 2: sending %d prefix+suffix prompts", num_prompts)
        send_completions(vllm.base_url(), phase2_prompts)
        if capture_failure is None:
            try:
                wait_for_captures(capture_dir, "loaded",
                                  expected=tp_size * sum(expected_load_blocks),
                                  timeout=180)
            except CaptureVerificationError as e:
                if not expect_verification_failure:
                    raise
                capture_failure = str(e)

        if capture_failure is not None:
            # The verification already failed at capture time (phase 2): the
            # region assertions below would only re-report the missing loaded
            # captures. Jump to the verdict, which reports the catch.
            env.assert_pinned_budget()
            logger.info(
                "verification failed as expected (capture-side, phase 2): %s",
                capture_failure)
            return

        # The loaded blocks must be exactly the tail window of each match, on
        # every tp rank -- not merely the right number of blocks.
        expected_keys = set()
        for toks, matched, load in zip(phase2_tokens, matched_blocks,
                                       expected_load_blocks):
            hashes = full_block_hashes(toks, mbs)
            for rank in range(tp_size):
                expected_keys.update(
                    (f"tp{rank}", h) for h in hashes[matched - load:matched])
        assert len(expected_keys) == tp_size * sum(expected_load_blocks)
        assert_loaded_blocks(capture_dir, expected_keys, tp_size)

        report = compare_captures(capture_dir)
        min_matched = tp_size * sum(expected_load_blocks)
        # The pinned-host budget is an invariant of every run that got this far,
        # the mutation meta-test included (design D5): check it before the
        # verdict below so a budget overrun can never hide behind a scenario's
        # expected outcome.
        env.assert_pinned_budget()
        if expect_verification_failure:
            # A capture-side assertion firing is the verification failing at
            # capture time (the null-page guard catches the slot shift before
            # any comparison runs) -- the same expected catch as a mismatched
            # report; a plain capture-count timeout is *not* (it means the
            # path under test never ran).
            if capture_failure is not None:
                logger.info(
                    "verification failed as expected (capture-side): %s",
                    capture_failure)
                return
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
