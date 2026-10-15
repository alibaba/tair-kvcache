"""Process lifecycle for the vLLM e2e harness: the real KVCM manager
binary and the real vLLM OpenAI server, plus ScenarioEnv which owns
their lifetimes for one scenario."""

from lib_utils import (
    MODEL_PATH, CaptureVerificationError, RetentionFacts, assert_loaded_blocks,
    assert_pinned_ok, assert_report_ok, block_tables_from_names,
    block_token_hash, compare_captures, count_captures, find_manager_binary,
    find_python, find_repo_root, free_port, full_block_hashes,
    get_manager_block_size, has_full_attention_group, is_hybrid_model,
    is_mla_model, is_multigroup_model, is_swa_model,
    make_base_prompts, model_fingerprint, retention_facts_from_logs,
    scenario_tp_size, send_completions, window_kind,
    shared_token_prefix_len, tokenize, wait_for_captures,
    wait_for_prefix_cached, wait_http, window_tail_blocks, window_tokens,
)


import glob
import json
import logging
import os
import re
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
#: is (architectures[0], declared window, size tier) from the checkpoint's
#: config, not a directory name: a copy, a rename or a symlinked checkpoint
#: must not turn a hard gate into a silent skip. The window separates
#: same-architecture siblings (gemma-3-4b 1024 vs gemma-3-1b/270m 512) and the
#: size tier -- (num_hidden_layers, hidden_size) -- the ones the window
#: cannot: gemma-3 12b/27b share the 4b's architecture and 1024-token window
#: but need ~3x its pinned budget, so they fall out of this table (a loud
#: WARNING, the same contract as any uncalibrated model; KVCM_E2E_PINNED_GIB
#: gates them until an e2e run calibrates an entry) instead of being killed by
#: the 4b threshold. Only models with a derived budget are gated.
PINNED_BUDGET_GIB = {
    # gemma-3-4b-it: exact pool 0.3125 GiB (1024 x 327_680 B).
    ("Gemma3ForConditionalGeneration", 1024, (34, 2560)): 0.4,
    # gemma-3-1b-it and gemma-3-270m-it: same family and window, separate tiers.
    ("Gemma3ForCausalLM", 512, (26, 1152)): 0.4,
    ("Gemma3ForCausalLM", 512, (18, 640)): 0.4,
    # Qwen3.5-4B: exact pool 0.999 GiB (62 x 17_301_504 B).
    ("Qwen3_5ForConditionalGeneration", None, (32, 2560)): 1.0,
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

#: Did *this* worker's pool plan run? The pinned total carries no rank, and a
#: stream can show two totals from one rank while another never planned at all
#: (the worker-init lines only prove the workers started). The harness's own
#: connector logs this line right after the parent's registration -- i.e. in the
#: very step that builds the pools and prints the total -- so its rank tags the
#: worker that reported, on top of the unkeyed budget check (whose max over all
#: lines is sound either way).
PLAN_LINE_RE = r"VerifyingConnector enabled.*tp_rank=(\d+)"

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
        # multigroup = hybrid OR windowed: both need a restart between the
        # phases, and prefix caching -- except for R-SWA (below). hybrid
        # additionally selects the mamba specific branches (long prompts,
        # recurrent-state assertions);
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
        # R-SWA is the multigroup exception: its vLLM config class
        # (UnlimitedOCRForCausalLMConfig) forbids prefix caching outright --
        # R-SWA decode KV is not a pure causal function of the prefix -- so
        # passing the flag would abort engine startup before any scenario
        # logic, including the scenarios' own later R-SWA skips, could run.
        # Those models stay multigroup (restart between phases, windowed
        # expectations) but start with prefix caching off, the same default
        # their config forces; an *explicit* request is still honored verbatim
        # (a scenario that asks for the flag without skipping R-SWA first
        # fails loudly at startup instead of silently running a different
        # orchestration).
        self.enable_prefix_caching = (
            (self.multigroup and self.window_kind != "rswa_gap")
            if enable_prefix_caching is None else enable_prefix_caching)
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

        Before the first engine start there is nothing registered to read, so
        the config-derived facts stand in (with a log line). *After* a start
        the absence of a registration is a disagreement, not a fallback: the
        orchestration was chosen because the config said "windowed", and
        verifying full-prefix expectations instead would pass every scenario
        without a single windowed group ever being exercised.
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
                assert self.vllm_starts == 0 or self.window is None, (
                    f"config-level detection says {MODEL_PATH} is windowed "
                    f"(kind={self.window_kind}, window={self.window}) but the "
                    f"connector registered no windowed group after "
                    f"{self.vllm_starts} engine start(s); the run would verify "
                    f"full-prefix expectations and never exercise a windowed "
                    f"group")
                logger.info(
                    "no WindowRetention in the connector logs yet; using the "
                    "config-derived retention (kind=%s window=%s)",
                    self.window_kind, self.window)
                return RetentionFacts(kind=self.window_kind,
                                      window_tokens=self.window)
        return self._retention

    def prompt_capacity(self, max_tokens: int) -> int:
        """Prompt tokens a request with that completion budget may use.

        The terms guard_prompt_fits checks, exported for scenarios that
        generate a long candidate on purpose and truncate it themselves:
        truncating to the limit without the margin would leave the gate
        skipping exactly the request that was just made to fit.
        """
        return max(0, self.max_model_len() - max_tokens - MODEL_LEN_MARGIN)

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

    def eagle_dropped_tokens(self) -> int:
        """Tokens vLLM drops from a windowed hit's matched tail (EAGLE/MTP).

        When the engine runs with a speculative group (KVCM_E2E_VLLM_ARGS
        enabling EAGLE/MTP), vLLM drops the last matched *hash* block before
        allocating a windowed hit, so the region the engine serves -- and the
        connector loads for -- ends one vLLM block before the match the
        connector reported; the production scheduler mirrors the same drop in
        ``ConnectorScheduler._window_blocks``. The connector registers the
        skipped drafter group at engine start (``skip eagle group``), and the
        drop unit is vLLM's own block size, from the scheduler-init line.
        Zero when no speculative group is registered (every default run: the
        geometry then degenerates to the match itself).
        """
        if not self.scan_connector_logs(r"skip eagle group \d+"):
            return 0
        sizes = [int(g[0]) for g in self.scan_connector_logs(
            r"scheduler inited.*vllm block size: (\d+)")]
        assert sizes, (
            "the connector skipped an EAGLE/MTP group but no scheduler-init "
            "line reports the vLLM block size; the dropped-block geometry "
            "cannot be built")
        return max(sizes)

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
            # One deployment can mix retention kinds (sliding and chunked-local
            # groups coexist); the config names one kind, and its window has to
            # be among that kind's registered values. The other kinds' tuples
            # are not compared -- they simply join the union.
            kinds = {k for k, _, _ in facts.tuples}
            assert self.window_kind in kinds, (
                f"retention kind mismatch: the config says {self.window_kind}, "
                f"the connector registered {sorted(kinds)}")
            if self.window_kind == "chunked_local":
                # Several sizes can live in one table: the config names one, and
                # it has to be among the registered ones (the union is computed
                # over all of them, see loaded_blocks).
                assert self.window in facts.chunk_sizes, (
                    f"chunk-size mismatch: the config says {self.window}, the "
                    f"connector registered {list(facts.chunk_sizes)}")
            else:
                windows = {w for k, w, _ in facts.tuples
                           if k == self.window_kind}
                assert self.window in windows, (
                    f"window mismatch: the config says {self.window}, the "
                    f"connector registered {sorted(windows)} for "
                    f"{self.window_kind}")
        kind = facts.kind or self.window_kind
        # The EAGLE/MTP drop only exists for *windowed* hits (the production
        # scheduler compensates it in _window_blocks, which is a windowed
        # clamp): a full-attention model's external allocation keeps the
        # whole reported match, so its geometry carries no drop.
        dropped = self.eagle_dropped_tokens() if kind is not None else 0
        return WindowGeometry(
            # The *registered* window, i.e. the widest one for a deployment with
            # several chunk sizes: the guards built on it (window_blocks, the
            # mid-prefix geometry, the window-crossing gates) have to bound the
            # widest local group, not the single size the config names.
            window=facts.window_tokens or self.window,
            kind=kind,
            full_attention_group=self.full_attention_group,
            extra_retained_tokens=facts.extra_retained_tokens,
            manager_block_size=self.manager_block_size(),
            retention_tuples=facts.tuples,
            chunk_sizes=facts.chunk_sizes,
            eagle_dropped_tokens=dropped,
        )

    def connector_block_tables(self) -> int:
        """How many vLLM block tables (KV-cache groups) the connector got.

        The load-failure reporting mode follows this *actual* shape, not the
        model family: the connector reports invalid blocks to vLLM only with a
        single block table (``len(load_req.all_block_ids) == 1``) and otherwise
        swallows the failure into a log line. Read from the registration-time
        shape -- the transfer-group names the connector registers *and* the
        tables it announces skipping (``skip ... group N``): a skipped
        EAGLE/MTP drafter table has no name yet still counts in
        ``all_block_ids``, so names alone would misread that deployment as
        single-table and wait for the block-id report it never sends. Packed
        tables are deduplicated by table index (several buckets share one
        table).
        """
        skipped = {int(g[0]) for g in self.scan_connector_logs(
            r"skip \w+ group (\d+) \(\d+ layers\)")}
        return block_tables_from_names(
            self.scan_connector_logs(r"register_kv_caches done.*"),
            sorted(skipped))

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
        # Each engine start writes its own vllm<suffix>.std* pair, and every
        # rank prints its pool plan once per start. Check the starts
        # independently: a stream that reported every rank in phase 1 must not
        # lend those ranks (or a duplicated line) to a restart whose own rank
        # never reported.
        checked = 0
        for path in sorted(glob.glob(os.path.join(self.vllm_dir,
                                                  "vllm*.std*"))):
            with open(path, errors="replace") as f:
                text = f.read()
            ranks = re.findall(RANK_LINE_RE, text)
            if not ranks:
                # No connector worker in this stream (the stdout half of a
                # server, or a start that never got that far): not a report to
                # gate, and any failure of such a start surfaces elsewhere.
                continue
            checked += 1
            assert_pinned_ok(
                # The pinned line itself has no rank; the worker-init line says
                # which workers started and the harness's own registration line
                # (logged in the same step that plans the pools) says whose plan
                # actually ran.
                re.findall(r"total pinned = ([\d.]+) GiB", text), limit,
                MODEL_PATH,
                expected_ranks=self.tp_size,
                expected_reports=self.tp_size,
                reported_ranks={int(r) for r, _ in ranks},
                declared_tp={int(tp) for _, tp in ranks},
                plan_ranks={int(g[0]) for g in
                            re.findall(PLAN_LINE_RE, text)},
                where=f" ({os.path.basename(path)})")
        assert checked, (
            f"no connector worker registration found under {self.vllm_dir}; "
            f"the pinned gate cannot be evaluated")

    def scan_connector_logs(self, pattern: str) -> list:
        """Regex-scan all vLLM std streams; returns the group tuple per match.

        *Every* match on a line is returned, not just the first: the connector
        prints one line per event, and an event can carry several groups the
        harness needs (a deployment line lists every group's WindowRetention,
        and a model with several windowed groups -- or several chunk sizes --
        only shows them all there).
        """
        out = []
        for path in glob.glob(os.path.join(self.vllm_dir, "vllm*.std*")):
            with open(path, errors="replace") as f:
                for line in f:
                    for m in re.finditer(pattern, line):
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
    #: Every registered registered ``(kind, window, extra)`` as it was
    #: registered (empty when the facts came from the config fallback): one
    #: deployment can mix retention kinds, and their kept regions are only
    #: partly nested, so the union is computed from these tuples per hit.
    retention_tuples: tuple = ()
    #: Every registered chunk size for chunked_local (empty otherwise): kept
    #: for messages and cross-checks; the tails come from ``retention_tuples``.
    chunk_sizes: tuple = ()
    #: Tokens an EAGLE/MTP group makes vLLM drop from a windowed hit's
    #: matched tail (see ScenarioEnv.eagle_dropped_tokens): the loaded region
    #: ends this far before the reported match. Zero without a speculative
    #: group.
    eagle_dropped_tokens: int = 0

    def effective_hit_tokens(self, matched_tokens: int) -> int:
        """Where the loaded suffix of a hit really ends, in tokens.

        With an EAGLE/MTP group registered, vLLM drops the last matched hash
        block before allocating the windowed hit (the production scheduler
        demands one vLLM block more of published data for exactly this,
        ``ConnectorScheduler._window_blocks``): the hit the engine serves --
        and the connector loads for -- ends ``eagle_dropped_tokens`` before
        the match the connector reported. Without a speculative group this is
        the match itself.
        """
        return max(0, matched_tokens - self.eagle_dropped_tokens)

    def hit_end_blocks(self, matched_tokens: int) -> int:
        """Manager block the loaded suffix of a hit really ends at.

        The scenarios' key slices end here, not at the reported match: with
        EAGLE/MTP the dropped hash block is never loaded, so ending at the
        match would expect a shifted region and fail a correct run.
        """
        return self.effective_hit_tokens(matched_tokens) // self.manager_block_size

    @property
    def prefill_unbounded(self) -> bool:
        """Does a registered group keep its whole prefill (R-SWA)?

        R-SWA evicts only the *decode* gap, so no prefill length bounds the
        retained region and guards like "the prefix crossed the retained span"
        cannot be decided from the prompt alone -- the caller has to skip (or
        decode past the R-SWA window) instead of trusting a finite span.
        """
        if self.retention_tuples:
            return any(kind == "rswa_gap" for kind, _, _ in self.retention_tuples)
        return self.kind == "rswa_gap"

    @property
    def retained_span_tokens(self) -> int:
        """Tokens the widest *bounded* local group can keep of a hit.

        ``window_blocks`` rounds this *up* to manager blocks, which is what the
        block-level guards want; the token-level condition "at least one block
        was freed" is ``H >= retained_span_tokens + manager_block_size`` (a hit
        only starts freeing blocks once it exceeds the span by a full block).
        Chunked-local contributes ``chunk - 1`` (its kept part is ``H % chunk``,
        never a whole chunk) and R-SWA contributes nothing at all: its prefill
        retention is unbounded, see ``prefill_unbounded``.
        """
        if self.retention_tuples:
            spans = [(w - 1 + e) if kind not in ("chunked_local", "rswa_gap")
                     else (w - 1) for kind, w, e in self.retention_tuples
                     if kind != "rswa_gap"]
            return max(spans) if spans else 0
        if self.window is None or self.kind == "rswa_gap":
            return 0
        return self.window - 1 + self.extra_retained_tokens

    @property
    def window_blocks(self) -> int:
        """Manager blocks the widest local group spans of one window.

        vLLM's ``_contiguous_blocks_for_hit`` for one group, maximized over the
        registered tuples: the guards built on it (the mid-prefix geometry, the
        "save crosses the window twice" assertion, the cross-request crossing
        gate) have to bound every local group, and a single tuple's window must
        never be recombined with another's extra.
        """
        if self.retention_tuples:
            spans = [(-(-w // self.manager_block_size)
                      if kind == "chunked_local"
                      else -(-(w - 1 + e) // self.manager_block_size))
                     for kind, w, e in self.retention_tuples]
            return max(spans)
        if self.window is None:
            return 0
        return -(-(self.window - 1 + self.extra_retained_tokens)
                 // self.manager_block_size)

    def local_tail(self, matched_tokens: int) -> int:
        """Manager blocks the *windowed* groups alone keep of that hit.

        Never includes a full-attention group's whole match: ``loaded_blocks``
        adds that on top for the load expectation, while the scenarios that
        reason about the windowed groups' own coverage -- the mid-prefix
        boundary, which must not land where a chunked-local group keeps
        nothing -- need this region by itself. Every kept region is a *suffix*
        of the hit, so the union over the registered tuples is the widest of
        their tails, evaluated per hit length (which is larger depends on the
        hit: sliding tails nest by span, chunked-local ones align to chunks).
        The hit is the *effective* one (see ``effective_hit_tokens``): with an
        EAGLE/MTP group the window vLLM allocates ends a hash block before
        the reported match, and the tail follows that end.
        """
        hit = self.effective_hit_tokens(matched_tokens)
        if self.retention_tuples:
            return max(window_tail_blocks(hit, w,
                                          self.manager_block_size,
                                          extra_retained=e, kind=kind)
                       for kind, w, e in self.retention_tuples)
        return window_tail_blocks(hit, self.window,
                                  self.manager_block_size,
                                  self.extra_retained_tokens, self.kind)

    def loaded_blocks(self, matched_tokens: int) -> int:
        """Manager blocks a hit of that size really loads (union over groups).

        The union of the windowed groups' tails (``local_tail``) and, when the
        model has one, a full-attention group's whole matched prefix -- which
        is what bounds the union for every windowed kind. Both are evaluated
        at the hit's effective end: an EAGLE/MTP drop moves the whole served
        region back by the dropped hash block.
        """
        hit = self.effective_hit_tokens(matched_tokens)
        if self.full_attention_group:
            return -(-hit // self.manager_block_size)
        return self.local_tail(matched_tokens)

    def window_tail(self, matched_tokens: int) -> int:
        """Manager blocks the windowed groups alone keep of that hit."""
        return self.local_tail(matched_tokens)


def run_e2e(scenario: str, tp_size: int, num_prompts: int,
            preferred_block_size: int, connector_name: str = "VerifyingConnector",
            expect_verification_failure: bool = False):
    """Run one full save-then-load verification scenario.

    Single-group full-attention models: prefix caching off, one server across
    both phases. Multi-group models (hybrid mamba + attention, or windowed
    attention): prefix caching on (per-group block tables, per-layer windows),
    the vLLM server is restarted between phases so phase 2 loads from KVCM
    instead of hitting the local prefix cache. R-SWA checkpoints are the
    exception on the flag: their config class forbids prefix caching, so they
    keep the multigroup orchestration (restart, windowed expectations) with
    prefix caching off.

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
        geom = env.window_geometry()
        if window is not None:
            # The guard below asserts "save crosses the *registered* window
            # twice", i.e. the widest local group -- and the config-level window
            # used to size the prompt can be narrower (a deployment may register
            # several chunk sizes). Extend the generated text until the
            # tokenized prompts really cross that window twice: the same
            # dynamic sizing the mid-prefix scenario does for its three windows.
            base_tokens = [tokenize(vllm.base_url(), p) for p in base_prompts]
            extension = 0
            while (extension < 32 and min(len(t) // mbs for t in base_tokens)
                   <= 2 * geom.window_blocks):
                extension += 1
                base_prompts = [
                    p + " " + " ".join(
                        f"Extended part {extension} sentence {j} of prompt {k} "
                        f"carries value {j * 23 + extension * 89 + k * 7}."
                        for j in range(40))
                    for k, p in enumerate(base_prompts)]
                base_tokens = [tokenize(vllm.base_url(), p)
                               for p in base_prompts]
            if extension:
                logger.info(
                    "extended the prompts by %d part(s) to cross the "
                    "%d-block registered window twice", extension,
                    geom.window_blocks)
        base_tokens = [tokenize(vllm.base_url(), p) for p in base_prompts]
        phase2_prompts = [p + s for p, s in zip(base_prompts, suffixes)]
        phase2_tokens = [tokenize(vllm.base_url(), p) for p in phase2_prompts]
        if window is not None and geom.chunk_sizes:
            # A chunked-local group keeps nothing when the *hit it is served*
            # ends exactly on one of its chunk boundaries (vLLM skips every
            # matched token). The hit is the shared prefix rounded down to
            # manager blocks, less any EAGLE/MTP drop (the served region ends
            # a hash block before the reported match), not the raw prompt
            # length: mbs=16 with a chunk of 128 turns a 390-token base into a
            # 384-token match, right on the boundary. So test the rounded
            # match at its effective end (suffix re-merge included) and nudge
            # the prompt until it lands inside every chunk (the same kind of
            # adjustment the mid-prefix scenario makes).
            def _hit_tokens(b: list, p2: list) -> int:
                save = len(b) // mbs
                return geom.effective_hit_tokens(
                    min(shared_token_prefix_len(b, p2) // mbs, save) * mbs)

            nudges = 0
            while (nudges < 8 and any(
                    _hit_tokens(b, p2) % chunk == 0
                    for b, p2 in zip(base_tokens, phase2_tokens)
                    for chunk in geom.chunk_sizes)):
                nudges += 1
                # Each step appends `nudges` distinct words, so the token count
                # moves by a different amount every time -- a fixed suffix could
                # keep landing on the same boundary (a chunk size dividing its
                # token count would never be escaped).
                extra = " ".join(f"Realign{nudges}x{j}"
                                 for j in range(nudges))
                base_prompts = [f"{p} {extra}" for p in base_prompts]
                base_tokens = [tokenize(vllm.base_url(), p)
                               for p in base_prompts]
                phase2_prompts = [p + s
                                  for p, s in zip(base_prompts, suffixes)]
                phase2_tokens = [tokenize(vllm.base_url(), p)
                                 for p in phase2_prompts]
            if nudges:
                logger.info("nudged the match off the chunk boundaries in %d "
                            "step(s)", nudges)
            if any(_hit_tokens(b, p2) % chunk == 0
                   for b, p2 in zip(base_tokens, phase2_tokens)
                   for chunk in geom.chunk_sizes):
                # Still on a boundary after the cap: the chunk is at most one
                # manager block wide, so *no* match can keep a block for it --
                # the windowed loaded region is empty by construction and the
                # exact-set assertion below would be vacuous.
                raise unittest.SkipTest(
                    f"no match length keeps a block for chunk size(s) "
                    f"{sorted(geom.chunk_sizes)} at mbs={mbs} (a chunk that "
                    f"small is inside one manager block); the windowed "
                    f"loaded-region assertion cannot be built")
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
            local_tails = [geom.window_tail(n * mbs) for n in matched_blocks]
            assert all(t >= 1 for t in local_tails), (
                f"matched prefixes keep no windowed block (windowed tails "
                f"{local_tails} of matches {matched_blocks}); the union below "
                f"would be carried by a full-attention group alone")
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
        # A capture-side assertion here is *not* a verdict, not even for the
        # mutation meta-test: phase 1 only writes references, read through
        # vLLM's own slot mapping, before any loaded or null-page verification
        # runs -- and the injected shift affects the connector's translation,
        # which phase 1 does not exercise. So it means the capture harness
        # itself is broken and always fails the run.
        wait_for_captures(capture_dir, "ref",
                          expected=tp_size * sum(expected_save_blocks),
                          timeout=180)
        capture_failure: Optional[str] = None

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
        # every tp rank -- not merely the right number of blocks. The region
        # ends at the hit's *effective* end (EAGLE/MTP drops the last matched
        # hash block of a windowed hit, see WindowGeometry.hit_end_blocks).
        expected_keys = set()
        for toks, matched, load in zip(phase2_tokens, matched_blocks,
                                       expected_load_blocks):
            hashes = full_block_hashes(toks, mbs)
            end = geom.hit_end_blocks(matched * mbs)
            for rank in range(tp_size):
                expected_keys.update(
                    (f"tp{rank}", h) for h in hashes[end - load:end])
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
