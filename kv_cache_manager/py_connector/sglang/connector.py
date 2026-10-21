import ctypes
import dataclasses
import functools
import hashlib
import logging
import os
import threading
import uuid
from collections import Counter
from types import MappingProxyType
from typing import AbstractSet, Any, Dict, List, Mapping, Optional
import time
import json

import requests
import torch
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorage,
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
    PoolName,
    PoolTransfer,
    PoolTransferResult,
    PoolHitPolicy,
)
from sglang.srt.mem_cache.memory_pool_host import HostKVCache

try:
    from sglang.srt.observability.metrics_collector import StorageMetrics
except ImportError:
    try:
        # Older sglang versions kept StorageMetrics here.
        from sglang.srt.metrics.collector import StorageMetrics  # ty: ignore[unresolved-import]
    except ImportError as e:
        raise ImportError(
            "Cannot import StorageMetrics from sglang. "
            "Tried sglang.srt.observability.metrics_collector and "
            "sglang.srt.metrics.collector. "
            "Please check your sglang version is compatible."
        ) from e

# get_attn_tp_group lives in sglang.srt.distributed since v0.5.9; the old
# alias sglang.srt.layers.dp_attention.get_attention_tp_group was removed in
# v0.5.16.
from sglang.srt.distributed import get_attn_tp_group, get_tp_group
from sglang.srt.layers.dp_attention import is_dp_attention_enabled

from kv_cache_manager.py_connector.common.manager_client import (
    KvCacheManagerClient,
    KvCacheManagerHTTPError,
    KvCacheManagerProtocolError,
)

# kvcm_py_client is the compiled pybind11 client; it ships no type stubs.
from kv_cache_manager.client.pybind import kvcm_py_client  # ty: ignore[unresolved-import]

# Stamped into the wheel at build time; absent in a source checkout.
from kv_cache_manager.py_connector.common._version_info import (  # ty: ignore[unresolved-import]
    FULL_VERSION,
    GIT_COMMIT,
    BUILD_TIME,
)

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class _PoolSpec:
    """How one sglang pool maps onto the location specs of this connector.

    ``_POOL_SPECS`` is the connector's knowledge of the pools sglang shipped
    before it: ``_init_kvcm_client`` derives the specs, the spec groups, the iov
    size and the SDK client config from the resolved rows, and the data path
    resolves a ``PoolTransfer`` through them.  A pool without a row here is
    derived from itself (``_derived_pool_spec``), so a new sglang side pool is
    backed without a connector change.
    """

    # sglang's PoolName member: the key of ``registered_pools`` and of a
    # PoolTransfer.
    pool: Any
    # Location-spec-group name.  A KVCM write session covers one group, so the
    # pools of one group are written together.  Each row emits one group entry,
    # so a row's group has to differ from every other row's: the Manager resolves
    # a group name to a single group and silently skips a duplicate's specs.
    group: str
    # Appended to "tp_{rank}" to name the pool's location specs.
    suffix: str
    # The anchor's suffix once a sidecar pool is registered as well.
    hybrid_suffix: Optional[str] = None
    # The KV pool: sglang hands exactly one per instance, and a flat model keeps
    # its historic "tp_{rank}" spec names so already-cached data stays readable.
    anchor: bool = False
    # The row was derived from a registered pool instead of coming from
    # ``_POOL_SPECS``.  A derived pool is backed like any other, but it does not
    # size the shared IOV area the HF3FS backend allocates at startup (see
    # ``_init_kvcm_client``): that area is paid for by every deployment,
    # including the ones this connector backed before the pool existed here.
    derived: bool = False
    # The pool is replicated rather than sharded across the TP ranks: an MLA
    # model writes rank 0's copy alone, so every rank reads and writes rank 0's
    # spec.  A sharded pool is the opposite -- sglang hands it to every rank,
    # which then each write their own slice into their own rank spec (upstream
    # HybridCacheController.should_backup(): Mamba/KDA only since v0.5.18, plus
    # mooncake's MHA draft pools, which this connector does not manage).
    shared_across_ranks: bool = True

    def spec_name(self, tp_rank: int, hybrid_stack: bool) -> str:
        """The location spec name of this pool on one TP rank."""
        suffix = self.suffix
        if hybrid_stack and self.hybrid_suffix is not None:
            suffix = self.hybrid_suffix
        return f"tp_{tp_rank}{suffix}"

    def spec_size(self, pool: Any) -> int:
        """Bytes of one block of this pool, counted by the pool itself.

        One page is ``page_size`` slots of ``get_size_per_token()`` bytes: a
        token-granular pool (the KV family, the indexer) multiplies, and a
        page-granular one (Mamba, whose host pool keeps one slot per page)
        reports ``page_size == 1`` and comes out as the identity.  Every host
        pool of one instance is built with the controller's page size, so the
        indexer's product equals the anchor's (the basis the table used before
        this refactor) and no deployment changes its registered sizes.

        sglang's DeepSeek-V4 host pools do not satisfy this: they report the
        bytes of one *page row* as their per-token size, so the product would be
        ``page_size`` times too large.  Such a pool only ever exists on an
        instance whose anchor is a ``LogicalHostPool``, which is refused before
        any row is resolved, so no row here is one of them.
        """
        return pool.get_size_per_token() * pool.page_size


_KV_SPEC = _PoolSpec(
    pool=PoolName.KV,
    group="Full",
    suffix="",
    hybrid_suffix="_full",
    anchor=True,
)
_MAMBA_SPEC = _PoolSpec(
    pool=PoolName.MAMBA,
    group="Linear",
    suffix="_linear",
    shared_across_ranks=False,
)
# Probed: a release without the enum member has no such pool to register, and
# the row then drops out of the table (naming only reads ``suffix``).
_INDEXER_SPEC = _PoolSpec(
    pool=getattr(PoolName, "INDEXER", None),
    group="Indexer",
    suffix="_indexer",
)

# In registration order: the anchor first, so the KV specs lead
# ``location_spec_infos`` and the KV group leads ``location_spec_groups``.
_POOL_SPECS: tuple[_PoolSpec, ...] = tuple(
    spec for spec in (_KV_SPEC, _MAMBA_SPEC, _INDEXER_SPEC) if spec.pool is not None
)


def _validate_pool_specs(table: tuple[_PoolSpec, ...]) -> None:
    """Reject a row set that would alias two pools onto one location spec.

    A row that breaks one of these invariants is silent in production: the
    Manager is handed two specs with the same name and different sizes (the SDK
    client keeps one of them), or it resolves a group name to a single group and
    skips the duplicate's specs.  Either way one pool's pages are read or
    written under another pool's identity and the only symptom is a cache that
    does not work.  Running this once at import turns a bad future row into an
    immediate failure instead of that.
    """

    def require(ok: bool, message: str) -> None:
        if not ok:
            raise ValueError(message)

    require(bool(table), "the table must hold at least the KV anchor")
    require(
        all(spec.pool is not None for spec in table),
        "a row was built without a PoolName member (it should not enter the "
        f"table at all): {table}",
    )

    pools = [spec.pool for spec in table]
    require(len(set(pools)) == len(pools), f"a pool has more than one row: {pools}")

    # Every spec suffix a row can emit -- its own, and the anchor's hybrid one
    # -- has to identify exactly one row, or two pools resolve to one spec.
    require(
        all(spec.suffix is not None for spec in table),
        f"a row has no suffix, its spec name would end in 'None': {table}",
    )
    emitted = [spec.suffix for spec in table] + [
        spec.hybrid_suffix for spec in table if spec.hybrid_suffix is not None
    ]
    require(
        len(set(emitted)) == len(emitted),
        f"two rows can emit the same spec suffix: {emitted}",
    )

    anchors = [spec for spec in table if spec.anchor]
    require(len(anchors) == 1, f"expected exactly one anchor row, got {len(anchors)}")
    anchor = anchors[0]
    require(anchor.pool is PoolName.KV, f"the anchor is the KV pool, not {anchor.pool}")
    # Every row needs a group: the Manager treats an empty group name as "no
    # constraint", drops that row's specs and leaves the pool permanently
    # missing -- the same silent failure a duplicate group causes.
    require(
        all(spec.group for spec in table),
        f"a row has an empty spec group: {[spec.group for spec in table]}",
    )
    # One row emits one group entry, so two rows sharing a group hand the
    # Manager a duplicate name: it resolves the name to a single group and skips
    # the other row's specs without an error.  (This also covers "a sidecar
    # must not reuse the anchor's group".)
    groups = [spec.group for spec in table]
    require(len(set(groups)) == len(groups), f"two rows share a spec group: {groups}")


# CI does not run this package's tests (they need a real sglang runtime), so the
# table is checked where it is defined rather than only from a test.
_validate_pool_specs(_POOL_SPECS)


def _pool_spec(pool_name: Any) -> Optional[_PoolSpec]:
    """The table row of a pool, or None when the table does not know it."""
    return next((spec for spec in _POOL_SPECS if spec.pool == pool_name), None)


# The Manager reads a location-spec group's hit semantics off the first
# character of the group name (manager/meta_searcher.cc::ClassifySpecGroups):
# 'F' requires every page of a prefix to carry the group's specs -- what
# PoolHitPolicy.ALL_PAGES asks for -- and 'L' only the trailing boundary page
# (TRAILING_PAGES); every other character is rejected outright.
#
# A registration cannot see a pool's hit policy: sglang hands a pool over as
# (name, host pool) and only a PoolTransfer carries ``hit_policy``, long after
# ``registerInstance`` froze the specs, with no update API behind it.  A derived
# row therefore claims the strict category -- the only choice that cannot hand
# out pages the pool does not have: over-strict costs hits, over-permissive
# would claim data.  The connector's own existence check follows the transfer's
# ``hit_policy``, never this name, and a pool whose policy is known belongs in
# ``_POOL_SPECS`` with the category that fits it.
_DERIVED_GROUP_PREFIX = "F"

# How many blocks of the largest pool the shared IOV area of an HF3FS backend
# must hold: the historical sizing of ``iov_size``, kept as the default because
# every deployment pays for that area twice at startup (see
# ``HiCacheKVCM._init_kvcm_client``).
_DEFAULT_IOV_BLOCK_COUNT = 1024

# The two storage types that are backed by the mempool SDK, as they appear in
# the Manager's ``storage_configs`` JSON: ``StorageConfig::ToRapidWriter`` writes
# ``ToString(DataStorageType)``, which spells DATA_STORAGE_TYPE_TAIR_MEMPOOL
# "pace" and DATA_STORAGE_TYPE_TAIR_MEMPOOL_SSD "pace_ssd"
# (``data_storage/storage_config.cc``).
_TAIR_MEMPOOL_STORAGE_TYPES = ("pace", "pace_ssd")


def _has_tair_mempool_storage(storage_configs: Any) -> Optional[bool]:
    """Whether a Tair mempool backend is among these storage configs.

    Reads the ``type`` of every entry of the Manager's ``storage_configs`` --
    the same JSON this connector hands to the client as
    ``InitParams.storage_configs``.  ``None`` means the list could not be read,
    which is not the same as "there is none": the caller may not treat an
    unreadable list as a consumer.
    """
    try:
        configs = (
            json.loads(storage_configs)
            if isinstance(storage_configs, str)
            else storage_configs
        )
        return any(
            isinstance(config, dict)
            and config.get("type") in _TAIR_MEMPOOL_STORAGE_TYPES
            for config in configs
        )
    except Exception:
        return None


# A transport-ish failure that says the Manager was not able to judge the
# registration: the leader election is still moving, and the client has already
# retried.  Its own text is the only signal the Manager client forwards (the
# status code is not carried on the exception), so it is matched as a substring
# and only to keep such a failure out of the "your layout was rejected" branch.
# A non-OK business status arrives as a bare ``AssertionError`` carrying only the
# Manager's message (``common/manager_client.py``), and the Manager can fail for a
# reason that a retry fixes -- persisting the instance info or the instance key to
# its storage backend, for instance.  So the default is ``unjudged``: only a
# message that positively identifies a permanent refusal is diagnosed as one.
_REGISTER_REJECTED_HINTS = (
    "mismatched fields",
    "already exists",
    "not found",
    "invalid",
    "unavailable",
)


def _collective_identity(name: Any) -> str:
    """The identity a pool has in the group's agreed collective sequence.

    The pool's text, which is this connector's pool identity everywhere else as
    well: ``registered_pools``, ``pool_specs`` and the ``results`` dict are all
    keyed by it, and sglang itself reads a result back as ``results.get(name)``
    then ``results.get(name.value)`` (``hybrid_cache_controller._page_backup``).
    The text is unambiguous because ``PoolName`` is a ``str`` enum -- ``hash``
    and ``==`` both come from the value, so ``PoolName.MAMBA`` and ``"mamba"``
    are *one* dictionary key, not two pools -- and ``_resolve_pool_specs``
    refuses to derive a row for a text a table row already claims.

    The spec name cannot be the identity instead: a sharded pool's spec is
    ``tp_{own_rank}...``, so it differs per rank, while the sequence has to be
    one key every rank spells the same.
    """
    return str(name)


def _derived_pool_spec(pool_name: Any) -> _PoolSpec:
    """The row of a pool that is not in ``_POOL_SPECS``, derived from itself.

    The pool name is all a registration carries and is unique per instance, so
    it prefixes both the spec suffix and the spec group.

    ``shared_across_ranks`` stays at its default (replicated), which is also
    sglang's own default: on an MLA model ``should_backup`` hands every pool
    except the sharded Mamba state to rank 0 alone, so the rank-0 protocol is the
    one that never waits for a peer that will not be called.  A pool that really
    is sharded across the TP ranks must therefore be a table row with
    ``shared_across_ranks=False``: a derived row is always treated as replicated,
    and treating a sharded pool as replicated makes every rank load rank 0's
    shard.
    """
    name = getattr(pool_name, "value", pool_name)
    return _PoolSpec(
        pool=pool_name,
        group=f"{_DERIVED_GROUP_PREFIX}{name}",
        suffix=f"_{name}",
        derived=True,
    )


@dataclasses.dataclass(frozen=True)
class _RegisteredPool:
    """A ``_PoolSpec`` resolved against the pools one instance registered."""

    spec: _PoolSpec
    # Bytes of one block, as frozen into the Manager registration.
    size: int
    # The pool's location spec names, indexed by TP rank.
    names: tuple[str, ...]
    # The rank whose spec this instance reads and writes for the pool.
    own_rank: int

    @property
    def name(self) -> str:
        """The location spec this instance reads and writes for the pool."""
        return self.names[self.own_rank]


@functools.lru_cache(maxsize=1)
def _sidecars_are_written_by_every_rank() -> bool:
    """Whether this sglang hands rank-sharded sidecars to every TP rank.

    ``HybridCacheController`` only started to run the whole backup path on
    every rank in v0.5.17: from there it defines its own
    ``backup_thread_func`` (hybrid_cache_controller.py:718 in v0.5.17, :739 in
    v0.5.19), which no longer inherits the base controller's ``backup_skip``
    gate (managers/cache_controller.py:1222 in v0.5.16), and filters the
    transfers down to the rank-owned pools (``should_backup`` since v0.5.18,
    an inline Mamba filter in v0.5.17).  Before that, a non-zero MLA rank never
    reached ``batch_set_v2`` at all, so a write that waits for it there would
    hang.

    Structural probe rather than a version comparison on purpose: an upstream
    shape we do not recognise falls back to the rank-0-only protocol -- safe,
    just unable to reuse the sidecar of a sharded pool -- instead of waiting
    for ranks that may never arrive.  Only a hybrid stack defines this class,
    and only a hybrid stack has sidecar pools to write.
    """
    try:
        from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
            HybridCacheController,
        )
    except ImportError:
        return False
    return "backup_thread_func" in vars(HybridCacheController)


class _UnsupportedBackendError(RuntimeError):
    """The connector cannot serve the pools and configuration it has.

    Raised by ``_ensure_client`` for a shape or configuration this connector
    cannot serve: a KV anchor that is not a host pool (sglang's DeepSeek-V4
    compressed pool) or an SDK client this configuration cannot create.  A
    request cannot fix either -- the pool set and the registration are frozen --
    so the data-path entries answer conservatively instead of failing the
    request, and the failure is reported once when it is detected
    (``_report_unsupported_kv_pool`` / ``_report_unusable_sdk_client``) so they
    need no log of their own.

    Only a later one-shot attempt can still succeed, and ``_ensure_client`` does
    re-attempt it on its next call: that is how a transfer runtime which failed
    to start for an environmental reason recovers.  ``finalize_registration``
    gets no second attempt -- the engine detaches the backend when it raises.

    ``reason`` is the operator-facing classification reported at startup, one
    phrase per cause, so "the transfer runtime could not be created" never reads
    as "the pool shape is not supported".
    """

    def __init__(
        self,
        message: str,
        *,
        reason: str,
    ) -> None:
        super().__init__(message)
        self.reason = reason


# Report key of the instance-wide failure "this instance has no SDK client".
# An opaque object, so it can neither be suppressed by nor suppress a pool's
# report in ``_reported_pools``.
_SDK_CLIENT_REPORT_KEY: Any = object()


def _register_failure_cause(error: BaseException) -> str:
    """Which of ``rejected`` / ``refused`` / ``unjudged`` a failure is.

    ``_report_register_failure`` explains why the exception *types* decide this
    and ``AssertionError`` cannot be the predicate; both the data-path report and
    the finalize label read this one classification, so one failure is never
    labelled two ways.
    """
    if isinstance(error, KvCacheManagerProtocolError):
        return "unjudged"
    if isinstance(error, KvCacheManagerHTTPError):
        status = getattr(getattr(error, "response", None), "status_code", None)
        return "refused" if status is not None and 400 <= status < 500 else "unjudged"
    if isinstance(error, AssertionError):
        message = str(error).lower()
        if any(hint in message for hint in _REGISTER_REJECTED_HINTS):
            return "rejected"
        # Not identifiable: the Manager refused for a reason this connector
        # cannot name, and retrying may still succeed.
        return "unjudged"
    return "unjudged"


def _describe_registration_failure(error: BaseException) -> str:
    """Label a failed one-shot registration for the operator's log line.

    The label reuses ``_register_failure_cause``, so a failure that the
    data-path report already called ``unjudged`` is not called ``permanent`` one
    line later.  Nothing here decides anything: the exception is raised
    unchanged, and whether a later attempt can succeed depends on the caller --
    the data path retries per call, the finalize hook does not.
    """
    if isinstance(error, _UnsupportedBackendError):
        return f"rejected: {error.reason}"
    if isinstance(error, KvCacheManagerProtocolError):
        return "unjudged: the Manager did not decide on the registration"
    if isinstance(error, KvCacheManagerHTTPError):
        status = getattr(getattr(error, "response", None), "status_code", None) or 0
        if 400 <= status < 500:
            return f"refused: the Manager rejected the request (HTTP {status})"
        return f"unjudged: the Manager failed with HTTP {status}"
    if isinstance(error, requests.RequestException):
        return "unjudged: the Manager is unreachable"
    if _register_failure_cause(error) == "rejected":
        return "rejected: the Manager refused the registration"
    # Not a Manager decision: the instance itself cannot serve (closed backend,
    # no KV pool registered yet).  A retry with the same setup cannot fix it.
    return f"local: {type(error).__name__}: this instance cannot be served"


class HiCacheKVCM(HiCacheStorage):
    """KVCM HiCache storage backend for sglang (v0.5.10 ~ v0.5.19).

    Which pools it backs, and how each of them maps onto KVCM location specs, is
    the module-level ``_POOL_SPECS`` table plus one row derived from every
    registered host pool the table does not know; everything per instance (spec
    names, spec sizes, groups, iov size, SDK client config) is derived from those
    rows by ``_resolve_pool_specs``.

    Pool registration contract -- what sglang hands the backend, in order:

    * ``register_mem_pool_host`` (v1) runs first on every attach. It carries
      the KV host pool for flat models on all supported versions and, since
      v0.5.19, for every model (sglang passes ``HostPoolGroup.anchor_entry.host_pool``).
      Up to v0.5.18 hybrid stacks pass the ``HostPoolGroup`` itself instead;
      a group is not a ``HostKVCache``, so its KV entry reaches the connector
      through ``register_mem_host_pool_v2`` right after.
    * ``register_mem_host_pool_v2`` carries one host pool per call and is the
      only source of sidecar pools (Mamba/Indexer/...). sglang calls it for
      every entry of the group, KV included.

    The location specs are frozen when the Manager registration and the SDK
    client are created (``_ensure_client``): ``registerInstance`` only accepts a
    repeat when every field matches and there is no spec update API.  sglang
    that knows the duck-typed ``finalize_registration`` hook says when the
    attach (draft pools included) is over -- before any request is served --
    and the hook is then the freeze point; a failure there fails the attach
    instead of degrading every storage call to a miss in a background thread.
    Upstream sglang has no such callback, so there the pool set is only final at
    the first data-path call -- the behavior of every release without the hook
    is unchanged.  Either way a pool that the v2 hook registers after the
    freeze is reported once, not recorded, and marked unusable: even if it
    reuses the name of an already-registered pool, its transfers are answered
    with misses instead of being paired with that pool's spec.  It is never
    silently assumed to work.  (The v1 hook runs once per attach -- it carries
    the KV anchor, or up to v0.5.18 the whole host pool group -- and has no
    late-pool bookkeeping of its own, so a re-attach has to build a new backend
    instance, which sglang does.)

    That also makes the pool set part of the deployment identity: a pool this
    connector backs after an upgrade but did not back before changes the
    registered specs, and the Manager refuses such a repeat under the same
    ``instance_id``.  Changing the pool set is therefore an operator action --
    a new ``instance_id`` -- not something this connector can repair at runtime.

    If the instance cannot be served at all, initialization reports it once and
    then fails: an anchor that is not a host pool (sglang's DeepSeek-V4
    compressed KV pool anchors on a tensorless ``LogicalHostPool``) or an SDK
    client the frozen configuration cannot create.  A data-path call then
    degrades to a miss, while ``finalize_registration`` fails the attach.  A
    sidecar registered under such a name, or one whose derived spec name would
    share another pool's spec, is simply not backed: its transfers are answered
    with a miss, never with someone else's data.

    The write paths are the only callers of the connector's TP collectives,
    and those need every rank of the group.  They therefore agree with the
    group on initialization before entering any of them: a rank that cannot
    create its client publishes that instead of returning, and the whole group
    then refuses the write (``_ensure_client_for_write``).  Reads are
    rank-local and never wait for another rank.

    Which ranks write what follows sglang's own split: an MLA KV pool is
    replicated and written by rank 0 alone, while a hybrid model's rank-sharded
    sidecars (Mamba/KDA state, e.g. Kimi-K3) are handed to every rank, which
    then each write their own slice into their own rank spec.  A sidecar write
    that spans the group is committed only for the blocks every rank wrote.
    """

    # Report keys already reported -- the pools that cannot be served (unknown to
    # this connector, registered after the specs were frozen, or named so that
    # their spec would alias another pool's) and the instance-wide failures
    # (``_SDK_CLIENT_REPORT_KEY``).  Process-wide so each key is reported once
    # instead of once per block batch, and so the report survives a backend
    # re-creation.
    _reported_pools: set = set()

    # Legacy entry points (batch_get/batch_set) already reported as not
    # implemented; same process-wide, report-once reasoning.
    _warned_legacy_ops: set = set()

    # Defaults for instances built without __init__ (tests build connectors by
    # hand to drive one code path); a real backend sets these in __init__.
    _client_ready: bool = False
    _closed: bool = False
    # Read by the write paths before the client exists, hence the default; a
    # real backend sets it from the storage config in __init__.
    is_mla_model: bool = False
    # Whether a round of _ensure_client_for_write() has already seen every
    # rank of the TP group ready (see that method).
    _tp_init_agreed: bool = False
    # Immutable so hand-built instances (tests bypass __init__) can read it
    # without sharing one mutable object: late names are added by rebinding,
    # never by mutating a shared set.
    _late_pools: AbstractSet[Any] = frozenset()
    # The spec table resolved against the pools this instance registered, the
    # anchor first; filled by _init_kvcm_client.  Read-only, and empty (never
    # shared mutable state) for instances built without it.
    pool_specs: Mapping[Any, _RegisteredPool] = MappingProxyType({})

    def __init__(self, storage_config: HiCacheStorageConfig, kwargs: Any) -> None:
        logger.warning(
            "KVCM sglang connector version: %s (commit: %s, build: %s)",
            FULL_VERSION,
            GIT_COMMIT,
            BUILD_TIME,
        )
        self.storage_config = storage_config
        # --hicache-storage-backend-extra-config '{"k":"v"}'
        # HiCacheStorageConfig types extra_config as Optional; sglang always
        # provides it for a hicache storage backend.
        self.extra_config: Dict[str, Any] = self.storage_config.extra_config or {}

        # deployment
        self.instance_group = self.extra_config["instance_group"]
        self.instance_id = self.extra_config["instance_id"]

        # Also read in _init_kvcm_client, but needed earlier: whether the write
        # paths run TP collectives is decided on the first write call, which is
        # before the client exists (an MLA write never uses the group).
        self.is_mla_model = self.storage_config.is_mla_model

        self._manager_client = KvCacheManagerClient.from_connector_config(
            self.extra_config
        )

        # PoolName -> the pool object; sglang's pool classes expose more than
        # the HostKVCache base (e.g. get_page_buffer_meta), hence Any.
        self.registered_pools: Dict[Any, Any] = {}

        # Pool names that arrived after the location specs were frozen (or
        # after close()): the Manager has no spec update API, and such a pool
        # may even reuse the name of an already-registered pool -- whose spec
        # the new host indices do not belong to.  The data path answers miss
        # for these names instead of pairing them with a registered spec.
        self._late_pools = set()

        # _init_lock guards the one-time client initialization; the flags are
        # instance state (class-level defaults keep hand-built instances
        # -- tests -- from raising on the first check).
        self._init_lock = threading.Lock()
        self._client_ready = False
        self._closed = False
        self._tp_init_agreed = False

        # SDK handles, created by _ensure_client() and dropped by close().
        self.transfer_client: Any = None
        self.init_params: Any = None
        # The host pool shm segments handed to the client (each one keeps its
        # mapping and allocator alive through ``owner``); held so that they
        # outlive the client, which holds raw addresses into them.
        self._host_pool_spans: List[Any] = []

        self.prefetch_pgs = []
        self.backup_pgs = []
        self.prefetch_bandwidth = []
        self.backup_bandwidth = []

        # Declare v1 interface support so that sglang's cache_controller uses
        # batch_set_v1/batch_get_v1 (zero-copy path) instead of the legacy
        # batch_set/batch_get. This avoids requiring users to manually add
        # "interface_v1": 1 in --hicache-storage-backend-extra-config.
        self.extra_config.setdefault("interface_v1", 1)

    def _init_parallel_context(self) -> None:
        """Resolve the TP group and create the connector's own gloo group.

        COLLECTIVE: ``torch.distributed.new_group`` requires every rank of the
        default group to call it in the same order, so this stays in the v1
        hook -- the attach path, where all ranks meet in lockstep. Everything
        else is deferred to the first storage call (``_ensure_client``).
        """
        # parallelism
        self.tp_rank = self.storage_config.tp_rank
        self.tp_size = self.storage_config.tp_size
        # TODO: pp
        self.dp_size = 1
        self.pp_size = 1

        tp_group = (
            get_attn_tp_group().cpu_group
            if is_dp_attention_enabled()
            else get_tp_group().cpu_group
        )
        self.tp_world_size = torch.distributed.get_world_size(group=tp_group)
        if self.tp_world_size > 1:
            group_ranks = torch.distributed.get_process_group_ranks(tp_group)
            self.storage_tp_group = torch.distributed.new_group(
                group_ranks, backend="gloo"
            )

    def _raise_if_closed(self) -> None:
        """Refuse to serve once ``close()`` retired this backend."""
        if self._closed:
            raise RuntimeError(
                "connector was closed; storage stays disabled until the "
                "backend is re-attached"
            )

    def _ensure_client(self) -> None:
        """Register with the Manager and create the SDK client, once.

        Two entry points, one implementation: the data path (the lazy path of
        every sglang release without the ``finalize_registration`` hook) and
        that hook, where sglang says its registration burst is over.  Either
        way the burst (v1 anchor, then one v2 call per pool) has to be over
        before the location specs are frozen into ``registerInstance`` -- a
        repeat with a different configuration is rejected and there is no spec
        update API.  Pools registered later are degraded, see
        ``register_mem_host_pool_v2``.

        No collective happens here, so ranks may initialize at different times
        and the read paths call this directly.  The write paths must agree on
        the outcome first, see ``_ensure_client_for_write``.

        Reporting is split by cause.  An ``_UnsupportedBackendError`` (a pool
        shape or a configuration this connector cannot serve) is reported once
        here, before it is raised, so the ``batch_*`` callers stay quiet on every
        later attempt.  Everything else -- a closed backend, no KV pool
        registered yet, a Manager error, an exception out of the SDK --
        propagates unreported and those callers log it on *every* attempt.  Both
        kinds answer conservatively and re-attempt initialization on their next
        call.
        """
        # Checked before the ready flag: close() tears the client down, so a
        # late storage call must fail with a clear message instead of using it.
        self._raise_if_closed()
        if self._client_ready:
            return
        with self._init_lock:
            if self._client_ready:
                return
            self._raise_if_closed()

            kv_pool = self.registered_pools.get(PoolName.KV)
            if kv_pool is None:
                raise RuntimeError(
                    "no KV host pool registered yet; storage calls require "
                    "sglang's register_mem_pool_host/register_mem_host_pool_v2"
                )
            if not isinstance(kv_pool, HostKVCache):
                # Reported once here instead of once per storage call: sglang's
                # DeepSeek-V4 compressed pool anchors its KV entry on a
                # LogicalHostPool, which has no get_size_per_token() to build
                # specs from.
                self._report_unsupported_kv_pool(kv_pool)
                raise _UnsupportedBackendError(
                    f"the KV anchor of this instance is a "
                    f"{type(kv_pool).__name__}, not a host pool; this pool "
                    f"shape is not supported by this connector",
                    reason="this instance's pool or spec shape is not supported",
                )
            self._init_kvcm_client()
            self._client_ready = True

    def _agree_collective_transfers(self, transfers: List[PoolTransfer]) -> List[str]:
        """The sequence every rank of the group runs its collectives in.

        ``batch_set_v2`` opens a write session and folds a commit flag per
        transfer, and both are collectives: a rank that enters them alone blocks
        its peers forever, and two ranks that enter them in a different order
        pair a session with the wrong transfer.  Neither the set nor the order
        can be read off the local list -- sglang filters replicated pools to
        rank 0 on MLA, a sidecar whose source pool is present on one rank only is
        resolvable on that rank alone, and the list is otherwise as ordered as
        the engine happened to build it -- so every rank reports its transfers as
        ``(identity, has_keys, eligible)`` and keeps the ones *every* rank
        reported as cooperative, sorted by that identity
        (``_collective_identity``).

        That triple is what makes the answer identical everywhere, and it is
        deliberately wider than the filter: only ``eligible`` -- does sglang hand
        this transfer to every rank at all -- has to hold on every rank, while
        ``has_keys`` is carried so the rule reads as what it is.  Which ranks have
        keys for a transfer never filters the sequence: such a rank is still a
        member of it and still joins its collectives (see ``batch_set_v2``), and
        dropping it here would leave its peers blocked.

        A pool that only some ranks reported drops out instead -- a rank handed
        no transfers at all reports nothing -- and dropping it is what keeps the
        group honest: no rank opens a session for it, each reports its own
        transfer as a miss, and nobody waits for a peer that is not in the call.
        One gather per call, against file I/O per block.
        """
        # The identity is a string: orderable (``PoolName`` members are not) and
        # picklable for the gather.
        local = sorted(
            (
                (
                    _collective_identity(transfer.name),
                    bool(transfer.keys),
                    self._transfer_is_written_by_every_rank(transfer),
                )
                for transfer in transfers
            ),
            key=lambda report: report[0],
        )
        cooperative = [identity for identity, _has_keys, eligible in local if eligible]
        if not self._write_path_uses_tp_collectives(transfers):
            # This write runs no TP collective at all -- a single rank, or the
            # rank-0 protocol an MLA sidecar write uses before sglang 0.5.17,
            # where only rank 0 is called.  Gathering here would hang the one
            # rank that calls it, so the local sequence *is* the schedule.
            return cooperative
        if self.storage_tp_group is None:
            return cooperative
        gathered: List[Any] = [None] * self.tp_world_size
        torch.distributed.all_gather_object(
            gathered, local, group=self.storage_tp_group
        )
        if any(report is None for report in gathered):
            # Only a group that did not answer leaves a slot empty, and a
            # schedule nobody agreed on is worse here than the local one.
            return cooperative
        # Per rank, not per report: a rank that lists one pool twice (the same
        # pool under a ``PoolName`` member and under its value text) still counts
        # as one rank, so the entry stays in the sequence instead of failing an
        # "every rank reported it" test it did in fact pass.
        reported = [
            {identity for identity, _has_keys, eligible in report if eligible}
            for report in gathered
        ]
        ranks_of_identity = Counter(
            identity for identities in reported for identity in identities
        )
        return sorted(
            identity
            for identity, ranks in ranks_of_identity.items()
            if ranks == len(reported)
        )

    def _report_rank_local_shared_transfer(self, pool_name: Any) -> None:
        """Report once that a group-wide pool was handed to this rank alone."""
        if not self._mark_pool_reported(("rank-local-shared", pool_name)):
            return
        logger.warning(
            "pool %s is written by every rank of the TP group, but only this "
            "rank was handed it for this call, so its peers would wait on the "
            "write session forever; it is not written and its transfers are "
            "reported as misses. A pool that only some ranks back belongs in the "
            "connector's table with the matching sharing mode.",
            pool_name,
        )

    def _transfer_is_written_by_every_rank(self, transfer: PoolTransfer) -> bool:
        """Whether sglang hands this transfer to every rank of the TP group.

        Non-MLA: yes, ``should_backup()`` answers True on every rank, so every
        rank writes every pool with its own rank spec.  MLA: only the pools the
        table marks as rank-sharded (``shared_across_ranks=False``, i.e. the
        Mamba/KDA state of e.g. Kimi-K3) are written from every rank; the
        remaining pools are filtered out on non-zero ranks and reach
        ``batch_set_v2`` from rank 0 alone, so no peer waits for them.  Which
        of the two protocols the installed sglang implements for those
        rank-sharded sidecars is decided by
        ``_sidecars_are_written_by_every_rank``.

        A row derived from a pool the table does not know is deliberately not
        consulted here: this method is called before ``_ensure_client`` may have
        resolved the per-instance rows, and a rank-dependent answer would split
        the TP group.  Not finding the pool keeps the rank-0 protocol, which is
        what sglang's own ``should_backup`` does for every pool but the Mamba
        state.
        """
        if not self.is_mla_model:
            return True
        spec = _pool_spec(transfer.name)
        return (
            spec is not None
            and not spec.shared_across_ranks
            and _sidecars_are_written_by_every_rank()
        )

    def _backs_a_sharded_sidecar(self) -> bool:
        """Whether this instance backs a pool the table marks as rank-sharded.

        A property of the registered pool *set*, not of the transfers one call
        was handed: the engine builds the same host pools on every rank, so
        every rank answers the same, which is what a group rendezvous needs (see
        ``_write_path_uses_tp_collectives``).  Such a pool is the only thing that
        makes sglang hand a sidecar write to every rank, hence the only reason a
        v2 write can reach a collective at all; a hybrid stack whose sidecars are
        all replicated (an MLA indexer, say) is written by rank 0 alone.
        """
        return any(
            not spec.shared_across_ranks and spec.pool in self.registered_pools
            for spec in _POOL_SPECS
        )

    def _write_path_uses_tp_collectives(
        self, pool_transfers: Optional[List[PoolTransfer]] = None
    ) -> bool:
        """Whether this write call runs collectives over the TP group.

        The condition in front of every collective the call will run, and it has
        to come out the same on every rank: the ranks rendezvous in
        ``_ensure_client_for_write`` *before* the first write collective, so a
        rank that answers False while a peer answers True leaves that peer
        waiting for good.  Only instance-wide inputs are therefore read --
        parallel sizes, the model kind, whether this sglang build writes sidecars
        from every rank, and whether the instance backs a sharded sidecar pool --
        never the contents of ``pool_transfers``: the transfer list is per rank
        by nature (a rank with no pages to back up is handed an empty list for
        the same call its peers are handed transfers for), and judging from it is
        exactly what splits the group.  ``pool_transfers`` is only read as
        "None = the v1 KV hook, a list = a v2 sidecar write".

        * single rank -> False, there is nobody to wait for;
        * non-MLA -> True, the v1 KV write and every sidecar are written by every
          rank;
        * MLA -> the v1 KV write is written by rank 0 alone (its KV pool is
          replicated), so False; a v2 sidecar write runs collectives exactly when
          this sglang hands sharded sidecars to every rank
          (``_sidecars_are_written_by_every_rank``) *and* the instance backs such
          a pool (``_backs_a_sharded_sidecar``).  Before sglang 0.5.17 only rank
          0 was called for sidecars, and an MLA stack without a sharded pool is
          called on rank 0 alone as well -- in both a rendezvous would wait for
          ranks that never arrive.
        """
        if getattr(self, "tp_world_size", 1) <= 1:
            return False
        if not self.is_mla_model:
            return True
        if pool_transfers is None:
            return False
        return _sidecars_are_written_by_every_rank() and self._backs_a_sharded_sidecar()

    def _ensure_client_for_write(
        self, pool_transfers: Optional[List[PoolTransfer]] = None
    ) -> None:
        """``_ensure_client`` plus the TP-group agreement the write paths need.

        The write paths are the only callers of the connector's TP collectives
        (``_batch_set``, ``batch_set_v2``), and a collective needs the whole
        group: a rank whose client cannot be created would otherwise answer
        conservatively and leave its peers waiting in
        ``broadcast_object_list`` / ``all_reduce`` for a rank that is already
        gone.  So the local outcome is published instead of being decided
        locally:

        * no collective on this path (single rank, or a write that runs rank-0
          only) -> plain local initialization, exactly as before;
        * a previous round already saw every rank ready -> return, so the
          steady state pays no extra collective;
        * otherwise one ``all_reduce(MIN)``, which every rank enters *before*
          the first write collective;
        * any rank failed -> *every* rank refuses the write, i.e. the callers
          turn it into their conservative result and no rank enters a write
          collective that another rank will never reach;
        * the group is missing -> refuse the write rather than run a
          collective nobody coordinates (``group=None`` means torch's
          *default* group, not "no group").

        ``pool_transfers`` is the sidecar write's transfer list: it decides,
        together with the model kind, whether this call runs collectives (see
        ``_write_path_uses_tp_collectives``), and the agreement therefore
        covers exactly the calls that will reach one.

        ``close()`` is checked first, so a retired backend fails here exactly
        like it does on the read paths, agreement or not.

        The failing rank does not return early: it records the error, joins
        the round and only then fails with it.  Readiness only ever turns from
        False to True, so a round that saw the whole group ready is final for
        this backend instance; until such a round happens every write call
        repeats it, which is what keeps a rank that failed once from slipping
        into a later write whose peers would already be waiting for it.
        Nothing is sticky, so the next call retries initialization.
        """
        self._raise_if_closed()
        if not self._write_path_uses_tp_collectives(pool_transfers):
            self._ensure_client()
            return
        if self._tp_init_agreed:
            return

        local_error: Optional[BaseException] = None
        try:
            self._ensure_client()
        except Exception as e:
            # Deliberately no early return: the ranks that did initialize are
            # already on their way to the write collectives and this rank has
            # to be in the round that tells them to stop.
            local_error = e

        # The agreement runs on the same group as the collectives it guards;
        # every rank that would reach them reaches this first.  A missing
        # group is a failure: peers may be about to wait here, and running the
        # write collectives on torch's default group instead is never wanted.
        group = getattr(self, "storage_tp_group", None)
        if group is None:
            raise RuntimeError(
                "the connector has no TP process group to agree on storage "
                "initialization with; refusing to write, because the write "
                "collectives would wait for ranks that cannot answer"
            )
        ready = torch.tensor([0 if local_error is not None else 1], dtype=torch.int)
        torch.distributed.all_reduce(
            ready, op=torch.distributed.ReduceOp.MIN, group=group
        )
        if not bool(ready.item()):
            if local_error is not None:
                raise local_error
            raise RuntimeError(
                "storage initialization failed on at least one rank of the TP "
                "group; this rank refuses the write so that no rank waits in a "
                "collective another rank will never reach"
            )
        self._tp_init_agreed = True

    def _init_kvcm_client(self) -> None:
        # model (is_mla_model is set in __init__ already: the write paths need
        # it before the client exists)
        self.model_name = self.storage_config.model_name
        self.kv_factor = 1 if self.is_mla_model else 2
        # The KV pool comes from the v1 hook (plain host pool) or from the v2
        # hook when v1 only carried a HostPoolGroup; _ensure_client rejects an
        # empty pool set before calling this.
        self.kv_dtype = self.registered_pools[PoolName.KV].dtype

        # manager
        self.block_size = self.mem_pool_host.page_size

        # The pool table resolved against what this instance registered, in
        # table order.  Everything below -- specs, groups, iov size, SDK spec
        # map -- is derived from it, and so is the data path's per-pool lookup.
        self.pool_specs = self._resolve_pool_specs()
        anchor = self.pool_specs[PoolName.KV]
        # The anchor's spec, under the names the data path uses.
        self.location_spec_name = anchor.name
        self.location_spec_size = anchor.size

        self.location_spec_infos = [
            {"name": name, "size": pool.size}
            for pool in self.pool_specs.values()
            for name in pool.names
        ]

        # LocationSpecGroup: KV specs are always prepared, but only sent when
        # extra pools exist (backward compat with older managers).
        self.location_spec_groups = [
            {"name": pool.spec.group, "spec_names": list(pool.names)}
            for pool in self.pool_specs.values()
            if not pool.spec.anchor
        ]
        if self.location_spec_groups:
            self.location_spec_groups.insert(
                0,
                {"name": anchor.spec.group, "spec_names": list(anchor.names)},
            )

        self.deployment = {
            "model_name": self.model_name,
            "tp_size": self.tp_size,
            "dp_size": self.dp_size,
            "pp_size": self.pp_size,
            "use_mla": self.is_mla_model,
            "dtype": str(self.kv_dtype)[6:],  # remove "torch."
        }

        register_request = {
            "trace_id": self._get_trace_id(),
            "instance_group": self.instance_group,
            "instance_id": self.instance_id,
            "model_deployment": self.deployment,
            "block_size": self.block_size,
            "location_spec_infos": self.location_spec_infos,
        }
        if self.location_spec_groups:
            register_request["location_spec_groups"] = self.location_spec_groups
        # The layout this run wants to freeze, in one comparable string: the
        # Manager keeps the first registration of an instance_id forever (it only
        # accepts a repeat when every field matches, and there is no spec update
        # API), so two different layouts under one id are a deployment error that
        # neither side can repair at runtime.  The digest is for the logs only.
        self.spec_fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "location_spec_infos": self.location_spec_infos,
                    "location_spec_groups": self.location_spec_groups,
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:16]
        logger.debug(
            "instance_id %s freezes %d location specs in %d groups, fingerprint %s",
            self.instance_id,
            len(self.location_spec_infos),
            len(self.location_spec_groups),
            self.spec_fingerprint,
        )
        # TODO: check conflict and update
        try:
            register_response = self._manager_client.register_instance(register_request)
        except Exception as e:
            self._report_register_failure(e)
            raise
        logger.debug(f"register_instance {register_response=}")

        self.storage_configs = register_response["storage_configs"]

        self.write_timeout_seconds = self.extra_config.get("write_timeout_seconds", 30)

        # sdk
        self.sdk_thread_num = self.extra_config.get("sdk_thread_num", 4)
        self.sdk_queue_size = self.extra_config.get("sdk_queue_size", 1000)
        self.sdk_get_timeout_ms = self.extra_config.get("sdk_get_timeout_ms", 15000)
        self.sdk_put_timeout_ms = self.extra_config.get("sdk_put_timeout_ms", 15000)

        self.read_iov_block_size = self.extra_config.get("read_iov_block_size", 0)
        self.write_iov_block_size = self.extra_config.get("write_iov_block_size", 0)

        # The HF3FS backend allocates this area twice at Init (one read and one
        # write iov, shared memory plus host-pinned through
        # Hf3fsSdk::InitIovHandle) and every client of the instance pays for it,
        # so it must not grow because a pool this connector did not back before is
        # backed now: a client that cannot be created takes *all* of the
        # instance's caching with it, while an area too small for one pool only
        # costs that pool's transfers (each one fails its own allocation and is
        # reported as a miss).  The default is therefore what it has always been
        # -- _DEFAULT_IOV_BLOCK_COUNT blocks of the largest *table* pool -- and a
        # pool the area cannot hold a full budget of is reported by capacity, not
        # by "is bigger than the pools it is sized from".
        #
        # TODO: per-pool IOV areas would remove the trade-off; that needs changes
        # to the HF3FS SDK.
        known_block = max(
            pool.size for pool in self.pool_specs.values() if not pool.spec.derived
        )
        # Must be an integer; a malformed value fails here, at startup, instead of
        # being guessed at.
        configured_iov_size = self.extra_config.get("iov_size_bytes")
        self.iov_size = int(
            known_block * _DEFAULT_IOV_BLOCK_COUNT
            if configured_iov_size is None
            else configured_iov_size
        )
        # ``self.iov_size`` reaches the data path only through the HF3FS SDK
        # configs (``parse_hf3fs_configs``); a deployment using any other storage
        # never allocates that area, so warning about its capacity would be a
        # false alarm.
        active_hf3fs = bool(self.parse_hf3fs_configs(self.storage_configs))
        if active_hf3fs:
            for pool in self.pool_specs.values():
                self._report_iov_capacity(pool)

        sdk_backend_configs = list(self.extra_config.get("sdk_backend_configs", []))

        hf3fs_configs = self.parse_hf3fs_configs(self.storage_configs)
        sdk_backend_configs.extend(hf3fs_configs)
        logger.debug(sdk_backend_configs)

        transfer_client_json = {
            "instance_group": self.instance_group,
            "instance_id": self.instance_id,
            "block_size": self.block_size,
            "sdk_config": {
                "thread_num": self.sdk_thread_num,
                "queue_size": self.sdk_queue_size,
                "sdk_backend_configs": sdk_backend_configs,
                "timeout_config": {
                    "get_timeout_ms": self.sdk_get_timeout_ms,
                    "put_timeout_ms": self.sdk_put_timeout_ms,
                },
            },
            # This instance's own spec per pool: the KV one plus one per sidecar.
            "location_spec_infos": {
                pool.name: pool.size for pool in self.pool_specs.values()
            },
        }
        self.transfer_client_config = json.dumps(transfer_client_json)

        # InitParams carries metadata consumed by the C++ SdkWrapper::Init:
        #   - self_location_spec_name: validated against location_spec_infos;
        #     also used to construct a unique Mooncake hostname when the
        #     Mooncake backend is present (format: {host}_{spec_name}_{rand}).
        #   - regist_spans: every host pool shm segment, which the backend
        #     registers for RDMA zero-copy in its own Init.
        self.init_params = kvcm_py_client.InitParams()
        self.init_params.role_type = kvcm_py_client.RoleType.WORKER
        self.init_params.self_location_spec_name = self.location_spec_name
        self.init_params.storage_configs = f"{self.storage_configs}"
        self._attach_host_pool_spans()

        self.transfer_client = kvcm_py_client.TransferClient.Create(
            self.transfer_client_config, self.init_params
        )
        if self.transfer_client is None:
            self._report_unusable_sdk_client()
            # This attempt cannot succeed -- the client is built from the
            # configuration frozen above -- but a later one-shot initialization
            # may, if the runtime failed for an environmental reason: the data
            # path re-attempts it, the finalize hook does not (the engine
            # detaches on the way out).  An ``assert`` would also vanish under
            # ``python -O`` and leave a backend without a client.
            raise _UnsupportedBackendError(
                "kvcm_py_client.TransferClient.Create returned no client for "
                "this instance's storage configuration and location specs",
                reason="the KVCM transfer runtime could not be created",
            )

    def _attach_host_pool_spans(self) -> None:
        """Hand every host pool shm segment to the client before it is created.

        The spans travel in ``InitParams.regist_spans``: the client passes them
        to the backend, which registers each one for RDMA zero-copy inside its
        own Init, so this connector never calls into the backend itself.

        Every segment is checked first, because the backend refuses the whole
        registration on the first one it cannot take and ``Create`` then returns
        no client at all -- that would cost every pool its storage, not just the
        broken segment.  Nothing here raises: a segment that fails a check is
        counted and left out, and the rest is handed over as usual.

        Nothing is handed over when the instance has no Tair mempool backend:
        the spans are dead weight there, and the cost is not free -- the client
        dups every fd and holds it until it is destroyed.
        """
        if self.extra_config.get("allocator") != "shm":
            logger.warning(
                'host pools are not shm-backed (no "allocator":"shm" in '
                "--hicache-storage-backend-extra-config), so no host segment is "
                "handed to the client and every transfer pays a bounce copy"
            )
            return
        if not hasattr(kvcm_py_client, "RegistSpan") or not hasattr(
            self.init_params, "regist_spans"
        ):
            logger.warning(
                "kvcm_py_client takes no host memory spans (open-source build or "
                "an older client), so no host segment is handed over and every "
                "transfer pays a bounce copy"
            )
            return

        spans: List[Any] = []
        pools = segments = skipped_pools = skipped_segments = 0
        skipped_no_backend = 0
        # Only a Tair mempool backend consumes these spans; every other backend
        # leaves them untouched, while the client still dups each fd and keeps
        # it for the instance's lifetime (``SdkWrapper::Init`` prepares the
        # registrations before it looks at the storage types).  So a deployment
        # without one is handed nothing at all.
        consumer = _has_tair_mempool_storage(self.storage_configs)
        if consumer is None:
            logger.warning(
                "PACE zero-copy: the storage configs cannot be read, so no host "
                "segment is handed over"
            )
        for pool_name, pool in self.registered_pools.items():
            allocator = getattr(pool, "allocator", None)
            # The allocator appends one segment per allocation; the pool's ``fd``
            # and the allocator's ``mm`` are only the first of them.
            fds = getattr(allocator, "fds", None)
            mms = getattr(allocator, "mms", None)
            if not fds or not mms:
                skipped_pools += 1
                logger.warning(
                    "PACE zero-copy: pool %s has no shm segments (allocator %s); "
                    "it is skipped and the other pools are unaffected",
                    pool_name,
                    type(allocator).__name__,
                )
                continue
            pools += 1
            for fd, mm in zip(fds, mms):
                segments += 1
                if consumer is not True:
                    skipped_no_backend += 1
                    continue
                span, defect = self._host_segment_span(fd, mm, allocator)
                if span is None:
                    skipped_segments += 1
                    logger.warning(
                        "PACE zero-copy: pool %s fd=%s is not handed over (%s); "
                        "its transfers fall back to a bounce copy",
                        pool_name,
                        fd,
                        defect,
                    )
                    continue
                spans.append(span)

        self._host_pool_spans = spans
        self.init_params.regist_spans = spans
        logger.info(
            "PACE zero-copy: pools=%d segments=%d handed=%d skipped_pools=%d "
            "skipped_segments=%d skipped_no_backend=%d",
            pools,
            segments,
            len(spans),
            skipped_pools,
            skipped_segments,
            skipped_no_backend,
        )

    @staticmethod
    def _host_segment_span(fd: Any, mm: Any, allocator: Any) -> tuple:
        """The ``RegistSpan`` for one shm segment, or ``(None, why not)``.

        The checks are the ones the backend's Init performs before it dups the
        fd: an open fd whose size still matches the mapping, and an fd this
        process can still dup.  ``owner`` keeps the mapping and the allocator
        alive -- the backend reaches them by raw address while the client lives.
        """
        try:
            # ``len(mm)`` is the page-aligned mapping, which is what the memfd
            # was sized to -- not the tensor's nbytes.
            size = len(mm)
            base = ctypes.addressof(ctypes.c_char.from_buffer(mm))
            if not isinstance(fd, int) or fd < 0:
                return None, f"fd {fd!r} is not an open descriptor"
            stat = os.fstat(fd)
            if stat.st_size != size:
                return None, f"the fd holds {stat.st_size} bytes, the mapping {size}"
            os.close(os.dup(fd))
            span = kvcm_py_client.RegistSpan()
            span.fd = fd
            span.size = size
            span.base = base
            span.owner = (mm, allocator)
        except Exception as e:
            return None, str(e)
        return span, None

    def _resolve_pool_specs(self) -> Mapping[Any, _RegisteredPool]:
        """Resolve the spec rows against the pools this instance registered.

        The table rows come first, in table order (anchor first), then one
        derived row per registered host pool the table does not know, in the
        order sglang registered them.  That order is this connector's own and it
        is stable (same engine build, same pool set, same order); the Manager
        neither keeps it for the spec groups -- it sorts them by name
        (``config/instance_info.cc::SortLocationSpecGroups``) -- nor derives any
        id from it.  What *is* order-sensitive is ``location_spec_infos``, which
        the Manager compares element by element against the stored copy
        (``config/instance_info.h::MismatchFields``), hence the stable order here.
        A pool that never registered has no entry.

        Two pools may never share a location spec or a spec group: the Manager
        would be handed the same spec name twice and the data path would write
        one pool's bytes under the other pool's name, which no pool-local test
        can see.  Table rows are held apart by the table's own invariants; a
        derived row is refused (that pool falls back to "not backed", i.e. to
        misses) instead, because its name comes from the engine.

        A backed sidecar also decides the anchor's spec name (``tp_{rank}`` vs
        ``tp_{rank}_full``), so the sidecar set is decided once here and every
        derived answer follows from it.  That predicate is the same fact as
        ``has_sidecars``, which cannot be used here because this method is what
        fills the state it reads.
        """
        static = [spec for spec in _POOL_SPECS if spec.pool in self.registered_pools]
        known = {spec.pool for spec in static}
        derived = [
            _derived_pool_spec(name)
            for name, pool in self.registered_pools.items()
            if name not in known and isinstance(pool, HostKVCache)
        ]

        def names_of(spec: _PoolSpec) -> set:
            """Every spec name the row can emit, whichever anchor naming wins."""
            return {
                spec.spec_name(rank, hybrid)
                for hybrid in (False, True)
                for rank in range(self.tp_size)
            }

        # The anchor claims both of its names: which one it takes depends on the
        # sidecars that survive below, and a derived pool must never land on
        # either (a pool called "full" would otherwise take the hybrid name).
        anchor = next((spec for spec in static if spec.anchor), None)
        claimed_names = names_of(anchor) if anchor is not None else set()
        claimed_groups = {anchor.group} if anchor is not None else set()
        sidecars: List[tuple[_PoolSpec, int]] = []
        rows = [(spec, False) for spec in static if not spec.anchor]
        rows += [(spec, True) for spec in derived]
        for spec, is_derived in rows:
            if names_of(spec) & claimed_names or spec.group in claimed_groups:
                if not is_derived:
                    # Unreachable: TestPoolSpecTable holds the table apart.
                    raise RuntimeError(f"two rows of _POOL_SPECS claim {spec.pool}")
                self._report_unknown_pool(spec.pool)
                continue
            size = spec.spec_size(self.registered_pools[spec.pool])
            if size <= 0:
                # A spec of zero bytes is not a layout the Manager or the SDK can
                # use (the SDK refuses a non-positive spec size outright), so the
                # pool is not backed at all rather than breaking the instance.
                if not is_derived:
                    # Unreachable for a host pool; the anchor contract is its own.
                    raise RuntimeError(f"{spec.pool} reports {size} bytes per block")
                self._report_unusable_pool_spec(spec.pool, size)
                continue
            claimed_names |= names_of(spec)
            claimed_groups.add(spec.group)
            sidecars.append((spec, size))

        hybrid_stack = bool(sidecars)

        def row_of(spec: _PoolSpec, size: int) -> _RegisteredPool:
            return _RegisteredPool(
                spec=spec,
                size=size,
                names=tuple(
                    spec.spec_name(rank, hybrid_stack) for rank in range(self.tp_size)
                ),
                own_rank=(
                    0
                    if spec.shared_across_ranks and self.is_mla_model
                    else self.tp_rank
                ),
            )

        resolved = {}
        if anchor is not None:
            resolved[anchor.pool] = row_of(
                anchor, anchor.spec_size(self.registered_pools[anchor.pool])
            )
        for spec, size in sidecars:
            resolved[spec.pool] = row_of(spec, size)
            if spec.pool not in known:
                logger.debug(
                    "pool %s is not in the spec table; derived spec %s (%d bytes "
                    "per block, group %s) from the pool itself",
                    spec.pool,
                    resolved[spec.pool].name,
                    size,
                    spec.group,
                )
        return MappingProxyType(resolved)

    # TODO: delete once the stacked pool PRs land -- nothing in this tree reads
    # them any more; use ``pool_specs`` or ``has_sidecars`` instead.  The same
    # applies to ``_tp_rank_to_spec_name`` / ``_tp_rank_to_linear_spec_name`` /
    # ``_tp_rank_to_indexer_spec_name`` below, whose only remaining caller is a
    # test fixture.
    @property
    def has_mamba(self) -> bool:
        """Whether this instance registered the Mamba/KDA pool.

        Read-only view of ``pool_specs``, which is resolved once per instance.
        """
        return _MAMBA_SPEC.pool in self.pool_specs

    @property
    def has_indexer(self) -> bool:
        """Whether this instance registered the NSA/DSA indexer pool.

        Read-only view of ``pool_specs``; always False on a release whose
        ``PoolName`` has no indexer member.
        """
        return _INDEXER_SPEC.pool is not None and _INDEXER_SPEC.pool in self.pool_specs

    @property
    def has_sidecars(self) -> bool:
        """Whether this instance registered a pool besides the KV anchor."""
        return any(not pool.spec.anchor for pool in self.pool_specs.values())

    def parse_hf3fs_configs(self, storage_configs: str) -> List[Dict[str, Any]]:
        hf3fs_configs = []
        storage_configs_json = json.loads(storage_configs)
        for storage_config in storage_configs_json:
            storage_type = storage_config["type"]
            if storage_type not in ("hf3fs", "vcns_hf3fs") or not storage_config.get(
                "is_available", True
            ):
                continue
            hf3fs_config = {
                "type": storage_type,
                "mountpoint": storage_config["storage_spec"]["mountpoint"],
                "root_dir": storage_config["storage_spec"]["root_dir"],
                "read_iov_block_size": self.read_iov_block_size,
                "read_iov_size": self.iov_size,
                "write_iov_block_size": self.write_iov_block_size,
                "write_iov_size": self.iov_size,
            }
            hf3fs_configs.append(hf3fs_config)
        return hf3fs_configs

    def register_mem_pool_host(self, mem_pool_host: HostKVCache) -> None:
        """v1 hook: the KV (anchor) host pool, or a <= 0.5.18 HostPoolGroup.

        The object is kept for the v1 data path (``get_page_buffer_meta`` and
        ``page_size``); pools themselves are registered from the v2 hook only,
        and the Manager is told about them on the first storage call.
        """
        # The pool objects expose more than the HostKVCache base
        # (get_page_buffer_meta & co).
        self.mem_pool_host: Any = mem_pool_host
        self._init_parallel_context()

        # A host pool handed over here is the KV pool: flat models on every
        # version, and every model on sglang >= 0.5.19 (sglang passes the
        # HostPoolGroup's anchor entry). Up to v0.5.18 a hybrid stack passes
        # the group itself, which is not a HostKVCache -- the KV entry then
        # arrives through register_mem_host_pool_v2 right after.
        if isinstance(mem_pool_host, HostKVCache):
            self.registered_pools[PoolName.KV] = mem_pool_host
        logger.info(
            "register_mem_pool_host: type=%s, kv_pool_registered=%s",
            type(mem_pool_host).__name__,
            PoolName.KV in self.registered_pools,
        )

    def register_mem_host_pool_v2(
        self, host_pool: HostKVCache, host_pool_name: str
    ) -> None:
        """v2 hook: one host pool per call, sidecars included.

        sglang calls this for every entry of its host pool group, the KV
        anchor included, right after the v1 hook. The pool set is final when
        the location specs are frozen -- at ``finalize_registration`` where
        sglang offers that hook, otherwise at the first data-path call. A pool
        showing up after that cannot be added to the Manager registration and
        is degraded (see the class docstring).

        The initialization lock is taken: a pool arriving while the specs are
        being frozen would otherwise be recorded but missing from the
        registration payload, i.e. silently unusable.
        """
        with self._init_lock:
            if self._client_ready or self._closed:
                # Too late for this instance.  Do not record it: if it reuses
                # a managed pool's name, the data path must not pick it up
                # (its memory is not the memory the registered spec refers
                # to).  ``_late_pools`` keeps the name unusable explicitly;
                # rebind instead of add() so the class-level default stays
                # immutable.
                self._late_pools = self._late_pools | {host_pool_name}
                self._report_late_pool(host_pool_name)
                return
            self.registered_pools[host_pool_name] = host_pool
            logger.info(
                "register_mem_host_pool_v2: pool=%s, type=%s",
                host_pool_name,
                type(host_pool).__name__,
            )

    def _report_late_pool(self, pool_name: Any) -> None:
        """Report once that a pool arrived after the specs were frozen."""
        if not self._mark_pool_reported(pool_name):
            return
        logger.error(
            "register_mem_host_pool_v2: pool %s was registered after the "
            "connector was initialized with the Manager; there is no "
            "specification update API, so this pool stays unsupported and its "
            "transfers are reported as misses.",
            pool_name,
        )

    def _mark_pool_reported(self, key: Any) -> bool:
        """Record that a report key has been reported, False if it already was.

        The key is usually a pool name, but a report that is not about a pool
        (a failed registration, for instance) passes a tuple of its own, or one
        of the module-level sentinels for an instance-wide failure; a tuple
        carries the instance id because two instances in one process are two
        deployments.
        """
        if key in self._reported_pools:
            return False
        self._reported_pools.add(key)
        return True

    def _report_register_failure(self, error: BaseException) -> None:
        """Report a failed registration once per instance and cause.

        ``_ensure_client`` retries on every storage call, so the same failure
        would otherwise be logged once per call.  Three causes are told apart,
        because only one of them can be fixed by changing the deployment's
        identity:

        * ``rejected`` -- the Manager answered and refused the registration as
          such (a non-OK business status).  If the id it refused is already
          registered with a different layout, the fix is a new ``instance_id``;
          that is the case the log leads with, and the Manager's own message
          (which names the mismatched fields) is attached;
        * ``refused`` -- the Manager refused the HTTP request itself (4xx);
        * ``unjudged`` -- nothing was decided: a 5xx, a broken envelope, a
          transport failure, or a leader election the client already retried.

        Only the first may name ``instance_id``; the other two are reported as
        what they are, and stay warnings.

        The classification cannot use ``isinstance(error, AssertionError)``: the
        Manager client's ``KvCacheManagerHTTPError`` and
        ``KvCacheManagerProtocolError`` inherit ``AssertionError`` as well (see
        ``common/manager_client.py``), so a 503 or a broken envelope would be
        read as "your layout was rejected" -- the one diagnosis whose fix costs a
        full cache restart.

        Deduplicated through ``_reported_pools`` like every other report, under a
        key that carries the instance id: two instances in one process are two
        deployments (sglang can detach and re-attach the backend with another
        ``instance_id`` at runtime), while a re-created backend of the *same*
        instance stays quiet because the diagnosis has not changed.
        """
        cause = _register_failure_cause(error)
        key = ("registerInstance", self.instance_id, cause)
        if not self._mark_pool_reported(key):
            logger.debug("registerInstance failed again (%s): %s", cause, error)
            return
        if cause == "unjudged":
            logger.warning(
                "registerInstance could not be judged by the Manager (transport, "
                "protocol, server or leader-election failure), so nothing is "
                "concluded about the layout of instance_id %s; a later attempt "
                "may still succeed. Error: %s",
                self.instance_id,
                error,
            )
            return
        if cause == "refused":
            logger.error(
                "the Manager refused the registration request of instance_id %s "
                "(instance_group %s) as a whole, so nothing is concluded about "
                "the pool set; check the request fields and whether the connector "
                "and the Manager agree on the API. Layout fingerprint %s. "
                "Error: %s",
                self.instance_id,
                self.instance_group,
                self.spec_fingerprint,
                error,
            )
            return
        logger.error(
            "the Manager rejected the registration of instance_id %s "
            "(instance_group %s) with the layout fingerprint %s (specs %s, "
            "groups %s). If this id is already registered with a different "
            "layout, that is a changed pool set -- registerInstance only accepts "
            "a repeat when every field matches and there is no spec update API, so "
            "it needs a new instance_id (and instance_group, if the pools moved "
            "between storage groups). Any other rejection is about the request "
            "itself, and the Manager's message says which field it refused: %s",
            self.instance_id,
            self.instance_group,
            self.spec_fingerprint,
            [spec["name"] for spec in self.location_spec_infos],
            [group["name"] for group in self.location_spec_groups],
            error,
        )

    def _report_unusable_pool_spec(self, pool_name: Any, size: int) -> None:
        """Report once that a pool's block size cannot become a location spec."""
        if not self._mark_pool_reported(pool_name):
            return
        logger.warning(
            "pool %s reports %d bytes per block; a location spec of zero bytes is "
            "not a layout the Manager or the SDK accepts, so this pool is not "
            "backed and its transfers are reported as misses.",
            pool_name,
            size,
        )

    def _report_iov_capacity(self, pool: _RegisteredPool) -> None:
        """Report a pool whose block does not fit the shared IOV area at all.

        The area is allocated per *block*, not per call: ``Hf3fsSdk::Get``/``Put``
        walk the operation one URI at a time, and each call allocates that block's
        bytes from ``Hf3fsMempool`` and frees them again before the next one.  So
        one block must fit; how many more fit only decides how many transfers can
        be in flight at once, and running out there fails that block's
        ``InitIovIor`` and reports it as a miss rather than corrupting anything.
        A pool whose single block does not fit is therefore never backed, and
        that is what this reports.
        """
        if pool.size <= 0:
            # Cannot happen for a backed row (see ``_resolve_pool_specs``); a
            # division here must never be able to fail the whole instance.
            return

        # Each direction is rounded up before use, with its own alignment:
        # ``Hf3fsSdk::InitIovHandle`` rounds the configured area up to a multiple
        # of that direction's ``iov_block_size`` and ``Hf3fsMempool::Alloc``
        # rounds the requested block up to the same value, so a block and an area
        # in one bucket fit even when the raw numbers say otherwise.  Read and
        # write areas are separate, so each has to fit on its own.
        def blocks(size: int, align: int) -> int:
            return (size + align - 1) // align if align > 1 else size

        fits = all(
            blocks(self.iov_size, align) >= blocks(pool.size, align)
            for align in (self.read_iov_block_size, self.write_iov_block_size)
        )
        if fits:
            return
        tokens = int(getattr(self.registered_pools[pool.spec.pool], "page_size", 1))
        logger.error(
            "pool %s moves %d bytes per block and the shared iov area "
            "(read_iov_size=write_iov_size=%d) does not hold even one of them, so "
            "every transfer of this pool fails to allocate and its data is never "
            "backed. Set iov_size_bytes to at least %d (one block; about %d tokens "
            "of this pool) -- larger values only allow more transfers in flight.",
            pool.spec.pool,
            pool.size,
            self.iov_size,
            pool.size,
            tokens,
        )

    def _report_unsupported_kv_pool(self, pool: Any) -> None:
        """Report once that the KV anchor is not a host pool.

        Seen with sglang's DeepSeek-V4 compressed KV pool, whose anchor is a
        ``LogicalHostPool``: it holds no tensors and no ``get_size_per_token``,
        so no location spec can be built for it.  The report is deduplicated
        through ``_reported_pools`` like every other unusable pool, so a
        re-created backend in the same process stays quiet.
        """
        if not self._mark_pool_reported(PoolName.KV):
            return
        logger.error(
            "the KV anchor of this instance is a %s, not a host pool: this "
            "pool shape (e.g. sglang's DeepSeek-V4 compressed KV pool) is not "
            "supported by this connector, so no location spec can be built for "
            "this instance.",
            type(pool).__name__,
        )

    def _report_unusable_sdk_client(self) -> None:
        """Report that this process has no SDK client for the instance, once.

        ``TransferClient.Create`` takes the frozen storage configuration and
        returns no client, i.e. nothing can be moved with the configuration this
        instance just registered.  The data-path entries answer
        ``_UnsupportedBackendError`` with a conservative result and no log of
        their own, so this is the one place an operator learns *why* nothing is
        cached -- the C++ side logs an error code without the instance or the
        specs.  Deduplicated like every other report, under a key that carries
        the instance id: a re-created backend of the same instance stays quiet,
        another instance in the same process is reported on its own.
        """
        # Per instance: two instances in one process are two deployments, and
        # the report names this instance's specs.
        if not self._mark_pool_reported((self.instance_id, _SDK_CLIENT_REPORT_KEY)):
            return
        logger.error(
            "instance %s has no KVCM transfer client: TransferClient.Create "
            "returned no client for this storage configuration, so none of its "
            "location specs can be served and its storage calls are answered "
            "with misses until a client can be created. specs: %s; "
            "storage_configs: %s",
            self.instance_id,
            ", ".join(
                f"{info['name']}={info['size']}B" for info in self.location_spec_infos
            ),
            self.storage_configs,
        )

    def finalize_registration(self) -> None:
        """Freeze the location specs once sglang reports every pool registered.

        Duck-typed hook, called by sglang that knows it after the last pool of
        the attach (draft pools included), i.e. exactly when the pool set
        becomes final, before any request is served, and propagated into
        scheduler construction.  Upstream sglang has no such callback, never
        calls this, and keeps freezing at the first data-path call -- unchanged.

        The work is ``_ensure_client``: ``_client_ready`` becomes the freeze
        point, so every pool registered before this call is in the spec set and
        one registered after it is late, i.e. reported once and answered with
        misses (there is no spec update API).  Idempotent, and safe to race the
        data path: both take the same lock and register once.  No collective
        runs here, so a rank that cannot register fails alone.

        Failures are raised unchanged; see ``_describe_registration_failure``
        for the transient/permanent split.  This is the one-shot attempt that
        does *not* get retried: the engine detaches the backend on the way out,
        so ``close()`` makes every later entry fail fast.  (A failed freeze
        leaves nothing frozen -- ``_client_ready`` stays False -- so without
        that detach a later data-path call would re-attempt initialization
        against the pool set of that moment; the Manager accepts a repeat of an
        identical registration and rejects a changed one.)
        """
        try:
            self._ensure_client()
        except Exception as e:
            logger.error(
                "finalize_registration: the one-shot Manager/SDK registration "
                "did not complete (%s); failing the attach instead of leaving "
                "every storage call to degrade to a miss in a background "
                "thread: %s",
                _describe_registration_failure(e),
                e,
            )
            raise

    def _batch_get(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        trace_id: str,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        # Prepare keys
        block_keys, len_prefix, len_new = self._prepare_block_keys(keys, extra_info)

        get_request = {
            "trace_id": trace_id,
            "block_keys": block_keys,
            "instance_id": self.instance_id,
            "query_type": "QT_PREFIX_MATCH",
            "block_mask": {"offset": len_prefix},
        }
        result = self._manager_client.get_cache_location(get_request)
        logger.debug(f"get_cache_location {result=}")
        locations = result["locations"]

        matched = len(locations)
        if matched == 0:
            return [False] * len_new

        # Data transfer preparation
        buffer_ptrs, buffer_sizes = self.mem_pool_host.get_page_buffer_meta(
            host_indices
        )
        buffer_matched = matched * self.kv_factor
        buffer_ptrs = buffer_ptrs[:buffer_matched]
        buffer_sizes = buffer_sizes[:buffer_matched]

        # Extract URIs and prepare buffers
        uris = self._extract_uris(locations)
        buffers = self._prepare_buffers(buffer_ptrs, buffer_sizes)
        assert len(uris) == len(buffers)
        # Perform data transfer
        start_time = time.perf_counter()
        result = self.transfer_client.LoadKvCaches(uris, buffers)
        end_time = time.perf_counter()
        self.prefetch_pgs.append(matched)
        self.prefetch_bandwidth.append(
            matched * self.location_spec_size / (1 << 30) / (end_time - start_time)
        )
        logger.debug(f"LoadKvCaches {result=}")

        flag = result == kvcm_py_client.ClientErrorCode.ER_OK
        if not flag:
            logger.error(f"{result}")
        return [flag] * len_new

    def batch_get_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        trace_id = self._get_trace_id()
        try:
            self._ensure_client()
            result = self._batch_get(
                keys=keys,
                host_indices=host_indices,
                trace_id=trace_id,
                extra_info=extra_info,
            )
            return result
        except _UnsupportedBackendError:
            # Already reported once by _ensure_client; keep answering
            # conservatively without logging on every call.
            return [False] * len(keys)
        except Exception as e:
            logger.error(f"batch_get_v1 failed: {trace_id=} {e=}")
            return [False] * len(keys)

    def batch_get_v2(
        self,
        transfers: List[PoolTransfer],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> dict[str, List[bool]]:
        results = {}
        trace_id = self._get_trace_id()
        try:
            self._ensure_client()
            for transfer in transfers:
                spec_name = self._get_extra_pool_spec_name(transfer.name)
                pool = self.registered_pools.get(transfer.name)
                keys = transfer.keys or []
                if not keys:
                    results[transfer.name] = []
                    continue
                if (
                    transfer.name in self._late_pools
                    or spec_name is None
                    or pool is None
                ):
                    self._report_unknown_pool(transfer.name)
                    results[transfer.name] = [False] * len(keys)
                    continue
                host_indices = self._transfer_host_indices(transfers, transfer)
                if host_indices is None:
                    logger.debug(
                        "pool %s: no host indices and no resolvable source pool "
                        "(%r); reporting a miss instead of moving data",
                        transfer.name,
                        transfer.indices_from_pool,
                    )
                    results[transfer.name] = [False] * len(keys)
                    continue

                block_keys, _, _ = self._prepare_block_keys(keys)
                get_request = {
                    "trace_id": trace_id,
                    "block_keys": block_keys,
                    "instance_id": self.instance_id,
                    "query_type": "QT_BATCH_GET",
                    "block_mask": {"offset": 0},
                }
                result = self._manager_client.get_cache_location(get_request)
                logger.debug(f"get_cache_location v2 {transfer.name} {result=}")
                locations = result["locations"]

                uris = []
                valid_indices = []
                for i, loc in enumerate(locations):
                    uri = self._extract_single_spec_uri(loc, spec_name)
                    if uri:
                        uris.append(uri)
                        valid_indices.append(i)

                if not uris:
                    results[transfer.name] = [False] * len(keys)
                    continue

                ptr_list, size_list, components = self._page_buffer_meta(
                    pool, transfer, host_indices
                )
                ptr_list = [
                    p
                    for i, p in enumerate(ptr_list)
                    if (i // components) in valid_indices
                ]
                size_list = [
                    s
                    for i, s in enumerate(size_list)
                    if (i // components) in valid_indices
                ]
                buffers = self._prepare_extra_pool_buffers(
                    ptr_list, size_list, components
                )
                assert len(uris) == len(buffers)

                start_time = time.perf_counter()
                load_result = self.transfer_client.LoadKvCaches(uris, buffers)
                end_time = time.perf_counter()
                flag = load_result == kvcm_py_client.ClientErrorCode.ER_OK
                if flag:
                    spec_size = self._get_extra_pool_spec_size(transfer.name)
                    self.prefetch_pgs.append(len(valid_indices))
                    self.prefetch_bandwidth.append(
                        len(valid_indices)
                        * spec_size
                        / (1 << 30)
                        / (end_time - start_time)
                    )
                per_key = [False] * len(keys)
                for idx in valid_indices:
                    per_key[idx] = flag
                results[transfer.name] = per_key

            return results
        except _UnsupportedBackendError:
            # Already reported once by _ensure_client; keep answering
            # conservatively without logging on every call.
            return {t.name: [False] * len(t.keys or []) for t in transfers}
        except Exception as e:
            logger.error(f"batch_get_v2 failed: {trace_id=} {e=}")
            return {t.name: [False] * len(t.keys or []) for t in transfers}

    def _batch_set(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        trace_id: str,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        # NOTE on cross-rank consistency:
        # start_write_cache and SaveKvCaches failures are synchronised
        # across ranks (via broadcast / all_reduce) so every rank
        # converges on the same result.
        #
        # finish_write_cache is called only on rank 0 *after* the
        # all_reduce has already completed. If it throws, rank 0
        # overrides flag to False while other ranks keep flag = True.
        # Adding a second all_reduce just for this error path would
        # penalise the hot path for a rare event. The practical
        # consequence is that rank 1+ callers may believe the write
        # succeeded, but the manager was never told to commit, so a
        # subsequent batch_get on those blocks will miss. This is an
        # accepted inconsistency -- the caller should tolerate cache
        # misses gracefully.

        # Prepare keys
        block_keys, len_prefix, len_new = self._prepare_block_keys(keys, extra_info)
        local_len_new = len_new  # Preserve local key count for return value
        local_hash = hash(
            (len_prefix, len_new, *block_keys)
        )  # Hash covers prefix/new boundary + all keys

        # Start write cache
        if self.tp_rank == 0:
            start_trace_id = f"start-{trace_id}"
            # When extra pools exist, use KV spec group to write KV specs only
            location_spec_group_names = (
                [self._get_kv_spec_group()] * len(block_keys)
                if self.has_sidecars
                else []
            )
            request = {
                "trace_id": start_trace_id,
                "instance_id": self.instance_id,
                "block_keys": block_keys,
                "location_spec_group_names": location_spec_group_names,
                "write_timeout_seconds": self.write_timeout_seconds,
            }
            logger.debug(f"start_write_cache {request=}")
            try:
                result = self._manager_client.start_write_cache(request)
            except Exception as e:
                logger.error(f"start_write_cache failed: {e}")
                result = None

            if self.tp_world_size > 1 and not self.is_mla_model:
                torch.distributed.broadcast_object_list(
                    [result, len_prefix, len_new, local_hash],
                    src=0,
                    group=self.storage_tp_group,
                )
        elif self.is_mla_model:
            logger.warning(
                f"_batch_set called on non-rank-0 (tp_rank={self.tp_rank}) "
                f"for MLA model; only rank 0 should write. Returning all False."
            )
            return [False] * len_new
        else:
            recv: List[Any] = [None, None, None, None]
            torch.distributed.broadcast_object_list(
                recv, src=0, group=self.storage_tp_group
            )
            result, len_prefix, len_new, rank0_hash = recv

        logger.debug(f"start_write_cache {result=}")

        # All ranks now share rank 0's len_prefix/len_new so that
        # _parse_block_mask produces the same save_indices everywhere,
        # preventing control-flow divergence (and NCCL hangs).
        # We also compare a hash of the full block_keys list: if a non-rank-0
        # rank's local block_keys differ from rank 0's, writing would corrupt
        # storage. The rank still participates in all_reduce with all-zero
        # flags so NCCL doesn't hang.
        skip_transfer = False
        if self.tp_rank != 0 and local_hash != rank0_hash:
            logger.warning(
                f"_batch_set: local block_keys hash ({local_hash}) != "
                f"rank 0 hash ({rank0_hash}), inputs diverged across TP ranks. "
                f"local_block_keys={block_keys}"
            )
            skip_transfer = True

        if result is None:
            return [False] * local_len_new

        locations = result["locations"]
        write_session_id = result["write_session_id"]
        block_mask = result["block_mask"]
        parsed = self._parse_block_mask(block_mask, len_prefix, len_new)

        finish_trace_id = f"finish-{trace_id}"

        # None means truly broken manager data — treat as write failure.
        if parsed is None:
            logger.warning(
                f"_batch_set: inconsistent block_mask from manager, "
                f"aborting write session {write_session_id}"
            )
            if self.tp_rank == 0:
                # Mark all locations as failed so manager cleans them up.
                try:
                    self._manager_client.finish_write_cache(
                        {
                            "trace_id": finish_trace_id,
                            "instance_id": self.instance_id,
                            "write_session_id": write_session_id,
                            "success_blocks": {
                                "bool_masks": {"values": [False] * len(locations)}
                            },
                        }
                    )
                except Exception as e:
                    logger.error(f"finish_write_cache failed: {e}")
            return [False] * local_len_new

        save_indices, prefix_write_count = parsed
        unmatched = len(save_indices)

        # Early return if all new blocks are already cached.
        if unmatched == 0:
            if self.tp_rank == 0:
                try:
                    self._manager_client.finish_write_cache(
                        {
                            "trace_id": finish_trace_id,
                            "instance_id": self.instance_id,
                            "write_session_id": write_session_id,
                            "success_blocks": {
                                "bool_masks": {"values": [False] * len(locations)}
                            },
                        }
                    )
                except Exception as e:
                    logger.error(f"finish_write_cache failed: {e}")
            return [False] * local_len_new if skip_transfer else [True] * local_len_new

        assert unmatched + prefix_write_count == len(locations)

        # Data transfer preparation and execution.
        # Skip prefix locations — sglang cannot write prefix blocks.
        # Best-effort: each rank writes only the blocks it has local data for.
        # A per-block flag vector is all_reduced (MIN) so only blocks written
        # by ALL ranks are considered successful.
        # Wrapped in try-except so that every rank always reaches the
        # all_reduce below, preventing cross-rank NCCL/gloo hangs.
        new_locations = locations[prefix_write_count:]
        per_block_flags = torch.zeros(unmatched, dtype=torch.int)
        if skip_transfer:
            # This rank's block_keys diverged from rank 0 — writing would
            # corrupt storage. Keep per_block_flags as all-zero and still
            # participate in all_reduce below.
            logger.warning(
                "_batch_set: skipping data transfer on this rank due to input divergence"
            )
        else:
            try:
                buffer_ptrs, buffer_sizes = self.mem_pool_host.get_page_buffer_meta(
                    host_indices
                )
                local_block_count = len(buffer_ptrs) // self.kv_factor

                # Determine which save_indices have local data available
                valid_save_mask = [(idx < local_block_count) for idx in save_indices]
                valid_save_set = set(
                    idx for idx, valid in zip(save_indices, valid_save_mask) if valid
                )
                num_valid = sum(valid_save_mask)

                if num_valid > 0:
                    buffer_ptrs = [
                        ptr
                        for i, ptr in enumerate(buffer_ptrs)
                        if (i // self.kv_factor) in valid_save_set
                    ]
                    buffer_sizes = [
                        sz
                        for i, sz in enumerate(buffer_sizes)
                        if (i // self.kv_factor) in valid_save_set
                    ]

                    # Extract URIs only for blocks with local data
                    valid_locations = [
                        loc
                        for loc, valid in zip(new_locations, valid_save_mask)
                        if valid
                    ]
                    uris = self._extract_uris(valid_locations)
                    buffers = self._prepare_buffers(buffer_ptrs, buffer_sizes)
                    assert len(uris) == len(buffers)

                    # Perform data transfer
                    start_time = time.perf_counter()
                    result = self.transfer_client.SaveKvCaches(uris, buffers)
                    end_time = time.perf_counter()
                    self.backup_pgs.append(num_valid)
                    self.backup_bandwidth.append(
                        num_valid
                        * self.location_spec_size
                        / (1 << 30)
                        / (end_time - start_time)
                    )
                    logger.debug(f"SaveKvCaches {result=}")

                    transfer_ok = result[0] == kvcm_py_client.ClientErrorCode.ER_OK
                    if not transfer_ok:
                        logger.error(f"SaveKvCaches error: {result}")
                else:
                    transfer_ok = True  # nothing to write on this rank

                # Mark blocks this rank successfully wrote
                for j, valid in enumerate(valid_save_mask):
                    if valid and transfer_ok:
                        per_block_flags[j] = 1

            except Exception as e:
                logger.error(f"Data transfer (SaveKvCaches) failed: {e}")
                # per_block_flags remains all zeros

        # Per-block all_reduce: only blocks ALL ranks wrote are marked success
        if self.tp_world_size > 1 and not self.is_mla_model:
            torch.distributed.all_reduce(
                per_block_flags,
                op=torch.distributed.ReduceOp.MIN,
                group=self.storage_tp_group,
            )

        new_block_success = [bool(per_block_flags[j]) for j in range(unmatched)]
        finish_mask = [False] * prefix_write_count + new_block_success
        if self.tp_rank == 0:
            try:
                self._manager_client.finish_write_cache(
                    {
                        "trace_id": finish_trace_id,
                        "instance_id": self.instance_id,
                        "write_session_id": write_session_id,
                        "success_blocks": {"bool_masks": {"values": finish_mask}},
                    }
                )
            except Exception as e:
                logger.error(f"finish_write_cache failed: {e}")
                # Mark all as failed on rank 0 for the return value.
                new_block_success = [False] * unmatched

        # Build result list: 1:1 positional mapping with input keys.
        # Use local_len_new so the return length matches the caller's input.
        # - keys not in save_indices → True (assumed cached / no-op)
        # - keys in save_indices → per-block success from all_reduce
        # When input diverged (skip_transfer), nothing was written and the
        # local keys don't match rank 0's — return all False.
        if skip_transfer:
            return [False] * local_len_new
        block_flag_map = {
            save_indices[j]: new_block_success[j] for j in range(unmatched)
        }
        result_list = [block_flag_map.get(i, True) for i in range(local_len_new)]
        return result_list

    def batch_set_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        trace_id = self._get_trace_id()
        try:
            # The group agreement, not just the local init: _batch_set runs
            # collectives that need every rank (see _ensure_client_for_write).
            self._ensure_client_for_write()
            result = self._batch_set(
                keys=keys,
                host_indices=host_indices,
                trace_id=trace_id,
                extra_info=extra_info,
            )
            return result
        except _UnsupportedBackendError:
            # Already reported once by _ensure_client; keep answering
            # conservatively without logging on every call.
            return [False] * len(keys)
        except Exception as e:
            logger.error(f"batch_set_v1 failed: {trace_id=} {e=}")
            return [False] * len(keys)

    def batch_set_v2(
        self,
        transfers: List[PoolTransfer],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> dict[str, List[bool]]:
        """Write extra pool data (e.g. Mamba/Indexer) in independent write sessions.

        Each PoolTransfer gets its own StartWriteCache -> Save -> FinishWriteCache
        using the pool's spec group, allowing writes to blocks whose KV cache
        was already committed in a separate write session.

        A transfer sglang hands to every rank (``_transfer_is_written_by_every_rank``)
        is written cooperatively, in the sequence the whole group agrees on
        (``_agree_collective_transfers``): rank 0 opens the session, every rank
        writes its own slice with its own rank spec, and the session is only
        committed for the blocks that every rank reported -- a rank-sharded
        sidecar is complete in storage only when all of its shards are.  A
        transfer only rank 0 receives is written here without peers.

        For a cooperative transfer every rank reaches both collectives
        (``broadcast_object_list`` for the session, ``_join_commit_reduce`` for
        the commit) for every transfer of the agreed sequence, whatever it
        decides locally: which ranks have keys, can resolve the indices, parse
        the mask or find anything left to write can differ, and what a rank is
        able to write decides only what it carries into them -- nothing, i.e.
        "wrote nothing" -- never whether it enters.  A branch that returned
        early leaves its peers blocked forever.
        """
        results = {}
        trace_id = self._get_trace_id()
        try:
            # The group agreement, for the calls that will reach a collective
            # below (see _ensure_client_for_write).
            self._ensure_client_for_write(transfers)
            # One sequence for the whole group: which transfers run collectively
            # *and* in which order has to be identical on every rank, because the
            # session broadcast and the commit reduce of the n-th transfer have
            # to meet the same transfer on the peers (see
            # _agree_collective_transfers).
            schedule = self._agree_collective_transfers(transfers)
            agreed = set(schedule)
            # The collectives run in that order.  A transfer only this rank was
            # handed has no position in ``schedule``, enters no collective, and
            # follows them; ``sorted`` is stable, so those keep the order sglang
            # gave them.
            order = {identity: index for index, identity in enumerate(schedule)}
            # One collective pass per pool, and that is a property of the
            # sequence itself: a pool listed twice (under a ``PoolName`` member
            # and under its value text, which is one pool here) would otherwise
            # open two sessions for one spec.  A transfer outside the sequence
            # keeps its own place among the ones that follow.
            ordered: List[PoolTransfer] = []
            seen: set = set()
            for transfer in sorted(
                transfers,
                key=lambda each: order.get(_collective_identity(each.name), len(order)),
            ):
                identity = _collective_identity(transfer.name)
                if identity in agreed:
                    if identity in seen:
                        continue
                    seen.add(identity)
                ordered.append(transfer)
            for transfer in ordered:
                keys = transfer.keys or []
                shared = _collective_identity(transfer.name) in agreed
                # Every rank sglang hands this transfer to takes part in the
                # session; the others are not called for it at all, so waiting
                # for them would hang.
                if not shared and self._transfer_is_written_by_every_rank(transfer):
                    self._report_rank_local_shared_transfer(transfer.name)
                    results[transfer.name] = [False] * len(keys)
                    continue

                # What this rank can write decides only what it carries into the
                # collectives below, never whether it enters them: which ranks
                # can resolve the indices, parse the mask or find anything left
                # to write differs, the collective does not, and a branch that
                # returned here leaves the peers of a shared transfer blocked
                # forever.
                spec_name = self._get_extra_pool_spec_name(transfer.name)
                pool = self.registered_pools.get(transfer.name)
                backed = (
                    spec_name is not None
                    and pool is not None
                    and transfer.name not in self._late_pools
                )
                host_indices: Optional[torch.Tensor] = None
                if keys and not backed:
                    self._report_unknown_pool(transfer.name)
                elif keys:
                    host_indices = self._transfer_host_indices(transfers, transfer)
                    if host_indices is None:
                        logger.debug(
                            "pool %s: no host indices and no resolvable source pool "
                            "(%r); refusing to write",
                            transfer.name,
                            transfer.indices_from_pool,
                        )

                if self.tp_rank == 0:
                    if host_indices is None:
                        # No session to open: there is nothing to fill it with on
                        # any rank that takes part (this rank has no keys, no
                        # spec for the pool, or no indices).  The broadcast below
                        # still runs, so the peers learn it instead of waiting.
                        write_result = None
                    else:
                        spec_group = self._get_extra_pool_spec_group(transfer.name)
                        block_keys, _, _ = self._prepare_block_keys(keys)
                        start_trace_id = f"start-v2-{trace_id}"
                        request = {
                            "trace_id": start_trace_id,
                            "instance_id": self.instance_id,
                            "block_keys": block_keys,
                            "location_spec_group_names": [spec_group] * len(block_keys),
                            "write_timeout_seconds": self.write_timeout_seconds,
                        }
                        try:
                            write_result = self._manager_client.start_write_cache(
                                request
                            )
                        except Exception as e:
                            logger.error(
                                f"start_write_cache failed on rank 0: {trace_id=} {e=}"
                            )
                            write_result = None
                    if shared and self.tp_world_size > 1:
                        torch.distributed.broadcast_object_list(
                            [write_result], src=0, group=self.storage_tp_group
                        )
                elif shared and self.tp_world_size > 1:
                    recv: List[Any] = [None]
                    torch.distributed.broadcast_object_list(
                        recv, src=0, group=self.storage_tp_group
                    )
                    write_result = recv[0]
                else:
                    logger.warning(
                        f"batch_set_v2 called on non-rank-0 (tp_rank={self.tp_rank}) "
                        f"for pool {transfer.name}, which sglang does not write from "
                        f"this rank; returning all False."
                    )
                    results[transfer.name] = [False] * len(keys)
                    continue
                if (
                    host_indices is None
                    or write_result is None
                    or spec_name is None
                    or pool is None
                ):
                    # Nothing to write here, but the peers of a shared transfer
                    # may have a session: this rank still enters the group's
                    # reduce (with zero) or they block on it forever.  Blocks are
                    # committed on the group's minimum, so a zero is exactly
                    # "this rank wrote nothing".  The spec and the pool are named
                    # for symmetry -- a rank without them resolved no indices
                    # either -- so the data path below reads a non-optional pair.
                    self._join_commit_reduce(False, shared)
                    results[transfer.name] = [False] * len(keys)
                    continue

                locations = write_result["locations"]
                write_session_id = write_result["write_session_id"]
                block_mask = write_result["block_mask"]
                finish_trace_id = f"finish-v2-{trace_id}"

                parsed = self._parse_block_mask(block_mask, 0, len(keys))

                if parsed is None:
                    logger.warning(
                        f"batch_set_v2: inconsistent block_mask from manager, "
                        f"aborting write session {write_session_id}"
                    )
                    if self.tp_rank == 0:
                        try:
                            self._manager_client.finish_write_cache(
                                {
                                    "trace_id": finish_trace_id,
                                    "instance_id": self.instance_id,
                                    "write_session_id": write_session_id,
                                    "success_blocks": {
                                        "bool_masks": {
                                            "values": [False] * len(locations)
                                        }
                                    },
                                }
                            )
                        except Exception as e:
                            logger.error(f"finish_write_cache failed: {e}")
                    self._join_commit_reduce(False, shared)
                    results[transfer.name] = [False] * len(keys)
                    continue

                save_indices, _prefix_write_count = parsed
                unmatched = len(save_indices)

                if unmatched == 0:
                    if self.tp_rank == 0:
                        try:
                            self._manager_client.finish_write_cache(
                                {
                                    "trace_id": finish_trace_id,
                                    "instance_id": self.instance_id,
                                    "write_session_id": write_session_id,
                                    "success_blocks": {
                                        "bool_masks": {
                                            "values": [False] * len(locations)
                                        }
                                    },
                                }
                            )
                        except Exception as e:
                            logger.error(f"finish_write_cache failed: {e}")
                    # Already in storage on this rank, so it contributes True;
                    # the reduce still has to happen, because a peer that cannot
                    # write this transfer is waiting on it.
                    cached = self._join_commit_reduce(True, shared)
                    results[transfer.name] = [cached] * len(keys)
                    continue

                assert len(save_indices) + _prefix_write_count == len(locations)

                # Data transfer preparation and execution.
                # Wrapped in try-except so that every rank always reaches the
                # all_reduce below, preventing cross-rank NCCL/gloo hangs.
                try:
                    ptr_list, size_list, components = self._page_buffer_meta(
                        pool, transfer, host_indices
                    )
                    save_set = set(save_indices)
                    ptr_list = [
                        p
                        for i, p in enumerate(ptr_list)
                        if (i // components) in save_set
                    ]
                    size_list = [
                        s
                        for i, s in enumerate(size_list)
                        if (i // components) in save_set
                    ]

                    uris = []
                    for loc in locations:
                        uri = self._extract_single_spec_uri(loc, spec_name)
                        if uri:
                            uris.append(uri)
                    buffers = self._prepare_extra_pool_buffers(
                        ptr_list, size_list, components
                    )
                    assert len(uris) == len(buffers)
                    start_time = time.perf_counter()
                    save_result = self.transfer_client.SaveKvCaches(uris, buffers)
                    end_time = time.perf_counter()
                    flag = save_result[0] == kvcm_py_client.ClientErrorCode.ER_OK
                    if flag:
                        spec_size = self._get_extra_pool_spec_size(transfer.name)
                        self.backup_pgs.append(unmatched)
                        self.backup_bandwidth.append(
                            unmatched * spec_size / (1 << 30) / (end_time - start_time)
                        )
                    if not flag:
                        logger.error(f"SaveKvCaches v2 error: {transfer.name}")
                except Exception as e:
                    logger.error(
                        f"Data transfer v2 (SaveKvCaches) failed: {transfer.name} {e}"
                    )
                    flag = False

                # The block is complete in storage only when every rank wrote
                # its slice: commit on the group's minimum.
                flag = self._join_commit_reduce(flag, shared)

                finish_mask = [flag] * len(locations)
                if self.tp_rank == 0:
                    try:
                        self._manager_client.finish_write_cache(
                            {
                                "trace_id": finish_trace_id,
                                "instance_id": self.instance_id,
                                "write_session_id": write_session_id,
                                "success_blocks": {
                                    "bool_masks": {"values": finish_mask}
                                },
                            }
                        )
                    except Exception as e:
                        logger.error(f"finish_write_cache failed: {e}")

                per_key = [True] * len(keys)
                for idx in save_indices:
                    per_key[idx] = flag
                results[transfer.name] = per_key

            return results
        except _UnsupportedBackendError:
            # Already reported once by _ensure_client; keep answering
            # conservatively without logging on every call.
            return {t.name: [False] * len(t.keys or []) for t in transfers}
        except Exception as e:
            logger.error(f"batch_set_v2 failed: {trace_id=} {e=}")
            return {t.name: [False] * len(t.keys or []) for t in transfers}

    def _batch_exists(
        self,
        keys: List[str],
        trace_id: str,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> int:
        block_keys, len_prefix, len_new = self._prepare_block_keys(keys, extra_info)
        get_request = {
            "trace_id": trace_id,
            "block_keys": block_keys,
            "instance_id": self.instance_id,
            "query_type": "QT_PREFIX_MATCH",
            "block_mask": {"offset": len_prefix},
        }
        result = self._manager_client.get_cache_location(get_request)
        logger.debug(f"get_cache_location {result=}")
        return len(result["locations"])

    def batch_exists(
        self,
        keys: List[str],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> int:
        trace_id = self._get_trace_id()
        try:
            self._ensure_client()
            result = self._batch_exists(
                keys=keys, trace_id=trace_id, extra_info=extra_info
            )
            return result
        except _UnsupportedBackendError:
            # Already reported once by _ensure_client; keep answering
            # conservatively without logging on every call.
            return 0
        except Exception as e:
            logger.error(f"batch_exists failed: {trace_id=} {e=}")
            return 0

    def batch_exists_v2(
        self,
        keys: List[str],
        pool_transfers: Optional[List[PoolTransfer]] = None,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> PoolTransferResult:
        trace_id = self._get_trace_id()
        try:
            self._ensure_client()
            # Reuse the same get_cache_location call as batch_exists,
            # but inspect per-location specs for extra pool existence.
            block_keys, len_prefix, len_new = self._prepare_block_keys(keys, extra_info)
            get_request = {
                "trace_id": trace_id,
                "block_keys": block_keys,
                "instance_id": self.instance_id,
                "query_type": "QT_PREFIX_MATCH",
                "block_mask": {"offset": len_prefix},
            }
            result = self._manager_client.get_cache_location(get_request)
            locations = result["locations"]

            # Count KV hit pages: prefix-match only locations that carry
            # the KV ("Full") spec.  get_cache_location returns any block
            # registered via start_write_cache regardless of spec group,
            # so we must filter by checking each location's specs.
            kv_hit_pages = 0
            for loc in locations:
                if any(
                    spec["name"] == self.location_spec_name
                    for spec in loc.get("location_specs", [])
                ):
                    kv_hit_pages += 1
                else:
                    break  # prefix match: stop at first gap
            pool_hit_pages = {PoolName.KV: kv_hit_pages} if kv_hit_pages else {}
            final_pages = kv_hit_pages

            # Check extra pool spec existence
            for transfer in pool_transfers or []:
                if final_pages == 0:
                    break
                boundary = self._check_pool_spec_existence(
                    locations, kv_hit_pages, transfer
                )
                pool_hit_pages[transfer.name] = boundary
                final_pages = min(final_pages, boundary)

            return PoolTransferResult(final_pages, pool_hit_pages)
        except _UnsupportedBackendError:
            # Already reported once by _ensure_client; keep answering
            # conservatively without logging on every call.
            return PoolTransferResult.empty()
        except Exception as e:
            logger.error(f"batch_exists_v2 failed: {trace_id=} {e=}")
            return PoolTransferResult.empty()

    def get_stats(self) -> Any:
        storage_metrics = StorageMetrics()
        storage_metrics.prefetch_pgs.extend(self.prefetch_pgs)
        storage_metrics.backup_pgs.extend(self.backup_pgs)
        storage_metrics.prefetch_bandwidth.extend(self.prefetch_bandwidth)
        storage_metrics.backup_bandwidth.extend(self.backup_bandwidth)
        self.prefetch_pgs.clear()
        self.backup_pgs.clear()
        self.prefetch_bandwidth.clear()
        self.backup_bandwidth.clear()
        return storage_metrics

    def close(self) -> None:
        """Release what this backend owns; sglang calls it on detach.

        sglang builds a fresh backend instance on re-attach, so nothing has to
        be re-armed here and the method is idempotent.

        close() takes the initialization lock, so it cannot interleave with
        the one-time init: either it wins and the init then fails on
        ``_closed``, or it waits and releases everything the init created.  A
        client created during the race can therefore never stay referenced
        (upstream also joins the storage threads before detaching, but the
        ordering is enforced here instead of assumed).

        Dropping ``transfer_client`` runs the pybind destructor of the SDK
        wrapper.  The Manager HTTP client (its session and leader-refresh
        thread) and the gloo group created for the TP collectives are per
        instance resources too, so they are released explicitly.

        The host pool spans go after the client: the backend owns their
        registration and holds raw addresses into them for the client's
        lifetime, so the mappings may only be dropped once it is gone.
        """
        with self._init_lock:
            if self._closed:
                return
            self._closed = True
            self.transfer_client = None
            self.init_params = None
            self._host_pool_spans = []
            self._manager_client.close()

            group = getattr(self, "storage_tp_group", None)
            if group is not None:
                self.storage_tp_group = None
                try:
                    torch.distributed.destroy_process_group(group)
                except Exception as e:
                    logger.warning("close: storage_tp_group not destroyed: %s", e)

    ##################################################

    def _tp_rank_to_spec_name(self, tp_rank: int) -> str:
        """The KV spec name: ``tp_{rank}``, ``tp_{rank}_full`` with sidecars."""
        return _KV_SPEC.spec_name(tp_rank, self.has_sidecars)

    def _tp_rank_to_linear_spec_name(self, tp_rank: int) -> str:
        return _MAMBA_SPEC.spec_name(tp_rank, self.has_sidecars)

    def _tp_rank_to_indexer_spec_name(self, tp_rank: int) -> str:
        return _INDEXER_SPEC.spec_name(tp_rank, self.has_sidecars)

    def _get_trace_id(self) -> str:
        return str(uuid.uuid1())

    def _sha256_to_int64(self, data: str) -> int:
        hash_digest = hashlib.sha256(data.encode("utf-8")).digest()
        return int.from_bytes(hash_digest[:8], "big", signed=True)

    def _prepare_block_keys(
        self, keys: List[str], extra_info: Optional[HiCacheStorageExtraInfo] = None
    ) -> tuple[List[int], int, int]:
        """Prepare block keys and return them along with the prefix offset."""
        prefix_keys = (
            extra_info.prefix_keys
            if (extra_info is not None) and (extra_info.prefix_keys is not None)
            else []
        )
        block_keys = prefix_keys + keys
        block_keys = [self._sha256_to_int64(block_key) for block_key in block_keys]
        return block_keys, len(prefix_keys), len(keys)

    def _extract_uris(self, locations: List[dict]) -> List[str]:
        """Extract URIs from locations for the current TP rank."""
        uris = []
        for location in locations:
            for location_spec in location["location_specs"]:
                if location_spec["name"] == self.location_spec_name:
                    uris.append(location_spec["uri"])
        return uris

    def _prepare_buffers(
        self, buffer_ptrs: List[int], buffer_sizes: List[int]
    ) -> List[kvcm_py_client.BlockBuffer]:
        """Prepare buffers for data transfer."""
        buffers = []
        for i in range(0, len(buffer_ptrs), self.kv_factor):
            buffer = kvcm_py_client.BlockBuffer()
            iovs = []
            for j in range(self.kv_factor):
                iov = kvcm_py_client.Iov()
                iov.type = kvcm_py_client.MemoryType.CPU
                iov.base = buffer_ptrs[i + j]
                iov.size = buffer_sizes[i + j]
                iov.ignore = False
                iovs.append(iov)
            buffer.iovs = iovs
            buffers.append(buffer)
        return buffers

    def _parse_block_mask(
        self, block_mask: dict, len_prefix: int, len_new: int
    ) -> Optional[tuple[List[int], int]]:
        """Parse block_mask from manager to determine which new-block indices need writing.

        Returns:
            tuple[List[int], int]:
                - save_indices: indices (relative to new blocks) that need writing.
                  Empty list means all new blocks are already cached.
                - prefix_write_count: number of prefix blocks the manager wants
                  written that we cannot fulfil (best-effort skip).
            None: manager returned truly broken data (e.g. incomplete bool_masks);
                  caller should treat as a total write failure.
        """
        save_indices = []
        prefix_write_count = 0
        if "offset" in block_mask:
            offset = block_mask["offset"]
            if offset < len_prefix:
                # Best-effort: prefix blocks [offset, len_prefix) can't be written
                # by sglang (no data available), but new blocks can still proceed.
                logger.warning(
                    f"_parse_block_mask: offset {offset} < len_prefix {len_prefix}, "
                    "prefix blocks will be skipped (best-effort)"
                )
                prefix_write_count = len_prefix - offset
                save_indices.extend(range(len_prefix, len_prefix + len_new))
            else:
                save_indices.extend(range(offset, len_prefix + len_new))
        else:
            # False: need to store
            bool_masks = block_mask.get("bool_masks", {}).get("values", [])
            if len(bool_masks) < len_prefix + len_new:
                # Incomplete mask data from manager.
                logger.warning(
                    f"_parse_block_mask: bool_masks length {len(bool_masks)} < "
                    f"expected {len_prefix + len_new}, treating as inconsistent state"
                )
                return None
            prefix_write_count = sum(1 for v in bool_masks[:len_prefix] if not v)
            if prefix_write_count > 0:
                logger.warning(
                    f"_parse_block_mask: {prefix_write_count} prefix blocks "
                    "not cached in bool_masks, will be skipped (best-effort)"
                )
            max_index = max([i for i, x in enumerate(bool_masks) if not x], default=-1)
            save_indices.extend(
                [i for i in range(len_prefix, max_index + 1) if not bool_masks[i]]
            )
        save_indices = [(i - len_prefix) for i in save_indices if i >= len_prefix]
        return save_indices, prefix_write_count

    def _extract_single_spec_uri(self, location: dict, spec_name: str) -> Optional[str]:
        """Extract the URI for a named spec from a single location dict."""
        for spec in location.get("location_specs", []):
            if spec["name"] == spec_name and spec.get("uri"):
                return spec["uri"]
        return None

    @staticmethod
    def _page_buffer_meta(
        pool: Any, transfer: PoolTransfer, host_indices: torch.Tensor
    ) -> tuple[List[int], List[int], int]:
        """Per-page zero-copy meta of an extra pool plus its IOVs per page.

        ``host_indices`` come from the transfer itself or from the pool it rides
        on (see ``_transfer_host_indices``).

        The component count is read back from the pool's own output instead of
        being derived from pool internals: mamba pools emit a temporal IOV only
        while the model has an SSM state, so conv-only models hand out one IOV
        per conv buffer.  Each page must also match exactly one key: callers
        slice the flat IOV list by page position and map it back to key
        position, so both a wrong component count and a page/key mismatch would
        silently shift every page's data.  Both are asserted here instead.
        """
        ptr_list, size_list = pool.get_page_buffer_meta(host_indices)
        num_pages = len(host_indices) // pool.page_size
        assert num_pages == len(transfer.keys or []), (
            f"pool transfer {transfer.name} carries {num_pages} host pages "
            f"(page size {pool.page_size}) for {len(transfer.keys or [])} keys"
        )
        assert num_pages > 0 and len(ptr_list) % num_pages == 0, (
            f"get_page_buffer_meta returned {len(ptr_list)} IOVs for "
            f"{len(host_indices)} host indices (page size {pool.page_size})"
        )
        return ptr_list, size_list, len(ptr_list) // num_pages

    def _get_kv_spec_group(self) -> str:
        """Spec group name used in start_write_cache for KV pool."""
        return _KV_SPEC.group

    def _get_extra_pool_spec_group(self, pool_name: Any) -> str:
        """Spec group name used in start_write_cache for an extra pool."""
        pool = self._extra_pool(pool_name)
        if pool is None:
            raise ValueError(f"Unknown extra pool: {pool_name}")
        return pool.spec.group

    def _get_extra_pool_spec_size(self, pool_name: Any) -> int:
        """Per-block spec size in bytes for bandwidth tracking."""
        pool = self._extra_pool(pool_name)
        return 0 if pool is None else pool.size

    def _extra_pool(self, pool_name: Any) -> Optional[_RegisteredPool]:
        """The pool this instance backs under that name, if it is an extra pool."""
        pool = self.pool_specs.get(pool_name)
        if pool is None or pool.spec.anchor:
            return None
        return pool

    def _transfer_host_indices(
        self, transfers: List[PoolTransfer], transfer: PoolTransfer
    ) -> Optional[torch.Tensor]:
        """The host indices a transfer moves, following a sidecar to its source.

        A sidecar sglang's tree builds carries ``indices_from_pool`` and no host
        indices of its own: it reuses another pool's slots and only the bytes
        behind them belong to the sidecar.  The engine normally materializes
        them before calling the backend (``HostPoolGroup.resolve_host_transfers``
        does the same lookup, ``pool_host/group.py``); when it has not, the
        sibling transfer of the same call is the source -- and one hop is all
        there is, because a source is a pool of its own.  A source outside the
        call (the KV anchor is not part of a v2 call) stays unresolved, and the
        caller answers conservatively instead of raising: an exception here would
        kill the backup thread it runs on.
        """
        if transfer.host_indices is not None:
            return transfer.host_indices
        source = transfer.indices_from_pool
        if source is None:
            return None
        return next(
            (
                other.host_indices
                for other in transfers
                if other.name == source and other.host_indices is not None
            ),
            None,
        )

    def _prepare_extra_pool_buffers(
        self, ptr_list: List[int], size_list: List[int], components_per_page: int
    ) -> List[kvcm_py_client.BlockBuffer]:
        """Convert get_page_buffer_meta output to BlockBuffer list.

        Each logical page maps to `components_per_page` IOVs in a single BlockBuffer.
        Works for both Mamba (temporal + conv components) and Indexer (single component).
        """
        buffers = []
        for i in range(0, len(ptr_list), components_per_page):
            buffer = kvcm_py_client.BlockBuffer()
            iovs = []
            for j in range(components_per_page):
                iov = kvcm_py_client.Iov()
                iov.type = kvcm_py_client.MemoryType.CPU
                iov.base = ptr_list[i + j]
                iov.size = size_list[i + j]
                iov.ignore = False
                iovs.append(iov)
            buffer.iovs = iovs
            buffers.append(buffer)
        return buffers

    def _get_extra_pool_spec_name(self, pool_name: Any) -> Optional[str]:
        """Map a PoolName to its KVCM location spec name for the current rank."""
        pool = self._extra_pool(pool_name)
        return None if pool is None else pool.name

    def _check_pool_spec_existence(
        self,
        locations: List[dict],
        kv_hit_pages: int,
        transfer: PoolTransfer,
    ) -> int:
        """Check how many pages have the extra pool's spec.

        Returns the number of contiguous prefix pages that satisfy
        transfer.hit_policy.  Falls back to 0 for unknown policies so
        a stale/future enum value never causes an UnboundLocalError crash.
        """
        spec_name = self._get_extra_pool_spec_name(transfer.name)
        if spec_name is None or transfer.name in self._late_pools:
            # No spec was registered for this pool -- it was never registered,
            # arrived after the specs were frozen, or its name would share
            # another pool's spec: its data is never written, so claiming hits
            # here would only make the caller move KV pages that batch_get_v2
            # then reports as misses.
            self._report_unknown_pool(transfer.name)
            return 0

        def has_spec(loc: dict) -> bool:
            return any(
                spec["name"] == spec_name for spec in loc.get("location_specs", [])
            )

        if transfer.hit_policy == PoolHitPolicy.ALL_PAGES:
            # First gap in the prefix is the boundary.
            return next(
                (i for i in range(kv_hit_pages) if not has_spec(locations[i])),
                kv_hit_pages,
            )

        if transfer.hit_policy == PoolHitPolicy.TRAILING_PAGES:
            trailing = max(1, len(transfer.keys) if transfer.keys else 1)
            for prefix_len in range(kv_hit_pages, 0, -1):
                if all(
                    has_spec(locations[i])
                    for i in range(max(0, prefix_len - trailing), prefix_len)
                ):
                    return prefix_len
            return 0

        logger.warning(
            "_check_pool_spec_existence: unknown PoolHitPolicy %r, defaulting to 0",
            transfer.hit_policy,
        )
        return 0

    def _join_commit_reduce(self, flag: bool, shared: bool) -> bool:
        """Fold this rank's contribution into the group's commit flag.

        A shared transfer commits a block only when every rank that takes part
        reported it written, so this is one ``all_reduce(MIN)`` over the group
        and *every* participating rank has to call it exactly once -- whatever
        it decided locally.  A branch that returns without it leaves its peers
        blocked on the collective forever, and whether the indices resolve, the
        mask parses or anything needs writing can differ per rank.
        """
        if not (shared and self.tp_world_size > 1):
            return flag
        flag_tensor = torch.tensor(flag, dtype=torch.int)
        torch.distributed.all_reduce(
            flag_tensor,
            op=torch.distributed.ReduceOp.MIN,
            group=self.storage_tp_group,
        )
        return bool(flag_tensor.item())

    def _report_unknown_pool(self, pool_name: Any) -> None:
        """Report once that sglang moved data for a pool this instance misses."""
        if not self._mark_pool_reported(pool_name):
            return
        logger.warning(
            "pool %s is not backed by this connector: no location spec was "
            "registered for it, so its transfers are reported as misses (the "
            "pool was never registered, arrived after the specs were frozen, "
            "or its name would share another pool's spec).",
            pool_name,
        )

    ##################################################

    def clear(self) -> None:
        raise NotImplementedError()

    def exists(self, key: str) -> bool:
        raise NotImplementedError()

    def get(
        self,
        key: str,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> torch.Tensor | None:
        raise NotImplementedError()

    def batch_get(
        self,
        keys: List[str],
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> List[torch.Tensor | None] | int:
        """Legacy page interface (sglang <= 0.5.18 draft/MTP path).

        KVCM backs KV/Mamba/Indexer location specs, not the draft pool, so the
        honest answer is "nothing stored": the caller skips every ``None``.

        Answering instead of raising is about visibility, not about keeping
        the request alive.  Upstream wraps the draft functions in a bare
        ``except Exception`` that logs at DEBUG, so the old
        ``NotImplementedError`` was swallowed silently; the one-time WARNING
        below names the missing L3 path instead.  (Only with ``interface_v1``
        forced to 0 would this interface also carry KV, where ``_page_backup``
        has no handler and a raise would kill the backup thread; the connector
        sets ``interface_v1 = 1`` itself, so that needs an explicit override.)
        """
        self._report_legacy_op("batch_get")
        return [None] * len(keys)

    def set(
        self,
        key: str,
        value: Optional[Any] = None,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        raise NotImplementedError()

    def batch_set(
        self,
        keys: List[str],
        values: Optional[Any] = None,
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        """Legacy page interface (sglang <= 0.5.18 draft/MTP path).

        Same reasoning as ``batch_get``: the draft wrapper swallows exceptions
        into a DEBUG log and discards the return value, so a failed write is
        equivalent in effect -- and the one-time WARNING below makes it
        visible.
        """
        self._report_legacy_op("batch_set")
        return False

    def _report_legacy_op(self, op: str) -> None:
        """Report once per legacy entry point that it is not implemented."""
        if op in self._warned_legacy_ops:
            return
        self._warned_legacy_ops.add(op)
        logger.warning(
            "%s: this backend only serves the v1/v2 location-spec interfaces; "
            "draft/MTP L3 is not backed by KVCM, so the call is answered "
            "conservatively (miss / failed write) instead of raising an error "
            "that upstream would only log at DEBUG.",
            op,
        )
