# GetHostCacheState stress tool

`get_host_cache_state_stress.sh` validates concurrent `ReportEvent` writes and
`GetHostCacheState` reads, including full-attention/Mamba component isolation.
Every KVCM request is sent with `curl`.

The script is self-documenting; use its help as the source of truth:

```bash
./tools/scripts/get_host_cache_state_stress.sh --help
```

The shortest isolated Mamba/full-attention reproduction is:

```bash
BOOTSTRAP_TEST_GROUP=1 RUN_CASES=mamba \
  ./tools/scripts/get_host_cache_state_stress.sh
```

It creates uniquely named test resources and removes them on exit by default.
The help output documents endpoint overrides, load controls, all cases, safety
behavior, result interpretation, and how to retain request/response artifacts.

## ReportEvent correctness and load tool

`report_event_load.py` mixes incremental ADD/DELETE events, periodic
authoritative snapshots, heartbeat events, and standalone reads. It establishes
an empty initial test baseline for deterministic shadow-state validation;
production reporters may send deltas immediately after REGISTER.

Every `ReportEvent` request is followed by `GetHostCacheState` and checked
against a per-host in-memory authoritative state. Report success, committed
snapshot token continuity, prefix-match contents, snapshot reconciliation, and
deleted-key invisibility are therefore validated together.

`s_version` is a reconciliation/cleanup generation, not a strict query fence.
Production queries may return well-formed older or legacy cache candidates
until a successful snapshot reclaims them.

Run it with a new instance id. By default the tool rejects a reporter that
already has a committed snapshot, because its startup empty snapshot would
replace all existing reporter state. `--allow-reuse-instance` is available only
for an intentional destructive reset.

The instance group must already contain the selected EventReport storage. By
default, `--storage-type auto` probes L2 and then L1P5 after `registerInstance`;
if neither backend is configured, the tool reports that
`event_report_storage_candidates` is missing instead of misdiagnosing it as an
instance-creation failure:

```bash
python3 tools/scripts/report_event_load.py \
  --base-url http://127.0.0.1:6382 \
  --instance-group vllm_kvcm_test_2 \
  --instance-id report_event_load_001 \
  --host-count 3 \
  --duration-sec 60 \
  --add-qps 30 \
  --delete-qps 30 \
  --get-qps 30 \
  --snapshot-interval-sec 35 \
  --snapshot-drop-ratio 0.05 \
  --add-batch-size 10 \
  --delete-batch-size 10 \
  --workers 64
```

`--snapshot-interval-sec` is per host and must not be shorter than the
EventReport backend's configured snapshot rate limit. Setting
`--snapshot-drop-ratio` above zero deliberately omits a portion of the current
host state from each full snapshot and verifies those omitted blocks are
eventually removed. If the backend responds with `SNAPSHOT_RATE_LIMITED`, the tool uses
`retry_after_ms` to stop resubmitting that host until its cooldown expires and
reports the configured interval as invalid.
