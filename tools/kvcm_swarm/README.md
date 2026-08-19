# kvcm_swarm

`kvcm_swarm` is a standalone C++ client-behavior runner and evidence system for
KVCM. It generates real HTTP/gRPC metadata traffic from simulated clients that
have their own identity, connections, sessions, timing, local cache and
lifecycle, and it writes a complete fact report.

```bash
bazel build //tools/kvcm_swarm
./bazel-bin/tools/kvcm_swarm/kvcm_swarm --config run_config.json
./bazel-bin/tools/kvcm_swarm/kvcm_swarm --config run_config.json --validate-only
```

* Design: [`docs/design/kvcm_swarm.md`](../../docs/design/kvcm_swarm.md)
* Implementation design: [`docs/design/kvcm_swarm_impl.md`](../../docs/design/kvcm_swarm_impl.md)
* CI orchestration and the scenario evaluator: `integration_test/swarm/`

## What it is and is not

* It is metadata-only: no KV bytes are transferred.
* It talks to KVCM through public APIs only. It never creates or modifies
  storages or instance groups: that belongs to `integration_test/swarm/`.
* It records facts. PASS/FAIL is decided out of process by
  `integration_test/swarm/evaluator.py`.
* Only plaintext HTTP and insecure gRPC are supported; HTTPS/TLS/mTLS are
  rejected, never silently downgraded.

## Layout

```text
app/        run coordinator (phases, stop, report) and preflight
scenario/   Jsonizable configuration objects, strict loading and validation
runtime/    executor, timers, RNG, admission lanes, async primitives
protocol/   aliases of the project-level async RPC protocol types
transport/  admission/evidence adapter over the project-level async RPC client
evidence/   observations, histograms, violation log and report rendering
clients/    behavior contract, registry, v6d_deployment, health_probe
test/       unit tests, including a V6D-free fake behavior
```

The reusable plaintext HTTP and insecure gRPC implementations live in
`kv_cache_manager/client/src/internal/async_rpc/`. They are internal project
infrastructure and are not exported from `kv_cache_manager_client.so`.

## Exit codes

| Code | Meaning |
| --- | --- |
| 0 | the run completed and the report was written |
| 2 | the configuration is invalid (local validation only, no RPC was sent) |
| 3 | preflight failed: the environment precondition is not met |
| 4 | initialize failed (for example a process could not register) |
| 5 | the report could not be produced |

The exit code says nothing about scenario thresholds. Sample counts, success
rates, latency and contract gating are the evaluator's job.

The first release gates small-scale correctness only. Configuration and
evidence for deployment-scale claims are added after the tool is ready and the
claimed scale has actually been exercised.
