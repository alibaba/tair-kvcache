# MetaService Reactor validation (2026-11-10)

All 13 MetaService unary gRPC methods and their HTTP equivalents share one bounded worker pool,
with a default waiting capacity of 2048. See the [design](../design/service_request_dispatch.md)
for configuration, metric semantics, overload responses and shutdown behavior.

## Source and environment

Branch: `feature/kvcm-v9-meta-reactor-20261110`, based on `release/kvcm_v9_fix_1106`
at `13e2046c2be6693ee9aab2a3237d5f617f02621f`.
Validation used a Linux x86-64 container, GCC 10.2.1 and Bazel 6.4.0.
Modified code/BUILD files were compared by SHA-256 with the tested copies.
Tests used `--config=debug --config=asan --test_env=ASAN_OPTIONS=detect_odr_violation=0`;
address and leak checking remained enabled. Results include valid Bazel cache hits.

## Results

| Scope | Result |
|---|---|
| Open-source `//kv_cache_manager/... //integration_test/...` | 161 targets passed, 1 skipped |
| Internal applicable unit-test set | 146 targets passed, 1 skipped |
| Standalone server scenarios | 7 passed |
| Python syntax and `git diff --check` | Passed |

Both modes skipped `SdkBufferCheckUtilTest`. The internal run excluded an existing HF3FS allocator
test with missing default BUILD dependencies and a dedicated external mount/service requirement.
These results do not claim that target passed.

Focused XML case counts, all with zero failures/errors:

| Target | Cases |
|---|---:|
| ServiceRequestExecutorTest | 15 |
| MetaServiceGRpcDispatchTest | 51 |
| CoroHttpDispatchTest | 7 |
| ServerMetricsTest | 5 |
| LoggingMetricsReporterTest | 6 |
| KmonitorMetricsReporterTest | 9 |

gRPC tests use 1/2/4 CQ pollers and cover real sockets, saturation, deadlines, cancellation,
leadership gates and startup failures. HTTP tests verify IO progress during blocked business/collector
work, resumption on IO, and Arena/cached JSON lifetimes.

The process test checks all generated methods over both protocols, followed by 64 mixed concurrent calls.
Seven scenarios cover separate/shared RPC ports with both shutdown policies, capacity 7, a Meta port
conflict, and an Admin port conflict after Meta starts. Reachability checks include invalid-argument
and missing-instance responses; completion counts are not business success rates.

## Logging metrics

Logging validation used a 100 ms interval, two workers and one HTTP IO thread. Default-capacity runs
reported 2048. One capacity-7 run produced:

| Metric | Value |
|---|---:|
| queue capacity / peak / time-weighted average | 7 / 7 / 5.6467 |
| rejected / completed, cumulative | 42 / 51 |
| wait MAX / AVG, microseconds | 116644 / 87279.4 |
| execution MAX / AVG, microseconds | 44268 / 27104 |

The test asserts nonzero timings, AVG <= MAX and bounded queue peaks. These values include full
logging overhead and are not a performance baseline. Console metrics are read from captured process
output; the default file logger writes to `logs/metrics.log`.

## Boundaries

Active business operations are not interrupted. Explicit client cancellation and HTTP disconnects
do not revoke queued operations. gRPC tasks whose server deadline has expired before worker execution
skip business and finish with DEADLINE_EXCEEDED. Completed-task metrics include these skipped callbacks.
An overloaded HTTP request is rejected before JSON decoding, including malformed JSON; its response
has a server-generated request ID because the caller's trace ID has not been decoded.

## Simplification review (2026-11-14)

The PR was reviewed again for admission, protocol handoff and shutdown lifetime:

- Combine gRPC preparation and business execution into one worker callback; remove
  the separate preparation callback and unreachable successful-error fallback.
- Allocate completion state only after checking queue admission, so full/closed
  rejections do not allocate a completion lease. Retain the outer lease reference
  to keep allocation-failure cleanup outside the queue mutex.
- Use the existing server-start exception handler instead of a nested duplicate.
- Add an ingress-failure regression: return UNKNOWN without submitting business
  work, then verify that the next RPC and the shared Admin service still work.

After these changes, external ASAN passed 161 targets with 1 skip; the internal
applicable set passed 146 targets with 1 skip, with the same exclusions above.
These totals include valid cache hits (37 external and 18 internal targets were
executed again). The gRPC dispatch target passed all 48 cases across 1/2/4 CQ
pollers. The real-process integration target passed all seven scenarios and its
Logging metric assertions. Changed code/BUILD files matched the tested copies in
both source modes by SHA-256.

The earlier sustained-load measurements used revision `ad4fc9e5`, before these
simplifications. They have not been rerun for this review revision; no new
performance improvement or deployment of this revision is claimed.

## Overload and failed-start regression checks (2026-11-14)

Three focused regressions were reproduced before their fixes:

- An expired queued gRPC request still executed business with 1/2/4 CQ pollers.
  Workers now check the server deadline before starting business.
- A full HTTP queue still decoded JSON on the IO thread before rejecting it.
  A shared admission check now rejects full/closed requests before parsing; it does
  not reserve a slot, and final TrySubmit still enforces the bound. Each rejection is counted once.
- A first-port bind failure skipped gRPC Shutdown and leaked its default quota resource user.
  The three poller configurations leaked 3660 bytes in total under ASAN/LSAN. A small
  patch calls Shutdown on that failure path, with the same patch enabled in both source modes.

After the fixes, external ASAN passed 161 targets with 1 skip (78 executed again), and
the internal applicable set passed 146 targets with 1 skip (61 executed again), with
the exclusions documented above. The 15 executor, 51 gRPC and 7 HTTP cases passed.
All seven real-process scenarios passed, including both port conflicts with the
configured default quota. The previous grpc_max_threads=-1 test workaround was removed;
address and leak checking remained enabled. These are correctness checks, not a new
five-hour performance result.

## Reproduce

```bash
bazelisk test --config=debug --config=asan \
  --test_env=ASAN_OPTIONS=detect_odr_violation=0 \
  //kv_cache_manager/... //integration_test/...

bazel-bin/integration_test/service_dispatch/service_dispatch_smoke \
  --binary bazel-bin/kv_cache_manager/kv_cache_manager_bin \
  --evidence-dir /path/to/evidence
```
