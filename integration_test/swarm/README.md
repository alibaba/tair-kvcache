# integration_test/swarm

CI orchestration for KVCM Swarm. Everything here runs **outside** the generator
process:

| File | Responsibility |
| --- | --- |
| `fixture.py` | creates an isolated event-report storage, instance group and quota, renders the effective C++ run configuration, tears the environment down |
| `runner.py` | starts the real `kvcm_swarm` binary and collects out-of-process facts |
| `evaluator.py` | turns one report plus one expectations file into PASS/FAIL, failing closed |
| `scenarios/` | C++ run-configuration templates (endpoints and identities are injected by the fixture) |
| `expectations/` | evaluator gates: minimum samples, success rate, latency, required contracts |

The fixture never proxies an RPC and never changes traffic because a gate was
not met. A unique instance group and a unique `instance_id` are generated per
run: the generator is deterministic, so reusing an instance would inherit the
previous run's locations and change replica-threshold behavior.

```bash
bazel test //integration_test/swarm/...
```

| Test | Covers |
| --- | --- |
| `test_swarm_http` | HTTP normal workload, multi-process shared `instance_id`, per-process contexts, phase buckets, group shapes |
| `test_swarm_grpc` | the same workload over insecure gRPC, one channel per endpoint |
| `test_swarm_capacity` | capacity-driven eviction with both writable and masked cold writes, C3 ordering, no `RemoveCache` |
| `test_swarm_health` | health probe standalone, and progressing while business RPCs greatly outnumber Executor workers |
| `test_swarm_drain` | drain order, shutdown flush, `HOST_DOWN`, residual reporting, and the bounded preflight failure exit code |
| `test_swarm_direct_json` | the binary driven by hand-written JSON with no Python in the loop, plus local-only rejection of an invalid configuration |
| `test_swarm_transport_contract` | every allowed API exercised successfully over both transports, with lane and endpoint separation |
| `test_swarm_evaluator` | the evaluator fails closed on empty reports, `NOT_RUN`, `INCONCLUSIVE`, missing fields and generator saturation |

The first release gates only small-scale correctness. Deployment-scale claims
and their scenario configurations are added after the tool is ready and each
claimed scale has been executed against a recorded topology.
