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
