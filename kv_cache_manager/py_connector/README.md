# py_connector

Python-side connectors between serving engines and KVCM:

- `vllm/` — vLLM v1 KV connector (scheduler + worker roles)
- `sglang/` — SGLang hicache storage backend
- `trtllm/` — TensorRT-LLM KV cache connector (example)
- `common/` — engine-agnostic plumbing (manager HTTP client, TP coordinator)
- `kernel/` — Triton gather/scatter kernels
- `test/` — pure-logic unit tests (no GPU / vLLM / pybind needed)

## Static checks (ruff + ty)

All Python under this directory is gated by three checks, configured in
`ruff.toml` / `ty.toml` next to this file. Run them manually from here:

```bash
cd kv_cache_manager/py_connector
ruff check .            # lint: ANN (annotations) + F (pyflakes)
ruff format --check .   # formatting
ty check                # type checking
```

Install the tools any way you like, e.g. `uv tool install ruff ty` or
`pip install ruff ty`. Both are dev-only tools and intentionally absent
from every runtime dependency list (`BUILD` / `requirements.txt`).

### Type environment

`ty` resolves third-party imports against the Python environment it
discovers: a virtualenv named `.venv` in this directory or any parent
(typically the repository root; it is git-ignored). Build one containing
the engine you work on, e.g. for the vLLM connector:

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python 'vllm==0.26.0' requests orjson pydantic
```

Extra packages per subpackage (install `--no-deps` to keep the pinned
vLLM intact):

- `sglang/` needs `sglang` (tested with 0.5.19; `--no-deps` avoids it
  resolving its own vLLM pin).
- `trtllm/` has no pip-installable source distribution: copy the
  `tensorrt_llm` package of a TensorRT-LLM checkout (v1.2.x) into the
  same site-packages
  (`cp -r <TensorRT-LLM>/tensorrt_llm .venv/lib/python3.12/site-packages/`).

A few imports cannot be resolved statically and carry inline
`# ty: ignore[unresolved-import]` comments where that is permanent:

- compiled modules without stubs: `kvcm_py_client` (pybind11) and
  `tensorrt_llm.bindings`;
- `_version_info`, generated into the wheel at build time (bazel
  `version_info_py`), absent from a source checkout;
- version-compatibility branches for older vLLM (0.22 / 0.23 era) and
  older SGLang import paths.

### pre-commit (optional)

`.pre-commit-config.yaml` at the repository root defines local hooks that
run the three checks over `kv_cache_manager/py_connector/` only. The
hooks are `language: system`: when `ruff` / `ty` are not on PATH they
exit 0 silently, and without `pre-commit` itself nothing runs at all.

```bash
pip install pre-commit   # or: uv tool install pre-commit
pre-commit install       # opt in
```
