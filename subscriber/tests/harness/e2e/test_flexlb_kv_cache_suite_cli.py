from __future__ import annotations

import json
from pathlib import Path

from harness.e2e.flexlb_kv_cache_suite import main


def test_plan_command_writes_engine_specific_manifest(tmp_path: Path) -> None:
    output_path = tmp_path / "manifest.json"

    exit_code = main(
        [
            "plan",
            "--engine-kind",
            "vllm",
            "--cache-unit-tokens",
            "64",
            "--max-context-tokens",
            "64000",
            "--output",
            str(output_path),
        ]
    )

    manifest = json.loads(output_path.read_text())
    assert exit_code == 0
    assert manifest["engine_kind"] == "vllm"
    assert manifest["cache_unit_tokens"] == 64
