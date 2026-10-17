from __future__ import annotations

from pathlib import Path

from harness.e2e.flexlb_suite.manifest import build_default_manifest
from harness.e2e.flexlb_suite.storage import read_manifest, write_manifest


def test_manifest_json_round_trip_preserves_tuple_contracts(tmp_path: Path) -> None:
    manifest = build_default_manifest(
        engine_kind="sglang",
        cache_unit_tokens=128,
        max_context_tokens=64_000,
    )
    path = tmp_path / "manifest.json"

    write_manifest(manifest, path)

    assert read_manifest(path) == manifest
