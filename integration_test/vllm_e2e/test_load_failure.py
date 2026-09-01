"""test_load_failure: storage loss between save and load must degrade, not kill.

Phase 1 saves a long prompt; the test then deletes the storage files of the
*tail half* of the manager blocks (resolved through the manager's ordered
getCacheLocation response, one file per block via key_count_per_file=1).
Phase 2 reloads the same prefix with ``block_per_load_task=1`` so every block
fails or succeeds independently.

Full-attention models (single group, report_failures=True): the connector
reports the failed blocks' vLLM block ids; with
``kv_load_failure_policy="recompute"`` (vLLM 0.26.0 defaults to "fail", which
turns any load failure into a 500) vLLM truncates the computed-token count at
the first invalid block and recomputes from there
(vllm/v1/core/sched/scheduler.py::_handle_invalid_blocks /
_update_requests_with_invalid_blocks). Asserts: the request returns a normal
completion, the failures were logged and reported, every *surviving* head
block's loaded KV is bit-exact, and any mismatching capture belongs to a
deleted (recomputed) block. Recomputed blocks are not held to bit-exactness:
they contain freshly recomputed KV whose numerics depend on prefill chunking,
which is vLLM's business, not the connector's.

Hybrid models (multiple groups, report_failures=False): vLLM's invalid-block
recovery only supports single-group block tables, so the connector only logs
the failure. Asserts: the request still returns (no hang, no crash) and the
failure was logged. KV content is NOT verified: with the failure swallowed
the affected blocks keep garbage by design.

This scenario also regression-tests the fail-reschedule loop fix: a request
whose external load failed must not re-match external blocks on requeue
(v1_connector.get_num_new_matched_tokens retry guard), otherwise the engine
loops load-fail-reschedule forever and the request hangs.
"""

import logging
import os
import unittest
from urllib.parse import urlparse

import requests

from e2e_lib import (
    ScenarioEnv, compare_captures, full_block_hashes, make_base_prompts,
    send_completions, tokenize, wait_for_captures, wait_for_prefix_cached,
)

logger = logging.getLogger("vllm_e2e")


def get_block_files(manager_uri: str, instance_id: str, token_ids: list[int],
                    spec_name: str | None = None) -> list[str]:
    """Per-manager-block storage file paths, in block order, from the manager's
    getCacheLocation response.

    Defaults to the spec every block is guaranteed to have: hybrid models
    publish per-block spec coverage, so a *state* group's spec is absent from
    the blocks where vLLM materialized no recurrent state. The attention spec
    is the one present on every published block -- and it is also the one whose
    loss makes a load fail, which is what this scenario needs.
    """
    r = requests.post(f"{manager_uri}/api/getCacheLocation", json={
        "trace_id": "e2e_block_files",
        "token_ids": token_ids,
        "instance_id": instance_id,
        "query_type": "QT_PREFIX_MATCH",
        "block_mask": {"offset": 0},
    }, timeout=10)
    r.raise_for_status()
    locations = r.json().get("locations", [])
    if spec_name is None:
        # The spec present in *every* location is the attention one; state specs
        # are sparse. Intersect to find it without knowing the group layout.
        common = None
        for location in locations:
            names = {s["name"] for s in location.get("location_specs", [])}
            common = names if common is None else (common & names)
        assert common, f"no spec is common to all {len(locations)} locations"
        spec_name = sorted(common)[0]
        logger.info("using spec %s (present on all %d blocks)", spec_name,
                    len(locations))
    files = []
    for location in locations:
        for spec in location.get("location_specs", []):
            if spec["name"] == spec_name:
                # uri: file://<path>?size=...
                files.append(urlparse(spec["uri"]).path)
    return files


class TestLoadFailure(unittest.TestCase):
    def test_load_failure(self):
        env = ScenarioEnv(
            "load_failure",
            extra_config_overrides={"block_per_load_task": 1},
            key_count_per_file=1,  # one file per block -> per-block failures
            kv_load_failure_policy="recompute",
        )
        try:
            env.start_manager()
            vllm = env.start_vllm(log_suffix="_p1" if env.hybrid else "")
            mbs = env.manager_block_size()

            prompt = make_base_prompts(1, env.hybrid)[0]
            suffix = " Now answer: what is 2+2?"
            toks = tokenize(vllm.base_url(), prompt)
            save_blocks = len(toks) // mbs
            self.assertGreaterEqual(save_blocks, 2)

            # ---- Phase 1: save everything.
            send_completions(vllm.base_url(), [prompt])
            wait_for_captures(env.capture_dir, "ref", expected=save_blocks,
                              timeout=180)
            self.assertTrue(wait_for_prefix_cached(
                env.manager.manager_uri(), env.instance_id, toks,
                min_blocks=save_blocks))

            # ---- Sabotage: delete the tail half of the blocks' files. The
            # head blocks stay loadable, so vLLM truncates at the first deleted
            # block and the surviving loads remain verifiable.
            files = get_block_files(env.manager.manager_uri(), env.instance_id,
                                    toks)
            self.assertEqual(len(files), save_blocks)
            keep = save_blocks // 2
            for path in files[keep:]:
                os.remove(path)
            logger.info("deleted %d/%d block files (kept blocks 0..%d)",
                        save_blocks - keep, save_blocks, keep - 1)

            if env.hybrid:
                vllm = env.restart_vllm(log_suffix="_p2")

            # ---- Phase 2: load with holes. The request must return normally.
            resp = send_completions(vllm.base_url(), [prompt + suffix])[0]
            self.assertTrue(resp["choices"][0]["text"])

            # The engine must survive and keep serving.
            resp2 = send_completions(vllm.base_url(), ["engine alive?"])[0]
            self.assertTrue(resp2["choices"][0]["text"])

            failed_tasks = env.scan_connector_logs(r"load task failed")
            self.assertTrue(failed_tasks, "no load failure was logged; the "
                            "sabotage did not break any loaded block")

            if env.hybrid:
                # report_failures=False path: swallowed but logged.
                swallowed = env.scan_connector_logs(
                    r"load failed for \d+/\d+ blocks .*hybrid")
                self.assertTrue(swallowed,
                                "hybrid load failure was not logged")
                return

            # Full-attention: vLLM was told about the invalid blocks...
            reported = env.scan_connector_logs(r"block_ids_with_load_errors")
            self.assertTrue(reported, "failed loads were not reported to vLLM")

            # ...and every surviving head block's loaded KV is bit-exact,
            # while any mismatch belongs to a deleted (recomputed) block.
            wait_for_captures(env.capture_dir, "loaded", expected=keep,
                              timeout=180)
            report = compare_captures(env.capture_dir, tp_size=1)
            hashes = full_block_hashes(toks, mbs)
            kept_keys = {("tp0", h) for h in hashes[:keep]}
            deleted_keys = {("tp0", h) for h in hashes[keep:]}
            failed_keys = {f["key"] for f in report["failures"]}
            self.assertFalse(
                failed_keys & kept_keys,
                f"surviving loaded blocks mismatched: {failed_keys & kept_keys}")
            self.assertTrue(
                failed_keys <= deleted_keys,
                f"mismatches outside the deleted blocks: "
                f"{failed_keys - deleted_keys}")
            matched_kept = kept_keys & set(report["matched_keys"])
            self.assertEqual(
                len(matched_kept), keep,
                f"only {len(matched_kept)}/{keep} surviving blocks verified")
        finally:
            env.stop()


if __name__ == "__main__":
    unittest.main()
