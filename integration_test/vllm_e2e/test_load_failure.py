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

Windowed multi-group models split by vLLM era: on vLLM >= 0.30 the connector
reports the failure at request granularity (KVConnectorTransferResults) and
vLLM recomputes the request -- asserted like the single-table case above. On
the pre-0.30 eras there is no request-level API, so the connector fails
closed (get_finished raises rather than resume the request on partially
loaded KV): the completion errors out, and that intended refusal -- plus the
request-granularity detection log that precedes it -- is what the scenario
asserts there.

This scenario also regression-tests the fail-reschedule loop fix: a request
whose external load failed must not re-match external blocks on requeue
(v1_connector.get_num_new_matched_tokens retry guard), otherwise the engine
loops load-fail-reschedule forever and the request hangs.
"""

import logging
import os
import sys
import unittest
from urllib.parse import urlparse

import requests

from e2e_lib import (
    ScenarioEnv, assert_no_capture_error, compare_captures, full_block_hashes,
    make_base_prompts, send_completions, supports_request_level_load_failures,
    tokenize, wait_for_captures, wait_for_prefix_cached,
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
            vllm = env.start_vllm(log_suffix="_p1" if env.multigroup else "")
            mbs = env.manager_block_size()

            prompt = make_base_prompts(1, env.hybrid, env.window)[0]
            suffix = " Now answer: what is 2+2?"
            toks = tokenize(vllm.base_url(), prompt)
            # Same gate as the other scaled-prompt scenarios: a window too wide
            # for --max-model-len must skip loudly instead of letting the HTTP
            # request fail before the storage-loss path is exercised.
            env.guard_prompt_fits(toks, "load_failure",
                                  max_tokens=len(tokenize(vllm.base_url(), suffix)) + 4)
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

            if env.multigroup:
                # Prefix caching stays on for multi-group models, so the restart
                # is what forces the load through KVCM.
                vllm = env.restart_vllm(log_suffix="_p2")

            # A windowed multi-group model on a vLLM without the request-level
            # load-failure API (pre-0.30) never reaches the branches below:
            # the connector deliberately fails closed there (get_finished
            # raises rather than resume the request on partially loaded KV),
            # so the completion errors out and the engine dies with it. Pin
            # that intended refusal here -- the request-level recompute the
            # newer eras perform is asserted in the main flow below.
            fail_closed = (env.connector_block_tables() > 1
                           and env.window_geometry().window is not None
                           and not supports_request_level_load_failures())
            if fail_closed:
                with self.assertRaises(requests.RequestException):
                    send_completions(vllm.base_url(), [prompt + suffix])
                # The connector still detected and logged the failure at
                # request granularity on its way out, and the sabotage path
                # really ran (individual load tasks failed).
                reported = env.scan_connector_logs(
                    r"load failed for \d+/\d+ blocks of req .*windowed "
                    r"multi-group")
                self.assertTrue(
                    reported,
                    "the windowed multi-group load failure was not detected "
                    "at request granularity on the fail-closed era")
                self.assertTrue(env.scan_connector_logs(r"load task failed"),
                                "no load failure was logged; the sabotage did "
                                "not break any loaded block")
                assert_no_capture_error(env.capture_dir)
                return

            # ---- Phase 2: load with holes. The request must return normally.
            resp = send_completions(vllm.base_url(), [prompt + suffix])[0]
            self.assertTrue(resp["choices"][0]["text"])

            # The engine must survive and keep serving.
            resp2 = send_completions(vllm.base_url(), ["engine alive?"])[0]
            self.assertTrue(resp2["choices"][0]["text"])

            failed_tasks = env.scan_connector_logs(r"load task failed")
            self.assertTrue(failed_tasks, "no load failure was logged; the "
                            "sabotage did not break any loaded block")

            if env.connector_block_tables() > 1:
                # Multi-block-table path: the connector cannot hand vLLM a
                # group-indexed invalid-block list. What happens next depends
                # on the layout: a *windowed* multi-group model reports the
                # failure at request granularity (vLLM >= 0.30 recomputes the
                # request instead of decoding on the hole), while a
                # non-windowed hybrid keeps the swallowed-and-logged contract.
                # Keyed on the *registered* block-table count, not on the
                # model family: a single-group windowed model reports like
                # full attention even though is_multigroup()/env.hybrid are
                # true for it. (Pre-0.30 windowed runs were already answered
                # by the fail-closed branch above.)
                if env.window_geometry().window is not None:
                    reported = env.scan_connector_logs(
                        r"load failed for \d+/\d+ blocks of req "
                        r".*reporting the request, not block ids")
                    self.assertTrue(
                        reported,
                        "the windowed multi-group load failure was not "
                        "reported at request granularity")
                else:
                    swallowed = env.scan_connector_logs(
                        r"load failed for \d+/\d+ blocks .*hybrid")
                    self.assertTrue(swallowed,
                                    "multi-block-table load failure was not logged")
                # No capture *count* is expected here (the loads failed on
                # purpose), but the marker still has to be read: it is the only
                # other channel a capture-side assertion or a broken capture
                # uses, and returning without it would accept either silently.
                assert_no_capture_error(env.capture_dir)
                return

            # Single block table: vLLM was told about the invalid blocks...
            reported = env.scan_connector_logs(r"block_ids_with_load_errors")
            self.assertTrue(reported, "failed loads were not reported to vLLM")

            # ...and every surviving head block's loaded KV is bit-exact,
            # while any mismatch belongs to a deleted (recomputed) block.
            # Key sets cover every tp rank: the sabotage deletes one rank's
            # files, but vLLM's invalid-block truncation is rank-global, so
            # on the other ranks the tail blocks are loaded and then
            # recomputed over -- their captures mismatch in low bits (a
            # different attention kernel path), which is exactly the
            # "recomputed" disposition, not a transfer failure.
            # vLLM allocated slots for the *original* external match (the whole
            # prompt), then invalid-block recovery truncated the request to the
            # first deleted block -- it does not go back and load the tail of
            # the shortened prefix. So the region that can have real loaded
            # captures is the originally allocated region intersected with the
            # undeleted head: the whole match for a full-attention group (and
            # for R-SWA), the tail window of the whole match for a purely
            # windowed model -- which for this ~2.5-window prompt lies entirely
            # in the deleted half, leaving nothing to verify beyond the failure
            # report itself.
            geom = env.window_geometry()
            # The originally allocated region starts where the windowed tail
            # of the *whole match* begins (the full group's region starts at
            # block 0), ending at the match's effective end: an EAGLE/MTP
            # drop moves the whole served region a hash block back.
            end = geom.hit_end_blocks(save_blocks * mbs)
            allocated_from = max(0, end - geom.window_tail(save_blocks * mbs))
            load_blocks = max(0, min(end, keep) - allocated_from)
            wait_for_captures(env.capture_dir, "loaded", expected=load_blocks,
                              timeout=180)
            report = compare_captures(env.capture_dir)
            hashes = full_block_hashes(toks, mbs)
            ranks = [f"tp{r}" for r in range(env.tp_size)]
            kept_keys = {(r, h) for r in ranks for h in hashes[:keep]}
            deleted_keys = {(r, h) for r in ranks for h in hashes[keep:]}
            loaded_keys = {(r, h) for r in ranks
                           for h in hashes[keep - load_blocks:keep]}
            failed_keys = {f["key"] for f in report["failures"]}
            self.assertFalse(
                failed_keys & loaded_keys,
                f"surviving loaded blocks mismatched: "
                f"{failed_keys & loaded_keys} (kept head: {keep}, loaded "
                f"region: [{keep - load_blocks}, {keep}))")
            self.assertTrue(
                failed_keys <= deleted_keys,
                f"mismatches outside the deleted blocks: "
                f"{failed_keys - deleted_keys}")
            matched_loaded = loaded_keys & set(report["matched_keys"])
            self.assertEqual(
                len(matched_loaded), load_blocks * env.tp_size,
                f"only {len(matched_loaded)}/{load_blocks * env.tp_size} "
                f"loaded surviving blocks verified (kept head: {keep}, "
                f"window: {geom.window})")
        finally:
            # Stop the processes even if the gate raises: a leaked manager or
            # engine would hold ports and GPU memory and cascade into the next
            # scenario. On the way out of a *failing* run the gate is skipped
            # entirely -- that failure must surface, not a gate that may not
            # even be evaluable (the engine logs exist only if a server got far
            # enough to print its pinned line).
            try:
                if sys.exc_info()[0] is None:
                    env.assert_pinned_budget()
            finally:
                env.stop()


if __name__ == "__main__":
    unittest.main()
