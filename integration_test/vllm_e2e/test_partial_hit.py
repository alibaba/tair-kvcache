"""test_partial_hit: incremental query + incremental save on a non-zero prefix.

The standard scenarios always start requests from zero computed blocks, so the
``computed_blocks > 0`` branch of ``get_num_new_matched_tokens`` (manager query
with a non-zero ``block_mask.offset``) and the ``block_mask.offset`` branch of
the save path (manager skipping already-stored blocks) are never exercised.
This scenario forces both, with prefix caching enabled for all model types:

* Stage 1: save prefix A (fresh prefill).
* Stage 2 (same server): send A+B. vLLM locally hits A, so the connector
  queries the manager with offset = locally computed blocks (> 0), then
  extends the save; the manager's start_write_cache response skips the
  already-stored A blocks via a non-zero block_mask offset.
* Stage 3 (restarted server, local cache cleared): send A+B+C. The connector
  externally matches A+B -- proving stage 2's incremental save committed --
  loads it, and the loaded KV is verified against the reference captures.

Windowed models (SWA): vLLM hands out slots per group -- the tail window of
the match for every group, plus the whole match for a full-attention group --
so stage 3 asserts the loaded blocks are exactly that region (see
``test_mid_prefix_hit`` for the case where the hit ends in the middle of a
saved request).

Log evidence asserted: a getCacheLocation request carrying a non-zero offset
(requires connector DEBUG logging, enabled via log_level).
"""

import logging
import unittest

from e2e_lib import (
    ScenarioEnv, assert_loaded_blocks, assert_report_ok, compare_captures,
    full_block_hashes, make_base_prompts, send_completions,
    shared_token_prefix_len, tokenize, wait_for_captures,
    wait_for_prefix_cached,
)

logger = logging.getLogger("vllm_e2e")


def skip_if_rswa(env) -> None:
    """R-SWA models cannot run this scenario (loudly, not silently).

    ``UnlimitedOCRForCausalLMConfig`` forbids prefix caching outright -- R-SWA
    decode KV is not a pure causal function of the prefix -- so the *local* hit
    on A that this scenario's manager-query offsets depend on can never happen,
    no matter what the harness asks vLLM for. Its window is a decode-time one
    as well, which is not the prefill geometry the mid-prefix variant builds
    its case from.
    """
    if env.window_kind == "rswa_gap":
        raise unittest.SkipTest(
            "R-SWA models disable vLLM prefix caching (the architecture's "
            "config class forces it off), so this scenario's premise -- a "
            "local hit on A with a nonzero manager-query offset -- cannot "
            "hold; the R-SWA coverage lives in the standard runner and the "
            "load-failure scenario")


class TestPartialHit(unittest.TestCase):
    def test_partial_hit(self):
        env = ScenarioEnv("partial_hit", enable_prefix_caching=True,
                          log_level="DEBUG")
        try:
            skip_if_rswa(env)
            env.start_manager()
            vllm = env.start_vllm(log_suffix="_s12")
            mbs = env.manager_block_size()

            base = make_base_prompts(1, env.hybrid, env.window)[0]
            # Three nested prompts: A < A+B < A+B+C.
            prompt_a = base
            prompt_ab = base + " Continuation section B. " + " ".join(
                f"Extra sentence {j} carries value {j * 13 + 7}."
                for j in range(90 if env.hybrid else 30))
            prompt_abc = prompt_ab + " Final question: what is 2+2?"

            toks_a = tokenize(vllm.base_url(), prompt_a)
            toks_ab = tokenize(vllm.base_url(), prompt_ab)
            toks_abc = tokenize(vllm.base_url(), prompt_abc)
            env.guard_prompt_fits(toks_abc, "partial_hit", max_tokens=4)
            blocks_a = len(toks_a) // mbs
            blocks_ab = len(toks_ab) // mbs
            shared_abc = min(shared_token_prefix_len(toks_ab, toks_abc) // mbs,
                             blocks_ab)
            load_blocks = env.window_geometry().loaded_blocks(shared_abc * mbs)
            self.assertGreaterEqual(blocks_a, 1)
            self.assertGreater(blocks_ab, blocks_a,
                               "B must add at least one manager block")
            logger.info("blocks: A=%d AB=%d shared(AB,ABC)=%d load=%d window=%s",
                        blocks_a, blocks_ab, shared_abc, load_blocks, env.window)

            # ---- Stage 1: save A.
            send_completions(vllm.base_url(), [prompt_a])
            wait_for_captures(env.capture_dir, "ref", expected=blocks_a, timeout=180)
            self.assertTrue(wait_for_prefix_cached(
                env.manager.manager_uri(), env.instance_id, toks_a,
                min_blocks=blocks_a))

            # ---- Stage 2: A hits the local prefix cache -> incremental
            # external query (non-zero offset) + incremental save of B.
            send_completions(vllm.base_url(), [prompt_ab])
            self.assertTrue(wait_for_prefix_cached(
                env.manager.manager_uri(), env.instance_id, toks_ab,
                min_blocks=blocks_ab))
            wait_for_captures(env.capture_dir, "ref", expected=blocks_ab, timeout=180)

            offsets = [int(g[0]) for g in env.scan_connector_logs(
                r"get_kvcache_location request:.*'offset': (\d+)")]
            self.assertTrue(any(o > 0 for o in offsets),
                            f"no incremental query with non-zero offset: {offsets}")

            # ---- Stage 3: restart (clear local cache) and load A+B.
            vllm = env.restart_vllm(log_suffix="_s3")
            send_completions(vllm.base_url(), [prompt_abc])
            wait_for_captures(env.capture_dir, "loaded",
                              expected=env.tp_size * load_blocks, timeout=180)

            hashes = full_block_hashes(toks_abc, mbs)
            assert_loaded_blocks(
                env.capture_dir,
                {(f"tp{r}", h) for r in range(env.tp_size)
                 for h in hashes[shared_abc - load_blocks:shared_abc]},
                env.tp_size)

            report = compare_captures(env.capture_dir)
            assert_report_ok(report, min_matched=env.tp_size * load_blocks)
            env.assert_pinned_budget()
        finally:
            env.stop()

    def test_mid_prefix_hit(self):
        """The external hit ends in the middle of a saved request.

        Every other scenario's match ends at (or near) the end of what was
        saved, so a connector that served "the tail of the stored prefix"
        would look correct. Here B is a strict token prefix of A plus a fresh
        suffix, so the match ends several windows before A's own tail: the
        blocks vLLM hands out slots for are A's *middle* blocks. Asserting the
        loaded blocks are exactly the window ending at B's match end -- and
        that no block of A's tail was loaded -- pins the load region to the
        hit, not to the stored request.

        Windowed models only: with a window the matched region is strictly
        larger than the loaded region, which is what this scenario contrasts.
        """
        env = ScenarioEnv("mid_prefix_hit", enable_prefix_caching=True,
                          log_level="DEBUG")
        try:
            skip_if_rswa(env)
            env.start_manager()
            if env.window is None:
                self.skipTest("mid-prefix window assertions need a windowed model")
            vllm = env.start_vllm(log_suffix="_p1")
            # The manager block size comes from the instance the connector
            # registers at engine startup, so the gate runs after start_vllm
            # (a skip below tears the server down cleanly through the finally).
            mbs = env.manager_block_size()
            geom = env.window_geometry()
            window_blocks = geom.window_blocks
            # B's match must end inside A, one window before A's own tail window,
            # so the prompt has to hold three windows. A window larger than a
            # third of the model length cannot satisfy that with a request the
            # server accepts: say so instead of failing an assertion that is
            # impossible to meet (the prompt would have to exceed
            # --max-model-len, and the server would reject it).
            needed_tokens = 3 * window_blocks * mbs
            if needed_tokens > env.max_model_len():
                self.skipTest(
                    f"a {env.window}-token window needs {needed_tokens} tokens "
                    f"for three windows, over the {env.max_model_len()} this "
                    f"server accepts; the mid-prefix case cannot be built")
            # The loaded region is the union of every group's slots: the window
            # for the sliding groups, the whole match for the full-attention
            # one (so it is a superset of the window).

            toks_a = tokenize(vllm.base_url(),
                              make_base_prompts(1, env.hybrid, env.window)[0])
            env.guard_prompt_fits(toks_a, "mid_prefix_hit", max_tokens=4)
            blocks_a = len(toks_a) // mbs
            toks_a = toks_a[:blocks_a * mbs]
            # B's match must end inside A but far enough in that the window it
            # is served from is a real one (not the prompt head), and far
            # enough from A's end that A's own tail window is a different
            # region: k < mid and mid + k <= blocks_a.
            mid = min(2 * window_blocks, blocks_a - window_blocks)
            self.assertGreaterEqual(mid, 2 * window_blocks,
                                    f"A too short ({blocks_a} blocks) for a "
                                    f"mid-prefix hit with a {window_blocks}-block "
                                    f"window")
            suffix = tokenize(vllm.base_url(), " Final question: what is 2+2?")
            toks_b = toks_a[:mid * mbs] + suffix
            logger.info("A=%d blocks, B matches %d blocks (window=%d blocks)",
                        blocks_a, mid, window_blocks)

            # ---- Phase 1: save A (token ids: exact cache keys).
            send_completions(vllm.base_url(), [toks_a], max_tokens=4)
            wait_for_captures(env.capture_dir, "ref", expected=blocks_a, timeout=180)
            self.assertTrue(wait_for_prefix_cached(
                env.manager.manager_uri(), env.instance_id, toks_a,
                min_blocks=blocks_a), "A was not committed to the manager")

            # ---- Phase 2: restart, then run B against A's middle.
            vllm = env.restart_vllm(log_suffix="_p2")
            send_completions(vllm.base_url(), [toks_b], max_tokens=4)

            matched = [int(g[0]) for g in
                       env.scan_connector_logs(r"matched (\d+) external tokens")]
            self.assertEqual(max(matched, default=0), mid * mbs,
                             f"the external match did not end at B's prefix: "
                             f"{matched} (want {mid * mbs}); A holds "
                             f"{blocks_a * mbs} tokens")
            self.assertFalse(env.scan_connector_logs(r"load task failed"),
                             "a load failed: a published block was not readable")

            hashes = full_block_hashes(toks_b, mbs)
            # The loaded region is the union of every group's slots: the window
            # of the hit for the sliding groups (the region a "just serve the
            # stored tail" implementation would *not* have loaded), plus the
            # whole match for the full-attention group.
            # Built here (not reused from the gate above): the registration
            # that supplies the extra retained tokens only exists once the
            # engine has started.
            covered = env.window_geometry().loaded_blocks(mid * mbs)
            expected_keys = {(f"tp{r}", h) for r in range(env.tp_size)
                             for h in hashes[mid - covered:mid]}
            # ... and that whole region is disjoint from A's own tail window
            # (the region a "just serve the stored tail" implementation would
            # have loaded), so the scenario can tell "region of the hit" from
            # "tail of the stored request" apart.
            self.assertLessEqual(mid, blocks_a - window_blocks,
                                 "the loaded region overlaps A's tail window; the "
                                 "scenario cannot tell 'region of the hit' from "
                                 "'tail of the stored request' apart")
            wait_for_captures(env.capture_dir, "loaded",
                              expected=env.tp_size * covered, timeout=180)
            assert_loaded_blocks(env.capture_dir, expected_keys, env.tp_size)

            report = compare_captures(env.capture_dir)
            assert_report_ok(report, min_matched=env.tp_size * covered)
            env.assert_pinned_budget()
        finally:
            env.stop()


if __name__ == "__main__":
    unittest.main()
