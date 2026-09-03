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

Log evidence asserted: a getCacheLocation request carrying a non-zero offset
(requires connector DEBUG logging, enabled via log_level).
"""

import logging
import unittest

from e2e_lib import (
    ScenarioEnv, assert_report_ok, compare_captures, make_base_prompts,
    send_completions, shared_token_prefix_len, tokenize, wait_for_captures,
    wait_for_prefix_cached,
)

logger = logging.getLogger("vllm_e2e")


class TestPartialHit(unittest.TestCase):
    def test_partial_hit(self):
        env = ScenarioEnv("partial_hit", enable_prefix_caching=True,
                          log_level="DEBUG")
        try:
            env.start_manager()
            vllm = env.start_vllm(log_suffix="_s12")
            mbs = env.manager_block_size()

            base = make_base_prompts(1, env.hybrid)[0]
            # Three nested prompts: A < A+B < A+B+C.
            prompt_a = base
            prompt_ab = base + " Continuation section B. " + " ".join(
                f"Extra sentence {j} carries value {j * 13 + 7}."
                for j in range(90 if env.hybrid else 30))
            prompt_abc = prompt_ab + " Final question: what is 2+2?"

            toks_a = tokenize(vllm.base_url(), prompt_a)
            toks_ab = tokenize(vllm.base_url(), prompt_ab)
            toks_abc = tokenize(vllm.base_url(), prompt_abc)
            blocks_a = len(toks_a) // mbs
            blocks_ab = len(toks_ab) // mbs
            shared_abc = shared_token_prefix_len(toks_ab, toks_abc) // mbs
            self.assertGreaterEqual(blocks_a, 1)
            self.assertGreater(blocks_ab, blocks_a,
                               "B must add at least one manager block")
            logger.info("blocks: A=%d AB=%d shared(AB,ABC)=%d",
                        blocks_a, blocks_ab, shared_abc)

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
                              expected=shared_abc, timeout=180)

            report = compare_captures(env.capture_dir, tp_size=1)
            assert_report_ok(report, min_matched=shared_abc)
        finally:
            env.stop()


if __name__ == "__main__":
    unittest.main()
