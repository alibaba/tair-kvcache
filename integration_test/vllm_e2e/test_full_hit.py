"""test_full_hit: full-prompt external hit must not crash the engine.

Regression test for the synchronous-load full-hit bug: this connector reports
external matches with load_kv_async=False, so vLLM schedules
``num_tokens - num_computed_tokens`` new tokens and asserts that count is > 0
(vllm/v1/core/sched/scheduler.py, waiting-queue loop: ``assert num_new_tokens
> 0``). Without capping, a prompt whose token count is an exact multiple of the
manager block size and whose blocks are all externally cached would make the
count 0 and kill the engine.

Phase 1 saves a prompt of exactly N manager blocks; phase 2 resends the very
same prompt (as explicit token ids, so tokenization cannot shift the length).
Asserts: the engine survives, the completion is well-formed, and the connector
reports 0 < matched < prompt tokens (the cap dropped at least the last block).

Runs against both full-attention and hybrid models via $KVCM_E2E_MODEL.
"""

import logging
import unittest

from e2e_lib import (
    ScenarioEnv, make_base_prompts, send_completions, tokenize,
    wait_for_prefix_cached,
)

logger = logging.getLogger("vllm_e2e")


class TestFullHit(unittest.TestCase):
    def test_full_hit(self):
        env = ScenarioEnv("full_hit")
        try:
            env.start_manager()
            vllm = env.start_vllm(log_suffix="_p1" if env.hybrid else "")

            mbs = env.manager_block_size()
            # Trim a long-enough prompt's token ids to an exact multiple of the
            # manager block size (>= 2 blocks so the cap has room to drop one).
            toks = tokenize(vllm.base_url(), make_base_prompts(1, env.hybrid)[0])
            num_blocks = len(toks) // mbs
            self.assertGreaterEqual(
                num_blocks, 2, f"prompt too short: {len(toks)} tokens, mbs={mbs}")
            prompt_ids = toks[:num_blocks * mbs]
            logger.info("full-hit prompt: %d tokens = %d x %d",
                        len(prompt_ids), num_blocks, mbs)

            # Phase 1: fresh prefill -> all blocks saved.
            resp1 = send_completions(vllm.base_url(), [prompt_ids])[0]
            self.assertTrue(resp1["choices"][0]["text"])
            self.assertTrue(wait_for_prefix_cached(
                env.manager.manager_uri(), env.instance_id, prompt_ids,
                min_blocks=num_blocks))

            if env.hybrid:
                # Clear the local prefix cache so phase 2 goes external.
                vllm = env.restart_vllm(log_suffix="_p2")

            # Phase 2: the exact same prompt -> full external hit. Without the
            # cap this crashes the engine (assert num_new_tokens > 0).
            resp2 = send_completions(vllm.base_url(), [prompt_ids])[0]
            self.assertTrue(resp2["choices"][0]["text"])

            # The engine must still be alive and serving.
            resp3 = send_completions(vllm.base_url(), ["sanity check prompt"])[0]
            self.assertTrue(resp3["choices"][0]["text"])

            # Connector-side evidence: matched > 0 (external hit happened) and
            # matched < prompt tokens (the cap left tokens to recompute).
            matched = [int(g[0]) for g in
                       env.scan_connector_logs(r"matched (\d+) external tokens")]
            self.assertTrue(matched, "no 'matched N external tokens' log found")
            hit = [m for m in matched if m > 0]
            self.assertTrue(hit, f"no positive external match in {matched}")
            self.assertTrue(all(m < len(prompt_ids) for m in hit),
                            f"match not capped below prompt len: {matched}")
        finally:
            env.stop()


if __name__ == "__main__":
    unittest.main()
