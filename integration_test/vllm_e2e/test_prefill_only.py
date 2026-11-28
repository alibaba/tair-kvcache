"""test_prefill_only: a request that ends in its prefill step must save every block.

A request with max_tokens=1 (what a PD-disaggregated prefill node serves)
finishes in the same engine step that computes its last prompt chunk. The
connector acquires the write locations for that chunk at the start of the step
and hands the save session to the worker in the *next* step's metadata. If the
engine is otherwise idle, that next step has no scheduled tokens: vLLM then
runs kv_connector_no_forward, which calls get_finished but not wait_for_save.
The save must still be submitted there; otherwise the last session is never
committed and the request is reported done without its last blocks.

Phase 1 (control): a max_tokens=4 request -- its last session is handed over
on a decode step -- must commit all its blocks.
Phase 2: a different prompt with max_tokens=1 and nothing else running must
commit all its blocks too.

Runs against both full-attention and hybrid models via $KVCM_E2E_MODEL.
"""

import logging
import unittest

from e2e_lib import (
    ScenarioEnv, make_base_prompts, send_completions, tokenize, wait_for_prefix_cached,
)

logger = logging.getLogger("vllm_e2e")


class TestPrefillOnly(unittest.TestCase):
    def test_prefill_only(self):
        env = ScenarioEnv("prefill_only")
        try:
            env.start_manager()
            vllm = env.start_vllm()
            mbs = env.manager_block_size()

            prompts = []
            for text in make_base_prompts(2, env.hybrid):
                toks = tokenize(vllm.base_url(), text)
                num_blocks = len(toks) // mbs
                self.assertGreaterEqual(
                    num_blocks, 2, f"prompt too short: {len(toks)} tokens, mbs={mbs}")
                prompts.append(toks[:num_blocks * mbs])

            for phase, (prompt_ids, max_tokens) in enumerate(
                    zip(prompts, (4, 1)), start=1):
                num_blocks = len(prompt_ids) // mbs
                resp = send_completions(vllm.base_url(), [prompt_ids],
                                        max_tokens=max_tokens, ignore_eos=True)[0]
                self.assertEqual(len(resp["choices"]), 1)
                logger.info("phase %d: %d tokens = %d x %d, max_tokens=%d",
                            phase, len(prompt_ids), num_blocks, mbs, max_tokens)
                # Nothing else is sent until the poll ends: the engine stays
                # idle after the request finishes.
                self.assertTrue(
                    wait_for_prefix_cached(env.manager.manager_uri(), env.instance_id,
                                           prompt_ids, min_blocks=num_blocks,
                                           timeout=60.0),
                    f"phase {phase} (max_tokens={max_tokens}): not all "
                    f"{num_blocks} blocks were committed")
        finally:
            env.stop()


if __name__ == "__main__":
    unittest.main()
