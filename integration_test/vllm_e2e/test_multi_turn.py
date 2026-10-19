"""test_multi_turn: decode-time incremental save feeds the next turn's hit.

Turn 1 sends a prompt and generates enough output tokens (max_tokens crossing
at least one manager block boundary) that the save threshold in
``build_connector_meta`` fires again during decode: blocks composed of
generated tokens are saved incrementally. Turn 2 sends prompt + turn-1 output
as its prompt (a real multi-turn conversation) and must externally match
*more* blocks than the turn-1 prompt alone covers -- proving decode-produced
blocks were saved -- and their loaded KV must verify against the references
captured during decode.

Full-attention (mbs=16): three decode blocks, same server both turns (prefix
caching off, the external hit is directly observable).
Hybrid (mbs=528): one decode block (528+ generated tokens); the server is
restarted before turn 2 because prefix caching must stay on for hybrid models
and would otherwise mask the external hit with a local one.
"""

import logging
import unittest

from e2e_lib import (
    ScenarioEnv, assert_loaded_blocks, assert_report_ok, compare_captures,
    full_block_hashes, send_completions, shared_token_prefix_len, tokenize,
    wait_for_captures, wait_for_prefix_cached,
)

logger = logging.getLogger("vllm_e2e")


#: Hoisted so the prompt-fit gate can size the turn-2 request before turn 2
#: exists (vLLM budgets prompt + output against one context window).
TURN2_SUFFIX_TEXT = " Now summarize the story in one word."


class TestMultiTurn(unittest.TestCase):
    def test_multi_turn(self):
        env = ScenarioEnv("multi_turn")
        try:
            env.start_manager()
            vllm = env.start_vllm(log_suffix="_t1" if env.multigroup else "")
            mbs = env.manager_block_size()

            turn1_prompt = ("A short story request. Please write a long, "
                            "detailed story about a robot that learns to paint.")
            prompt_tokens = tokenize(vllm.base_url(), turn1_prompt)

            prompt_blocks = len(prompt_tokens) // mbs

            # ---- Turn 1: generate output crossing >= 1 manager block
            # boundary (3 blocks for full-attn's mbs=16; 1 block for hybrid's
            # mbs=528 to keep decode time bounded).
            max_tokens = (mbs + 32) if env.hybrid else (mbs * 3 + 5)
            # vLLM budgets prompt plus requested output against the same
            # context window, so the fit gate has to see both -- and the
            # request that has to fit is turn 2's, i.e. prompt + this output +
            # the turn-2 suffix + its own (small) completion.
            env.guard_prompt_fits(
                prompt_tokens, "multi_turn",
                max_tokens=max_tokens
                + len(tokenize(vllm.base_url(), TURN2_SUFFIX_TEXT)) + 4)
            resp = send_completions(
                vllm.base_url(), [turn1_prompt], max_tokens=max_tokens,
                ignore_eos=True, return_token_ids=True)[0]
            choice = resp["choices"][0]
            output_ids = choice["token_ids"]
            self.assertEqual(len(output_ids), max_tokens)
            turn1_ids = choice["prompt_token_ids"] + output_ids
            # The connector tracks tokens when they are *scheduled as input*;
            # the very last sampled token never re-enters a step, so at most
            # (len - 1) // mbs blocks can have been committed.
            turn1_blocks = (len(turn1_ids) - 1) // mbs
            self.assertGreater(
                turn1_blocks, prompt_blocks,
                "turn 1 output did not cross a manager block boundary")
            logger.info("turn1: %d prompt + %d output tokens = %d blocks "
                        "(prompt alone: %d)", len(choice["prompt_token_ids"]),
                        len(output_ids), turn1_blocks, prompt_blocks)

            # Decode-produced blocks must be committed: the manager holds the
            # full prompt+output prefix, more blocks than the prompt covers.
            self.assertTrue(wait_for_prefix_cached(
                env.manager.manager_uri(), env.instance_id, turn1_ids,
                min_blocks=turn1_blocks))
            wait_for_captures(env.capture_dir, "ref", expected=turn1_blocks,
                              timeout=180)

            if env.multigroup:
                # Prefix caching is on for multi-group models; restart so turn
                # 2's hit comes from KVCM, not the local prefix cache.
                vllm = env.restart_vllm(log_suffix="_t2")

            # ---- Turn 2: conversation continues; prompt embeds turn 1's
            # prompt + output as token ids (immune to detokenization drift).
            turn2_suffix = tokenize(vllm.base_url(), TURN2_SUFFIX_TEXT)
            turn2_ids = turn1_ids + turn2_suffix
            shared_blocks = min(
                shared_token_prefix_len(turn1_ids, turn2_ids) // mbs,
                turn1_blocks)
            self.assertGreater(shared_blocks, prompt_blocks,
                               "turn 2 shares no decode-produced block")
            resp2 = send_completions(vllm.base_url(), [turn2_ids])[0]
            self.assertTrue(resp2["choices"][0]["text"])

            # The external hit must cover decode-produced blocks.
            matched = [int(g[0]) for g in
                       env.scan_connector_logs(r"matched (\d+) external tokens")]
            best = max(matched, default=0)
            self.assertGreater(
                best, prompt_blocks * mbs,
                f"external hit ({best} tokens) does not exceed the prompt-only "
                f"coverage ({prompt_blocks * mbs} tokens): decode-time saves "
                f"were not used")

            # And the loaded decode-block KV must verify -- the loaded region
            # is what vLLM gave slots for: the tail window of every group, plus
            # the whole match for a full-attention group.
            load_blocks = env.window_geometry().loaded_blocks(shared_blocks * mbs)
            hashes = full_block_hashes(turn2_ids, mbs)
            wait_for_captures(env.capture_dir, "loaded",
                              expected=env.tp_size * load_blocks, timeout=180)
            assert_loaded_blocks(
                env.capture_dir,
                {(f"tp{r}", h) for r in range(env.tp_size)
                 for h in hashes[shared_blocks - load_blocks:shared_blocks]},
                env.tp_size)
            report = compare_captures(env.capture_dir)
            assert_report_ok(report, min_matched=env.tp_size * load_blocks)
            env.assert_pinned_budget()
        finally:
            env.stop()


if __name__ == "__main__":
    unittest.main()
