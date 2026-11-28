"""test_save_after_prefill: the KV the connector saves must be the KV the forward pass wrote.

The save path gathers a request's blocks out of HBM on the connector's save
stream. With vLLM's async scheduling (on by default) the engine runs one step
ahead of the GPU, so the first save session after a long prefill can reach the
worker while that prefill is still executing. Unless the gather is ordered after
the forward pass, it copies slots the forward has not written yet, and every
later load of those blocks returns that stale KV without any error.

The default reference capture of VerifyingConnector synchronizes the device in
wait_for_save, which hides this: it changes the timing and reads the cache at the
same moment as the save. This scenario sets KVCM_E2E_DEFER_REF=1 so the reference
is captured after each request has finished (the KV the local prefix cache would
serve), and runs with CUDA graphs (--no-enforce-eager), the serving default.

Phase 1 sends long prompts one at a time; vLLM is restarted so phase 2 loads from
KVCM; phase 2 sends prefix+suffix prompts; the loaded KV must equal the deferred
references bit for bit.

Runs against both full-attention and hybrid models via $KVCM_E2E_MODEL.
"""

import json
import logging
import os
import unittest

from e2e_lib import (
    ScenarioEnv, assert_report_ok, compare_captures, send_completions,
    shared_token_prefix_len, tokenize, wait_for_captures, wait_for_prefix_cached,
)

logger = logging.getLogger("vllm_e2e")

NUM_PROMPTS = 6
# Long prompts (~2300 tokens) make the prefill long enough for the first save
# session to arrive while it is still running.
NUM_SENTENCES = 160


def long_prompts(n: int) -> list[str]:
    # Same scheme as make_base_prompts: unique counters give every manager block
    # unique token content, so captures cannot collide across prompts.
    return [
        f"Prompt number {i}. " + " ".join(
            f"Sentence {j} of prompt {i} has value {j * 7 + i * 131}."
            for j in range(NUM_SENTENCES)
        )
        for i in range(n)
    ]


class TestSaveAfterPrefill(unittest.TestCase):
    def test_save_after_prefill(self):
        os.environ["KVCM_E2E_DEFER_REF"] = "1"
        vllm_args = json.loads(os.environ.get("KVCM_E2E_VLLM_ARGS", "{}"))
        vllm_args.setdefault("no-enforce-eager", None)
        os.environ["KVCM_E2E_VLLM_ARGS"] = json.dumps(vllm_args)

        env = ScenarioEnv("save_after_prefill")
        try:
            env.start_manager()
            vllm = env.start_vllm(log_suffix="_p1")

            base = long_prompts(NUM_PROMPTS)
            phase2 = [p + f" Now answer question {i}: what is 2+2?" for i, p in enumerate(base)]
            mbs = env.manager_block_size()
            base_tokens = [tokenize(vllm.base_url(), p) for p in base]
            phase2_tokens = [tokenize(vllm.base_url(), p) for p in phase2]
            expected_save = [len(t) // mbs for t in base_tokens]
            expected_load = [
                min(shared_token_prefix_len(b, p2) // mbs, s)
                for b, p2, s in zip(base_tokens, phase2_tokens, expected_save)
            ]
            self.assertTrue(all(n >= 1 for n in expected_load),
                            f"prompts too short (mbs={mbs}): {expected_load}")
            logger.info("mbs=%d prompt tokens=%s save blocks=%s load blocks=%s", mbs,
                        [len(t) for t in base_tokens], expected_save, expected_load)

            # Phase 1: one request at a time, so each prefill runs alone and its
            # first save session is handed over while the engine is otherwise idle.
            for p in base:
                send_completions(vllm.base_url(), [p])
            wait_for_captures(env.capture_dir, "ref", expected=sum(expected_save), timeout=180)
            for toks, blocks in zip(base_tokens, expected_save):
                self.assertTrue(
                    wait_for_prefix_cached(env.manager.manager_uri(), env.instance_id,
                                           toks, min_blocks=blocks),
                    "save was not committed to the manager in time")

            # Drop the local prefix cache so phase 2 loads from KVCM.
            vllm = env.restart_vllm(log_suffix="_p2")
            for p in phase2:
                send_completions(vllm.base_url(), [p])
            wait_for_captures(env.capture_dir, "loaded", expected=sum(expected_load), timeout=180)

            report = compare_captures(env.capture_dir)
            assert_report_ok(report, min_matched=sum(expected_load), tp_size=1)
            logger.info("save_after_prefill PASSED: %s", json.dumps(
                {k: v for k, v in report.items() if k not in ("failures", "matched_keys")},
                default=str))
        finally:
            env.stop()


if __name__ == "__main__":
    unittest.main()
