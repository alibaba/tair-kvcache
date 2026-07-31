"""test_cross_request_prefix: a shorter request must not resume from a prefix
whose recurrent (mamba) boundary state was never written.

Why this scenario exists
-----------------------
Every other e2e scenario re-sends the *same or a longer* prompt, so the block
that ends an external match is always a block whose mamba state vLLM actually
materialized. Hybrid ``mamba_cache_mode="align"`` materializes a state only at
segment boundaries: the interior manager blocks of a saved request map to
vLLM's null block and hold no state at all.

This scenario is the missing cross-request case:

* **Phase 1** -- request A (long, several manager blocks) is saved. Which of A's
  blocks actually carry state is measured *independently* of the connector's
  publication logic, from the harness's own reference captures (the verifying
  connector captures state layers only for non-null state blocks).
* **Phase 2** -- request B (a token-level *prefix* of A ending inside one of A's
  blocks) runs against a *fresh instance id* on a restarted server, so nothing
  is cached anywhere: its greedy output is the ground truth.
* **Phase 3** -- a restarted server (empty local prefix cache) runs B against
  A's cached blocks, so B's external match ends on an *interior* block of A.

If interior blocks are published as fully cached even though their state was
never written, phase 3 loads a never-written state URI. The hybrid load failure
cannot be reported to vLLM (single-group invalid-block recovery upstream), so B
silently decodes from a garbage recurrent state and its output diverges from
phase 2's ground truth. That divergence is the primary assertion.

Correct behaviour: the manager knows -- through per-key
``location_spec_group_names`` -- exactly which blocks carry state, and the
connector truncates the external match to the last state-complete block. B then
resumes from a real boundary state (or recomputes), producing the ground-truth
output.

Full-attention models have no state groups: they must show *uniform* spec
coverage and *no* truncation, which makes this scenario a regression guard for
them too.
"""

import glob
import logging
import os
import unittest

import requests

from e2e_lib import (
    ScenarioEnv, full_block_hashes, send_completions, tokenize,
    wait_for_captures, wait_for_prefix_cached,
)

logger = logging.getLogger("vllm_e2e")


def per_block_spec_names(manager_uri: str, instance_id: str,
                         token_ids: list[int]) -> list[list[str]]:
    """Per-manager-block location spec names, in block order, as the manager
    reports them to a prefix-match query -- exactly what the connector's load
    path sees in ``get_num_new_matched_tokens``."""
    r = requests.post(f"{manager_uri}/api/getCacheLocation", json={
        "trace_id": "e2e_specs",
        "token_ids": token_ids,
        "instance_id": instance_id,
        "query_type": "QT_PREFIX_MATCH",
        "block_mask": {"offset": 0},
    }, timeout=10)
    r.raise_for_status()
    return [sorted(s["name"] for s in loc.get("location_specs", []))
            for loc in r.json().get("locations", [])]


def captured_state_presence(capture_dir: str, token_ids: list[int],
                            manager_block_size: int) -> list[bool]:
    """Per-manager-block "vLLM materialized a recurrent state here", measured
    from the harness's reference captures.

    ``test_connector.VerifyingConnector`` captures a state group's layers only
    when the manager block maps to a real (non-null) state block, and stores
    them as a ``list[Tensor]`` (attention layers are a single Tensor). So a
    capture containing a list-valued layer is a block whose state exists.

    This is independent of the connector's manager-facing publication logic,
    which is what the scenario verifies.
    """
    import torch

    presence = []
    for idx, token_hash in enumerate(full_block_hashes(token_ids, manager_block_size)):
        matches = glob.glob(os.path.join(capture_dir, f"ref_tp0_{token_hash}.pt"))
        assert matches, f"no reference capture for manager block {idx}"
        rec = torch.load(matches[0], map_location="cpu", weights_only=True)
        presence.append(any(isinstance(v, (list, tuple))
                            for v in rec["kv"].values()))
    return presence


def long_prompt(num_sentences: int) -> str:
    """Deterministic prompt whose every sentence is unique, so no two manager
    blocks share token content (and therefore no two share a cache key)."""
    return "Report A. " + " ".join(
        f"Item {j} of the ledger records the value {j * 17 + 3} and the tag "
        f"{(j * 31) % 97}."
        for j in range(num_sentences))


def completion_token_ids(base_url: str, token_ids: list[int],
                         max_tokens: int) -> list[int]:
    """Greedy continuation of an explicit token-id prompt, as token ids."""
    resp = send_completions(base_url, [token_ids], max_tokens=max_tokens,
                            ignore_eos=True, return_token_ids=True)[0]
    out = list(resp["choices"][0]["token_ids"])
    assert len(out) == max_tokens, f"short completion: {len(out)}"
    return out


class TestCrossRequestPrefix(unittest.TestCase):
    # Long enough that a corrupted recurrent state cannot plausibly reproduce
    # the reference continuation token for token.
    MAX_TOKENS = 32
    # Mirrors the harness's --max-model-len; prompt + output must fit.
    MAX_MODEL_LEN = 4096

    def test_cross_request_prefix(self):
        env = ScenarioEnv("cross_request_prefix", log_level="DEBUG")
        real_instance_id = env.instance_id
        try:
            env.start_manager()

            # ---- Phase 1: save request A, long enough to span several manager
            # blocks so interior (state-less) blocks exist.
            vllm = env.start_vllm(log_suffix="_p1")
            mbs = env.manager_block_size()
            # Hybrid models pin the manager block to the scheduler block (528),
            # so they need a far longer prompt to span several blocks.
            toks_a = tokenize(vllm.base_url(), long_prompt(260 if env.hybrid else 60))
            # Cap the length so prompt + MAX_TOKENS stays inside the model len.
            blocks_a = min(len(toks_a) // mbs,
                           (self.MAX_MODEL_LEN - self.MAX_TOKENS) // mbs)
            self.assertGreaterEqual(
                blocks_a, 4,
                f"only {blocks_a} manager blocks available (mbs={mbs}); this "
                f"scenario needs interior blocks")
            # Send A as explicit token ids so its cache keys are exactly toks_a.
            toks_a = toks_a[:blocks_a * mbs]
            logger.info("A: %d tokens = %d x %d", len(toks_a), blocks_a, mbs)
            completion_token_ids(vllm.base_url(), toks_a, max_tokens=4)
            wait_for_captures(env.capture_dir, "ref", expected=blocks_a, timeout=180)
            self.assertTrue(wait_for_prefix_cached(
                env.manager.manager_uri(), env.instance_id, toks_a,
                min_blocks=blocks_a), "A was not committed to the manager")

            # ---- Ground truth: which of A's blocks really have a state?
            has_state = captured_state_presence(env.capture_dir, toks_a, mbs)
            logger.info("A state materialized per block (from captures): %s",
                        has_state)
            # ---- And what does the manager advertise per block?
            specs = per_block_spec_names(env.manager.manager_uri(),
                                         env.instance_id, toks_a)
            self.assertEqual(len(specs), blocks_a)
            union = sorted(set().union(*[set(s) for s in specs]))
            complete = [s == union for s in specs]
            logger.info("A per-block coverage (union=%s): %s", union,
                        [("full" if c else "missing:%s" % sorted(set(union) - set(s)))
                         for c, s in zip(complete, specs)])

            # ---- Choose B: a strict prefix of A ending mid-block whose last
            # *candidate* block (where the external match would end) carries no
            # state. That is the case the bug corrupts. The null-state layout
            # depends on how vLLM's save batches lined up with segment
            # boundaries, so derive it from the measured ground truth.
            if env.hybrid:
                choices = [n for n in range(2, blocks_a) if not has_state[n - 1]]
                self.assertTrue(
                    choices,
                    f"no state-less interior block in A (state: {has_state}); "
                    f"the scenario cannot exercise truncation")
                candidates = max(choices)
            else:
                candidates = blocks_a - 1
            toks_b = toks_a[:candidates * mbs + mbs // 2]
            logger.info("B: %d tokens, %d candidate blocks (last has_state=%s)",
                        len(toks_b), candidates, has_state[candidates - 1])

            # ---- Phase 2: ground truth for B -- fresh instance id (no KVCM
            # entries) on a restarted server (no local prefix cache).
            env.instance_id = real_instance_id + "-ref"
            vllm = env.restart_vllm(log_suffix="_ref")
            ref_out = completion_token_ids(vllm.base_url(), toks_b, self.MAX_TOKENS)
            logger.info("B reference output: %s", ref_out)

            # ---- Phase 3: B against A's cached blocks.
            env.instance_id = real_instance_id
            vllm = env.restart_vllm(log_suffix="_p3")
            got_out = completion_token_ids(vllm.base_url(), toks_b, self.MAX_TOKENS)
            logger.info("B cached output:    %s", got_out)

            matched = [int(g[0]) for g in
                       env.scan_connector_logs(r"matched (\d+) external tokens")]
            logger.info("connector external match(es): %s", matched)

            # ---- Primary assertion: reusing A's prefix must not change B's
            # output. A mismatch means B resumed from state it never had.
            self.assertEqual(
                got_out, ref_out,
                "B's output changed when it reused A's cached prefix: the "
                "external match ended on a block whose recurrent state was "
                f"never written (external matches: {matched})")

            # ---- A load failure must never be silently swallowed.
            self.assertFalse(
                env.scan_connector_logs(r"load task failed"),
                "a load failed in phase 3: a published block was not readable")
            self.assertFalse(
                env.scan_connector_logs(r"load failed for \d+/\d+ blocks"),
                "a hybrid load failure was swallowed in phase 3")

            # ---- Structural evidence.
            matched_blocks = max(matched) // mbs if matched else 0
            if not env.hybrid:
                # No state groups: uniform coverage, nothing to truncate.
                self.assertTrue(all(complete),
                                f"full-attention coverage is not uniform: {specs}")
                self.assertEqual(
                    matched_blocks, candidates,
                    f"full-attention match was truncated: {matched_blocks} of "
                    f"{candidates} blocks")
                return

            # Hybrid: what the manager advertises must match reality...
            self.assertEqual(
                complete, has_state,
                f"advertised spec coverage {complete} does not match the "
                f"blocks that really have a state {has_state}: state-less "
                f"blocks are published as fully cached")
            # ...and the match must stop at the last state-complete block.
            expected = 0
            for i in range(candidates):
                if has_state[i]:
                    expected = i + 1
            self.assertEqual(
                matched_blocks, expected,
                f"match not truncated to the last state-complete block: got "
                f"{matched_blocks} blocks, expected {expected} (state of "
                f"candidates: {has_state[:candidates]})")
            self.assertTrue(
                env.scan_connector_logs(r"truncated external match"),
                "the truncation was not logged")
        finally:
            env.stop()


if __name__ == "__main__":
    unittest.main()
