"""Direct tests for the shared write-session vocabulary and manager payload."""

import unittest
from unittest.mock import MagicMock

from kv_cache_manager.py_connector.vllm.save_session import (
    SaveSession,
    finish_abandoned,
    finish_save_session,
    finish_write_cache_payload,
    flush_mask,
)


class TestSaveSessionPayload(unittest.TestCase):
    def test_payload_has_the_manager_shape_and_copies_the_mask(self):
        mask = [True, False]
        payload = finish_write_cache_payload(
            instance_id="instance", write_session_id="session-123", mask=mask
        )
        mask[0] = False

        self.assertEqual(payload["trace_id"], "finish_session-")
        self.assertEqual(payload["instance_id"], "instance")
        self.assertEqual(payload["write_session_id"], "session-123")
        self.assertEqual(
            payload["success_blocks"]["bool_masks"]["values"], [True, False]
        )

    def test_single_rank_flush_uses_verdicts_but_multi_rank_claims_nothing(self):
        verdicts = [True, False, True]
        self.assertEqual(flush_mask(verdicts, world=1), verdicts)
        self.assertEqual(flush_mask(verdicts, world=2), [False, False, False])

    def test_abandoned_session_finishes_every_block_false(self):
        manager = MagicMock()
        accepted = finish_abandoned(
            manager_client=manager,
            instance_id="instance",
            write_session_id="orphan",
            num_blocks=3,
            reason="test retirement",
        )

        self.assertTrue(accepted)
        payload = manager.finish_write_cache.call_args.args[0]
        self.assertEqual(payload["success_blocks"]["bool_masks"]["values"], [False] * 3)

    def test_finish_failure_is_returned_not_raised(self):
        manager = MagicMock()
        manager.finish_write_cache.side_effect = RuntimeError("unavailable")

        accepted = finish_save_session(
            manager_client=manager,
            instance_id="instance",
            write_session_id="orphan",
            mask=[False],
            reason="test failure",
        )

        self.assertFalse(accepted)

    def test_session_veto_is_terminal_state_owned_by_the_record(self):
        session = SaveSession("req", "sess", 2, [False, False])
        self.assertFalse(session.vetoed)
        session.vetoed = True
        session.verdicts[:] = [True, True]
        self.assertTrue(session.vetoed)


if __name__ == "__main__":
    unittest.main()
