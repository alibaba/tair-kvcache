"""Write-session ownership shared by the scheduler, worker and data plane.

A write session is a lease: until ``finishWriteCache`` says which blocks were
really written, the manager must eventually reclaim all of them. The data plane
owns the open-session record because it produces the verdicts; the worker and
scheduler own manager calls. These small types and helpers are the hand-off, so
an engine-specific backend does not need another session vocabulary or payload.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Iterable, List, Protocol, Sequence, Tuple, runtime_checkable

from kv_cache_manager.py_connector.common.logger import logger


@dataclass
class SaveSession:
    """One data-plane save whose verdict has not reached the manager yet.

    ``verdicts`` starts all false and becomes this rank's per-block mask.
    ``vetoed`` is terminal: once source-page ownership is lost, a late all-rank
    success must not publish bytes whose provenance can no longer be proved.
    """

    req_id: str
    write_session_id: str
    num_blocks: int
    verdicts: List[bool]
    vetoed: bool = False


class SaveDisposition(Enum):
    """The atomically-closed state of a session at its publication point."""

    PUBLISH = auto()
    VETO = auto()
    ALREADY_REPORTED = auto()


@runtime_checkable
class SaveSessionBook(Protocol):
    """The session-ledger seam a worker uses to close or flush a data plane."""

    def open_save_sessions(self) -> List[SaveSession]: ...

    def take_open_save_sessions(self) -> List[SaveSession]: ...

    def complete_save_session_flush(self, sessions: Sequence[SaveSession]) -> None: ...

    def save_session_flush_state(
        self, session: SaveSession
    ) -> Tuple[bool, List[bool]]: ...

    def veto_save_sessions(self, req_ids: Iterable[str]) -> int: ...

    def close_save_session(self, write_session_id: str) -> SaveDisposition: ...


def flush_mask(verdicts: Sequence[bool], *, world: int) -> List[bool]:
    """Return the mask one process may publish without its peers' verdicts.

    A single-rank process has the complete verdict. With several ranks it only
    has its own share, so claiming success would be unsupported: publish none.
    """
    return list(verdicts) if world == 1 else [False] * len(verdicts)


def finish_write_cache_payload(
    *, instance_id: str, write_session_id: str, mask: Sequence[bool]
) -> dict[str, Any]:
    """Build the manager's ``finishWriteCache`` request body.

    The values are copied: asynchronous callers must not observe a mask that a
    later coordinator callback mutates. One boolean per manager block is the
    wire contract; this helper deliberately does not infer or pad a short mask.
    """
    return {
        "trace_id": f"finish_{write_session_id[:8]}",
        "instance_id": instance_id,
        "write_session_id": write_session_id,
        "success_blocks": {"bool_masks": {"values": list(mask)}},
    }


def finish_save_session(
    *,
    manager_client: Any,
    instance_id: str,
    write_session_id: str,
    mask: Sequence[bool],
    reason: str,
) -> bool:
    """Finish one lease, returning whether the manager accepted the verdict."""
    try:
        manager_client.finish_write_cache(
            finish_write_cache_payload(
                instance_id=instance_id,
                write_session_id=write_session_id,
                mask=mask,
            )
        )
    except Exception:
        logger.warning(
            "finish_write_cache failed for session %s (%d block(s) stay "
            "unpublished until the manager expires them); reason: %s",
            write_session_id,
            len(mask),
            reason,
            exc_info=True,
        )
        return False
    logger.info(
        "finished save session %s with %d/%d block(s) written (reason: %s)",
        write_session_id,
        sum(mask),
        len(mask),
        reason,
    )
    return True


def finish_abandoned(
    *,
    manager_client: Any,
    instance_id: str,
    write_session_id: str,
    num_blocks: int,
    reason: str,
) -> bool:
    """Close a session no data plane will run, publishing no blocks."""
    logger.error(
        "save session %s (%d block(s)) was abandoned before transfer; "
        "finishing every block as unwritten: %s",
        write_session_id,
        num_blocks,
        reason,
    )
    return finish_save_session(
        manager_client=manager_client,
        instance_id=instance_id,
        write_session_id=write_session_id,
        mask=[False] * num_blocks,
        reason=reason,
    )
