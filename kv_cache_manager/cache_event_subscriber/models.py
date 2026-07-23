from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class SourceNotReady(RuntimeError):
    """The engine is reachable but has not exposed an authoritative baseline."""


@dataclass(frozen=True, order=True)
class LocationSpec:
    name: str
    uri: str


@dataclass(frozen=True)
class BlockRecord:
    block_key: int
    medium: str
    group_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class EngineUpdate:
    """One source update whose cursor is committed only after KVCM ACKs it."""

    full_snapshot: bool
    blocks: tuple[BlockRecord, ...] = ()
    upserts: tuple[BlockRecord, ...] = ()
    removals: tuple[BlockRecord, ...] = ()
    commit_token: Any = None

    def __post_init__(self) -> None:
        if self.full_snapshot and (self.upserts or self.removals):
            raise ValueError("a full snapshot cannot also contain deltas")

    @property
    def empty(self) -> bool:
        return not self.full_snapshot and not self.upserts and not self.removals
