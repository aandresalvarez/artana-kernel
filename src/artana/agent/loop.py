from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DraftVerifyLoopConfig:
    draft_model: str
    verify_model: str


__all__ = ["DraftVerifyLoopConfig"]
