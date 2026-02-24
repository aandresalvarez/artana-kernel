from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

_STEP_KEY: ContextVar[str | None] = ContextVar("artana_tool_request_step_key", default=None)
_PARENT_STEP_KEY: ContextVar[str | None] = ContextVar(
    "artana_tool_request_parent_step_key",
    default=None,
)


@contextmanager
def tool_request_context(*, step_key: str | None, parent_step_key: str | None) -> Iterator[None]:
    step_token = _STEP_KEY.set(step_key)
    parent_token = _PARENT_STEP_KEY.set(parent_step_key)
    try:
        yield
    finally:
        _STEP_KEY.reset(step_token)
        _PARENT_STEP_KEY.reset(parent_token)


def current_tool_step_key() -> str | None:
    return _STEP_KEY.get()


def current_parent_step_key() -> str | None:
    return _PARENT_STEP_KEY.get()

