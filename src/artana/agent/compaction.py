from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel

from artana.events import ChatMessage


class CompactionSummary(BaseModel):
    summary: str


@dataclass(frozen=True, slots=True)
class CompactionStrategy:
    max_history_messages: int = 40
    trigger_at_messages: int | None = None
    keep_recent_messages: int = 10
    summarize_with_model: str = "gpt-4o-mini"
    max_context_tokens: int | None = None
    context_utilization_ratio: float = 0.8

    def __post_init__(self) -> None:
        if self.max_history_messages < 1:
            raise ValueError("max_history_messages must be >= 1.")
        if self.trigger_at_messages is not None and self.trigger_at_messages < 1:
            raise ValueError("trigger_at_messages must be >= 1 when provided.")
        if self.keep_recent_messages < 0:
            raise ValueError("keep_recent_messages must be >= 0.")
        if self.max_context_tokens is not None and self.max_context_tokens < 1:
            raise ValueError("max_context_tokens must be >= 1 when provided.")
        if self.context_utilization_ratio < 0.0:
            raise ValueError("context_utilization_ratio must be >= 0.0.")

    def should_compact(self, *, messages: tuple[ChatMessage, ...], model: str) -> bool:
        history_limit = (
            self.trigger_at_messages
            if self.trigger_at_messages is not None
            else self.max_history_messages
        )
        if len(messages) >= history_limit:
            return True
        if self.max_context_tokens is None:
            return False
        if self.context_utilization_ratio <= 0.0:
            return False
        ratio_limit = self.max_context_tokens * self.context_utilization_ratio
        return estimate_tokens(messages=messages, model=model) >= int(ratio_limit)


def estimate_tokens(messages: tuple[ChatMessage, ...], model: str) -> int:
    try:
        from litellm import token_counter

        count = token_counter(
            messages=[{"role": message.role, "content": message.content} for message in messages],
            model=model,
        )
        if isinstance(count, int):
            return count
    except Exception:
        pass
    total_chars = sum(len(message.content) for message in messages)
    return max(1, int(total_chars / 4))


__all__ = ["CompactionStrategy", "CompactionSummary", "estimate_tokens"]
