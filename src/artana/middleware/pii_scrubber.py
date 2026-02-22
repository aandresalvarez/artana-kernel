from __future__ import annotations

import re

from artana.events import ChatMessage
from artana.middleware.base import ModelInvocation
from artana.models import TenantContext
from artana.ports.model import ModelUsage


class PIIScrubberMiddleware:
    def __init__(self) -> None:
        self._patterns: tuple[tuple[re.Pattern[str], str], ...] = (
            (
                re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
                "[REDACTED_EMAIL]",
            ),
            (
                re.compile(r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b"),
                "[REDACTED_PHONE]",
            ),
            (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[REDACTED_SSN]"),
        )

    async def prepare_model(self, invocation: ModelInvocation) -> ModelInvocation:
        redacted_prompt = self._redact_text(invocation.prompt)
        redacted_messages = tuple(
            ChatMessage(role=message.role, content=self._redact_text(message.content))
            for message in invocation.messages
        )
        return invocation.with_updates(
            prompt=redacted_prompt,
            messages=redacted_messages,
        )

    async def before_model(self, *, run_id: str, tenant: TenantContext) -> None:
        return None

    async def after_model(
        self, *, run_id: str, tenant: TenantContext, usage: ModelUsage
    ) -> None:
        return None

    def _redact_text(self, input_text: str) -> str:
        result = input_text
        for pattern, replacement in self._patterns:
            result = pattern.sub(replacement, result)
        return result

