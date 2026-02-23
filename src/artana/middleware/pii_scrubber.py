from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence

from artana.json_utils import canonical_json_dumps
from artana.middleware.base import ModelInvocation
from artana.models import TenantContext
from artana.ports.model import ModelUsage


class PIIScrubberMiddleware:
    """Demo-only regex scrubber for basic examples.

    This middleware is intentionally minimal and does not provide production-grade
    PII detection/coverage guarantees.
    """

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
            message.model_copy(update={"content": self._redact_text(message.content)})
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

    async def prepare_tool_request(
        self,
        *,
        run_id: str,
        tenant: TenantContext,
        tool_name: str,
        arguments_json: str,
    ) -> str:
        return self._redact_json_payload(arguments_json)

    async def prepare_tool_result(
        self,
        *,
        run_id: str,
        tenant: TenantContext,
        tool_name: str,
        result_json: str,
    ) -> str:
        return self._redact_json_payload(result_json)

    def _redact_text(self, input_text: str) -> str:
        result = input_text
        for pattern, replacement in self._patterns:
            result = pattern.sub(replacement, result)
        return result

    def _redact_json_payload(self, payload_json: str) -> str:
        try:
            parsed = json.loads(payload_json)
        except json.JSONDecodeError:
            return self._redact_text(payload_json)
        redacted = self._redact_value(parsed)
        return canonical_json_dumps(redacted)

    def _redact_value(self, value: object) -> object:
        if isinstance(value, str):
            return self._redact_text(value)
        if isinstance(value, Mapping):
            return {str(key): self._redact_value(nested) for key, nested in value.items()}
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return [self._redact_value(item) for item in value]
        return value
