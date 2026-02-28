from __future__ import annotations

import hashlib
from datetime import datetime
from typing import TYPE_CHECKING

from artana.canonicalization import canonical_json_dumps

if TYPE_CHECKING:
    from artana.events import EventPayload, EventType


def payload_to_canonical_json(payload: EventPayload) -> str:
    payload_dict = payload.model_dump(mode="json")
    if (
        payload_dict.get("kind") == "model_requested"
        and payload_dict.get("allowed_tools_hash") is None
    ):
        payload_dict.pop("allowed_tools_hash", None)
    if payload_dict.get("kind") == "model_requested":
        if payload_dict.get("allowed_tool_signatures") == []:
            payload_dict.pop("allowed_tool_signatures", None)
        if payload_dict.get("context_version") is None:
            payload_dict.pop("context_version", None)
        if payload_dict.get("model_cycle_id") is None:
            payload_dict.pop("model_cycle_id", None)
        messages = payload_dict.get("messages")
        if isinstance(messages, list):
            for message in messages:
                if not isinstance(message, dict):
                    continue
                if message.get("tool_call_id") is None:
                    message.pop("tool_call_id", None)
                if message.get("name") is None:
                    message.pop("name", None)
                if message.get("tool_calls") is None:
                    message.pop("tool_calls", None)
    if payload_dict.get("kind") == "model_terminal":
        tool_calls = payload_dict.get("tool_calls")
        if isinstance(tool_calls, list):
            for tool_call in tool_calls:
                if not isinstance(tool_call, dict):
                    continue
                if tool_call.get("tool_call_id") is None:
                    tool_call.pop("tool_call_id", None)
        for optional_key in (
            "failure_reason",
            "error_category",
            "error_class",
            "http_status",
            "provider_request_id",
            "diagnostics_json",
            "output_json",
            "prompt_tokens",
            "completion_tokens",
            "cost_usd",
            "api_mode_used",
            "response_id",
            "step_key",
        ):
            if payload_dict.get(optional_key) is None:
                payload_dict.pop(optional_key, None)
    if payload_dict.get("kind") == "tool_requested":
        # Keep step_key=null for backward compatibility with historical hash chains.
        # Legacy events included `step_key` even when null.
        if payload_dict.get("semantic_idempotency_key") is None:
            payload_dict.pop("semantic_idempotency_key", None)
        if payload_dict.get("intent_id") is None:
            payload_dict.pop("intent_id", None)
        if payload_dict.get("amount_usd") is None:
            payload_dict.pop("amount_usd", None)
    return canonical_json_dumps(payload_dict)


def compute_allowed_tools_hash(tool_names: list[str]) -> str:
    joined = ",".join(sorted(tool_names))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def compute_event_hash(
    *,
    event_id: str,
    run_id: str,
    tenant_id: str,
    seq: int,
    event_type: EventType,
    prev_event_hash: str | None,
    timestamp: datetime,
    parent_step_key: str | None,
    payload: EventPayload,
) -> str:
    hash_fields = [
        event_id,
        run_id,
        tenant_id,
        str(seq),
        event_type.value,
        prev_event_hash or "",
        timestamp.isoformat(),
    ]
    if parent_step_key is not None:
        hash_fields.append(parent_step_key)
    hash_fields.append(payload_to_canonical_json(payload))
    joined = "|".join(hash_fields)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()
