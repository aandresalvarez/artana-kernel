from __future__ import annotations

import ast
import json
from datetime import datetime, timedelta, timezone
from string import Formatter
from typing import Mapping

from artana._kernel.tool_request_context import current_parent_step_key, current_tool_step_key
from artana._kernel.types import ApprovalRequiredError, PolicyViolationError
from artana.canonicalization import canonical_json_dumps, canonicalize_json_object
from artana.events import EventType, KernelEvent, RunSummaryPayload, ToolRequestedPayload
from artana.json_utils import sha256_hex
from artana.middleware.base import ModelInvocation, PreparedToolRequest
from artana.models import TenantContext
from artana.ports.model import ModelUsage
from artana.safety import (
    ApprovalGatePolicy,
    IntentPlanRecord,
    InvariantRule,
    SafetyPolicyConfig,
    SemanticIdempotencyRequirement,
    ToolLimitPolicy,
)
from artana.store.base import EventStore, SupportsToolPolicyAggregation

_MISSING = object()


class SafetyPolicyMiddleware:
    def __init__(
        self,
        *,
        config: SafetyPolicyConfig,
        store: EventStore | None = None,
    ) -> None:
        self._config = config
        self._store = store

    def bind_store(self, store: EventStore) -> None:
        self._store = store

    async def prepare_model(self, invocation: ModelInvocation) -> ModelInvocation:
        return invocation

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
    ) -> str | PreparedToolRequest:
        policy = self._config.policy_for(tool_name)
        if policy is None:
            return arguments_json
        store = self._require_store()
        canonical_arguments_json = canonicalize_json_object(arguments_json)
        parsed_arguments = _parse_arguments(canonical_arguments_json)
        events = await store.get_events_for_run(run_id)
        step_key = current_tool_step_key()
        parent_step_key = current_parent_step_key()
        fingerprint = _decision_fingerprint(
            tenant_id=tenant.tenant_id,
            tool_name=tool_name,
            arguments_json=canonical_arguments_json,
            step_key=step_key,
        )
        replay_candidate = _is_replay_candidate(
            events=events,
            tool_name=tool_name,
            arguments_json=canonical_arguments_json,
            step_key=step_key,
        )

        intent_id: str | None = None
        if policy.intent is not None and policy.intent.require_intent:
            intent_id = self._resolve_intent_id(
                events=events,
                tool_name=tool_name,
                max_age_seconds=policy.intent.max_age_seconds,
            )
            if intent_id is None:
                await self._emit_policy_decision(
                    run_id=run_id,
                    tenant_id=tenant.tenant_id,
                    tool_name=tool_name,
                    fingerprint=fingerprint,
                    outcome="deny",
                    rule_id="intent",
                    reason="missing_intent_plan",
                    step_key=step_key,
                    parent_step_key=parent_step_key,
                )
                raise PolicyViolationError(
                    code="missing_intent_plan",
                    message=f"Tool {tool_name!r} requires an active intent plan.",
                    tool_name=tool_name,
                    fingerprint=fingerprint,
                )

        semantic_key: str | None = None
        if policy.semantic_idempotency is not None:
            semantic_key = self._derive_semantic_key(
                policy=policy.semantic_idempotency,
                tenant=tenant,
                tool_name=tool_name,
                arguments=parsed_arguments,
                fingerprint=fingerprint,
            )
            if not replay_candidate:
                await self._enforce_semantic_uniqueness(
                    store=store,
                    run_id=run_id,
                    tenant=tenant,
                    tool_name=tool_name,
                    semantic_key=semantic_key,
                    arguments_json=canonical_arguments_json,
                    step_key=step_key,
                    fingerprint=fingerprint,
                    parent_step_key=parent_step_key,
                )

        amount_usd: float | None = None
        if policy.limits is not None:
            amount_usd = self._extract_amount_if_present(
                policy=policy.limits,
                arguments=parsed_arguments,
                tool_name=tool_name,
                fingerprint=fingerprint,
            )
            if not replay_candidate:
                await self._enforce_limits(
                    store=store,
                    run_id=run_id,
                    tenant=tenant,
                    tool_name=tool_name,
                    limits=policy.limits,
                    amount_usd=amount_usd,
                    fingerprint=fingerprint,
                    step_key=step_key,
                    parent_step_key=parent_step_key,
                )

        if policy.approval is not None:
            await self._enforce_approval(
                store=store,
                run_id=run_id,
                tenant=tenant,
                tool_name=tool_name,
                approval=policy.approval,
                fingerprint=fingerprint,
                arguments_json=canonical_arguments_json,
                semantic_idempotency_key=semantic_key,
                step_key=step_key,
                parent_step_key=parent_step_key,
            )

        if len(policy.invariants) > 0:
            await self._enforce_invariants(
                events=events,
                run_id=run_id,
                tenant=tenant,
                tool_name=tool_name,
                invariants=policy.invariants,
                arguments=parsed_arguments,
                fingerprint=fingerprint,
                step_key=step_key,
                parent_step_key=parent_step_key,
            )

        await self._emit_policy_decision(
            run_id=run_id,
            tenant_id=tenant.tenant_id,
            tool_name=tool_name,
            fingerprint=fingerprint,
            outcome="allow",
            rule_id="pipeline",
            reason="policy_checks_passed",
            step_key=step_key,
            parent_step_key=parent_step_key,
        )
        return PreparedToolRequest(
            arguments_json=canonical_arguments_json,
            semantic_idempotency_key=semantic_key,
            intent_id=intent_id,
            amount_usd=amount_usd,
        )

    async def prepare_tool_result(
        self,
        *,
        run_id: str,
        tenant: TenantContext,
        tool_name: str,
        result_json: str,
    ) -> str:
        return result_json

    def _require_store(self) -> EventStore:
        if self._store is None:
            raise RuntimeError("SafetyPolicyMiddleware store is not configured.")
        return self._store

    def _resolve_intent_id(
        self,
        *,
        events: list[KernelEvent],
        tool_name: str,
        max_age_seconds: int | None,
    ) -> str | None:
        now = datetime.now(timezone.utc)
        for event in reversed(events):
            if event.event_type != EventType.RUN_SUMMARY:
                continue
            payload = event.payload
            if not isinstance(payload, RunSummaryPayload):
                continue
            if payload.summary_type != "policy::intent_plan":
                continue
            try:
                intent = IntentPlanRecord.model_validate_json(payload.summary_json)
            except Exception:
                continue
            if len(intent.applies_to_tools) > 0 and tool_name not in intent.applies_to_tools:
                continue
            intent_created_at = intent.created_at
            if intent_created_at.tzinfo is None:
                intent_created_at = intent_created_at.replace(tzinfo=timezone.utc)
            if max_age_seconds is not None:
                max_age = timedelta(seconds=max_age_seconds)
                if now - intent_created_at > max_age:
                    continue
            return intent.intent_id
        return None

    async def _enforce_semantic_uniqueness(
        self,
        *,
        store: EventStore,
        run_id: str,
        tenant: TenantContext,
        tool_name: str,
        semantic_key: str,
        arguments_json: str,
        step_key: str | None,
        fingerprint: str,
        parent_step_key: str | None,
    ) -> None:
        if not isinstance(store, SupportsToolPolicyAggregation):
            raise RuntimeError(
                "SafetyPolicyMiddleware semantic idempotency requires a store implementing "
                "SupportsToolPolicyAggregation."
            )
        latest = await store.get_latest_tool_semantic_outcome(
            tenant_id=tenant.tenant_id,
            tool_name=tool_name,
            semantic_idempotency_key=semantic_key,
        )
        if latest is None:
            return
        if (
            latest.run_id == run_id
            and step_key is not None
            and latest.request_step_key == step_key
            and canonicalize_json_object(latest.request_arguments_json)
            == canonicalize_json_object(arguments_json)
        ):
            return
        if latest.outcome == "unknown_outcome":
            await self._emit_policy_decision(
                run_id=run_id,
                tenant_id=tenant.tenant_id,
                tool_name=tool_name,
                fingerprint=fingerprint,
                outcome="deny",
                rule_id="semantic_idempotency",
                reason="semantic_requires_reconciliation",
                step_key=step_key,
                parent_step_key=parent_step_key,
            )
            raise PolicyViolationError(
                code="semantic_requires_reconciliation",
                message=(
                    f"Tool {tool_name!r} semantic key {semantic_key!r} has unknown outcome "
                    "and must be reconciled first."
                ),
                tool_name=tool_name,
                fingerprint=fingerprint,
            )
        if latest.outcome == "success":
            await self._emit_policy_decision(
                run_id=run_id,
                tenant_id=tenant.tenant_id,
                tool_name=tool_name,
                fingerprint=fingerprint,
                outcome="deny",
                rule_id="semantic_idempotency",
                reason="semantic_duplicate",
                step_key=step_key,
                parent_step_key=parent_step_key,
            )
            raise PolicyViolationError(
                code="semantic_duplicate",
                message=(
                    f"Tool {tool_name!r} semantic key {semantic_key!r} was already executed "
                    "successfully."
                ),
                tool_name=tool_name,
                fingerprint=fingerprint,
            )

    async def _enforce_limits(
        self,
        *,
        store: EventStore,
        run_id: str,
        tenant: TenantContext,
        tool_name: str,
        limits: ToolLimitPolicy,
        amount_usd: float | None,
        fingerprint: str,
        step_key: str | None,
        parent_step_key: str | None,
    ) -> None:
        if not isinstance(store, SupportsToolPolicyAggregation):
            raise RuntimeError(
                "SafetyPolicyMiddleware limits require a store implementing "
                "SupportsToolPolicyAggregation."
            )
        if limits.max_calls_per_run is not None:
            count_for_run = await store.get_tool_request_count_for_run(
                run_id=run_id,
                tool_name=tool_name,
            )
            if count_for_run >= limits.max_calls_per_run:
                await self._emit_policy_decision(
                    run_id=run_id,
                    tenant_id=tenant.tenant_id,
                    tool_name=tool_name,
                    fingerprint=fingerprint,
                    outcome="deny",
                    rule_id="limit:max_calls_per_run",
                    reason=f"count={count_for_run},limit={limits.max_calls_per_run}",
                    step_key=step_key,
                    parent_step_key=parent_step_key,
                )
                raise PolicyViolationError(
                    code="tool_limit_exceeded",
                    message=(
                        f"Tool {tool_name!r} exceeded max_calls_per_run="
                        f"{limits.max_calls_per_run}."
                    ),
                    tool_name=tool_name,
                    fingerprint=fingerprint,
                )
        if limits.max_calls_per_tenant_window is not None:
            if limits.tenant_window_seconds is None:
                raise RuntimeError("tenant_window_seconds is required for tenant-window limits.")
            since = datetime.now(timezone.utc) - timedelta(seconds=limits.tenant_window_seconds)
            count_for_tenant = await store.get_tool_request_count_for_tenant_since(
                tenant_id=tenant.tenant_id,
                tool_name=tool_name,
                since=since,
            )
            if count_for_tenant >= limits.max_calls_per_tenant_window:
                await self._emit_policy_decision(
                    run_id=run_id,
                    tenant_id=tenant.tenant_id,
                    tool_name=tool_name,
                    fingerprint=fingerprint,
                    outcome="deny",
                    rule_id="limit:max_calls_per_tenant_window",
                    reason=(
                        f"count={count_for_tenant},limit={limits.max_calls_per_tenant_window},"
                        f"window_seconds={limits.tenant_window_seconds}"
                    ),
                    step_key=step_key,
                    parent_step_key=parent_step_key,
                )
                raise PolicyViolationError(
                    code="tool_rate_limited",
                    message=(
                        f"Tool {tool_name!r} exceeded tenant window limit "
                        f"{limits.max_calls_per_tenant_window}."
                    ),
                    tool_name=tool_name,
                    fingerprint=fingerprint,
                )
        if limits.max_amount_usd_per_call is not None:
            if amount_usd is None:
                raise PolicyViolationError(
                    code="tool_amount_missing",
                    message=(
                        f"Tool {tool_name!r} requires amount extraction via amount_arg_path "
                        "for max_amount_usd_per_call checks."
                    ),
                    tool_name=tool_name,
                    fingerprint=fingerprint,
                )
            if amount_usd > limits.max_amount_usd_per_call:
                await self._emit_policy_decision(
                    run_id=run_id,
                    tenant_id=tenant.tenant_id,
                    tool_name=tool_name,
                    fingerprint=fingerprint,
                    outcome="deny",
                    rule_id="limit:max_amount_usd_per_call",
                    reason=f"amount={amount_usd},limit={limits.max_amount_usd_per_call}",
                    step_key=step_key,
                    parent_step_key=parent_step_key,
                )
                raise PolicyViolationError(
                    code="tool_amount_exceeded",
                    message=(
                        f"Tool {tool_name!r} amount {amount_usd} exceeds "
                        f"max_amount_usd_per_call={limits.max_amount_usd_per_call}."
                    ),
                    tool_name=tool_name,
                    fingerprint=fingerprint,
                )

    async def _enforce_approval(
        self,
        *,
        store: EventStore,
        run_id: str,
        tenant: TenantContext,
        tool_name: str,
        approval: ApprovalGatePolicy,
        fingerprint: str,
        arguments_json: str,
        semantic_idempotency_key: str | None,
        step_key: str | None,
        parent_step_key: str | None,
    ) -> None:
        approval_key = _approval_key(
            tenant_id=tenant.tenant_id,
            tool_name=tool_name,
            arguments_json=arguments_json,
            semantic_idempotency_key=semantic_idempotency_key,
        )
        summary_type = f"policy::approval::{approval_key}"
        summary = await store.get_latest_run_summary(run_id, summary_type)
        if _approval_is_valid(summary=summary, approval=approval):
            return
        await self._emit_policy_decision(
            run_id=run_id,
            tenant_id=tenant.tenant_id,
            tool_name=tool_name,
            fingerprint=fingerprint,
            outcome="deny",
            rule_id=f"approval:{approval.mode}",
            reason="approval_required",
            step_key=step_key,
            parent_step_key=parent_step_key,
        )
        raise ApprovalRequiredError(
            tool_name=tool_name,
            approval_key=approval_key,
            mode=approval.mode,
            critic_model=approval.critic_model,
            fingerprint=fingerprint,
            arguments_json=arguments_json,
            message=f"Tool {tool_name!r} requires {approval.mode} approval.",
        )

    async def _enforce_invariants(
        self,
        *,
        events: list[KernelEvent],
        run_id: str,
        tenant: TenantContext,
        tool_name: str,
        invariants: tuple[InvariantRule, ...],
        arguments: dict[str, object],
        fingerprint: str,
        step_key: str | None,
        parent_step_key: str | None,
    ) -> None:
        for invariant in invariants:
            if invariant.type == "required_arg_true":
                if invariant.field is None:
                    continue
                value = _resolve_arg(arguments, invariant.field)
                if value is not True:
                    await self._emit_policy_decision(
                        run_id=run_id,
                        tenant_id=tenant.tenant_id,
                        tool_name=tool_name,
                        fingerprint=fingerprint,
                        outcome="deny",
                        rule_id="invariant:required_arg_true",
                        reason=f"field={invariant.field}",
                        step_key=step_key,
                        parent_step_key=parent_step_key,
                    )
                    raise PolicyViolationError(
                        code="invariant_violation",
                        message=(
                            f"Tool {tool_name!r} invariant required_arg_true failed for "
                            f"field={invariant.field!r}."
                        ),
                        tool_name=tool_name,
                        fingerprint=fingerprint,
                    )
            elif invariant.type == "email_domain_allowlist":
                if invariant.field is None:
                    continue
                value = _resolve_arg(arguments, invariant.field)
                if not isinstance(value, str) or "@" not in value:
                    raise PolicyViolationError(
                        code="invariant_violation",
                        message=(
                            f"Tool {tool_name!r} invariant email_domain_allowlist requires a "
                            f"valid email string at field={invariant.field!r}."
                        ),
                        tool_name=tool_name,
                        fingerprint=fingerprint,
                    )
                domain = value.split("@", 1)[1].lower()
                allowed = {item.lower() for item in invariant.allowed_domains}
                if domain not in allowed:
                    await self._emit_policy_decision(
                        run_id=run_id,
                        tenant_id=tenant.tenant_id,
                        tool_name=tool_name,
                        fingerprint=fingerprint,
                        outcome="deny",
                        rule_id="invariant:email_domain_allowlist",
                        reason=f"domain={domain}",
                        step_key=step_key,
                        parent_step_key=parent_step_key,
                    )
                    raise PolicyViolationError(
                        code="invariant_violation",
                        message=(
                            f"Tool {tool_name!r} email domain {domain!r} is not in the "
                            "allowlist."
                        ),
                        tool_name=tool_name,
                        fingerprint=fingerprint,
                    )
            elif invariant.type == "recipient_must_be_verified":
                if invariant.recipient_arg_path is None:
                    continue
                recipient = _resolve_arg(arguments, invariant.recipient_arg_path)
                if not isinstance(recipient, str):
                    raise PolicyViolationError(
                        code="invariant_violation",
                        message=(
                            f"Tool {tool_name!r} invariant recipient_must_be_verified requires "
                            f"a string recipient at {invariant.recipient_arg_path!r}."
                        ),
                        tool_name=tool_name,
                        fingerprint=fingerprint,
                    )
                if not _recipient_is_verified(
                    events=events,
                    summary_type=invariant.verification_summary_type,
                    recipient=recipient,
                ):
                    await self._emit_policy_decision(
                        run_id=run_id,
                        tenant_id=tenant.tenant_id,
                        tool_name=tool_name,
                        fingerprint=fingerprint,
                        outcome="deny",
                        rule_id="invariant:recipient_must_be_verified",
                        reason=f"recipient={recipient}",
                        step_key=step_key,
                        parent_step_key=parent_step_key,
                    )
                    raise PolicyViolationError(
                        code="invariant_violation",
                        message=(
                            f"Tool {tool_name!r} recipient {recipient!r} is not marked as verified."
                        ),
                        tool_name=tool_name,
                        fingerprint=fingerprint,
                    )
            elif invariant.type == "custom_json_rule":
                if invariant.json_path is None or invariant.operator is None:
                    continue
                current_value = _resolve_arg(arguments, invariant.json_path)
                expected_value = _load_json_value_or_raise(invariant.value_json)
                if not _evaluate_custom_rule(
                    left=current_value,
                    operator=invariant.operator,
                    right=expected_value,
                ):
                    await self._emit_policy_decision(
                        run_id=run_id,
                        tenant_id=tenant.tenant_id,
                        tool_name=tool_name,
                        fingerprint=fingerprint,
                        outcome="deny",
                        rule_id="invariant:custom_json_rule",
                        reason=f"path={invariant.json_path},operator={invariant.operator}",
                        step_key=step_key,
                        parent_step_key=parent_step_key,
                    )
                    raise PolicyViolationError(
                        code="invariant_violation",
                        message=(
                            f"Tool {tool_name!r} custom_json_rule failed at path "
                            f"{invariant.json_path!r}."
                        ),
                        tool_name=tool_name,
                        fingerprint=fingerprint,
                    )
            elif invariant.type == "ast_validation_passed":
                if invariant.field is None:
                    continue
                code = _resolve_arg(arguments, invariant.field)
                if not isinstance(code, str):
                    raise PolicyViolationError(
                        code="invariant_violation",
                        message=(
                            f"Tool {tool_name!r} invariant ast_validation_passed requires "
                            f"a string code payload at field={invariant.field!r}."
                        ),
                        tool_name=tool_name,
                        fingerprint=fingerprint,
                    )
                ast_ok, ast_error = _validate_python_ast(code)
                if not ast_ok:
                    await self._emit_policy_decision(
                        run_id=run_id,
                        tenant_id=tenant.tenant_id,
                        tool_name=tool_name,
                        fingerprint=fingerprint,
                        outcome="deny",
                        rule_id="invariant:ast_validation_passed",
                        reason=ast_error or f"field={invariant.field}",
                        step_key=step_key,
                        parent_step_key=parent_step_key,
                    )
                    raise PolicyViolationError(
                        code="invariant_violation",
                        message=(
                            f"Tool {tool_name!r} ast_validation_passed failed for "
                            f"field={invariant.field!r}: {ast_error or 'invalid syntax'}."
                        ),
                        tool_name=tool_name,
                        fingerprint=fingerprint,
                    )
            elif invariant.type == "linter_passed":
                if invariant.field is None:
                    continue
                code = _resolve_arg(arguments, invariant.field)
                if not isinstance(code, str):
                    raise PolicyViolationError(
                        code="invariant_violation",
                        message=(
                            f"Tool {tool_name!r} invariant linter_passed requires "
                            f"a string code payload at field={invariant.field!r}."
                        ),
                        tool_name=tool_name,
                        fingerprint=fingerprint,
                    )
                lint_ok, lint_error = _validate_lint(code)
                if not lint_ok:
                    await self._emit_policy_decision(
                        run_id=run_id,
                        tenant_id=tenant.tenant_id,
                        tool_name=tool_name,
                        fingerprint=fingerprint,
                        outcome="deny",
                        rule_id="invariant:linter_passed",
                        reason=lint_error or f"field={invariant.field}",
                        step_key=step_key,
                        parent_step_key=parent_step_key,
                    )
                    raise PolicyViolationError(
                        code="invariant_violation",
                        message=(
                            f"Tool {tool_name!r} linter_passed failed for "
                            f"field={invariant.field!r}: {lint_error or 'lint failed'}."
                        ),
                        tool_name=tool_name,
                        fingerprint=fingerprint,
                    )

    def _derive_semantic_key(
        self,
        *,
        policy: SemanticIdempotencyRequirement,
        tenant: TenantContext,
        tool_name: str,
        arguments: dict[str, object],
        fingerprint: str,
    ) -> str:
        template_fields = _template_fields(policy.template)
        substitutions: dict[str, str] = {}
        for field_name in template_fields:
            value = _resolve_template_field(
                field_name=field_name,
                tenant=tenant,
                tool_name=tool_name,
                arguments=arguments,
            )
            if value is _MISSING:
                raise PolicyViolationError(
                    code="semantic_missing_field",
                    message=(
                        f"Semantic idempotency template field {field_name!r} is missing "
                        f"for tool {tool_name!r}."
                    ),
                    tool_name=tool_name,
                    fingerprint=fingerprint,
                )
            substitutions[field_name] = _stringify_template_value(value)
        for required_field in policy.required_fields:
            value = _resolve_template_field(
                field_name=required_field,
                tenant=tenant,
                tool_name=tool_name,
                arguments=arguments,
            )
            if value is _MISSING:
                raise PolicyViolationError(
                    code="semantic_missing_field",
                    message=(
                        f"Semantic idempotency required field {required_field!r} is missing "
                        f"for tool {tool_name!r}."
                    ),
                    tool_name=tool_name,
                    fingerprint=fingerprint,
                )
            substitutions.setdefault(required_field, _stringify_template_value(value))
        try:
            return policy.template.format_map(substitutions)
        except Exception as exc:
            raise PolicyViolationError(
                code="semantic_template_error",
                message=(
                    f"Failed to derive semantic idempotency key for tool {tool_name!r}: {exc}"
                ),
                tool_name=tool_name,
                fingerprint=fingerprint,
            ) from exc

    def _extract_amount_if_present(
        self,
        *,
        policy: ToolLimitPolicy,
        arguments: dict[str, object],
        tool_name: str,
        fingerprint: str,
    ) -> float | None:
        if policy.amount_arg_path is None:
            return None
        value = _resolve_arg(arguments, policy.amount_arg_path)
        if value is _MISSING:
            return None
        if isinstance(value, bool):
            raise PolicyViolationError(
                code="tool_amount_invalid",
                message=f"Tool {tool_name!r} amount value at {policy.amount_arg_path!r} is bool.",
                tool_name=tool_name,
                fingerprint=fingerprint,
            )
        if isinstance(value, int):
            return float(value)
        if isinstance(value, float):
            return value
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError as exc:
                raise PolicyViolationError(
                    code="tool_amount_invalid",
                    message=(
                        f"Tool {tool_name!r} amount value at {policy.amount_arg_path!r} "
                        f"is not a valid number: {value!r}."
                    ),
                    tool_name=tool_name,
                    fingerprint=fingerprint,
                ) from exc
        raise PolicyViolationError(
            code="tool_amount_invalid",
            message=(
                f"Tool {tool_name!r} amount value at {policy.amount_arg_path!r} has unsupported "
                f"type {type(value)!r}."
            ),
            tool_name=tool_name,
            fingerprint=fingerprint,
        )

    async def _emit_policy_decision(
        self,
        *,
        run_id: str,
        tenant_id: str,
        tool_name: str,
        fingerprint: str,
        outcome: str,
        rule_id: str,
        reason: str,
        step_key: str | None,
        parent_step_key: str | None,
    ) -> None:
        store = self._require_store()
        await store.append_event(
            run_id=run_id,
            tenant_id=tenant_id,
            event_type=EventType.RUN_SUMMARY,
            parent_step_key=parent_step_key,
            payload=RunSummaryPayload(
                summary_type="policy_decision",
                summary_json=json.dumps(
                    {
                        "tool_name": tool_name,
                        "fingerprint": fingerprint,
                        "outcome": outcome,
                        "rule_id": rule_id,
                        "reason": reason,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                step_key=step_key,
            ),
        )


def _parse_arguments(arguments_json: str) -> dict[str, object]:
    raw = json.loads(arguments_json)
    if not isinstance(raw, dict):
        raise ValueError("Tool arguments must decode to an object.")
    parsed: dict[str, object] = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            raise ValueError("Tool argument keys must be strings.")
        parsed[key] = value
    return parsed


def _resolve_arg(arguments: Mapping[str, object], path: str) -> object:
    if path == "":
        return _MISSING
    current: object = arguments
    for part in path.split("."):
        if part == "":
            return _MISSING
        if not isinstance(current, Mapping):
            return _MISSING
        if part not in current:
            return _MISSING
        current = current[part]
    return current


def _template_fields(template: str) -> tuple[str, ...]:
    fields: list[str] = []
    formatter = Formatter()
    for _, field_name, _, _ in formatter.parse(template):
        if field_name is None or field_name == "":
            continue
        fields.append(field_name)
    return tuple(fields)


def _resolve_template_field(
    *,
    field_name: str,
    tenant: TenantContext,
    tool_name: str,
    arguments: Mapping[str, object],
) -> object:
    if field_name == "tenant_id":
        return tenant.tenant_id
    if field_name == "tool_name":
        return tool_name
    return _resolve_arg(arguments, field_name)


def _stringify_template_value(value: object) -> str:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return str(value)
    return canonical_json_dumps(value)


def _is_replay_candidate(
    *,
    events: list[KernelEvent],
    tool_name: str,
    arguments_json: str,
    step_key: str | None,
) -> bool:
    for event in reversed(events):
        if event.event_type != EventType.TOOL_REQUESTED:
            continue
        payload = event.payload
        if not isinstance(payload, ToolRequestedPayload):
            continue
        if payload.tool_name != tool_name:
            continue
        if payload.step_key != step_key:
            continue
        if canonicalize_json_object(payload.arguments_json) != canonicalize_json_object(
            arguments_json
        ):
            continue
        return True
    return False


def _decision_fingerprint(
    *,
    tenant_id: str,
    tool_name: str,
    arguments_json: str,
    step_key: str | None,
) -> str:
    token = f"{tenant_id}|{tool_name}|{step_key}|{arguments_json}"
    return sha256_hex(token)


def _approval_key(
    *,
    tenant_id: str,
    tool_name: str,
    arguments_json: str,
    semantic_idempotency_key: str | None,
) -> str:
    token = (
        f"{tenant_id}|{tool_name}|{arguments_json}|"
        f"{semantic_idempotency_key if semantic_idempotency_key is not None else ''}"
    )
    return sha256_hex(token)


def _approval_is_valid(*, summary: RunSummaryPayload | None, approval: ApprovalGatePolicy) -> bool:
    if summary is None:
        return False
    try:
        payload: object = json.loads(summary.summary_json)
    except json.JSONDecodeError:
        return False
    if not isinstance(payload, dict):
        return False
    approved = payload.get("approved")
    if approved is not True:
        return False
    mode = payload.get("mode")
    if mode != approval.mode:
        return False
    approved_at_obj = payload.get("approved_at")
    if approval.approval_ttl_seconds is None:
        return True
    if not isinstance(approved_at_obj, str):
        return False
    try:
        approved_at = datetime.fromisoformat(approved_at_obj)
    except ValueError:
        return False
    if approved_at.tzinfo is None:
        approved_at = approved_at.replace(tzinfo=timezone.utc)
    ttl = timedelta(seconds=approval.approval_ttl_seconds)
    return datetime.now(timezone.utc) - approved_at <= ttl


def _recipient_is_verified(
    *,
    events: list[KernelEvent],
    summary_type: str,
    recipient: str,
) -> bool:
    for event in reversed(events):
        if event.event_type != EventType.RUN_SUMMARY:
            continue
        payload = event.payload
        if not isinstance(payload, RunSummaryPayload):
            continue
        if payload.summary_type != summary_type:
            continue
        try:
            summary_payload: object = json.loads(payload.summary_json)
        except json.JSONDecodeError:
            continue
        if not isinstance(summary_payload, dict):
            continue
        if summary_payload.get("recipient") != recipient:
            continue
        if summary_payload.get("verified") is True:
            return True
    return False


def _load_json_value_or_raise(value_json: str | None) -> object:
    if value_json is None:
        raise ValueError("custom_json_rule requires value_json.")
    return json.loads(value_json)


def _validate_python_ast(code: str) -> tuple[bool, str | None]:
    try:
        ast.parse(code)
    except SyntaxError as exc:
        details = exc.msg
        if exc.lineno is not None:
            details = f"line {exc.lineno}: {details}"
        return False, details
    return True, None


def _validate_lint(code: str) -> tuple[bool, str | None]:
    ast_ok, ast_error = _validate_python_ast(code)
    if not ast_ok:
        return False, ast_error
    for line_number, line in enumerate(code.splitlines(), start=1):
        if "\t" in line:
            return False, f"line {line_number}: tabs are not allowed"
        if line.rstrip(" \t") != line:
            return False, f"line {line_number}: trailing whitespace is not allowed"
        if len(line) > 120:
            return False, f"line {line_number}: line length exceeds 120 characters"
    return True, None


def _evaluate_custom_rule(*, left: object, operator: str, right: object) -> bool:
    if left is _MISSING:
        return False
    if operator == "eq":
        return left == right
    if operator == "ne":
        return left != right
    if operator == "gt":
        return _safe_ordered_compare(left=left, right=right, op="gt")
    if operator == "gte":
        return _safe_ordered_compare(left=left, right=right, op="gte")
    if operator == "lt":
        return _safe_ordered_compare(left=left, right=right, op="lt")
    if operator == "lte":
        return _safe_ordered_compare(left=left, right=right, op="lte")
    if operator == "in":
        if isinstance(right, list):
            return left in right
        return False
    if operator == "not_in":
        if isinstance(right, list):
            return left not in right
        return False
    return False


def _safe_ordered_compare(*, left: object, right: object, op: str) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return False
    if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
        return False
    if op == "gt":
        return left > right
    if op == "gte":
        return left >= right
    if op == "lt":
        return left < right
    if op == "lte":
        return left <= right
    return False
