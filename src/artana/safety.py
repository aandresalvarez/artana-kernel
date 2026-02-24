from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field, model_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class IntentRequirement(BaseModel):
    require_intent: bool = True
    max_age_seconds: int | None = Field(default=None, gt=0)


class SemanticIdempotencyRequirement(BaseModel):
    template: str = Field(min_length=1)
    required_fields: tuple[str, ...] = Field(default_factory=tuple)


class ToolLimitPolicy(BaseModel):
    max_calls_per_run: int | None = Field(default=None, gt=0)
    max_calls_per_tenant_window: int | None = Field(default=None, gt=0)
    tenant_window_seconds: int | None = Field(default=None, gt=0)
    max_amount_usd_per_call: float | None = Field(default=None, gt=0.0)
    amount_arg_path: str | None = None

    @model_validator(mode="after")
    def validate_combinations(self) -> "ToolLimitPolicy":
        if self.max_calls_per_tenant_window is not None and self.tenant_window_seconds is None:
            raise ValueError(
                "tenant_window_seconds is required when max_calls_per_tenant_window is set."
            )
        if self.max_amount_usd_per_call is not None and self.amount_arg_path is None:
            raise ValueError("amount_arg_path is required when max_amount_usd_per_call is set.")
        return self


class ApprovalGatePolicy(BaseModel):
    mode: Literal["human", "critic"]
    critic_model: str | None = None
    approval_ttl_seconds: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_mode(self) -> "ApprovalGatePolicy":
        if self.mode == "critic" and self.critic_model is None:
            raise ValueError("critic_model is required for critic approval mode.")
        return self


class InvariantRule(BaseModel):
    type: Literal[
        "required_arg_true",
        "email_domain_allowlist",
        "recipient_must_be_verified",
        "custom_json_rule",
    ]
    field: str | None = None
    allowed_domains: tuple[str, ...] = Field(default_factory=tuple)
    recipient_arg_path: str | None = None
    verification_summary_type: str = "policy::recipient_verification"
    json_path: str | None = None
    operator: Literal["eq", "ne", "gt", "gte", "lt", "lte", "in", "not_in"] | None = None
    value_json: str | None = None

    @model_validator(mode="after")
    def validate_rule_shape(self) -> "InvariantRule":
        if self.type == "required_arg_true" and self.field is None:
            raise ValueError("required_arg_true invariant requires field.")
        if self.type == "email_domain_allowlist":
            if self.field is None:
                raise ValueError("email_domain_allowlist invariant requires field.")
            if len(self.allowed_domains) == 0:
                raise ValueError("email_domain_allowlist invariant requires allowed_domains.")
        if self.type == "recipient_must_be_verified" and self.recipient_arg_path is None:
            raise ValueError("recipient_must_be_verified invariant requires recipient_arg_path.")
        if self.type == "custom_json_rule":
            if self.json_path is None or self.operator is None:
                raise ValueError("custom_json_rule invariant requires json_path and operator.")
            if self.value_json is None:
                raise ValueError("custom_json_rule invariant requires value_json.")
        return self


class ToolSafetyPolicy(BaseModel):
    intent: IntentRequirement | None = None
    semantic_idempotency: SemanticIdempotencyRequirement | None = None
    limits: ToolLimitPolicy | None = None
    approval: ApprovalGatePolicy | None = None
    invariants: tuple[InvariantRule, ...] = Field(default_factory=tuple)


class SafetyPolicyConfig(BaseModel):
    tools: dict[str, ToolSafetyPolicy] = Field(default_factory=dict)

    def policy_for(self, tool_name: str) -> ToolSafetyPolicy | None:
        return self.tools.get(tool_name)


class IntentPlanRecord(BaseModel):
    intent_id: str = Field(min_length=1)
    goal: str = Field(min_length=1)
    why: str = Field(min_length=1)
    success_criteria: str = Field(min_length=1)
    assumed_state: str = Field(min_length=1)
    applies_to_tools: tuple[str, ...] = Field(default_factory=tuple)
    created_at: datetime = Field(default_factory=utc_now)

