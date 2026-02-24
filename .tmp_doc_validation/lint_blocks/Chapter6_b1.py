from artana.kernel import ArtanaKernel, KernelPolicy
from artana.middleware import SafetyPolicyMiddleware
from artana.safety import (
    IntentRequirement,
    SafetyPolicyConfig,
    SemanticIdempotencyRequirement,
    ToolLimitPolicy,
    ToolSafetyPolicy,
)
from artana.store import SQLiteStore

safety = SafetyPolicyMiddleware(
    config=SafetyPolicyConfig(
        tools={
            "send_invoice": ToolSafetyPolicy(
                intent=IntentRequirement(require_intent=True, max_age_seconds=3600),
                semantic_idempotency=SemanticIdempotencyRequirement(
                    template="send_invoice:{tenant_id}:{billing_period}",
                    required_fields=("billing_period",),
                ),
                limits=ToolLimitPolicy(
                    max_calls_per_run=2,
                    max_calls_per_tenant_window=5,
                    tenant_window_seconds=3600,
                    max_amount_usd_per_call=500.0,
                    amount_arg_path="amount_usd",
                ),
            )
        }
    )
)

kernel = ArtanaKernel(
    store=SQLiteStore("chapter6_safety.db"),
    model_port=DemoModelPort(),  # your ModelPort implementation
    middleware=ArtanaKernel.default_middleware_stack(safety=safety),
    policy=KernelPolicy.enforced_v2(),
)