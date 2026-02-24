import asyncio

from pydantic import BaseModel

from artana.agent import SingleStepModelClient
from artana.kernel import ArtanaKernel, KernelPolicy
from artana.middleware import SafetyPolicyMiddleware
from artana.models import TenantContext
from artana.ports.model import ModelRequest, ModelResult, ModelUsage
from artana.safety import (
    IntentRequirement,
    SafetyPolicyConfig,
    SemanticIdempotencyRequirement,
    ToolLimitPolicy,
    ToolSafetyPolicy,
)
from artana.store import SQLiteStore


class Decision(BaseModel):
    ok: bool


class DemoModelPort:
    async def complete(self, request: ModelRequest[Decision]) -> ModelResult[Decision]:
        return ModelResult(
            output=Decision(ok=True),
            usage=ModelUsage(prompt_tokens=3, completion_tokens=2, cost_usd=0.0),
        )


async def main():
    safety = SafetyPolicyMiddleware(
        config=SafetyPolicyConfig(
            tools={
                "transfer_funds": ToolSafetyPolicy(
                    intent=IntentRequirement(require_intent=True),
                    semantic_idempotency=SemanticIdempotencyRequirement(
                        template="transfer:{tenant_id}:{account_id}:{amount}",
                        required_fields=("account_id", "amount"),
                    ),
                    limits=ToolLimitPolicy(
                        max_calls_per_run=3,
                        max_amount_usd_per_call=500.0,
                        amount_arg_path="amount",
                    ),
                )
            }
        )
    )

    kernel = ArtanaKernel(
        store=SQLiteStore("chapter4_step1.db"),
        model_port=DemoModelPort(),
        middleware=ArtanaKernel.default_middleware_stack(safety=safety),
        policy=KernelPolicy.enforced_v2(),
    )

    tenant = TenantContext(
        tenant_id="enterprise_user",
        capabilities=frozenset(),
        budget_usd_limit=1.0,
    )

    client = SingleStepModelClient(kernel=kernel)

    result = await client.step(
        run_id="enforced_run",
        tenant=tenant,
        model="demo-model",
        prompt="Verify policy enforcement.",
        output_schema=Decision,
        step_key="policy_step",
    )

    print(result.output)
    await kernel.close()


asyncio.run(main())