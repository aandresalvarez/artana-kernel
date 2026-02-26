 
# Chapter 4: Ultimate Architecture

(Custom Loops, Strict Policies, Drift Control, and Platform Orchestration)

This chapter demonstrates:

* Enforced enterprise policy
* Custom bare-metal execution loops
* Drift-aware evolution
* Supervisor-level orchestration
* External orchestrator integration
* Ledger observability at scale

Code block contract for this chapter:

* `python` blocks are standalone runnable scripts.
* `pycon` blocks are in-context snippets and may require surrounding setup.

---

# Step 1 — Enforced Enterprise Kernel (Mandatory Middleware)

For OS-grade safety controls, production systems should use `KernelPolicy.enforced_v2()`.

```python
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
```

In `enforced_v2` mode:

* Missing middleware = kernel initialization error
* Tool IO hooks required
* Budget and capability checks mandatory
* Safety policy middleware mandatory

---

# Step 2 — Bare-Metal Custom Loop (Direct Kernel Control)

Sometimes you need full control.

This pattern bypasses AutonomousAgent and Harness entirely.

```python
import asyncio
import json

from pydantic import BaseModel

from artana.events import ChatMessage
from artana.kernel import ArtanaKernel, ModelInput
from artana.models import TenantContext
from artana.ports.model import (
    ModelCallOptions,
    ModelRequest,
    ModelResult,
    ModelUsage,
    ToolCall,
)
from artana.store import SQLiteStore


class DebateResponse(BaseModel):
    text: str


class StoreArgumentArgs(BaseModel):
    value: str


class DebateModelPort:
    async def complete(self, request: ModelRequest[DebateResponse]) -> ModelResult[DebateResponse]:
        last = request.messages[-1].content
        output = request.output_schema.model_validate({"text": f"Reply to: {last}"})

        tool_calls = ()
        if "store this" in last.lower():
            tool_calls = (
                ToolCall(
                    tool_name="store_argument",
                    arguments_json='{"value":"important"}',
                    tool_call_id="call_1",
                ),
            )

        return ModelResult(
            output=output,
            usage=ModelUsage(prompt_tokens=10, completion_tokens=5, cost_usd=0.0),
            tool_calls=tool_calls,
        )


async def main():
    kernel = ArtanaKernel(
        store=SQLiteStore("chapter4_step2.db"),
        model_port=DebateModelPort(),
    )

    @kernel.tool()
    async def store_argument(value: str) -> str:
        return json.dumps({"stored": value})

    tenant = TenantContext(
        tenant_id="research",
        capabilities=frozenset(),
        budget_usd_limit=5.0,
    )

    run_id = "debate_run"
    await kernel.start_run(tenant=tenant, run_id=run_id)

    transcript = [ChatMessage(role="system", content="You are debating.")]

    result = await kernel.step_model(
        run_id=run_id,
        tenant=tenant,
        model="demo-model",
        input=ModelInput.from_messages(
            transcript + [ChatMessage(role="user", content="Store this idea")]
        ),
        output_schema=DebateResponse,
        model_options=ModelCallOptions(api_mode="auto"),
        step_key="turn_1",
    )

    for tool in result.tool_calls:
        tool_result = await kernel.step_tool(
            run_id=run_id,
            tenant=tenant,
            tool_name=tool.tool_name,
            arguments=StoreArgumentArgs(value="important"),
            step_key="tool_1",
        )
        print(tool_result.result_json)

    await kernel.close()


asyncio.run(main())
```

`ModelCallOptions(api_mode="auto")` is the standard baseline for model calls in kernel loops.

Use this pattern when:

* Building custom reasoning loops
* Mixing multiple models
* Building research-grade experimental systems

---

# Step 3 — Drift-Aware Evolution (Fork-On-Drift)

Long-lived systems evolve.

Artana supports controlled run forking.

Snippet (in-context, not standalone):

```pycon
from artana.kernel import ModelInput
from artana.ports.model import ModelCallOptions

result = await kernel.step_model(
    run_id="long_run",
    tenant=tenant,
    model="demo-model",
    input=ModelInput.from_prompt("New improved prompt"),
    output_schema=Decision,
    model_options=ModelCallOptions(api_mode="auto", verbosity="medium"),
    step_key="analysis_step",
    replay_policy="fork_on_drift",
)
```

If model inputs/options change:

* Original run remains immutable
* New forked run is created
* REPLAYED_WITH_DRIFT event is recorded

This enables:

* Versioned experiments
* Controlled upgrades
* Safe prompt refactoring

---

# Step 4 — Harness + Supervisor (Platform-Level Orchestration)

Production systems should coordinate harnesses, not raw agents.

```python
import asyncio
from artana.harness import IncrementalTaskHarness, SupervisorHarness, TaskUnit
from artana.kernel import ArtanaKernel
from artana.models import TenantContext
from artana.store import SQLiteStore


class DeploymentHarness(IncrementalTaskHarness):

    async def define_tasks(self):
        return [
            TaskUnit(id="build", description="Build artifacts"),
            TaskUnit(id="deploy", description="Deploy services"),
        ]

    async def work_on(self, task: TaskUnit):
        print("Executing:", task.id)


class DeploymentSupervisor(SupervisorHarness):
    async def step(self, *, context):
        deployment = DeploymentHarness(kernel=self.kernel, tenant=context.tenant)
        return await self.run_child(
            harness=deployment,
            run_id=f"{context.run_id}::deployment",
            tenant=context.tenant,
            model=context.model,
        )


async def main():
    kernel = ArtanaKernel(
        store=SQLiteStore("chapter4_step4.db"),
        model_port=None,
    )

    tenant = TenantContext(
        tenant_id="ops",
        capabilities=frozenset(),
        budget_usd_limit=5.0,
    )

    supervisor = DeploymentSupervisor(kernel=kernel, tenant=tenant)
    result = await supervisor.run(run_id="deployment_run")

    print("Deployment state:", result)
    await kernel.close()


asyncio.run(main())
```

This gives:

* Clean-state enforcement
* Incremental discipline
* Replay-safe orchestration
* Multi-harness composition

---

# Step 5 — External Orchestrator Integration (Temporal-Style)

Artana is orchestration-agnostic.

Example: integrating with an external scheduler.

```python
import asyncio
from pydantic import BaseModel

from artana.kernel import ArtanaKernel, ModelInput
from artana.models import TenantContext
from artana.ports.model import (
    ModelCallOptions,
    ModelRequest,
    ModelResult,
    ModelUsage,
)
from artana.store import SQLiteStore


class Report(BaseModel):
    summary: str


class DemoModelPort:
    async def complete(self, request: ModelRequest[Report]) -> ModelResult[Report]:
        return ModelResult(
            output=Report(summary="generated report"),
            usage=ModelUsage(prompt_tokens=4, completion_tokens=3, cost_usd=0.0),
        )


async def generate_report(workflow_id: str, account_id: str) -> str:
    kernel = ArtanaKernel(
        store=SQLiteStore("chapter4_step5.db"),
        model_port=DemoModelPort(),
    )

    tenant = TenantContext(
        tenant_id=account_id,
        capabilities=frozenset(),
        budget_usd_limit=5.0,
    )

    await kernel.start_run(tenant=tenant, run_id=workflow_id)

    result = await kernel.step_model(
        run_id=workflow_id,
        tenant=tenant,
        model="demo-model",
        input=ModelInput.from_prompt(f"Generate report for {account_id}"),
        output_schema=Report,
        model_options=ModelCallOptions(api_mode="auto"),
        step_key="report_step",
    )

    await kernel.close()
    return result.output.summary


async def main():
    print(await generate_report("workflow_123", "acct_42"))


asyncio.run(main())
```

Key property:

External orchestrator manages scheduling.
Artana manages durable execution and replay.

---

# Step 6 — Ledger-Level Observability at Scale

Artana’s event store is a queryable audit log.

Snippet (in-context, not standalone; assumes an existing SQLite event DB):

```pycon
import sqlite3

connection = sqlite3.connect("chapter4_step1.db")

rows = connection.execute(
    """
    SELECT
        tenant_id,
        SUM(CAST(json_extract(payload_json, '$.cost_usd') AS FLOAT)) AS total_spend,
        COUNT(*) AS model_calls
    FROM kernel_events
    WHERE event_type = 'model_completed'
    GROUP BY tenant_id
    ORDER BY total_spend DESC
    """
).fetchall()

for row in rows:
    print(row)

connection.close()
```

This enables:

* Cost dashboards
* Drift detection
* Replay audits
* Capability decision tracking
* Regulatory reporting

---

# Final Architecture Summary

Production Artana systems combine:

| Layer        | Role                              |
| ------------ | --------------------------------- |
| Kernel       | Deterministic execution OS        |
| Harness      | Structured incremental discipline |
| Supervisor   | Multi-agent orchestration         |
| Workflow     | Crash-proof deterministic steps   |
| Middleware   | Security + budget enforcement     |
| ReplayPolicy | Evolution control                 |
| Ledger       | Immutable audit trail             |

---
 
