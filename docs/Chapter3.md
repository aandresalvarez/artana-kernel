 
# Chapter 3: Production Mode (Resilience, Drift, Supervision, and Scale)

This chapter focuses on:

* Two-phase tool safety
* Drift-aware replay
* Long-running harness recovery
* Hybrid deterministic + LLM workflows
* Progressive skills under discipline
* Adapter portability
* Ledger integrity

## Chapter Metadata

- Audience: Engineers deploying Artana in failure-prone or evolving environments.
- Prerequisites: Chapters 1–2 complete; familiarity with harness lifecycle and replay basics.
- Estimated time: 40 minutes.
- Expected outcome: You can choose replay policies intentionally, recover from tool uncertainty, and design crash-safe long-running runs.

Code block contract for this chapter:

* `python` blocks are standalone runnable scripts.
* `pycon` blocks are in-context snippets and may assume existing kernel/store/runtime state.

---

# Step 1 — Two-Phase Tool Safety + Reconciliation (Real-World Failure)

This pattern protects against:

* Network failures
* Unknown provider outcomes
* Duplicate execution

```python
import asyncio
import json

from pydantic import BaseModel

from artana.kernel import ArtanaKernel, ToolExecutionFailedError
from artana.models import TenantContext
from artana.ports.model import ModelRequest, ModelResult, ModelUsage
from artana.ports.tool import ToolExecutionContext, ToolUnknownOutcomeError
from artana.store import SQLiteStore


class NoopOutput(BaseModel):
    ok: bool


class NoopModelPort:
    async def complete(self, request: ModelRequest[NoopOutput]) -> ModelResult[NoopOutput]:
        return ModelResult(
            output=NoopOutput(ok=True),
            usage=ModelUsage(prompt_tokens=1, completion_tokens=1, cost_usd=0.0),
        )


class ChargeArgs(BaseModel):
    amount_cents: int
    card_id: str


async def main():
    kernel = ArtanaKernel(
        store=SQLiteStore("chapter3_step1.db"),
        model_port=NoopModelPort(),
    )

    provider_state = {"first_attempt": True}

    @kernel.tool(requires_capability="payments:charge")
    async def charge_credit_card(
        amount_cents: int,
        card_id: str,
        artana_context: ToolExecutionContext,
    ) -> str:
        if provider_state["first_attempt"]:
            provider_state["first_attempt"] = False
            raise ToolUnknownOutcomeError("network timeout after provider accepted charge")

        return json.dumps({
            "status": "charged",
            "idempotency_key": artana_context.idempotency_key,
        })

    tenant = TenantContext(
        tenant_id="billing_team",
        capabilities=frozenset({"payments:charge"}),
        budget_usd_limit=5.0,
    )

    await kernel.start_run(tenant=tenant, run_id="payment_run")

    args = ChargeArgs(amount_cents=1000, card_id="card_123")

    try:
        await kernel.step_tool(
            run_id="payment_run",
            tenant=tenant,
            tool_name="charge_credit_card",
            arguments=args,
            step_key="charge_step",
        )
    except ToolExecutionFailedError:
        print("Reconciliation required")

    result = await kernel.reconcile_tool(
        run_id="payment_run",
        tenant=tenant,
        tool_name="charge_credit_card",
        arguments=args,
        step_key="charge_step",
    )

    print("Reconciled:", result)
    await kernel.close()


asyncio.run(main())
```

This is safe, idempotent, replayable, and production-grade.

---

# Step 2 — Drift-Aware Replay in Long-Running Systems

In production, prompts evolve.

ReplayPolicy allows safe evolution.

Snippet (in-context, not standalone):

```pycon
from artana.kernel import ReplayPolicy
from artana.ports.model import ModelCallOptions

# Strict replay (default safety)
await kernel.step_model(
    ...,
    model_options=ModelCallOptions(api_mode="auto", verbosity="medium"),
    replay_policy="strict",
)

# Allow prompt drift while preserving prior outputs
await kernel.step_model(
    ...,
    model_options=ModelCallOptions(api_mode="responses", previous_response_id="resp_prev"),
    replay_policy="allow_prompt_drift",
)

# Fork run automatically if model inputs/options change
await kernel.step_model(
    ...,
    model_options=ModelCallOptions(api_mode="auto"),
    replay_policy="fork_on_drift",
)
```

Drift matching includes prompt/messages plus model options (api mode, reasoning, verbosity, previous response chain) and Responses input items.

Production guidance:

| Scenario              | Replay Mode        |
| --------------------- | ------------------ |
| Regulated finance     | strict             |
| Iterative product dev | allow_prompt_drift |
| Experimental research | fork_on_drift      |

---

# Step 3 — Long-Running Recovery with Incremental Harness

Production systems must survive crashes mid-run.

```python
import asyncio

from artana import (
    ArtanaKernel,
    IncrementalTaskHarness,
    MockModelPort,
    SQLiteStore,
    TaskUnit,
    TenantContext,
)


class MigrationHarness(IncrementalTaskHarness):

    async def define_tasks(self):
        return [
            TaskUnit(id="backup", description="Backup DB"),
            TaskUnit(id="migrate", description="Run migrations"),
            TaskUnit(id="verify", description="Verify schema"),
        ]

    async def work_on(self, task: TaskUnit):
        print("Executing:", task.id)


async def main():
    kernel = ArtanaKernel(
        store=SQLiteStore("chapter3_step3.db"),
        model_port=MockModelPort(output={"message": "unused"}),
    )

    tenant = TenantContext(
        tenant_id="ops",
        capabilities=frozenset(),
        budget_usd_limit=5.0,
    )

    harness = MigrationHarness(kernel=kernel, tenant=tenant)

    await harness.run("migration_run")

    # Simulate restart:
    await harness.run("migration_run")

    await kernel.close()


asyncio.run(main())
```

If the process crashes mid-task:

* Task state remains persisted
* Partial transitions rejected
* Clean-state validation enforced

---

# Step 4 — Hybrid AI + Deterministic Workflow (Safe Orchestration)

Production systems mix:

* Deterministic Python
* LLM reasoning
* Checkpointed workflow steps

```python
import asyncio

from pydantic import BaseModel

from artana.kernel import ArtanaKernel, WorkflowContext, json_step_serde
from artana.agent import SingleStepModelClient
from artana.models import TenantContext
from artana.ports.model import ModelRequest, ModelResult, ModelUsage
from artana.store import SQLiteStore


class Intent(BaseModel):
    question: str


class Email(BaseModel):
    body: str


class HybridModel:
    async def complete(self, request: ModelRequest[BaseModel]) -> ModelResult[BaseModel]:
        if "question" in request.output_schema.model_fields:
            output = request.output_schema.model_validate({"question": "What is revenue?"})
        else:
            output = request.output_schema.model_validate({"body": "Revenue is $8.3M."})
        return ModelResult(
            output=output,
            usage=ModelUsage(prompt_tokens=5, completion_tokens=5, cost_usd=0.0),
        )


async def heavy_math():
    return {"revenue": 8300000}


async def main():
    kernel = ArtanaKernel(
        store=SQLiteStore("chapter3_step4.db"),
        model_port=HybridModel(),
    )

    tenant = TenantContext(
        tenant_id="finance",
        capabilities=frozenset(),
        budget_usd_limit=5.0,
    )

    client = SingleStepModelClient(kernel)

    async def workflow(ctx: WorkflowContext):
        intent = await client.step(
            run_id=ctx.run_id,
            tenant=ctx.tenant,
            model="demo-model",
            prompt="Extract intent",
            output_schema=Intent,
            step_key="intent",
        )

        math = await ctx.step(
            name="compute",
            action=heavy_math,
            serde=json_step_serde(),
        )

        email = await client.step(
            run_id=ctx.run_id,
            tenant=ctx.tenant,
            model="demo-model",
            prompt=f"{intent.output.question}. Revenue: {math['revenue']}",
            output_schema=Email,
            step_key="email",
        )

        return email.output.body

    result = await kernel.run_workflow(
        run_id="hybrid_run",
        tenant=tenant,
        workflow=workflow,
    )

    print(result.output)
    await kernel.close()


asyncio.run(main())
```

Deterministic + AI + replay = production-safe orchestration.

---

# Step 5 — Production Middleware (Enforced Modes)

Baseline production enforcement:

Snippet (in-context, not standalone):

```pycon
from artana.kernel import KernelPolicy
from artana.middleware import (
    PIIScrubberMiddleware,
    QuotaMiddleware,
    CapabilityGuardMiddleware,
)

kernel = ArtanaKernel(
    store=SQLiteStore("prod.db"),
    model_port=HybridModel(),
    middleware=[
        PIIScrubberMiddleware(),
        QuotaMiddleware(),
        CapabilityGuardMiddleware(),
    ],
    policy=KernelPolicy.enforced(),
)
```

`KernelPolicy.enforced()` requires:

* PII scrubber
* Quota middleware
* Capability guard

For side-effect-heavy production systems, prefer `KernelPolicy.enforced_v2()` and add
`SafetyPolicyMiddleware` to activate intent plans, semantic idempotency, limits, approvals,
and invariants.

This prevents unsafe deployments.

---

# Step 6 — Progressive Skills Under Discipline

Progressive skills allow dynamic tool exposure.

Snippet (in-context, not standalone):

```pycon
from artana.agent import AutonomousAgent

agent = AutonomousAgent(kernel=kernel)

# load_skill() must be called before using certain tools
```

Production tip:

* Combine progressive skills with capability guard
* Require explicit capability for high-risk tools

---

# Step 7 — Ledger Integrity + Audit

Every run is verifiable:

Snippet (in-context, not standalone):

```pycon
from artana.store import SQLiteStore

events = await kernel.get_events(run_id="migration_run")

for event in events:
    print(event.seq, event.event_type)

store = SQLiteStore("chapter3_step3.db")
valid = await store.verify_run_chain("migration_run")
print("Ledger valid:", valid)
await store.close()
```

Production uses:

* Cost aggregation
* Summary inspection
* Drift detection events
* Forked run tracking

---

# Step 8 — Adapter Swap (SQLite → PostgresStore)

Production swaps store implementation, not business logic.

Snippet (in-context, not standalone):

```pycon
from artana.store import PostgresStore

store = PostgresStore(
    dsn="postgresql://user:pass@db:5432/artana",
    min_pool_size=2,
    max_pool_size=20,
)
```

Kernel logic remains identical.

Only the persistence backend changes.

---

# Production Principles Summary

| Principle               | Artana Mechanism           |
| ----------------------- | -------------------------- |
| Idempotent side effects | step_tool + reconcile_tool |
| Crash safety            | WorkflowContext            |
| Long-running discipline | IncrementalTaskHarness     |
| Drift control           | ReplayPolicy               |
| Policy enforcement      | KernelPolicy.enforced_v2   |
| Audit ledger            | verify_run_chain           |
| Structured continuity   | artifacts + summaries      |
| Safe scaling            | SupervisorHarness          |

---

# Final Production Mental Model

Production Artana systems should:

* Use Harness for long-running tasks
* Use Workflow for deterministic orchestration
* Use enforced middleware (`enforced_v2` for OS-grade safety)
* Use replay policies intentionally
* Store structured artifacts
* Validate clean state before sleep
* Audit ledger integrity

## You Should Now Be Able To

- Recover safely from unknown tool outcomes without duplicating side effects.
- Pick the correct replay policy (`strict`, `allow_prompt_drift`, or `fork_on_drift`) for your environment.
- Resume long-running harness and workflow runs after interruptions with deterministic state.

## Next Chapter

Continue to [Chapter 4: Ultimate Architecture](./Chapter4.md) for advanced orchestration, custom loops, and platform-level composition patterns.

 
