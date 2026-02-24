# 📂 Project File Contents

- 📁 **docs**
  - [`Chapter1.md`](#docs-chapter1md)
  - [`Chapter2.md`](#docs-chapter2md)
  - [`Chapter3.md`](#docs-chapter3md)
  - [`Chapter4.md`](#docs-chapter4md)
  - [`Chapter5.md`](#docs-chapter5md)
  - [`Chapter6.md`](#docs-chapter6md)
  - [`deep_traceability.md`](#docs-deep_traceabilitymd)
  - [`kernel_contracts.md`](#docs-kernel_contractsmd)

---

# Target Folder: docs

## Folder: docs

### File: `docs/Chapter1.md`
<a name="docs-chapter1md"></a>
```markdown
 

# 🚀 The Complete Beginner’s Guide to Artana (Modern Edition)

Artana is built in **three layers**:

1. **Kernel** → Durable execution OS (replay-safe, crash-proof)
2. **Agent** → Multi-turn intelligent reasoning
3. **Harness** → Structured long-running discipline

This guide walks you from deterministic steps → tools → workflows → agents → harnesses.

All examples are runnable.

---

# 🧠 Step 1 — Deterministic Model Steps (The Kernel)

Every model step in Artana:

* Is persisted
* Is replay-safe
* Requires a `step_key`
* Can be resumed safely

```python
import asyncio
from typing import TypeVar

from pydantic import BaseModel

from artana.agent import SingleStepModelClient
from artana.kernel import ArtanaKernel
from artana.models import TenantContext
from artana.ports.model import ModelRequest, ModelResult, ModelUsage
from artana.store import SQLiteStore

OutputT = TypeVar("OutputT", bound=BaseModel)


class HelloResult(BaseModel):
    message: str


class DemoModelPort:
    async def complete(self, request: ModelRequest[OutputT]) -> ModelResult[OutputT]:
        output = request.output_schema.model_validate(
            {"message": "Hello from Artana!"}
        )
        return ModelResult(
            output=output,
            usage=ModelUsage(prompt_tokens=5, completion_tokens=5, cost_usd=0.0),
        )


async def main():
    kernel = ArtanaKernel(
        store=SQLiteStore("step1.db"),
        model_port=DemoModelPort(),
    )

    tenant = TenantContext(
        tenant_id="demo_user",
        capabilities=frozenset(),
        budget_usd_limit=1.0,
    )

    client = SingleStepModelClient(kernel=kernel)

    result = await client.step(
        run_id="hello_run",
        tenant=tenant,
        model="demo-model",
        prompt="Say hello",
        output_schema=HelloResult,
        step_key="hello_step",  # 🔑 required for replay safety
    )

    print(result.output)
    await kernel.close()


asyncio.run(main())
```

🔑 **Important:**
`step_key` ensures deterministic replay.
Never reuse a step_key for different logic.

---

# 🛠 Step 2 — Tools + Idempotency

Tools are durable and idempotent.

Every tool can receive:

```python
artana_context: ToolExecutionContext
```

Use `artana_context.idempotency_key` for safe retries.

```python
import asyncio
import json

from pydantic import BaseModel

from artana.kernel import ArtanaKernel
from artana.models import TenantContext
from artana.store import SQLiteStore
from artana.ports.model import ModelRequest, ModelResult, ModelUsage
from artana.ports.tool import ToolExecutionContext


class Decision(BaseModel):
    ok: bool


class DemoModelPort:
    async def complete(self, request: ModelRequest[Decision]) -> ModelResult[Decision]:
        return ModelResult(
            output=Decision(ok=True),
            usage=ModelUsage(prompt_tokens=1, completion_tokens=1, cost_usd=0.0),
        )


async def main():
    kernel = ArtanaKernel(
        store=SQLiteStore("step2.db"),
        model_port=DemoModelPort(),
    )

    @kernel.tool()
    async def transfer_money(
        amount: int,
        to_user: str,
        artana_context: ToolExecutionContext,
    ) -> str:
        return json.dumps({
            "amount": amount,
            "to_user": to_user,
            "idempotency_key": artana_context.idempotency_key
        })

    tenant = TenantContext(
        tenant_id="demo_user",
        capabilities=frozenset(),
        budget_usd_limit=1.0,
    )

    await kernel.start_run(tenant=tenant, run_id="tool_run")

    result = await kernel.step_tool(
        run_id="tool_run",
        tenant=tenant,
        tool_name="transfer_money",
        arguments=BaseModel.model_validate({"amount": 10, "to_user": "alice"}),
        step_key="transfer_step",
    )

    print(result.result_json)
    await kernel.close()


asyncio.run(main())
```

---

# 🔁 Step 3 — Crash-Proof Workflows

Workflows checkpoint each step automatically.

If the process crashes, you resume safely.

```python
import asyncio
from artana.kernel import ArtanaKernel, WorkflowContext
from artana.models import TenantContext
from artana.store import SQLiteStore


async def main():
    kernel = ArtanaKernel(
        store=SQLiteStore("workflow.db"),
        model_port=None,  # not needed here
    )

    tenant = TenantContext(
        tenant_id="workflow_user",
        capabilities=frozenset(),
        budget_usd_limit=1.0,
    )

    async def my_workflow(ctx: WorkflowContext):
        step1 = await ctx.step(
            name="compute_value",
            action=lambda: asyncio.sleep(0, result=42),
            serde=ctx.json_step_serde(),
        )

        if step1 == 42:
            await ctx.pause(reason="Confirm value before proceeding")

        return "Finished"

    first = await kernel.run_workflow(
        run_id="workflow_run",
        tenant=tenant,
        workflow=my_workflow,
    )

    print("status:", first.status)
    await kernel.close()


asyncio.run(main())
```

---

# 🤖 Step 4 — Autonomous Agent (Multi-Turn)

For multi-turn reasoning:

```python
import asyncio
from pydantic import BaseModel

from artana.agent import AutonomousAgent
from artana.kernel import ArtanaKernel
from artana.models import TenantContext
from artana.store import SQLiteStore


class Report(BaseModel):
    text: str


class DemoModelPort:
    async def complete(self, request):
        return type(request).output_schema.model_validate({"text": "Demo report"})


async def main():
    kernel = ArtanaKernel(
        store=SQLiteStore("agent.db"),
        model_port=DemoModelPort(),
    )

    tenant = TenantContext(
        tenant_id="agent_user",
        capabilities=frozenset(),
        budget_usd_limit=1.0,
    )

    agent = AutonomousAgent(kernel=kernel)

    result = await agent.run(
        run_id="agent_run",
        tenant=tenant,
        model="demo-model",
        prompt="Write a short report",
        output_schema=Report,
    )

    print(result.text)
    await kernel.close()


asyncio.run(main())
```

Use AutonomousAgent for short-running or exploratory reasoning.

---

# 🏗 Step 5 — Harnesses (Long-Running Structured Agents)

Harnesses are for **long-running structured work**.

They enforce:

* Incremental progress
* One task completion per session
* Clean state
* Structured summaries

```python
import asyncio
from artana.harness import IncrementalTaskHarness, TaskUnit
from artana.kernel import ArtanaKernel
from artana.models import TenantContext
from artana.store import SQLiteStore


class ResearchHarness(IncrementalTaskHarness):

    async def define_tasks(self):
        return [
            TaskUnit(id="collect", description="Collect data"),
            TaskUnit(id="analyze", description="Analyze data"),
            TaskUnit(id="summarize", description="Write summary"),
        ]

    async def work_on(self, task: TaskUnit):
        print("Working on:", task.id)


async def main():
    kernel = ArtanaKernel(
        store=SQLiteStore("harness.db"),
        model_port=None,
    )

    tenant = TenantContext(
        tenant_id="research_team",
        capabilities=frozenset(),
        budget_usd_limit=1.0,
    )

    harness = ResearchHarness(kernel=kernel, tenant=tenant)

    progress = await harness.run("research_run")

    print("Task states:", progress)
    await kernel.close()


asyncio.run(main())
```

Harnesses automatically:

* Persist task progress
* Prevent multiple DONE transitions per session
* Enforce clean state before sleep

---

# 🗂 Step 6 — Artifacts (Structured Continuity)

Artifacts store structured durable state.

```python
await harness.set_artifact(key="plan", value={"phase": 1})
plan = await harness.get_artifact(key="plan")
print(plan)
```

Artifacts are stored as structured run summaries.

---

# 🧭 Step 7 — Supervisor Harness (Multi-Agent)

Compose harnesses safely.

```python
from artana.harness import SupervisorHarness

supervisor = SupervisorHarness(kernel)

result = await supervisor.run_child(
    harness=ResearchHarness(kernel),
    run_id="child_run"
)
```

---

# 🏁 Final Mental Model

| Layer           | Purpose                            |
| --------------- | ---------------------------------- |
| Kernel          | Durable execution, replay safety   |
| Workflow        | Crash-proof orchestration          |
| AutonomousAgent | Multi-turn reasoning               |
| Harness         | Structured long-running discipline |

---

# 🧠 Key Principles

* Always use stable `step_key`
* Tools must be idempotent
* Harness enforces discipline
* Replay modes allow evolution
* Artifacts store structured continuity

 
```

### File: `docs/Chapter2.md`
<a name="docs-chapter2md"></a>
```markdown
 
# Chapter 2: Scaling Up (Harnesses, Supervision, and Production Discipline)

This chapter focuses on production patterns:

* Multi-agent orchestration
* Long-running incremental harnesses
* Structured artifacts
* Replay modes
* Middleware enforcement
* Ledger & observability

All examples are runnable and reflect the current API.

---

# Step 1 — Structured Multi-Agent Supervision (Harness-Based Swarms)

Instead of directly spawning subagents from tools, modern Artana prefers **SupervisorHarness**.

```python
import asyncio
from pydantic import BaseModel

from artana.harness import IncrementalTaskHarness, SupervisorHarness, TaskUnit
from artana.kernel import ArtanaKernel
from artana.models import TenantContext
from artana.store import SQLiteStore


class ResearchHarness(IncrementalTaskHarness):

    async def define_tasks(self):
        return [
            TaskUnit(id="fact", description="Provide a historical fact"),
        ]

    async def work_on(self, task: TaskUnit):
        print("Research task executed:", task.id)


async def main():
    kernel = ArtanaKernel(
        store=SQLiteStore("chapter2_step1.db"),
        model_port=None,
    )

    tenant = TenantContext(
        tenant_id="manager",
        capabilities=frozenset(),
        budget_usd_limit=5.0,
    )

    supervisor = SupervisorHarness(kernel=kernel, tenant=tenant)
    child_harness = ResearchHarness(kernel=kernel, tenant=tenant)

    result = await supervisor.run_child(
        harness=child_harness,
        run_id="swarm_run_01"
    )

    print("Child task states:", result)
    await kernel.close()


asyncio.run(main())
```

🔎 Why this is better:

* Supervisor controls structure.
* Child harness enforces incremental discipline.
* Kernel guarantees replay integrity.

---

# Step 2 — Long-Running Incremental Harness Discipline

This replaces ad-hoc autonomous loops with structured continuity.

```python
import asyncio
from artana.harness import IncrementalTaskHarness, TaskUnit
from artana.kernel import ArtanaKernel
from artana.models import TenantContext
from artana.store import SQLiteStore


class DataPipelineHarness(IncrementalTaskHarness):

    async def define_tasks(self):
        return [
            TaskUnit(id="ingest", description="Ingest data"),
            TaskUnit(id="transform", description="Transform data"),
            TaskUnit(id="validate", description="Validate results"),
        ]

    async def work_on(self, task: TaskUnit):
        print("Executing:", task.id)


async def main():
    kernel = ArtanaKernel(
        store=SQLiteStore("chapter2_step2.db"),
        model_port=None,
    )

    tenant = TenantContext(
        tenant_id="pipeline_team",
        capabilities=frozenset(),
        budget_usd_limit=5.0,
    )

    harness = DataPipelineHarness(kernel=kernel, tenant=tenant)

    progress = await harness.run("pipeline_run_001")
    print("Progress snapshot:", progress)

    await kernel.close()


asyncio.run(main())
```

What this enforces:

* Only one task → DONE per session
* No deletion of tasks
* Clean state before sleep
* Structured task_progress summary

This is how you scale agents safely across days.

---

# Step 3 — Replay Modes (Production Evolution)

Long-running agents evolve. Prompts change. Policies update.

Artana supports safe replay policies.

```python
from artana.kernel import ReplayPolicy

harness = DataPipelineHarness(
    kernel=kernel,
    tenant=tenant,
)

# Strict mode (default safety)
await harness.run("run_strict")

# Allow minor prompt drift
harness = DataPipelineHarness(
    kernel=kernel,
    tenant=tenant,
    replay_policy="allow_prompt_drift",
)

await harness.run("run_drift_safe")
```

Replay modes:

| Mode               | Behavior                         |
| ------------------ | -------------------------------- |
| strict             | Fail if prompt changes           |
| allow_prompt_drift | Replay safely with drift summary |
| fork_on_drift      | Fork run if logic changed        |

This is critical for long-lived systems.

---

# Step 4 — Structured Artifacts (Durable State)

Artifacts allow structured continuity across sessions.

```python
await harness.set_artifact(key="schema_version", value={"v": 2})
schema = await harness.get_artifact(key="schema_version")
print("Schema:", schema)
```

Artifacts are stored as structured run summaries:

* `artifact::<key>`
* Immutable history
* Latest snapshot retrievable in O(1)

Use artifacts for:

* Plans
* Schemas
* Maps
* Checkpoints
* External system IDs

---

# Step 5 — Middleware Enforcement (Security + Budget + Custom Rules)

You can layer custom policies safely.

```python
from artana.middleware import (
    PIIScrubberMiddleware,
    QuotaMiddleware,
    CapabilityGuardMiddleware,
)
from artana.middleware.base import KernelMiddleware, ModelInvocation


class BlockKeywordMiddleware(KernelMiddleware):

    async def prepare_model(self, invocation: ModelInvocation):
        if "forbidden" in invocation.prompt.lower():
            raise ValueError("Blocked keyword detected.")
        return invocation

    async def before_model(self, **kwargs): return None
    async def after_model(self, **kwargs): return None
    async def prepare_tool_request(self, **kwargs): return kwargs["arguments_json"]
    async def prepare_tool_result(self, **kwargs): return kwargs["result_json"]


kernel = ArtanaKernel(
    store=SQLiteStore("chapter2_step5.db"),
    model_port=DemoModelPort(),
    middleware=[
        PIIScrubberMiddleware(),
        QuotaMiddleware(),
        CapabilityGuardMiddleware(),
        BlockKeywordMiddleware(),
    ],
)
```

Order is enforced automatically:

1. PII scrub
2. Quota
3. Capability guard
4. Safety policy (if configured)
5. Custom middleware

---

# Step 6 — Audit Ledger (Immutable Event Log)

Every run is a verifiable ledger.

```python
events = await kernel.get_events(run_id="pipeline_run_001")

for event in events:
    print(event.seq, event.event_type)

verified = await kernel._store.verify_run_chain("pipeline_run_001")
print("Chain valid:", verified)
```

You can audit:

* Model usage
* Tool calls
* Cost aggregation
* Drift events
* Replay forks
* Summaries

This makes Artana suitable for regulated environments.

---

# Step 7 — Observability Tool (`query_event_history`)

Autonomous agents can inspect themselves.

```python
# query_event_history is automatically registered
# when AutonomousAgent is used

# The agent can call:
# query_event_history(limit=10, event_type="all")
```

This enables:

* Self-debugging
* Self-reflection
* Drift awareness
* Recovery reasoning

---

# Chapter 2 Summary

In production, you should:

* Use Harness for long-running tasks
* Use SupervisorHarness for orchestration
* Store structured artifacts
* Enforce incremental discipline
* Choose replay mode deliberately
* Layer middleware carefully
* Rely on immutable event ledger
* Use summaries instead of scanning full history
 

```

### File: `docs/Chapter3.md`
<a name="docs-chapter3md"></a>
```markdown
 
# Chapter 3: Production Mode (Resilience, Drift, Supervision, and Scale)

This chapter focuses on:

* Two-phase tool safety
* Drift-aware replay
* Long-running harness recovery
* Hybrid deterministic + LLM workflows
* Progressive skills under discipline
* Adapter portability
* Ledger integrity

All examples use current APIs and are copy-paste runnable.

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

```python
from artana.kernel import ReplayPolicy

# Strict replay (default safety)
await kernel.step_model(..., replay_policy="strict")

# Allow prompt drift while preserving prior outputs
await kernel.step_model(..., replay_policy="allow_prompt_drift")

# Fork run automatically if prompt changed
await kernel.step_model(..., replay_policy="fork_on_drift")
```

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
from artana.harness import IncrementalTaskHarness, TaskUnit
from artana.kernel import ArtanaKernel
from artana.models import TenantContext
from artana.store import SQLiteStore


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
        model_port=None,
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

    client = SingleStepModelClient(kernel=kernel)

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

# Step 5 — Production Middleware (Enforced Mode)

Production environments should enable enforcement mode:

```python
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

Enforced mode requires:

* PII scrubber
* Quota middleware
* Capability guard
* Tool IO hooks

This prevents unsafe deployments.

---

# Step 6 — Progressive Skills Under Discipline

Progressive skills allow dynamic tool exposure.

```python
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

```python
events = await kernel.get_events("migration_run")

for event in events:
    print(event.seq, event.event_type)

valid = await kernel._store.verify_run_chain("migration_run")
print("Ledger valid:", valid)
```

Production uses:

* Cost aggregation
* Summary inspection
* Drift detection events
* Forked run tracking

---

# Step 8 — Adapter Swap (SQLite → Postgres)

Production swaps store implementation, not business logic.

```python
class PostgresStore(SQLiteStore):
    """Production store adapter implementing EventStore interface."""
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
| Policy enforcement      | KernelPolicy.enforced      |
| Audit ledger            | verify_run_chain           |
| Structured continuity   | artifacts + summaries      |
| Safe scaling            | SupervisorHarness          |

---

# Final Production Mental Model

Production Artana systems should:

* Use Harness for long-running tasks
* Use Workflow for deterministic orchestration
* Use enforced middleware
* Use replay policies intentionally
* Store structured artifacts
* Validate clean state before sleep
* Audit ledger integrity

 
```

### File: `docs/Chapter4.md`
<a name="docs-chapter4md"></a>
```markdown
 
# Chapter 4: Ultimate Architecture

(Custom Loops, Strict Policies, Drift Control, and Platform Orchestration)

This chapter demonstrates:

* Enforced enterprise policy
* Custom bare-metal execution loops
* Drift-aware evolution
* Supervisor-level orchestration
* External orchestrator integration
* Ledger observability at scale

All examples are runnable.

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
from artana.ports.model import ModelRequest, ModelResult, ModelUsage, ToolCall
from artana.store import SQLiteStore


class DebateResponse(BaseModel):
    text: str


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
        step_key="turn_1",
    )

    for tool in result.tool_calls:
        tool_result = await kernel.step_tool(
            run_id=run_id,
            tenant=tenant,
            tool_name=tool.tool_name,
            arguments=BaseModel.model_validate({"value": "important"}),
            step_key="tool_1",
        )
        print(tool_result.result_json)

    await kernel.close()


asyncio.run(main())
```

Use this pattern when:

* Building custom reasoning loops
* Mixing multiple models
* Building research-grade experimental systems

---

# Step 3 — Drift-Aware Evolution (Fork-On-Drift)

Long-lived systems evolve.

Artana supports controlled run forking.

```python
result = await kernel.step_model(
    run_id="long_run",
    tenant=tenant,
    model="demo-model",
    input=ModelInput.from_prompt("New improved prompt"),
    output_schema=Decision,
    step_key="analysis_step",
    replay_policy="fork_on_drift",
)
```

If prompt changes:

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

    supervisor = SupervisorHarness(kernel=kernel, tenant=tenant)
    deployment = DeploymentHarness(kernel=kernel, tenant=tenant)

    result = await supervisor.run_child(
        harness=deployment,
        run_id="deployment_run",
    )

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
from artana.ports.model import ModelRequest, ModelResult, ModelUsage
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

```python
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
 

```

### File: `docs/Chapter5.md`
<a name="docs-chapter5md"></a>
```markdown
# Chapter 5: Distributed Scaling & Multi-Tenant Deployment

This chapter demonstrates:

* Multi-tenant isolation
* Horizontal scaling patterns
* Worker architecture
* Queue-based execution
* Long-running harness recovery
* Deployment topology
* Production safety checklist

---

# 🏗️ Step 1 — Multi-Tenant Isolation (First-Class Concept)

In Artana, tenants are explicit.

Every run is tied to:

```python
TenantContext(
    tenant_id="tenant_name",
    capabilities=frozenset({...}),
    budget_usd_limit=...
)
```

Example:

```python
from artana.kernel import ArtanaKernel
from artana.models import TenantContext
from artana.store import SQLiteStore

kernel = ArtanaKernel(
    store=SQLiteStore("multi_tenant.db"),
    model_port=DemoModelPort(),
)

tenant_a = TenantContext(
    tenant_id="tenant_a",
    capabilities=frozenset({"analytics"}),
    budget_usd_limit=10.0,
)

tenant_b = TenantContext(
    tenant_id="tenant_b",
    capabilities=frozenset(),
    budget_usd_limit=2.0,
)
```

Every run enforces:

* Budget
* Capabilities
* Policy
* Ledger separation

Isolation is guaranteed at the run level.

---

# ⚙️ Step 2 — Horizontal Scaling Pattern

Artana Kernel is stateless.

State lives in:

* EventStore
* MemoryStore
* ExperienceStore

This enables horizontal scaling.

### Worker Pattern

Each worker process:

```python
from artana import ArtanaKernel, PostgresStore
from artana.ports.model_adapter import LiteLLMAdapter

kernel = ArtanaKernel(
    store=PostgresStore("postgresql://user:pass@db:5432/artana"),  # shared DB
    model_port=LiteLLMAdapter(...),
    middleware=ArtanaKernel.default_middleware_stack(),
)
```

Workers can:

* Load any run
* Resume safely
* Replay deterministically
* Continue long-running harness

No in-memory coordination required.

---

# 🔁 Step 3 — Queue + Worker Architecture

Example using a simple async queue:

```python
import asyncio

task_queue = asyncio.Queue()

async def worker():
    while True:
        run_id, tenant = await task_queue.get()
        harness = DeploymentHarness(kernel=kernel, tenant=tenant)
        await harness.run(run_id)
        task_queue.task_done()
```

Key insight:

* Workers can crash
* On restart, they resume from durable state
* Harness enforces clean-state validation
* Kernel guarantees replay

This enables:

* Kubernetes auto-scaling
* Serverless execution
* Background job systems

---

# 🧠 Step 4 — Long-Running Harness Recovery

If a worker crashes mid-task:

```python
await harness.run("migration_run")
```

On restart:

```python
await harness.run("migration_run")
```

Because:

* TaskProgressSnapshot is persisted
* Tool resolutions are reconciled
* Partial states rejected
* step_key prevents duplication

Recovery is deterministic.

---

# 🗃️ Step 5 — Distributed Event Store (PostgresStore)

`PostgresStore` is implemented in the Artana library and can be used directly:

```python
from artana.store import PostgresStore

store = PostgresStore(
    dsn="postgresql://user:pass@db:5432/artana",
    min_pool_size=2,
    max_pool_size=20,
    command_timeout_seconds=30.0,
)
```

Then pass it into `ArtanaKernel`; all kernel logic remains identical.

Implementation notes:

* Event append is transactional
* Per-run sequencing is protected with advisory locks
* Hash-chain ledger semantics remain the same as SQLite
* Replay and idempotency behavior do not change across stores

Benefits:

* Multi-worker concurrency
* High durability
* Strong transactional guarantees
* Cloud-native deployments

EventStore is the only scaling boundary.

---

# 🌍 Step 6 — Multi-Region Deployment Strategy

Architecture recommendation:

```
[Load Balancer]
      |
[Stateless API Layer]
      |
[Worker Pool]
      |
[Shared Postgres Event Store]
      |
[Model Provider APIs]
```

Rules:

* API nodes stateless
* Workers stateless
* All state in DB
* Idempotency enforced
* Replay safe

This allows:

* Blue/green deploys
* Canary releases
* Zero-downtime upgrades

---

# 🔄 Step 7 — Rolling Upgrade Strategy (Replay Safe)

When deploying new code:

1. Old workers finish current runs
2. New workers start
3. replay_policy="allow_prompt_drift" during transition
4. Monitor REPLAYED_WITH_DRIFT events

If incompatible change:

Use:

```python
replay_policy="fork_on_drift"
```

Old runs remain untouched.

New runs fork cleanly.

---

# 📊 Step 8 — Observability & Monitoring

Monitor:

* Model cost aggregation
* Drift events
* Unknown tool outcomes
* BudgetExceededError
* ReplayConsistencyError

Example cost dashboard query:

```sql
SELECT
  tenant_id,
  COALESCE(SUM((payload_json::jsonb ->> 'cost_usd')::double precision), 0.0) AS spend
FROM kernel_events
WHERE event_type = 'model_completed'
GROUP BY tenant_id;
```

Production metrics to track:

* Avg cost per run
* Drift rate
* Fork rate
* Tool failure rate
* Harness clean-state violations

---

# 🔐 Step 9 — Security Hardening Checklist

Production kernel should:

* Use KernelPolicy.enforced()
* Enable PII scrubber
* Enforce quota
* Enforce capability guard
* Validate tool idempotency
* Monitor drift

Optional:

* Encrypt EventStore at rest
* Encrypt tool payloads
* Sign event_hash externally
* Stream ledger to SIEM

---

# 📦 Step 10 — Production Deployment Template

Minimal production instantiation:

```python
kernel = ArtanaKernel(
    store=PostgresStore("postgresql://..."),
    model_port=LiteLLMAdapter(...),
    middleware=ArtanaKernel.default_middleware_stack(),
    policy=KernelPolicy.enforced(),
)
```

Workers:

```python
harness = MyHarness(kernel=kernel, tenant=tenant)
await harness.run(run_id)
```

That’s it.

Everything else is architecture.

---

# 🧠 Final Production Mental Model

In distributed production:

| Layer        | Responsibility                 |
| ------------ | ------------------------------ |
| EventStore   | Source of truth                |
| Kernel       | Deterministic execution        |
| Harness      | Discipline & incremental logic |
| Middleware   | Enforcement                    |
| Workers      | Stateless executors            |
| Orchestrator | Scheduling                     |

Artana is:

> A deterministic execution substrate that survives crashes, drift, and scaling.

---

# 🏁 Production Readiness Checklist

Before deploying:

* [ ] All tools idempotent
* [ ] step_key stable
* [ ] replay_policy chosen intentionally
* [ ] KernelPolicy.enforced enabled
* [ ] Budget limits configured
* [ ] Ledger verification tested
* [ ] Drift handling strategy decided
* [ ] Artifact schema versioned
* [ ] Observability dashboards configured

---

 

```

### File: `docs/Chapter6.md`
<a name="docs-chapter6md"></a>
```markdown
# Chapter 6: OS-Grade Safety V2 and Harness Reality

This chapter covers what is now available in Artana:

* `KernelPolicy.enforced_v2()` for OS-grade safety
* Declarative per-tool policy with `SafetyPolicyMiddleware`
* Intent plans, semantic idempotency, limits, approvals, invariants
* First-class harness APIs (`HarnessContext`, `TaskUnit`, artifacts)

---

# Step 1 — Boot an Enforced V2 Kernel

```python
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
```

What `enforced_v2` guarantees at boot:

* PII scrubber required
* quota middleware required
* capability guard required
* safety policy middleware required

---

# Step 2 — Record a Typed Intent Plan Before Side Effects

```python
from artana.safety import IntentPlanRecord

await kernel.record_intent_plan(
    run_id="billing_run",
    tenant=tenant,
    intent=IntentPlanRecord(
        intent_id="intent_2026_02",
        goal="Send February invoice",
        why="Monthly billing close",
        success_criteria="Invoice sent exactly once",
        assumed_state="Customer account is active and approved",
        applies_to_tools=("send_invoice",),
    ),
)
```

If a configured tool requires intent and none exists (or it is stale), the tool call is blocked.
Autonomous workflows can also write this via the runtime tool `record_intent_plan`.

---

# Step 3 — Semantic Idempotency Prevents Business-Level Duplicates

For policy-configured tools, semantic keys are derived deterministically from template fields.

Example key:

`send_invoice:{tenant_id}:{billing_period}`

If the same semantic key already completed successfully, Artana blocks the new call with policy violation (`semantic_duplicate`).
If the prior outcome is unknown, Artana blocks until reconciliation (`semantic_requires_reconciliation`).

---

# Step 4 — Limits, Amount Controls, and Deterministic Invariants

`ToolLimitPolicy` can enforce:

* max calls per run
* max calls per tenant in a UTC time window
* max amount per call using an argument path

`InvariantRule` supports built-ins:

* `required_arg_true`
* `email_domain_allowlist`
* `recipient_must_be_verified`
* `custom_json_rule`

All violations are hard blocks and are audited via `policy_decision` run summaries.

---

# Step 5 — Approval Gates (Human + Critic)

Human approval flow:

```python
from pydantic import BaseModel
from artana.kernel import ApprovalRequiredError

class SendInvoiceArgs(BaseModel):
    billing_period: str
    amount_usd: float

try:
    await kernel.step_tool(
        run_id="billing_run",
        tenant=tenant,
        tool_name="send_invoice",
        arguments=SendInvoiceArgs(billing_period="2026-02", amount_usd=120.0),
        step_key="invoice_send",
    )
except ApprovalRequiredError as exc:
    await kernel.approve_tool_call(
        run_id="billing_run",
        tenant=tenant,
        approval_key=exc.approval_key,
        mode="human",
        reason="Finance manager approved",
    )
    await kernel.step_tool(
        run_id="billing_run",
        tenant=tenant,
        tool_name="send_invoice",
        arguments=SendInvoiceArgs(billing_period="2026-02", amount_usd=120.0),
        step_key="invoice_send",
    )
```

Critic approval flow is kernel-managed and replay-safe. It runs a deterministic model step and either:

* records approval and continues
* blocks with `critic_denied`

---

# Step 6 — Harness APIs Are First-Class

Artana now has an explicit harness substrate:

* `HarnessContext` and `BaseHarness`
* `IncrementalTaskHarness` with typed `TaskUnit`
* `SupervisorHarness` for composition
* built-in artifact helpers (`set_artifact`, `get_artifact`)

Example artifact usage:

```python
await harness.set_artifact(key="plan", value={"version": 2, "status": "approved"})
artifact = await harness.get_artifact(key="plan")
```

Artifacts are currently persisted as `run_summary` entries under `artifact::<key>`.
This gives durable retrieval without introducing a separate event type.

---

# Step 7 — Layer Selection Guide

Use this decision rule:

* **Kernel**: when you want minimal deterministic primitives (`step_model`, `step_tool`, `run_workflow`)
* **Harness**: when you need structured long-running discipline and typed task progress
* **AutonomousAgent**: when model-led loops are desired, with kernel safety underneath

For formal contracts, see:

* `docs/kernel_contracts.md`
* `docs/deep_traceability.md`

```

### File: `docs/deep_traceability.md`
<a name="docs-deep_traceabilitymd"></a>
```markdown
# Deep Traceability Guide

This document describes the deep traceability features added to Artana harness and kernel execution.

## Goals

Deep traceability is designed to make runs easy to inspect without breaking replay determinism:

- structured lifecycle boundaries
- hierarchical step linking
- stage-level cost and timeline visibility
- drift and validation trace channels
- query and streaming hooks for runtime observability

## Feature Map

### 1. Harness lifecycle events

Lifecycle events are first-class `EventType` entries:

- `harness_initialized`
- `harness_wake`
- `harness_stage`
- `harness_sleep`
- `harness_failed`

These are emitted by `BaseHarness.run(...)` around initialize/wake/work/sleep boundaries.

### 2. Structured trace channels (`run_summary`)

Trace channels use `RunSummaryPayload.summary_type` with a `trace::...` namespace.

Current built-in channels:

- `trace::state_transition`
- `trace::round`
- `trace::cost`
- `trace::cost_snapshot`
- `trace::drift`
- `trace::tool_validation`

You can emit additional custom channels with:

```python
await harness.write_summary(
    summary_type="trace::my_channel",
    payload={"k": "v"},
)
```

### 3. Step hierarchy (`parent_step_key`)

All kernel events support `parent_step_key`, including hash chaining for auditability.

Propagation exists through:

- model request/completion/replay-drift events
- tool request/completion/reconciliation events
- harness events and summary emissions

This allows tree reconstruction:

```text
run
  -> harness stage
    -> model step
    -> tool step
    -> trace summary
```

### 4. Automatic failure boundary

If harness execution or sleep fails, `harness_failed` is appended before raising.

Payload fields:

- `error_type`
- `message`
- `last_step_key`

### 5. Deterministic cost and timeline summaries

At stage close, harness emits:

- `trace::cost`
- `trace::cost_snapshot`

Both include:

- `stage`
- `round`
- `model_cost`
- `tool_cost`
- `total_cost`
- `logical_duration_ms`

`logical_duration_ms` is measured with monotonic clock and stored as deterministic ledger data.

### 6. Drift trace channel

When a model step has drift metadata, harness emits:

- `trace::drift`

Payload includes:

- `step_key`
- `drift_fields`
- `forked`

### 7. Live event streaming hook

`SQLiteStore` supports an optional async callback:

```python
from artana.store import SQLiteStore

async def on_event(event):
    print(event.seq, event.event_type.value)

store = SQLiteStore("artana_state.db", on_event=on_event)
```

The callback runs after each successful append.

### 8. Trace query API

Kernel exposes:

```python
summary = await kernel.explain_run(run_id)
```

Returned keys:

- `status`
- `last_stage`
- `last_tool`
- `drift_count`
- `drift_events`
- `failure_reason`
- `failure_step`
- `cost_total`

### 9. Trace level modes

Harness supports:

- `minimal`
- `stage`
- `verbose`

Usage:

```python
await harness.run(run_id="run_1", tenant=tenant, trace_level="verbose")
```

Behavior:

- `minimal`: no stage/verbose trace summaries, only core run behavior
- `stage`: lifecycle and stage-level trace summaries
- `verbose`: stage plus detailed tool/model validation summaries

## API surface

### Kernel

- `ArtanaKernel.explain_run(run_id)`
- `ArtanaKernel.get_latest_summary(...)` (compat helper)
- `ArtanaKernel.append_run_summary(..., parent_step_key=...)`
- `ArtanaKernel.append_harness_event(..., parent_step_key=...)`

### Harness

- `BaseHarness.run(..., trace_level=...)`
- `BaseHarness.emit_summary(..., parent_step_key=...)`
- `BaseHarness.run_model(..., parent_step_key=...)`
- `BaseHarness.run_tool(..., parent_step_key=...)`

### Store

- `SQLiteStore(..., on_event=...)`

## Typical tracing flow

For one harness run, you typically see:

1. `run_started`
2. `harness_initialized`
3. `harness_wake`
4. `harness_stage` (initialize/wake/work/sleep)
5. `trace::state_transition` summaries
6. model/tool events (if used)
7. `trace::round`, `trace::cost`, `trace::cost_snapshot`
8. `harness_sleep`
9. optional `harness_failed` (if exception)

## Determinism and safety notes

- all traces are ledger-backed events/summaries
- `parent_step_key` participates in event hash computation
- replay logic is preserved for model and tool execution
- trace channels are additive and do not alter kernel replay guarantees

## Reference tests

Traceability behavior is covered in:

- `tests/test_harness_layer.py`
- `tests/test_improvements_features.py`
- `tests/test_sqlite_store.py`

```

### File: `docs/kernel_contracts.md`
<a name="docs-kernel_contractsmd"></a>
```markdown
# Kernel Contracts

This document is the operational contract for replay, tool compatibility, and policy behavior.
The machine-generated behavior index lives at `docs/kernel_behavior_index.json` and is
validated in CI.

## Replay Policy

`step_model` and `KernelModelClient.step` support:

- `strict` (default): exact replay requires matching prompt, messages, model, tool signatures, and `step_key`.
- `allow_prompt_drift`: if prompt/messages drift for the same `(model, step_key, tool signatures)`, replay the prior completion and append `replayed_with_drift`.
- `fork_on_drift`: if drift is detected for the same `(model, step_key, tool signatures)`, fork into `run_id::fork::<hash>` and execute there.

## Kernel Policy Modes

- `permissive`: no required middleware.
- `enforced`: requires `PIIScrubberMiddleware`, `QuotaMiddleware`, and `CapabilityGuardMiddleware`.
- `enforced_v2`: requires all `enforced` middleware plus `SafetyPolicyMiddleware`.

## Model Request Invariants

Each `model_requested` event stores:

- `allowed_tools`: sorted tool names
- `allowed_tool_signatures`: `name + tool_version + schema_version + schema_hash`
- `allowed_tools_hash`: hash of tool signatures (not just tool names)
- `context_version`:
  - `system_prompt_hash`
  - `context_builder_version`
  - `compaction_version`

Replay validates only signature-based hashes/tokens.

## Tool Determinism Invariants

- Tool arguments are canonicalized as sorted JSON objects before matching and storage.
- Tool idempotency key input is `(run_id, tool_name, seq)`.
- Tool request events persist `tool_version` and `schema_version`.
- `tool_requested` payload optionally persists:
  - `semantic_idempotency_key`
  - `intent_id`
  - `amount_usd`

## Safety Policy Invariants

When `SafetyPolicyMiddleware` is configured for a tool, `prepare_tool_request` evaluates rules in this order:

1. intent requirement
2. semantic idempotency
3. tool limits/rate checks
4. approval gates
5. deterministic invariants

Each decision appends `run_summary` with:
- `summary_type=policy_decision`
- JSON payload containing:
  - `tool_name`
  - `fingerprint`
  - `outcome` (`allow` or `deny`)
  - `rule_id`
  - `reason`

### Semantic Idempotency

- Semantic keys are derived deterministically from configured templates.
- Duplicate prior `success` outcomes are blocked (`semantic_duplicate`).
- Prior `unknown_outcome` for the same semantic key is blocked until reconciliation (`semantic_requires_reconciliation`).

### Approval Gates

- Approval records are run summaries under `summary_type=policy::approval::<approval_key>`.
- Human mode raises `ApprovalRequiredError` until approval is recorded.
- Critic mode uses deterministic kernel model steps with key prefix `critic::<tool>::`.
- Critic denials raise `PolicyViolationError(code="critic_denied")`.

### Intent Plans

- Intent records are run summaries under `summary_type=policy::intent_plan`.
- `record_intent_plan(...)` stores typed `IntentPlanRecord` payloads.
- Missing or stale intent plans for configured tools raise
  `PolicyViolationError(code="missing_intent_plan")`.

## Tool IO Policy Hooks

Kernel middleware now includes tool hooks:

- `prepare_tool_request(run_id, tenant, tool_name, arguments_json)`
- `prepare_tool_result(run_id, tenant, tool_name, result_json)`

Hooks run in middleware order, enabling policy enforcement on tool input/output in addition to model prompt/messages.

## Agent Observability

The autonomous agent emits `run_summary` events for model and tool steps.
These summaries are queryable via `query_event_history`.
Kernel model steps also emit `run_summary` entries with `summary_type=capability_decision`
that explain why each tool was allowed or filtered.

## Harness and Artifact Contracts

Artana exposes first-class harness APIs:

- `HarnessContext`
- `BaseHarness`
- `IncrementalTaskHarness`
- `TaskUnit`
- `SupervisorHarness`

Artifacts in harnesses are persisted as run summaries with `summary_type=artifact::<key>`.
`set_artifact(...)` writes `{"value": ...}` payloads and `get_artifact(...)` resolves the latest value.

```
