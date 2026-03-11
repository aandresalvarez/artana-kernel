 

# 🚀 The Complete Beginner’s Guide to Artana (Modern Edition)

Artana is built in **three layers**:

1. **Kernel** → Durable execution OS (replay-safe, crash-proof)
2. **Agent** → Multi-turn intelligent reasoning
3. **Harness** → Structured long-running discipline

This guide walks you from deterministic steps → tools → workflows → agents → harnesses.

## Chapter Metadata

- Audience: Engineers onboarding to Artana for the first time.
- Prerequisites: Python 3.12+, repository cloned, `uv sync --all-groups`.
- Estimated time: 35–45 minutes.
- Expected outcome: You can run a local Artana flow end-to-end and understand when to choose Kernel, Workflow, Agent, or Harness layers.

Code block contract for this chapter:

* `python` blocks are standalone runnable scripts.
* `pycon` blocks are in-context snippets and assume surrounding variables/state.

---

# 🧠 Step 1 — Deterministic Model Steps (The Kernel)

Goal: first successful local run in under 10 minutes.

Every model step in Artana:

* Is persisted
* Is replay-safe
* Supports optional deterministic `step_key`
* Can be resumed safely

Minimal first run:

```python
import asyncio

from pydantic import BaseModel

from artana import (
    ArtanaKernel,
    MockModelPort,
    SingleStepModelClient,
    SQLiteStore,
    TenantContext,
)


class HelloResult(BaseModel):
    message: str


async def main():
    kernel = ArtanaKernel(
        store=SQLiteStore("step1.db"),
        model_port=MockModelPort(output={"message": "Hello from Artana!"}),
    )

    tenant = TenantContext(
        tenant_id="demo_user",
        capabilities=frozenset(),
        budget_usd_limit=1.0,
    )

    client = SingleStepModelClient(kernel)

    result = await client.step(
        run_id="hello_run",
        tenant=tenant,
        model="demo-model",
        prompt="Say hello",
        output_schema=HelloResult,
    )

    print(result.output)
    await kernel.close()


asyncio.run(main())
```

Recommended deterministic style:

```python
from artana import StepKey

step = StepKey(namespace="chapter1_step1")
result = await client.step(
    run_id="hello_run",
    tenant=tenant,
    model="demo-model",
    prompt="Say hello",
    output_schema=HelloResult,
    step_key=step.next("model"),
)
```

🔑 **Important:** use explicit `step_key` for stable workflow-style orchestration and drift policies.
Use `StepKey(namespace=...)` when generating keys across loops.
If you omit `step_key` in `KernelModelClient` / `SingleStepModelClient`, Artana generates a deterministic step key for high-level ergonomics.

`api_mode="auto"` is the normal model flow. Artana uses Responses when supported and falls back to chat-completions when needed.

---

# 🛠 Step 2 — Tools + Idempotency

Tools are durable and idempotent.

Every tool can receive:

```text
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


class TransferArgs(BaseModel):
    amount: int
    to_user: str


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

    @kernel.tool(side_effect=True)
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
        arguments=TransferArgs(amount=10, to_user="alice"),
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

from artana import (
    ArtanaKernel,
    MockModelPort,
    SQLiteStore,
    TenantContext,
    WorkflowContext,
    json_step_serde,
)


async def main():
    kernel = ArtanaKernel(
        store=SQLiteStore("workflow.db"),
        model_port=MockModelPort(output={"message": "unused"}),
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
            serde=json_step_serde(),
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
from artana.ports.model import ModelRequest, ModelResult, ModelUsage
from artana.store import SQLiteStore


class Report(BaseModel):
    text: str


class DemoModelPort:
    async def complete(self, request: ModelRequest[Report]) -> ModelResult[Report]:
        return ModelResult(
            output=Report(text="Demo report"),
            usage=ModelUsage(prompt_tokens=4, completion_tokens=3, cost_usd=0.0),
        )


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

# 🏗 Step 5 — Harnesses (Strong-Model Default + Durable Substrate)

For 2026-style model-driven work, prefer the strong-model harness path:

* `StrongModelAgentHarness` when you want `AutonomousAgent` + `ContextBuilder` + optional draft/verify + acceptance gates
* domain templates (`ResearchHarness`, `CodingHarness`, `ReviewHarness`, `CurationHarness`) when you want an operating mode instead of a blank subclass
* `StrongModelHarness` when you want a thinner durable wrapper without the agent loop

`IncrementalTaskHarness` remains the lower-level durable substrate underneath that posture.
It still matters, but it should not be the first mental model.

That substrate enforces:

* Incremental progress
* One task completion per session
* Clean state
* Structured summaries

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
        model_port=MockModelPort(output={"message": "unused"}),
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

The substrate automatically:

* Persist task progress
* Prevent multiple DONE transitions per session
* Enforce clean state before sleep

If you want the recommended opinionated paths, see:

* `examples/10_live_manual_agent_harness.py` for the coding-shaped strong-model harness
* `examples/11_durable_release_harness.py` for the governed review + side-effect pattern
* `examples/12_research_strong_model_harness.py` for the research-shaped strong-model harness

---

# 🗂 Step 6 — Artifacts (Structured Continuity)

Artifacts store structured durable state.

Snippet (in-context, not standalone):

```pycon
await harness.set_artifact(key="plan", value={"phase": 1})
plan = await harness.get_artifact(key="plan")
print(plan)
```

Artifacts are stored as structured run summaries.

---

# 🧭 Step 7 — Supervisor Harness (Multi-Agent)

Compose harnesses safely.

Snippet (in-context, not standalone):

```pycon
from artana.harness import SupervisorHarness

class ResearchSupervisor(SupervisorHarness):
    async def step(self, *, context):
        child = ResearchHarness(kernel=self.kernel, tenant=context.tenant)
        return await self.run_child(
            harness=child,
            run_id=f"{context.run_id}::child",
            tenant=context.tenant,
            model=context.model,
        )

supervisor = ResearchSupervisor(kernel=kernel, tenant=tenant)
result = await supervisor.run(run_id="supervisor_run")
```

---

# ⚖️ Step 8 — Draft vs Verify Model Calls

Harness helpers support two-model loops without changing kernel semantics.

Snippet (in-context, not standalone):

```pycon
draft = await harness.run_draft_model(
    prompt="Brainstorm implementation options",
    output_schema=Decision,
    model_options=ModelCallOptions(api_mode="auto", reasoning_effort="none"),
)

verify = await harness.run_verify_model(
    prompt="Check correctness and edge cases",
    output_schema=Decision,
    model_options=ModelCallOptions(api_mode="responses", reasoning_effort="high"),
)
```

Use `run_draft_model` for low-cost exploration and `run_verify_model` for final checks.

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

## You Should Now Be Able To

- Run a complete local-first Artana flow without external model dependencies.
- Decide when to rely on auto-generated step keys vs explicit `StepKey` control.
- Build first durable tool, workflow, agent, and harness executions on the same kernel.

## Next Chapter

Continue to [Chapter 2: Scaling Up](./Chapter2.md) to learn durable harness progression, supervisor orchestration, and production discipline.

 
