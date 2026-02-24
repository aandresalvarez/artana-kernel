 

# 🚀 The Complete Beginner’s Guide to Artana (Modern Edition)

Artana is built in **three layers**:

1. **Kernel** → Durable execution OS (replay-safe, crash-proof)
2. **Agent** → Multi-turn intelligent reasoning
3. **Harness** → Structured long-running discipline

This guide walks you from deterministic steps → tools → workflows → agents → harnesses.

Standalone scripts in this chapter are runnable. Short snippet blocks are
in-context API references and assume surrounding variables.

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
from artana.kernel import ArtanaKernel, WorkflowContext, json_step_serde
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

 
