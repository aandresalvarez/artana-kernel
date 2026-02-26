 
# Chapter 2: Scaling Up (Harnesses, Supervision, and Production Discipline)

This chapter focuses on production patterns:

* Multi-agent orchestration
* Long-running incremental harnesses
* Structured artifacts
* Replay modes
* Middleware enforcement
* Ledger & observability

## Chapter Metadata

- Audience: Engineers who completed Chapter 1 and are moving to durable multi-session execution.
- Prerequisites: Chapter 1 complete; comfort with `TaskUnit`, run IDs, and basic tooling.
- Estimated time: 35 minutes.
- Expected outcome: You can structure long-running harness workflows with supervisor coordination and production-safe middleware/audit practices.

Code block contract for this chapter:

* `python` blocks are standalone runnable scripts.
* `pycon` blocks are in-context snippets and may assume existing `kernel`, `tenant`, or harness state.

---

# Step 1 — Structured Multi-Agent Supervision (Harness-Based Swarms)

Instead of directly spawning subagents from tools, modern Artana prefers **SupervisorHarness**.

```python
import asyncio

from artana import (
    ArtanaKernel,
    IncrementalTaskHarness,
    MockModelPort,
    SQLiteStore,
    SupervisorHarness,
    TaskUnit,
    TenantContext,
)


class ResearchHarness(IncrementalTaskHarness):

    async def define_tasks(self):
        return [
            TaskUnit(id="fact", description="Provide a historical fact"),
        ]

    async def work_on(self, task: TaskUnit):
        print("Research task executed:", task.id)


class SwarmSupervisor(SupervisorHarness):
    async def step(self, *, context):
        child = ResearchHarness(kernel=self.kernel, tenant=context.tenant)
        return await self.run_child(
            harness=child,
            run_id=f"{context.run_id}::research",
            tenant=context.tenant,
            model=context.model,
        )


async def main():
    kernel = ArtanaKernel(
        store=SQLiteStore("chapter2_step1.db"),
        model_port=MockModelPort(output={"message": "unused"}),
    )

    tenant = TenantContext(
        tenant_id="manager",
        capabilities=frozenset(),
        budget_usd_limit=5.0,
    )

    supervisor = SwarmSupervisor(kernel=kernel, tenant=tenant)
    result = await supervisor.run(run_id="swarm_run_01")

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

from artana import (
    ArtanaKernel,
    IncrementalTaskHarness,
    MockModelPort,
    SQLiteStore,
    TaskUnit,
    TenantContext,
)


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
        model_port=MockModelPort(output={"message": "unused"}),
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

Snippet (in-context, not standalone):

```pycon
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
| strict             | Fail if model inputs/options change |
| allow_prompt_drift | Replay safely with drift summary |
| fork_on_drift      | Fork run if logic changed        |

This is critical for long-lived systems.

---

# Step 4 — Structured Artifacts (Durable State)

Artifacts allow structured continuity across sessions.

Snippet (in-context, not standalone):

```pycon
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

Snippet (in-context, not standalone):

```pycon
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

Capability visibility helper (in-context, not standalone):

```pycon
capabilities_view = await kernel.describe_capabilities(tenant=tenant)
visible_tools = kernel.list_tools_for_tenant(tenant=tenant)
print(capabilities_view["final_allowed_tools"])
print([tool.name for tool in visible_tools])
```

---

# Step 6 — Audit Ledger (Immutable Event Log)

Every run is a verifiable ledger.

Snippet (in-context, not standalone):

```pycon
from artana.store import SQLiteStore

events = await kernel.get_events(run_id="pipeline_run_001")

for event in events:
    print(event.seq, event.event_type)

store = SQLiteStore("chapter2_step5.db")
verified = await store.verify_run_chain("pipeline_run_001")
print("Chain valid:", verified)
await store.close()
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

```text
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

# Step 8 — Productized Harness Templates and Tool Bundles

Use built-in templates/bundles instead of rewriting common coding-agent plumbing.

```pycon
from artana import DraftReviewVerifySupervisor, ObservabilityTools
from artana.tools import CodingHarnessTools

coding_tools = CodingHarnessTools(sandbox_root="/tmp/agent_workspace")
observability_tools = ObservabilityTools(root="/var/tmp/agent_observability")

# Attach one registry as kernel tool_port per worker process.
kernel = ArtanaKernel(
    store=SQLiteStore("chapter2_tools.db"),
    model_port=DemoModelPort(),
    tool_port=coding_tools.registry(),
)

# Capability expectations for coding bundle:
# - coding:worktree
# - coding:read
# - coding:write
```

Supervisor template usage:

```pycon
supervisor = DraftReviewVerifySupervisor(
    kernel=kernel,
    tenant=tenant,
    drafter=drafter_harness,
    reviewer=reviewer_harness,
    verifier=verifier_harness,
)

result = await supervisor.run(run_id="draft_review_verify_run")
print(result.approved)
```

Safety notes:

* Keep `sandbox_root` and observability roots isolated per environment.
* Grant only required capabilities to each tenant.
* Use `@kernel.tool(side_effect=True)` for mutating tools to enforce idempotency context.

---

# Chapter 2 Summary

In production, you should:

* Use Harness for long-running tasks
* Use SupervisorHarness for orchestration
* Store structured artifacts
* Enforce incremental discipline
* Choose replay mode deliberately
* Layer middleware carefully
* Prefer built-in templates (`DraftReviewVerifySupervisor`) and bundles (`CodingHarnessTools`, `ObservabilityTools`)
* Rely on immutable event ledger
* Use summaries instead of scanning full history

## You Should Now Be Able To

- Run durable harness sessions repeatedly with deterministic incremental progress.
- Compose child harnesses behind a supervisor run topology.
- Inspect capability filtering and event-ledger history for production troubleshooting.

## Next Chapter

Continue to [Chapter 3: Production Mode](./Chapter3.md) to harden failure handling, replay strategy, and crash recovery.
 
