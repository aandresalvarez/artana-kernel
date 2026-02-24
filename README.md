

# Artana Kernel (MVP)

Artana is a policy-enforced, event-sourced execution kernel for AI systems. It provides durable, replay-safe, governed execution for LLM-driven workflows — without requiring a full distributed workflow platform.

**Artana is not:**

- A cognitive reasoning framework (it does not enforce A* search or prompt strategies)
- A workflow DSL (it does not invent new conditional or loop syntax)
- A replacement for Temporal

**Artana is:**

- A domain-specific execution primitive and OS-level hypervisor for AI workloads.

## Positioning

Modern AI systems operate across three layers:

### Agent Layer
(LangChain, PydanticAI, OpenAI Agents SDK)
- Prompt orchestration
- Tool selection
- Structured outputs
- Multi-agent coordination

These systems help models reason. They do not provide strong guarantees around durability, governance, replay semantics, or side-effect safety.

### Workflow Layer
(Temporal and similar engines)
- Durable execution
- Deterministic replay
- Distributed scheduling
- At-least-once Activities
- Production-scale orchestration

Temporal is a general-purpose workflow engine. It does not natively model:
- Token budgeting
- AI-specific governance
- Prompt-level policy enforcement
- LLM-aware execution semantics

### Artana's Layer — AI Execution Kernel
Artana sits between them. It provides:
- Deterministic replay semantics for LLM + tool execution
- Two-phase side-effect handling
- Capability gating for AI tools
- Prompt-level middleware enforcement
- Token budgeting and reconciliation
- Multi-tenant isolation
- Replay-safe crash recovery

Artana is the safety and durability layer your AI agent should run on. It complements Temporal and strengthens agent frameworks.

## What This Release Implements

Initial implementation aligned with the Artana Kernel PRD:

- Strictly typed kernel and event models (`mypy --strict`)
- Event-sourced stores with hash-chained cryptographic ledgers:
  - `SQLiteStore` for local/single-worker setups
  - `PostgresStore` for multi-worker/shared-database deployments
- Ports for model and tools:
  - `LiteLLMAdapter` (implements `ModelPort`)
  - `LocalToolRegistry` (implements `ToolPort`)
- Core kernel APIs:
  - `start_run` / `load_run` — authoritative run lifecycle
  - `step_model` / `step_tool` — deterministic, atomic execution primitives
  - replay policy modes on `step_model`: `strict | allow_prompt_drift | fork_on_drift`
  - `reconcile_tool` — resolve `unknown_outcome` tool requests safely
  - `pause` / `resume` — human-in-the-loop durable interrupts
- **The Agent Runtime:**
  - `AutonomousAgent` — An out-of-the-box `Model -> Tool -> Model` loop that automatically executes tools and manages conversation memory on top of the Kernel.
  - `CompactionStrategy` — proactive context compaction (message-count and token-threshold triggers) with replay-safe summaries.
  - `ContextBuilder` — pluggable prompt assembly pipeline (identity + long-term memory + inter-run experience learnings + progressive skills + short-term history).
  - `SQLiteExperienceStore` — tenant/task-scoped inter-run learning store for reusable `WIN_PATTERN`, `ANTI_PATTERN`, and `FACT` rules.
  - Optional post-run reflection (`auto_reflect=True`) to extract and persist reusable rules with deterministic replay-safe step keys.
  - Progressive skill disclosure via built-in `load_skill(...)` meta-tool, capability-scoped so tenants only see/load authorized skills.
  - Built-in long-term memory tools: `core_memory_append`, `core_memory_replace`, `core_memory_search`.
  - `SubAgentFactory` — sub-agent delegation with run lineage (`parent::sub_agent::idempotency_key`) and parent tenant inheritance (capabilities + budget).
- **The Workflow Runtime:**
  - `run_workflow` — Durable workflow with `WorkflowContext`, deterministic Python logic execution (`ctx.step`), and `ctx.pause`.
- Tool registration via `@kernel.tool(requires_capability="...")`
- Middleware stack: `PIIScrubberMiddleware`, `QuotaMiddleware`, `CapabilityGuardMiddleware`
- Optional OS-grade Safety Policy V2 (`KernelPolicy.enforced_v2()` + `SafetyPolicyMiddleware`) for:
  - typed intent plans before side-effect tools
  - semantic idempotency dedupe
  - per-tool limits (run + tenant window + amount)
  - human/critic approval gates
  - deterministic invariants
- Replay-safe two-phase tool execution (protects against double-charging/executing on crash recovery)
- Tool signature hashing (`name + version + schema hash`) for replay compatibility checks
- Context version tracking in `model_requested` events (`system_prompt_hash`, context builder, compaction)
- Agent `run_summary` events for lightweight autonomous observability
- Kernel `capability_decision` run summaries for per-tool allow/filter reasoning
- Kernel policy APIs for safety workflows:
  - `record_intent_plan(...)`
  - `approve_tool_call(...)`
- Extended kernel syscalls for orchestration/scheduling:
  - `get_run_status(...)`, `list_active_runs(...)`, `resume_point(...)`
  - `checkpoint(...)`
  - `set_artifact(...)`, `get_artifact(...)`, `list_artifacts(...)`
  - `block_run(...)`, `unblock_run(...)`
  - `stream_events(...)`
  - `acquire_run_lease(...)`, `renew_run_lease(...)`, `release_run_lease(...)`
  - `explain_tool_allowlist(...)`
  - `canonicalize_tool_args(...)`, `tool_fingerprint(...)`
- Tool request payload extensions for policy traceability:
  - `semantic_idempotency_key`
  - `intent_id`
  - `amount_usd`
- Kernel contracts reference: `docs/kernel_contracts.md`
- Deep traceability reference: `docs/deep_traceability.md`
- Generated behavior index reference: `docs/kernel_behavior_index.json`
- OS-grade safety and harness chapter: `docs/Chapter6.md`

## Core Guarantees

### Durable AI Execution
All model and tool interactions are persisted as append-only events.
**Replay:**
- Never re-calls completed model steps
- Never re-executes completed tool steps
- Detects incomplete side-effects and forces reconciliation
- Resumes deterministically in milliseconds

### Two-Phase Tool Semantics
Each tool invocation emits:
1. `tool_requested` (Before execution)
2. `tool_completed` (After execution)

If a worker dies between these events, Artana halts replay and demands reconciliation, ensuring side-effects (like DB writes or payments) are never accidentally duplicated.

### Governance by Default
Every outbound call passes through middleware:
- PII scrubbing (Demo implementation provided)
- Budget reservation
- Capability validation
- Audit logging

### Budget Enforcement
Token usage is tracked via LiteLLM metadata. Execution is terminated if budgets are exceeded. Budgeting is enforced at runtime — not just as post-run reporting.

### Multi-Tenant Isolation
All runs require a `TenantContext`:
- `tenant_id`
- `capabilities`
- `budget_usd_limit`

The Kernel automatically filters the tool list based on the tenant's capabilities *before* the LLM ever sees the prompt.

## Developer Guardrails

Set up local enforcement once per clone:

```bash
uv sync --all-groups
uv run pre-commit install
```

Run the full guardrail suite at any time:

```bash
uv run pre-commit run --all-files
```

Pre-commit mirrors CI quality gates and best-practices contracts:
- `ruff check .`
- `mypy --strict src tests`
- `pytest`
- `python scripts/generate_kernel_behavior_index.py --check`

Best-practices invariants are also enforced in tests (`tests/test_best_practices_contract.py`) to prevent regressions in architecture boundaries, middleware ordering, typing hygiene, and side-effect safety patterns.

## Mental Models

Artana gives you two ways to build, both backed by the exact same secure Kernel.

### 1. The Autonomous Agent (The "Steering Wheel")
Use this when you want the LLM to dynamically decide which tools to use and how many times to loop until it reaches a final answer. The Kernel safely injects the tools the Tenant is allowed to use.

```python
agent = AutonomousAgent(kernel=kernel)

report = await agent.run(
    run_id="research_01",
    tenant=tenant,
    model="gpt-4o",
    prompt="Find the CEO of Acme Corp.",
    output_schema=CompanyReport
)
```

For long-running agents, configure the runtime explicitly:

```python
from artana import (
    AutonomousAgent,
    CompactionStrategy,
    ContextBuilder,
    SQLiteExperienceStore,
    SQLiteMemoryStore,
)

memory_store = SQLiteMemoryStore("agent_memory.db")
experience_store = SQLiteExperienceStore("tenant_experience.db")
context_builder = ContextBuilder(
    identity="You are a senior data analyst.",
    memory_store=memory_store,
    experience_store=experience_store,
    task_category="Financial_Reporting",
    progressive_skills=True,
)
agent = AutonomousAgent(
    kernel=kernel,
    context_builder=context_builder,
    compaction=CompactionStrategy(
        trigger_at_messages=40,
        keep_recent_messages=10,
        summarize_with_model="gpt-4o-mini",
    ),
    auto_reflect=True,
    reflection_model="gpt-4o-mini",
)
```

### 2. The Durable Workflow (Manual Control)
Use this when you need strict, deterministic pipelines with Human-In-The-Loop pauses and pure Python logic. The LLM is restricted to single extraction steps.

```python
async def my_workflow(ctx: WorkflowContext):
    # 1. Atomic LLM call
    facts = await model_client.step(..., step_key="extract")
    
    # 2. Deterministic Python logic (cached to DB)
    derived = await ctx.step(name="math", action=run_math, serde=...)
    
    # 3. Durable Pause (Server can safely restart here)
    await ctx.pause(reason="Human approval required", context=derived)
    
    return {"status": "saved"}

result = await kernel.run_workflow(run_id="run_1", tenant=tenant, workflow=my_workflow)
```

### 3. The Harness SDK (Developer-Friendly Default)
Use this when you want durable long-running work with minimal kernel plumbing.

```python
from artana import IncrementalTaskHarness, TaskUnit

class ResearchHarness(IncrementalTaskHarness):
    async def define_tasks(self) -> list[TaskUnit]:
        return [
            TaskUnit(id="collect_data", description="Collect datasets"),
            TaskUnit(id="analyze", description="Analyze patterns"),
            TaskUnit(id="write_summary", description="Write final report"),
        ]

    async def work_on(self, task: TaskUnit) -> None:
        if task.id == "collect_data":
            ...
        elif task.id == "analyze":
            ...
        else:
            ...

harness = ResearchHarness(kernel=kernel, tenant=tenant)
snapshot = await harness.run(run_id="research_run_01")
```

What the SDK handles automatically:
- run creation and wake/sleep lifecycle
- wake reorientation summaries
- task progress persistence and one-task-per-session advancement
- replay-safe model/tool execution helpers (`run_model`, `run_tool`)
- clean-state validation before sleep

## Architecture

Artana follows a strict Ports & Adapters model combined with an Event-Sourced Middleware stack:

```text
[ Developer's Python Code (Agent or Workflow) ]
                 ↓
[ Artana Kernel (The Orchestrator) ] 
                 ↓
[ Middleware Stack ] -> 1. PII Scrubber
                     -> 2. Quota Check
                     -> 3. Capability Guard
                     -> 4. Safety Policy (optional, required for enforced_v2)
                 ↓
    [ Port Interfaces ]
       ↙                 ↘
[ ModelPort ]         [ ToolPort ]
 (LiteLLM)         (Local Function Registry)
       ↘                 ↙
   [ EventStore (SQLiteStore | PostgresStore) ]
```

## Detailed API Reference

### Core Data Types

```python
TenantContext(
    tenant_id: str,
    capabilities: frozenset[str] = frozenset(),
    budget_usd_limit: float,  # > 0.0
)

KernelPolicy.enforced() -> KernelPolicy
KernelPolicy.enforced_v2() -> KernelPolicy
ModelInput.from_prompt(prompt: str) -> ModelInput
ModelInput.from_messages(messages: Sequence[ChatMessage]) -> ModelInput
```

### ArtanaKernel Initialization
```python
from artana import ArtanaKernel, PostgresStore, SQLiteStore

kernel = ArtanaKernel(
    store=SQLiteStore("artana_state.db"),
    model_port=LiteLLMAdapter(),
    middleware=ArtanaKernel.default_middleware_stack(),
    policy=KernelPolicy.enforced(),
)

# Multi-worker / shared DB deployment:
kernel = ArtanaKernel(
    store=PostgresStore("postgresql://user:pass@localhost:5432/artana"),
    model_port=LiteLLMAdapter(),
    middleware=ArtanaKernel.default_middleware_stack(),
    policy=KernelPolicy.enforced(),
)
```

Use `KernelPolicy.enforced()` for baseline enforcement. Use
`KernelPolicy.enforced_v2()` + `SafetyPolicyMiddleware` when tools can create
external side effects (payments, emails, deletes, transfers).

Opt-in OS-grade safety policy:
```python
from artana import (
    ArtanaKernel,
    IntentRequirement,
    KernelPolicy,
    SafetyPolicyConfig,
    SemanticIdempotencyRequirement,
    ToolLimitPolicy,
    ToolSafetyPolicy,
)
from artana.middleware import SafetyPolicyMiddleware

safety = SafetyPolicyMiddleware(
    config=SafetyPolicyConfig(
        tools={
            "send_invoice": ToolSafetyPolicy(
                intent=IntentRequirement(require_intent=True),
                semantic_idempotency=SemanticIdempotencyRequirement(
                    template="send_invoice:{tenant_id}:{billing_period}",
                    required_fields=("billing_period",),
                ),
                limits=ToolLimitPolicy(max_calls_per_run=1),
            )
        }
    )
)

kernel = ArtanaKernel(
    store=SQLiteStore("artana_state.db"),
    model_port=LiteLLMAdapter(),
    middleware=ArtanaKernel.default_middleware_stack(safety=safety),
    policy=KernelPolicy.enforced_v2(),
)
```

### Tool Registration
```python
from artana.ports.tool import ToolExecutionContext

@kernel.tool(requires_capability="finance:write")
async def transfer_funds(
    account_id: str,
    amount: str,
    artana_context: ToolExecutionContext,
) -> str:
    # For side-effect calls, always forward artana_context.idempotency_key
    # to the external API (Stripe, SendGrid, etc.) to prevent duplicate effects.
    idempotency_key = artana_context.idempotency_key
    _ = idempotency_key
    return "Success"
```

For side-effect tools, treating idempotency as optional is unsafe. Always accept
`artana_context: ToolExecutionContext` and pass `artana_context.idempotency_key`
to downstream APIs.

### Autonomous Agent API
A runtime wrapper over the kernel that manages conversation history, automatic tool execution,
context compaction, long-term memory, inter-run experience learning, and progressive skill disclosure.
```python
from artana.agent import (
    AutonomousAgent,
    CompactionStrategy,
    ContextBuilder,
    SQLiteExperienceStore,
)
from artana.agent.memory import SQLiteMemoryStore

memory_store = SQLiteMemoryStore("agent_memory.db")
experience_store = SQLiteExperienceStore("tenant_experience.db")
context_builder = ContextBuilder(
    memory_store=memory_store,
    experience_store=experience_store,
    task_category="Financial_Reporting",
    progressive_skills=True,
)

agent = AutonomousAgent(
    kernel=kernel,
    context_builder=context_builder,
    compaction=CompactionStrategy(trigger_at_messages=40, keep_recent_messages=10),
    auto_reflect=True,
    reflection_model="gpt-4o-mini",
)
result = await agent.run(
    run_id="run_123",
    tenant=tenant,
    model="gpt-4o",
    system_prompt="You are a helpful agent.",
    prompt="Do the task.",
    output_schema=FinalDecision,
    max_iterations=15
)
```

Experience models are strict and reusable in both autonomous and workflow code:
`RuleType`, `ExperienceRule`, and `ReflectionResult`.

When `progressive_skills=True`:
- The initial prompt includes a lightweight capability-filtered skills panel.
- `load_skill(skill_name=...)` returns full schema/instructions only for authorized tools.
- Unauthorized skill loads return a deterministic error payload (`"error": "forbidden_skill"`).

When `experience_store` and `task_category` are configured:
- The system prompt includes:
  - `[PAST LEARNINGS FOR THIS TASK]`
  - prioritized historical rules for that exact `tenant_id` + task category
- With `auto_reflect=True`, the agent runs a deterministic reflection step
  (`turn_{iteration}_reflection`) at the end of successful runs and persists extracted rules.

### Sub-Agent Delegation API
Create tools that run specialized child agents with inherited governance and durable lineage.
```python
from artana.agent import SubAgentFactory

# Parent tenant context is inherited from ToolExecutionContext automatically.
factory = SubAgentFactory(kernel=kernel)

factory.create(
    name="run_researcher",
    output_schema=ResearchResult,
    model="gpt-4o-mini",
    system_prompt="You are a specialized researcher.",
    requires_capability="spawn_researcher",
)
```

### Workflow Context API
For deterministic, step-by-step processes using pure Python.
```python
await kernel.run_workflow(
    run_id="run_123",
    tenant=tenant,
    workflow=my_async_workflow_function,
)

# Inside the workflow function:
await ctx.step(name="step_1", action=my_func, serde=pydantic_step_serde(MyModel))
await ctx.pause(reason="Needs review", step_key="approval_gate")
```

### Raw Kernel Primitives
If you are building your own custom orchestration loop:
```python
await kernel.step_model(
    run_id=run_id,
    tenant=tenant,
    model="gpt-4o",
    input=ModelInput.from_prompt("..."),
    output_schema=MySchema,
    step_key="turn_1"
)

await kernel.step_tool(
    run_id=run_id,
    tenant=tenant,
    tool_name="transfer_funds",
    arguments=ArgsSchema(...),
    step_key="tool_1"
)
```

Additional orchestration syscalls:
```python
status = await kernel.get_run_status(run_id=run_id)
resume_point = await kernel.resume_point(run_id=run_id)
active = await kernel.list_active_runs(tenant_id=tenant.tenant_id)

await kernel.checkpoint(
    run_id=run_id,
    tenant=tenant,
    name="phase_collect",
    payload={"round": 1, "done": True},
)

await kernel.set_artifact(
    run_id=run_id,
    tenant=tenant,
    key="report",
    value={"version": 2},
)
report = await kernel.get_artifact(run_id=run_id, key="report")

await kernel.block_run(
    run_id=run_id,
    tenant=tenant,
    reason="Waiting for external approval",
    unblock_key="approval_123",
)
await kernel.unblock_run(
    run_id=run_id,
    tenant=tenant,
    unblock_key="approval_123",
)
```

### Safety Workflow APIs
Use these with `SafetyPolicyMiddleware` when a tool requires intent planning or approvals.

```python
from artana import IntentPlanRecord

await kernel.record_intent_plan(
    run_id=run_id,
    tenant=tenant,
    intent=IntentPlanRecord(
        intent_id="intent_2026_02",
        goal="Send February invoice",
        why="Billing cycle closed",
        success_criteria="Invoice sent exactly once",
        assumed_state="Customer account is active",
        applies_to_tools=("send_invoice",),
    ),
)

await kernel.approve_tool_call(
    run_id=run_id,
    tenant=tenant,
    approval_key="<deterministic_approval_key>",
    mode="human",
    reason="Finance manager approved",
)
```

Tool gateway helpers:
```python
canonical_args_json, schema_hash = kernel.canonicalize_tool_args(
    tool_name="transfer_funds",
    arguments={"account_id": "acc_1", "amount": 12.5},
)
fingerprint = kernel.tool_fingerprint(tool_name="transfer_funds")
```

## Examples

Run examples from the repository root:

- **`01_durable_chat_replay.py`**: Basic capability-scoped tool execution and durable replay.
- **`02_real_litellm_chat.py`**: Real OpenAI calls proving replay invariants without duplicate network requests.
- **`03_fact_extraction_triplets.py`**: Single-step structured extraction.
- **`04_autonomous_agent_research.py`**: Demonstrates the `AutonomousAgent` loop dynamically selecting tools to accomplish a goal.
- **`05_hard_triplets_workflow.py`**: Demonstrates the strict `run_workflow` pattern interleaving LLM calls, deterministic Python math, and a durable Human-In-The-Loop pause.
- **`06_triplets_swarm.py`**: Demonstrates multi-agent orchestration (lead agent + extractor/adjudicator sub-agents + deterministic graph math tool) on the shared Kernel ledger.
- **`07_adaptive_agent_learning.py`**: Demonstrates inter-run experience learning where Run 1 discovers a durable rule and Run 2 succeeds immediately using injected past learnings.
- **`golden_example.py`**: Canonical production-leaning example testing unknown tool outcomes and reconciliation.

## Growth Path

- **Phase 1 (Shipped):** SQLite backend, strict replay semantics, Autonomous Agent runtime (compaction/memory/progressive skills/sub-agents), inter-run Experience Engine (tenant/task rule memory + reflection), Workflow contexts.
- **Phase 2 (In progress):** Postgres backend for multi-worker concurrency (now available), plus Snapshotting and Advanced Observability (Logfire tracing decorators).
- **Phase 3:** Optional integration with Temporal, distributed execution model, enterprise policy engines (OPA).
 
