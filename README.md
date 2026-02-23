# Artana Kernel (MVP)

Artana is a policy-enforced, event-sourced execution kernel for AI systems. It provides durable, replay-safe, governed execution for LLM-driven workflows — without requiring a full distributed workflow platform.

**Artana is not:**

- An agent framework
- A reasoning orchestrator
- A workflow DSL
- A replacement for Temporal

**Artana is:**

- A domain-specific execution primitive for AI workloads.

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

These must be built on top.

### Artana's Layer — AI Execution Kernel

Artana sits between them. It provides:

- Deterministic replay semantics for LLM + tool execution
- Two-phase side-effect handling
- Capability gating for AI tools
- Prompt-level middleware enforcement
- Token budgeting and reconciliation
- Multi-tenant isolation
- Replay-safe crash recovery

Artana is the safety and durability layer your AI agent should run on. It complements Temporal. It strengthens agent frameworks.

## What This MVP Implements

Initial implementation aligned with the Artana Kernel PRD:

- Strictly typed kernel and event models (mypy --strict, no Any)
- Event-sourced SQLite store (`SQLiteStore`):
  - `append_event`
  - `get_events_for_run`
  - `verify_run_chain`
- Ports for model and tools:
  - `LiteLLMAdapter` (implements `ModelPort`)
  - `LocalToolRegistry` (implements `ToolPort`)
- Core kernel APIs:
  - `start_run` — kernel-issued run IDs for authoritative run lifecycle
  - `load_run` — load an existing run reference
  - `step_model` — deterministic model step execution
  - `step_tool` — deterministic tool step execution
  - `reconcile_tool` — resolve `unknown_outcome` tool requests using original idempotency key
  - `pause` — request human-in-the-loop pause with optional context
  - `resume` — append a resume boundary event with optional human input
  - `run_workflow` — durable workflow with `WorkflowContext`, step serde, replay
  - `KernelModelClient.chat(...)` (agent layer) — convenience wrapper around `step_model`
  - `default_middleware_stack` — deterministic middleware ordering helper
  - `KernelPolicy.enforced()` — startup guard requiring governance middleware
- Tool registration via `@kernel.tool(requires_capability="...")` decorator
- Middleware stack: `PIIScrubberMiddleware`, `QuotaMiddleware`, `CapabilityGuardMiddleware`
- `PIIScrubberMiddleware` is intentionally demo-level; use production DLP/classification controls for real PII handling.
- Replay-safe two-phase tool execution
- Deterministic crash recovery
- Cryptographic audit hash-chain with verification
- Async pytest coverage for: sequencing correctness, replay determinism, quota enforcement, middleware filtering/redaction, crash recovery, audit integrity

## Core Guarantees

### Durable AI Execution

All model and tool interactions are persisted as append-only events.

**Replay:**

- Never re-calls completed model steps
- Never re-executes completed tool steps
- Detects incomplete side-effects
- Resumes deterministically

### Two-Phase Tool Semantics

Each tool invocation emits:

- `tool_requested`
- `tool_completed`

This prevents accidental double side-effects. Artana does not assume tools are safe. It enforces execution boundaries explicitly.

### Governance by Default

Every outbound call passes through middleware:

- PII scrubbing
- Budget reservation
- Capability validation
- Audit logging

There is no bypass path.

`PIIScrubberMiddleware` is intentionally simple regex redaction for examples and tests. It is not production-grade PII detection.

### Budget Enforcement

Token usage is tracked via LiteLLM metadata. Execution is terminated if budgets are exceeded. Budgeting is enforced at runtime — not as reporting.

### Multi-Tenant Isolation

All runs require:

- `tenant_id`
- `capabilities`
- `budget_usd_limit`

All event logs are tenant-scoped.

### Tamper-Evident Audit Trail

Events form a hash-chain ledger. Audit verification is built in.

## Mental Model

Developers write normal Python:

```python
run = await kernel.start_run(tenant=tenant)
step = await kernel.step_model(
    run_id=run.run_id,
    tenant=tenant,
    model="gpt-4o-mini",
    input=ModelInput.from_prompt("Should we transfer?"),
    output_schema=Decision,
)
decision = step.output
```

Artana ensures: durability, governance, side-effect safety, budget control, deterministic replay, auditability. You control orchestration. Artana controls execution integrity.

## Architecture

Artana follows a strict Ports & Adapters model:

```
Developer Code → Kernel → Middleware Stack → ModelPort / ToolPort → EventStore
```

No direct LiteLLM or DB calls from userland. All I/O is event-wrapped.

### Code Structure

```
artana/
├── __init__.py          # ArtanaKernel, TenantContext, WorkflowContext, SQLiteStore, etc.
├── kernel.py            # Re-exports from _kernel
├── models.py            # TenantContext
├── events.py            # KernelEvent, EventType, payloads, compute_event_hash
├── _kernel/
│   ├── core.py          # ArtanaKernel
│   ├── workflow_runtime.py  # run_workflow, WorkflowContext, StepSerde
│   ├── replay.py        # replay matching + tenant validation helpers
│   ├── model_cycle.py   # get_or_execute_model_step
│   ├── tool_cycle.py    # replay-aware tool step execution
│   ├── policies.py      # middleware pipeline + tool capability assertion helpers
│   └── types.py         # ModelInput, StepModelResult, StepToolResult, KernelPolicy, RunHandle
├── agent.py             # KernelModelClient (ChatClient compatibility alias)
├── agent_runtime.py     # AgentRuntime loop abstraction over kernel primitives
├── middleware/          # PIIScrubberMiddleware, QuotaMiddleware, CapabilityGuardMiddleware
├── ports/
│   ├── model.py        # LiteLLMAdapter, ModelPort
│   ├── model_adapter.py
│   ├── model_types.py
│   └── tool.py         # LocalToolRegistry, ToolPort
└── store/
    ├── base.py         # EventStore protocol
    └── sqlite.py       # SQLiteStore
```

## Growth Path

- **Phase 1 (MVP):** SQLite, single worker, strict replay semantics
- **Phase 2:** Postgres backend, concurrency controls, snapshotting, observability
- **Phase 3:** Optional integration with Temporal, distributed execution model, enterprise policy engines

Artana is compatible with Temporal. It can run inside Temporal Activities. It can provide AI execution guarantees even in distributed workflow systems.

## Quickstart

```bash
uv sync
uv run pytest
uv run mypy src
```

Strict typing and tests are mandatory. No `Any`. No hidden side-effects. No untracked execution.

### Authoritative Mode (Recommended)

For production-like usage, prefer:

- `run = await kernel.start_run(tenant=tenant)` instead of caller-generated run IDs
- `middleware=ArtanaKernel.default_middleware_stack()` to lock middleware order
- `policy=KernelPolicy.enforced()` to fail fast if required middleware is missing
- `LiteLLMAdapter(fail_on_unknown_cost=True)` to fail fast on unknown pricing metadata

This turns governance and replay guarantees into startup/runtime contracts, not conventions.

For side-effect tools, unknown outcomes are fail-closed: Artana halts replay and requires
`kernel.reconcile_tool(...)` before continuing.

### Minimal Usage

```python
from pydantic import BaseModel

from artana import ArtanaKernel, KernelPolicy, ModelInput, TenantContext
from artana.store import SQLiteStore
from artana.ports.model import LiteLLMAdapter

class Decision(BaseModel):
    approved: bool
    reason: str

kernel = ArtanaKernel(
    store=SQLiteStore(":memory:"),
    model_port=LiteLLMAdapter(),
    middleware=ArtanaKernel.default_middleware_stack(),
    policy=KernelPolicy.enforced(),
)
tenant = TenantContext(
    tenant_id="org_1",
    capabilities=frozenset({"finance:read"}),
    budget_usd_limit=1.0,
)
run = await kernel.start_run(tenant=tenant)

step = await kernel.step_model(
    run_id=run.run_id,
    tenant=tenant,
    model="gpt-4o-mini",
    input=ModelInput.from_prompt("Should we approve?"),
    output_schema=Decision,
)
decision = step.output
```

## Detailed API Reference (Latest)

This section documents the current API surface.

### Imports

```python
from pydantic import BaseModel

from artana import (
    AgentRuntime,
    ArtanaKernel,
    KernelModelClient,
    KernelPolicy,
    ModelInput,
    TenantContext,
    WorkflowContext,
    json_step_serde,
    pydantic_step_serde,
)
from artana.kernel import CapabilityDeniedError, ReplayConsistencyError, ToolExecutionFailedError
from artana.ports.model import LiteLLMAdapter
from artana.store import SQLiteStore
```

### Core Data Types

```python
TenantContext(
    tenant_id: str,
    capabilities: frozenset[str] = frozenset(),
    budget_usd_limit: float,  # > 0.0
)

KernelPolicy(mode: Literal["permissive", "enforced"] = "permissive")
KernelPolicy.enforced() -> KernelPolicy

ModelInput.from_prompt(prompt: str) -> ModelInput
ModelInput.from_messages(messages: Sequence[ChatMessage], *, prompt: str | None = None) -> ModelInput
```

`StepModelResult` fields:
- `run_id: str`
- `seq: int`
- `output: BaseModel`
- `usage: ModelUsage`
- `tool_calls: tuple[ToolCall, ...]`
- `replayed: bool`

`StepToolResult` fields:
- `run_id: str`
- `seq: int`
- `tool_name: str`
- `result_json: str`
- `replayed: bool`

`PauseTicket` fields:
- `run_id: str`
- `ticket_id: str`
- `seq: int`
- `reason: str`

### ArtanaKernel

Constructor:

```python
ArtanaKernel(
    *,
    store: EventStore,
    model_port: ModelPort,
    tool_port: ToolPort | None = None,
    middleware: Sequence[KernelMiddleware] | None = None,
    policy: KernelPolicy | None = None,
)
```

Recommended deterministic middleware setup:

```python
ArtanaKernel.default_middleware_stack(
    *,
    pii: bool = True,
    quota: bool = True,
    capabilities: bool = True,
) -> tuple[KernelMiddleware, ...]
```

Lifecycle:

```python
await kernel.start_run(*, tenant: TenantContext, run_id: str | None = None) -> RunRef
await kernel.load_run(*, run_id: str) -> RunRef
await kernel.close() -> None
```

Model step:

```python
await kernel.step_model(
    *,
    run_id: str,
    tenant: TenantContext,
    model: str,
    input: ModelInput,
    output_schema: type[OutputT],
    step_key: str | None = None,
) -> StepModelResult[OutputT]
```

Notes:
- If a matching model cycle already exists for `(run_id, step_key, request shape)`, result is replayed with `replayed=True`.
- If no match exists, kernel appends `model_requested` then `model_completed`.

Tool registration and execution:

```python
@kernel.tool(requires_capability: str | None = None)
async def my_tool(...) -> str | ToolExecutionResult:
    ...

await kernel.step_tool(
    *,
    run_id: str,
    tenant: TenantContext,
    tool_name: str,
    arguments: BaseModel,
    step_key: str | None = None,
) -> StepToolResult

await kernel.reconcile_tool(
    *,
    run_id: str,
    tenant: TenantContext,
    tool_name: str,
    arguments: BaseModel,
    step_key: str | None = None,
) -> str
```

Notes:
- Tool execution is two-phase: `tool_requested` is appended before call, `tool_completed` after call.
- Unknown outcomes are fail-closed (`ToolExecutionFailedError`) and must be reconciled via `reconcile_tool(...)`.
- Stable `step_key` values are recommended for deterministic replay over time.

Human-in-the-loop boundaries:

```python
await kernel.pause(
    *,
    run_id: str,
    tenant: TenantContext,
    reason: str,
    context: BaseModel | None = None,
    step_key: str | None = None,
) -> PauseTicket

await kernel.resume(
    *,
    run_id: str,
    tenant: TenantContext,
    human_input: BaseModel | None = None,
) -> RunRef
```

Workflow runtime:

```python
await kernel.run_workflow(
    *,
    run_id: str | None,
    tenant: TenantContext,
    workflow: Callable[[WorkflowContext], Awaitable[WorkflowOutputT]],
) -> WorkflowRunResult[WorkflowOutputT]
```

`WorkflowRunResult` fields:
- `run_id: str`
- `status: Literal["complete", "paused"]`
- `output: WorkflowOutputT | None`
- `pause_ticket: PauseTicket | None`

Inside `WorkflowContext`:

```python
await ctx.step(
    *,
    name: str,
    action: Callable[[], Awaitable[StepT]],
    serde: StepSerde[StepT],
) -> StepT

await ctx.pause(
    reason: str,
    *,
    context: BaseModel | None = None,
    step_key: str | None = None,
) -> PauseTicket  # raises internal pause interrupt and returns via WorkflowRunResult
```

### KernelModelClient

```python
chat = KernelModelClient(kernel=kernel)

await chat.chat(
    *,
    run_id: str,
    tenant: TenantContext,
    model: str,
    prompt: str,
    output_schema: type[OutputT],
    step_key: str | None = None,
) -> StepModelResult[OutputT]
```

Notes:
- `KernelModelClient.chat` auto-starts the run if `run_id` does not exist.
- `chat(...)` delegates to `kernel.step_model(...)` and preserves replay behavior.
- `ChatClient` remains available as a compatibility alias to `KernelModelClient`.

### AgentRuntime

```python
runtime = AgentRuntime(kernel=kernel)
```

Notes:
- `AgentRuntime` is an optional loop layer above `ArtanaKernel`.
- Use `run_turn(...)` for one replay-safe model turn with message accumulation.
- Use `run_until(...)` with a policy callback to stop/continue the loop.

### Event Store

`SQLiteStore` implements `EventStore`:

```python
SQLiteStore(
    database_path: str,
    *,
    busy_timeout_ms: int = 5000,
    max_retry_attempts: int = 5,
    retry_backoff_seconds: float = 0.02,
)

await store.append_event(
    *,
    run_id: str,
    tenant_id: str,
    event_type: EventType,
    payload: EventPayload,
) -> KernelEvent

await store.get_events_for_run(run_id: str) -> list[KernelEvent]
await store.verify_run_chain(run_id: str) -> bool
await store.close() -> None

# Optional fast-path API used by QuotaMiddleware when available:
await store.get_model_cost_sum_for_run(run_id: str) -> float
```

### Error Surface

Common exceptions:
- `CapabilityDeniedError`: tenant lacks required capability for requested tool.
- `ToolExecutionFailedError`: tool failed, unknown outcome halt, or replayed non-success completion.
- `ReplayConsistencyError`: replay invariants broken (mismatched event history).
- `ValueError`: missing/unknown run, invalid lifecycle usage.

### Real OpenAI Usage

```bash
set -a; source .env; set +a
uv run python examples/02_real_litellm_chat.py
```

The script proves deterministic replay by calling `KernelModelClient.chat()` twice for the same `run_id` and asserting:

- second response has `replayed=True`
- event count does not grow on replay
- output is identical to the first response

### Golden Example

```bash
set -a; source .env; set +a
uv run python examples/golden_example.py
```

The golden example is the canonical implementation of Artana invariants for:

- kernel-issued run lifecycle (`start_run`)
- enforced middleware governance
- deterministic replay without duplicate events

### Workflow Runtime

Durable steps with replay:

```python
from artana import ArtanaKernel, TenantContext, WorkflowContext, json_step_serde

async def fetch_data() -> dict:
    return {"status": "ok"}

async def my_workflow(ctx: WorkflowContext) -> dict:
    data = await ctx.step(name="fetch", action=fetch_data, serde=json_step_serde())
    return data

result = await kernel.run_workflow(
    run_id=None,
    tenant=tenant,
    workflow=my_workflow,
)
# result.output, result.status ("complete" | "paused"), result.pause_ticket
```

See also: `examples/README.md`

---

*Artana provides durable, policy-enforced AI execution semantics — as a lightweight kernel that complements both agent frameworks and workflow engines.*
