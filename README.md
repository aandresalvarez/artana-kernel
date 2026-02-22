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
  - `chat` — single model call with structured output and tool execution
  - `execute_tool` — execute or replay a tool for a run
  - `reconcile_tool` — resolve `unknown_outcome` tool requests using original idempotency key
  - `pause_for_human` — request human-in-the-loop pause
  - `resume` — derive run state (paused, pending_tool, ready, complete)
  - `run_workflow` — durable workflow with `WorkflowContext`, step serde, replay
  - `default_middleware_stack` — deterministic middleware ordering helper
  - `KernelPolicy.enforced()` — startup guard requiring governance middleware
- Tool registration via `@kernel.tool(requires_capability="...")` decorator
- Middleware stack: `PIIScrubberMiddleware`, `QuotaMiddleware`, `CapabilityGuardMiddleware`
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
response = await kernel.chat(
    run_id=run.run_id,
    prompt="Should we transfer?",
    model="gpt-4o-mini",
    tenant=tenant,
    output_schema=Decision,
)
decision = response.output  # ChatResponse[OutputT]
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
│   ├── replay.py        # derive_run_resume_state, validate_tenant_for_run
│   ├── model_cycle.py   # get_or_execute_model_step
│   ├── tool_cycle.py    # execute_or_replay_tools_for_model, execute_tool_with_replay
│   ├── policies.py     # apply_prepare_model_middleware, enforce_capability_scope
│   └── types.py         # ChatResponse, PauseTicket, RunResumeState, KernelPolicy, RunHandle
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

from artana import ArtanaKernel, KernelPolicy, TenantContext
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

response = await kernel.chat(
    run_id=run.run_id,
    prompt="Should we approve?",
    model="gpt-4o-mini",
    tenant=tenant,
    output_schema=Decision,
)
decision = response.output
```

### Real OpenAI Usage

```bash
set -a; source .env; set +a
uv run python examples/02_real_litellm_chat.py
```

The script proves deterministic replay by calling `kernel.chat()` twice for the same `run_id` and asserting:

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
