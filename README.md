

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

## What This MVP Implements

Initial implementation aligned with the Artana Kernel PRD:

- Strictly typed kernel and event models (`mypy --strict`)
- Event-sourced SQLite store (`SQLiteStore`) with hash-chained cryptographic ledgers
- Ports for model and tools:
  - `LiteLLMAdapter` (implements `ModelPort`)
  - `LocalToolRegistry` (implements `ToolPort`)
- Core kernel APIs:
  - `start_run` / `load_run` — authoritative run lifecycle
  - `step_model` / `step_tool` — deterministic, atomic execution primitives
  - `reconcile_tool` — resolve `unknown_outcome` tool requests safely
  - `pause` / `resume` — human-in-the-loop durable interrupts
- **The Agent Runtime:**
  - `AutonomousAgent` — An out-of-the-box `Model -> Tool -> Model` loop that automatically executes tools and manages conversation memory on top of the Kernel.
  - `CompactionStrategy` — proactive context compaction (message-count and token-threshold triggers) with replay-safe summaries.
  - `ContextBuilder` — pluggable prompt assembly pipeline (identity + long-term memory + progressive skills + short-term history).
  - Progressive skill disclosure via built-in `load_skill(...)` meta-tool.
  - Built-in long-term memory tools: `core_memory_append`, `core_memory_replace`, `core_memory_search`.
  - `SubAgentFactory` — sub-agent delegation with run lineage (`parent::sub_agent::idempotency_key`) and shared tenant governance.
- **The Workflow Runtime:**
  - `run_workflow` — Durable workflow with `WorkflowContext`, deterministic Python logic execution (`ctx.step`), and `ctx.pause`.
- Tool registration via `@kernel.tool(requires_capability="...")`
- Middleware stack: `PIIScrubberMiddleware`, `QuotaMiddleware`, `CapabilityGuardMiddleware`
- Replay-safe two-phase tool execution (protects against double-charging/executing on crash recovery)

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
    SQLiteMemoryStore,
)

memory_store = SQLiteMemoryStore("agent_memory.db")
context_builder = ContextBuilder(
    identity="You are a senior data analyst.",
    memory_store=memory_store,
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
)
```

### 2. The Durable Workflow (Manual Control)
Use this when you need strict, deterministic pipelines with Human-In-The-Loop pauses and pure Python logic. The LLM is restricted to single extraction steps.

```python
async def my_workflow(ctx: WorkflowContext):
    # 1. Atomic LLM call
    facts = await chat.chat(..., step_key="extract")
    
    # 2. Deterministic Python logic (cached to DB)
    derived = await ctx.step(name="math", action=run_math, serde=...)
    
    # 3. Durable Pause (Server can safely restart here)
    await ctx.pause(reason="Human approval required", context=derived)
    
    return {"status": "saved"}

result = await kernel.run_workflow(run_id="run_1", tenant=tenant, workflow=my_workflow)
```

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
                 ↓
    [ Port Interfaces ]
       ↙                 ↘
[ ModelPort ]         [ ToolPort ]
 (LiteLLM)         (Local Function Registry)
       ↘                 ↙
   [ EventStore (SQLiteStore) ]
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
ModelInput.from_prompt(prompt: str) -> ModelInput
ModelInput.from_messages(messages: Sequence[ChatMessage]) -> ModelInput
```

### ArtanaKernel Initialization
```python
kernel = ArtanaKernel(
    store=SQLiteStore("artana_state.db"),
    model_port=LiteLLMAdapter(),
    middleware=ArtanaKernel.default_middleware_stack(),
    policy=KernelPolicy.enforced(),
)
```

### Tool Registration
```python
@kernel.tool(requires_capability="finance:write")
async def transfer_funds(account_id: str, amount: str) -> str:
    return "Success"
```

### Autonomous Agent API
A runtime wrapper over the kernel that manages conversation history, automatic tool execution,
context compaction, long-term memory, and progressive skill disclosure.
```python
from artana.agent import AutonomousAgent, CompactionStrategy, ContextBuilder
from artana.agent.memory import SQLiteMemoryStore

memory_store = SQLiteMemoryStore("agent_memory.db")
context_builder = ContextBuilder(memory_store=memory_store, progressive_skills=True)

agent = AutonomousAgent(
    kernel=kernel,
    context_builder=context_builder,
    compaction=CompactionStrategy(trigger_at_messages=40, keep_recent_messages=10),
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

### Sub-Agent Delegation API
Create tools that run specialized child agents with inherited governance and durable lineage.
```python
from artana.agent import SubAgentFactory

factory = SubAgentFactory(kernel=kernel, tenant=tenant)

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

## Examples

Run examples from the repository root:

- **`01_durable_chat_replay.py`**: Basic capability-scoped tool execution and durable replay.
- **`02_real_litellm_chat.py`**: Real OpenAI calls proving replay invariants without duplicate network requests.
- **`03_fact_extraction_triplets.py`**: Single-step structured extraction.
- **`04_autonomous_agent_research.py`**: Demonstrates the `AutonomousAgent` loop dynamically selecting tools to accomplish a goal.
- **`05_hard_triplets_workflow.py`**: Demonstrates the strict `run_workflow` pattern interleaving LLM calls, deterministic Python math, and a durable Human-In-The-Loop pause.
- **`06_triplets_swarm.py`**: Demonstrates multi-agent orchestration (lead agent + extractor/adjudicator sub-agents + deterministic graph math tool) on the shared Kernel ledger.
- **`golden_example.py`**: Canonical production-leaning example testing unknown tool outcomes and reconciliation.

## Growth Path

- **Phase 1 (MVP - Current):** SQLite, single worker, strict replay semantics, Autonomous Agent runtime (compaction/memory/progressive skills/sub-agents), Workflow contexts.
- **Phase 2:** Postgres backend (for multi-worker concurrency), Snapshotting, Advanced Observability (Logfire tracing decorators).
- **Phase 3:** Optional integration with Temporal, distributed execution model, enterprise policy engines (OPA).
 
