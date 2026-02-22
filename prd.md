Here is the detailed Product Requirements Document (PRD) for the **Artana Kernel MVP**. This document serves as the foundational blueprint for your engineering team to begin development immediately.

***

# Product Requirements Document (PRD): Artana Kernel MVP

## 1. Vision & Identity
**Artana** is a policy-enforced, event-sourced execution kernel for autonomous AI agents. 

As LLMs (like o1/o3, GPT-5) natively absorb complex planning, reasoning, and routing, traditional DAG-based workflow orchestrators are becoming obsolete. However, enterprise constraints remain absolute: AI cannot operate in production without strict budget caps, cryptographic audit logs, multi-tenant isolation, data redaction (PII), and crash-safe resumability.

Artana abandons the concept of "managing how the AI thinks" (no YAML DAGs, no AST parsing) and focuses entirely on "governing what the AI does." It acts as a Hypervisor, allowing developers to write standard, native Python code while Artana invisibly wraps all I/O in a durable, governed transaction layer.

## 2. Technical Stack & Strict Constraints
To ensure a high-performance, maintainable, and enterprise-grade foundation, the MVP will strictly adhere to the following stack and constraints:

*   **Project Management:** `uv` (for lightning-fast dependency resolution, locking, and virtualenv management).
*   **Language:** Modern Python (3.12+).
*   **LLM Interface:** `LiteLLM` (Replaces `pydantic-ai`. We require raw control over message histories and tool-call payloads to ensure deterministic event-sourcing. LiteLLM provides standard normalization across OpenAI, Anthropic, Gemini, etc., without obscuring the execution loop).
*   **Data Validation:** `pydantic` (v2).
*   **State Backend (MVP):** `sqlite3` / `aiosqlite`. Designed to be easily swapped for Postgres in v1.1.
*   **Testing:** `pytest` with `pytest-asyncio`. Test coverage must be treated as a first-class citizen, specifically testing crash-recovery and replay mechanics.
*   **The "No `Any`" Rule (Strict Typing):** The codebase will enforce `mypy --strict`. The use of `typing.Any` is strictly forbidden in the core kernel. All inputs, outputs, state payloads, and configurations must be typed using Generics (`TypeVar`) or explicit `pydantic.BaseModel` schemas.

## 3. Core Enterprise Features (Scope of MVP)

### A. Event-Sourced Durable Execution (Crash-Safety & HITL)
*   State is not saved as a "snapshot." It is saved as an append-only sequence of atomic events (`MODEL_REQUESTED`, `MODEL_COMPLETED`, `TOOL_REQUESTED`, etc.).
*   **Two-Phase Tool Execution:** To prevent re-executing side-effects (e.g., charging a credit card twice), tools emit a `TOOL_REQUESTED` event *before* execution and a `TOOL_COMPLETED` event *after*.
*   **Human-In-The-Loop (HITL):** The kernel provides a `pause_for_human()` interrupt that halts the Python process and flushes state to SQLite. A `resume()` function fast-forwards the event log, skipping already-completed LLM and Tool calls, and resumes native Python execution exactly where it left off.

### B. Multi-Tenant Capability Governance
*   Every run requires a strictly typed `TenantContext` (defining tenant ID, budget, and capabilities).
*   Tools are registered with explicit `requires_capability` flags. The Kernel strips unauthorized tools from the LLM's prompt before the request is sent over the network.

### C. Resource Budgeting (Quotas)
*   Runs are initialized with a `budget_usd_limit`.
*   A Quota middleware tracks tokens via `LiteLLM` response metadata, calculates exact cost, and aborts the run if the threshold is breached mid-execution.

### D. Cryptographic Auditability
*   The event ledger acts as an irrefutable "Flight Recorder." Every event includes a timestamp, sequence ID, and tenant ID.

## 4. Architecture & Data Flow

The architecture follows a strict **Ports & Adapters (Hexagonal)** pattern combined with a **Middleware Interceptor** stack.

```text
[ Developer's Standard Python Code ]
                 ↓
      ( calls kernel.chat() )
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

## 5. API Design Specification

The developer experience must feel like a standard library, not a heavy framework.

### Setup & Tool Registration
```python
from artana.kernel import ArtanaKernel
from artana.store import SQLiteStore
from artana.ports.model import LiteLLMAdapter
from artana.middleware import QuotaMiddleware

# 1. Boot the Kernel
kernel = ArtanaKernel(
    store=SQLiteStore("artana_state.db"),
    model_port=LiteLLMAdapter(),
    middleware=[QuotaMiddleware()]
)

# 2. Register tools with capability requirements
@kernel.tool(requires_capability="finance:read")
async def get_balance(account_id: str) -> str:
    return "Balance is $10,000"
```

### Application Code (The Workflow)
```python
from artana.models import TenantContext
from pydantic import BaseModel

class TransferDecision(BaseModel):
    approved: bool
    reason: str

async def transfer_workflow(user_prompt: str, run_id: str | None = None):
    tenant = TenantContext(
        tenant_id="org_123",
        capabilities=["finance:read", "finance:write"],
        budget_usd_limit=0.50
    )
    
    # 3. Kernel chat enforces schema, permissions, and writes events to SQLite
    decision_response = await kernel.chat(
        run_id=run_id,
        prompt=user_prompt,
        model="gpt-4o-mini",
        tenant=tenant,
        output_schema=TransferDecision # Strict Pydantic enforcement
    )
    
    decision: TransferDecision = decision_response.output
    
    if not decision.approved:
        # 4. Durable Pause: Stops execution, flushes state.
        pause_ticket = await kernel.pause_for_human(
            run_id=run_id,
            reason=f"Requires human review: {decision.reason}"
        )
        return {"status": "paused", "ticket": pause_ticket}
    
    return {"status": "complete"}
```

## 6. Data Model: The Event Schema

To enforce strict typing (`no Any`), the event payload uses a discriminated union of strictly defined Pydantic models.

```python
from typing import Generic, TypeVar, Literal
from pydantic import BaseModel, Field
from datetime import datetime, timezone

class BaseEventPayload(BaseModel):
    pass

class ModelRequestedPayload(BaseEventPayload):
    model: str
    messages: list[dict[str, str]]

class ToolRequestedPayload(BaseEventPayload):
    tool_name: str
    arguments: str # JSON string of args

class ToolCompletedPayload(BaseEventPayload):
    tool_name: str
    result: str

EventPayloadT = TypeVar("EventPayloadT", bound=BaseEventPayload)

class KernelEvent(BaseModel, Generic[EventPayloadT]):
    event_id: str
    run_id: str
    tenant_id: str
    seq: int
    event_type: Literal["model_requested", "tool_requested", "tool_completed", "pause_requested"]
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    payload: EventPayloadT
```

## 7. Execution Plan (Milestones)

### Step 1: Project Scaffolding (Day 1)
*   Initialize project with `uv init artana-kernel`.
*   Set up `pyproject.toml` with dependencies: `pydantic`, `litellm`, `aiosqlite`, `pytest`, `mypy`.
*   Configure `mypy` for absolute strictness (`strict = true`, `disallow_untyped_defs = true`, `disallow_any_generics = true`).

### Step 2: The Core Interfaces & State (Days 2-3)
*   Define `TenantContext` and the `KernelEvent` Pydantic models.
*   Implement `SQLiteStore` with `append_event` and `get_events_for_run` methods.
*   Write `pytest` suite to verify database locks and sequential event retrieval.

### Step 3: Ports & The Execution Loop (Days 4-6)
*   Implement `LiteLLMAdapter` implementing `ModelPort`. Ensure it maps LiteLLM responses into typed Artana structures.
*   Implement `LocalToolRegistry` implementing `ToolPort`.
*   Build the `ExecutionKernel.chat()` loop:
    1. Check Tenant capabilities against tools.
    2. Write `MODEL_REQUESTED` event.
    3. Call `LiteLLM`.
    4. Write `MODEL_COMPLETED` event.
    5. Handle Tool Calls (Write `TOOL_REQUESTED` -> Execute -> Write `TOOL_COMPLETED`).

### Step 4: Middleware & Replay (Days 7-10)
*   Implement `QuotaMiddleware` using pre/post request hooks.
*   Implement the `resume()` logic: When `kernel.chat()` or `kernel.execute_tool()` is called, it checks the SQLite event log via `run_id`. If a matching event exists for the current sequence step, it returns the cached response instead of executing the network/function call.
*   Write integration tests simulating a crashed process mid-tool-call and verify deterministic recovery.