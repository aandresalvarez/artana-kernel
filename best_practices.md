# Artana Kernel — Mantra

Guidelines for maintaining code quality, architecture integrity, and long-term maintainability. Align with the [PRD](prd.md) for vision, stack, and data flow.

---

## 1. Single Responsibility

- **One module, one reason to change.** Each file or class should have a single, well-defined responsibility.
- Prefer small, focused functions over large ones. If a function does more than one thing, split it.
- When a module grows beyond its purpose, extract a new module rather than expanding the existing one.

---

## 2. Respect the Architecture

- **Ports & Adapters (Hexagonal):** Code must live in the correct layer. Do not bypass ports (e.g. kernel or middleware must not call SQLite or LiteLLM directly).
- **Middleware order** is defined in the PRD (PII Scrubber → Quota Check → Capability Guard). New middleware must plug into this stack explicitly.
- **Data flow:** Developer code → Kernel → Middleware → Ports → Adapters → EventStore. Depend on **port interfaces** (`ModelPort`, `ToolPort`, event store interface), not on concrete adapters.
- Use the PRD’s naming and placement: `TenantContext`, `KernelEvent`, event type literals, and the documented module roles.

---

## 3. Strict Typing (No `Any`)

- **No `typing.Any`** in the core kernel. All inputs, outputs, state payloads, and configuration must be typed.
- Use **Generics** (`TypeVar`) or **explicit Pydantic models** for event payloads and port contracts.
- Code must pass **`mypy --strict`** (including `disallow_untyped_defs`, `disallow_any_generics`). Fix type errors; do not silence them with casts or ignores without a documented justification.

---

## 4. Events as the Source of Truth

- State is derived **only** from the append-only event log. No hidden mutable state that is not reflected in events.
- **Replay must be deterministic:** given the same event sequence, resume must produce the same behavior.
- Two-phase tool execution: emit `TOOL_REQUESTED` before execution and `TOOL_COMPLETED` after. Never skip or reorder events for convenience.

---

## 5. Always Write Tests

- **All new behavior must have tests.** Prefer unit tests for pure logic; integration tests for kernel, store, and middleware interactions.
- **Critical paths** that must be covered:
  - **Crash-recovery and replay:** simulate a crash mid-tool-call and verify deterministic recovery.
  - **Quota breach** and **capability denial** behavior.
  - **Pause/resume** (`pause_for_human`, `resume`) and event flushing.
- Use **pytest** with **pytest-asyncio** for async code. Treat test coverage as a first-class requirement.

---

## 6. File Size and Structure

- **Keep modules under ~300 lines.** If a file exceeds this, split by responsibility (e.g. separate event types, middleware, or port implementations).
- The threshold is a guideline to avoid monolithic files; the real rule is single responsibility. A file under 300 lines that does too many things should still be split.

---

## 7. Failure and Recovery

- **Document and test** failure modes: what happens on quota breach, capability denial, store failure, or LLM timeout.
- Critical paths (pause/resume, quota, capabilities) should have a short comment or docstring when behavior is non-obvious, and must have corresponding tests.

---

## 8. Explicit Over Implicit

- Prefer explicit capability requirements on tools, typed `TenantContext`, and explicit event types over magic strings or implicit defaults.
- Configuration (budget, model, tenant) should be passed explicitly; avoid global or hidden configuration.

---

## 9. Quality Gates

- **CI** must run on every change:
  - `mypy --strict`
  - `pytest`
  - Linter/formatter (e.g. ruff) as configured in the project.
- Do not merge with failing type-check or tests; do not disable checks without team agreement and a tracked follow-up.

---

## 10. Remote Execution & Distributed Safety

- **Never Block the Event Loop:** The kernel relies entirely on `asyncio` for high-concurrency multi-tenancy. All network calls (LLMs, databases, remote tools) **must** use asynchronous libraries (`httpx`, `aiosqlite`). A single synchronous HTTP request will stall the entire kernel for all users.
- **Idempotency is Mandatory:** Tools execute across network boundaries. If a connection drops, the Kernel cannot know if the remote server processed the request. Tools that perform external mutations (e.g., payments, writes) must accept and enforce an **idempotency key** derived from the `run_id` and `seq`.
- **Defensive Networking & Retries:** Assume the network will fail. All remote adapters must enforce strict timeouts. Wrap remote calls in exponential backoff/retry logic for transient errors (429s, 503s) before failing the run.
- **Serializable Boundaries:** Data crossing the network must be strictly JSON-serializable Pydantic models. Do not pass bare Python objects or functions across the `ToolPort` boundary.
- **Zero-Trust Logging:** Never serialize API keys or credentials into the `EventStore`. The event log is immutable; credentials must be injected by the Port at the moment of the network request and scrubbed from the logged payload.
- **Context Propagation:** Inject `run_id` and `event_id` into the HTTP headers of all outbound requests to remote tools to ensure distributed traceability.
- **Worker Statelessness:** Assume the Kernel process will be killed at any millisecond. Never store run-specific state in local memory expecting it to be there for the next step.
- **Exactly-Once *effects* are a contract (not a hope):** The Kernel guarantees *deterministic replay*, but remote systems can only guarantee *idempotent effects*. Every remote "write" must be designed so repeated delivery does not repeat the side effect (dedupe key, upsert semantics, conditional writes, Stripe-style idempotency, etc.).
- **Two-phase remote operations are mandatory:** For any tool with side effects:
  - append `TOOL_REQUESTED` (with args + idempotency_key) **before** sending the request;
  - append `TOOL_COMPLETED` (with result or failure classification) **after** the response;
  - on recovery, if `TOOL_REQUESTED` exists without `TOOL_COMPLETED`, the default is **halt + reconcile** unless the tool declares safe retry semantics.
- **Classify errors and retries by policy, not by vibes:** Every remote adapter must map failures into a small, typed set:
  - `Transient` (retryable: 429/503/timeouts);
  - `Permanent` (fail fast: 4xx validation/authz);
  - `UnknownOutcome` (request may have succeeded; requires reconciliation).
  Retries are allowed only for `Transient`, and must be bounded (max attempts + max total wall time).
- **Timeouts are layered and explicit:** Define separate budgets for: connect timeout, request timeout, overall operation deadline, stream idle timeout (no chunks received). Timeouts are part of the adapter config and must be covered by tests.
- **Backpressure is a first-class feature:** Remote calls must not be unbounded. Enforce per-tenant concurrency limits (semaphore), global worker caps, and queues with bounded size and clear shedding behavior (reject vs degrade).
- **Protocol versioning is mandatory at the boundary:** Every remote tool call must include `tool_version` (semantic version) and `schema_version` (payload contract version), with strict validation on both sides. Never "best-effort parse" across versions; fail with a typed `ContractViolation`.
- **Remote responses must be self-describing and auditable:** Require remote tools to return: `result` (typed), `effect_id` (remote-side unique identifier, if side effects occur), `received_idempotency_key`, `server_time` / `request_id` (for support + correlation). Store these in `TOOL_COMPLETED` payload (redacted where necessary).
- **Secrets never cross the contract boundary:** Tools receive **capabilities** and **opaque references**, not raw credentials. Credentials are injected only inside the adapter at call time; event logs store only redacted inputs + opaque secret references (if any).
- **Reconciliation is a supported mode, not an incident:** Define a standard reconciliation hook: `tool.reconcile(run_id, seq, idempotency_key) -> ToolCompletedPayload | ReconcilePending`, so operators have a consistent way to resolve `UnknownOutcome` without hacking the DB.
- **Streaming is treated as a remote distributed system:** If responses stream: either store chunks deterministically, or store the final assembled response + usage metadata. If you cannot guarantee deterministic reconstruction, disable streaming for that adapter.

---

## Summary

| Principle              | Action |
|------------------------|--------|
| Single responsibility  | One module, one reason to change; split when it grows. |
| Architecture           | Respect PRD layers and ports; depend on interfaces. |
| Typing                 | No `Any`; `mypy --strict` passes. |
| Events                 | Event log is the only source of truth; replay is deterministic. |
| Tests                  | All new behavior tested; crash-recovery and replay covered. |
| File size              | ~300 lines per file; split by responsibility. |
| Failure paths          | Document and test quota, capabilities, pause/resume. |
| Explicit over implicit | Typed context, explicit config, no magic. |
| CI                     | mypy, pytest, lint/format on every change. |
| **Distributed Safety** | **Async I/O, idempotent two-phase effects, typed retry policy, layered timeouts, backpressure, versioned boundaries, auditable responses, reconciliation hook, no secrets in events.** |
