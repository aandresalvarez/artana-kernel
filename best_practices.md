Here is the updated **Artana Kernel — Mantra**. It has been expanded to codify the recent massive architectural leaps: **Harness Engineering**, **Replay Policies (Drift/Forking)**, **Tool IO Middleware**, and **Inter-Run Memory**. 

This document serves as the absolute engineering baseline for your team.

***

# Artana Kernel — Mantra

Guidelines for maintaining code quality, architecture integrity, and long-term maintainability. Align with the PRD for vision, stack, and data flow.

---

## 1. Single Responsibility

- **One module, one reason to change.** Each file or class should have a single, well-defined responsibility.
- Prefer small, focused functions over large ones. If a function does more than one thing, split it.
- When a module grows beyond its purpose, extract a new module rather than expanding the existing one.

---

## 2. Respect the Architecture (The Map vs. The Engine)

- **Ports & Adapters (Hexagonal):** Code must live in the correct layer. Do not bypass ports. The Agent/Harness layers must NEVER reach into `kernel._store` or `kernel._model_port` directly. Use public Kernel APIs.
- **The Kernel owns the Physics:** Durability, quota, capability routing, PII scrubbing, and event sourcing live in the `_kernel` and `middleware`. The Kernel *does not* own loops or prompts.
- **The Harness owns the Logic:** `artana.agent` and `artana.harness` own the control flow (`while` loops, strict `ctx.step` pipelines, context compaction, and learning).
- **Data flow:** Developer Code (Harness/Agent) → Kernel → Middleware Stack → Ports → Adapters → EventStore. 

---

## 3. Harness Engineering & Environment Discipline

- **Environments over Loops:** Treat Harnesses as durable, versioned environments. Use structured artifacts (via `RUN_SUMMARY` / `set_artifact`) to maintain state across resets, rather than relying solely on raw context windows.
- **Strict Lifecycle Hooks:** Subclasses of `BaseHarness` must respect `on_initialize`, `on_wake` (Session Reorientation), and `on_sleep` (Clean State Validation).
- **Incremental Discipline:** For complex autonomous tasks, enforce state machines (e.g., `IncrementalTaskHarness` using `TaskUnit`). Agents must not be allowed to go to sleep leaving their environment in a partial or broken state.

---

## 4. Strict Typing (No `Any`)

- **No `typing.Any`** in the core kernel. All inputs, outputs, state payloads, and configuration must be typed.
- Use **Generics** (`TypeVar`) or **explicit Pydantic models** for event payloads and port contracts.
- Code must pass **`mypy --strict`** (including `disallow_untyped_defs`, `disallow_any_generics`). Fix type errors; do not silence them with casts or ignores without a documented justification.

---

## 5. Events as the Source of Truth & Replay Semantics

- State is derived **only** from the append-only cryptographic event log. 
- **Tool Signature Hashing:** The Kernel validates replay not just on tool names, but on exact schema hashes (`ToolSignatureRecord`). If a tool's JSON schema changes, the Kernel must detect the drift.
- **Replay Policies are Explicit:** Replay is handled via explicit `ReplayPolicy` (`strict`, `allow_prompt_drift`, `fork_on_drift`).
- **Forking Timelines:** When using `fork_on_drift`, the Kernel creates a new timeline branch (`forked_from_run_id`) rather than destroying or corrupting the original run log.

---

## 6. IO Boundaries and Governance (Middleware)

- **Middleware order is absolute:** PII Scrubber → Quota Check → Capability Guard. New middleware must explicitly define its priority.
- **Tool IO is Untrusted:** PII scrubbing and governance do not just apply to LLM prompts. They apply to `prepare_tool_request` (before it hits the real world) and `prepare_tool_result` (before it re-enters the LLM context).
- **Tool Capabilities are strictly enforced:** Agents only see the tools permitted by their `TenantContext.capabilities`. The Kernel handles the filtering; the Harness never sees the unauthorized tools.

---

## 7. Context, Compaction, and Learning

- **The Kernel does not summarize.** Context window management is the responsibility of the Harness/Agent (via `CompactionStrategy`).
- **Experience is Inter-Run:** Repetitive tasks should utilize the `ExperienceStore` to automatically extract and inject `Win-Patterns` and `Anti-Patterns` across different run IDs.
- **Progressive Disclosure:** Large toolsets should not be injected raw. Use text menus and the `load_skill` pattern to save tokens and preserve the LLM's attention span.

---

## 8. Remote Execution & Distributed Safety

- **Never Block the Event Loop:** The kernel relies entirely on `asyncio` for high-concurrency multi-tenancy. All network calls **must** use asynchronous libraries.
- **Idempotency is Mandatory:** Tools that perform external mutations must accept and enforce an **idempotency key** derived from the `run_id`, `tool_name`, and `seq`.
- **Two-phase remote operations:** For any tool with side effects:
  - Append `TOOL_REQUESTED` (with args + idempotency_key) **before** sending the request.
  - Append `TOOL_COMPLETED` (with result/outcome) **after** the response.
- **Unknown Outcomes default to Halt:** If a worker dies mid-tool-execution and restarts, an unmatched `TOOL_REQUESTED` must result in a `ToolExecutionFailedError(unknown_outcome)` that forces developer reconciliation (`reconcile_tool_with_replay`). Never guess. Never double-execute.
- **Secrets never cross the contract boundary:** Tools receive capabilities and opaque references. Credentials are injected only inside the adapter at call time; event logs store only redacted inputs.

---

## 9. Quality Gates & Testing

- **All new behavior must have tests.**
- **Critical paths** that must be covered:
  - **Crash-recovery and replay:** simulate a crash mid-tool-call and verify deterministic recovery / `unknown_outcome` catches.
  - **Replay Drift:** Test `fork_on_drift` tree generation.
  - **Quota breach** and **capability denial** behavior.
  - **Pause/resume** (`pause_for_human`, `resume`) and event flushing.
- **CI** must run on every change: `mypy --strict`, `pytest`, `ruff`. Do not merge with failing checks.

---

## 10. Explicit Over Implicit & File Structure

- **Keep modules under ~300 lines for leaf modules.** If a file exceeds this, split by responsibility.
- **Orchestration exception:** Kernel/Harness/store orchestrators may exceed ~300 lines when they protect cross-cutting invariants (replay, lifecycle, durability). In those cases, extract parsing/serialization/policy helpers into dedicated modules and keep behavior guarded by tests.
- Configuration (budget, model, tenant) should be passed explicitly. Avoid global, hidden, or environmental configuration variables bleeding into the Kernel layer. 
- Use the PRD’s naming conventions: `TenantContext`, `KernelEvent`, `HarnessContext`, `TaskUnit`.

---

## Summary

| Principle | Action |
| :--- | :--- |
| **Single Responsibility** | One module, one reason to change; split when it grows. |
| **Architecture** | Harness owns Logic/Loops. Kernel owns Physics/Safety. Depend on Ports. |
| **Typing** | No `Any`; `mypy --strict` passes. |
| **Events & Replay** | Event log is the absolute source of truth. Manage drift explicitly via `ReplayPolicy` (`fork_on_drift`). |
| **Harness Discipline** | Enforce `on_wake` (reorient) and `on_sleep` (clean state). Use structured Artifacts. |
| **Governance** | Middleware protects BOTH Model I/O and Tool I/O. Capabilities determine tool visibility. |
| **Tests** | All new behavior tested; crash-recovery and replay covered. |
| **Distributed Safety** | Async I/O, two-phase tool logging, strict idempotency keys, unknown outcome halting. |
| **Memory** | Agent manages Compaction. `ExperienceStore` manages inter-run learning. |
| **CI** | mypy, pytest, lint/format on every change. |
