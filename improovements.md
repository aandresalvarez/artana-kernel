Excellent.
Let’s treat this like an RFC.

Below is a structured architecture document you can share with your team.

---

# RFC: Deep Traceability for Artana Harnesses

**Status:** Proposal
**Audience:** Core Kernel & Harness Engineering
**Goal:** Make harness execution fully observable, debuggable, replay-auditable, and production-traceable without breaking determinism.

---

# 1. Executive Summary

Artana already provides:

* Deterministic event ledger
* Replay safety
* Step-level `step_key`
* Tool idempotency
* Drift detection
* Budget enforcement
* Structured summaries

However, when a harness fails in production, developers currently need to manually inspect raw events.

We propose introducing **Deep Harness Traceability**, a structured observability layer on top of the deterministic ledger.

The objective is:

> Every harness execution should be explainable in 30 seconds.

---

# 2. Problem Statement

Current challenges:

1. Raw event log is too low-level for fast debugging.
2. Harness lifecycle phases are not explicitly marked in ledger.
3. No structured trace boundaries per logical stage.
4. Failures are captured implicitly (exceptions), not as structured trace artifacts.
5. No hierarchy between harness → model → tool → workflow steps.
6. No timeline-level view.
7. No cost-per-stage visibility.
8. No live trace streaming.

We need traceability without:

* Breaking replay determinism
* Introducing side-channel state
* Weakening audit guarantees
* Polluting core kernel abstraction

---

# 3. Design Principles

1. Deterministic first — tracing must be ledger-backed.
2. Structured, not textual — no freeform logs.
3. Step-boundary aware.
4. Harness lifecycle visible.
5. Cheap to query.
6. Scalable for distributed workers.
7. Zero runtime performance penalty in non-debug mode.

---

# 4. Proposed Features

---

# Feature 1 — Harness Lifecycle Events

## Problem

We currently have no first-class events for:

* Harness initialization
* Wake phase
* Sleep phase
* Structured failure boundaries

## Proposal

Add new event types:

```python
class EventType(StrEnum):
    HARNESS_INITIALIZED
    HARNESS_WAKE
    HARNESS_SLEEP
    HARNESS_FAILED
    HARNESS_STAGE
```

### Example Event

```json
{
  "event_type": "harness_stage",
  "payload": {
    "stage": "proposer",
    "round": 2,
    "claims_count": 12
  }
}
```

### Why?

* Clear lifecycle segmentation
* Faster debugging
* Structured observability
* No need to inspect raw model events

---

# Feature 2 — Structured Debug Trace Channel

Introduce a new summary type namespace:

```
summary_type="trace::<channel>"
```

Examples:

* trace::round
* trace::tool_validation
* trace::cost_snapshot
* trace::state_transition

This avoids adding new event types while preserving structure.

Example:

```python
await harness.write_summary(
    summary_type="trace::round",
    payload={
        "round": 3,
        "confidence": 0.71,
        "claims": 9,
    },
)
```

Benefits:

* Lightweight
* Backward compatible
* Fully ledger-backed

---

# Feature 3 — Step Hierarchy Indexing

Add structured trace hierarchy:

```
run
 ├── harness lifecycle
 │    ├── stage
 │    │    ├── model call
 │    │    ├── tool call
 │    │    └── summary
```

Implementation:

Add parent_step_key field to events.

Example:

```json
{
  "step_key": "round_2.proposer",
  "parent_step_key": "round_2"
}
```

This enables:

* Tree reconstruction
* Stage-level tracing
* Visual trace viewer
* Span grouping

---

# Feature 4 — Automatic Failure Boundary Emission

Currently failures throw exceptions.

Proposal:

BaseHarness.run() should emit structured failure event before raising.

```python
{
  "event_type": "harness_failed",
  "payload": {
    "error_type": "ValidationError",
    "message": "...",
    "last_step_key": "round_3.repair"
  }
}
```

Benefits:

* No silent crash ambiguity
* Exact failure boundary logged
* Fast post-mortem

---

# Feature 5 — Deterministic Cost Breakdown

Add optional automatic cost aggregation per stage.

Emit:

```
summary_type="trace::cost"
```

Payload:

```json
{
  "stage": "round_2",
  "model_cost": 0.012,
  "tool_cost": 0.0,
  "total_cost": 0.012
}
```

Implementation:

* Track cost_usd from MODEL_COMPLETED
* Associate with current harness stage

Benefits:

* Cost visibility per debate round
* Budget spike detection
* Optimization guidance

---

# Feature 6 — Drift Trace Channel

Currently drift is emitted as REPLAYED_WITH_DRIFT.

Enhancement:

Emit structured drift summary:

```
summary_type="trace::drift"
```

Payload:

```json
{
  "step_key": "round_3.proposer",
  "drift_fields": ["prompt"],
  "forked": false
}
```

This makes drift visually searchable.

---

# Feature 7 — Live Event Streaming Hook

Extend EventStore to support optional on_event callback:

```python
class EventStore:
    def __init__(..., on_event: Callable[[KernelEvent], Awaitable[None]] | None = None)
```

This enables:

* WebSocket streaming
* Console tracing
* Prometheus metrics
* OpenTelemetry spans
* Debug CLI

Example usage:

```python
async def on_event(event):
    print(event.event_type, event.seq)

store = SQLiteStore("db", on_event=on_event)
```

No core logic changes required.

---

# Feature 8 — Trace Query API

Add high-level helper:

```python
await kernel.explain_run(run_id)
```

Returns:

```python
{
  "status": "failed",
  "last_stage": "round_3.repair",
  "last_tool": "verify_claim_tool",
  "cost_total": 0.32,
  "drift_events": 1,
  "failure_reason": "ValidationError"
}
```

This is built purely from ledger.

---

# Feature 9 — Trace Level Modes

Add optional harness parameter:

```python
trace_level="minimal" | "stage" | "verbose"
```

* minimal → only failures
* stage → stage boundaries
* verbose → every model/tool/summary

No change to determinism — only additional summaries emitted.

---

# Feature 10 — Deterministic Timeline View

Introduce optional timestamp normalization:

Add logical_duration_ms in summary:

```json
{
  "stage": "round_2",
  "duration_ms": 1840
}
```

Measured inside harness, stored in ledger.

Enables:

* Latency profiling
* Bottleneck detection
* Production SLO tracking

---

# 5. Proposed Minimal Implementation Plan

Phase 1 (Low Risk):

* Add harness_failed summary
* Add trace::round channel
* Add cost snapshot summary
* Add explain_run helper

Phase 2 (Medium):

* Add parent_step_key support
* Add live on_event hook
* Add drift trace channel

Phase 3 (Advanced):

* OpenTelemetry integration
* Prometheus metrics exporter
* Web UI trace explorer
* Multi-run correlation viewer

---

# 6. Backward Compatibility

All features:

* Ledger-based
* Optional
* Do not break replay
* Do not modify core determinism logic
* Do not change event hashing semantics

---

# 7. Example: Fully Traced Debate Round

Ledger would contain:

```
RUN_STARTED
HARNESS_INITIALIZED
HARNESS_WAKE
trace::round
MODEL_REQUESTED (round_1.proposer)
MODEL_COMPLETED
MODEL_REQUESTED (round_1.critic)
MODEL_COMPLETED
trace::cost
trace::drift (if any)
HARNESS_SLEEP
```

A developer can answer:

* Where did it fail?
* Which model call?
* Which tool?
* Which stage?
* How much did it cost?
* Was there drift?
* Was there replay?

In under 30 seconds.

---

# 8. Final Outcome

With these features, Artana becomes:

> A fully auditable, traceable execution substrate for long-running intelligent systems.

Not just deterministic.

Not just replay-safe.

But **deeply observable.**

 