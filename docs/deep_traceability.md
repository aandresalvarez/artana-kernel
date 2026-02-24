# Deep Traceability Guide

This document describes the deep traceability features added to Artana harness and kernel execution.

## Goals

Deep traceability is designed to make runs easy to inspect without breaking replay determinism:

- structured lifecycle boundaries
- hierarchical step linking
- stage-level cost and timeline visibility
- drift and validation trace channels
- query and streaming hooks for runtime observability

## Feature Map

### 1. Harness lifecycle events

Lifecycle events are first-class `EventType` entries:

- `harness_initialized`
- `harness_wake`
- `harness_stage`
- `harness_sleep`
- `harness_failed`

These are emitted by `BaseHarness.run(...)` around initialize/wake/work/sleep boundaries.

### 2. Structured trace channels (`run_summary`)

Trace channels use `RunSummaryPayload.summary_type` with a `trace::...` namespace.

Current built-in channels:

- `trace::state_transition`
- `trace::round`
- `trace::cost`
- `trace::cost_snapshot`
- `trace::drift`
- `trace::tool_validation`

You can emit additional custom channels with:

```python
await harness.write_summary(
    summary_type="trace::my_channel",
    payload={"k": "v"},
)
```

### 3. Step hierarchy (`parent_step_key`)

All kernel events support `parent_step_key`, including hash chaining for auditability.

Propagation exists through:

- model request/completion/replay-drift events
- tool request/completion/reconciliation events
- harness events and summary emissions

This allows tree reconstruction:

```text
run
  -> harness stage
    -> model step
    -> tool step
    -> trace summary
```

### 4. Automatic failure boundary

If harness execution or sleep fails, `harness_failed` is appended before raising.

Payload fields:

- `error_type`
- `message`
- `last_step_key`

### 5. Deterministic cost and timeline summaries

At stage close, harness emits:

- `trace::cost`
- `trace::cost_snapshot`

Both include:

- `stage`
- `round`
- `model_cost`
- `tool_cost`
- `total_cost`
- `logical_duration_ms`

`logical_duration_ms` is measured with monotonic clock and stored as deterministic ledger data.

### 6. Drift trace channel

When a model step has drift metadata, harness emits:

- `trace::drift`

Payload includes:

- `step_key`
- `drift_fields`
- `forked`

### 7. Live event streaming hook

`SQLiteStore` supports an optional async callback:

```python
from artana.store import SQLiteStore

async def on_event(event):
    print(event.seq, event.event_type.value)

store = SQLiteStore("artana_state.db", on_event=on_event)
```

The callback runs after each successful append.

### 8. Trace query API

Kernel exposes:

```python
summary = await kernel.explain_run(run_id)
```

Returned keys:

- `status`
- `last_stage`
- `last_tool`
- `drift_count`
- `drift_events`
- `failure_reason`
- `failure_step`
- `cost_total`

### 9. Trace level modes

Harness supports:

- `minimal`
- `stage`
- `verbose`

Usage:

```python
await harness.run(run_id="run_1", tenant=tenant, trace_level="verbose")
```

Behavior:

- `minimal`: no stage/verbose trace summaries, only core run behavior
- `stage`: lifecycle and stage-level trace summaries
- `verbose`: stage plus detailed tool/model validation summaries

## API surface

### Kernel

- `ArtanaKernel.explain_run(run_id)`
- `ArtanaKernel.get_latest_summary(...)` (compat helper)
- `ArtanaKernel.append_run_summary(..., parent_step_key=...)`
- `ArtanaKernel.append_harness_event(..., parent_step_key=...)`

### Harness

- `BaseHarness.run(..., trace_level=...)`
- `BaseHarness.emit_summary(..., parent_step_key=...)`
- `BaseHarness.run_model(..., parent_step_key=...)`
- `BaseHarness.run_tool(..., parent_step_key=...)`

### Store

- `SQLiteStore(..., on_event=...)`

## Typical tracing flow

For one harness run, you typically see:

1. `run_started`
2. `harness_initialized`
3. `harness_wake`
4. `harness_stage` (initialize/wake/work/sleep)
5. `trace::state_transition` summaries
6. model/tool events (if used)
7. `trace::round`, `trace::cost`, `trace::cost_snapshot`
8. `harness_sleep`
9. optional `harness_failed` (if exception)

## Determinism and safety notes

- all traces are ledger-backed events/summaries
- `parent_step_key` participates in event hash computation
- replay logic is preserved for model and tool execution
- trace channels are additive and do not alter kernel replay guarantees

## Reference tests

Traceability behavior is covered in:

- `tests/test_harness_layer.py`
- `tests/test_improvements_features.py`
- `tests/test_sqlite_store.py`
