# Deep Traceability Guide

This document describes the deep traceability features added to Artana harness and kernel execution.

Code block contract for this document:

* `pycon` blocks are in-context snippets.
* Runnable end-to-end scripts live in `examples/` and target tests under `tests/`.

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

Agent-level run summaries also include:

- `agent_model_step`
- `agent_verify_step`
- `agent_acceptance_gate`
- `agent_tool_step`

You can emit additional custom channels with:

```pycon
await harness.write_summary(
    summary_type="trace::my_channel",
    payload={"k": "v"},
)
```

### 3. Step hierarchy (`parent_step_key`)

All kernel events support `parent_step_key`, including hash chaining for auditability.

Propagation exists through:

- model request/terminal/replay-drift events
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

### 7. Live event callbacks + streaming

Both stores support an optional async callback on append:

```pycon
from artana.store import PostgresStore, SQLiteStore

async def on_event(event):
    print(event.seq, event.event_type.value)

store = SQLiteStore("artana_state.db", on_event=on_event)
pg_store = PostgresStore("postgresql://user:pass@localhost:5432/artana", on_event=on_event)
```

The callback runs after each successful append.

Kernel also exposes store-agnostic event streaming:

```pycon
async for event in kernel.stream_events(run_id="run_1", since_seq=0, follow=True):
    print(event.seq, event.event_type.value)
```

### 7b. Draft/Verify + Acceptance Gate Trace Shape

When `AutonomousAgent` runs with `DraftVerifyLoopConfig` and `AcceptanceSpec`:

- draft model turns emit `summary_type=agent_model_step`
- gate tool checks emit `summary_type=agent_acceptance_gate`
- verifier turns emit `summary_type=agent_verify_step`

This allows deterministic analysis of why a run continued vs. finalized.

### 8. Trace query API

Kernel exposes:

```pycon
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

```pycon
await harness.run(run_id="run_1", tenant=tenant, trace_level="verbose")
```

Behavior:

- `minimal`: no stage/verbose trace summaries, only core run behavior
- `stage`: lifecycle and stage-level trace summaries
- `verbose`: stage plus detailed tool/model validation summaries

### 10. CLI Trace Inspection

Operational trace inspection is available through CLI commands:

```pycon
artana run status <run_id> --db .state.db --json
artana run summaries <run_id> --db .state.db --limit 20 --json
artana run artifacts <run_id> --db .state.db --json
artana run tail <run_id> --db .state.db --since-seq 0
```

Use `--dsn postgresql://...` for shared deployments.

### 11. External tracing status (deferred)

Current production path is ledger-native traceability:

- lifecycle events
- `trace::...` run summaries
- CLI/event-stream inspection

External tracing decorators (Logfire/OpenTelemetry) are intentionally deferred.
Trigger conditions for enabling them:

- need to correlate kernel spans with external services/APIs across process boundaries
- incident-response workflows requiring centralized distributed-trace timelines
- clear SLO/MTTR evidence that built-in ledger traces are insufficient

Until those triggers appear, Artana keeps observability deterministic and store-native.

## API surface

### Kernel

- `ArtanaKernel.explain_run(run_id)`
- `ArtanaKernel.get_latest_summary(...)` (compat helper)
- `ArtanaKernel.append_run_summary(..., parent_step_key=...)`
- `ArtanaKernel.append_harness_event(..., parent_step_key=...)`
- `ArtanaKernel.stream_events(run_id, since_seq=0, follow=False, ...)`
- `ArtanaKernel.describe_capabilities(tenant=...)`
- `ArtanaKernel.list_tools_for_tenant(tenant=...)`

### Harness

- `BaseHarness.run(..., trace_level=...)`
- `BaseHarness.emit_summary(..., parent_step_key=...)`
- `BaseHarness.run_model(..., model_options=..., parent_step_key=...)`
- `BaseHarness.run_draft_model(..., model_options=..., parent_step_key=...)`
- `BaseHarness.run_verify_model(..., model_options=..., parent_step_key=...)`
- `BaseHarness.run_tool(..., parent_step_key=...)`

### Store

- `SQLiteStore(..., on_event=...)`
- `PostgresStore(..., on_event=...)`

## Typical tracing flow

For one harness run, you typically see:

1. `run_started`
2. `harness_initialized`
3. `harness_wake`
4. `harness_stage` (initialize/wake/work/sleep)
5. `trace::state_transition` summaries
6. model/tool events (if used)
7. optional agent summaries (`agent_model_step`, `agent_acceptance_gate`, `agent_verify_step`)
8. `trace::round`, `trace::cost`, `trace::cost_snapshot`
9. `harness_sleep`
10. optional `harness_failed` (if exception)

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
- `tests/test_postgres_store.py`
