# Artana Kernel (MVP Start)

Initial implementation for the Artana Kernel PRD:

- Strictly typed kernel and event models.
- Event-sourced SQLite store (`append_event`, `get_events_for_run`).
- Ports for model and tools, including a LiteLLM adapter.
- Core kernel APIs (`chat`, `execute_tool`, `pause_for_human`, `resume`).
- Middleware stack components (`PIIScrubberMiddleware`, `QuotaMiddleware`, `CapabilityGuardMiddleware`).
- Replay-safe two-phase tool execution and crash recovery.
- Cryptographic audit hash-chain on persisted events with verification support.
- Async pytest coverage for sequencing, replay, quotas, middleware filtering/redaction, recovery, and audit integrity.

## Quickstart

```bash
uv sync
uv run pytest
uv run mypy src
```
