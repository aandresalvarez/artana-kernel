# Changelog

All notable changes to `artana-kernel` are documented in this file.

## [Unreleased]

### Added

- `KernelModelClient.capabilities()` and `ModelClientCapabilities` for step-kwarg capability checks.
- Mixed-version `KernelModelClient.step(...)` compatibility fallback that retries without unsupported kwargs and emits a warning.
- Canonical run progress APIs:
  - `ArtanaKernel.get_run_progress(run_id)`
  - `ArtanaKernel.stream_run_progress(run_id, since_seq=0, follow=False, ...)`
- Progress types:
  - `RunProgressStatus`
  - `RunProgress`
- Postgres read-path retry hardening for transient connection-lifecycle failures
  (for example connection-closed mid-operation).
- Store schema metadata contract:
  - `StoreSchemaInfo`
  - `SQLiteStore.get_schema_info()`
  - `PostgresStore.get_schema_info()`
- Compatibility documentation:
  - `docs/compatibility_matrix.md`
- Terminal model event contract:
  - `model_terminal` event with `outcome` (`completed|failed|timeout|cancelled|abandoned`)
  - correlation fields (`model_cycle_id`, `source_model_requested_event_id`)
  - normalized diagnostics (`failure_reason`, `error_category`, `error_class`, `http_status`, `provider_request_id`, `elapsed_ms`)
- Deterministic replay behavior for failed model cycles (non-retrying by default) and explicit retry knob (`retry_failed_step=True`).
- Kernel stale-run cleanup API:
  - `ArtanaKernel.cleanup_stale_model_runs(...)`
  - synthetic terminal closure events for orphaned `model_requested` runs.
- Run state snapshot enrichments:
  - `error_category`
  - `diagnostics_json`
  - failure transitions on non-completed model terminal outcomes.

### Docs

- Expanded kernel contracts for run progress and mixed-version client behavior.
- Added release/compatibility references in `README.md`.
- Documented event-loop ownership expectations and FastAPI lifespan usage for long-lived kernel/store instances.
- Documented Postgres transient read retry semantics in contracts/compatibility docs.
- Updated contracts/traceability docs for `model_terminal`, failed-cycle replay behavior, and stale-run cleanup semantics.

## [0.1.0] - 2026-02-26

### Added

- Initial public release with durable kernel replay, tool safety policies, harness APIs, and run lifecycle/status surfaces.
