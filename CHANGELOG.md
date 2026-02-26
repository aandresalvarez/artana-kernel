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
- Store schema metadata contract:
  - `StoreSchemaInfo`
  - `SQLiteStore.get_schema_info()`
  - `PostgresStore.get_schema_info()`
- Compatibility documentation:
  - `docs/compatibility_matrix.md`

### Docs

- Expanded kernel contracts for run progress and mixed-version client behavior.
- Added release/compatibility references in `README.md`.

## [0.1.0] - 2026-02-26

### Added

- Initial public release with durable kernel replay, tool safety policies, harness APIs, and run lifecycle/status surfaces.
