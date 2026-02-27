# Compatibility Matrix

This matrix defines runtime API and store schema compatibility for `0.x` releases.

## Runtime API compatibility

| Package version | Runtime API contract | Notes |
| --- | --- | --- |
| `0.1.0` | Baseline kernel lifecycle APIs (`get_run_status`, `list_active_runs`, `resume_point`) | Initial public contract for run lifecycle and replay policy behavior. |
| `0.1.x` (future patches) | Backward-compatible with `0.1.0` public APIs | No breaking runtime API changes in patch releases. |

## Client compatibility behavior

| Client surface | Compatibility guarantee |
| --- | --- |
| `KernelModelClient.step(...)` / `SingleStepModelClient.step(...)` | Accepts `replay_policy` and `context_version`; when bound kernel does not support these kwargs, retries once without unsupported kwargs and emits a warning. |
| `KernelModelClient.capabilities()` | Exposes support flags for `replay_policy` and `context_version` on the bound kernel instance. |

## Progress API compatibility

| API | Status values emitted today | Reserved values |
| --- | --- | --- |
| `get_run_progress(...)` / `stream_run_progress(...)` | `running`, `completed`, `failed` | `queued`, `cancelled` |

## Store schema compatibility

| Store backend | Schema version | Compatibility expectation |
| --- | --- | --- |
| SQLite (`SQLiteStore`) | `1` | Compatible across `0.1.x` unless explicitly noted in `CHANGELOG.md`. |
| Postgres (`PostgresStore`) | `1` | Compatible across `0.1.x` unless explicitly noted in `CHANGELOG.md`. |

Both backends expose their declared schema via `get_schema_info()`.

## Store runtime behavior compatibility

| Store backend | Behavior | Compatibility expectation |
| --- | --- | --- |
| Postgres (`PostgresStore`) | Read-path retries for transient connection-lifecycle failures with pool invalidation/reconnect | Supported in current `0.1.x`; tuned by `max_retry_attempts` and `retry_backoff_seconds`. |

## Upgrade guidance

- Read `CHANGELOG.md` before upgrading.
- Treat minor `0.x` bumps as potentially breaking; breaking changes will include explicit upgrade notes.
- Treat patch bumps as backward-compatible unless a release note explicitly states otherwise.
