# Run State Benchmark Report

## Configuration

- generated_at_utc: `2026-02-26T21:20:08.289030+00:00`
- single_run_event_count: `10000`
- tenant_run_count: `100`

## SQLite

| method | path | p50 (ms) | p95 (ms) | threshold p95 (ms) | pass |
| --- | --- | ---: | ---: | ---: | :---: |
| `get_run_status` | `snapshot` | 0.08 | 0.13 | 20.00 | yes |
| `explain_run` | `snapshot` | 0.08 | 0.08 | 25.00 | yes |
| `list_active_runs` | `snapshot` | 0.57 | 0.67 | 150.00 | yes |
| `get_run_status` | `fallback` | 234.55 | 263.20 | 20.00 | no |
| `explain_run` | `fallback` | 231.34 | 244.03 | 25.00 | no |
| `list_active_runs` | `fallback` | 247.67 | 259.72 | 150.00 | no |

## Postgres

Skipped. Set `ARTANA_TEST_POSTGRES_DSN` to include Postgres benchmark data.

Note: `resume_point` uses the same snapshot-backed row as `get_run_status`; its latency tracks that path closely.
