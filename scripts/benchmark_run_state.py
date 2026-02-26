#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import TypeVar

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pydantic import BaseModel  # noqa: E402

from artana.events import (  # noqa: E402
    EventPayload,
    EventType,
    HarnessSleepPayload,
    KernelEvent,
    RunStartedPayload,
    RunSummaryPayload,
)
from artana.kernel import ArtanaKernel  # noqa: E402
from artana.models import TenantContext  # noqa: E402
from artana.ports.model import ModelRequest, ModelResult, ModelUsage  # noqa: E402
from artana.store import PostgresStore, SQLiteStore  # noqa: E402
from artana.store.base import (  # noqa: E402
    EventStore,
    SupportsModelCostAggregation,
    SupportsRunIndexing,
)

OutputModelT = TypeVar("OutputModelT", bound=BaseModel)


class _UnusedModelPort:
    async def complete(self, request: ModelRequest[OutputModelT]) -> ModelResult[OutputModelT]:
        output = request.output_schema.model_validate({})
        return ModelResult(
            output=output,
            usage=ModelUsage(prompt_tokens=0, completion_tokens=0, cost_usd=0.0),
        )


class _SnapshotBlindStore(EventStore, SupportsRunIndexing, SupportsModelCostAggregation):
    def __init__(self, inner: EventStore) -> None:
        self._inner = inner

    async def append_event(
        self,
        *,
        run_id: str,
        tenant_id: str,
        event_type: EventType,
        payload: EventPayload,
        parent_step_key: str | None = None,
    ) -> KernelEvent:
        return await self._inner.append_event(
            run_id=run_id,
            tenant_id=tenant_id,
            event_type=event_type,
            payload=payload,
            parent_step_key=parent_step_key,
        )

    async def get_events_for_run(self, run_id: str) -> list[KernelEvent]:
        return await self._inner.get_events_for_run(run_id)

    async def get_latest_run_summary(
        self,
        run_id: str,
        summary_type: str,
    ) -> RunSummaryPayload | None:
        return await self._inner.get_latest_run_summary(run_id, summary_type)

    async def verify_run_chain(self, run_id: str) -> bool:
        return await self._inner.verify_run_chain(run_id)

    async def close(self) -> None:
        await self._inner.close()

    async def list_run_ids(
        self,
        *,
        tenant_id: str | None = None,
        since: datetime | None = None,
    ) -> list[str]:
        if not isinstance(self._inner, SupportsRunIndexing):
            raise RuntimeError("Wrapped store does not support run indexing.")
        return await self._inner.list_run_ids(tenant_id=tenant_id, since=since)

    async def get_model_cost_sum_for_run(self, run_id: str) -> float:
        if not isinstance(self._inner, SupportsModelCostAggregation):
            raise RuntimeError("Wrapped store does not support model cost aggregation.")
        return await self._inner.get_model_cost_sum_for_run(run_id)


@dataclass(frozen=True, slots=True)
class Metric:
    method: str
    path: str
    p50_ms: float
    p95_ms: float
    threshold_ms: float


def _tenant(*, tenant_id: str) -> TenantContext:
    return TenantContext(
        tenant_id=tenant_id,
        capabilities=frozenset(),
        budget_usd_limit=10.0,
    )


async def _seed_single_run(
    store: EventStore,
    *,
    run_id: str,
    tenant_id: str,
    event_count: int,
) -> None:
    await store.append_event(
        run_id=run_id,
        tenant_id=tenant_id,
        event_type=EventType.RUN_STARTED,
        payload=RunStartedPayload(),
    )
    for index in range(max(0, event_count - 2)):
        await store.append_event(
            run_id=run_id,
            tenant_id=tenant_id,
            event_type=EventType.RUN_SUMMARY,
            payload=RunSummaryPayload(
                summary_type="trace::round",
                summary_json='{"stage":"work","round":1}',
                step_key=f"seed_{index}",
            ),
        )
    await store.append_event(
        run_id=run_id,
        tenant_id=tenant_id,
        event_type=EventType.HARNESS_SLEEP,
        payload=HarnessSleepPayload(status="completed"),
    )


async def _seed_tenant_runs(
    store: EventStore,
    *,
    tenant_id: str,
    run_count: int,
) -> None:
    for index in range(run_count):
        run_id = f"run_index_{index:03d}"
        await store.append_event(
            run_id=run_id,
            tenant_id=tenant_id,
            event_type=EventType.RUN_STARTED,
            payload=RunStartedPayload(),
        )


async def _sample(
    method: Callable[[], Awaitable[object]],
    *,
    iterations: int,
) -> tuple[float, float]:
    durations: list[float] = []
    for _ in range(iterations):
        started = perf_counter()
        await method()
        durations.append((perf_counter() - started) * 1000.0)
    p50 = statistics.median(durations)
    p95 = sorted(durations)[max(0, int(iterations * 0.95) - 1)]
    return p50, p95


async def _benchmark_store(
    *,
    label: str,
    store: EventStore,
    events_per_run: int,
    run_count: int,
) -> list[Metric]:
    tenant_id = f"tenant_bench_{label}"
    target_run = "run_target"
    await _seed_single_run(
        store,
        run_id=target_run,
        tenant_id=tenant_id,
        event_count=events_per_run,
    )
    await _seed_tenant_runs(
        store,
        tenant_id=tenant_id,
        run_count=run_count,
    )

    snapshot_kernel = ArtanaKernel(store=store, model_port=_UnusedModelPort())
    fallback_kernel = ArtanaKernel(
        store=_SnapshotBlindStore(store),
        model_port=_UnusedModelPort(),
    )
    tenant = _tenant(tenant_id=tenant_id)

    try:
        sqlite_thresholds = {
            "get_run_status": 20.0,
            "explain_run": 25.0,
            "list_active_runs": 150.0,
        }
        postgres_thresholds = {
            "get_run_status": 35.0,
            "explain_run": 40.0,
            "list_active_runs": 250.0,
        }
        thresholds = sqlite_thresholds if label == "sqlite" else postgres_thresholds

        metrics: list[Metric] = []
        for path_name, kernel in (
            ("snapshot", snapshot_kernel),
            ("fallback", fallback_kernel),
        ):
            status_p50, status_p95 = await _sample(
                lambda: kernel.get_run_status(run_id=target_run),
                iterations=40,
            )
            metrics.append(
                Metric(
                    method="get_run_status",
                    path=path_name,
                    p50_ms=status_p50,
                    p95_ms=status_p95,
                    threshold_ms=thresholds["get_run_status"],
                )
            )

            explain_p50, explain_p95 = await _sample(
                lambda: kernel.explain_run(target_run),
                iterations=40,
            )
            metrics.append(
                Metric(
                    method="explain_run",
                    path=path_name,
                    p50_ms=explain_p50,
                    p95_ms=explain_p95,
                    threshold_ms=thresholds["explain_run"],
                )
            )

            list_p50, list_p95 = await _sample(
                lambda: kernel.list_active_runs(tenant_id=tenant.tenant_id),
                iterations=20,
            )
            metrics.append(
                Metric(
                    method="list_active_runs",
                    path=path_name,
                    p50_ms=list_p50,
                    p95_ms=list_p95,
                    threshold_ms=thresholds["list_active_runs"],
                )
            )
        return metrics
    finally:
        await fallback_kernel.close()
        await snapshot_kernel.close()


def _render_markdown(
    *,
    sqlite_metrics: list[Metric],
    postgres_metrics: list[Metric] | None,
    events_per_run: int,
    run_count: int,
) -> str:
    lines = [
        "# Run State Benchmark Report",
        "",
        "## Configuration",
        "",
        f"- generated_at_utc: `{datetime.now(timezone.utc).isoformat()}`",
        f"- single_run_event_count: `{events_per_run}`",
        f"- tenant_run_count: `{run_count}`",
        "",
        "## SQLite",
        "",
        "| method | path | p50 (ms) | p95 (ms) | threshold p95 (ms) | pass |",
        "| --- | --- | ---: | ---: | ---: | :---: |",
    ]
    for metric in sqlite_metrics:
        passed = "yes" if metric.p95_ms <= metric.threshold_ms else "no"
        lines.append(
            f"| `{metric.method}` | `{metric.path}` | "
            f"{metric.p50_ms:.2f} | {metric.p95_ms:.2f} | {metric.threshold_ms:.2f} | {passed} |"
        )

    lines.append("")
    lines.append("## Postgres")
    lines.append("")
    if postgres_metrics is None:
        lines.append(
            "Skipped. Set `ARTANA_TEST_POSTGRES_DSN` to include Postgres benchmark data."
        )
    else:
        lines.extend(
            [
                "| method | path | p50 (ms) | p95 (ms) | threshold p95 (ms) | pass |",
                "| --- | --- | ---: | ---: | ---: | :---: |",
            ]
        )
        for metric in postgres_metrics:
            passed = "yes" if metric.p95_ms <= metric.threshold_ms else "no"
            lines.append(
                f"| `{metric.method}` | `{metric.path}` | "
                f"{metric.p50_ms:.2f} | {metric.p95_ms:.2f} | "
                f"{metric.threshold_ms:.2f} | {passed} |"
            )

    lines.append("")
    lines.append(
        "Note: `resume_point` uses the same snapshot-backed row as `get_run_status`; "
        "its latency tracks that path closely."
    )
    lines.append("")
    return "\n".join(lines)


async def _run(args: argparse.Namespace) -> str:
    sqlite_path = Path(args.sqlite_path).expanduser().resolve()
    if sqlite_path.exists():
        sqlite_path.unlink()

    sqlite_store = SQLiteStore(str(sqlite_path))
    try:
        sqlite_metrics = await _benchmark_store(
            label="sqlite",
            store=sqlite_store,
            events_per_run=args.events_per_run,
            run_count=args.run_count,
        )
    finally:
        await sqlite_store.close()
        if sqlite_path.exists():
            sqlite_path.unlink()

    postgres_metrics: list[Metric] | None = None
    if args.postgres_dsn is not None:
        postgres_store = PostgresStore(args.postgres_dsn)
        try:
            postgres_metrics = await _benchmark_store(
                label="postgres",
                store=postgres_store,
                events_per_run=args.events_per_run,
                run_count=args.run_count,
            )
        finally:
            await postgres_store.close()

    return _render_markdown(
        sqlite_metrics=sqlite_metrics,
        postgres_metrics=postgres_metrics,
        events_per_run=args.events_per_run,
        run_count=args.run_count,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events-per-run", type=int, default=10_000)
    parser.add_argument("--run-count", type=int, default=100)
    parser.add_argument(
        "--sqlite-path",
        default=str(REPO_ROOT / ".tmp_benchmark_run_state.db"),
    )
    parser.add_argument("--postgres-dsn", default=None)
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "output" / "run_state_benchmark.md"),
    )
    args = parser.parse_args()
    if args.events_per_run < 2:
        raise ValueError("--events-per-run must be >= 2.")
    if args.run_count < 1:
        raise ValueError("--run-count must be >= 1.")

    markdown = asyncio.run(_run(args))
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    print(f"Wrote benchmark report to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
