from __future__ import annotations

import json
from pathlib import Path

import pytest

from artana import ArtanaKernel, MockModelPort, SQLiteStore
from artana.events import (
    EventType,
    HarnessFailedPayload,
    HarnessSleepPayload,
    HarnessStagePayload,
)
from artana.models import TenantContext


def _tenant() -> TenantContext:
    return TenantContext(
        tenant_id="org_progress",
        capabilities=frozenset(),
        budget_usd_limit=10.0,
    )


@pytest.mark.asyncio
async def test_get_run_progress_uses_task_progress_summary(tmp_path: Path) -> None:
    store = SQLiteStore(str(tmp_path / "state.db"))
    kernel = ArtanaKernel(
        store=store,
        model_port=MockModelPort(output={"ok": True}),
    )
    run_id = "run_progress_summary"
    tenant = _tenant()

    try:
        await kernel.start_run(tenant=tenant, run_id=run_id)
        await kernel.append_run_summary(
            run_id=run_id,
            tenant=tenant,
            summary_type="task_progress",
            summary_json=json.dumps(
                {
                    "units": [
                        {"id": "collect", "description": "Collect", "state": "done"},
                        {"id": "draft", "description": "Draft", "state": "in_progress"},
                        {"id": "verify", "description": "Verify", "state": "pending"},
                    ]
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
        )

        progress = await kernel.get_run_progress(run_id=run_id)

        assert progress.status == "running"
        assert progress.percent == 33
        assert progress.current_stage == "draft"
        assert progress.completed_stages == ("collect",)
        assert progress.started_at <= progress.updated_at
        assert progress.eta_seconds is not None
    finally:
        await kernel.close()


@pytest.mark.asyncio
async def test_get_run_progress_falls_back_to_last_stage_without_task_progress(
    tmp_path: Path,
) -> None:
    store = SQLiteStore(str(tmp_path / "state.db"))
    kernel = ArtanaKernel(
        store=store,
        model_port=MockModelPort(output={"ok": True}),
    )
    run_id = "run_progress_fallback_stage"
    tenant = _tenant()

    try:
        await kernel.start_run(tenant=tenant, run_id=run_id)
        await store.append_event(
            run_id=run_id,
            tenant_id=tenant.tenant_id,
            event_type=EventType.HARNESS_STAGE,
            payload=HarnessStagePayload(stage="draft"),
        )

        progress = await kernel.get_run_progress(run_id=run_id)

        assert progress.status == "running"
        assert progress.percent == 0
        assert progress.current_stage == "draft"
        assert progress.completed_stages == ()
        assert progress.eta_seconds is None
    finally:
        await kernel.close()


@pytest.mark.asyncio
async def test_get_run_progress_terminal_states(tmp_path: Path) -> None:
    store = SQLiteStore(str(tmp_path / "state.db"))
    kernel = ArtanaKernel(
        store=store,
        model_port=MockModelPort(output={"ok": True}),
    )
    tenant = _tenant()

    try:
        completed_run_id = "run_progress_completed"
        await kernel.start_run(tenant=tenant, run_id=completed_run_id)
        await store.append_event(
            run_id=completed_run_id,
            tenant_id=tenant.tenant_id,
            event_type=EventType.HARNESS_SLEEP,
            payload=HarnessSleepPayload(status="completed"),
        )

        failed_run_id = "run_progress_failed"
        await kernel.start_run(tenant=tenant, run_id=failed_run_id)
        await store.append_event(
            run_id=failed_run_id,
            tenant_id=tenant.tenant_id,
            event_type=EventType.HARNESS_FAILED,
            payload=HarnessFailedPayload(
                error_type="runtime_error",
                message="boom",
            ),
        )

        completed_progress = await kernel.get_run_progress(run_id=completed_run_id)
        failed_progress = await kernel.get_run_progress(run_id=failed_run_id)

        assert completed_progress.status == "completed"
        assert completed_progress.percent == 100
        assert completed_progress.eta_seconds == 0

        assert failed_progress.status == "failed"
        assert failed_progress.percent == 0
    finally:
        await kernel.close()


@pytest.mark.asyncio
async def test_stream_run_progress_deduplicates_identical_updates(tmp_path: Path) -> None:
    store = SQLiteStore(str(tmp_path / "state.db"))
    kernel = ArtanaKernel(
        store=store,
        model_port=MockModelPort(output={"ok": True}),
    )
    run_id = "run_progress_stream"
    tenant = _tenant()
    summary_json = json.dumps(
        {
            "units": [
                {"id": "collect", "description": "Collect", "state": "done"},
                {"id": "draft", "description": "Draft", "state": "pending"},
            ]
        },
        ensure_ascii=False,
        sort_keys=True,
    )

    try:
        await kernel.start_run(tenant=tenant, run_id=run_id)
        await kernel.append_run_summary(
            run_id=run_id,
            tenant=tenant,
            summary_type="task_progress",
            summary_json=summary_json,
        )
        await kernel.append_run_summary(
            run_id=run_id,
            tenant=tenant,
            summary_type="task_progress",
            summary_json=summary_json,
        )

        updates = [
            progress
            async for progress in kernel.stream_run_progress(
                run_id=run_id,
                since_seq=1,
            )
        ]

        assert len(updates) == 1
        assert updates[0].percent == 50
    finally:
        await kernel.close()
