from __future__ import annotations

import json
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TypeVar

import pytest
from pydantic import BaseModel

import artana.cli as cli_module
from artana import ArtanaKernel
from artana.events import KernelEvent
from artana.models import TenantContext
from artana.ports.model import ModelRequest, ModelResult, ModelUsage
from artana.store import SQLiteStore

OutputT = TypeVar("OutputT", bound=BaseModel)


class DummyModelPort:
    async def complete(self, request: ModelRequest[OutputT]) -> ModelResult[OutputT]:
        output = request.output_schema.model_validate({})
        return ModelResult(
            output=output,
            usage=ModelUsage(prompt_tokens=0, completion_tokens=0, cost_usd=0.0),
        )


def _tenant() -> TenantContext:
    return TenantContext(
        tenant_id="org_cli",
        capabilities=frozenset(),
        budget_usd_limit=1.0,
    )


def _other_tenant() -> TenantContext:
    return TenantContext(
        tenant_id="org_cli_other",
        capabilities=frozenset(),
        budget_usd_limit=1.0,
    )


async def _seed_runs(db_path: Path) -> None:
    kernel = ArtanaKernel(store=SQLiteStore(str(db_path)), model_port=DummyModelPort())
    try:
        await kernel.start_run(tenant=_tenant(), run_id="run_cli_one")
        await kernel.start_run(tenant=_tenant(), run_id="run_cli_two")
        await kernel.start_run(tenant=_other_tenant(), run_id="run_cli_other")
        await kernel.checkpoint(
            run_id="run_cli_one",
            tenant=_tenant(),
            name="phase_collect",
            payload={"done": True},
            step_key="checkpoint_1",
        )
        await kernel.set_artifact(
            run_id="run_cli_one",
            tenant=_tenant(),
            key="report",
            value={"status": "ok"},
            step_key="artifact_report_1",
        )
    finally:
        await kernel.close()


async def _read_first_event(db_path: Path, run_id: str) -> KernelEvent:
    store = SQLiteStore(str(db_path))
    try:
        return (await store.get_events_for_run(run_id))[0]
    finally:
        await store.close()


async def _invoke_cli(argv: list[str]) -> int:
    parser = cli_module._build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 1
    try:
        return await cli_module._run_command(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


@pytest.mark.asyncio
async def test_cli_run_list_and_tail_and_verify(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    db_path = tmp_path / "state.db"
    await _seed_runs(db_path)

    code_list = await _invoke_cli(
        ["run", "list", "--db", str(db_path), "--tenant", _tenant().tenant_id]
    )
    output_list = capsys.readouterr().out
    assert code_list == 0
    assert "run_cli_one" in output_list
    assert "run_cli_two" in output_list
    assert "run_cli_other" not in output_list

    missing_tenant_code = await _invoke_cli(["run", "list", "--db", str(db_path)])
    missing_tenant = capsys.readouterr()
    assert missing_tenant_code == 2
    assert "required: --tenant" in missing_tenant.err

    code_tail = await _invoke_cli(
        ["run", "tail", "run_cli_one", "--db", str(db_path), "--tenant", _tenant().tenant_id]
    )
    output_tail = capsys.readouterr().out
    assert code_tail == 0
    assert "run_started" in output_tail

    code_verify = await _invoke_cli(
        [
            "run",
            "verify-ledger",
            "run_cli_one",
            "--db",
            str(db_path),
            "--tenant",
            _tenant().tenant_id,
        ]
    )
    output_verify = capsys.readouterr().out.strip()
    assert code_verify == 0
    assert output_verify == "valid"


@pytest.mark.asyncio
async def test_cli_json_status_summaries_and_artifacts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    db_path = tmp_path / "state.db"
    await _seed_runs(db_path)

    code_status = await _invoke_cli(
        [
            "run",
            "status",
            "run_cli_one",
            "--db",
            str(db_path),
            "--tenant",
            _tenant().tenant_id,
            "--json",
        ]
    )
    payload_status = json.loads(capsys.readouterr().out)
    assert code_status == 0
    assert payload_status["run_id"] == "run_cli_one"
    assert payload_status["status"] in {"active", "paused", "failed", "completed"}

    code_summaries = await _invoke_cli(
        [
            "run",
            "summaries",
            "run_cli_one",
            "--db",
            str(db_path),
            "--tenant",
            _tenant().tenant_id,
            "--json",
        ]
    )
    payload_summaries = json.loads(capsys.readouterr().out)
    assert code_summaries == 0
    assert payload_summaries["run_id"] == "run_cli_one"
    assert any(
        item.get("summary_type") == "checkpoint::phase_collect"
        for item in payload_summaries["summaries"]
    )

    code_artifacts = await _invoke_cli(
        [
            "run",
            "artifacts",
            "run_cli_one",
            "--db",
            str(db_path),
            "--tenant",
            _tenant().tenant_id,
            "--json",
        ]
    )
    payload_artifacts = json.loads(capsys.readouterr().out)
    assert code_artifacts == 0
    assert payload_artifacts["run_id"] == "run_cli_one"
    assert payload_artifacts["artifacts"]["report"]["status"] == "ok"

    code_verify = await _invoke_cli(
        [
            "run",
            "verify-ledger",
            "run_cli_one",
            "--db",
            str(db_path),
            "--tenant",
            _tenant().tenant_id,
            "--json",
        ]
    )
    payload_verify = json.loads(capsys.readouterr().out)
    assert code_verify == 0
    assert payload_verify["valid"] is True


@pytest.mark.asyncio
async def test_cli_tail_follow_uses_streaming_access_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    db_path = tmp_path / "state.db"
    await _seed_runs(db_path)

    store = SQLiteStore(str(db_path))
    first_event = await _read_first_event(db_path, "run_cli_one")
    observed = {"history_reads": 0, "stream_calls": 0}

    async def fail_get_events_for_run(run_id: str) -> list[object]:
        observed["history_reads"] += 1
        raise AssertionError("tail --follow should not preload run history.")

    async def stream_events(
        run_id: str,
        *,
        since_seq: int = 0,
        follow: bool = False,
        poll_interval_seconds: float = 0.5,
        idle_timeout_seconds: float | None = None,
    ) -> AsyncIterator[KernelEvent]:
        observed["stream_calls"] += 1
        assert run_id == "run_cli_one"
        assert since_seq == 0
        assert follow is True
        yield first_event

    monkeypatch.setattr(store, "get_events_for_run", fail_get_events_for_run)
    monkeypatch.setattr(store, "stream_events", stream_events)
    monkeypatch.setattr(cli_module, "_open_store", lambda *, db, dsn: store)

    code = await _invoke_cli(
        [
            "run",
            "tail",
            "run_cli_one",
            "--db",
            str(db_path),
            "--tenant",
            _tenant().tenant_id,
            "--follow",
        ]
    )
    output = capsys.readouterr().out

    assert code == 0
    assert observed["history_reads"] == 0
    assert observed["stream_calls"] == 1
    assert "run_started" in output


@pytest.mark.asyncio
async def test_cli_init_scaffold_profiles(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    enforced_path = tmp_path / "starter_enforced"
    dev_path = tmp_path / "starter_dev"

    code_enforced = await _invoke_cli(["init", str(enforced_path)])
    out_enforced = capsys.readouterr().out
    assert code_enforced == 0
    assert "Initialized Artana project" in out_enforced
    enforced_app = (enforced_path / "app.py").read_text(encoding="utf-8")
    assert "KernelPolicy.enforced()" in enforced_app

    code_dev = await _invoke_cli(["init", str(dev_path), "--profile", "dev"])
    out_dev = capsys.readouterr().out
    assert code_dev == 0
    assert "Initialized Artana project" in out_dev
    dev_app = (dev_path / "app.py").read_text(encoding="utf-8")
    assert "KernelPolicy.enforced()" not in dev_app
