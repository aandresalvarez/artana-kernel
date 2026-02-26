from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import TypeVar

import pytest
from pydantic import BaseModel

from artana import ArtanaKernel
from artana.cli import main as cli_main
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


async def _seed_runs(db_path: Path) -> None:
    kernel = ArtanaKernel(store=SQLiteStore(str(db_path)), model_port=DummyModelPort())
    try:
        await kernel.start_run(tenant=_tenant(), run_id="run_cli_one")
        await kernel.start_run(tenant=_tenant(), run_id="run_cli_two")
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


def test_cli_run_list_and_tail_and_verify(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    db_path = tmp_path / "state.db"
    asyncio.run(_seed_runs(db_path))

    code_list = cli_main(["run", "list", "--db", str(db_path)])
    output_list = capsys.readouterr().out
    assert code_list == 0
    assert "run_cli_one" in output_list
    assert "run_cli_two" in output_list

    code_tail = cli_main(["run", "tail", "run_cli_one", "--db", str(db_path)])
    output_tail = capsys.readouterr().out
    assert code_tail == 0
    assert "run_started" in output_tail

    code_verify = cli_main(["run", "verify-ledger", "run_cli_one", "--db", str(db_path)])
    output_verify = capsys.readouterr().out.strip()
    assert code_verify == 0
    assert output_verify == "valid"


def test_cli_json_status_summaries_and_artifacts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    db_path = tmp_path / "state.db"
    asyncio.run(_seed_runs(db_path))

    code_status = cli_main(
        ["run", "status", "run_cli_one", "--db", str(db_path), "--json"]
    )
    payload_status = json.loads(capsys.readouterr().out)
    assert code_status == 0
    assert payload_status["run_id"] == "run_cli_one"
    assert payload_status["status"] in {"active", "paused", "failed", "completed"}

    code_summaries = cli_main(
        ["run", "summaries", "run_cli_one", "--db", str(db_path), "--json"]
    )
    payload_summaries = json.loads(capsys.readouterr().out)
    assert code_summaries == 0
    assert payload_summaries["run_id"] == "run_cli_one"
    assert any(
        item.get("summary_type") == "checkpoint::phase_collect"
        for item in payload_summaries["summaries"]
    )

    code_artifacts = cli_main(
        ["run", "artifacts", "run_cli_one", "--db", str(db_path), "--json"]
    )
    payload_artifacts = json.loads(capsys.readouterr().out)
    assert code_artifacts == 0
    assert payload_artifacts["run_id"] == "run_cli_one"
    assert payload_artifacts["artifacts"]["report"]["status"] == "ok"

    code_verify = cli_main(
        ["run", "verify-ledger", "run_cli_one", "--db", str(db_path), "--json"]
    )
    payload_verify = json.loads(capsys.readouterr().out)
    assert code_verify == 0
    assert payload_verify["valid"] is True


def test_cli_init_scaffold_profiles(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    enforced_path = tmp_path / "starter_enforced"
    dev_path = tmp_path / "starter_dev"

    code_enforced = cli_main(["init", str(enforced_path)])
    out_enforced = capsys.readouterr().out
    assert code_enforced == 0
    assert "Initialized Artana project" in out_enforced
    enforced_app = (enforced_path / "app.py").read_text(encoding="utf-8")
    assert "KernelPolicy.enforced()" in enforced_app

    code_dev = cli_main(["init", str(dev_path), "--profile", "dev"])
    out_dev = capsys.readouterr().out
    assert code_dev == 0
    assert "Initialized Artana project" in out_dev
    dev_app = (dev_path / "app.py").read_text(encoding="utf-8")
    assert "KernelPolicy.enforced()" not in dev_app
